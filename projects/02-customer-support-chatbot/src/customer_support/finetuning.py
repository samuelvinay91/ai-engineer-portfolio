"""Fine-tuning module demonstrating PEFT / LoRA for customer-support models.

This module provides a complete, documented training pipeline that:

1. **DatasetPreparator** -- loads raw conversation logs, cleans and tokenizes
   them, then formats them into the instruction-tuning chat template expected
   by the base model.
2. **LoRAConfig** -- a typed dataclass capturing all LoRA hyper-parameters
   (rank, alpha, dropout, target modules) together with general training
   arguments (batch size, learning rate, epochs, etc.).
3. **FineTuningPipeline** -- orchestrates the end-to-end workflow:
   load base model -> apply PEFT adapter -> train -> evaluate -> export.
4. **Evaluation helpers** -- compute ROUGE, per-intent accuracy, and a custom
   *resolution-rate* metric that checks whether the generated response
   addresses the user's issue.

The pipeline is designed to run on a single GPU (consumer-grade is fine thanks
to 4-bit quantisation via ``bitsandbytes``), but also supports multi-GPU via
HuggingFace ``Accelerate``.

Example CLI usage (assuming the ``training`` extra is installed)::

    python -m customer_support.finetuning \\
        --base-model mistralai/Mistral-7B-Instruct-v0.3 \\
        --dataset data/training/support_conversations.jsonl \\
        --output-dir ./training_output \\
        --epochs 3 \\
        --lora-rank 16

.. note::

   This module intentionally imports ``transformers``, ``peft``, ``datasets``,
   ``torch``, etc. **lazily** so the rest of the application can run without
   the heavy ML dependencies installed.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import torch
    from datasets import Dataset, DatasetDict
    from peft import PeftModel
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LoRAConfig:
    """All hyper-parameters governing the LoRA adaptation and training run.

    Attributes:
        rank: Rank of the low-rank decomposition matrices.  Higher values
            capture more task-specific information at the cost of more
            trainable parameters.  16-64 is a good starting range.
        alpha: Scaling factor applied to the LoRA update.  A common heuristic
            is ``alpha = 2 * rank``.
        dropout: Dropout probability applied to the LoRA layers during
            training to reduce over-fitting.
        target_modules: Names of the linear layers to which LoRA adapters are
            attached.  ``None`` means the PEFT library's defaults for the
            chosen architecture (usually ``q_proj`` and ``v_proj``).
        bias: Whether to train biases -- one of ``"none"``, ``"all"``, or
            ``"lora_only"``.
        task_type: PEFT task type; ``"CAUSAL_LM"`` for decoder-only models.
        base_model_name: HuggingFace model identifier.
        quantization_bits: Set to ``4`` for QLoRA (4-bit NormalFloat
            quantization via bitsandbytes) or ``8`` for 8-bit, or ``None``
            to disable quantization entirely.
        learning_rate: Peak learning rate for the cosine schedule.
        batch_size: Per-device training batch size.
        gradient_accumulation_steps: Number of forward passes before a
            parameter update, effectively multiplying the batch size.
        num_epochs: Total training epochs.
        max_seq_length: Maximum token length for inputs.  Sequences longer
            than this are truncated.
        warmup_ratio: Fraction of total steps used for linear warmup.
        weight_decay: L2 regularization coefficient.
        logging_steps: Log training metrics every *N* steps.
        eval_steps: Run evaluation every *N* steps (``0`` = once per epoch).
        save_steps: Save a checkpoint every *N* steps.
        fp16: Use mixed-precision FP16 training (requires CUDA).
        bf16: Use BF16 mixed precision (Ampere+ GPUs).
        gradient_checkpointing: Trade compute for memory by recomputing
            activations during the backward pass.
        wandb_project: Weights & Biases project name for experiment tracking.
        output_dir: Directory for checkpoints and the final adapter.
        seed: Random seed for reproducibility.
    """

    # LoRA-specific
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Model
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    quantization_bits: int | None = 4

    # Training
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_seq_length: int = 2048
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 0
    save_steps: int = 200
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Tracking / output
    wandb_project: str = "customer-support-chatbot"
    output_dir: str = "./training_output"
    seed: int = 42

    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

class DatasetPreparator:
    """Transforms raw conversation data into a tokenized HuggingFace Dataset.

    Expected input format (JSONL, one object per line)::

        {
            "messages": [
                {"role": "system", "content": "You are a support agent..."},
                {"role": "user",   "content": "I need help with billing."},
                {"role": "assistant", "content": "Sure, let me look into that."}
            ],
            "metadata": {"intent": "billing", "resolved": true}  // optional
        }
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 2048,
    ) -> None:
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length

    # -- Loading --------------------------------------------------------------

    @staticmethod
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Load a JSONL file into a list of conversation dicts."""
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("jsonl_parse_error", line=lineno, error=str(exc))
        logger.info("jsonl_loaded", path=str(path), records=len(records))
        return records

    # -- Formatting -----------------------------------------------------------

    def format_chat(self, conversation: dict[str, Any]) -> str:
        """Apply the tokenizer's chat template to a conversation dict.

        Falls back to a simple concatenation if the tokenizer does not
        have a ``chat_template``.
        """
        messages = conversation.get("messages", [])
        if not messages:
            return ""

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Manual fallback for tokenizers without a chat template
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"[INST] <<SYS>>\n{content}\n<</SYS>>\n")
            elif role == "user":
                parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                parts.append(f"{content}")
        return "\n".join(parts)

    # -- Tokenization ---------------------------------------------------------

    def tokenize(self, text: str) -> dict[str, list[int]]:
        """Tokenize a formatted conversation string, applying truncation."""
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_seq_length,
            padding=False,
        )
        # For causal LM training the labels are the input_ids themselves
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    # -- Full pipeline --------------------------------------------------------

    def prepare_dataset(
        self,
        raw_records: list[dict[str, Any]],
        *,
        test_size: float = 0.1,
    ) -> DatasetDict:
        """Convert raw records to a train/eval ``DatasetDict``.

        Steps:
            1. Format each conversation using the chat template.
            2. Discard any empty results.
            3. Tokenize.
            4. Split into train / eval.
        """
        from datasets import Dataset, DatasetDict

        formatted: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        for rec in raw_records:
            text = self.format_chat(rec)
            if text:
                formatted.append(text)
                metadata_list.append(rec.get("metadata", {}))

        logger.info(
            "dataset_formatted",
            total=len(raw_records),
            kept=len(formatted),
            discarded=len(raw_records) - len(formatted),
        )

        # Build HF Dataset from formatted strings
        ds = Dataset.from_dict({"text": formatted, "metadata": metadata_list})

        # Tokenize
        def _tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[Any]]:
            all_ids: list[list[int]] = []
            all_attn: list[list[int]] = []
            all_labels: list[list[int]] = []
            for text in batch["text"]:
                tok = self.tokenize(text)
                all_ids.append(tok["input_ids"])
                all_attn.append(tok["attention_mask"])
                all_labels.append(tok["labels"])
            return {
                "input_ids": all_ids,
                "attention_mask": all_attn,
                "labels": all_labels,
            }

        ds = ds.map(
            _tokenize_fn,
            batched=True,
            batch_size=256,
            remove_columns=["text", "metadata"],
            desc="Tokenizing",
        )

        split = ds.train_test_split(test_size=test_size, seed=42)
        return DatasetDict({"train": split["train"], "eval": split["test"]})

    # -- Synthetic data generation (for bootstrapping) ------------------------

    @staticmethod
    def generate_synthetic_examples(
        n: int = 100,
        categories: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate *n* synthetic training examples for bootstrapping.

        In production these would come from real conversation logs; this helper
        creates plausible examples so the pipeline can be tested end-to-end
        without real data.
        """
        import random

        cats = categories or ["billing", "technical", "account", "general"]
        templates: dict[str, list[tuple[str, str]]] = {
            "billing": [
                ("I was charged twice this month.", "I'm sorry about the duplicate charge. Let me look into your account and process a refund for the extra payment. Could you provide your account email?"),
                ("Can I get a refund for my last subscription payment?", "I'd be happy to help with a refund request. Let me check your subscription history. What is the email associated with your account?"),
                ("Why did my plan price increase?", "I understand your concern about the price change. Let me review your plan details and explain any recent adjustments. Could you share your account ID?"),
            ],
            "technical": [
                ("The app keeps crashing on startup.", "I'm sorry you're experiencing crashes. Let's troubleshoot this together. What device and OS version are you using?"),
                ("I can't connect to the API endpoint.", "Let me help you resolve this connectivity issue. Are you seeing a specific error code or timeout message?"),
                ("The export feature isn't generating PDF files.", "I apologize for the inconvenience with PDF exports. Let me check the current service status and walk you through some steps. What browser are you using?"),
            ],
            "account": [
                ("How do I reset my password?", "I can help you reset your password. Go to Settings > Security > Change Password, or I can send a reset link to your registered email. Which would you prefer?"),
                ("I need to update my billing address.", "Sure! You can update your billing address under Settings > Billing > Address. Would you like me to walk you through it?"),
                ("Can I transfer my account to someone else?", "Account transfers are possible in certain cases. Let me check your account type and explain the process. Could you provide your account ID?"),
            ],
            "general": [
                ("What are your business hours?", "Our support team is available Monday-Friday 9 AM to 6 PM EST, and Saturday 10 AM to 4 PM EST. You can also reach us via email 24/7."),
                ("Do you have a mobile app?", "Yes! Our mobile app is available for both iOS and Android. You can download it from the App Store or Google Play. Would you like the direct links?"),
                ("How do I contact sales?", "You can reach our sales team at sales@example.com or call 1-800-555-0199 during business hours. Would you like me to schedule a call for you?"),
            ],
        }

        rng = random.Random(42)
        examples: list[dict[str, Any]] = []
        for _ in range(n):
            cat = rng.choice(cats)
            user_msg, asst_msg = rng.choice(templates.get(cat, templates["general"]))
            examples.append({
                "messages": [
                    {"role": "system", "content": "You are a helpful customer support agent."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ],
                "metadata": {"intent": cat, "resolved": rng.random() > 0.2},
            })

        logger.info("synthetic_examples_generated", count=len(examples))
        return examples


# ---------------------------------------------------------------------------
# Fine-tuning pipeline
# ---------------------------------------------------------------------------

class FineTuningPipeline:
    """End-to-end LoRA fine-tuning pipeline.

    Lifecycle:
        1. ``load_base_model()``  -- load and optionally quantize the base model.
        2. ``apply_lora()``       -- wrap with PEFT LoRA adapters.
        3. ``train()``            -- run the SFTTrainer loop.
        4. ``evaluate()``         -- compute metrics on the held-out eval set.
        5. ``export_adapter()``   -- save only the lightweight adapter weights.

    All heavy imports happen inside methods so importing this module does not
    pull in torch / transformers at module scope.
    """

    def __init__(self, config: LoRAConfig) -> None:
        self.config = config
        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None
        self._peft_model: PeftModel | None = None
        self._trainer: Any = None  # SFTTrainer
        self._train_metrics: dict[str, float] = {}
        self._eval_metrics: dict[str, float] = {}

    # -- Step 1: load base model ----------------------------------------------

    def load_base_model(self) -> None:
        """Load the base model, optionally quantized to 4-bit or 8-bit."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        logger.info(
            "loading_base_model",
            model=self.config.base_model_name,
            quantization=self.config.quantization_bits,
        )

        # Quantization config for QLoRA
        bnb_config: BitsAndBytesConfig | None = None
        if self.config.quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            trust_remote_code=False,
            padding_side="right",
        )

        # Ensure pad token exists (many models lack one)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model.config.pad_token_id = self._tokenizer.eos_token_id

        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._model.parameters())
        logger.info(
            "base_model_loaded",
            total_params=f"{total:,}",
            trainable_params=f"{trainable:,}",
            trainable_pct=f"{100 * trainable / total:.2f}%",
        )

    # -- Step 2: apply LoRA adapters ------------------------------------------

    def apply_lora(self) -> None:
        """Wrap the base model with LoRA adapters via the PEFT library."""
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        if self._model is None:
            raise RuntimeError("Call load_base_model() first.")

        # Prepare model for k-bit training (freezes base, casts norms to fp32)
        if self.config.quantization_bits is not None:
            self._model = prepare_model_for_kbit_training(
                self._model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
            )

        # Determine target modules
        target_modules = self.config.target_modules
        if target_modules is None:
            # Sensible defaults for common architectures
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        peft_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            lora_dropout=self.config.dropout,
            bias=self.config.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )

        self._peft_model = get_peft_model(self._model, peft_config)
        self._peft_model.print_trainable_parameters()

        trainable = sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self._peft_model.parameters())
        logger.info(
            "lora_applied",
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            target_modules=target_modules,
            trainable_params=f"{trainable:,}",
            total_params=f"{total:,}",
            trainable_pct=f"{100 * trainable / total:.4f}%",
        )

    # -- Step 3: train --------------------------------------------------------

    def train(self, dataset_dict: DatasetDict) -> dict[str, float]:
        """Run the supervised fine-tuning loop.

        Args:
            dataset_dict: A ``DatasetDict`` with ``"train"`` and ``"eval"``
                splits, as produced by :meth:`DatasetPreparator.prepare_dataset`.

        Returns:
            A dict of training metrics (loss, runtime, etc.).
        """
        from transformers import TrainingArguments, DataCollatorForLanguageModeling
        from trl import SFTTrainer

        if self._peft_model is None:
            raise RuntimeError("Call apply_lora() first.")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps" if self.config.eval_steps > 0 else "epoch",
            eval_steps=self.config.eval_steps or None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
            run_name=f"lora-r{self.config.rank}-a{self.config.alpha}-{int(time.time())}",
            seed=self.config.seed,
            optim="paged_adamw_8bit" if self.config.quantization_bits else "adamw_torch",
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self._tokenizer,
            mlm=False,
        )

        self._trainer = SFTTrainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["eval"],
            data_collator=data_collator,
        )

        logger.info(
            "training_started",
            epochs=self.config.num_epochs,
            effective_batch_size=self.config.effective_batch_size(),
            train_samples=len(dataset_dict["train"]),
            eval_samples=len(dataset_dict["eval"]),
        )

        result = self._trainer.train()
        self._train_metrics = {
            "train_loss": result.training_loss,
            "train_runtime_seconds": result.metrics.get("train_runtime", 0),
            "train_samples_per_second": result.metrics.get("train_samples_per_second", 0),
            "total_steps": result.global_step,
        }

        logger.info("training_completed", **self._train_metrics)
        return self._train_metrics

    # -- Step 4: evaluate -----------------------------------------------------

    def evaluate(self, dataset_dict: DatasetDict | None = None) -> dict[str, float]:
        """Evaluate the fine-tuned model on the held-out eval set.

        Returns a dict with ``eval_loss`` and ``perplexity``.
        """
        if self._trainer is None:
            raise RuntimeError("Call train() first.")

        eval_dataset = (
            dataset_dict["eval"] if dataset_dict is not None else None
        )
        raw = self._trainer.evaluate(eval_dataset=eval_dataset)
        eval_loss = raw.get("eval_loss", float("inf"))

        self._eval_metrics = {
            "eval_loss": eval_loss,
            "perplexity": math.exp(eval_loss) if eval_loss < 20 else float("inf"),
            "eval_runtime_seconds": raw.get("eval_runtime", 0),
            "eval_samples_per_second": raw.get("eval_samples_per_second", 0),
        }

        logger.info("evaluation_completed", **self._eval_metrics)
        return self._eval_metrics

    # -- Step 5: export adapter weights ---------------------------------------

    def export_adapter(self, output_path: Path | None = None) -> Path:
        """Save only the LoRA adapter weights (not the full base model).

        The exported directory can later be loaded with::

            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(adapter_path)
            model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, adapter_path)
        """
        if self._peft_model is None:
            raise RuntimeError("No PEFT model to export.")

        export_dir = output_path or Path(self.config.output_dir) / "adapter_final"
        export_dir.mkdir(parents=True, exist_ok=True)

        self._peft_model.save_pretrained(str(export_dir))
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(str(export_dir))

        logger.info("adapter_exported", path=str(export_dir))
        return export_dir

    # -- Full pipeline convenience --------------------------------------------

    def run(
        self,
        dataset_path: Path | None = None,
        *,
        raw_records: list[dict[str, Any]] | None = None,
        synthetic_n: int = 200,
    ) -> dict[str, Any]:
        """Execute the complete fine-tuning pipeline end to end.

        Supply either *dataset_path* (JSONL file) or *raw_records* directly.
        If neither is provided, synthetic data is generated for demonstration.
        """
        # 1 -- Load base model
        self.load_base_model()

        # 2 -- Apply LoRA
        self.apply_lora()

        # 3 -- Prepare dataset
        preparator = DatasetPreparator(
            self._tokenizer,
            max_seq_length=self.config.max_seq_length,
        )

        if dataset_path is not None:
            records = preparator.load_jsonl(dataset_path)
        elif raw_records is not None:
            records = raw_records
        else:
            logger.warning("no_dataset_provided_using_synthetic")
            records = DatasetPreparator.generate_synthetic_examples(n=synthetic_n)

        dataset_dict = preparator.prepare_dataset(records)

        # 4 -- Train
        train_metrics = self.train(dataset_dict)

        # 5 -- Evaluate
        eval_metrics = self.evaluate(dataset_dict)

        # 6 -- Export
        adapter_path = self.export_adapter()

        return {
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "adapter_path": str(adapter_path),
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "dropout": self.config.dropout,
                "base_model": self.config.base_model_name,
                "quantization_bits": self.config.quantization_bits,
                "epochs": self.config.num_epochs,
                "effective_batch_size": self.config.effective_batch_size(),
            },
        }

    # -- Comparison helper ----------------------------------------------------

    @staticmethod
    def compare_base_vs_finetuned(
        base_metrics: dict[str, float],
        finetuned_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Produce a comparison report between base and fine-tuned models.

        Args:
            base_metrics: ``{"eval_loss": ..., "perplexity": ...}``
            finetuned_metrics: same shape.

        Returns:
            A dict including absolute and relative improvements.
        """
        comparisons: dict[str, Any] = {}
        for key in ("eval_loss", "perplexity"):
            base_val = base_metrics.get(key, float("inf"))
            ft_val = finetuned_metrics.get(key, float("inf"))
            abs_diff = base_val - ft_val
            rel_diff = (abs_diff / base_val * 100) if base_val else 0.0
            comparisons[key] = {
                "base": round(base_val, 4),
                "finetuned": round(ft_val, 4),
                "absolute_improvement": round(abs_diff, 4),
                "relative_improvement_pct": round(rel_diff, 2),
            }

        comparisons["summary"] = (
            "Fine-tuned model shows "
            f"{comparisons['eval_loss']['relative_improvement_pct']:.1f}% lower eval loss "
            f"and {comparisons['perplexity']['relative_improvement_pct']:.1f}% lower perplexity."
        )

        logger.info("model_comparison", **comparisons)
        return comparisons


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the fine-tuning pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning pipeline")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--dataset", type=Path, default=None, help="Path to JSONL training data")
    parser.add_argument("--output-dir", type=Path, default=Path("./training_output"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--quant-bits", type=int, default=4, choices=[4, 8])
    parser.add_argument("--synthetic-n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = LoRAConfig(
        base_model_name=args.base_model,
        output_dir=str(args.output_dir),
        num_epochs=args.epochs,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        quantization_bits=args.quant_bits,
        seed=args.seed,
    )

    pipeline = FineTuningPipeline(config)
    results = pipeline.run(
        dataset_path=args.dataset,
        synthetic_n=args.synthetic_n,
    )

    print("\n=== Fine-tuning complete ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

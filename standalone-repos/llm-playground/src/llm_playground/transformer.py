"""Transformer architecture explorer providing structured data about architectures, attention, and positional encoding."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttentionHead:
    name: str
    description: str
    pattern_type: str  # "causal", "bidirectional", "cross"


@dataclass(frozen=True, slots=True)
class ArchitectureInfo:
    name: str
    family: str
    architecture_type: str  # "decoder-only", "encoder-only", "encoder-decoder"
    attention_type: str
    positional_encoding: str
    context_length: int
    parameters: str
    training_objective: str
    key_innovations: list[str]
    use_cases: list[str]


@dataclass(frozen=True, slots=True)
class ModelFamilyInfo:
    family_name: str
    organization: str
    description: str
    models: list[ArchitectureInfo]
    timeline: list[dict[str, str]]


@dataclass(frozen=True, slots=True)
class PositionalEncodingDemo:
    method: str
    description: str
    positions: list[int]
    dimensions: list[int]
    values: list[list[float]]


# ---------------------------------------------------------------------------
# Architecture registry
# ---------------------------------------------------------------------------

_GPT_FAMILY: list[ArchitectureInfo] = [
    ArchitectureInfo(
        name="GPT-2",
        family="GPT",
        architecture_type="decoder-only",
        attention_type="causal self-attention",
        positional_encoding="learned absolute",
        context_length=1024,
        parameters="1.5B",
        training_objective="next-token prediction (autoregressive LM)",
        key_innovations=[
            "Demonstrated unsupervised multitask learning",
            "Layer normalization moved to input of each sub-block",
            "Scaled initialization by depth",
        ],
        use_cases=["text generation", "summarization", "translation (zero-shot)"],
    ),
    ArchitectureInfo(
        name="GPT-3",
        family="GPT",
        architecture_type="decoder-only",
        attention_type="causal self-attention with alternating dense and sparse",
        positional_encoding="learned absolute",
        context_length=2048,
        parameters="175B",
        training_objective="next-token prediction (autoregressive LM)",
        key_innovations=[
            "In-context learning (few-shot, one-shot, zero-shot)",
            "Scaling laws validated at 175B parameters",
            "Sparse attention patterns in alternating layers",
        ],
        use_cases=["few-shot learning", "code generation", "creative writing"],
    ),
    ArchitectureInfo(
        name="GPT-4",
        family="GPT",
        architecture_type="decoder-only",
        attention_type="causal self-attention (details undisclosed)",
        positional_encoding="undisclosed (likely RoPE or ALiBi)",
        context_length=128_000,
        parameters="undisclosed (rumored MoE ~1.8T)",
        training_objective="next-token prediction + RLHF",
        key_innovations=[
            "Multimodal (text + vision)",
            "Significantly improved reasoning and safety",
            "128K context window",
        ],
        use_cases=["complex reasoning", "multimodal analysis", "code", "agents"],
    ),
]

_LLAMA_FAMILY: list[ArchitectureInfo] = [
    ArchitectureInfo(
        name="LLaMA 2",
        family="LLaMA",
        architecture_type="decoder-only",
        attention_type="grouped-query attention (GQA)",
        positional_encoding="RoPE (Rotary Position Embedding)",
        context_length=4096,
        parameters="7B / 13B / 70B",
        training_objective="next-token prediction + RLHF (chat variants)",
        key_innovations=[
            "Grouped-query attention for efficient inference",
            "RoPE positional encoding for length generalization",
            "Open-weight release enabling community research",
        ],
        use_cases=["open-source chat", "fine-tuning", "research"],
    ),
    ArchitectureInfo(
        name="LLaMA 3.1",
        family="LLaMA",
        architecture_type="decoder-only",
        attention_type="grouped-query attention (GQA)",
        positional_encoding="RoPE with extended context",
        context_length=128_000,
        parameters="8B / 70B / 405B",
        training_objective="next-token prediction + RLHF + DPO",
        key_innovations=[
            "128K context length via RoPE scaling",
            "405B dense model competitive with GPT-4",
            "Multilingual and tool-use capabilities",
        ],
        use_cases=["multilingual", "long-context", "tool use", "code generation"],
    ),
]

_CLAUDE_FAMILY: list[ArchitectureInfo] = [
    ArchitectureInfo(
        name="Claude 3.5 Sonnet",
        family="Claude",
        architecture_type="decoder-only",
        attention_type="undisclosed (likely optimized self-attention)",
        positional_encoding="undisclosed",
        context_length=200_000,
        parameters="undisclosed",
        training_objective="RLHF + Constitutional AI (CAI)",
        key_innovations=[
            "Constitutional AI for alignment",
            "200K context window",
            "Strong instruction following and safety",
        ],
        use_cases=["long-document analysis", "coding", "reasoning", "safety-critical"],
    ),
    ArchitectureInfo(
        name="Claude 4 Opus",
        family="Claude",
        architecture_type="decoder-only",
        attention_type="undisclosed",
        positional_encoding="undisclosed",
        context_length=200_000,
        parameters="undisclosed",
        training_objective="RLHF + Constitutional AI (CAI)",
        key_innovations=[
            "State-of-the-art reasoning",
            "Extended thinking capabilities",
            "Hybrid extended-thinking + tool-use",
        ],
        use_cases=["deep research", "complex reasoning", "agentic tasks", "coding"],
    ),
]

_ENCODER_MODELS: list[ArchitectureInfo] = [
    ArchitectureInfo(
        name="BERT",
        family="BERT",
        architecture_type="encoder-only",
        attention_type="bidirectional self-attention",
        positional_encoding="learned absolute + segment embeddings",
        context_length=512,
        parameters="110M (base) / 340M (large)",
        training_objective="masked language modeling (MLM) + next sentence prediction (NSP)",
        key_innovations=[
            "Bidirectional pre-training via masking",
            "Fine-tuning paradigm for downstream tasks",
            "Segment embeddings for sentence-pair tasks",
        ],
        use_cases=["classification", "NER", "question answering", "embeddings"],
    ),
]

_ENCODER_DECODER_MODELS: list[ArchitectureInfo] = [
    ArchitectureInfo(
        name="T5",
        family="T5",
        architecture_type="encoder-decoder",
        attention_type="bidirectional (encoder) + causal cross-attention (decoder)",
        positional_encoding="relative positional bias",
        context_length=512,
        parameters="60M to 11B",
        training_objective="span corruption (denoising)",
        key_innovations=[
            "Text-to-text framing for all NLP tasks",
            "Relative positional biases instead of sinusoidal",
            "Systematic study of transfer learning approaches",
        ],
        use_cases=["translation", "summarization", "question answering", "classification"],
    ),
]


# ---------------------------------------------------------------------------
# Explorer class
# ---------------------------------------------------------------------------


class TransformerExplorer:
    """Provides structured data about transformer architectures, attention, and positional encoding."""

    def get_all_architectures(self) -> list[ArchitectureInfo]:
        return (
            _GPT_FAMILY
            + _LLAMA_FAMILY
            + _CLAUDE_FAMILY
            + _ENCODER_MODELS
            + _ENCODER_DECODER_MODELS
        )

    def get_architecture(self, name: str) -> ArchitectureInfo | None:
        for arch in self.get_all_architectures():
            if arch.name.lower() == name.lower():
                return arch
        return None

    def get_model_families(self) -> list[ModelFamilyInfo]:
        return [
            ModelFamilyInfo(
                family_name="GPT",
                organization="OpenAI",
                description="Generative Pre-trained Transformer series. Pioneered large-scale "
                "autoregressive language models and in-context learning.",
                models=_GPT_FAMILY,
                timeline=[
                    {"year": "2018", "event": "GPT-1 released (117M params)"},
                    {"year": "2019", "event": "GPT-2 released (1.5B params)"},
                    {"year": "2020", "event": "GPT-3 released (175B params)"},
                    {"year": "2023", "event": "GPT-4 released (multimodal)"},
                    {"year": "2024", "event": "GPT-4o released (omni-modal)"},
                ],
            ),
            ModelFamilyInfo(
                family_name="LLaMA",
                organization="Meta AI",
                description="Large Language Model Meta AI. Open-weight models that democratized "
                "LLM research and enabled widespread fine-tuning.",
                models=_LLAMA_FAMILY,
                timeline=[
                    {"year": "2023-02", "event": "LLaMA 1 released (7B-65B)"},
                    {"year": "2023-07", "event": "LLaMA 2 released with RLHF chat variants"},
                    {"year": "2024-04", "event": "LLaMA 3 released (8B/70B)"},
                    {"year": "2024-07", "event": "LLaMA 3.1 released (up to 405B, 128K ctx)"},
                ],
            ),
            ModelFamilyInfo(
                family_name="Claude",
                organization="Anthropic",
                description="Claude model family built with Constitutional AI. Emphasizes "
                "safety, helpfulness, and long-context understanding.",
                models=_CLAUDE_FAMILY,
                timeline=[
                    {"year": "2023-03", "event": "Claude 1 released"},
                    {"year": "2023-07", "event": "Claude 2 released (100K context)"},
                    {"year": "2024-03", "event": "Claude 3 family (Haiku/Sonnet/Opus)"},
                    {"year": "2024-06", "event": "Claude 3.5 Sonnet released"},
                    {"year": "2025-05", "event": "Claude 4 Opus released"},
                ],
            ),
        ]

    def get_architecture_comparison(self) -> dict[str, Any]:
        return {
            "architecture_types": {
                "encoder-only": {
                    "description": "Processes input bidirectionally. Best for understanding tasks.",
                    "attention": "Full bidirectional self-attention",
                    "examples": ["BERT", "RoBERTa", "ALBERT", "DeBERTa"],
                    "strengths": ["classification", "NER", "embeddings", "similarity"],
                },
                "decoder-only": {
                    "description": "Autoregressive generation. Each token attends only to previous tokens.",
                    "attention": "Causal (masked) self-attention",
                    "examples": ["GPT-4", "Claude", "LLaMA", "Mistral"],
                    "strengths": ["text generation", "chat", "code", "reasoning"],
                },
                "encoder-decoder": {
                    "description": "Encoder reads full input; decoder generates output autoregressively.",
                    "attention": "Bidirectional (encoder) + causal cross-attention (decoder)",
                    "examples": ["T5", "BART", "mBART", "Flan-T5"],
                    "strengths": ["translation", "summarization", "seq2seq tasks"],
                },
            },
            "training_objectives": {
                "CLM": "Causal Language Modeling -- predict the next token (GPT, LLaMA, Claude)",
                "MLM": "Masked Language Modeling -- predict masked tokens (BERT)",
                "span_corruption": "Span Corruption -- reconstruct corrupted spans (T5)",
                "RLHF": "Reinforcement Learning from Human Feedback -- align with human preferences",
                "DPO": "Direct Preference Optimization -- simplified alignment without reward model",
                "CAI": "Constitutional AI -- self-supervised alignment via principles (Claude)",
            },
        }

    # -- attention visualization data ---------------------------------------

    def get_attention_patterns(self, seq_length: int = 8) -> dict[str, Any]:
        """Generate attention weight matrices for different pattern types."""
        causal_mask = np.tril(np.ones((seq_length, seq_length)))
        causal_weights = causal_mask / causal_mask.sum(axis=-1, keepdims=True)

        bidirectional = np.ones((seq_length, seq_length)) / seq_length

        # Sliding window attention (like Mistral)
        window = 3
        sliding = np.zeros((seq_length, seq_length))
        for i in range(seq_length):
            start = max(0, i - window + 1)
            sliding[i, start : i + 1] = 1.0
        sliding = sliding / sliding.sum(axis=-1, keepdims=True)

        return {
            "causal": {
                "description": "Causal (autoregressive) -- each position attends only to earlier positions",
                "weights": causal_weights.tolist(),
                "mask": causal_mask.tolist(),
            },
            "bidirectional": {
                "description": "Bidirectional -- each position attends to all positions equally",
                "weights": bidirectional.tolist(),
            },
            "sliding_window": {
                "description": f"Sliding window (size={window}) -- each position attends to a local window",
                "weights": sliding.tolist(),
            },
        }

    # -- positional encoding demo -------------------------------------------

    def get_positional_encoding_demo(
        self,
        max_position: int = 64,
        d_model: int = 128,
    ) -> list[PositionalEncodingDemo]:
        """Generate sinusoidal positional encoding values for visualization."""
        positions = list(range(min(max_position, 32)))
        dims = list(range(0, min(d_model, 16), 2))

        sinusoidal_values: list[list[float]] = []
        for pos in positions:
            row: list[float] = []
            for d in dims:
                angle = pos / (10_000 ** (d / d_model))
                row.append(round(math.sin(angle), 6))
                row.append(round(math.cos(angle), 6))
            sinusoidal_values.append(row)

        dim_labels = []
        for d in dims:
            dim_labels.extend([d, d + 1])

        return [
            PositionalEncodingDemo(
                method="sinusoidal",
                description=(
                    "Original Transformer (Vaswani et al. 2017). Uses sin/cos functions "
                    "at different frequencies. Fixed, not learned."
                ),
                positions=positions,
                dimensions=dim_labels,
                values=sinusoidal_values,
            ),
            PositionalEncodingDemo(
                method="RoPE",
                description=(
                    "Rotary Position Embedding (Su et al. 2021). Encodes position by "
                    "rotating query/key vectors. Enables relative position awareness and "
                    "extrapolation to longer sequences. Used by LLaMA, Mistral, Gemma."
                ),
                positions=positions,
                dimensions=dim_labels,
                values=sinusoidal_values,  # simplified representation
            ),
            PositionalEncodingDemo(
                method="ALiBi",
                description=(
                    "Attention with Linear Biases (Press et al. 2022). Adds a linear bias "
                    "based on distance directly to attention scores. No positional embeddings "
                    "added to token representations. Excellent length extrapolation."
                ),
                positions=positions,
                dimensions=list(range(8)),  # 8 heads
                values=[
                    [round(-0.5**h * abs(i - j), 4) for h in range(8) for j in range(1)]
                    for i in positions
                ],
            ),
        ]

    # -- pre-training concepts ----------------------------------------------

    def get_pretraining_overview(self) -> dict[str, Any]:
        return {
            "stages": [
                {
                    "name": "Data Collection",
                    "description": (
                        "Gathering massive text corpora from the web (Common Crawl, Wikipedia, "
                        "books, code repositories). Scale ranges from hundreds of GB to multiple TB."
                    ),
                    "challenges": [
                        "Copyright and licensing concerns",
                        "Bias in web-crawled data",
                        "Ensuring diversity of domains and languages",
                        "PII and sensitive content filtering",
                    ],
                },
                {
                    "name": "Data Cleaning",
                    "description": (
                        "Filtering, deduplication, and quality assessment. Removes boilerplate, "
                        "near-duplicates, toxic content, and low-quality text."
                    ),
                    "techniques": [
                        "MinHash / SimHash deduplication",
                        "Perplexity-based quality filtering",
                        "Language identification and filtering",
                        "Heuristic rules (line length, symbol ratio, repetition)",
                        "Classifier-based quality scoring",
                    ],
                },
                {
                    "name": "Tokenization",
                    "description": (
                        "Converting raw text to integer sequences. BPE (Byte Pair Encoding) is the "
                        "dominant approach, balancing vocabulary size with coverage."
                    ),
                    "algorithms": [
                        "BPE -- iteratively merges most frequent byte pairs (GPT, LLaMA)",
                        "WordPiece -- similar to BPE but uses likelihood-based merges (BERT)",
                        "Unigram -- starts with large vocab and prunes (T5, mBART)",
                        "SentencePiece -- language-agnostic tokenization (LLaMA, T5)",
                    ],
                },
                {
                    "name": "Pre-Training",
                    "description": (
                        "Self-supervised training on the tokenized corpus. The model learns "
                        "language structure, facts, and reasoning patterns."
                    ),
                    "details": [
                        "Objective: next-token prediction (decoder) or masked LM (encoder)",
                        "Hardware: thousands of GPUs/TPUs for weeks to months",
                        "Optimization: AdamW, cosine learning rate schedule, gradient clipping",
                        "Cost: millions of dollars for frontier models",
                    ],
                },
                {
                    "name": "Post-Training Alignment",
                    "description": (
                        "Fine-tuning on curated instruction data and aligning with human "
                        "preferences via RLHF, DPO, or Constitutional AI."
                    ),
                    "methods": [
                        "Supervised Fine-Tuning (SFT) on instruction-response pairs",
                        "RLHF: train reward model then optimize policy with PPO",
                        "DPO: directly optimize on preference pairs without reward model",
                        "Constitutional AI: self-critique guided by principles",
                    ],
                },
            ],
        }

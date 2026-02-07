"""Comprehensive tokenizer module demonstrating BPE, WordPiece, and Unigram tokenization.

Provides side-by-side comparison of tokenization strategies used across major LLM
providers (OpenAI via tiktoken, HuggingFace models), with utilities for token
counting, encoding/decoding, and vocabulary analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog
import tiktoken
from tokenizers import Tokenizer as HFTokenizerBase
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

TIKTOKEN_ENCODINGS: dict[str, str] = {
    "cl100k_base": "GPT-4 / GPT-3.5-turbo",
    "o200k_base": "GPT-4o",
    "p50k_base": "text-davinci-003 / Codex",
    "r50k_base": "GPT-3 (davinci)",
}

HUGGINGFACE_TOKENIZERS: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "Llama 3.1 (BPE)",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B (BPE/SentencePiece)",
    "google/gemma-2-9b-it": "Gemma 2 (SentencePiece)",
    "bert-base-uncased": "BERT (WordPiece)",
    "google/flan-t5-base": "Flan-T5 (SentencePiece/Unigram)",
}


@dataclass(frozen=True, slots=True)
class TokenizeResult:
    tokenizer_name: str
    text: str
    tokens: list[int]
    token_strings: list[str]
    num_tokens: int
    encoding_time_ms: float


@dataclass(frozen=True, slots=True)
class VocabStats:
    tokenizer_name: str
    vocab_size: int
    tokenizer_type: str
    description: str
    special_tokens: list[str]
    sample_tokens: list[str]


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    text: str
    results: list[TokenizeResult]
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer service
# ---------------------------------------------------------------------------


class TokenizerService:
    """Unified interface for tiktoken and HuggingFace tokenizers."""

    def __init__(self) -> None:
        self._tiktoken_cache: dict[str, tiktoken.Encoding] = {}
        self._hf_cache: dict[str, Any] = {}

    # -- internal loaders ---------------------------------------------------

    def _get_tiktoken(self, name: str) -> tiktoken.Encoding:
        if name not in self._tiktoken_cache:
            self._tiktoken_cache[name] = tiktoken.get_encoding(name)
        return self._tiktoken_cache[name]

    def _get_hf_tokenizer(self, name: str) -> Any:
        if name not in self._hf_cache:
            from transformers import AutoTokenizer

            self._hf_cache[name] = AutoTokenizer.from_pretrained(
                name, trust_remote_code=True
            )
        return self._hf_cache[name]

    def _is_tiktoken(self, name: str) -> bool:
        return name in TIKTOKEN_ENCODINGS

    # -- public API ---------------------------------------------------------

    def encode(self, text: str, tokenizer_name: str) -> TokenizeResult:
        start = time.perf_counter()

        if self._is_tiktoken(tokenizer_name):
            enc = self._get_tiktoken(tokenizer_name)
            tokens = enc.encode(text, allowed_special="all")
            token_strings = [enc.decode([t]) for t in tokens]
        else:
            tok = self._get_hf_tokenizer(tokenizer_name)
            encoding = tok(text, add_special_tokens=False)
            tokens = encoding["input_ids"]
            token_strings = tok.convert_ids_to_tokens(tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.debug(
            "tokenize_complete",
            tokenizer=tokenizer_name,
            num_tokens=len(tokens),
            elapsed_ms=round(elapsed_ms, 2),
        )

        return TokenizeResult(
            tokenizer_name=tokenizer_name,
            text=text,
            tokens=tokens,
            token_strings=token_strings,
            num_tokens=len(tokens),
            encoding_time_ms=round(elapsed_ms, 2),
        )

    def decode(self, tokens: list[int], tokenizer_name: str) -> str:
        if self._is_tiktoken(tokenizer_name):
            return self._get_tiktoken(tokenizer_name).decode(tokens)
        return self._get_hf_tokenizer(tokenizer_name).decode(tokens)

    def compare_tokenizers(
        self, text: str, tokenizer_names: list[str]
    ) -> ComparisonResult:
        results = [self.encode(text, name) for name in tokenizer_names]

        token_counts = {r.tokenizer_name: r.num_tokens for r in results}
        most_efficient = min(token_counts, key=token_counts.get)  # type: ignore[arg-type]
        least_efficient = max(token_counts, key=token_counts.get)  # type: ignore[arg-type]

        summary = {
            "token_counts": token_counts,
            "most_efficient": most_efficient,
            "least_efficient": least_efficient,
            "efficiency_ratio": round(
                token_counts[least_efficient] / max(token_counts[most_efficient], 1),
                2,
            ),
        }

        return ComparisonResult(text=text, results=results, summary=summary)

    def get_vocab_stats(self, tokenizer_name: str) -> VocabStats:
        if self._is_tiktoken(tokenizer_name):
            enc = self._get_tiktoken(tokenizer_name)
            special = list(enc.special_tokens_set)
            # tiktoken exposes max_token_value; vocab_size ~ max_token_value + 1
            vocab_size = enc.max_token_value + 1
            sample = [enc.decode([i]) for i in range(100, 120)]
            return VocabStats(
                tokenizer_name=tokenizer_name,
                vocab_size=vocab_size,
                tokenizer_type="BPE (tiktoken)",
                description=TIKTOKEN_ENCODINGS.get(tokenizer_name, "Unknown"),
                special_tokens=special,
                sample_tokens=sample,
            )

        tok = self._get_hf_tokenizer(tokenizer_name)
        vocab_size = tok.vocab_size
        special = list(tok.all_special_tokens)
        sample = [tok.decode([i]) for i in range(100, 120)]
        return VocabStats(
            tokenizer_name=tokenizer_name,
            vocab_size=vocab_size,
            tokenizer_type="HuggingFace AutoTokenizer",
            description=HUGGINGFACE_TOKENIZERS.get(tokenizer_name, tokenizer_name),
            special_tokens=special,
            sample_tokens=sample,
        )

    def list_available(self) -> dict[str, dict[str, str]]:
        return {
            "tiktoken": TIKTOKEN_ENCODINGS,
            "huggingface": HUGGINGFACE_TOKENIZERS,
        }

    # -- educational utility ------------------------------------------------

    @staticmethod
    def train_bpe_demo(corpus: list[str], vocab_size: int = 500) -> dict[str, Any]:
        """Train a tiny BPE tokenizer from scratch on the provided corpus.

        This is purely educational -- it shows how BPE merges work on a small
        dataset so callers can visualize the vocabulary construction process.
        """
        tokenizer = HFTokenizerBase(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            show_progress=False,
        )
        tokenizer.train_from_iterator(corpus, trainer=trainer)

        vocab = tokenizer.get_vocab()
        sample_encoding = tokenizer.encode(corpus[0] if corpus else "hello world")

        return {
            "vocab_size": len(vocab),
            "sample_tokens": sample_encoding.tokens,
            "sample_ids": sample_encoding.ids,
            "first_50_vocab": sorted(vocab, key=vocab.get)[:50],  # type: ignore[arg-type]
        }

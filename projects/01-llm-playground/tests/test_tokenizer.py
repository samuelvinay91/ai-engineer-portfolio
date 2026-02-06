"""Tests for the TokenizerService covering encoding, decoding, comparison, and vocab stats."""

from __future__ import annotations

import pytest

from llm_playground.tokenizer import TokenizerService


@pytest.fixture(scope="module")
def svc() -> TokenizerService:
    return TokenizerService()


class TestEncode:
    def test_encode_cl100k_returns_tokens(self, svc: TokenizerService) -> None:
        result = svc.encode("Hello, world!", "cl100k_base")
        assert result.num_tokens > 0
        assert len(result.tokens) == result.num_tokens
        assert len(result.token_strings) == result.num_tokens
        assert result.tokenizer_name == "cl100k_base"
        assert result.text == "Hello, world!"

    def test_encode_o200k_returns_tokens(self, svc: TokenizerService) -> None:
        result = svc.encode("Hello, world!", "o200k_base")
        assert result.num_tokens > 0

    def test_encode_empty_string(self, svc: TokenizerService) -> None:
        result = svc.encode("", "cl100k_base")
        assert result.num_tokens == 0
        assert result.tokens == []

    def test_encode_unicode(self, svc: TokenizerService) -> None:
        result = svc.encode("Bonjour le monde! Hola mundo!", "cl100k_base")
        assert result.num_tokens > 0

    def test_encode_long_text(self, svc: TokenizerService) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 100
        result = svc.encode(text, "cl100k_base")
        assert result.num_tokens > 100

    def test_encode_records_timing(self, svc: TokenizerService) -> None:
        result = svc.encode("test", "cl100k_base")
        assert result.encoding_time_ms >= 0

    def test_encode_invalid_tokenizer_raises(self, svc: TokenizerService) -> None:
        with pytest.raises(Exception):
            svc.encode("hello", "nonexistent_tokenizer_xyz")


class TestDecode:
    def test_roundtrip_cl100k(self, svc: TokenizerService) -> None:
        original = "Hello, world!"
        encoded = svc.encode(original, "cl100k_base")
        decoded = svc.decode(encoded.tokens, "cl100k_base")
        assert decoded == original

    def test_roundtrip_o200k(self, svc: TokenizerService) -> None:
        original = "Tokenization is fascinating."
        encoded = svc.encode(original, "o200k_base")
        decoded = svc.decode(encoded.tokens, "o200k_base")
        assert decoded == original

    def test_decode_empty_list(self, svc: TokenizerService) -> None:
        decoded = svc.decode([], "cl100k_base")
        assert decoded == ""


class TestCompareTokenizers:
    def test_compare_two_tiktoken(self, svc: TokenizerService) -> None:
        result = svc.compare_tokenizers(
            "Hello, world!", ["cl100k_base", "o200k_base"]
        )
        assert len(result.results) == 2
        assert result.text == "Hello, world!"
        assert "token_counts" in result.summary
        assert "most_efficient" in result.summary
        assert "least_efficient" in result.summary
        assert result.summary["efficiency_ratio"] >= 1.0

    def test_compare_single_tokenizer(self, svc: TokenizerService) -> None:
        result = svc.compare_tokenizers("test", ["cl100k_base"])
        assert len(result.results) == 1
        assert result.summary["efficiency_ratio"] == 1.0

    def test_compare_different_text_lengths(self, svc: TokenizerService) -> None:
        short = svc.compare_tokenizers("Hi", ["cl100k_base", "o200k_base"])
        long = svc.compare_tokenizers(
            "This is a significantly longer sentence for comparison.",
            ["cl100k_base", "o200k_base"],
        )
        assert short.results[0].num_tokens <= long.results[0].num_tokens


class TestVocabStats:
    def test_vocab_stats_cl100k(self, svc: TokenizerService) -> None:
        stats = svc.get_vocab_stats("cl100k_base")
        assert stats.vocab_size > 0
        assert stats.tokenizer_type == "BPE (tiktoken)"
        assert len(stats.sample_tokens) > 0
        assert stats.tokenizer_name == "cl100k_base"

    def test_vocab_stats_o200k(self, svc: TokenizerService) -> None:
        stats = svc.get_vocab_stats("o200k_base")
        assert stats.vocab_size > stats.__class__.__new__(stats.__class__).vocab_size or stats.vocab_size > 0


class TestListAvailable:
    def test_list_returns_both_categories(self, svc: TokenizerService) -> None:
        available = svc.list_available()
        assert "tiktoken" in available
        assert "huggingface" in available
        assert len(available["tiktoken"]) >= 2


class TestTrainBPEDemo:
    def test_train_small_corpus(self) -> None:
        corpus = [
            "The cat sat on the mat.",
            "The dog ran in the park.",
            "A bird flew over the lake.",
        ] * 10
        result = TokenizerService.train_bpe_demo(corpus, vocab_size=200)
        assert result["vocab_size"] <= 200
        assert len(result["sample_tokens"]) > 0
        assert len(result["sample_ids"]) > 0
        assert len(result["first_50_vocab"]) <= 50

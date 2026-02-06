"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-playground"
        assert "version" in data


class TestTokenizeEndpoints:
    @pytest.mark.asyncio
    async def test_tokenize_default_tokenizer(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/tokenize",
            json={"text": "Hello, world!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] > 0
        assert data["tokenizer_name"] == "cl100k_base"
        assert len(data["tokens"]) == data["num_tokens"]
        assert len(data["token_strings"]) == data["num_tokens"]

    @pytest.mark.asyncio
    async def test_tokenize_specific_tokenizer(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/tokenize",
            json={"text": "Testing o200k", "tokenizer_name": "o200k_base"},
        )
        assert resp.status_code == 200
        assert resp.json()["tokenizer_name"] == "o200k_base"

    @pytest.mark.asyncio
    async def test_tokenize_empty_text_rejected(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/tokenize",
            json={"text": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_tokenize_invalid_tokenizer(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/tokenize",
            json={"text": "hello", "tokenizer_name": "nonexistent"},
        )
        assert resp.status_code == 400


class TestCompareTokenizers:
    @pytest.mark.asyncio
    async def test_compare_default_tokenizers(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/compare-tokenizers",
            json={"text": "Hello, world!"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3
        assert "summary" in data
        assert "most_efficient" in data["summary"]

    @pytest.mark.asyncio
    async def test_compare_custom_tokenizers(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/compare-tokenizers",
            json={
                "text": "Compare these!",
                "tokenizer_names": ["cl100k_base", "o200k_base"],
            },
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 2


class TestDecodeEndpoint:
    @pytest.mark.asyncio
    async def test_decode_valid_tokens(self, client: AsyncClient) -> None:
        # First encode, then decode
        encode_resp = await client.post(
            "/api/v1/tokenize",
            json={"text": "Hello"},
        )
        tokens = encode_resp.json()["tokens"]

        decode_resp = await client.post(
            "/api/v1/decode",
            json={"tokens": tokens, "tokenizer_name": "cl100k_base"},
        )
        assert decode_resp.status_code == 200
        assert decode_resp.json()["text"] == "Hello"


class TestListEndpoints:
    @pytest.mark.asyncio
    async def test_list_tokenizers(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/tokenizers")
        assert resp.status_code == 200
        data = resp.json()
        assert "tiktoken" in data
        assert "huggingface" in data

    @pytest.mark.asyncio
    async def test_list_models(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "anthropic" in data
        assert "openai" in data
        assert "open_source" in data
        assert len(data["all"]) > 0

    @pytest.mark.asyncio
    async def test_list_strategies(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/strategies")
        assert resp.status_code == 200
        strategies = resp.json()["strategies"]
        names = {s["name"] for s in strategies}
        assert "greedy" in names
        assert "top_k" in names
        assert "top_p" in names


class TestArchitectureEndpoints:
    @pytest.mark.asyncio
    async def test_get_architectures(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/architectures")
        assert resp.status_code == 200
        archs = resp.json()["architectures"]
        assert len(archs) > 0
        names = {a["name"] for a in archs}
        assert "GPT-4" in names
        assert "BERT" in names

    @pytest.mark.asyncio
    async def test_get_single_architecture(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/architectures/GPT-4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "GPT-4"
        assert data["family"] == "GPT"
        assert len(data["key_innovations"]) > 0

    @pytest.mark.asyncio
    async def test_get_unknown_architecture_404(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/architectures/NonExistentModel")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_model_families(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/model-families")
        assert resp.status_code == 200
        families = resp.json()["families"]
        family_names = {f["family_name"] for f in families}
        assert "GPT" in family_names
        assert "LLaMA" in family_names
        assert "Claude" in family_names

    @pytest.mark.asyncio
    async def test_architecture_comparison(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/architecture-comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert "architecture_types" in data
        assert "decoder-only" in data["architecture_types"]
        assert "encoder-only" in data["architecture_types"]
        assert "encoder-decoder" in data["architecture_types"]

    @pytest.mark.asyncio
    async def test_attention_patterns(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/attention-patterns", params={"seq_length": 4})
        assert resp.status_code == 200
        data = resp.json()
        assert "causal" in data
        assert "bidirectional" in data
        assert "sliding_window" in data
        # Verify causal mask shape
        assert len(data["causal"]["weights"]) == 4
        assert len(data["causal"]["weights"][0]) == 4

    @pytest.mark.asyncio
    async def test_attention_patterns_invalid_length(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/attention-patterns", params={"seq_length": 100})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_positional_encoding(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/positional-encoding")
        assert resp.status_code == 200
        encodings = resp.json()["encodings"]
        assert len(encodings) >= 2
        methods = {e["method"] for e in encodings}
        assert "sinusoidal" in methods
        assert "RoPE" in methods

    @pytest.mark.asyncio
    async def test_pretraining_overview(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/pretraining")
        assert resp.status_code == 200
        stages = resp.json()["stages"]
        assert len(stages) >= 4
        stage_names = {s["name"] for s in stages}
        assert "Data Collection" in stage_names
        assert "Tokenization" in stage_names


class TestSamplingSim:
    @pytest.mark.asyncio
    async def test_simulate_greedy(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/simulate-sampling",
            json={"strategy": "greedy"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "greedy"
        assert "selected_token" in data
        # Greedy: filtered_probs should have exactly one 1.0
        assert max(data["filtered_probs"]) == 1.0

    @pytest.mark.asyncio
    async def test_simulate_top_k(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/simulate-sampling",
            json={"strategy": "top_k", "top_k": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "top_k"
        nonzero = [p for p in data["filtered_probs"] if p > 0]
        assert len(nonzero) <= 3

    @pytest.mark.asyncio
    async def test_simulate_top_p(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/simulate-sampling",
            json={"strategy": "top_p", "top_p": 0.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "top_p"


class TestVocabStats:
    @pytest.mark.asyncio
    async def test_vocab_stats_cl100k(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/vocab-stats/cl100k_base")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vocab_size"] > 0
        assert data["tokenizer_type"] == "BPE (tiktoken)"
        assert len(data["sample_tokens"]) > 0


class TestTrainBPE:
    @pytest.mark.asyncio
    async def test_train_bpe_demo(self, client: AsyncClient) -> None:
        corpus = ["the cat sat on the mat"] * 20
        resp = await client.post(
            "/api/v1/train-bpe",
            json={"corpus": corpus, "vocab_size": 100},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["vocab_size"] <= 100
        assert len(data["sample_tokens"]) > 0

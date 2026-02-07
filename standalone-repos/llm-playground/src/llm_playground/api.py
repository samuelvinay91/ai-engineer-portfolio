"""FastAPI application exposing tokenization, generation, and architecture exploration endpoints."""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common import ErrorResponse, HealthResponse

from llm_playground.config import ModelInfo, get_settings
from llm_playground.generation import (
    GenerationRequest,
    GenerationService,
    GenerationStrategy,
    simulate_sampling,
)
from llm_playground.tokenizer import TokenizerService
from llm_playground.transformer import TransformerExplorer

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="LLM Playground",
        description="Interactive exploration of tokenization, text generation strategies, and transformer architectures",
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- singletons ---------------------------------------------------------
    tokenizer_svc = TokenizerService()
    generation_svc = GenerationService(settings)
    transformer_explorer = TransformerExplorer()

    # -- error handling -----------------------------------------------------

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=str(exc.detail),
                status_code=exc.status_code,
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                status_code=500,
            ).model_dump(),
        )

    # -- health -------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service=settings.app_name,
            version=settings.app_version,
        )

    # -- tokenizer routes ---------------------------------------------------

    class TokenizeRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=100_000)
        tokenizer_name: str = Field(default="cl100k_base")

    class DecodeRequest(BaseModel):
        tokens: list[int]
        tokenizer_name: str = Field(default="cl100k_base")

    class CompareTokenizersRequest(BaseModel):
        text: str = Field(..., min_length=1, max_length=100_000)
        tokenizer_names: list[str] = Field(
            default=["cl100k_base", "o200k_base", "p50k_base"]
        )

    class TrainBPERequest(BaseModel):
        corpus: list[str] = Field(..., min_length=1)
        vocab_size: int = Field(default=500, ge=50, le=10_000)

    @app.post("/api/v1/tokenize")
    async def tokenize(req: TokenizeRequest) -> dict[str, Any]:
        try:
            result = tokenizer_svc.encode(req.text, req.tokenizer_name)
            return {
                "tokenizer_name": result.tokenizer_name,
                "text": result.text,
                "tokens": result.tokens,
                "token_strings": result.token_strings,
                "num_tokens": result.num_tokens,
                "encoding_time_ms": result.encoding_time_ms,
            }
        except Exception as exc:
            log.error("tokenize_error", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/decode")
    async def decode(req: DecodeRequest) -> dict[str, str]:
        try:
            text = tokenizer_svc.decode(req.tokens, req.tokenizer_name)
            return {"text": text, "tokenizer_name": req.tokenizer_name}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/compare-tokenizers")
    async def compare_tokenizers(req: CompareTokenizersRequest) -> dict[str, Any]:
        try:
            result = tokenizer_svc.compare_tokenizers(req.text, req.tokenizer_names)
            return {
                "text": result.text,
                "results": [
                    {
                        "tokenizer_name": r.tokenizer_name,
                        "tokens": r.tokens,
                        "token_strings": r.token_strings,
                        "num_tokens": r.num_tokens,
                        "encoding_time_ms": r.encoding_time_ms,
                    }
                    for r in result.results
                ],
                "summary": result.summary,
            }
        except Exception as exc:
            log.error("compare_error", error=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/train-bpe")
    async def train_bpe(req: TrainBPERequest) -> dict[str, Any]:
        try:
            return tokenizer_svc.train_bpe_demo(req.corpus, req.vocab_size)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/v1/tokenizers")
    async def list_tokenizers() -> dict[str, dict[str, str]]:
        return tokenizer_svc.list_available()

    @app.get("/api/v1/vocab-stats/{tokenizer_name:path}")
    async def vocab_stats(tokenizer_name: str) -> dict[str, Any]:
        try:
            stats = tokenizer_svc.get_vocab_stats(tokenizer_name)
            return {
                "tokenizer_name": stats.tokenizer_name,
                "vocab_size": stats.vocab_size,
                "tokenizer_type": stats.tokenizer_type,
                "description": stats.description,
                "special_tokens": stats.special_tokens,
                "sample_tokens": stats.sample_tokens,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- generation routes --------------------------------------------------

    @app.post("/api/v1/generate")
    async def generate(req: GenerationRequest) -> dict[str, Any]:
        try:
            result = await generation_svc.generate(req)
            return {
                "text": result.text,
                "model": result.model,
                "strategy": result.strategy,
                "tokens_used": result.tokens_used,
                "generation_time_ms": result.generation_time_ms,
                "metadata": result.metadata,
            }
        except Exception as exc:
            log.error("generation_error", error=str(exc))
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/v1/generate/stream")
    async def generate_stream(req: GenerationRequest) -> EventSourceResponse:
        async def event_generator():
            try:
                async for chunk in generation_svc.generate_stream(req):
                    yield {"event": "token", "data": json.dumps({"text": chunk})}
                yield {"event": "done", "data": json.dumps({"status": "complete"})}
            except Exception as exc:
                log.error("stream_error", error=str(exc))
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}),
                }

        return EventSourceResponse(event_generator())

    class SamplingSimRequest(BaseModel):
        vocab_labels: list[str] = Field(
            default=["the", "a", "cat", "dog", "sat", "on", "mat", "ran", "big", "red"]
        )
        logits: list[float] = Field(
            default=[2.0, 1.5, 1.8, 1.2, 0.9, 0.5, 0.3, 0.7, 0.4, 0.6]
        )
        strategy: GenerationStrategy = GenerationStrategy.TOP_P
        temperature: float = 1.0
        top_k: int = 5
        top_p: float = 0.9

    @app.post("/api/v1/simulate-sampling")
    async def simulate_sampling_endpoint(req: SamplingSimRequest) -> dict[str, Any]:
        import numpy as np

        return simulate_sampling(
            vocab_labels=req.vocab_labels,
            logits=np.array(req.logits),
            strategy=req.strategy,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

    # -- model info routes --------------------------------------------------

    @app.get("/api/v1/models")
    async def list_models() -> dict[str, list[str]]:
        return {
            "anthropic": ModelInfo.ANTHROPIC,
            "openai": ModelInfo.OPENAI,
            "open_source": ModelInfo.OPEN_SOURCE,
            "all": ModelInfo.all_models(),
        }

    @app.get("/api/v1/strategies")
    async def list_strategies() -> dict[str, list[dict[str, str]]]:
        return {
            "strategies": [
                {
                    "name": GenerationStrategy.GREEDY.value,
                    "description": "Always select the highest-probability token. Deterministic but can be repetitive.",
                },
                {
                    "name": GenerationStrategy.BEAM_SEARCH.value,
                    "description": "Maintain top-k hypotheses and select the highest-scoring complete sequence.",
                },
                {
                    "name": GenerationStrategy.TOP_K.value,
                    "description": "Sample from the top-k most probable tokens. Balances diversity and quality.",
                },
                {
                    "name": GenerationStrategy.TOP_P.value,
                    "description": "Sample from the smallest set of tokens whose cumulative probability exceeds p (nucleus sampling).",
                },
                {
                    "name": GenerationStrategy.TEMPERATURE.value,
                    "description": "Scale logits by temperature before softmax. Higher = more random, lower = more deterministic.",
                },
            ]
        }

    # -- architecture routes ------------------------------------------------

    @app.get("/api/v1/architectures")
    async def get_architectures() -> dict[str, Any]:
        archs = transformer_explorer.get_all_architectures()
        return {
            "architectures": [
                {
                    "name": a.name,
                    "family": a.family,
                    "architecture_type": a.architecture_type,
                    "attention_type": a.attention_type,
                    "positional_encoding": a.positional_encoding,
                    "context_length": a.context_length,
                    "parameters": a.parameters,
                    "training_objective": a.training_objective,
                    "key_innovations": a.key_innovations,
                    "use_cases": a.use_cases,
                }
                for a in archs
            ]
        }

    @app.get("/api/v1/architectures/{name}")
    async def get_architecture(name: str) -> dict[str, Any]:
        arch = transformer_explorer.get_architecture(name)
        if arch is None:
            raise HTTPException(status_code=404, detail=f"Architecture '{name}' not found")
        return {
            "name": arch.name,
            "family": arch.family,
            "architecture_type": arch.architecture_type,
            "attention_type": arch.attention_type,
            "positional_encoding": arch.positional_encoding,
            "context_length": arch.context_length,
            "parameters": arch.parameters,
            "training_objective": arch.training_objective,
            "key_innovations": arch.key_innovations,
            "use_cases": arch.use_cases,
        }

    @app.get("/api/v1/model-families")
    async def get_model_families() -> dict[str, Any]:
        families = transformer_explorer.get_model_families()
        return {
            "families": [
                {
                    "family_name": f.family_name,
                    "organization": f.organization,
                    "description": f.description,
                    "model_count": len(f.models),
                    "timeline": f.timeline,
                }
                for f in families
            ]
        }

    @app.get("/api/v1/architecture-comparison")
    async def get_architecture_comparison() -> dict[str, Any]:
        return transformer_explorer.get_architecture_comparison()

    @app.get("/api/v1/attention-patterns")
    async def get_attention_patterns(seq_length: int = 8) -> dict[str, Any]:
        if seq_length < 2 or seq_length > 64:
            raise HTTPException(status_code=400, detail="seq_length must be between 2 and 64")
        return transformer_explorer.get_attention_patterns(seq_length)

    @app.get("/api/v1/positional-encoding")
    async def get_positional_encoding(
        max_position: int = 64, d_model: int = 128
    ) -> dict[str, Any]:
        demos = transformer_explorer.get_positional_encoding_demo(max_position, d_model)
        return {
            "encodings": [
                {
                    "method": d.method,
                    "description": d.description,
                    "positions": d.positions,
                    "dimensions": d.dimensions,
                    "values": d.values,
                }
                for d in demos
            ]
        }

    @app.get("/api/v1/pretraining")
    async def get_pretraining_overview() -> dict[str, Any]:
        return transformer_explorer.get_pretraining_overview()

    return app


app = create_app()

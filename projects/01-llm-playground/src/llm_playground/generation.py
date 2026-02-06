"""Text generation strategies demonstrating greedy, beam, top-k, top-p, and temperature sampling.

Wraps Anthropic and OpenAI APIs for cloud-based generation, and provides local
simulation utilities that illustrate how each decoding strategy affects token
selection distributions.
"""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import anthropic
import numpy as np
import openai
import structlog
from pydantic import BaseModel, Field

from llm_playground.config import Settings, get_settings

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class GenerationStrategy(StrEnum):
    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"


class GenerationRequest(BaseModel):
    prompt: str
    model: str | None = None
    strategy: GenerationStrategy = GenerationStrategy.TOP_P
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    beam_width: int = Field(default=4, ge=1, le=10)
    repetition_penalty: float = Field(default=1.0, ge=0.5, le=2.0)
    system_prompt: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model: str
    strategy: str
    tokens_used: int
    generation_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sampling simulations (educational, CPU-only)
# ---------------------------------------------------------------------------


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by temperature before softmax."""
    if temperature <= 0:
        return logits
    return logits / temperature


def apply_repetition_penalty(
    logits: np.ndarray, generated_ids: list[int], penalty: float
) -> np.ndarray:
    logits = logits.copy()
    for token_id in set(generated_ids):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()


def greedy_select(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    indices_to_remove = np.argsort(logits)[:-k]
    logits = logits.copy()
    logits[indices_to_remove] = -math.inf
    return logits


def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    sorted_indices = np.argsort(logits)[::-1]
    probs = softmax(logits)
    cumulative = np.cumsum(probs[sorted_indices])
    cutoff_idx = int(np.searchsorted(cumulative, p)) + 1
    keep = set(sorted_indices[:cutoff_idx].tolist())
    logits = logits.copy()
    for i in range(len(logits)):
        if i not in keep:
            logits[i] = -math.inf
    return logits


def sample_from_logits(logits: np.ndarray) -> int:
    probs = softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


def simulate_sampling(
    vocab_labels: list[str],
    logits: np.ndarray,
    strategy: GenerationStrategy,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> dict[str, Any]:
    """Run a single-step sampling simulation and return visualization data.

    Returns the probability distributions before and after filtering so the
    caller can render charts showing how each strategy reshapes token selection.
    """
    original_probs = softmax(logits).tolist()
    modified = apply_temperature(logits, temperature)

    if strategy == GenerationStrategy.GREEDY:
        selected_idx = greedy_select(modified)
        final_probs = [0.0] * len(modified)
        final_probs[selected_idx] = 1.0
    elif strategy == GenerationStrategy.TOP_K:
        modified = top_k_filter(modified, top_k)
        final_probs = softmax(modified).tolist()
        selected_idx = sample_from_logits(modified)
    elif strategy == GenerationStrategy.TOP_P:
        modified = top_p_filter(modified, top_p)
        final_probs = softmax(modified).tolist()
        selected_idx = sample_from_logits(modified)
    else:
        final_probs = softmax(modified).tolist()
        selected_idx = sample_from_logits(modified)

    return {
        "strategy": strategy.value,
        "vocab_labels": vocab_labels,
        "original_probs": original_probs,
        "filtered_probs": final_probs,
        "selected_index": selected_idx,
        "selected_token": vocab_labels[selected_idx],
    }


# ---------------------------------------------------------------------------
# Cloud generation service
# ---------------------------------------------------------------------------


class GenerationService:
    """Orchestrates text generation via Anthropic and OpenAI APIs."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._anthropic: anthropic.AsyncAnthropic | None = None
        self._openai: openai.AsyncOpenAI | None = None

    @property
    def anthropic_client(self) -> anthropic.AsyncAnthropic:
        if self._anthropic is None:
            self._anthropic = anthropic.AsyncAnthropic(
                api_key=self._settings.anthropic_api_key or None
            )
        return self._anthropic

    @property
    def openai_client(self) -> openai.AsyncOpenAI:
        if self._openai is None:
            self._openai = openai.AsyncOpenAI(
                api_key=self._settings.openai_api_key or None
            )
        return self._openai

    def _resolve_model(self, request: GenerationRequest) -> str:
        return request.model or self._settings.default_model

    def _is_anthropic(self, model: str) -> bool:
        return model.startswith("claude")

    def _build_api_params(self, request: GenerationRequest) -> dict[str, float]:
        params: dict[str, float] = {}
        match request.strategy:
            case GenerationStrategy.GREEDY:
                params["temperature"] = 0.0
            case GenerationStrategy.TOP_K:
                params["temperature"] = request.temperature
                params["top_k"] = request.top_k
            case GenerationStrategy.TOP_P:
                params["temperature"] = request.temperature
                params["top_p"] = request.top_p
            case GenerationStrategy.TEMPERATURE:
                params["temperature"] = request.temperature
            case _:
                params["temperature"] = request.temperature
        return params

    # -- non-streaming ------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        model = self._resolve_model(request)
        params = self._build_api_params(request)
        start = time.perf_counter()

        log.info("generation_start", model=model, strategy=request.strategy.value)

        if self._is_anthropic(model):
            result = await self._generate_anthropic(model, request, params)
        else:
            result = await self._generate_openai(model, request, params)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "generation_complete",
            model=model,
            tokens=result.tokens_used,
            elapsed_ms=round(elapsed_ms, 1),
        )
        return result

    async def _generate_anthropic(
        self,
        model: str,
        request: GenerationRequest,
        params: dict[str, float],
    ) -> GenerationResult:
        # Anthropic does not support top_p as a separate param; temperature is primary
        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            api_kwargs["system"] = request.system_prompt
        if "temperature" in params:
            api_kwargs["temperature"] = params["temperature"]
        if "top_k" in params:
            api_kwargs["top_k"] = int(params["top_k"])
        if "top_p" in params:
            api_kwargs["top_p"] = params["top_p"]

        response = await self.anthropic_client.messages.create(**api_kwargs)
        text = response.content[0].text if response.content else ""
        usage = response.usage
        return GenerationResult(
            text=text,
            model=model,
            strategy=request.strategy.value,
            tokens_used=(usage.input_tokens + usage.output_tokens),
            generation_time_ms=0,
            metadata={
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "stop_reason": response.stop_reason,
            },
        )

    async def _generate_openai(
        self,
        model: str,
        request: GenerationRequest,
        params: dict[str, float],
    ) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
        }
        if "temperature" in params:
            api_kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            api_kwargs["top_p"] = params["top_p"]

        response = await self.openai_client.chat.completions.create(**api_kwargs)
        choice = response.choices[0]
        usage = response.usage
        tokens = (usage.prompt_tokens + usage.completion_tokens) if usage else 0
        return GenerationResult(
            text=choice.message.content or "",
            model=model,
            strategy=request.strategy.value,
            tokens_used=tokens,
            generation_time_ms=0,
            metadata={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "finish_reason": choice.finish_reason,
            },
        )

    # -- streaming ----------------------------------------------------------

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[str]:
        model = self._resolve_model(request)
        params = self._build_api_params(request)

        log.info("stream_start", model=model, strategy=request.strategy.value)

        if self._is_anthropic(model):
            async for chunk in self._stream_anthropic(model, request, params):
                yield chunk
        else:
            async for chunk in self._stream_openai(model, request, params):
                yield chunk

    async def _stream_anthropic(
        self,
        model: str,
        request: GenerationRequest,
        params: dict[str, float],
    ) -> AsyncIterator[str]:
        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.system_prompt:
            api_kwargs["system"] = request.system_prompt
        if "temperature" in params:
            api_kwargs["temperature"] = params["temperature"]
        if "top_k" in params:
            api_kwargs["top_k"] = int(params["top_k"])
        if "top_p" in params:
            api_kwargs["top_p"] = params["top_p"]

        async with self.anthropic_client.messages.stream(**api_kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _stream_openai(
        self,
        model: str,
        request: GenerationRequest,
        params: dict[str, float],
    ) -> AsyncIterator[str]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": True,
        }
        if "temperature" in params:
            api_kwargs["temperature"] = params["temperature"]
        if "top_p" in params:
            api_kwargs["top_p"] = params["top_p"]

        response = await self.openai_client.chat.completions.create(**api_kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

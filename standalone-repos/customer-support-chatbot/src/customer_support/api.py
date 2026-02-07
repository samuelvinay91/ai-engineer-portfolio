"""FastAPI application for the customer-support chatbot.

Endpoints:
    POST   /api/v1/chat                   -- synchronous chat completion
    POST   /api/v1/chat/stream            -- streaming chat via SSE
    GET    /api/v1/conversations/{id}      -- retrieve conversation history
    POST   /api/v1/classify               -- classify a message's intent
    GET    /api/v1/prompts                 -- list registered prompt templates
    POST   /api/v1/prompts/test           -- render a prompt template with params
    GET    /health                         -- health / readiness check
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, AsyncGenerator

import anthropic
import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from customer_support.classifier import (
    ClassificationResult,
    Intent,
    IntentClassifier,
    route_to_template,
)
from customer_support.config import Settings, get_settings
from customer_support.conversation import (
    ConversationManager,
    ConversationSession,
    ConversationState,
    SentimentLevel,
)
from customer_support.prompts import (
    ChainOfThoughtTemplate,
    PromptRegistry,
    get_default_registry,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Payload for the synchronous /chat endpoint."""

    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = Field(default=None, description="Resume an existing session")
    customer_name: str | None = None
    template_override: str | None = Field(
        default=None, description="Force a specific prompt template"
    )
    use_chain_of_thought: bool = False


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str | None = None
    confidence: float | None = None
    conversation_state: str
    sentiment_trend: str = "stable"
    turn_count: int = 0


class StreamChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None
    customer_name: str | None = None


class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    context: list[dict[str, str]] | None = None


class ClassifyResponse(BaseModel):
    primary_intent: str
    primary_confidence: float
    secondary_intents: list[dict[str, Any]]
    reasoning: str
    needs_escalation: bool
    latency_ms: float


class PromptTestRequest(BaseModel):
    template_name: str
    knowledge_base: str = ""
    extra_sections: dict[str, str] | None = None


class PromptTestResponse(BaseModel):
    rendered_prompt: str
    template_name: str
    version: str | None = None
    character_count: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    uptime_seconds: float = 0.0


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_start_time: float = 0.0


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the fully configured FastAPI application."""
    global _start_time
    _start_time = time.monotonic()

    settings = settings or get_settings()

    app = FastAPI(
        title="Customer Support Chatbot API",
        version="0.1.0",
        description="Production customer-support chatbot with PEFT/LoRA and prompt engineering.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store settings and singletons in app state
    app.state.settings = settings
    app.state.registry = get_default_registry(settings.prompt_template_dir)
    app.state.conversation_manager: ConversationManager | None = None
    app.state.classifier: IntentClassifier | None = None

    # -- Lifecycle events -----------------------------------------------------

    @app.on_event("startup")
    async def _startup() -> None:
        logger.info(
            "app_starting",
            environment=settings.environment.value,
            model=settings.default_model,
        )

        # Attempt Redis connection (non-fatal if unavailable)
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await redis_client.ping()
            app.state.conversation_manager = ConversationManager(
                redis_client,
                max_history=settings.max_conversation_history,
                ttl_seconds=settings.conversation_ttl_seconds,
                summary_threshold=settings.context_summary_threshold,
            )
            logger.info("redis_connected", url=settings.redis_url)
        except Exception:
            logger.warning("redis_unavailable_using_in_memory_fallback")
            app.state.conversation_manager = None

        # Initialise classifier
        app.state.classifier = IntentClassifier(_make_llm_callable(settings))

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        logger.info("app_shutting_down")

    # -- Register routes ------------------------------------------------------

    app.include_router(_build_router())

    return app


# ---------------------------------------------------------------------------
# LLM callable factory
# ---------------------------------------------------------------------------


def _make_llm_callable(settings: Settings):
    """Return an async callable that invokes the configured LLM provider."""
    api_key = settings.anthropic_api_key.get_secret_value()

    async def _call_anthropic(
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        client = anthropic.AsyncAnthropic(api_key=api_key or "dummy-key")
        resp = await client.messages.create(
            model=settings.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        return resp.content[0].text

    return _call_anthropic


def _make_streaming_llm(settings: Settings):
    """Return an async generator callable for streaming responses."""
    api_key = settings.anthropic_api_key.get_secret_value()

    async def _stream_anthropic(
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        client = anthropic.AsyncAnthropic(api_key=api_key or "dummy-key")
        async with client.messages.stream(
            model=settings.default_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    return _stream_anthropic


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter


def _build_router() -> APIRouter:
    router = APIRouter()

    # -- Health ---------------------------------------------------------------

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            uptime_seconds=round(time.monotonic() - _start_time, 2),
        )

    # -- Chat (synchronous) ---------------------------------------------------

    @router.post("/api/v1/chat", response_model=ChatResponse, tags=["chat"])
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        settings: Settings = request.app.state.settings
        registry: PromptRegistry = request.app.state.registry
        conv_mgr: ConversationManager | None = request.app.state.conversation_manager
        classifier: IntentClassifier | None = request.app.state.classifier

        session_id = body.session_id or uuid.uuid4().hex

        # 1 -- Classify intent
        classification: ClassificationResult | None = None
        template_name = body.template_override or "general_support"
        if classifier is not None:
            try:
                classification = await classifier.classify(body.message)
                route = route_to_template(classification)
                template_name = body.template_override or route.template_name
            except Exception:
                logger.exception("classification_failed")

        # 2 -- Build system prompt
        try:
            system_prompt = registry.build_system_prompt(
                template_name,
                knowledge_base="",
            )
        except KeyError:
            system_prompt = registry.build_system_prompt("general_support")

        # 3 -- Manage conversation context
        context_messages: list[dict[str, str]] = []
        session: ConversationSession | None = None

        if conv_mgr is not None:
            session = await conv_mgr.add_message(
                session_id, "user", body.message,
                sentiment=SentimentLevel.NEUTRAL,
            )
            context_messages = conv_mgr.prepare_context(session)
        else:
            context_messages = [{"role": "user", "content": body.message}]

        # Optionally apply chain-of-thought wrapping
        if body.use_chain_of_thought and context_messages:
            last = context_messages[-1]
            if last["role"] == "user":
                context_messages[-1] = {
                    "role": "user",
                    "content": ChainOfThoughtTemplate.wrap(last["content"]),
                }

        # 4 -- Call LLM
        llm = _make_llm_callable(settings)
        try:
            reply = await llm(
                system_prompt,
                context_messages,
                temperature=settings.model_temperature,
                max_tokens=settings.model_max_tokens,
            )
        except Exception as exc:
            logger.exception("llm_call_failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM provider error: {exc}",
            ) from exc

        # 5 -- Store assistant reply
        if conv_mgr is not None and session is not None:
            session = await conv_mgr.add_message(session_id, "assistant", reply)
            await conv_mgr.auto_transition(session)

        # 6 -- Compute sentiment trend
        trend = "stable"
        if session is not None:
            trend = ConversationManager.compute_sentiment_trend(
                session.sentiment_history
            )

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            intent=classification.primary_intent.value if classification else None,
            confidence=classification.primary_confidence if classification else None,
            conversation_state=(
                session.state.value if session else ConversationState.GREETING.value
            ),
            sentiment_trend=trend,
            turn_count=session.turn_count if session else 1,
        )

    # -- Chat (streaming) -----------------------------------------------------

    @router.post("/api/v1/chat/stream", tags=["chat"])
    async def chat_stream(body: StreamChatRequest, request: Request):
        settings: Settings = request.app.state.settings
        registry: PromptRegistry = request.app.state.registry
        conv_mgr: ConversationManager | None = request.app.state.conversation_manager

        session_id = body.session_id or uuid.uuid4().hex

        system_prompt = registry.build_system_prompt("general_support")

        context_messages: list[dict[str, str]] = []
        if conv_mgr is not None:
            session = await conv_mgr.add_message(session_id, "user", body.message)
            context_messages = conv_mgr.prepare_context(session)
        else:
            context_messages = [{"role": "user", "content": body.message}]

        stream_llm = _make_streaming_llm(settings)

        async def _event_generator() -> AsyncGenerator[dict[str, str], None]:
            full_reply_parts: list[str] = []
            try:
                async for chunk in stream_llm(
                    system_prompt,
                    context_messages,
                    temperature=settings.model_temperature,
                    max_tokens=settings.model_max_tokens,
                ):
                    full_reply_parts.append(chunk)
                    yield {"event": "token", "data": chunk}
            except Exception as exc:
                logger.exception("stream_llm_error")
                yield {"event": "error", "data": str(exc)}
                return

            full_reply = "".join(full_reply_parts)
            if conv_mgr is not None:
                await conv_mgr.add_message(session_id, "assistant", full_reply)

            yield {
                "event": "done",
                "data": f'{{"session_id": "{session_id}"}}',
            }

        return EventSourceResponse(_event_generator())

    # -- Conversation history -------------------------------------------------

    @router.get("/api/v1/conversations/{session_id}", tags=["conversations"])
    async def get_conversation(session_id: str, request: Request):
        conv_mgr: ConversationManager | None = request.app.state.conversation_manager
        if conv_mgr is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation storage unavailable (Redis not connected).",
            )

        session = await conv_mgr.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session {session_id!r} not found.",
            )

        return {
            "session_id": session.session_id,
            "state": session.state.value,
            "turn_count": session.turn_count,
            "summary": session.summary,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp,
                    "sentiment": m.sentiment.value,
                }
                for m in session.messages
            ],
            "sentiment_history": session.sentiment_history,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        }

    # -- Intent classification ------------------------------------------------

    @router.post("/api/v1/classify", response_model=ClassifyResponse, tags=["classification"])
    async def classify_intent(body: ClassifyRequest, request: Request) -> ClassifyResponse:
        classifier: IntentClassifier | None = request.app.state.classifier
        if classifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Classifier not initialized.",
            )

        result = await classifier.classify(
            body.message,
            conversation_context=body.context,
        )
        d = result.to_dict()
        return ClassifyResponse(**d)

    # -- Prompt templates -----------------------------------------------------

    @router.get("/api/v1/prompts", tags=["prompts"])
    async def list_prompts(request: Request):
        registry: PromptRegistry = request.app.state.registry
        templates = []
        for name in registry.list_templates():
            tpl = registry.get(name)
            if tpl is not None:
                templates.append({
                    "name": tpl.name,
                    "description": tpl.description,
                    "current_version": (
                        tpl.current_version.version if tpl.current_version else None
                    ),
                    "chain_of_thought_enabled": tpl.chain_of_thought_enabled,
                    "few_shot_example_count": len(tpl.few_shot_examples),
                })
        return {"templates": templates, "total": len(templates)}

    @router.post(
        "/api/v1/prompts/test",
        response_model=PromptTestResponse,
        tags=["prompts"],
    )
    async def test_prompt(body: PromptTestRequest, request: Request) -> PromptTestResponse:
        registry: PromptRegistry = request.app.state.registry
        try:
            rendered = registry.build_system_prompt(
                body.template_name,
                knowledge_base=body.knowledge_base,
                extra_sections=body.extra_sections,
            )
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Template {body.template_name!r} not found.",
            )

        tpl = registry.get(body.template_name)
        version = tpl.current_version.version if tpl and tpl.current_version else None

        return PromptTestResponse(
            rendered_prompt=rendered,
            template_name=body.template_name,
            version=version,
            character_count=len(rendered),
        )

    # -- Error handlers -------------------------------------------------------

    @router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
    async def _catch_all(path: str):
        raise HTTPException(status_code=404, detail=f"Route /{path} not found.")

    return router

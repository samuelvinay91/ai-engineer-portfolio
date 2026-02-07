"""Ingestion pipeline: load -> chunk -> embed -> store in Qdrant.

Orchestrates the full document ingestion lifecycle with progress tracking
and error handling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Sequence

import structlog
from qdrant_client import AsyncQdrantClient, models

from agent_rag.config import settings
from agent_rag.ingestion.chunker import BaseChunker, Chunk, get_chunker
from agent_rag.ingestion.embedder import BaseEmbedder, get_embedder
from agent_rag.ingestion.loader import DocumentLoader, LoadedDocument

logger = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    """Result of a document ingestion operation."""

    document_id: str
    source: str
    chunks_created: int
    vectors_stored: int
    collection: str
    duration_ms: float
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and self.vectors_stored > 0


@dataclass
class IngestionStats:
    """Aggregate statistics for a batch ingestion."""

    documents_processed: int = 0
    documents_failed: int = 0
    total_chunks: int = 0
    total_vectors: int = 0
    results: list[IngestionResult] = field(default_factory=list)


class IngestionPipeline:
    """End-to-end ingestion: load -> chunk -> embed -> store.

    Coordinates the DocumentLoader, chunker, embedder, and Qdrant client
    to ingest documents into a vector store.
    """

    def __init__(
        self,
        *,
        loader: DocumentLoader | None = None,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._loader = loader or DocumentLoader()
        self._chunker = chunker or get_chunker()
        self._embedder = embedder or get_embedder()
        self._collection = collection_name or settings.qdrant_collection
        self._qdrant = qdrant_client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=(
                settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        collections = await self._qdrant.get_collections()
        existing = {c.name for c in collections.collections}

        if self._collection not in existing:
            await self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._embedder.dimensions,
                    distance=models.Distance.COSINE,
                    on_disk=settings.qdrant_on_disk,
                ),
            )
            # Create payload indices for filtering
            for field_name in ("document_id", "level", "section_title"):
                await self._qdrant.create_payload_index(
                    collection_name=self._collection,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

            logger.info(
                "collection_created",
                name=self._collection,
                dimensions=self._embedder.dimensions,
            )

    # ------------------------------------------------------------------
    # Single document ingestion
    # ------------------------------------------------------------------

    async def ingest_document(
        self,
        source: str | Path,
        *,
        mime_type: str | None = None,
        stream: BinaryIO | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest a single document through the full pipeline."""
        start = datetime.now(timezone.utc)
        errors: list[str] = []

        # 1. Load
        try:
            doc = self._loader.load(source, mime_type=mime_type, stream=stream)
        except Exception as exc:
            return IngestionResult(
                document_id="",
                source=str(source),
                chunks_created=0,
                vectors_stored=0,
                collection=self._collection,
                duration_ms=_elapsed_ms(start),
                errors=[f"Load failed: {exc}"],
            )

        if doc.is_empty:
            return IngestionResult(
                document_id=doc.id,
                source=str(source),
                chunks_created=0,
                vectors_stored=0,
                collection=self._collection,
                duration_ms=_elapsed_ms(start),
                errors=["Document is empty"],
            )

        # 2. Chunk
        try:
            chunks = self._chunker.chunk(
                doc.content,
                document_id=doc.id,
                document_source=str(source),
                extra_metadata=extra_metadata,
            )
        except Exception as exc:
            return IngestionResult(
                document_id=doc.id,
                source=str(source),
                chunks_created=0,
                vectors_stored=0,
                collection=self._collection,
                duration_ms=_elapsed_ms(start),
                errors=[f"Chunking failed: {exc}"],
            )

        if not chunks:
            return IngestionResult(
                document_id=doc.id,
                source=str(source),
                chunks_created=0,
                vectors_stored=0,
                collection=self._collection,
                duration_ms=_elapsed_ms(start),
                errors=["No chunks produced"],
            )

        # 3. Embed
        try:
            texts = [c.text for c in chunks]
            vectors = await self._embedder.embed_texts(texts)
        except Exception as exc:
            return IngestionResult(
                document_id=doc.id,
                source=str(source),
                chunks_created=len(chunks),
                vectors_stored=0,
                collection=self._collection,
                duration_ms=_elapsed_ms(start),
                errors=[f"Embedding failed: {exc}"],
            )

        # 4. Store in Qdrant
        try:
            await self.ensure_collection()
            points = _build_points(chunks, vectors, doc)
            await self._qdrant.upsert(
                collection_name=self._collection,
                points=points,
            )
        except Exception as exc:
            errors.append(f"Storage failed: {exc}")

        duration = _elapsed_ms(start)
        stored = len(vectors) if not errors else 0

        logger.info(
            "document_ingested",
            document_id=doc.id,
            source=str(source),
            chunks=len(chunks),
            vectors=stored,
            duration_ms=round(duration, 1),
        )

        return IngestionResult(
            document_id=doc.id,
            source=str(source),
            chunks_created=len(chunks),
            vectors_stored=stored,
            collection=self._collection,
            duration_ms=duration,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Batch / directory ingestion
    # ------------------------------------------------------------------

    async def ingest_directory(
        self,
        directory: str | Path,
        *,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> IngestionStats:
        """Ingest all supported documents in a directory."""
        docs = self._loader.load_directory(
            directory, recursive=recursive, extensions=extensions
        )
        return await self._ingest_loaded_docs(docs)

    async def ingest_texts(
        self,
        texts: Sequence[str],
        *,
        source_prefix: str = "text",
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionStats:
        """Ingest a list of plain text strings directly."""
        docs: list[LoadedDocument] = []
        for i, text in enumerate(texts):
            from agent_rag.ingestion.loader import (
                DocumentFormat,
                DocumentMetadata,
                _compute_hash,
                _generate_doc_id,
            )

            content_hash = _compute_hash(text)
            doc_id = _generate_doc_id(f"{source_prefix}_{i}", content_hash)
            meta = DocumentMetadata(
                source=f"{source_prefix}_{i}",
                format=DocumentFormat.TXT,
                word_count=len(text.split()),
                content_hash=content_hash,
            )
            docs.append(LoadedDocument(id=doc_id, content=text, metadata=meta))

        return await self._ingest_loaded_docs(docs, extra_metadata=extra_metadata)

    async def _ingest_loaded_docs(
        self,
        docs: list[LoadedDocument],
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionStats:
        stats = IngestionStats()
        for doc in docs:
            result = await self.ingest_document(
                doc.metadata.source, extra_metadata=extra_metadata
            )
            stats.results.append(result)
            if result.success:
                stats.documents_processed += 1
                stats.total_chunks += result.chunks_created
                stats.total_vectors += result.vectors_stored
            else:
                stats.documents_failed += 1
        return stats

    # ------------------------------------------------------------------
    # Queries on the collection
    # ------------------------------------------------------------------

    async def list_documents(self) -> list[dict[str, Any]]:
        """List unique documents in the collection by scrolling payloads."""
        try:
            records, _ = await self._qdrant.scroll(
                collection_name=self._collection,
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []

        seen: dict[str, dict[str, Any]] = {}
        for record in records:
            payload = record.payload or {}
            doc_id = payload.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "source": payload.get("document_source", ""),
                    "title": payload.get("title", ""),
                    "format": payload.get("format", ""),
                }
        return list(seen.values())

    async def get_collections(self) -> list[dict[str, Any]]:
        """Return information about available Qdrant collections."""
        collections = await self._qdrant.get_collections()
        result: list[dict[str, Any]] = []
        for c in collections.collections:
            info = await self._qdrant.get_collection(c.name)
            result.append({
                "name": c.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
            })
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_points(
    chunks: list[Chunk],
    vectors: list[list[float]],
    doc: LoadedDocument,
) -> list[models.PointStruct]:
    """Build Qdrant point structs from chunks and their embeddings."""
    points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        payload: dict[str, Any] = {
            "text": chunk.text,
            "document_id": chunk.metadata.document_id,
            "document_source": chunk.metadata.document_source,
            "chunk_index": chunk.metadata.chunk_index,
            "total_chunks": chunk.metadata.total_chunks,
            "level": chunk.metadata.level,
            "section_title": chunk.metadata.section_title,
            "parent_chunk_id": chunk.metadata.parent_chunk_id,
            "title": doc.metadata.title,
            "author": doc.metadata.author,
            "format": doc.metadata.format.value,
            "token_estimate": chunk.token_estimate,
        }
        points.append(
            models.PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload=payload,
            )
        )
    return points


def _elapsed_ms(start: datetime) -> float:
    return (datetime.now(timezone.utc) - start).total_seconds() * 1000

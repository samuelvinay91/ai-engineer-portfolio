"""Advanced chunking strategies for document processing.

Provides four strategies, each suited to different use-cases:
- FixedSizeChunker:  simple fixed-token windows (fast, predictable)
- SemanticChunker:   splits by semantic boundaries (paragraphs, sections)
- HierarchicalChunker: multi-level chunks (document -> section -> paragraph)
- RecursiveChunker:  recursive text splitting with overlap (LangChain-style)
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_rag.config import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ChunkMetadata:
    """Metadata linking a chunk back to its parent document and section."""

    document_id: str
    document_source: str
    chunk_index: int
    total_chunks: int = 0
    level: str = "paragraph"          # "document" | "section" | "paragraph"
    section_title: str = ""
    page_number: int | None = None
    parent_chunk_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """An individual text chunk ready for embedding."""

    id: str
    text: str
    metadata: ChunkMetadata
    token_estimate: int = 0

    def __post_init__(self) -> None:
        if self.token_estimate == 0:
            # Rough token estimate: ~4 characters per token
            self.token_estimate = max(1, len(self.text) // 4)


def _make_chunk_id() -> str:
    return f"chunk_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Base class for all chunking strategies."""

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        document_source: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Split *text* into a list of Chunk objects."""


# ---------------------------------------------------------------------------
# 1. Fixed-size chunker
# ---------------------------------------------------------------------------

class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size windows by approximate token count.

    Simple and predictable.  Good for homogeneous corpora where semantic
    boundaries are less important than even coverage.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        # Approximate chars per token
        self._char_size = self.chunk_size * 4
        self._char_overlap = self.chunk_overlap * 4

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        document_source: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = start + self._char_size
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break

            meta = ChunkMetadata(
                document_id=document_id,
                document_source=document_source,
                chunk_index=idx,
                level="paragraph",
                extra=extra_metadata or {},
            )
            chunks.append(Chunk(id=_make_chunk_id(), text=chunk_text, metadata=meta))
            start = end - self._char_overlap
            idx += 1

        for c in chunks:
            c.metadata.total_chunks = len(chunks)

        return chunks


# ---------------------------------------------------------------------------
# 2. Semantic chunker
# ---------------------------------------------------------------------------

# Patterns ranked by "strength" of boundary
_SECTION_RE = re.compile(r"\n#{1,4}\s+.+")           # Markdown headings
_DOUBLE_NEWLINE = re.compile(r"\n\s*\n")              # Paragraph break
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")           # Sentence boundary


class SemanticChunker(BaseChunker):
    """Split text along semantic boundaries (sections, paragraphs, sentences).

    Tries to keep chunks under *max_tokens* while respecting natural
    boundaries.  Falls back to sentence splitting when paragraphs are too
    large.
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens or settings.chunk_size
        self._max_chars = self.max_tokens * 4

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        document_source: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        # First split by sections (markdown headings)
        raw_sections = _SECTION_RE.split(text)
        headings = _SECTION_RE.findall(text)

        paragraphs: list[tuple[str, str]] = []  # (section_title, paragraph_text)
        current_heading = ""

        for i, section in enumerate(raw_sections):
            if i > 0 and i - 1 < len(headings):
                current_heading = headings[i - 1].strip().lstrip("#").strip()

            for para in _DOUBLE_NEWLINE.split(section):
                para = para.strip()
                if para:
                    paragraphs.append((current_heading, para))

        # Merge small paragraphs, split large ones
        chunks: list[Chunk] = []
        buffer = ""
        buffer_heading = ""
        idx = 0

        for heading, para in paragraphs:
            if len(buffer) + len(para) + 2 > self._max_chars and buffer:
                chunks.append(self._make_chunk(
                    buffer, idx, document_id, document_source,
                    buffer_heading, extra_metadata,
                ))
                idx += 1
                buffer = ""

            if len(para) > self._max_chars:
                # Flush buffer
                if buffer:
                    chunks.append(self._make_chunk(
                        buffer, idx, document_id, document_source,
                        buffer_heading, extra_metadata,
                    ))
                    idx += 1
                    buffer = ""
                # Split by sentences
                for sentence_chunk in self._split_long(para):
                    chunks.append(self._make_chunk(
                        sentence_chunk, idx, document_id, document_source,
                        heading, extra_metadata,
                    ))
                    idx += 1
            else:
                if buffer:
                    buffer += "\n\n"
                buffer += para
                buffer_heading = heading or buffer_heading

        if buffer.strip():
            chunks.append(self._make_chunk(
                buffer, idx, document_id, document_source,
                buffer_heading, extra_metadata,
            ))

        for c in chunks:
            c.metadata.total_chunks = len(chunks)

        return chunks

    def _split_long(self, text: str) -> list[str]:
        sentences = _SENTENCE_RE.split(text)
        parts: list[str] = []
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) + 1 > self._max_chars and buf:
                parts.append(buf.strip())
                buf = ""
            buf += " " + sent
        if buf.strip():
            parts.append(buf.strip())
        return parts

    @staticmethod
    def _make_chunk(
        text: str,
        idx: int,
        doc_id: str,
        doc_source: str,
        section: str,
        extra: dict[str, Any] | None,
    ) -> Chunk:
        meta = ChunkMetadata(
            document_id=doc_id,
            document_source=doc_source,
            chunk_index=idx,
            level="paragraph",
            section_title=section,
            extra=extra or {},
        )
        return Chunk(id=_make_chunk_id(), text=text, metadata=meta)


# ---------------------------------------------------------------------------
# 3. Hierarchical chunker
# ---------------------------------------------------------------------------

class HierarchicalChunker(BaseChunker):
    """Create multi-level chunks: document -> section -> paragraph.

    Each level references its parent, enabling multi-granularity retrieval.
    A retriever can search at the paragraph level for precision, then
    retrieve the enclosing section for more context.
    """

    def __init__(self, max_tokens: int | None = None) -> None:
        self.max_tokens = max_tokens or settings.chunk_size
        self._max_chars = self.max_tokens * 4

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        document_source: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        extra = extra_metadata or {}

        # Level 0: full document summary chunk
        doc_chunk_id = _make_chunk_id()
        doc_summary = text[: self._max_chars].strip()
        chunks.append(Chunk(
            id=doc_chunk_id,
            text=doc_summary,
            metadata=ChunkMetadata(
                document_id=document_id,
                document_source=document_source,
                chunk_index=0,
                level="document",
                extra=extra,
            ),
        ))

        # Split into sections by headings
        section_splits = re.split(r"(?=\n#{1,3}\s+)", text)
        chunk_idx = 1

        for sec_text in section_splits:
            sec_text = sec_text.strip()
            if not sec_text:
                continue

            # Extract section title
            heading_match = re.match(r"^#{1,3}\s+(.+)", sec_text)
            section_title = heading_match.group(1).strip() if heading_match else ""

            # Level 1: section chunk
            section_chunk_id = _make_chunk_id()
            section_content = sec_text[: self._max_chars * 2].strip()
            chunks.append(Chunk(
                id=section_chunk_id,
                text=section_content,
                metadata=ChunkMetadata(
                    document_id=document_id,
                    document_source=document_source,
                    chunk_index=chunk_idx,
                    level="section",
                    section_title=section_title,
                    parent_chunk_id=doc_chunk_id,
                    extra=extra,
                ),
            ))
            chunk_idx += 1

            # Level 2: paragraph chunks within section
            paragraphs = _DOUBLE_NEWLINE.split(sec_text)
            for para in paragraphs:
                para = para.strip()
                if not para or len(para) < (settings.min_chunk_size * 4):
                    continue

                chunks.append(Chunk(
                    id=_make_chunk_id(),
                    text=para[: self._max_chars],
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        document_source=document_source,
                        chunk_index=chunk_idx,
                        level="paragraph",
                        section_title=section_title,
                        parent_chunk_id=section_chunk_id,
                        extra=extra,
                    ),
                ))
                chunk_idx += 1

        for c in chunks:
            c.metadata.total_chunks = len(chunks)

        return chunks


# ---------------------------------------------------------------------------
# 4. Recursive chunker
# ---------------------------------------------------------------------------

class RecursiveChunker(BaseChunker):
    """Recursive text splitting with overlap (LangChain-style).

    Tries a hierarchy of separators:
      1. Double newline (paragraph)
      2. Single newline
      3. Sentence boundary
      4. Word boundary
    Each level is tried before falling back to the next.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._char_size = self.chunk_size * 4
        self._char_overlap = self.chunk_overlap * 4

    def chunk(
        self,
        text: str,
        *,
        document_id: str = "",
        document_source: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        raw = self._recursive_split(text, self._SEPARATORS)
        chunks: list[Chunk] = []

        for idx, piece in enumerate(raw):
            meta = ChunkMetadata(
                document_id=document_id,
                document_source=document_source,
                chunk_index=idx,
                level="paragraph",
                extra=extra_metadata or {},
            )
            chunks.append(Chunk(id=_make_chunk_id(), text=piece, metadata=meta))

        for c in chunks:
            c.metadata.total_chunks = len(chunks)

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self._char_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []

        parts = text.split(sep)
        merged: list[str] = []
        buffer = ""

        for part in parts:
            candidate = f"{buffer}{sep}{part}" if buffer else part
            if len(candidate) > self._char_size:
                if buffer:
                    merged.append(buffer.strip())
                # If a single part is still too large, recurse with finer separator
                if len(part) > self._char_size and remaining_seps:
                    merged.extend(self._recursive_split(part, remaining_seps))
                    buffer = ""
                else:
                    buffer = part
            else:
                buffer = candidate

        if buffer.strip():
            merged.append(buffer.strip())

        # Apply overlap by prepending tail of previous chunk
        if self._char_overlap > 0 and len(merged) > 1:
            overlapped: list[str] = [merged[0]]
            for i in range(1, len(merged)):
                prev_tail = merged[i - 1][-self._char_overlap:]
                overlapped.append(prev_tail + " " + merged[i])
            return overlapped

        return merged


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_chunker(strategy: str | None = None) -> BaseChunker:
    """Return a chunker instance for the configured strategy."""
    name = (strategy or settings.chunk_strategy.value).lower()
    mapping: dict[str, type[BaseChunker]] = {
        "fixed": FixedSizeChunker,
        "semantic": SemanticChunker,
        "hierarchical": HierarchicalChunker,
        "recursive": RecursiveChunker,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ValueError(f"Unknown chunking strategy: {name}. Choose from {list(mapping)}")
    return cls()

"""Context Engineering module.

Constructs, compresses, and ranks context windows for LLM consumption.
Key concerns:
- Token budget management: ensure context fits within model limits
- Relevance ordering: most important passages first (primacy bias)
- Redundancy removal: avoid duplicate or near-duplicate content
- Compression: summarize verbose passages when space is tight
- Sliding window with importance weighting for long contexts
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import structlog

from agent_rag.config import settings
from agent_rag.retrieval.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token for English text."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# ContextRanker
# ---------------------------------------------------------------------------

@dataclass
class ScoredPassage:
    """A passage with a composite relevance score for ranking."""

    chunk: RetrievedChunk
    composite_score: float
    position: int  # Original position (lower = retrieved earlier)


class ContextRanker:
    """Order context passages by relevance using multiple signals.

    Combines:
    - Retrieval score (from the retriever or reranker)
    - Position bonus (earlier results tend to be more relevant)
    - Diversity penalty (down-rank near-duplicates)
    - Level bonus (section-level chunks may provide better context)
    """

    def __init__(
        self,
        position_weight: float = 0.15,
        diversity_weight: float = 0.1,
        level_bonus: dict[str, float] | None = None,
    ) -> None:
        self._position_weight = position_weight
        self._diversity_weight = diversity_weight
        self._level_bonus = level_bonus or {
            "document": 0.0,
            "section": 0.05,
            "paragraph": 0.0,
        }

    def rank(self, chunks: Sequence[RetrievedChunk]) -> list[ScoredPassage]:
        """Rank chunks by composite relevance score."""
        if not chunks:
            return []

        # Normalize retrieval scores to [0, 1]
        max_score = max(c.score for c in chunks) or 1.0
        min_score = min(c.score for c in chunks)
        score_range = max_score - min_score or 1.0

        scored: list[ScoredPassage] = []
        seen_hashes: set[str] = set()

        for i, chunk in enumerate(chunks):
            # Normalize retrieval score
            norm_score = (chunk.score - min_score) / score_range

            # Position bonus (earlier = better, exponential decay)
            pos_bonus = self._position_weight * (0.9 ** i)

            # Diversity penalty
            text_hash = hashlib.md5(chunk.text[:200].encode()).hexdigest()[:8]
            diversity_penalty = 0.0
            if text_hash in seen_hashes:
                diversity_penalty = self._diversity_weight
            seen_hashes.add(text_hash)

            # Level bonus
            level_bonus = self._level_bonus.get(chunk.level, 0.0)

            composite = norm_score + pos_bonus - diversity_penalty + level_bonus

            scored.append(ScoredPassage(
                chunk=chunk,
                composite_score=composite,
                position=i,
            ))

        scored.sort(key=lambda s: s.composite_score, reverse=True)
        return scored


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """Compress context passages while preserving key information.

    Strategies:
    1. Truncation: Cut long passages at sentence boundaries
    2. Deduplication: Remove near-duplicate passages
    3. Extractive compression: Keep only sentences with query term overlap
    """

    def __init__(
        self,
        max_passage_tokens: int = 300,
        similarity_threshold: float = 0.8,
    ) -> None:
        self._max_passage_tokens = max_passage_tokens
        self._sim_threshold = similarity_threshold

    def compress(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        query: str = "",
    ) -> list[RetrievedChunk]:
        """Compress a list of chunks by removing redundancy and truncating."""
        if not chunks:
            return []

        # 1. Deduplicate
        unique = self._deduplicate(chunks)

        # 2. Truncate long passages
        compressed: list[RetrievedChunk] = []
        for chunk in unique:
            if estimate_tokens(chunk.text) > self._max_passage_tokens:
                truncated_text = self._truncate_at_sentence(
                    chunk.text, self._max_passage_tokens
                )
                # Create a modified copy
                compressed.append(RetrievedChunk(
                    id=chunk.id,
                    text=truncated_text,
                    score=chunk.score,
                    document_id=chunk.document_id,
                    document_source=chunk.document_source,
                    level=chunk.level,
                    section_title=chunk.section_title,
                    parent_chunk_id=chunk.parent_chunk_id,
                    chunk_index=chunk.chunk_index,
                    metadata=chunk.metadata,
                ))
            else:
                compressed.append(chunk)

        # 3. Extractive compression if query is provided
        if query:
            compressed = self._extractive_compress(compressed, query)

        return compressed

    def _deduplicate(
        self,
        chunks: Sequence[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Remove near-duplicate passages using shingling."""
        unique: list[RetrievedChunk] = []
        seen_shingles: list[set[str]] = []

        for chunk in chunks:
            shingles = self._compute_shingles(chunk.text)
            is_dup = False
            for prev_shingles in seen_shingles:
                if not shingles or not prev_shingles:
                    continue
                overlap = len(shingles & prev_shingles) / max(
                    len(shingles | prev_shingles), 1
                )
                if overlap >= self._sim_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(chunk)
                seen_shingles.append(shingles)

        return unique

    @staticmethod
    def _compute_shingles(text: str, k: int = 3) -> set[str]:
        """Compute character k-shingles for near-duplicate detection."""
        words = text.lower().split()
        if len(words) < k:
            return {" ".join(words)}
        return {" ".join(words[i: i + k]) for i in range(len(words) - k + 1)}

    @staticmethod
    def _truncate_at_sentence(text: str, max_tokens: int) -> str:
        """Truncate text at a sentence boundary within the token budget."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Find the last sentence boundary before the limit
        truncated = text[:max_chars]
        for end_char in [".", "!", "?", "\n"]:
            last_pos = truncated.rfind(end_char)
            if last_pos > max_chars // 2:
                return truncated[: last_pos + 1]

        return truncated.rstrip() + "..."

    @staticmethod
    def _extractive_compress(
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[RetrievedChunk]:
        """Keep sentences that have high overlap with query terms."""
        query_terms = set(query.lower().split())
        if not query_terms:
            return chunks

        result: list[RetrievedChunk] = []
        for chunk in chunks:
            sentences = chunk.text.replace("\n", " ").split(". ")
            relevant_sentences: list[str] = []

            for sent in sentences:
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                # Keep sentence if it has any query term overlap
                if overlap > 0 or len(relevant_sentences) < 2:
                    relevant_sentences.append(sent)

            compressed_text = ". ".join(relevant_sentences)
            if compressed_text and not compressed_text.endswith("."):
                compressed_text += "."

            result.append(RetrievedChunk(
                id=chunk.id,
                text=compressed_text,
                score=chunk.score,
                document_id=chunk.document_id,
                document_source=chunk.document_source,
                level=chunk.level,
                section_title=chunk.section_title,
                parent_chunk_id=chunk.parent_chunk_id,
                chunk_index=chunk.chunk_index,
                metadata=chunk.metadata,
            ))

        return result


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------

@dataclass
class BuiltContext:
    """A constructed context window with metadata."""

    text: str
    num_passages: int
    total_tokens: int
    truncated: bool = False


class ContextBuilder:
    """Construct optimal context windows for LLM consumption.

    Orchestrates ranking, compression, and token budget management to
    produce the best possible context within the model's limits.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        ranker: ContextRanker | None = None,
        compressor: ContextCompressor | None = None,
        enable_compression: bool | None = None,
    ) -> None:
        self._max_tokens = max_tokens or settings.context_max_tokens
        self._ranker = ranker or ContextRanker()
        self._compressor = compressor or ContextCompressor()
        self._compression = (
            enable_compression
            if enable_compression is not None
            else settings.context_compression_enabled
        )

    def build(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        max_tokens: int | None = None,
        query: str = "",
    ) -> str:
        """Build a context string from retrieved chunks.

        Args:
            chunks: Retrieved chunks to include.
            max_tokens: Override token budget.
            query: Optional query for compression relevance.

        Returns:
            Formatted context string within the token budget.
        """
        if not chunks:
            return "No relevant context found."

        budget = max_tokens or self._max_tokens

        # 1. Rank
        scored = self._ranker.rank(list(chunks))
        ranked_chunks = [s.chunk for s in scored]

        # 2. Compress if enabled
        if self._compression:
            ranked_chunks = self._compressor.compress(ranked_chunks, query=query)

        # 3. Build within token budget
        passages: list[str] = []
        tokens_used = 0
        passage_num = 1

        for chunk in ranked_chunks:
            passage_tokens = estimate_tokens(chunk.text)
            header_tokens = 10  # Approximate header size

            if tokens_used + passage_tokens + header_tokens > budget:
                # Try to fit a truncated version
                remaining = budget - tokens_used - header_tokens
                if remaining > 50:  # Only if meaningful space remains
                    truncated = chunk.text[: remaining * 4]
                    last_period = truncated.rfind(".")
                    if last_period > len(truncated) // 2:
                        truncated = truncated[: last_period + 1]
                    passages.append(self._format_passage(passage_num, chunk, truncated))
                break

            passages.append(self._format_passage(passage_num, chunk, chunk.text))
            tokens_used += passage_tokens + header_tokens
            passage_num += 1

        if not passages:
            return "No relevant context found."

        return "\n\n".join(passages)

    def build_detailed(
        self,
        chunks: Sequence[RetrievedChunk],
        *,
        max_tokens: int | None = None,
        query: str = "",
    ) -> BuiltContext:
        """Build context with detailed metadata about the result."""
        text = self.build(chunks, max_tokens=max_tokens, query=query)
        tokens = estimate_tokens(text)
        budget = max_tokens or self._max_tokens

        return BuiltContext(
            text=text,
            num_passages=text.count("[Passage"),
            total_tokens=tokens,
            truncated=tokens >= budget * 0.95,
        )

    @staticmethod
    def _format_passage(num: int, chunk: RetrievedChunk, text: str) -> str:
        """Format a single passage with header metadata."""
        header_parts = [f"[Passage {num}]"]
        if chunk.section_title:
            header_parts.append(f"Section: {chunk.section_title}")
        if chunk.document_source:
            source_name = chunk.document_source.split("/")[-1]
            header_parts.append(f"Source: {source_name}")
        header = " | ".join(header_parts)
        return f"{header}\n{text}"

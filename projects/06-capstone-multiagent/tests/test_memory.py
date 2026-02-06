"""Tests for the three-tier memory system.

Covers short-term, working, and long-term memory independently.  Long-term
memory tests use an in-memory SQLite database to avoid requiring PostgreSQL.
"""

from __future__ import annotations

import time

import pytest

from capstone.memory import (
    ConversationMessage,
    MemoryEntry,
    MemoryManager,
    ShortTermMemory,
    WorkingMemory,
)


# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------


class TestShortTermMemory:
    """Tests for the sliding-window conversation memory."""

    def test_add_and_retrieve_messages(self, short_term_memory: ShortTermMemory):
        """Messages are stored and retrievable."""
        short_term_memory.add_message("user", "Hello")
        short_term_memory.add_message("assistant", "Hi there!")

        messages = short_term_memory.messages
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].content == "Hi there!"

    def test_sliding_window_eviction(self):
        """Oldest messages are evicted when the window overflows."""
        mem = ShortTermMemory(session_id="test", max_messages=3)
        for i in range(5):
            mem.add_message("user", f"Message {i}")

        messages = mem.messages
        assert len(messages) == 3
        assert messages[0].content == "Message 2"
        assert messages[-1].content == "Message 4"

    def test_get_context_window(self, short_term_memory: ShortTermMemory):
        """Context window returns formatted messages for LLM injection."""
        short_term_memory.add_message("user", "Q1")
        short_term_memory.add_message("assistant", "A1")
        short_term_memory.add_message("user", "Q2")

        context = short_term_memory.get_context_window(last_n=2)
        assert len(context) == 2
        assert context[0]["role"] == "assistant"
        assert context[1]["content"] == "Q2"

    def test_summarize(self, short_term_memory: ShortTermMemory):
        """Summary produces a compact text representation."""
        short_term_memory.add_message("user", "What is AI?")
        short_term_memory.add_message("assistant", "AI stands for Artificial Intelligence.")

        summary = short_term_memory.summarize()
        assert "user" in summary
        assert "AI" in summary

    def test_clear(self, short_term_memory: ShortTermMemory):
        """Clear removes all messages."""
        short_term_memory.add_message("user", "Test")
        short_term_memory.clear()
        assert len(short_term_memory.messages) == 0

    def test_serialization_roundtrip(self, short_term_memory: ShortTermMemory):
        """Messages survive JSON serialization and deserialization."""
        short_term_memory.add_message("user", "Hello")
        short_term_memory.add_message("assistant", "World")

        data = short_term_memory.to_json()
        restored = ShortTermMemory.from_json("test-session", data)

        assert len(restored.messages) == 2
        assert restored.messages[0].role == "user"
        assert restored.messages[1].content == "World"

    def test_session_id_property(self, short_term_memory: ShortTermMemory):
        """Session ID is accessible via property."""
        assert short_term_memory.session_id == "test-session"

    def test_empty_summarize(self):
        """Summarizing empty memory returns empty string."""
        mem = ShortTermMemory(session_id="empty")
        assert mem.summarize() == ""


# ---------------------------------------------------------------------------
# Working memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    """Tests for the task scratch-pad memory."""

    def test_store_and_retrieve(self, working_memory: WorkingMemory):
        """Entries can be stored and retrieved by task_id + key."""
        working_memory.store("task-1", "step1", "Intermediate result A")
        entry = working_memory.retrieve("task-1", "step1")
        assert entry is not None
        assert entry.content == "Intermediate result A"

    def test_retrieve_nonexistent(self, working_memory: WorkingMemory):
        """Retrieving a non-existent entry returns None."""
        assert working_memory.retrieve("task-x", "missing") is None

    def test_get_all_for_task(self, working_memory: WorkingMemory):
        """All entries for a task can be retrieved at once."""
        working_memory.store("task-2", "a", "Result A")
        working_memory.store("task-2", "b", "Result B")
        working_memory.store("task-3", "x", "Different task")

        entries = working_memory.get_all_for_task("task-2")
        assert len(entries) == 2

    def test_clear_task(self, working_memory: WorkingMemory):
        """All entries for a task can be cleared."""
        working_memory.store("task-4", "a", "X")
        working_memory.store("task-4", "b", "Y")
        count = working_memory.clear_task("task-4")
        assert count == 2
        assert working_memory.get_all_for_task("task-4") == []

    def test_ttl_expiration(self):
        """Expired entries are pruned on retrieval."""
        mem = WorkingMemory(ttl_seconds=0)  # immediate expiry
        entry = mem.store("task-5", "k", "Will expire")
        # Force the created_at to the past
        entry.created_at = time.time() - 10

        assert mem.retrieve("task-5", "k") is None

    def test_cleanup_expired(self):
        """Bulk cleanup removes all expired entries."""
        mem = WorkingMemory(ttl_seconds=0)
        entry1 = mem.store("task-6", "a", "Old")
        entry2 = mem.store("task-6", "b", "Old too")
        entry1.created_at = time.time() - 10
        entry2.created_at = time.time() - 10

        count = mem.cleanup_expired()
        assert count == 2

    def test_agent_name_metadata(self, working_memory: WorkingMemory):
        """Agent name is stored with the entry."""
        working_memory.store("task-7", "k", "V", agent_name="researcher")
        entry = working_memory.retrieve("task-7", "k")
        assert entry is not None
        assert entry.agent_name == "researcher"


# ---------------------------------------------------------------------------
# Memory manager
# ---------------------------------------------------------------------------


class TestMemoryManager:
    """Tests for the unified memory manager."""

    def test_get_short_term_creates_on_first_access(self, memory_manager: MemoryManager):
        """Short-term memory is lazily created per session."""
        mem = memory_manager.get_short_term("new-session")
        assert isinstance(mem, ShortTermMemory)
        assert mem.session_id == "new-session"

    def test_get_short_term_reuses_existing(self, memory_manager: MemoryManager):
        """Same session returns the same short-term memory object."""
        mem1 = memory_manager.get_short_term("s1")
        mem2 = memory_manager.get_short_term("s1")
        assert mem1 is mem2

    def test_add_and_get_context(self, memory_manager: MemoryManager):
        """Messages added via the manager are retrievable as context."""
        memory_manager.add_message("s2", "user", "Hello")
        memory_manager.add_message("s2", "assistant", "World")
        context = memory_manager.get_context("s2")
        assert len(context) == 2

    def test_working_memory_access(self, memory_manager: MemoryManager):
        """Working memory is accessible via the manager."""
        working = memory_manager.working
        assert isinstance(working, WorkingMemory)
        working.store("t1", "k", "v")
        assert working.retrieve("t1", "k") is not None

    @pytest.mark.asyncio
    async def test_long_term_none_without_db(self, memory_manager: MemoryManager):
        """Long-term memory is None before initialization."""
        assert memory_manager.long_term is None

    @pytest.mark.asyncio
    async def test_search_long_term_returns_empty_without_db(
        self, memory_manager: MemoryManager
    ):
        """Searching long-term memory without a DB returns an empty list."""
        results = await memory_manager.search_long_term("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_store_long_term_returns_none_without_db(
        self, memory_manager: MemoryManager
    ):
        """Storing to long-term memory without a DB returns None."""
        result = await memory_manager.store_long_term("s1", "content")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_session_summary(self, memory_manager: MemoryManager):
        """Session summary includes short-term and long-term info."""
        memory_manager.add_message("summary-sess", "user", "Test message")
        summary = await memory_manager.get_session_summary("summary-sess")

        assert summary["session_id"] == "summary-sess"
        assert summary["short_term"]["message_count"] == 1
        assert "long_term" in summary


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TestMemoryModels:
    """Tests for memory data models."""

    def test_memory_entry_defaults(self):
        """MemoryEntry has sensible defaults."""
        entry = MemoryEntry(session_id="s1", content="test")
        assert entry.entry_id  # auto-generated
        assert entry.relevance_score == 1.0
        assert entry.content_type == "text"

    def test_conversation_message_defaults(self):
        """ConversationMessage has a timestamp by default."""
        msg = ConversationMessage(role="user", content="Hello")
        assert msg.timestamp > 0
        assert msg.metadata == {}

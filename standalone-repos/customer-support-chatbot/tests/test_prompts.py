"""Tests for the prompt engineering module.

Covers:
- SystemPromptBuilder composition and section ordering
- PromptTemplate versioning
- FewShotManager CRUD and rendering
- PromptRegistry loading from YAML
- ChainOfThoughtTemplate wrapping
- End-to-end build_system_prompt via the registry
"""

from __future__ import annotations

from pathlib import Path

import pytest

from customer_support.prompts import (
    ChainOfThoughtTemplate,
    FewShotExample,
    FewShotManager,
    PromptRegistry,
    PromptTemplate,
    PromptVersion,
    SupportCategory,
    SystemPromptBuilder,
    get_default_registry,
)


# ---------------------------------------------------------------------------
# SystemPromptBuilder
# ---------------------------------------------------------------------------


class TestSystemPromptBuilder:
    """Tests for the fluent SystemPromptBuilder."""

    def test_build_empty(self) -> None:
        prompt = SystemPromptBuilder().build()
        assert prompt == ""

    def test_build_with_role_only(self) -> None:
        prompt = SystemPromptBuilder().with_role("Test Agent").build()
        assert "<role>" in prompt
        assert "Test Agent" in prompt
        assert "</role>" in prompt

    def test_build_section_ordering(self) -> None:
        """Role should appear before knowledge_base, which appears before tone."""
        prompt = (
            SystemPromptBuilder()
            .with_tone("friendly")
            .with_knowledge_base("KB content")
            .with_role("Agent role")
            .build()
        )
        role_pos = prompt.index("<role>")
        kb_pos = prompt.index("<knowledge_base>")
        tone_pos = prompt.index("<tone_guidelines>")
        assert role_pos < kb_pos < tone_pos

    def test_build_with_all_sections(self) -> None:
        prompt = (
            SystemPromptBuilder()
            .with_role("Support Agent")
            .with_knowledge_base("Product FAQ here")
            .with_tone("professional", "empathetic")
            .with_escalation_rules("Escalate if angry.")
            .with_response_format("Use bullet points.")
            .with_guardrails("No secrets.")
            .build()
        )
        assert "<role>" in prompt
        assert "<knowledge_base>" in prompt
        assert "<tone_guidelines>" in prompt
        assert "<escalation_rules>" in prompt
        assert "<response_format>" in prompt
        assert "<guardrails>" in prompt
        assert "professional, empathetic" in prompt

    def test_chain_of_thought_visible(self) -> None:
        prompt = (
            SystemPromptBuilder()
            .with_role("Agent")
            .with_chain_of_thought(visible=True)
            .build()
        )
        assert "<chain_of_thought>" in prompt
        assert "<thinking>" in prompt

    def test_chain_of_thought_hidden(self) -> None:
        prompt = (
            SystemPromptBuilder()
            .with_role("Agent")
            .with_chain_of_thought(visible=False)
            .build()
        )
        assert "<chain_of_thought>" in prompt
        assert "Do NOT include this reasoning" in prompt

    def test_few_shot_examples_included(self) -> None:
        mgr = FewShotManager()
        mgr.add(FewShotExample(
            user_message="Hello",
            assistant_response="Hi there!",
            category="test",
        ))
        prompt = (
            SystemPromptBuilder()
            .with_role("Agent")
            .with_few_shot_examples(mgr, "test")
            .build()
        )
        assert "<examples>" in prompt
        assert "Hello" in prompt
        assert "Hi there!" in prompt

    def test_custom_section(self) -> None:
        prompt = (
            SystemPromptBuilder()
            .with_custom_section("product_catalog", "Widget A, Widget B")
            .build()
        )
        assert "<product_catalog>" in prompt
        assert "Widget A" in prompt


# ---------------------------------------------------------------------------
# FewShotManager
# ---------------------------------------------------------------------------


class TestFewShotManager:
    def test_add_and_get(self) -> None:
        mgr = FewShotManager()
        ex = FewShotExample(user_message="Q", assistant_response="A", category="cat1")
        mgr.add(ex)
        assert mgr.get("cat1") == [ex]
        assert mgr.get("nonexistent") == []

    def test_limit(self) -> None:
        mgr = FewShotManager()
        for i in range(10):
            mgr.add(FewShotExample(
                user_message=f"Q{i}",
                assistant_response=f"A{i}",
                category="cat",
            ))
        assert len(mgr.get("cat", limit=3)) == 3

    def test_render_empty(self) -> None:
        mgr = FewShotManager()
        assert mgr.render("empty") == ""

    def test_render_nonempty(self) -> None:
        mgr = FewShotManager()
        mgr.add(FewShotExample(user_message="Hi", assistant_response="Hello", category="greet"))
        rendered = mgr.render("greet")
        assert "<examples>" in rendered
        assert "<user>Hi</user>" in rendered
        assert "<assistant>Hello</assistant>" in rendered

    def test_categories(self) -> None:
        mgr = FewShotManager()
        mgr.add(FewShotExample(user_message="Q", assistant_response="A", category="b"))
        mgr.add(FewShotExample(user_message="Q", assistant_response="A", category="a"))
        assert mgr.categories == ["a", "b"]

    def test_len(self) -> None:
        mgr = FewShotManager()
        assert len(mgr) == 0
        mgr.add(FewShotExample(user_message="Q", assistant_response="A", category="x"))
        mgr.add(FewShotExample(user_message="Q", assistant_response="A", category="y"))
        assert len(mgr) == 2


# ---------------------------------------------------------------------------
# PromptTemplate & PromptVersion
# ---------------------------------------------------------------------------


class TestPromptTemplate:
    def test_snapshot_creates_version(self) -> None:
        tpl = PromptTemplate(name="test", system_prompt="You are helpful.")
        v = tpl.snapshot("1.0.0", description="Initial")
        assert v.version == "1.0.0"
        assert v.sha256  # non-empty hash
        assert tpl.current_version is v
        assert len(tpl.versions) == 1

    def test_multiple_snapshots(self) -> None:
        tpl = PromptTemplate(name="test", system_prompt="v1")
        tpl.snapshot("1.0.0")
        tpl.system_prompt = "v2"
        tpl.snapshot("2.0.0")
        assert len(tpl.versions) == 2
        assert tpl.current_version.version == "2.0.0"

    def test_version_hash_differs(self) -> None:
        v1 = PromptVersion(version="1", template="Hello", created_at="now")
        v2 = PromptVersion(version="2", template="World", created_at="now")
        assert v1.sha256 != v2.sha256


# ---------------------------------------------------------------------------
# PromptRegistry (YAML loading)
# ---------------------------------------------------------------------------


class TestPromptRegistry:
    def test_load_from_directory(self, template_dir: Path) -> None:
        registry = PromptRegistry()
        loaded = registry.load_from_directory(template_dir)
        assert loaded >= 3  # general, technical, billing

    def test_list_templates(self, prompt_registry: PromptRegistry) -> None:
        names = prompt_registry.list_templates()
        assert "general_support" in names
        assert "technical_support" in names
        assert "billing_support" in names

    def test_get_existing(self, prompt_registry: PromptRegistry) -> None:
        tpl = prompt_registry.get("general_support")
        assert tpl is not None
        assert tpl.name == "general_support"
        assert tpl.description  # non-empty

    def test_get_missing(self, prompt_registry: PromptRegistry) -> None:
        assert prompt_registry.get("nonexistent_template") is None

    def test_contains(self, prompt_registry: PromptRegistry) -> None:
        assert "general_support" in prompt_registry
        assert "nonexistent" not in prompt_registry

    def test_len(self, prompt_registry: PromptRegistry) -> None:
        assert len(prompt_registry) >= 3

    def test_few_shot_examples_loaded(self, prompt_registry: PromptRegistry) -> None:
        tpl = prompt_registry.get("billing_support")
        assert tpl is not None
        assert len(tpl.few_shot_examples) >= 2

    def test_version_auto_created(self, prompt_registry: PromptRegistry) -> None:
        tpl = prompt_registry.get("technical_support")
        assert tpl is not None
        assert tpl.current_version is not None
        assert tpl.current_version.version == "1.0.0"

    def test_chain_of_thought_flag(self, prompt_registry: PromptRegistry) -> None:
        tech = prompt_registry.get("technical_support")
        general = prompt_registry.get("general_support")
        assert tech is not None and tech.chain_of_thought_enabled is True
        assert general is not None and general.chain_of_thought_enabled is False

    def test_build_system_prompt_general(self, prompt_registry: PromptRegistry) -> None:
        prompt = prompt_registry.build_system_prompt("general_support")
        assert "<role>" in prompt
        assert "<tone_guidelines>" in prompt
        assert "<escalation_rules>" in prompt
        assert len(prompt) > 100

    def test_build_system_prompt_technical_includes_cot(
        self, prompt_registry: PromptRegistry
    ) -> None:
        prompt = prompt_registry.build_system_prompt("technical_support")
        assert "<chain_of_thought>" in prompt

    def test_build_system_prompt_with_knowledge_base(
        self, prompt_registry: PromptRegistry
    ) -> None:
        kb = "Acme Widget Pro costs $99/month."
        prompt = prompt_registry.build_system_prompt(
            "billing_support",
            knowledge_base=kb,
        )
        assert "<knowledge_base>" in prompt
        assert "Acme Widget Pro" in prompt

    def test_build_system_prompt_with_extra_sections(
        self, prompt_registry: PromptRegistry
    ) -> None:
        prompt = prompt_registry.build_system_prompt(
            "general_support",
            extra_sections={"customer_tier": "Enterprise, priority support"},
        )
        assert "<customer_tier>" in prompt
        assert "Enterprise" in prompt

    def test_build_system_prompt_unknown_template_raises(
        self, prompt_registry: PromptRegistry
    ) -> None:
        with pytest.raises(KeyError, match="Unknown template"):
            prompt_registry.build_system_prompt("nonexistent")

    def test_load_missing_directory_returns_zero(self, tmp_path: Path) -> None:
        registry = PromptRegistry()
        loaded = registry.load_from_directory(tmp_path / "does_not_exist")
        assert loaded == 0


# ---------------------------------------------------------------------------
# ChainOfThoughtTemplate
# ---------------------------------------------------------------------------


class TestChainOfThoughtTemplate:
    def test_wrap_includes_message(self) -> None:
        wrapped = ChainOfThoughtTemplate.wrap("My account is locked.")
        assert "My account is locked." in wrapped
        assert "<customer_message>" in wrapped
        assert "<reasoning_framework>" in wrapped

    def test_wrap_includes_all_steps(self) -> None:
        wrapped = ChainOfThoughtTemplate.wrap("Test")
        assert "Step 1" in wrapped
        assert "Step 5" in wrapped


# ---------------------------------------------------------------------------
# SupportCategory enum
# ---------------------------------------------------------------------------


class TestSupportCategory:
    def test_values(self) -> None:
        assert SupportCategory.GENERAL == "general_support"
        assert SupportCategory.TECHNICAL == "technical_support"
        assert SupportCategory.BILLING == "billing_support"


# ---------------------------------------------------------------------------
# get_default_registry singleton
# ---------------------------------------------------------------------------


class TestDefaultRegistry:
    def test_returns_registry(self, template_dir: Path) -> None:
        # Reset the global to ensure a fresh load
        import customer_support.prompts as mod
        mod._default_registry = None

        registry = get_default_registry(template_dir)
        assert len(registry) >= 3
        # Built-in few-shot examples should be seeded
        assert len(registry.few_shot_manager) >= 3

        # Clean up
        mod._default_registry = None

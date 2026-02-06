"""Advanced prompt engineering module.

Provides a composable, version-controlled system for building LLM prompts
used across the customer-support chatbot.  Key capabilities:

* **SystemPromptBuilder** -- fluent builder that assembles system prompts from
  discrete sections (role, knowledge base, tone, escalation rules, format).
* **PromptTemplate / PromptRegistry** -- YAML-backed templates with semantic
  versioning so prompt changes can be tracked and rolled back.
* **FewShotManager** -- manages exemplar (input, output) pairs for few-shot
  in-context learning.
* **ChainOfThoughtTemplate** -- wrapper that injects structured reasoning
  instructions for complex issue resolution.

Three built-in template families ship with the package:
``general_support``, ``technical_support``, and ``billing_support``.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import textwrap
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SupportCategory(StrEnum):
    """Top-level support categories used by prompt selection."""

    GENERAL = "general_support"
    TECHNICAL = "technical_support"
    BILLING = "billing_support"


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """A single few-shot exemplar."""

    user_message: str
    assistant_response: str
    category: str = "general"
    tags: tuple[str, ...] = ()

    def render(self) -> str:
        return (
            f"<example>\n"
            f"  <user>{self.user_message}</user>\n"
            f"  <assistant>{self.assistant_response}</assistant>\n"
            f"</example>"
        )


@dataclass(frozen=True, slots=True)
class PromptVersion:
    """Immutable snapshot of a prompt template at a specific version."""

    version: str
    template: str
    created_at: str
    description: str = ""
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            digest = hashlib.sha256(self.template.encode()).hexdigest()[:12]
            object.__setattr__(self, "sha256", digest)


@dataclass(slots=True)
class PromptTemplate:
    """A named, versioned prompt template loaded from YAML.

    Attributes:
        name: Unique template identifier (e.g. ``general_support``).
        description: Human-readable description.
        system_prompt: The Jinja2-compatible system prompt body.
        few_shot_examples: Optional exemplars to prepend.
        metadata: Arbitrary key-value metadata from the YAML file.
        versions: Chronological list of prompt versions.
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    few_shot_examples: list[FewShotExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    versions: list[PromptVersion] = field(default_factory=list)
    chain_of_thought_enabled: bool = False

    @property
    def current_version(self) -> PromptVersion | None:
        return self.versions[-1] if self.versions else None

    def snapshot(self, version_tag: str, *, description: str = "") -> PromptVersion:
        """Create a new immutable version snapshot of the current prompt."""
        pv = PromptVersion(
            version=version_tag,
            template=self.system_prompt,
            created_at=dt.datetime.now(dt.UTC).isoformat(),
            description=description,
        )
        self.versions.append(pv)
        logger.info(
            "prompt_version_created",
            template=self.name,
            version=version_tag,
            sha256=pv.sha256,
        )
        return pv


# ---------------------------------------------------------------------------
# FewShotManager
# ---------------------------------------------------------------------------

class FewShotManager:
    """Manages a corpus of few-shot examples, indexed by category."""

    def __init__(self) -> None:
        self._examples: dict[str, list[FewShotExample]] = {}

    def add(self, example: FewShotExample) -> None:
        self._examples.setdefault(example.category, []).append(example)

    def add_many(self, examples: list[FewShotExample]) -> None:
        for ex in examples:
            self.add(ex)

    def get(self, category: str, *, limit: int = 3) -> list[FewShotExample]:
        """Return up to *limit* examples for the given category."""
        return self._examples.get(category, [])[:limit]

    def render(self, category: str, *, limit: int = 3) -> str:
        """Render examples as XML-style blocks suitable for system prompts."""
        examples = self.get(category, limit=limit)
        if not examples:
            return ""
        rendered = "\n".join(ex.render() for ex in examples)
        return f"<examples>\n{rendered}\n</examples>"

    @property
    def categories(self) -> list[str]:
        return sorted(self._examples.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._examples.values())


# ---------------------------------------------------------------------------
# SystemPromptBuilder
# ---------------------------------------------------------------------------

class SystemPromptBuilder:
    """Fluent builder for composing multi-section system prompts.

    Usage::

        prompt = (
            SystemPromptBuilder()
            .with_role("Acme Corp Customer Support Agent")
            .with_knowledge_base(kb_text)
            .with_tone("professional", "empathetic", "concise")
            .with_escalation_rules(rules)
            .with_response_format(fmt)
            .with_few_shot_examples(manager, "billing")
            .build()
        )
    """

    def __init__(self) -> None:
        self._sections: dict[str, str] = {}
        self._few_shot_block: str = ""
        self._cot_instructions: str = ""

    # -- Section setters (fluent) ---------------------------------------------

    def with_role(self, role_description: str) -> SystemPromptBuilder:
        """Define the agent's identity and primary responsibilities."""
        self._sections["role"] = textwrap.dedent(f"""\
            <role>
            {role_description.strip()}
            </role>""")
        return self

    def with_knowledge_base(self, knowledge: str) -> SystemPromptBuilder:
        """Inject company-specific knowledge (FAQ, policies, product info)."""
        self._sections["knowledge_base"] = textwrap.dedent(f"""\
            <knowledge_base>
            {knowledge.strip()}
            </knowledge_base>""")
        return self

    def with_tone(self, *adjectives: str) -> SystemPromptBuilder:
        """Set the desired tone / style for responses."""
        tone_list = ", ".join(adjectives)
        self._sections["tone"] = textwrap.dedent(f"""\
            <tone_guidelines>
            Always respond in a tone that is: {tone_list}.
            - Use the customer's name when available.
            - Avoid jargon unless the customer uses it first.
            - Mirror the customer's communication style while maintaining professionalism.
            - Show empathy for frustrations before jumping to solutions.
            </tone_guidelines>""")
        return self

    def with_escalation_rules(self, rules: str) -> SystemPromptBuilder:
        """Provide escalation criteria and instructions."""
        self._sections["escalation"] = textwrap.dedent(f"""\
            <escalation_rules>
            {rules.strip()}
            </escalation_rules>""")
        return self

    def with_response_format(self, format_instructions: str) -> SystemPromptBuilder:
        """Constrain the structure / format of responses."""
        self._sections["format"] = textwrap.dedent(f"""\
            <response_format>
            {format_instructions.strip()}
            </response_format>""")
        return self

    def with_guardrails(self, guardrails: str) -> SystemPromptBuilder:
        """Add safety guardrails and content policies."""
        self._sections["guardrails"] = textwrap.dedent(f"""\
            <guardrails>
            {guardrails.strip()}
            </guardrails>""")
        return self

    def with_few_shot_examples(
        self,
        manager: FewShotManager,
        category: str,
        *,
        limit: int = 3,
    ) -> SystemPromptBuilder:
        """Attach few-shot examples from the manager."""
        self._few_shot_block = manager.render(category, limit=limit)
        return self

    def with_chain_of_thought(self, *, visible: bool = False) -> SystemPromptBuilder:
        """Enable chain-of-thought reasoning instructions.

        Args:
            visible: If True the reasoning is shown to the user inside
                ``<thinking>`` tags.  If False the model is instructed to
                produce reasoning internally and only surface the final answer.
        """
        if visible:
            self._cot_instructions = textwrap.dedent("""\
                <chain_of_thought>
                For complex issues, think step-by-step before responding:
                1. Identify the core problem or question.
                2. List relevant policies, known issues, or documentation.
                3. Consider possible solutions in order of simplicity.
                4. Determine whether escalation is needed.
                5. Formulate a clear, actionable response.

                Wrap your reasoning in <thinking>...</thinking> tags so the
                customer can follow your logic.
                </chain_of_thought>""")
        else:
            self._cot_instructions = textwrap.dedent("""\
                <chain_of_thought>
                Before responding to the customer, silently reason through the
                following steps.  Do NOT include this reasoning in your visible
                response -- only output the final answer.

                Internal reasoning steps:
                1. Classify the issue (billing / technical / account / general).
                2. Check the knowledge base for relevant policies or solutions.
                3. Determine whether additional information is needed from the user.
                4. Assess whether escalation criteria are met.
                5. Compose a helpful, empathetic response.
                </chain_of_thought>""")
        return self

    def with_custom_section(self, name: str, content: str) -> SystemPromptBuilder:
        """Add an arbitrary named section."""
        self._sections[name] = textwrap.dedent(f"""\
            <{name}>
            {content.strip()}
            </{name}>""")
        return self

    # -- Build ----------------------------------------------------------------

    def build(self) -> str:
        """Assemble all sections into the final system prompt string."""
        parts: list[str] = []

        # Deterministic ordering: role -> knowledge_base -> tone -> escalation
        # -> format -> guardrails -> everything else alphabetically
        priority = ["role", "knowledge_base", "tone", "escalation", "format", "guardrails"]
        for key in priority:
            if key in self._sections:
                parts.append(self._sections[key])

        for key in sorted(self._sections):
            if key not in priority:
                parts.append(self._sections[key])

        if self._cot_instructions:
            parts.append(self._cot_instructions)

        if self._few_shot_block:
            parts.append(self._few_shot_block)

        prompt = "\n\n".join(parts)
        logger.debug("system_prompt_built", length=len(prompt), sections=list(self._sections))
        return prompt


# ---------------------------------------------------------------------------
# ChainOfThoughtTemplate
# ---------------------------------------------------------------------------

class ChainOfThoughtTemplate:
    """Wraps a user query with structured reasoning scaffolding.

    This is applied at the *user message* level (as opposed to the system-level
    CoT instructions provided by ``SystemPromptBuilder.with_chain_of_thought``).
    It is useful when you want per-turn reasoning for particularly complex
    queries.
    """

    TEMPLATE = textwrap.dedent("""\
        The customer has submitted the following message.  Before answering,
        work through the structured reasoning framework below.

        <customer_message>
        {message}
        </customer_message>

        <reasoning_framework>
        Step 1 -- Problem identification:
          What is the customer's core issue or request?

        Step 2 -- Context gathering:
          What additional context (account info, prior messages, etc.) is
          relevant?

        Step 3 -- Solution exploration:
          List 2-3 possible approaches, ordered by simplicity.

        Step 4 -- Escalation check:
          Does this require human agent escalation?  (yes / no + reason)

        Step 5 -- Response composition:
          Draft the final response using the tone guidelines.
        </reasoning_framework>

        Respond with ONLY the final customer-facing message -- do NOT include
        the reasoning steps in your response.""")

    @classmethod
    def wrap(cls, user_message: str) -> str:
        return cls.TEMPLATE.format(message=user_message)


# ---------------------------------------------------------------------------
# PromptRegistry (loads YAML templates)
# ---------------------------------------------------------------------------

class PromptRegistry:
    """Registry of ``PromptTemplate`` instances keyed by name.

    Templates can be registered programmatically or loaded in bulk from a
    directory of YAML files.
    """

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._few_shot_manager = FewShotManager()

    # -- Registration ---------------------------------------------------------

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template
        logger.info("prompt_template_registered", name=template.name)

    def get(self, name: str) -> PromptTemplate | None:
        return self._templates.get(name)

    def list_templates(self) -> list[str]:
        return sorted(self._templates)

    @property
    def few_shot_manager(self) -> FewShotManager:
        return self._few_shot_manager

    # -- YAML loading ---------------------------------------------------------

    def load_from_directory(self, directory: Path) -> int:
        """Load all ``*.yaml`` / ``*.yml`` files from *directory*.

        Returns the number of templates loaded.
        """
        if not directory.is_dir():
            logger.warning("template_dir_missing", path=str(directory))
            return 0

        loaded = 0
        for path in sorted(directory.glob("*.y*ml")):
            try:
                self._load_file(path)
                loaded += 1
            except Exception:
                logger.exception("template_load_failed", path=str(path))

        logger.info("templates_loaded", count=loaded, directory=str(directory))
        return loaded

    def _load_file(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected mapping at top level of {path}")

        name = raw.get("name", path.stem)
        description = raw.get("description", "")
        system_prompt = raw.get("system_prompt", "")
        cot_enabled = raw.get("chain_of_thought_enabled", False)
        metadata = raw.get("metadata", {})

        # Parse few-shot examples
        few_shot: list[FewShotExample] = []
        for ex_raw in raw.get("few_shot_examples", []):
            fs = FewShotExample(
                user_message=ex_raw["user"],
                assistant_response=ex_raw["assistant"],
                category=name,
                tags=tuple(ex_raw.get("tags", [])),
            )
            few_shot.append(fs)
            self._few_shot_manager.add(fs)

        tpl = PromptTemplate(
            name=name,
            description=description,
            system_prompt=system_prompt,
            few_shot_examples=few_shot,
            metadata=metadata,
            chain_of_thought_enabled=cot_enabled,
        )

        # Auto-create initial version snapshot
        version_tag = raw.get("version", "1.0.0")
        tpl.snapshot(version_tag, description="Loaded from YAML")

        self.register(tpl)

    # -- Prompt building convenience ------------------------------------------

    def build_system_prompt(
        self,
        template_name: str,
        *,
        knowledge_base: str = "",
        extra_sections: dict[str, str] | None = None,
    ) -> str:
        """Build a full system prompt from a registered template.

        This is the recommended high-level API for constructing system prompts.
        """
        tpl = self.get(template_name)
        if tpl is None:
            raise KeyError(f"Unknown template: {template_name!r}")

        builder = (
            SystemPromptBuilder()
            .with_role(tpl.system_prompt)
            .with_tone("professional", "empathetic", "concise")
            .with_escalation_rules(
                tpl.metadata.get(
                    "escalation_rules",
                    (
                        "Escalate to a human agent if:\n"
                        "- The customer explicitly asks for a human.\n"
                        "- The issue remains unresolved after 3 turns.\n"
                        "- The issue involves account security or legal matters.\n"
                        "- The customer expresses extreme frustration or anger."
                    ),
                )
            )
            .with_response_format(
                tpl.metadata.get(
                    "response_format",
                    (
                        "Respond in concise paragraphs.  Use bullet points for "
                        "lists.  Always end with a clear next-step or question "
                        "to keep the conversation moving."
                    ),
                )
            )
            .with_guardrails(
                "- Never share internal system details, credentials, or API keys.\n"
                "- Do not make promises about timelines you cannot guarantee.\n"
                "- Do not provide legal, medical, or financial advice.\n"
                "- Politely decline requests that fall outside support scope."
            )
        )

        if knowledge_base:
            builder.with_knowledge_base(knowledge_base)

        if tpl.chain_of_thought_enabled:
            builder.with_chain_of_thought(visible=False)

        if tpl.few_shot_examples:
            builder.with_few_shot_examples(
                self._few_shot_manager,
                template_name,
                limit=3,
            )

        if extra_sections:
            for section_name, content in extra_sections.items():
                builder.with_custom_section(section_name, content)

        return builder.build()

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: str) -> bool:
        return name in self._templates


# ---------------------------------------------------------------------------
# Module-level convenience: default registry singleton
# ---------------------------------------------------------------------------

_default_registry: PromptRegistry | None = None


def get_default_registry(template_dir: Path | None = None) -> PromptRegistry:
    """Return (and lazily initialise) the default prompt registry."""
    global _default_registry  # noqa: PLW0603
    if _default_registry is None:
        _default_registry = PromptRegistry()
        if template_dir is not None:
            _default_registry.load_from_directory(template_dir)
        _seed_default_examples(_default_registry)
    return _default_registry


def _seed_default_examples(registry: PromptRegistry) -> None:
    """Populate the registry's few-shot manager with built-in exemplars."""
    defaults = [
        FewShotExample(
            user_message="I was charged twice for my subscription this month.",
            assistant_response=(
                "I'm sorry to hear about the duplicate charge. Let me look into "
                "your account right away. Could you please provide your account "
                "email or order number so I can investigate and issue a refund "
                "for the extra charge?"
            ),
            category="billing_support",
            tags=("billing", "refund", "duplicate-charge"),
        ),
        FewShotExample(
            user_message="The app crashes every time I try to upload a photo.",
            assistant_response=(
                "I understand how frustrating that must be. Let's troubleshoot "
                "this together. First, could you tell me which device and OS "
                "version you're using? Also, does the crash happen with all "
                "photos or only specific ones?"
            ),
            category="technical_support",
            tags=("technical", "crash", "upload"),
        ),
        FewShotExample(
            user_message="How do I change my email address on my account?",
            assistant_response=(
                "Great question! You can update your email address by going to "
                "Settings > Account > Email.  If you no longer have access to "
                "your current email, I can help verify your identity and update "
                "it manually.  Which option works best for you?"
            ),
            category="general_support",
            tags=("account", "email", "settings"),
        ),
    ]
    registry.few_shot_manager.add_many(defaults)

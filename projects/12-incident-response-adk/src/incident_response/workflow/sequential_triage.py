"""Sequential Triage Pipeline.

Demonstrates ADK's SequentialAgent pattern by chaining:
1. ContextEnricherAgent - enrich alert with operational context
2. TriageAgent - classify severity (P1-P4)
3. ResponderAssignerAgent - assign on-call and escalation path

Each agent receives the output context of the previous agent.
"""

from __future__ import annotations

from incident_response.agents.base import SequentialAgent
from incident_response.agents.enricher import ContextEnricherAgent
from incident_response.agents.responder import ResponderAssignerAgent
from incident_response.agents.triage import TriageAgent


def build_sequential_triage() -> SequentialAgent:
    """Construct the sequential triage pipeline.

    Returns:
        A SequentialAgent that chains enrichment -> triage -> responder
        assignment in order.
    """
    enricher = ContextEnricherAgent()
    triage = TriageAgent()
    responder = ResponderAssignerAgent()

    return SequentialAgent(
        name="sequential_triage_pipeline",
        description=(
            "Sequential pipeline that enriches an alert with context, "
            "classifies severity, and assigns the appropriate responder. "
            "Mirrors Google ADK SequentialAgent pattern."
        ),
        sub_agents=[enricher, triage, responder],
    )

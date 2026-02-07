"""Compliance audit agent implementations.

Each agent wraps a lightweight ChatAgent abstraction inspired by the
Microsoft Agent Framework.  Agents fall back to heuristic analysis
when no LLM API key is configured.
"""

from compliance_audit.agents.base import AgentResponse, ChatAgent
from compliance_audit.agents.classifier import ClassifierAgent
from compliance_audit.agents.gdpr_checker import GDPRCheckerAgent
from compliance_audit.agents.remediation import RemediationAgent
from compliance_audit.agents.risk_scorer import RiskScorerAgent
from compliance_audit.agents.soc2_checker import SOC2CheckerAgent
from compliance_audit.agents.sox_checker import SOXCheckerAgent

__all__ = [
    "AgentResponse",
    "ChatAgent",
    "ClassifierAgent",
    "GDPRCheckerAgent",
    "RemediationAgent",
    "RiskScorerAgent",
    "SOC2CheckerAgent",
    "SOXCheckerAgent",
]

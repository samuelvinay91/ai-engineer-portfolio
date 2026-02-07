"""Workflow composition for the Incident Response Orchestrator.

Demonstrates ADK's three workflow agent types:
- SequentialAgent: enrichment -> triage -> responder assignment
- ParallelAgent: concurrent log, metric, and config diagnostics
- LoopAgent: remediation -> verification escalation loop
"""

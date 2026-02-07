"""Parallel Diagnostics Pipeline.

Demonstrates ADK's ParallelAgent pattern by running three diagnostic
agents concurrently:
1. LogAnalyzerAgent - search logs for error patterns
2. MetricsCheckerAgent - check metrics for anomalies
3. ConfigAuditorAgent - detect configuration drift

All agents run simultaneously via asyncio.gather and their results
are merged into the shared context.
"""

from __future__ import annotations

from incident_response.agents.base import ParallelAgent
from incident_response.agents.config_auditor import ConfigAuditorAgent
from incident_response.agents.log_analyzer import LogAnalyzerAgent
from incident_response.agents.metrics_checker import MetricsCheckerAgent


def build_parallel_diagnostics() -> ParallelAgent:
    """Construct the parallel diagnostics pipeline.

    Returns:
        A ParallelAgent that runs log, metrics, and config analysis
        concurrently using asyncio.gather.
    """
    log_analyzer = LogAnalyzerAgent()
    metrics_checker = MetricsCheckerAgent()
    config_auditor = ConfigAuditorAgent()

    return ParallelAgent(
        name="parallel_diagnostics",
        description=(
            "Parallel diagnostic pipeline that simultaneously analyzes "
            "logs, metrics, and configuration for the affected service. "
            "Mirrors Google ADK ParallelAgent pattern."
        ),
        sub_agents=[log_analyzer, metrics_checker, config_auditor],
    )

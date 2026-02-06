"""Capstone Multi-Agent AI Platform.

A production-grade orchestration platform that coordinates specialized AI agents
to solve complex, multi-faceted tasks. Combines techniques from all portfolio
projects into a unified system:

- **Researcher Agent**: Deep research with multi-step web search and synthesis
- **Coder Agent**: Code generation, review, bug fixing, and explanation
- **Analyst Agent**: Structured data analysis, insights, and trend identification
- **Writer Agent**: Content creation across styles with SEO optimization

The platform uses LangGraph for stateful orchestration, implementing a supervisor
pattern where a meta-agent decomposes tasks, delegates to specialists, and
aggregates results into coherent output.
"""

__version__ = "0.1.0"

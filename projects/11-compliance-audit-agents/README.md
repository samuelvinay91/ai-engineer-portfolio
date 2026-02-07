# Compliance Audit Agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/samuelvinay91/compliance-audit-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelvinay91/compliance-audit-agents/actions)

Multi-agent compliance audit system powered by **Microsoft Agent Framework** patterns. Dispatches specialized agents to check SOX, GDPR, and SOC2 compliance with graph-based workflows, middleware pipelines, and human-in-the-loop approval.

---

## What This Demonstrates

| Concept | Implementation |
|---------|---------------|
| **Microsoft Agent Framework** | ChatAgent abstraction, graph-based workflows, type-based routing |
| **Graph Workflow** | Directed pipeline: classify → route → check → score → remediate → approve |
| **Type-Based Routing** | Classified transactions route to SOX/GDPR/SOC2 checker agents automatically |
| **Middleware Pipeline** | Audit trail logging, PII redaction, telemetry (OpenTelemetry-compatible) |
| **Human-in-the-Loop** | Approval gate before audit report finalization |
| **Enterprise Patterns** | Compliance domain, risk scoring, remediation recommendations |

---

## Architecture

```
                          Compliance Audit Workflow
                          ========================

[Ingest Transactions]
        |
  [ClassifierAgent] ──── type-based routing
        |
   ┌────┼────┐
   ▼    ▼    ▼
 [SOX] [GDPR] [SOC2]    ← Domain checker agents (parallel)
   └────┼────┘
        ▼
  [Aggregate Findings]
        |
  [RiskScorerAgent] ──── probability × impact scoring
        |
  [RemediationAgent] ──── generates action items
        |
  [Approval Gate] ──── human approve/reject
        |
  [Final Report]

  ┌─────────────────────────────────────────┐
  │         Middleware Pipeline              │
  │  AuditTrail → PII Redaction → Telemetry │
  └─────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/samuelvinay91/compliance-audit-agents.git
cd compliance-audit-agents

# Build and run
docker build -t compliance-audit-agents .
docker run -p 8012:8000 --env-file .env compliance-audit-agents
```

The API will be available at **http://localhost:8012**. Docs at **http://localhost:8012/docs**.

### Option 2: Local Development

```bash
git clone https://github.com/samuelvinay91/compliance-audit-agents.git
cd compliance-audit-agents

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
python -m compliance_audit.main
```

### Option 3: uv (Fast)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m compliance_audit.main
```

> **No API keys required!** The system works out of the box using heuristic-based compliance checking. Add OpenAI/Anthropic keys for LLM-powered analysis.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/audits` | Submit transactions for compliance audit |
| GET | `/api/v1/audits/{id}` | Get audit session status and findings |
| GET | `/api/v1/audits/{id}/stream` | SSE stream of real-time audit progress |
| POST | `/api/v1/audits/{id}/approve` | Approve audit findings |
| POST | `/api/v1/audits/{id}/reject` | Reject findings with comments |
| GET | `/api/v1/reports/{id}` | Download structured audit report |
| GET | `/api/v1/policies` | List loaded compliance policies |
| POST | `/api/v1/policies/check` | Quick single-transaction compliance check |
| GET | `/health` | Health check |

---

## Example Usage

```bash
# Submit transactions for audit
curl -X POST http://localhost:8012/api/v1/audits \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "id": "txn-001",
        "timestamp": "2025-01-15T10:30:00Z",
        "actor": "john.smith",
        "action": "approve_and_execute_payment",
        "resource": "invoice-5001",
        "metadata": {"amount": 150000, "approved_by": "john.smith"},
        "department": "finance"
      }
    ]
  }'

# Stream audit progress
curl http://localhost:8012/api/v1/audits/{session_id}/stream

# Approve findings
curl -X POST http://localhost:8012/api/v1/audits/{session_id}/approve
```

---

## Agents

| Agent | Role | Regulation |
|-------|------|-----------|
| **ClassifierAgent** | Categorizes transactions by regulation type | All |
| **SOXCheckerAgent** | Checks segregation of duties, financial controls | SOX |
| **GDPRCheckerAgent** | Checks data privacy, consent, retention | GDPR |
| **SOC2CheckerAgent** | Checks access control, encryption, availability | SOC2 |
| **RiskScorerAgent** | Assigns severity × probability risk scores | All |
| **RemediationAgent** | Generates remediation action items | All |

---

## Middleware

| Middleware | Purpose |
|-----------|---------|
| **AuditTrailMiddleware** | Logs every agent invocation with timestamp, input/output |
| **PIIRedactionMiddleware** | Scrubs SSN, email, phone, credit card from outputs |
| **TelemetryMiddleware** | Collects timing metrics, token usage per agent |

---

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/compliance_audit
pytest tests/ -v -m "not slow and not integration"
```

---

## Project Structure

```
compliance-audit-agents/
├── src/compliance_audit/
│   ├── agents/           # ChatAgent-based compliance agents
│   ├── workflow/          # Graph-based audit pipeline
│   ├── middleware/        # Audit trail, PII redaction, telemetry
│   ├── mock_data/         # Realistic test policies & transactions
│   ├── api.py             # FastAPI application
│   ├── config.py          # Settings
│   ├── models.py          # Pydantic domain models
│   ├── streaming.py       # SSE event stream
│   └── main.py            # Entry point
├── tests/
├── k8s/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

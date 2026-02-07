# Incident Response Orchestrator

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/samuelvinay91/incident-response-adk/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelvinay91/incident-response-adk/actions)

Automated IT incident response system powered by **Google Agent Development Kit (ADK)** patterns. Triages alerts, runs parallel diagnostics, and loops through escalation/remediation until resolution. Showcases all 3 ADK workflow agent types: **SequentialAgent**, **ParallelAgent**, **LoopAgent**.

---

## What This Demonstrates

| Concept | Implementation |
|---------|---------------|
| **Google ADK** | SequentialAgent, ParallelAgent, LoopAgent, LlmAgent patterns |
| **SequentialAgent** | Alert enrichment pipeline: Enrich → Triage → Assign Responder |
| **ParallelAgent** | Concurrent diagnostics: Log Analyzer + Metrics Checker + Config Auditor |
| **LoopAgent** | Escalation loop: Remediate → Verify → Escalate (max 3 iterations) |
| **LlmAgent** | Intelligent triage with severity classification |
| **Infrastructure Tools** | Mock kubectl, log queries, metrics, health checks |
| **SRE Patterns** | Runbooks, escalation levels, on-call rotation |

---

## Architecture

```
                      Incident Response Pipeline
                      ==========================

  [Alert In]
      |
      ▼
  SequentialAgent (Triage Pipeline)
  ┌─────────────────────────────────────────┐
  │ [ContextEnricher] → [TriageAgent] → [ResponderAssigner] │
  └─────────────────────────────────────────┘
      |
      ▼
  ParallelAgent (Diagnostics)
  ┌───────────────┬────────────────┬──────────────┐
  │ [LogAnalyzer] │ [MetricsChecker] │ [ConfigAuditor] │
  └───────────────┴────────────────┴──────────────┘
      |
      ▼
  [Aggregate Diagnostics]
      |
      ▼
  LoopAgent (Escalation, max_iterations=3)
  ┌──────────────────────────────────────┐
  │ [Remediator] → [Verifier]           │
  │      ↑              |               │
  │      |         fixed? ──→ [Resolved]│
  │  [Escalate]         |               │
  │      ↑         not fixed?           │
  │      └──────────────┘               │
  └──────────────────────────────────────┘
      |
  [Resolved] or [Human Takeover]
```

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/samuelvinay91/incident-response-adk.git
cd incident-response-adk

docker build -t incident-response-adk .
docker run -p 8013:8000 --env-file .env incident-response-adk
```

The API will be available at **http://localhost:8013**. Docs at **http://localhost:8013/docs**.

### Option 2: Local Development

```bash
git clone https://github.com/samuelvinay91/incident-response-adk.git
cd incident-response-adk

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
python -m incident_response.main
```

### Option 3: uv (Fast)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m incident_response.main
```

> **No API keys required!** Built-in mock infrastructure and heuristic agents provide full functionality out of the box.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/incidents` | Submit alert for automated response |
| GET | `/api/v1/incidents/{id}` | Get incident status and diagnostics |
| GET | `/api/v1/incidents/{id}/stream` | SSE stream of response progress |
| POST | `/api/v1/incidents/{id}/escalate` | Manually escalate to next level |
| POST | `/api/v1/incidents/{id}/resolve` | Mark incident as resolved |
| POST | `/api/v1/incidents/{id}/takeover` | Human takes over from automation |
| GET | `/api/v1/incidents` | List all incidents with filters |
| GET | `/api/v1/runbooks` | List available remediation runbooks |
| GET | `/api/v1/diagnostics/{id}` | Get detailed diagnostic results |
| GET | `/health` | Health check |

---

## Example Usage

```bash
# Submit an alert
curl -X POST http://localhost:8013/api/v1/incidents \
  -H "Content-Type: application/json" \
  -d '{
    "alert": {
      "id": "alert-001",
      "source": "prometheus",
      "title": "High CPU on payment-service",
      "description": "CPU usage exceeded 95% for 5 minutes",
      "service": "payment-service",
      "host": "k8s-node-01",
      "timestamp": "2025-01-15T10:30:00Z",
      "raw_data": {"cpu_percent": 97.5}
    }
  }'

# Stream incident response progress
curl http://localhost:8013/api/v1/incidents/{session_id}/stream

# Human takeover
curl -X POST http://localhost:8013/api/v1/incidents/{session_id}/takeover
```

---

## Mock Infrastructure

The project includes a full mock infrastructure for zero-config demos:

| Component | Details |
|-----------|---------|
| **10 Services** | payment-service, order-processor, api-gateway, auth-service, etc. |
| **8 Alert Types** | CPU spike, OOM, 5xx surge, cert expiry, disk full, memory leak, etc. |
| **5 Runbooks** | restart, scale_up, rollback, clear_cache, rotate_certs |
| **On-Call Rotation** | Mock team assignments per service |
| **Baseline Metrics** | CPU, memory, error rate, latency per service |

---

## Testing

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/incident_response
pytest tests/ -v -m "not slow and not integration"
```

---

## Project Structure

```
incident-response-adk/
├── src/incident_response/
│   ├── agents/           # ADK-pattern agents (enricher, triage, diagnostics)
│   ├── workflow/          # SequentialAgent, ParallelAgent, LoopAgent
│   ├── tools/             # Mock infrastructure tools & runbooks
│   ├── mock_data/         # Alerts, services, metrics
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

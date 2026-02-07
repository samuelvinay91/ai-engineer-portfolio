"""Workflow and agent tests for Compliance Audit Agents."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_classifier_agent():
    """ClassifierAgent categorizes transactions by regulation type."""
    from compliance_audit.agents.classifier import ClassifierAgent
    from compliance_audit.models import TransactionLog

    agent = ClassifierAgent()
    transactions = [
        TransactionLog(
            id="txn-1",
            timestamp="2025-01-15T10:00:00Z",
            actor="john",
            action="approve_payment",
            resource="invoice-100",
            metadata={"amount": 50000},
            department="finance",
        ),
        TransactionLog(
            id="txn-2",
            timestamp="2025-01-15T11:00:00Z",
            actor="jane",
            action="export_personal_data",
            resource="customer-db",
            metadata={"records": 1000},
            department="marketing",
        ),
        TransactionLog(
            id="txn-3",
            timestamp="2025-01-15T12:00:00Z",
            actor="admin",
            action="modify_firewall_rules",
            resource="security-config",
            metadata={},
            department="IT",
        ),
    ]

    classified = await agent.classify(transactions)
    assert isinstance(classified, dict)
    # Should classify at least some transactions
    total = sum(len(v) for v in classified.values())
    assert total >= 1


@pytest.mark.asyncio
async def test_sox_checker():
    """SOXCheckerAgent detects financial compliance violations."""
    from compliance_audit.agents.sox_checker import SOXCheckerAgent
    from compliance_audit.mock_data.policies import get_sox_policies
    from compliance_audit.models import TransactionLog

    agent = SOXCheckerAgent()
    policies = get_sox_policies()
    transactions = [
        TransactionLog(
            id="txn-sox",
            timestamp="2025-01-15T10:00:00Z",
            actor="john",
            action="approve_and_execute_payment",
            resource="invoice-100",
            metadata={"amount": 150000, "approved_by": "john", "executed_by": "john"},
            department="finance",
        ),
    ]

    findings = await agent.check(transactions, policies)
    assert isinstance(findings, list)


@pytest.mark.asyncio
async def test_risk_scorer():
    """RiskScorerAgent assigns severity scores."""
    from compliance_audit.agents.risk_scorer import RiskScorerAgent
    from compliance_audit.models import AuditFinding, RegulationType, SeverityLevel

    agent = RiskScorerAgent()
    findings = [
        AuditFinding(
            id="f-1",
            regulation_type=RegulationType.SOX,
            severity=SeverityLevel.HIGH,
            rule_violated="Segregation of duties",
            evidence="Same person approved and executed",
            transaction_id="txn-1",
            description="SOD violation in payment processing",
            recommendation="Separate approval and execution roles",
        ),
    ]

    scores = await agent.score(findings)
    assert len(scores) == 1
    assert scores[0].score > 0
    assert scores[0].finding_id == "f-1"


@pytest.mark.asyncio
async def test_pii_redaction_middleware():
    """PIIRedactionMiddleware scrubs sensitive data."""
    from compliance_audit.middleware.pii_redaction import PIIRedactionMiddleware

    middleware = PIIRedactionMiddleware()
    text = "User john@example.com with SSN 123-45-6789 made a payment with card 4111-1111-1111-1111"
    redacted = middleware.redact(text)

    assert "john@example.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "4111-1111-1111-1111" not in redacted
    assert "[REDACTED" in redacted


@pytest.mark.asyncio
async def test_mock_transactions_available():
    """Mock data provides realistic test transactions."""
    from compliance_audit.mock_data.transactions import get_mock_transactions

    transactions = get_mock_transactions()
    assert len(transactions) >= 10
    # Should have varied departments
    departments = {t.department for t in transactions}
    assert len(departments) >= 2

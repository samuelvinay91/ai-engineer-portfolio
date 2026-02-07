"""Mock transaction logs for demonstration.

Generates 35 realistic enterprise transactions including clean ones,
SOX violations, GDPR violations, and SOC 2 violations across multiple
departments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from compliance_audit.models import TransactionLog

_BASE_TIME = datetime(2025, 1, 15, 9, 0, 0, tzinfo=timezone.utc)


def _ts(hours: int = 0, days: int = 0) -> datetime:
    """Helper to create timestamps relative to the base time."""
    return _BASE_TIME + timedelta(hours=hours, days=days)


# ---------------------------------------------------------------------------
# Clean transactions (no violations)
# ---------------------------------------------------------------------------

_CLEAN_TRANSACTIONS: list[TransactionLog] = [
    TransactionLog(
        id="tx-clean-001",
        timestamp=_ts(hours=1),
        actor="alice.johnson",
        action="view",
        resource="quarterly-report-q4",
        metadata={"read_only": True, "audit_trail": True},
        department="finance",
    ),
    TransactionLog(
        id="tx-clean-002",
        timestamp=_ts(hours=2),
        actor="bob.smith",
        action="submit",
        resource="expense-report-2025-001",
        metadata={
            "amount": 450.00,
            "approved_by": "carol.davis",
            "audit_trail": True,
        },
        department="finance",
    ),
    TransactionLog(
        id="tx-clean-003",
        timestamp=_ts(hours=3),
        actor="carol.davis",
        action="approve",
        resource="expense-report-2025-001",
        metadata={
            "amount": 450.00,
            "submitted_by": "bob.smith",
            "audit_trail": True,
        },
        department="finance",
    ),
    TransactionLog(
        id="tx-clean-004",
        timestamp=_ts(hours=4),
        actor="dave.wilson",
        action="read",
        resource="customer-analytics-dashboard",
        metadata={
            "encrypted": True,
            "mfa": True,
            "consent": True,
        },
        department="marketing",
    ),
    TransactionLog(
        id="tx-clean-005",
        timestamp=_ts(hours=5),
        actor="eve.martinez",
        action="deploy",
        resource="api-gateway-v2.3",
        metadata={
            "change_ticket": "CHG-2025-042",
            "approved_by": "frank.chen",
            "mfa": True,
            "encrypted": True,
        },
        department="engineering",
    ),
]

# ---------------------------------------------------------------------------
# SOX violation transactions
# ---------------------------------------------------------------------------

_SOX_VIOLATIONS: list[TransactionLog] = [
    # Segregation of duties: same person approves and executes
    TransactionLog(
        id="tx-sox-001",
        timestamp=_ts(days=1, hours=1),
        actor="george.baker",
        action="approve",
        resource="wire-transfer-intl-9847",
        metadata={
            "amount": 125000.00,
            "destination": "offshore-account-447",
            "audit_trail": True,
        },
        department="finance",
    ),
    TransactionLog(
        id="tx-sox-002",
        timestamp=_ts(days=1, hours=2),
        actor="george.baker",
        action="execute",
        resource="wire-transfer-intl-9847",
        metadata={
            "amount": 125000.00,
            "destination": "offshore-account-447",
        },
        department="finance",
    ),
    # High-value transaction without approval
    TransactionLog(
        id="tx-sox-003",
        timestamp=_ts(days=2, hours=1),
        actor="hannah.lee",
        action="process",
        resource="vendor-payment-batch-2025-Q1",
        metadata={
            "amount": 750000.00,
            "vendor": "Global Consulting LLC",
        },
        department="accounting",
    ),
    # Unauthorized access to financial system
    TransactionLog(
        id="tx-sox-004",
        timestamp=_ts(days=2, hours=3),
        actor="intern.james",
        action="modify",
        resource="general-ledger-entries",
        metadata={
            "unauthorized": True,
            "original_amount": 10000.00,
            "modified_amount": 15000.00,
        },
        department="finance",
    ),
    # Missing audit trail on journal entry
    TransactionLog(
        id="tx-sox-005",
        timestamp=_ts(days=3, hours=1),
        actor="karen.white",
        action="submit",
        resource="journal-entry-JE-2025-1847",
        metadata={
            "amount": 89000.00,
            "type": "financial adjustment",
        },
        department="accounting",
    ),
    # Another segregation of duties issue
    TransactionLog(
        id="tx-sox-006",
        timestamp=_ts(days=3, hours=4),
        actor="leon.harris",
        action="authorize",
        resource="purchase-order-PO-5523",
        metadata={
            "amount": 62000.00,
            "vendor": "TechSupplies Inc",
            "audit_trail": True,
        },
        department="procurement",
    ),
    TransactionLog(
        id="tx-sox-007",
        timestamp=_ts(days=3, hours=5),
        actor="leon.harris",
        action="disburse",
        resource="purchase-order-PO-5523",
        metadata={
            "amount": 62000.00,
            "vendor": "TechSupplies Inc",
        },
        department="procurement",
    ),
]

# ---------------------------------------------------------------------------
# GDPR violation transactions
# ---------------------------------------------------------------------------

_GDPR_VIOLATIONS: list[TransactionLog] = [
    # Processing personal data without consent
    TransactionLog(
        id="tx-gdpr-001",
        timestamp=_ts(days=4, hours=1),
        actor="marketing.bot",
        action="process",
        resource="customer-profiles-eu",
        metadata={
            "data_type": "personal",
            "purpose": "targeted advertising",
            "record_count": 15000,
        },
        department="marketing",
    ),
    # Data retention violation
    TransactionLog(
        id="tx-gdpr-002",
        timestamp=_ts(days=4, hours=3),
        actor="data.admin",
        action="configure",
        resource="customer-database-retention",
        metadata={
            "retention_days": 730,
            "data_type": "personal customer records",
        },
        department="data-engineering",
    ),
    # Cross-border transfer without safeguards
    TransactionLog(
        id="tx-gdpr-003",
        timestamp=_ts(days=5, hours=1),
        actor="sync.service",
        action="transfer",
        resource="user-data-replication",
        metadata={
            "destination_country": "US",
            "data_type": "personal",
            "record_count": 50000,
        },
        department="infrastructure",
    ),
    # Right-to-delete not fulfilled
    TransactionLog(
        id="tx-gdpr-004",
        timestamp=_ts(days=5, hours=3),
        actor="privacy.team",
        action="erasure",
        resource="data-subject-request-DSR-4421",
        metadata={
            "subject_email": "hans.mueller@example.de",
            "requested_date": "2024-12-01",
            "status": "pending",
            "completed": False,
        },
        department="privacy",
    ),
    # PII exposure in logs
    TransactionLog(
        id="tx-gdpr-005",
        timestamp=_ts(days=6, hours=1),
        actor="logging.service",
        action="write",
        resource="application-logs",
        metadata={
            "log_entry": "User login: john.doe@company.com, SSN: 123-45-6789",
            "encrypted": False,
        },
        department="engineering",
    ),
    # Another consent violation with profiling
    TransactionLog(
        id="tx-gdpr-006",
        timestamp=_ts(days=6, hours=3),
        actor="analytics.engine",
        action="process",
        resource="user-behavior-profiling",
        metadata={
            "data_type": "personal browsing history",
            "purpose": "profiling",
            "subjects": 8500,
        },
        department="data-science",
    ),
    # Transfer to non-EU country
    TransactionLog(
        id="tx-gdpr-007",
        timestamp=_ts(days=7, hours=1),
        actor="backup.service",
        action="replicate",
        resource="customer-backups",
        metadata={
            "transfer_destination": "India",
            "data_type": "personal",
            "encrypted": True,
        },
        department="infrastructure",
    ),
]

# ---------------------------------------------------------------------------
# SOC 2 violation transactions
# ---------------------------------------------------------------------------

_SOC2_VIOLATIONS: list[TransactionLog] = [
    # Missing MFA
    TransactionLog(
        id="tx-soc2-001",
        timestamp=_ts(days=8, hours=1),
        actor="admin.root",
        action="login",
        resource="production-database-admin",
        metadata={
            "mfa": False,
            "auth_method": "password",
            "ip_address": "203.0.113.42",
        },
        department="infrastructure",
    ),
    # Unencrypted data storage
    TransactionLog(
        id="tx-soc2-002",
        timestamp=_ts(days=8, hours=3),
        actor="data.pipeline",
        action="store",
        resource="analytics-data-lake",
        metadata={
            "encrypted": False,
            "data_classification": "confidential",
            "record_count": 200000,
        },
        department="data-engineering",
    ),
    # Unauthorized configuration change
    TransactionLog(
        id="tx-soc2-003",
        timestamp=_ts(days=9, hours=1),
        actor="junior.dev",
        action="modify",
        resource="firewall-configuration",
        metadata={
            "change_type": "rule change",
            "opened_port": 8080,
            "environment": "production",
        },
        department="engineering",
    ),
    # Unauthorized access attempt
    TransactionLog(
        id="tx-soc2-004",
        timestamp=_ts(days=9, hours=3),
        actor="unknown.actor",
        action="access denied",
        resource="secrets-vault",
        metadata={
            "attempt_type": "privilege escalation",
            "source_ip": "10.0.0.99",
            "target_role": "admin",
        },
        department="security",
    ),
    # Missing incident tracking
    TransactionLog(
        id="tx-soc2-005",
        timestamp=_ts(days=10, hours=1),
        actor="monitoring.system",
        action="detect",
        resource="vulnerability-scan-results",
        metadata={
            "severity": "critical",
            "cve": "CVE-2025-1234",
            "affected_systems": 12,
        },
        department="security",
    ),
    # Another MFA violation
    TransactionLog(
        id="tx-soc2-006",
        timestamp=_ts(days=10, hours=3),
        actor="contractor.external",
        action="access",
        resource="source-code-repository",
        metadata={
            "mfa_enabled": False,
            "auth_method": "basic",
            "access_level": "write",
        },
        department="engineering",
    ),
    # Unencrypted backup
    TransactionLog(
        id="tx-soc2-007",
        timestamp=_ts(days=11, hours=1),
        actor="backup.cron",
        action="create",
        resource="database-backup-nightly",
        metadata={
            "encryption": False,
            "backup_size_gb": 45,
            "destination": "s3://backups-unencrypted/",
        },
        department="infrastructure",
    ),
    # Security incident without tracking
    TransactionLog(
        id="tx-soc2-008",
        timestamp=_ts(days=11, hours=4),
        actor="soc.analyst",
        action="investigate",
        resource="suspicious-outbound-traffic",
        metadata={
            "incident_type": "data exfiltration attempt",
            "severity": "high",
            "source_ip": "10.0.5.22",
        },
        department="security",
    ),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MOCK_TRANSACTIONS: list[TransactionLog] = (
    _CLEAN_TRANSACTIONS
    + _SOX_VIOLATIONS
    + _GDPR_VIOLATIONS
    + _SOC2_VIOLATIONS
)


def get_mock_transactions() -> list[TransactionLog]:
    """Return all mock transactions.

    Convenience alias for ``MOCK_TRANSACTIONS``.
    """
    return list(MOCK_TRANSACTIONS)


def generate_transactions(
    include_clean: bool = True,
    include_sox: bool = True,
    include_gdpr: bool = True,
    include_soc2: bool = True,
) -> list[TransactionLog]:
    """Generate a configurable set of mock transactions.

    Parameters
    ----------
    include_clean:
        Include transactions with no violations.
    include_sox:
        Include transactions with SOX violations.
    include_gdpr:
        Include transactions with GDPR violations.
    include_soc2:
        Include transactions with SOC 2 violations.
    """
    transactions: list[TransactionLog] = []
    if include_clean:
        transactions.extend(_CLEAN_TRANSACTIONS)
    if include_sox:
        transactions.extend(_SOX_VIOLATIONS)
    if include_gdpr:
        transactions.extend(_GDPR_VIOLATIONS)
    if include_soc2:
        transactions.extend(_SOC2_VIOLATIONS)
    return transactions

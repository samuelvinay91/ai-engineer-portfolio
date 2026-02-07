"""Mock compliance policy documents for demonstration.

Provides realistic enterprise policy rules for SOX, GDPR, and SOC 2
compliance frameworks.  Each policy contains five rules that the
checker agents evaluate transactions against.
"""

from __future__ import annotations

from compliance_audit.models import PolicyDocument, RegulationType

# ---------------------------------------------------------------------------
# SOX policies
# ---------------------------------------------------------------------------

_SOX_POLICY = PolicyDocument(
    id="pol-sox-001",
    name="Sarbanes-Oxley Financial Controls Policy",
    regulation_type=RegulationType.SOX,
    content=(
        "This policy establishes internal controls over financial reporting "
        "as required by the Sarbanes-Oxley Act of 2002. It applies to all "
        "financial transactions, reporting processes, and related IT systems "
        "within the organization."
    ),
    rules=[
        (
            "SOX-R1: Segregation of Duties -- No single individual may both "
            "initiate and approve a financial transaction. Approval and "
            "execution functions must be performed by separate personnel."
        ),
        (
            "SOX-R2: Financial Authorization Controls -- All financial "
            "transactions exceeding $50,000 require documented managerial "
            "approval. Transactions exceeding $500,000 require executive-level "
            "sign-off."
        ),
        (
            "SOX-R3: Audit Trail Requirements -- Every financial transaction "
            "must maintain a complete, immutable audit trail including "
            "timestamps, actor identification, approval chain, and change history."
        ),
        (
            "SOX-R4: Access Controls -- Access to financial systems and data "
            "must follow least-privilege principles with quarterly access "
            "reviews and immediate revocation upon role change or termination."
        ),
        (
            "SOX-R5: Financial Reporting Integrity -- All journal entries, "
            "reconciliations, and financial reports must be reviewed by a "
            "qualified individual independent of the preparer before submission."
        ),
    ],
)

# ---------------------------------------------------------------------------
# GDPR policies
# ---------------------------------------------------------------------------

_GDPR_POLICY = PolicyDocument(
    id="pol-gdpr-001",
    name="General Data Protection Regulation Compliance Policy",
    regulation_type=RegulationType.GDPR,
    content=(
        "This policy establishes data protection requirements in accordance "
        "with the EU General Data Protection Regulation (GDPR). It governs "
        "the collection, processing, storage, and transfer of personal data "
        "for all EU/EEA data subjects."
    ),
    rules=[
        (
            "GDPR-R1: Lawful Basis for Processing -- All processing of personal "
            "data must have a documented lawful basis under Article 6 GDPR. "
            "Consent must be freely given, specific, informed, and unambiguous."
        ),
        (
            "GDPR-R2: Data Retention Limits -- Personal data must not be "
            "retained longer than necessary for its original purpose. Maximum "
            "retention period is 365 days unless a legal obligation requires "
            "longer retention."
        ),
        (
            "GDPR-R3: International Data Transfers -- Transfer of personal "
            "data outside the EU/EEA is prohibited unless the destination "
            "country has an adequacy decision or appropriate safeguards "
            "(e.g., Standard Contractual Clauses) are in place."
        ),
        (
            "GDPR-R4: Right to Erasure -- Data subject erasure requests must "
            "be fulfilled within 30 calendar days. All copies of the personal "
            "data must be deleted from production systems and backups within "
            "90 days."
        ),
        (
            "GDPR-R5: Data Security -- Personal data must be encrypted at "
            "rest (AES-256) and in transit (TLS 1.2+). PII must never appear "
            "in application logs, error messages, or unencrypted communications."
        ),
    ],
)

# ---------------------------------------------------------------------------
# SOC 2 policies
# ---------------------------------------------------------------------------

_SOC2_POLICY = PolicyDocument(
    id="pol-soc2-001",
    name="SOC 2 Trust Services Criteria Compliance Policy",
    regulation_type=RegulationType.SOC2,
    content=(
        "This policy implements controls required by the AICPA SOC 2 "
        "Trust Services Criteria. It covers Security, Availability, "
        "Processing Integrity, Confidentiality, and Privacy for all "
        "organizational information systems."
    ),
    rules=[
        (
            "SOC2-R1: Multi-Factor Authentication -- All access to production "
            "systems, administrative interfaces, and sensitive data must "
            "require multi-factor authentication (MFA). Password-only "
            "authentication is prohibited."
        ),
        (
            "SOC2-R2: Encryption Standards -- All data at rest must be "
            "encrypted using AES-256 or equivalent. All data in transit must "
            "use TLS 1.2 or higher. Encryption keys must be rotated annually."
        ),
        (
            "SOC2-R3: Change Management -- All changes to production systems "
            "must follow the change management process: documented change "
            "request, impact assessment, peer review, approval, and "
            "post-deployment verification."
        ),
        (
            "SOC2-R4: Access Control -- Access to systems and data must "
            "follow the principle of least privilege. Access must be reviewed "
            "quarterly and revoked immediately upon role change or termination. "
            "Unauthorized access attempts must trigger immediate alerts."
        ),
        (
            "SOC2-R5: Incident Response -- All security incidents must be "
            "logged in the incident management system with a unique tracking "
            "ID within 1 hour of detection. Incident response playbooks must "
            "be followed and post-incident reviews completed within 5 business "
            "days."
        ),
    ],
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

MOCK_POLICIES: list[PolicyDocument] = [
    _SOX_POLICY,
    _GDPR_POLICY,
    _SOC2_POLICY,
]


def get_policies_by_type(
    regulation_type: RegulationType,
) -> list[PolicyDocument]:
    """Return policies matching the given regulation type."""
    return [p for p in MOCK_POLICIES if p.regulation_type == regulation_type]


def get_sox_policies() -> list[PolicyDocument]:
    """Return SOX compliance policies."""
    return get_policies_by_type(RegulationType.SOX)


def get_gdpr_policies() -> list[PolicyDocument]:
    """Return GDPR compliance policies."""
    return get_policies_by_type(RegulationType.GDPR)


def get_soc2_policies() -> list[PolicyDocument]:
    """Return SOC 2 compliance policies."""
    return get_policies_by_type(RegulationType.SOC2)

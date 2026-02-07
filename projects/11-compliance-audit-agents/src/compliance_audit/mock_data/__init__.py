"""Mock policy documents and transaction logs for demonstration."""

from compliance_audit.mock_data.policies import (
    MOCK_POLICIES,
    get_gdpr_policies,
    get_policies_by_type,
    get_soc2_policies,
    get_sox_policies,
)
from compliance_audit.mock_data.transactions import (
    MOCK_TRANSACTIONS,
    generate_transactions,
    get_mock_transactions,
)

__all__ = [
    "MOCK_POLICIES",
    "MOCK_TRANSACTIONS",
    "generate_transactions",
    "get_gdpr_policies",
    "get_mock_transactions",
    "get_policies_by_type",
    "get_soc2_policies",
    "get_sox_policies",
]

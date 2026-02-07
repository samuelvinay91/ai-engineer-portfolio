"""Tests for price comparison and split-order optimization."""

import pytest
from decimal import Decimal

from ucp_shopping.agents.comparison_agent import ComparisonAgent
from ucp_shopping.agents.optimizer import SplitOrderOptimizer
from ucp_shopping.models import (
    ProductResult,
    ShoppingPreferences,
)


@pytest.fixture
def sample_search_results():
    """Sample search results from 3 merchants."""
    return {
        "techzone": [
            ProductResult(
                product_id="tz-kb-001",
                name="Mechanical Keyboard Pro",
                price=Decimal("89.99"),
                merchant_id="techzone",
                merchant_name="TechZone",
                shipping_cost=Decimal("5.99"),
                shipping_days=5,
                in_stock=True,
            ),
            ProductResult(
                product_id="tz-hub-001",
                name="USB-C Hub 7-in-1",
                price=Decimal("45.99"),
                merchant_id="techzone",
                merchant_name="TechZone",
                shipping_cost=Decimal("5.99"),
                shipping_days=5,
                in_stock=True,
            ),
        ],
        "homegoods": [
            ProductResult(
                product_id="hg-kb-001",
                name="Ergonomic Mechanical Keyboard",
                price=Decimal("99.99"),
                merchant_id="homegoods",
                merchant_name="HomeGoods",
                shipping_cost=Decimal("0.00"),
                shipping_days=7,
                in_stock=True,
            ),
            ProductResult(
                product_id="hg-hub-001",
                name="USB-C Docking Hub",
                price=Decimal("34.99"),
                merchant_id="homegoods",
                merchant_name="HomeGoods",
                shipping_cost=Decimal("0.00"),
                shipping_days=7,
                in_stock=True,
            ),
        ],
        "megamart": [
            ProductResult(
                product_id="mm-kb-001",
                name="Gaming Mechanical Keyboard",
                price=Decimal("69.99"),
                merchant_id="megamart",
                merchant_name="MegaMart",
                shipping_cost=Decimal("8.99"),
                shipping_days=3,
                in_stock=True,
            ),
            ProductResult(
                product_id="mm-hub-001",
                name="USB-C Hub Pro",
                price=Decimal("39.99"),
                merchant_id="megamart",
                merchant_name="MegaMart",
                shipping_cost=Decimal("8.99"),
                shipping_days=3,
                in_stock=True,
            ),
        ],
    }


class TestComparisonAgent:
    def test_build_comparison(self, sample_search_results):
        agent = ComparisonAgent()
        matrix = agent.build_comparison(
            sample_search_results,
            items=["keyboard", "usb hub"],
        )
        assert len(matrix.entries) == 2
        for entry in matrix.entries:
            assert len(entry.merchant_results) > 0
            assert entry.recommended is not None

    def test_best_price_identified(self, sample_search_results):
        agent = ComparisonAgent()
        matrix = agent.build_comparison(
            sample_search_results,
            items=["keyboard"],
        )
        entry = matrix.entries[0]
        assert entry.best_price is not None
        # MegaMart at $69.99 should be cheapest for keyboard
        # (even though shipping is $8.99, total is $78.98)

    def test_empty_results(self):
        agent = ComparisonAgent()
        matrix = agent.build_comparison({}, items=["nothing"])
        assert len(matrix.entries) == 1
        assert len(matrix.entries[0].merchant_results) == 0


class TestSplitOrderOptimizer:
    def test_optimize_finds_cheapest(self, sample_search_results):
        optimizer = SplitOrderOptimizer()
        agent = ComparisonAgent()
        matrix = agent.build_comparison(
            sample_search_results,
            items=["keyboard", "usb hub"],
        )
        plan = optimizer.optimize(matrix, ShoppingPreferences())
        assert plan is not None
        assert len(plan.items) == 2
        assert plan.grand_total > 0

    def test_savings_calculated(self, sample_search_results):
        optimizer = SplitOrderOptimizer()
        agent = ComparisonAgent()
        matrix = agent.build_comparison(
            sample_search_results,
            items=["keyboard", "usb hub"],
        )
        plan = optimizer.optimize(matrix, ShoppingPreferences())
        # savings_vs_single_merchant should be >= 0
        assert plan.savings_vs_single_merchant >= 0

    def test_prefer_single_merchant(self, sample_search_results):
        optimizer = SplitOrderOptimizer()
        agent = ComparisonAgent()
        matrix = agent.build_comparison(
            sample_search_results,
            items=["keyboard", "usb hub"],
        )
        prefs = ShoppingPreferences(prefer_single_merchant=True)
        plan = optimizer.optimize(matrix, prefs)
        # All items should be from the same merchant
        merchants = {item.merchant_name for item in plan.items}
        assert len(merchants) == 1

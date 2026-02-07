"""Tests for checkout state machine and session management."""

import pytest
from decimal import Decimal

from ucp_merchant.checkout.state_machine import CheckoutStateMachine
from ucp_merchant.checkout.session import CheckoutSessionManager
from ucp_merchant.checkout.extensions import FulfillmentExtension, DiscountExtension
from ucp_merchant.checkout.totals import TotalsCalculator
from ucp_merchant.catalog.products import ProductCatalog
from ucp_merchant.models import CheckoutState, Address


@pytest.fixture
def catalog():
    return ProductCatalog()


@pytest.fixture
def session_manager(catalog):
    fulfillment = FulfillmentExtension()
    discount = DiscountExtension()
    totals = TotalsCalculator(tax_rate=Decimal("0.0875"))
    return CheckoutSessionManager(
        catalog=catalog,
        fulfillment=fulfillment,
        discount=discount,
        totals=totals,
    )


class TestStateMachine:
    def test_valid_transition_incomplete_to_ready(self):
        sm = CheckoutStateMachine()
        assert sm.can_transition(CheckoutState.INCOMPLETE, CheckoutState.READY_FOR_COMPLETE)

    def test_valid_transition_ready_to_completed(self):
        sm = CheckoutStateMachine()
        assert sm.can_transition(CheckoutState.READY_FOR_COMPLETE, CheckoutState.COMPLETED)

    def test_invalid_transition_completed_to_incomplete(self):
        sm = CheckoutStateMachine()
        assert not sm.can_transition(CheckoutState.COMPLETED, CheckoutState.INCOMPLETE)

    def test_any_state_can_be_abandoned(self):
        sm = CheckoutStateMachine()
        for state in [CheckoutState.INCOMPLETE, CheckoutState.REQUIRES_ESCALATION,
                      CheckoutState.READY_FOR_COMPLETE]:
            assert sm.can_transition(state, CheckoutState.ABANDONED)


class TestCheckoutSession:
    def test_create_session(self, session_manager, catalog):
        products = catalog.list_products(limit=2)
        line_items = [{"product_id": p.id, "quantity": 1} for p in products]
        session = session_manager.create_session(line_items)
        assert session.state == CheckoutState.INCOMPLETE
        assert len(session.line_items) == 2
        assert session.subtotal > 0

    def test_update_address(self, session_manager, catalog):
        products = catalog.list_products(limit=1)
        session = session_manager.create_session(
            [{"product_id": products[0].id, "quantity": 1}]
        )
        address = Address(
            line1="123 Main St", city="SF",
            state="CA", postal_code="94105", country="US",
        )
        updated = session_manager.update_session(session.id, shipping_address=address)
        assert updated.shipping_address is not None
        assert updated.shipping_address.city == "SF"

    def test_state_transitions_to_ready(self, session_manager, catalog):
        products = catalog.list_products(limit=1)
        session = session_manager.create_session(
            [{"product_id": products[0].id, "quantity": 1}]
        )
        assert session.state == CheckoutState.INCOMPLETE

        # Add address
        session_manager.update_session(
            session.id,
            shipping_address=Address(
                line1="123 Main St", city="SF",
                state="CA", postal_code="94105", country="US",
            ),
        )

        # Select shipping
        updated = session_manager.update_session(
            session.id, shipping_option_id="standard"
        )
        assert updated.state == CheckoutState.READY_FOR_COMPLETE

    def test_complete_session(self, session_manager, catalog):
        products = catalog.list_products(limit=1)
        session = session_manager.create_session(
            [{"product_id": products[0].id, "quantity": 1}]
        )
        session_manager.update_session(
            session.id,
            shipping_address=Address(
                line1="123 Main St", city="SF",
                state="CA", postal_code="94105", country="US",
            ),
        )
        session_manager.update_session(session.id, shipping_option_id="standard")

        completed = session_manager.complete_session(session.id)
        assert completed.state == CheckoutState.COMPLETED
        assert completed.order_id is not None

    def test_cannot_complete_incomplete(self, session_manager, catalog):
        products = catalog.list_products(limit=1)
        session = session_manager.create_session(
            [{"product_id": products[0].id, "quantity": 1}]
        )
        with pytest.raises(ValueError):
            session_manager.complete_session(session.id)


class TestFulfillmentExtension:
    def test_get_shipping_options(self):
        ext = FulfillmentExtension()
        options = ext.get_shipping_options(subtotal=Decimal("50.00"))
        assert len(options) >= 3
        names = [o.name for o in options]
        assert "Standard" in names
        assert "Express" in names

    def test_free_shipping_threshold(self):
        ext = FulfillmentExtension()
        options = ext.get_shipping_options(subtotal=Decimal("150.00"))
        free = [o for o in options if o.cost == Decimal("0.00")]
        assert len(free) >= 1


class TestDiscountExtension:
    def test_valid_code(self):
        ext = DiscountExtension()
        code = ext.validate_code("SAVE10")
        assert code is not None
        assert code.percentage == Decimal("10")

    def test_invalid_code(self):
        ext = DiscountExtension()
        code = ext.validate_code("INVALID")
        assert code is None

    def test_freeship_code(self):
        ext = DiscountExtension()
        code = ext.validate_code("FREESHIP")
        assert code is not None
        assert code.free_shipping is True


class TestTotalsCalculator:
    def test_calculate_with_tax(self):
        calc = TotalsCalculator(tax_rate=Decimal("0.10"))
        # Simple test: $100 subtotal, 10% tax = $110
        from ucp_merchant.models import CheckoutSession, LineItem
        session = CheckoutSession(
            id="test",
            state=CheckoutState.INCOMPLETE,
            line_items=[
                LineItem(
                    product_id="p1", name="Test", quantity=1,
                    unit_price=Decimal("100.00"), total=Decimal("100.00"),
                )
            ],
            subtotal=Decimal("100.00"),
        )
        result = calc.calculate(session)
        assert result.tax == Decimal("10.00")
        assert result.total == Decimal("110.00")

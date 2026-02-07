"""Tests for UCP Merchant Server API."""

import pytest
from decimal import Decimal
from httpx import ASGITransport, AsyncClient

from ucp_merchant.main import create_app
from ucp_merchant.config import MerchantSettings


@pytest.fixture
def settings():
    return MerchantSettings(
        environment="testing",
        merchant_name="Test Store",
        ap2_enabled=True,
        mcp_enabled=True,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    async def test_health_check(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ucp-merchant-server"


class TestDiscovery:
    async def test_well_known_ucp(self, client):
        resp = await client.get("/.well-known/ucp")
        assert resp.status_code == 200
        data = resp.json()
        assert "services" in data
        assert data["version"] == "2026-01-11"

    async def test_manifest_has_capabilities(self, client):
        resp = await client.get("/.well-known/ucp")
        data = resp.json()
        services = data["services"]
        assert len(services) > 0
        shopping = services[0]
        assert "capabilities" in shopping
        cap_names = [c["name"] for c in shopping["capabilities"]]
        assert "dev.ucp.shopping.checkout" in cap_names

    async def test_manifest_has_extensions(self, client):
        resp = await client.get("/.well-known/ucp")
        data = resp.json()
        shopping = data["services"][0]
        assert "extensions" in shopping
        ext_names = [e["name"] for e in shopping["extensions"]]
        assert "dev.ucp.shopping.fulfillment" in ext_names
        assert "dev.ucp.shopping.discount" in ext_names

    async def test_negotiate_capabilities(self, client):
        resp = await client.post("/api/v1/negotiate", json={
            "agent_capabilities": ["dev.ucp.shopping.checkout"],
            "agent_extensions": ["dev.ucp.shopping.fulfillment"],
            "agent_payment_handlers": ["dev.ucp.mock_payment"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "supported_capabilities" in data
        assert "dev.ucp.shopping.checkout" in data["supported_capabilities"]


class TestCatalog:
    async def test_list_products(self, client):
        resp = await client.get("/api/v1/catalog/products")
        assert resp.status_code == 200
        data = resp.json()
        assert "products" in data
        assert "total" in data
        assert data["total"] > 0

    async def test_search_products(self, client):
        resp = await client.get("/api/v1/catalog/products", params={"q": "laptop"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 0

    async def test_filter_by_category(self, client):
        resp = await client.get("/api/v1/catalog/products", params={"category": "Laptops"})
        assert resp.status_code == 200
        data = resp.json()
        for p in data["products"]:
            assert p["category"] == "Laptops"

    async def test_filter_by_price_range(self, client):
        resp = await client.get("/api/v1/catalog/products", params={
            "min_price": 50, "max_price": 200
        })
        assert resp.status_code == 200
        data = resp.json()
        for p in data["products"]:
            assert 50 <= float(p["price"]) <= 200

    async def test_get_product_by_id(self, client):
        # First get a product list
        resp = await client.get("/api/v1/catalog/products", params={"limit": 1})
        products = resp.json()["products"]
        if products:
            product_id = products[0]["id"]
            resp = await client.get(f"/api/v1/catalog/products/{product_id}")
            assert resp.status_code == 200
            assert resp.json()["id"] == product_id

    async def test_get_categories(self, client):
        resp = await client.get("/api/v1/catalog/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert "categories" in data
        assert len(data["categories"]) > 0

    async def test_sort_by_price(self, client):
        resp = await client.get("/api/v1/catalog/products", params={"sort_by": "price_asc"})
        assert resp.status_code == 200
        products = resp.json()["products"]
        if len(products) >= 2:
            prices = [float(p["price"]) for p in products]
            assert prices == sorted(prices)


class TestCheckout:
    async def _create_checkout(self, client):
        resp = await client.get("/api/v1/catalog/products", params={"limit": 2})
        products = resp.json()["products"][:2]
        line_items = [
            {"product_id": p["id"], "quantity": 1}
            for p in products
        ]
        resp = await client.post("/api/v1/checkout/sessions", json={
            "line_items": line_items
        })
        return resp

    async def test_create_session(self, client):
        resp = await self._create_checkout(client)
        assert resp.status_code == 201
        data = resp.json()
        assert data["state"] == "incomplete"
        assert len(data["line_items"]) > 0
        assert "id" in data

    async def test_get_session(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        resp = await client.get(f"/api/v1/checkout/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == session_id

    async def test_update_session_address(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        resp = await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_address": {
                "line1": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["shipping_address"] is not None
        assert data["shipping_address"]["city"] == "San Francisco"

    async def test_select_shipping(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        # Set address first
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_address": {
                "line1": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
            }
        })
        # Select shipping
        resp = await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_option_id": "standard"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["selected_shipping"] is not None

    async def test_apply_discount(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        resp = await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "discount_code": "SAVE10"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["applied_discounts"]) > 0

    async def test_checkout_state_machine(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        assert create_resp.json()["state"] == "incomplete"

        # Add address
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_address": {
                "line1": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
            }
        })

        # Select shipping
        resp = await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_option_id": "standard"
        })
        data = resp.json()
        assert data["state"] == "ready_for_complete"

    async def test_complete_checkout(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]

        # Fill required fields
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_address": {
                "line1": "123 Main St",
                "city": "San Francisco",
                "state": "CA",
                "postal_code": "94105",
                "country": "US",
            }
        })
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_option_id": "standard"
        })

        resp = await client.post(f"/api/v1/checkout/sessions/{session_id}/complete")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "completed"
        assert data.get("order_id") is not None

    async def test_cannot_complete_incomplete_session(self, client):
        create_resp = await self._create_checkout(client)
        session_id = create_resp.json()["id"]
        resp = await client.post(f"/api/v1/checkout/sessions/{session_id}/complete")
        assert resp.status_code == 400


class TestPayments:
    async def test_generate_key_pair(self, client):
        resp = await client.post("/api/v1/payments/keys")
        assert resp.status_code == 201
        data = resp.json()
        assert "key_id" in data
        assert "public_key_jwk" in data
        assert data["algorithm"] == "ES256"

    async def test_get_public_key(self, client):
        create_resp = await client.post("/api/v1/payments/keys")
        key_id = create_resp.json()["key_id"]
        resp = await client.get(f"/api/v1/payments/keys/{key_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["kty"] == "EC"
        assert data["crv"] == "P-256"

    async def test_verify_test_mandate(self, client):
        # Generate key
        key_resp = await client.post("/api/v1/payments/keys")
        key_id = key_resp.json()["key_id"]

        # Create checkout session
        products_resp = await client.get("/api/v1/catalog/products", params={"limit": 1})
        product = products_resp.json()["products"][0]
        create_resp = await client.post("/api/v1/checkout/sessions", json={
            "line_items": [{"product_id": product["id"], "quantity": 1}]
        })
        session_id = create_resp.json()["id"]

        # Create test mandate
        resp = await client.post("/api/v1/payments/test-mandate", json={
            "session_id": session_id,
            "key_id": key_id,
        })
        assert resp.status_code == 200
        mandate = resp.json()
        assert "signature" in mandate

        # Verify it
        resp = await client.post("/api/v1/payments/verify-mandate", json=mandate)
        assert resp.status_code == 200
        result = resp.json()
        assert result["valid"] is True


class TestOrders:
    async def _create_completed_checkout(self, client):
        products_resp = await client.get("/api/v1/catalog/products", params={"limit": 1})
        product = products_resp.json()["products"][0]
        create_resp = await client.post("/api/v1/checkout/sessions", json={
            "line_items": [{"product_id": product["id"], "quantity": 1}]
        })
        session_id = create_resp.json()["id"]
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_address": {
                "line1": "123 Main St", "city": "SF",
                "state": "CA", "postal_code": "94105", "country": "US",
            }
        })
        await client.put(f"/api/v1/checkout/sessions/{session_id}", json={
            "shipping_option_id": "standard"
        })
        resp = await client.post(f"/api/v1/checkout/sessions/{session_id}/complete")
        return resp.json()

    async def test_get_order(self, client):
        session = await self._create_completed_checkout(client)
        order_id = session["order_id"]
        resp = await client.get(f"/api/v1/orders/{order_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == order_id
        assert data["state"] == "confirmed"

    async def test_simulate_fulfillment(self, client):
        session = await self._create_completed_checkout(client)
        order_id = session["order_id"]
        resp = await client.post(f"/api/v1/testing/simulate-fulfillment/{order_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "shipped"
        assert data.get("tracking_number") is not None


class TestMCP:
    async def test_list_tools(self, client):
        resp = await client.post("/api/v1/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        assert len(data["tools"]) >= 6
        tool_names = [t["name"] for t in data["tools"]]
        assert "search_products" in tool_names
        assert "create_checkout" in tool_names

    async def test_execute_search_tool(self, client):
        resp = await client.post("/api/v1/mcp/tools/search_products/execute", json={
            "arguments": {"query": "laptop"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

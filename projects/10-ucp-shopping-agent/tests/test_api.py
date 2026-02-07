"""Tests for UCP Shopping Agent API."""

import pytest
from httpx import ASGITransport, AsyncClient

from ucp_shopping.main import create_app
from ucp_shopping.config import ShoppingSettings


@pytest.fixture
def settings():
    return ShoppingSettings(
        environment="testing",
        openai_api_key="test-key",
        human_confirmation_required=False,
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
        assert data["service"] == "ucp-shopping-agent"


class TestMockMerchants:
    async def test_techzone_discovery(self, client):
        resp = await client.get("/merchants/techzone/.well-known/ucp")
        assert resp.status_code == 200
        data = resp.json()
        assert "services" in data

    async def test_homegoods_discovery(self, client):
        resp = await client.get("/merchants/homegoods/.well-known/ucp")
        assert resp.status_code == 200

    async def test_megamart_discovery(self, client):
        resp = await client.get("/merchants/megamart/.well-known/ucp")
        assert resp.status_code == 200

    async def test_techzone_catalog(self, client):
        resp = await client.get("/merchants/techzone/api/v1/catalog/products")
        assert resp.status_code == 200
        data = resp.json()
        assert "products" in data
        assert data["total"] > 0

    async def test_search_across_merchants(self, client):
        for merchant in ["techzone", "homegoods", "megamart"]:
            resp = await client.get(
                f"/merchants/{merchant}/api/v1/catalog/products",
                params={"q": "keyboard"},
            )
            assert resp.status_code == 200


class TestMerchantDiscovery:
    async def test_list_merchants(self, client):
        resp = await client.get("/api/v1/merchants")
        assert resp.status_code == 200
        data = resp.json()
        assert "merchants" in data
        assert len(data["merchants"]) >= 3

    async def test_discover_merchants(self, client):
        resp = await client.post("/api/v1/merchants/discover", json={
            "urls": ["http://test/merchants/techzone"]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "discovered" in data


class TestComparison:
    async def test_compare_product(self, client):
        resp = await client.post("/api/v1/compare", json={
            "query": "mechanical keyboard",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data

    async def test_optimize_split_order(self, client):
        resp = await client.post("/api/v1/optimize", json={
            "items": ["mechanical keyboard", "USB hub"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "plan" in data


class TestShopping:
    async def test_submit_shopping_request(self, client):
        resp = await client.post("/api/v1/shop", json={
            "query": "Find me a mechanical keyboard under $100",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["state"] in [
            "planning", "discovering", "searching",
            "comparing", "completed",
        ]

    async def test_get_shopping_session(self, client):
        create_resp = await client.post("/api/v1/shop", json={
            "query": "laptop",
        })
        session_id = create_resp.json()["session_id"]
        resp = await client.get(f"/api/v1/shop/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == session_id

    async def test_cancel_shopping(self, client):
        create_resp = await client.post("/api/v1/shop", json={
            "query": "mouse",
        })
        session_id = create_resp.json()["session_id"]
        resp = await client.post(f"/api/v1/shop/{session_id}/cancel")
        assert resp.status_code == 200


class TestMCPTools:
    async def test_list_tools(self, client):
        resp = await client.post("/api/v1/mcp/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert "tools" in data
        tool_names = [t["name"] for t in data["tools"]]
        assert "shop" in tool_names
        assert "compare_prices" in tool_names

    async def test_execute_compare_tool(self, client):
        resp = await client.post("/api/v1/mcp/tools/compare_prices/execute", json={
            "arguments": {"query": "keyboard"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

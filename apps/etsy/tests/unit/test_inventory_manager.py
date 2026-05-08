"""Unit tests for InventoryManager — focuses on the GET-vs-PUT schema asymmetry.

The Etsy v3 inventory PUT body must NOT include read-only fields the GET
response carries (product_id, offering_id, is_deleted) and must send price
as a float, not as the Money dict the GET returns.

Verified against the official OpenAPI spec on 2026-05-08:
https://www.etsy.com/openapi/generated/oas/3.0.0.json
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from etsy_mcp.managers.inventory_manager import (
    InventoryManager,
    _money_to_float,
    _strip_offering,
    _strip_product,
    _strip_property_value,
)


# --- pure-function strippers -------------------------------------------------


def test_strip_product_removes_read_only_fields() -> None:
    raw = {
        "product_id": 12345678,
        "is_deleted": False,
        "sku": "ABC-123",
        "property_values": [],
        "offerings": [],
    }
    out = _strip_product(raw)
    assert "product_id" not in out
    assert "is_deleted" not in out
    assert out["sku"] == "ABC-123"
    assert out["property_values"] == []
    assert out["offerings"] == []


def test_strip_offering_removes_read_only_fields_and_converts_price() -> None:
    raw = {
        "offering_id": 99999999,
        "is_deleted": False,
        "price": {"amount": 450, "divisor": 100, "currency_code": "EUR"},
        "quantity": 41,
        "is_enabled": True,
        "readiness_state_id": None,
    }
    out = _strip_offering(raw)
    assert "offering_id" not in out
    assert "is_deleted" not in out
    assert out["price"] == 4.5
    assert out["quantity"] == 41
    assert out["is_enabled"] is True


def test_strip_offering_passes_through_plain_float_price() -> None:
    """Callers that already constructed a PUT-shape offering must not be broken."""
    raw = {"price": 4.5, "quantity": 10}
    assert _strip_offering(raw) == {"price": 4.5, "quantity": 10}


def test_strip_property_value_keeps_only_writable_keys() -> None:
    raw = {
        "property_id": 200,
        "value_ids": [1, 2, 3],
        "scale_id": 19,
        "property_name": "Größe",
        "values": ["S", "M", "L"],
        "extra_garbage": "should be dropped",
    }
    out = _strip_property_value(raw)
    assert "extra_garbage" not in out
    assert out["property_id"] == 200
    assert out["value_ids"] == [1, 2, 3]
    assert out["scale_id"] == 19
    assert out["property_name"] == "Größe"
    assert out["values"] == ["S", "M", "L"]


def test_money_to_float_handles_zero_divisor_defensively() -> None:
    assert _money_to_float({"amount": 1234, "divisor": 0, "currency_code": "EUR"}) == 1234.0


def test_money_to_float_passthrough_on_non_money() -> None:
    assert _money_to_float(4.5) == 4.5
    assert _money_to_float(None) is None
    assert _money_to_float({"foo": "bar"}) == {"foo": "bar"}


# --- end-to-end update() ----------------------------------------------------


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.get = AsyncMock()
    client.put = AsyncMock()
    return client


def _simple_listing_inventory() -> dict[str, Any]:
    """Shape of GET /listings/{id}/inventory for a no-variant listing.

    Mirrors what Kathrin's Mappe-Glitter listing returned on 2026-05-08
    (one product, one offering, price as Money dict, read-only fields).
    """
    return {
        "products": [
            {
                "product_id": 22222222,
                "is_deleted": False,
                "sku": "MAPPE-GLITTER",
                "property_values": [],
                "offerings": [
                    {
                        "offering_id": 33333333,
                        "is_deleted": False,
                        "price": {"amount": 450, "divisor": 100, "currency_code": "EUR"},
                        "quantity": 41,
                        "is_enabled": True,
                    }
                ],
            }
        ],
        "price_on_property": [],
        "quantity_on_property": [],
        "sku_on_property": [],
    }


@pytest.mark.asyncio
async def test_update_strips_read_only_fields_in_put_body(mock_client: AsyncMock) -> None:
    """The bug we found 2026-05-08: PUT body must not contain product_id /
    is_deleted / offering_id, and price must be a float."""
    mock_client.get.return_value = _simple_listing_inventory()
    mock_client.put.return_value = {"products": [], "success": True}

    mgr = InventoryManager(client=mock_client)
    incoming = {
        "products": [
            {
                "product_id": 22222222,
                "sku": "MAPPE-GLITTER",
                "offerings": [
                    {
                        "offering_id": 33333333,
                        "quantity": 100,
                    }
                ],
            }
        ]
    }
    await mgr.update(listing_id=760203240, inventory=incoming)

    mock_client.put.assert_awaited_once()
    sent_body = mock_client.put.await_args.kwargs["json"]

    sent_product = sent_body["products"][0]
    assert "product_id" not in sent_product
    assert "is_deleted" not in sent_product
    assert sent_product["sku"] == "MAPPE-GLITTER"

    sent_offering = sent_product["offerings"][0]
    assert "offering_id" not in sent_offering
    assert "is_deleted" not in sent_offering
    assert sent_offering["quantity"] == 100
    assert sent_offering["price"] == 4.5  # converted from Money dict


@pytest.mark.asyncio
async def test_update_offering_quantity_strips_fields_too(mock_client: AsyncMock) -> None:
    """The convenience wrapper must also produce a clean PUT body."""
    mock_client.get.return_value = _simple_listing_inventory()
    mock_client.put.return_value = {"products": [], "success": True}

    mgr = InventoryManager(client=mock_client)
    await mgr.update_offering_quantity(
        listing_id=760203240,
        product_id=22222222,
        offering_id=33333333,
        quantity=100,
    )

    sent_body = mock_client.put.await_args.kwargs["json"]
    product = sent_body["products"][0]
    assert "product_id" not in product
    assert "is_deleted" not in product
    offering = product["offerings"][0]
    assert "offering_id" not in offering
    assert "is_deleted" not in offering
    assert offering["quantity"] == 100
    assert offering["price"] == 4.5

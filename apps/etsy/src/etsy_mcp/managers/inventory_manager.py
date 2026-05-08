"""Inventory manager — wraps Etsy ListingInventory + Product + Offering endpoints.

5 operations:
- get: fetch full inventory tree for a listing
- update: fetch-merge-PUT on inventory with 6-tuple identity key for offering merging
- get_product: fetch a specific product
- get_offering: fetch a specific product offering
- update_offering_quantity: convenience wrapper that does get -> modify -> put

The 6-tuple identity key (sku, property_values_sorted, quantity, price, is_enabled,
offering_id) prevents collapsing distinct offerings that happen to share some
fields. Critical for variation-heavy listings (e.g. MX/SRV-style multi-axis).

GET-vs-PUT schema asymmetry: the GET response includes read-only fields
(product_id, offering_id, is_deleted) and `price` as a Money dict. The PUT
body must NOT include those read-only fields, and `price` must be a float.
We strip + convert before sending. Etsy rejects the request otherwise with
"Validation error: Array contains invalid keys: product_id,is_deleted".

Managers return raw Etsy response dicts. Tool layer handles envelopes.
"""

from __future__ import annotations

import logging
from typing import Any

from etsy_core.client import EtsyClient

logger = logging.getLogger(__name__)


def _offering_identity_key(offering: dict[str, Any]) -> tuple:
    """Build a STABLE identity for an offering — must not change when the
    caller patches mutable fields like quantity or price.

    Identity rules:
    - If `offering_id` (or its alias `product_offering_id`) is present, use
      it. That's Etsy's canonical identifier and never changes.
    - Otherwise (new offering being created), fall back to (sku, property_values).
      That uniquely identifies a variation row within a product.

    Earlier versions of this function included quantity, price, and is_enabled
    in the identity, which meant the same offering_id had a different identity
    after patching quantity — defeating the merge. The PUT then contained two
    offerings sharing one offering_id, and Etsy applied the last-wins value.
    """
    offering_id = offering.get("offering_id") or offering.get("product_offering_id")
    if offering_id is not None:
        return ("by_id", offering_id)

    sku = offering.get("sku")
    pv_raw = offering.get("property_values") or []
    pv_pairs = []
    for pv in pv_raw:
        pid = pv.get("property_id")
        vids = pv.get("value_ids") or []
        for vid in vids:
            pv_pairs.append((pid, vid))
    pv_sorted = tuple(sorted(pv_pairs))
    return ("new", sku, pv_sorted)


# Per Etsy OpenAPI spec for PUT /v3/application/listings/{listing_id}/inventory.
# Sourced from https://www.etsy.com/openapi/generated/oas/3.0.0.json (verified
# 2026-05-08). Anything outside these sets is rejected with HTTP 400 / "Array
# contains invalid keys".
_PRODUCT_PUT_FIELDS: frozenset[str] = frozenset({"sku", "property_values", "offerings"})
_OFFERING_PUT_FIELDS: frozenset[str] = frozenset(
    {"price", "quantity", "is_enabled", "readiness_state_id"}
)
_PROPERTY_VALUE_PUT_FIELDS: frozenset[str] = frozenset(
    {"property_id", "value_ids", "scale_id", "property_name", "values"}
)
_INVENTORY_TOP_LEVEL_PUT_FIELDS: frozenset[str] = frozenset(
    {
        "products",
        "price_on_property",
        "quantity_on_property",
        "readiness_state_on_property",
        "sku_on_property",
    }
)


def _money_to_float(value: Any) -> Any:
    """Convert Etsy Money dict (GET response) to float (PUT body).

    GET returns `price` as `{amount, divisor, currency_code}`. PUT requires a
    plain number. Pass plain numbers through untouched so callers that already
    constructed PUT-shape data aren't broken.
    """
    if isinstance(value, dict) and "amount" in value and "divisor" in value:
        amount = value.get("amount") or 0
        divisor = value.get("divisor") or 1
        if divisor == 0:
            return float(amount)
        return float(amount) / float(divisor)
    return value


def _strip_property_value(pv: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pv.items() if k in _PROPERTY_VALUE_PUT_FIELDS}


def _strip_offering(offering: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in offering.items() if k in _OFFERING_PUT_FIELDS}
    if "price" in out:
        out["price"] = _money_to_float(out["price"])
    return out


def _strip_product(product: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {k: v for k, v in product.items() if k in _PRODUCT_PUT_FIELDS}
    if "property_values" in out and isinstance(out["property_values"], list):
        out["property_values"] = [_strip_property_value(pv) for pv in out["property_values"]]
    if "offerings" in out and isinstance(out["offerings"], list):
        out["offerings"] = [_strip_offering(off) for off in out["offerings"]]
    return out


def _product_identity_key(product: dict[str, Any]) -> tuple:
    """Build a STABLE identity for a product.

    Same rule as offerings: prefer Etsy's product_id when present, fall back
    to (sku, property_values) for new products. Older versions included
    product_id at the end of an sku+pv tuple, which split distinct products
    that happened to share sku+pv but had different IDs.
    """
    product_id = product.get("product_id")
    if product_id is not None:
        return ("by_id", product_id)

    sku = product.get("sku")
    pv_raw = product.get("property_values") or []
    pv_pairs = []
    for pv in pv_raw:
        pid = pv.get("property_id")
        vids = pv.get("value_ids") or []
        for vid in vids:
            pv_pairs.append((pid, vid))
    pv_sorted = tuple(sorted(pv_pairs))
    return ("new", sku, pv_sorted)


class InventoryManager:
    """Manages Etsy ListingInventory + Product + Offering operations."""

    def __init__(self, client: EtsyClient) -> None:
        self.client = client

    async def get(self, listing_id: int) -> dict[str, Any]:
        """GET /listings/{listing_id}/inventory"""
        return await self.client.get(f"/listings/{listing_id}/inventory")

    async def get_product(self, listing_id: int, product_id: int) -> dict[str, Any]:
        """GET /listings/{listing_id}/inventory/products/{product_id}"""
        return await self.client.get(
            f"/listings/{listing_id}/inventory/products/{product_id}"
        )

    async def get_offering(
        self,
        listing_id: int,
        product_id: int,
        product_offering_id: int,
    ) -> dict[str, Any]:
        """GET /listings/{listing_id}/inventory/products/{product_id}/offerings/{product_offering_id}"""
        return await self.client.get(
            f"/listings/{listing_id}/inventory/products/{product_id}/offerings/{product_offering_id}"
        )

    async def update(
        self,
        listing_id: int,
        inventory: dict[str, Any],
    ) -> dict[str, Any]:
        """PUT /listings/{listing_id}/inventory with fetch-merge-put semantics.

        Fetches current inventory, merges caller's partial product/offering
        updates using a 6-tuple identity key on offerings to avoid collapsing
        distinct variation rows. Sends the full inventory document back.
        """
        current = await self.get(listing_id)
        current_products: list[dict[str, Any]] = list(current.get("products") or [])
        incoming_products: list[dict[str, Any]] = list(inventory.get("products") or [])

        # Index current products by identity key
        current_by_key: dict[tuple, dict[str, Any]] = {}
        for prod in current_products:
            current_by_key[_product_identity_key(prod)] = prod

        merged_products: list[dict[str, Any]] = []
        seen_keys: set[tuple] = set()

        for inc_prod in incoming_products:
            key = _product_identity_key(inc_prod)
            base = current_by_key.get(key)
            if base is None:
                # New product, take incoming as-is
                merged_products.append(inc_prod)
                seen_keys.add(key)
                continue

            seen_keys.add(key)
            merged: dict[str, Any] = {**base, **inc_prod}

            # Merge offerings using 6-tuple identity
            base_offerings = base.get("offerings") or []
            inc_offerings = inc_prod.get("offerings") or []
            base_off_by_key: dict[tuple, dict[str, Any]] = {
                _offering_identity_key(o): o for o in base_offerings
            }
            merged_offerings: list[dict[str, Any]] = []
            seen_off_keys: set[tuple] = set()

            for inc_off in inc_offerings:
                ok = _offering_identity_key(inc_off)
                base_off = base_off_by_key.get(ok)
                if base_off is None:
                    merged_offerings.append(inc_off)
                else:
                    merged_offerings.append({**base_off, **inc_off})
                seen_off_keys.add(ok)

            # Carry over base offerings the caller did not touch
            for ok, base_off in base_off_by_key.items():
                if ok not in seen_off_keys:
                    merged_offerings.append(base_off)

            merged["offerings"] = merged_offerings
            merged_products.append(merged)

        # Carry over current products the caller did not touch
        for key, prod in current_by_key.items():
            if key not in seen_keys:
                merged_products.append(prod)

        # Strip read-only fields (product_id, offering_id, is_deleted) and
        # convert price Money dicts to floats before PUT. Etsy's PUT schema is
        # narrower than its GET schema; sending GET-shape data unchanged
        # produces "Validation error: Array contains invalid keys".
        stripped_products = [_strip_product(p) for p in merged_products]

        payload: dict[str, Any] = {"products": stripped_products}
        # Pass through top-level inventory hints if caller supplied them.
        # Order matches the OpenAPI spec; readiness_state_on_property is also
        # accepted but rarely set, kept consistent with the rest.
        for k in (
            "price_on_property",
            "quantity_on_property",
            "readiness_state_on_property",
            "sku_on_property",
        ):
            if k in inventory:
                payload[k] = inventory[k]
            elif k in current:
                payload[k] = current[k]

        return await self.client.put(
            f"/listings/{listing_id}/inventory",
            json=payload,
            idempotent=True,
        )

    async def update_offering_quantity(
        self,
        listing_id: int,
        product_id: int,
        offering_id: int,
        quantity: int,
    ) -> dict[str, Any]:
        """Convenience: get -> modify single offering quantity -> put.

        Uses the full-inventory fetch-merge-put path so the 6-tuple identity
        merge protects every other offering.
        """
        current = await self.get(listing_id)
        partial: dict[str, Any] = {"products": []}

        for prod in current.get("products") or []:
            if prod.get("product_id") != product_id:
                continue
            patched_offerings: list[dict[str, Any]] = []
            for off in prod.get("offerings") or []:
                off_id = off.get("offering_id") or off.get("product_offering_id")
                if off_id == offering_id:
                    patched_offerings.append({**off, "quantity": int(quantity)})
                else:
                    patched_offerings.append(off)
            partial["products"].append({**prod, "offerings": patched_offerings})
            break

        if not partial["products"]:
            raise ValueError(
                f"product_id={product_id} not found on listing {listing_id}"
            )

        return await self.update(listing_id, partial)

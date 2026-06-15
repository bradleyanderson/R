#!/usr/bin/env python3
"""
Vintage Bait — bulk product sync.

Reads product-designs/manifest/*.csv and upserts matching products in the
connected Shopify store via the Admin GraphQL API: creates/updates the
product (productSet), ensures it's attached to its collection, uploads
design artwork if present under product-designs/artwork/, and syncs
inventory at the default location.

Idempotent: re-running updates existing products (matched by handle)
instead of creating duplicates.

Setup:
    pip install -r scripts/requirements.txt
    cp .env.example .env   # fill in SHOPIFY_STORE_DOMAIN / SHOPIFY_ADMIN_ACCESS_TOKEN

Usage:
    python scripts/bulk_create_products.py [--manifest path/to/file.csv ...] [--dry-run]

NOTE ON GRAPHQL SHAPES: the mutation/field names below (productSet,
collectionAddProductsV2, stagedUploadsCreate, productCreateMedia,
inventorySetQuantities) reflect the Admin API as of the 2024-2025 releases.
Before running against a live store, validate the operations in this file
with the connected Shopify MCP's `graphql_schema` / `validate_graphql_codeblocks`
tools (or `shopify-cli`'s GraphiQL) in case your store's API version differs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARTWORK_DIR = ROOT / "product-designs" / "artwork"
DEFAULT_MANIFESTS = [
    ROOT / "product-designs" / "manifest" / "vintage-apparel.csv",
    ROOT / "product-designs" / "manifest" / "fishing-merch.csv",
]

API_VERSION = os.environ.get("SHOPIFY_API_VERSION", "2025-01")


class ShopifyClient:
    def __init__(self, store_domain: str, access_token: str):
        self.endpoint = f"https://{store_domain}/admin/api/{API_VERSION}/graphql.json"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "X-Shopify-Access-Token": access_token,
                "Content-Type": "application/json",
            }
        )

    def execute(self, query: str, variables: dict | None = None) -> dict:
        resp = self.session.post(
            self.endpoint, json={"query": query, "variables": variables or {}}, timeout=30
        )
        resp.raise_for_status()
        payload = resp.json()
        if "errors" in payload:
            raise RuntimeError(f"GraphQL error: {json.dumps(payload['errors'], indent=2)}")
        return payload["data"]


def check_user_errors(result: dict, key: str, errors_key: str = "userErrors") -> None:
    errors = result.get(key, {}).get(errors_key) or []
    if errors:
        raise RuntimeError(f"{key} returned errors: {json.dumps(errors, indent=2)}")


def get_default_location_id(client: ShopifyClient) -> str:
    data = client.execute("query { locations(first: 1) { nodes { id name } } }")
    nodes = data["locations"]["nodes"]
    if not nodes:
        raise RuntimeError("Store has no locations configured.")
    return nodes[0]["id"]


def get_or_create_collection_id(client: ShopifyClient, handle: str) -> str:
    data = client.execute(
        "query($handle: String!) { collectionByHandle(handle: $handle) { id } }",
        {"handle": handle},
    )
    existing = data.get("collectionByHandle")
    if existing:
        return existing["id"]

    title = handle.replace("-", " ").title()
    print(f"  collection '{handle}' not found — creating '{title}'")
    data = client.execute(
        """
        mutation($input: CollectionInput!) {
          collectionCreate(input: $input) {
            collection { id }
            userErrors { field message }
          }
        }
        """,
        {"input": {"title": title, "handle": handle}},
    )
    check_user_errors(data, "collectionCreate")
    return data["collectionCreate"]["collection"]["id"]


def get_product_id_by_handle(client: ShopifyClient, handle: str) -> str | None:
    data = client.execute(
        "query($handle: String!) { productByHandle(handle: $handle) { id } }",
        {"handle": handle},
    )
    product = data.get("productByHandle")
    return product["id"] if product else None


def build_product_set_input(row: dict, location_id: str, existing_id: str | None) -> dict:
    tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
    quantity = int(row["initial_inventory"] or 0)

    product_input: dict[str, Any] = {
        "handle": row["handle"],
        "title": row["title"],
        "descriptionHtml": f"<p>{row['description']}</p>",
        "productType": row["product_type"],
        "tags": tags,
        "status": "DRAFT",
    }
    if existing_id:
        product_input["id"] = existing_id

    option_name = row.get("option_name", "").strip()
    option_values = [v.strip() for v in row.get("option_values", "").split("|") if v.strip()]

    if option_name and option_values:
        product_input["productOptions"] = [
            {"name": option_name, "values": [{"name": v} for v in option_values]}
        ]
        product_input["variants"] = [
            {
                "optionValues": [{"optionName": option_name, "name": v}],
                "price": row["price"],
                "sku": f"{row['handle']}-{v.lower()}",
                "inventoryItem": {"tracked": True},
                "inventoryQuantities": [
                    {"locationId": location_id, "name": "available", "quantity": quantity}
                ],
            }
            for v in option_values
        ]
    else:
        product_input["variants"] = [
            {
                "price": row["price"],
                "sku": row["handle"],
                "inventoryItem": {"tracked": True},
                "inventoryQuantities": [
                    {"locationId": location_id, "name": "available", "quantity": quantity}
                ],
            }
        ]

    return product_input


def upsert_product(client: ShopifyClient, row: dict, location_id: str) -> dict:
    existing_id = get_product_id_by_handle(client, row["handle"])
    product_input = build_product_set_input(row, location_id, existing_id)

    data = client.execute(
        """
        mutation($input: ProductSetInput!) {
          productSet(input: $input, synchronous: true) {
            product {
              id
              handle
              variants(first: 50) {
                nodes { id inventoryItem { id } }
              }
            }
            userErrors { field message }
          }
        }
        """,
        {"input": product_input},
    )
    check_user_errors(data, "productSet")
    return data["productSet"]["product"]


def attach_to_collection(client: ShopifyClient, product_id: str, collection_id: str) -> None:
    data = client.execute(
        """
        mutation($id: ID!, $productIds: [ID!]!) {
          collectionAddProductsV2(id: $id, productIds: $productIds) {
            userErrors { field message }
          }
        }
        """,
        {"id": collection_id, "productIds": [product_id]},
    )
    check_user_errors(data, "collectionAddProductsV2")


def maybe_upload_design(client: ShopifyClient, product_id: str, design_file: str, alt: str) -> None:
    if not design_file:
        return
    path = ARTWORK_DIR / design_file
    if not path.exists():
        print(f"  no artwork at {path.relative_to(ROOT)} — skipping image upload")
        return

    staged = client.execute(
        """
        mutation($input: [StagedUploadInput!]!) {
          stagedUploadsCreate(input: $input) {
            stagedTargets { url resourceUrl parameters { name value } }
            userErrors { field message }
          }
        }
        """,
        {
            "input": [
                {
                    "filename": path.name,
                    "mimeType": "image/png",
                    "httpMethod": "POST",
                    "resource": "IMAGE",
                }
            ]
        },
    )
    check_user_errors(staged, "stagedUploadsCreate")
    target = staged["stagedUploadsCreate"]["stagedTargets"][0]

    with open(path, "rb") as f:
        upload_resp = requests.post(
            target["url"],
            data={p["name"]: p["value"] for p in target["parameters"]},
            files={"file": f},
            timeout=60,
        )
    upload_resp.raise_for_status()

    media = client.execute(
        """
        mutation($productId: ID!, $media: [CreateMediaInput!]!) {
          productCreateMedia(productId: $productId, media: $media) {
            mediaUserErrors { field message }
          }
        }
        """,
        {
            "productId": product_id,
            "media": [
                {
                    "originalSource": target["resourceUrl"],
                    "alt": alt,
                    "mediaContentType": "IMAGE",
                }
            ],
        },
    )
    check_user_errors(media, "productCreateMedia", errors_key="mediaUserErrors")


def sync_variant_inventory(client: ShopifyClient, inventory_item_id: str, location_id: str, quantity: int) -> None:
    data = client.execute(
        """
        mutation($input: InventorySetQuantitiesInput!) {
          inventorySetQuantities(input: $input) {
            userErrors { field message }
          }
        }
        """,
        {
            "input": {
                "name": "available",
                "reason": "correction",
                "ignoreCompareQuantity": True,
                "quantities": [
                    {
                        "inventoryItemId": inventory_item_id,
                        "locationId": location_id,
                        "quantity": quantity,
                    }
                ],
            }
        },
    )
    check_user_errors(data, "inventorySetQuantities")


def load_manifest(path: pathlib.Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=pathlib.Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifests = args.manifest or DEFAULT_MANIFESTS

    store_domain = os.environ.get("SHOPIFY_STORE_DOMAIN")
    access_token = os.environ.get("SHOPIFY_ADMIN_ACCESS_TOKEN")
    if not store_domain or not access_token:
        print("Set SHOPIFY_STORE_DOMAIN and SHOPIFY_ADMIN_ACCESS_TOKEN (see .env.example)", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[dry-run] would sync {sum(len(load_manifest(m)) for m in manifests)} products "
              f"to {store_domain} (API {API_VERSION})")
        for manifest in manifests:
            for row in load_manifest(manifest):
                print(f"  - {row['handle']} -> collection '{row['collection_handle']}'")
        return 0

    client = ShopifyClient(store_domain, access_token)
    location_id = get_default_location_id(client)
    print(f"Using location {location_id}")

    collection_cache: dict[str, str] = {}

    for manifest in manifests:
        print(f"\n== {manifest.relative_to(ROOT)} ==")
        for row in load_manifest(manifest):
            print(f"- {row['handle']}")
            product = upsert_product(client, row, location_id)

            collection_handle = row["collection_handle"]
            if collection_handle not in collection_cache:
                collection_cache[collection_handle] = get_or_create_collection_id(client, collection_handle)
            attach_to_collection(client, product["id"], collection_cache[collection_handle])

            maybe_upload_design(client, product["id"], row.get("design_file", ""), row["title"])

            quantity = int(row["initial_inventory"] or 0)
            for variant in product["variants"]["nodes"]:
                sync_variant_inventory(client, variant["inventoryItem"]["id"], location_id, quantity)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Vintage Bait — Printify product sync.

Creates or updates products on Printify from a JSON config
(scripts/printify_products.json) and publishes them to the connected
Shopify store. Images are uploaded from public URLs you supply in the config.

Workflow:
  1. Create a Printify account and connect your Shopify store.
  2. Add PRINTIFY_API_KEY and PRINTIFY_SHOP_ID to .env.
  3. Run with --discover to find blueprint/print-provider IDs:
       python scripts/printify_create_products.py --discover --type "t-shirt"
  4. Fill in scripts/printify_products.json with blueprint_id,
     print_provider_id, variant_ids, and image_url for each product.
  5. Run normally to create and publish:
       python scripts/printify_create_products.py
     Or dry-run to preview without creating:
       python scripts/printify_create_products.py --dry-run

Setup:
    pip install -r scripts/requirements.txt
    cp .env.example .env  # fill in PRINTIFY_API_KEY + PRINTIFY_SHOP_ID
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
PRODUCTS_CONFIG = ROOT / "scripts" / "printify_products.json"

PRINTIFY_API = "https://api.printify.com/v1"


class PrintifyClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "VintageBait/1.0",
        })

    def get(self, path: str) -> dict | list:
        resp = self.session.get(f"{PRINTIFY_API}{path}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{PRINTIFY_API}{path}", json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()


def get_shop_id(client: PrintifyClient) -> str:
    shops = client.get("/shops.json")
    if not shops:
        print("No Printify shops found. Connect your Shopify store in the Printify dashboard first.", file=sys.stderr)
        sys.exit(1)
    if len(shops) == 1:
        return str(shops[0]["id"])
    print("Multiple shops found:")
    for s in shops:
        print(f"  {s['id']}: {s['title']} ({s['sales_channel']})")
    print("Set PRINTIFY_SHOP_ID in .env to the ID you want to use.", file=sys.stderr)
    sys.exit(1)


def discover_blueprints(client: PrintifyClient, keyword: str) -> None:
    print(f"Searching blueprints for: {keyword!r}\n")
    blueprints = client.get("/catalog/blueprints.json")
    matches = [b for b in blueprints if keyword.lower() in b["title"].lower()]
    if not matches:
        print("No matches found. Try a different keyword (e.g. 'tee', 'hoodie', 'hat', 'sticker', 'tumbler').")
        return
    for b in matches[:10]:
        print(f"  blueprint_id={b['id']}  {b['title']}")
        providers = client.get(f"/catalog/blueprints/{b['id']}/print_providers.json")
        for p in providers[:3]:
            print(f"    print_provider_id={p['id']}  {p['title']} ({p['location']['country']})")
    print("\nRun with --variants --blueprint <id> --provider <id> to see variant IDs.")


def discover_variants(client: PrintifyClient, blueprint_id: int, provider_id: int) -> None:
    variants = client.get(f"/catalog/blueprints/{blueprint_id}/print_providers/{provider_id}/variants.json")
    print(f"Variants for blueprint {blueprint_id}, provider {provider_id}:\n")
    for v in variants.get("variants", []):
        opts = ", ".join(f"{k}={v2}" for k, v2 in v.get("options", {}).items())
        print(f"  variant_id={v['id']}  {v['title']}  ({opts})")


def upload_image(client: PrintifyClient, image_url: str, file_name: str) -> str:
    print(f"  Uploading image: {file_name}")
    result = client.post("/uploads/images.json", {
        "file_name": file_name,
        "url": image_url,
    })
    return result["id"]


def create_product(client: PrintifyClient, shop_id: str, cfg: dict, image_id: str) -> str:
    print(f"  Creating product: {cfg['title']}")

    variant_ids = cfg["variant_ids"]
    price_cents = int(float(cfg["price"]) * 100)

    payload = {
        "title": cfg["title"],
        "description": cfg["description"],
        "blueprint_id": cfg["blueprint_id"],
        "print_provider_id": cfg["print_provider_id"],
        "variants": [
            {"id": vid, "price": price_cents, "is_enabled": True}
            for vid in variant_ids
        ],
        "print_areas": [
            {
                "variant_ids": variant_ids,
                "placeholders": [
                    {
                        "position": cfg.get("print_position", "front"),
                        "images": [
                            {
                                "id": image_id,
                                "x": cfg.get("image_x", 0.5),
                                "y": cfg.get("image_y", 0.5),
                                "scale": cfg.get("image_scale", 1.0),
                                "angle": 0,
                            }
                        ],
                    }
                ],
            }
        ],
    }

    if cfg.get("tags"):
        payload["tags"] = cfg["tags"] if isinstance(cfg["tags"], list) else cfg["tags"].split(",")

    result = client.post(f"/shops/{shop_id}/products.json", payload)
    return result["id"]


def publish_product(client: PrintifyClient, shop_id: str, product_id: str, title: str) -> None:
    print(f"  Publishing to Shopify: {title}")
    client.post(f"/shops/{shop_id}/products/{product_id}/publish.json", {
        "title": True,
        "description": True,
        "images": True,
        "variants": True,
        "tags": True,
        "keyFeatures": True,
        "shipping_template": True,
    })


def load_config() -> list[dict]:
    if not PRODUCTS_CONFIG.exists():
        print(f"Config not found: {PRODUCTS_CONFIG}", file=sys.stderr)
        print("Create it from scripts/printify_products.example.json or run --discover first.", file=sys.stderr)
        sys.exit(1)
    with open(PRODUCTS_CONFIG, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--discover", action="store_true", help="List blueprints matching a keyword")
    parser.add_argument("--type", default="", help="Keyword for --discover search (e.g. 'hoodie')")
    parser.add_argument("--variants", action="store_true", help="Show variants for a blueprint+provider")
    parser.add_argument("--blueprint", type=int, help="Blueprint ID for --variants")
    parser.add_argument("--provider", type=int, help="Print provider ID for --variants")
    parser.add_argument("--dry-run", action="store_true", help="Preview products without creating them")
    args = parser.parse_args()

    api_key = os.environ.get("PRINTIFY_API_KEY")
    if not api_key:
        print("Set PRINTIFY_API_KEY in .env (Printify dashboard → My Profile → Connections → API)", file=sys.stderr)
        return 1

    client = PrintifyClient(api_key)

    if args.discover:
        discover_blueprints(client, args.type or "t-shirt")
        return 0

    if args.variants:
        if not args.blueprint or not args.provider:
            print("--variants requires --blueprint <id> and --provider <id>", file=sys.stderr)
            return 1
        discover_variants(client, args.blueprint, args.provider)
        return 0

    shop_id = os.environ.get("PRINTIFY_SHOP_ID") or get_shop_id(client)
    print(f"Using Printify shop: {shop_id}\n")

    products = load_config()

    if args.dry_run:
        print(f"[dry-run] Would create {len(products)} products in shop {shop_id}:")
        for p in products:
            print(f"  - {p['title']} | blueprint={p['blueprint_id']} provider={p['print_provider_id']}")
            print(f"    image: {p.get('image_url', '(missing)')}")
        return 0

    for p in products:
        print(f"\n— {p['title']}")
        if not p.get("image_url"):
            print("  SKIP: no image_url set in config")
            continue
        image_id = upload_image(client, p["image_url"], f"{p['handle']}.png")
        product_id = create_product(client, shop_id, p, image_id)
        publish_product(client, shop_id, product_id, p["title"])
        print(f"  Done. Printify product ID: {product_id}")

    print("\nAll products created and published to Shopify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Vintage Bait — interactive Printify image mapper.

Run this ONCE on your local machine to:
  1. Download all images from Google Drive (opens in your browser for auth if needed)
  2. Show each image so you can assign it to a product
  3. Write the completed printify_products.json ready for printify_create_products.py

Usage:
    pip install -r scripts/requirements.txt
    python scripts/setup_printify_images.py

Requirements: you must be signed into Google in your default browser, OR
the Drive files must be set to "Anyone with the link can view" (public).
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import subprocess
import sys
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "scripts" / "printify_products.json"

DRIVE_FILE_IDS = [
    "17HffqwRJO3ylZ3YucE7L1jZVphPVH-Et",
    "18BvvV8QvsXaXMPATkTZZWEUQN2Hv_wD7",
    "1B8qDOO5zp94324GMk-AyJ36geitxPVpE",
    "1Itqs66uRzhn7ehBJ0iJ2GDSLq6uQYnA2",
    "1KBofuEbKZC6wUUhfydtu0cDasJ9D-6qu",
    "1QlYhtknB4SVzZXHipoTB_hCQuS3Yi-wQ",
    "1RSVslzOjoLY4mMVC1tXcojZQfXkeM4b8",
    "1Xm75fG4qvNc-hDxNRNSY0VVF_BXvZDbK",
    "1ZzE_o7LLU8949avFPp_G8VWdbgpEzVK1",
    "1agMNQrT58aDGESbgxRU3O5x-XtudhuNi",
    "1jSXsCT9pV-RvVi1JwuYiYMKg2puCspo4",
    "1kS5TEXtu3aC8JCW4vcJbqgPFsqYaxTBY",
    "1l4aQ6BMRCxz68fRDviF-PNcmWezYcRaH",
    "1lAjrqf8-RHE97vLQnl1_XsDXiC3a_ZIQ",
    "1pyGHooswPgy2pTGqMb8Ip-Bzm15Fhtgs",
    "1siv7w8GlBVC8EtFskPaX4Btb9Bp96cXJ",
    "1tBW4mSMqUVm5T8wKxJH-Qu0JIw4yp-ze",
    "1wmS5M8j5kkrUho__cmC3Ev5BRqaecX46",
]

PRODUCTS = [
    {"handle": "faded-tarpon-tee",               "label": "Faded Tarpon Tee (T-Shirt)"},
    {"handle": "sun-bleached-hoodie",             "label": "Sun-Bleached Hoodie"},
    {"handle": "weathered-trucker-cap",           "label": "Weathered Trucker Cap"},
    {"handle": "camp-crewneck",                   "label": "Camp Crewneck"},
    {"handle": "hook-and-line-sticker-pack",      "label": "Hook & Line Sticker Pack"},
    {"handle": "tackle-box-koozie",               "label": "Tackle Box Koozie"},
    {"handle": "old-guide-trucker-hat",           "label": "Old Guide Trucker Hat"},
    {"handle": "the-one-that-got-away-tumbler",   "label": "The One That Got Away Tumbler"},
    {"handle": "_skip",                           "label": "SKIP (not a product image)"},
]

def direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def open_image(url: str) -> None:
    system = platform.system()
    print(f"\n  Opening: {url}")
    try:
        if system == "Darwin":
            subprocess.run(["open", url], check=False)
        elif system == "Linux":
            subprocess.run(["xdg-open", url], check=False)
        else:
            import webbrowser
            webbrowser.open(url)
    except Exception as e:
        print(f"  (Couldn't auto-open: {e})")

def pick_product(image_num: int, total: int, url: str) -> str | None:
    print(f"\n{'─'*60}")
    print(f"Image {image_num}/{total}")
    print(f"URL: {url}")
    print()
    for i, p in enumerate(PRODUCTS):
        print(f"  {i+1}. {p['label']}")
    print()
    open_image(url)
    while True:
        raw = input("Which product? (enter number, or 's' to skip): ").strip()
        if raw.lower() == "s" or raw == str(len(PRODUCTS)):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(PRODUCTS) - 1:
            return PRODUCTS[int(raw) - 1]["handle"]
        print("  Enter a number from the list above.")

def main() -> int:
    print("Vintage Bait — Printify image mapper")
    print("=====================================")
    print(f"Mapping {len(DRIVE_FILE_IDS)} images to {len(PRODUCTS)-1} products.\n")
    print("For each image, it will open in your browser.")
    print("Type the number of the product it belongs to, or 's' to skip.\n")
    input("Press Enter to start...")

    # Load existing config
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config: list[dict] = json.load(f)

    # Build handle → config index map
    config_by_handle = {p["handle"]: p for p in config}

    # Accumulate assignments: handle → list of URLs
    assignments: dict[str, list[str]] = {p["handle"]: [] for p in config}

    total = len(DRIVE_FILE_IDS)
    for i, fid in enumerate(DRIVE_FILE_IDS, 1):
        url = direct_url(fid)
        handle = pick_product(i, total, url)
        if handle:
            assignments[handle].append(url)
            print(f"  ✓ Assigned to: {config_by_handle[handle]['label'] if handle in config_by_handle else handle}")
        else:
            print("  — Skipped")

    # Write first assigned URL as image_url for each product
    updated = 0
    for product in config:
        h = product["handle"]
        urls = assignments.get(h, [])
        if urls:
            product["image_url"] = urls[0]
            product["additional_image_urls"] = urls[1:] if len(urls) > 1 else []
            updated += 1
            print(f"  {h}: {urls[0][:60]}...")

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Updated {updated} products in {CONFIG_PATH.relative_to(ROOT)}")
    print("\nNext steps:")
    print("  1. Make sure PRINTIFY_API_KEY and PRINTIFY_SHOP_ID are in .env")
    print("  2. Run: python scripts/printify_create_products.py --dry-run")
    print("  3. If that looks good: python scripts/printify_create_products.py")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

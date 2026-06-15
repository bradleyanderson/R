# CLAUDE.md — Vintage Bait Shopify Theme

## Project overview

This is the Shopify Online Store 2.0 theme and store-management tooling for
**Vintage Bait**, a merchandise brand selling vintage-style casual apparel
and fishing-themed merch. The theme is a customization layer on top of
Shopify's **Dawn** reference theme.

**Brand voice:** worn-in, faded earth tones, a little weathered, a little
nostalgic — think an old tackle shop sign, not a slick outdoor-gear startup.
Copy can be warm and a little tongue-in-cheek ("built to look broken-in from
day one") but product info must stay accurate.

## Where things live

- `assets/vintage-bait.css` — the design system (palette, typography,
  components). Faded earth-tone palette: cream/parchment background,
  faded olive primary, rust accent, tobacco-brown secondary, charcoal text.
- `sections/`, `snippets/` — custom Liquid sections/snippets, all prefixed
  `vintage-` / `vb-` so they don't collide with Dawn's own files.
- `templates/` — JSON templates wiring the custom sections into pages.
- `layout/THEME_LIQUID_ADDITIONS.md` — the (small) edits needed in Dawn's
  `layout/theme.liquid` to load the stylesheet, fonts, and grain overlay.
- `product-designs/` — the product catalog manifest (CSV) and artwork. This
  is the source of truth for what products should exist.
- `scripts/bulk_create_products.py` — syncs the manifest to Shopify
  (products, collections, inventory, images) via the Admin GraphQL API.
- `.claude/skills/` — Claude Code skills for recurring store operations.

## Store configuration

Credentials live in `.env` (copy from `.env.example`), **never** hardcoded
or committed:

- `SHOPIFY_STORE_DOMAIN`, `SHOPIFY_ADMIN_ACCESS_TOKEN` — Admin API (catalog/
  inventory scripts).
- `SHOPIFY_CLI_THEME_TOKEN`, `STAGING_THEME_ID`, `PRODUCTION_THEME_ID` —
  theme deploys (CI secrets, see `.github/workflows/`).

## Live store state

- Store: `ujhza4-fz.myshopify.com` ("Vintage Bait", Basic plan).
- **The live/published theme is Shopify's "Horizon" theme (role `MAIN`), not Dawn.**
  `layout/THEME_LIQUID_ADDITIONS.md` and `config/settings_data.vintage-schemes.json`
  in this repo describe edits written against **Dawn 11+** — Horizon uses a different
  section architecture (generic `hero`/`product-list`/`main-collection` sections with
  `color_palette` instead of Dawn's `color_schemes`/`color_scheme` settings, plus
  block-based settings like `padding-block-start`). Treat those two files as a
  reference for *what* needs wiring (stylesheet, fonts, grain overlay, color
  palette), not as drop-in edits for the current live theme.
- An **UNPUBLISHED "Vintage Bait" theme** (duplicated from Horizon) already has the
  Vintage Bait overlay applied via the Admin API:
  - `assets/vintage-bait.css`, `sections/vintage-hero.liquid`,
    `sections/vintage-collection-grid.liquid`, `sections/vintage-badge.liquid`,
    `snippets/grain-overlay.liquid`, `snippets/vintage-badge-stamp.liquid`,
    `snippets/icon-fishhook.liquid` — pushed as-is (additive, `vb-*`/`vintage-*`
    namespaced).
  - `layout/theme.liquid` — added the stylesheet/Google Fonts `<link>`s in `<head>`
    and `{% render 'grain-overlay' %}` right after `<body ...>`.
  - `config/settings_data.json` — merged `scheme-1` / `scheme-2` / `scheme-3-tobacco`
    into `current.color_schemes` (additive; Horizon had no `color_schemes` key).
  - `templates/index.json` — replaced Horizon's default `hero`/`product-list` home
    layout with `vintage_hero` → `vintage_collection_grid` → a repurposed
    `product-list` section (Horizon's `_product-card`/`_product-list-*` blocks,
    pointed at the `vintage-apparel` collection) → `vintage_seal`.
  - `templates/collection.vintage-apparel.json` / `templates/collection.fishing-merch.json`
    were **not** applied — they reference Dawn-only section types
    (`main-collection-banner`, `main-collection-product-grid`) that don't exist in
    Horizon. Horizon's existing dynamic `templates/collection.json` (reads
    `closest.collection.title`/`description`) already covers both collections.
  - Preview at `https://ujhza4-fz.myshopify.com/?preview_theme_id=188419965218`
    before publishing. Don't publish without reviewing the preview first.
- Catalog: **Vintage Apparel** and **Fishing Merch** collections exist, each with the
  products from `product-designs/manifest/*.csv` created as `ACTIVE` (no artwork
  uploaded yet — no image-generation tool has been connected to this project).

## Branch strategy

- `main` — integration branch. All theme/catalog changes land here first.
- `staging` — deploys to the staging theme (`STAGING_THEME_ID`) on every
  push. Use this to preview changes on the real storefront before going live.
- `production` — deploys to the live theme (`PRODUCTION_THEME_ID`). Only
  promote from `staging` after it's been reviewed there.

## Do's and don'ts

- **Do** make catalog changes (new products, price/inventory updates) by
  editing `product-designs/manifest/*.csv` and re-running
  `bulk_create_products.py` — not by hand in Shopify Admin — so the manifest
  stays the source of truth.
- **Do** preview theme changes on the `staging` theme before promoting to
  `production`.
- **Don't** run `shopify theme push --live` (or push directly to
  `production`) without the change having been verified on staging first.
- **Don't** commit `.env`, access tokens, or theme tokens.
- **Don't** hand-edit Dawn's core files when a Vintage Bait override/section
  will do — keep customizations additive (`vb-*` / `vintage-*` namespaced)
  so upstream Dawn updates stay low-conflict.
- **Don't** delete or unpublish products/collections without confirming with
  the user first — these actions are visible to customers.

## Local development

```
shopify theme dev --store=<your-store>.myshopify.com
```

## Useful references

- `product-designs/README.md` — manifest format and artwork conventions.
- `layout/THEME_LIQUID_ADDITIONS.md` — exact theme.liquid edits required.
- `config/settings_data.vintage-schemes.json` — color scheme definitions to
  merge into `config/settings_data.json`.

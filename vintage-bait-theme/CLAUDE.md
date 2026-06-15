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

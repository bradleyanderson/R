# Vintage Bait — Shopify Theme & Store Tooling

A Shopify Online Store 2.0 theme and catalog-management toolkit for
**Vintage Bait** — vintage-style casual apparel and fishing-themed
merchandise. Faded earth tones, worn-in textures, two flagship collections:
**Vintage Apparel** and **Fishing Merch**.

> **Repo note**: this project currently lives at
> `vintage-bait-theme/` inside the `bradleyanderson/R` repo (an unrelated
> AI-platform repo) because the GitHub integration in this session isn't
> scoped to create a new repository. It's structured as a self-contained
> project so it can be split out cleanly:
>
> ```bash
> git subtree split -P vintage-bait-theme -b vintage-bait-theme-export
> # then push that branch to a new, empty GitHub repo as its `main`
> ```

## What's here

This is an **overlay** on top of Shopify's free [Dawn](https://github.com/Shopify/dawn)
reference theme — it doesn't vendor Dawn itself (avoids drift from upstream
updates). It adds:

- `assets/vintage-bait.css` — the Vintage Bait design system: faded
  earth-tone palette, vintage typography, grain/texture effects, worn-in
  buttons and cards — all CSS, no raster assets required.
- `sections/vintage-hero.liquid`, `vintage-collection-grid.liquid`,
  `vintage-badge.liquid` — custom Online Store 2.0 sections.
- `snippets/grain-overlay.liquid`, `vintage-badge-stamp.liquid`,
  `icon-fishhook.liquid` — reusable components.
- `templates/index.json`, `collection.vintage-apparel.json`,
  `collection.fishing-merch.json` — page templates wiring it together.
- `config/settings_data.vintage-schemes.json` — color scheme definitions.
- `product-designs/` — catalog manifest + artwork conventions.
- `scripts/bulk_create_products.py` — syncs the manifest to Shopify
  (products, collections, inventory, images).
- `.claude/skills/` — Claude Code skills for ongoing store operations.
- `.github/workflows/` — staging/production theme deploy pipelines.

## Design system

| Token            | Hex       | Use                                  |
| ---------------- | --------- | ------------------------------------- |
| `--vb-cream`      | `#F1E6D2` | Page background (parchment)          |
| `--vb-cream-dark` | `#E3D5B8` | Card / alternating section background |
| `--vb-olive`      | `#6B6B47` | Primary accent (faded olive)          |
| `--vb-rust`       | `#B5552B` | CTAs / buttons (sun-faded rust)       |
| `--vb-tobacco`    | `#6E4A2E` | Borders, footer (tobacco brown)       |
| `--vb-charcoal`   | `#2E2A24` | Text                                  |
| `--vb-mustard`    | `#C9A227` | Highlights / stamps                   |

Typography: **Oswald** (display/headings, uppercase, wide letter-spacing) +
**Inter** (body). "Worn-in" texture comes from an SVG grain overlay
(`snippets/grain-overlay.liquid`), distressed borders/shadows, and a sepia
filter on product imagery — see `assets/vintage-bait.css` for details.

No logo, hero photography, or apparel print artwork exists yet — these need
to be supplied or generated separately (no image-generation tool was
connected when this was built). `product-designs/artwork/` has placeholder
folders and a naming convention ready for when artwork is available.

## Setup

1. **Get a Dawn checkout**:
   ```bash
   git clone https://github.com/Shopify/dawn.git
   ```
   Note the Dawn version/tag you start from.
2. **Copy this project's files into the Dawn checkout**, preserving paths
   (`assets/`, `sections/`, `snippets/`, `templates/`, `config/`).
3. **Apply the small `theme.liquid` edits** in
   `layout/THEME_LIQUID_ADDITIONS.md` (load the stylesheet/fonts, render the
   grain overlay).
4. **Merge color schemes**: add the entries from
   `config/settings_data.vintage-schemes.json` into your Dawn checkout's
   `config/settings_data.json` (`color_schemes` object).
5. **Preview locally**:
   ```bash
   shopify theme dev --store=<your-store>.myshopify.com
   ```
6. **Catalog setup**: `cp .env.example .env`, fill in Admin API credentials,
   then `python scripts/bulk_create_products.py --dry-run` followed by the
   real run. See `product-designs/README.md`.

## Branch strategy

`main` (integration) → `staging` (preview theme) → `production` (live
theme). See `CLAUDE.md` for the full do's/don'ts and `.github/workflows/`
for the deploy pipelines (require repo secrets — see workflow file
comments).

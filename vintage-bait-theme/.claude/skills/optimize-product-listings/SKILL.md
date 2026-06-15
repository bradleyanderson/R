---
name: optimize-product-listings
description: Review and improve Vintage Bait product titles, descriptions, SEO fields, and tags for consistency and discoverability.
---

# Optimize product listings

Improves existing product listings — copy, SEO metadata, and tags — without
changing prices, inventory, or images. Keep the Vintage Bait brand voice:
worn-in, faded earth tones, a little nostalgic, but accurate about
materials/fit.

## Workflow

1. Pull current listings via the Shopify Admin MCP's `search_products` /
   `get-product` (or `search_products` scoped to a collection).
2. For each product, review:
   - **Title**: clear, on-brand, includes the product type (e.g. "Faded
     Tarpon Tee" not just "Tarpon Tee Shirt Item").
   - **Description**: rewrite for brand voice and readability if it's
     generic or thin. Keep factual claims (materials, fit, sizing) accurate
     — don't invent details not present in the manifest or current listing.
   - **SEO title/description**: set via `update-product` or
     `graphql_mutation` (`seo: { title, description }` on `ProductInput`) —
     concise, includes product type + key descriptors.
   - **Tags**: ensure consistent tagging per `product-designs/README.md`
     conventions (collection tags like `vintage`/`apparel` or
     `fishing`/`outdoors`/`merch`, plus product-type tags). Add missing tags,
     don't remove existing ones without checking they're not used by a
     collection's automated rules.
3. Apply changes via `update-product` (Shopify Admin MCP) or
   `graphql_mutation` for fields without a dedicated tool (e.g. `seo`,
   metafields).
4. If a description is rewritten, also update the corresponding
   `description` cell in `product-designs/manifest/*.csv` so the manifest
   stays in sync with what's live.
5. Summarize what changed per product (title/description/SEO/tags) so the
   user can review.

## Guardrails

- Don't change price, inventory, images, or published status — that's
  `manage-product-catalog` / `sync-inventory`'s job.
- Don't fabricate product specs (materials, origin, sizing) — if unsure,
  flag it for the user rather than guessing.

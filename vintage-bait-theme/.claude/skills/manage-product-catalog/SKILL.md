---
name: manage-product-catalog
description: Add, update, or remove Vintage Bait products by editing the catalog manifest and syncing it to Shopify.
---

# Manage product catalog

The product catalog manifest (`product-designs/manifest/*.csv`) is the
source of truth for what products exist. This skill wraps
`scripts/bulk_create_products.py`, which syncs the manifest to Shopify
(products, collection membership, images, inventory) via the Admin
GraphQL API.

## Workflow

1. **Adding/updating a product**: edit the relevant CSV
   (`vintage-apparel.csv` or `fishing-merch.csv`) — add a new row or edit an
   existing one. Required columns are documented in
   `product-designs/README.md`. Keep `handle` stable; it's the idempotency
   key.
2. If the product has artwork, place it under
   `product-designs/artwork/<collection>/` using the naming convention in
   `product-designs/README.md`, and reference it in the `design_file`
   column. If artwork doesn't exist yet, leave the column pointing at the
   expected filename — the sync skips image upload until the file exists.
3. **Dry run first**:
   ```
   python scripts/bulk_create_products.py --dry-run
   ```
   Review the list of products that will be created/updated.
4. **Apply**:
   ```
   python scripts/bulk_create_products.py
   ```
   New products are created as **drafts** (`status: DRAFT`) — review them
   in Shopify Admin and publish manually when ready. This skill should not
   change a product's published status without the user asking explicitly.
5. Report what changed: which handles were created vs. updated, and any
   `userErrors` returned by Shopify.

## Removing a product

Removing a row from the manifest does **not** delete the product in
Shopify — the script only creates/updates. To remove a product, ask the
user to confirm, then use the Shopify Admin MCP tools
(`bulk-update-product-status` to archive, or a `graphql_mutation` for
`productDelete`) directly. Deleting/archiving products is visible to
customers and should always be confirmed first.

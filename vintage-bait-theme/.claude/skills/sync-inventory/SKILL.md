---
name: sync-inventory
description: Compare actual Shopify inventory levels against the catalog manifest and reconcile differences, with confirmation before applying changes.
---

# Sync inventory

Reconciles live Shopify inventory ("available" quantity at the default
location) against the `initial_inventory` values in
`product-designs/manifest/*.csv`. Use this for periodic stock reconciliation
— e.g. after a manual count, or after the manifest's expected quantities
change.

## Workflow

1. For each product in the manifest, look up current inventory via the
   Shopify Admin MCP's `get-inventory-levels` (or
   `scripts/bulk_create_products.py`'s GraphQL helpers for a scripted pass).
2. Build a diff table: handle, variant, manifest quantity vs. live quantity.
3. Present the diff to the user. **Do not apply changes automatically** —
   inventory corrections affect what customers can buy and should be
   reviewed.
4. On confirmation, apply corrections via `set-inventory` (Shopify Admin
   MCP) or `inventorySetQuantities` (as used in
   `scripts/bulk_create_products.py`).
5. If the live quantity should become the new "expected" baseline instead
   (e.g. a real sale happened, not a discrepancy), update
   `initial_inventory` in the manifest to match instead of pushing the old
   manifest value back to Shopify.

## Notes

- This skill is read-heavy by design — most runs should end with "here's
  the diff, what would you like to do?" rather than a write.
- For bulk re-creation/sync of products (not just inventory), use the
  `manage-product-catalog` skill instead.

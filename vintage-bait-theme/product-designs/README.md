# Product designs & catalog manifest

This folder is the source of truth for the Vintage Bait product catalog. The
`manifest/` CSVs describe every product; `scripts/bulk_create_products.py`
(at the project root) reads them and creates/updates products, collection
membership, and inventory in Shopify via the Admin GraphQL API.

## Artwork naming convention

Place print/design artwork in `artwork/<collection-slug>/`, named:

```
<collection-slug>__<product-handle>__<variant>.png
```

Examples:

```
artwork/apparel/vintage-apparel__faded-tarpon-tee__front.png
artwork/fishing/fishing-merch__hook-and-line-sticker-pack__default.png
```

No artwork exists yet for the initial catalog (no image-generation tool is
connected) — the `design_file` column in the manifests references the
expected filenames so artwork can be dropped in later without changing the
manifest. Until a file exists at that path, `bulk_create_products.py` skips
image upload for that product (the product is still created/updated).

## Manifest columns (`manifest/*.csv`)

| Column              | Description                                                                 |
| ------------------- | ---------------------------------------------------------------------------- |
| `handle`            | Unique product handle (URL slug). Used as the idempotency key.              |
| `title`             | Product title.                                                               |
| `collection_handle` | Handle of the collection this product belongs to.                           |
| `product_type`      | Shopify product type (e.g. `T-Shirt`, `Sticker`).                            |
| `tags`              | Comma-separated tags (quote the whole field since it contains commas).      |
| `price`             | Base price, e.g. `28.00`.                                                    |
| `description`       | Plain-text product description (rewritten into HTML by the script).         |
| `design_file`       | Expected artwork filename under `artwork/` (relative path). May not exist yet. |
| `option_name`       | Variant option name, e.g. `Size`. Leave blank for single-variant products.  |
| `option_values`     | Pipe-separated option values, e.g. `S\|M\|L\|XL`. Leave blank if no options. |
| `initial_inventory` | Starting "available" quantity per variant at the default location.          |

## Workflow

1. Edit/add rows in `manifest/vintage-apparel.csv` or `manifest/fishing-merch.csv`.
2. Run `python scripts/bulk_create_products.py` (reads `.env` for store
   credentials — see `.env.example`).
3. The script is idempotent: re-running it updates existing products
   (matched by `handle`) rather than duplicating them.

See the `manage-product-catalog` and `sync-inventory` skills in
`.claude/skills/` for the Claude Code workflows that wrap this script.

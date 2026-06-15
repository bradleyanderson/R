---
name: deploy-theme
description: Push the current branch's theme code to the matching Shopify theme (staging or production), with confirmation before any production deploy.
---

# Deploy theme

Pushes this repo's theme files to the corresponding Shopify theme using the
Shopify CLI. Branch determines target:

- `staging` branch -> `$STAGING_THEME_ID`
- `production` branch -> `$PRODUCTION_THEME_ID`
- any other branch -> ask the user which theme to preview against, or use
  `shopify theme dev` for an ephemeral local preview instead of pushing.

## Workflow

1. Confirm the current git branch and that the working tree is clean
   (`git status`). If there are uncommitted changes, ask whether to commit
   first.
2. Confirm `SHOPIFY_CLI_THEME_TOKEN`, `SHOPIFY_STORE_DOMAIN`, and the
   relevant theme ID env var are set (see `.env.example`).
3. **Staging**: run
   ```
   shopify theme push --store=$SHOPIFY_STORE_DOMAIN --theme=$STAGING_THEME_ID
   ```
   Report the preview URL Shopify CLI prints.
4. **Production**: this is a live-storefront change.
   - First verify the same changes are already deployed and look correct on
     staging.
   - Summarize what's changing (files touched, e.g. `git diff
     staging...production --stat`).
   - Ask the user to explicitly confirm before running:
     ```
     shopify theme push --store=$SHOPIFY_STORE_DOMAIN --theme=$PRODUCTION_THEME_ID --allow-live
     ```
   - Never run this automatically as part of a larger task — production
     deploys are a discrete, confirmed step.

## Notes

- This skill assumes the repo has been merged into a full Dawn checkout per
  `README.md` / `layout/THEME_LIQUID_ADDITIONS.md` — `shopify theme push`
  pushes the whole theme directory, not just the Vintage Bait overlay files.
- For automated deploys, see `.github/workflows/deploy-staging.yml` and
  `deploy-production.yml` — this skill is for ad-hoc/manual pushes.

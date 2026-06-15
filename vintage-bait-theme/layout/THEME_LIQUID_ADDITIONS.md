# Additions to `layout/theme.liquid`

This project doesn't vendor Dawn's `layout/theme.liquid` (avoid drift from
upstream). After copying this project's files into your Dawn checkout, make
these two small additions to `layout/theme.liquid`.

## 1. Load the design system stylesheet and fonts

In the `<head>`, alongside Dawn's existing `base.css` tag, add:

```liquid
{{ 'vintage-bait.css' | asset_url | stylesheet_tag }}

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link
  href="https://fonts.googleapis.com/css2?family=Oswald:wght@400;500;700&family=Inter:wght@400;500;600&display=swap"
  rel="stylesheet"
>
```

## 2. Render the grain overlay

Immediately after the opening `<body ...>` tag, add:

```liquid
{% render 'grain-overlay' %}
```

That's it — everything else (palette, typography, components) is scoped to
`vb-*` classes in `assets/vintage-bait.css` and the custom sections/snippets
in this project, so it won't conflict with Dawn's own styles.

## 3. Color schemes

Merge `config/settings_data.vintage-schemes.json` into the `color_schemes`
object of your `config/settings_data.json` (see that file's `_comment` for
details). `templates/index.json` and the vintage sections reference
`scheme-1` / `scheme-2` / `scheme-3-tobacco` by id.

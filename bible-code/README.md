# Bible Code Explorer

A static, client-side web app for exploring **Equidistant Letter Sequences
(ELS)** — the so-called "Bible Code" — in the Hebrew text of the Torah
(Genesis through Deuteronomy, Westminster Leningrad Codex, public domain).

**Live app:** https://bradleyanderson.github.io/R/

## Features

- Search for one or more Hebrew words across a configurable skip range
- Built-in on-screen Hebrew keyboard (no Hebrew input method required)
- Quick-pick example words
- Results sorted by smallest skip (the most statistically notable), with
  verse references
- Letter-grid visualizer (right-to-left, top-to-bottom like Hebrew text)
  showing each match's ELS line, with overlapping "crossing" matches
  highlighted together
- Restrict the search to the whole Torah or a single book

## Running locally

This is a plain static site — no build step required.

```sh
cd bible-code
npx http-server -p 8080
# open http://localhost:8080
```

## Data

`data/torah.json` contains:

- `text`: the full consonantal text of Genesis–Deuteronomy as one
  continuous string (305,172 letters), with all niqqud, cantillation marks,
  and morphological separators stripped
- `verses`: an index mapping each verse to its start offset and length in
  `text`, used to show book/chapter/verse references
- `books`: start offset and length of each of the five books within `text`

Source: [openscriptures/morphhb](https://github.com/openscriptures/morphhb)
(Westminster Leningrad Codex, public domain).

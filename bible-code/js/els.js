// Equidistant Letter Sequence (ELS) search engine for the Torah text.
// All positions are 0-based indices into a single continuous string of
// Hebrew consonants spanning Genesis through Deuteronomy.

const TorahData = {
  text: '',
  verses: [],
  books: [],
  loaded: false,
};

async function loadTorahData() {
  if (TorahData.loaded) return TorahData;
  const res = await fetch('data/torah.json');
  const data = await res.json();
  TorahData.text = data.text;
  TorahData.verses = data.verses;
  TorahData.books = data.books;
  TorahData.loaded = true;
  return TorahData;
}

// Binary search: returns the verse record containing text index `idx`.
function findVerse(idx) {
  const verses = TorahData.verses;
  let lo = 0, hi = verses.length - 1, ans = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (verses[mid].s <= idx) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return verses[ans];
}

function refForIndex(idx) {
  const v = findVerse(idx);
  if (!v) return '';
  return `${v.b} ${v.c}:${v.v}`;
}

// Search for `term` (a string of Hebrew consonants, length >= 2) across
// skip distances from minSkip to maxSkip (inclusive, both positive).
// Also searches the reversed term, reported as a negative skip, which
// covers the "reading backwards" direction.
//
// `range` optionally restricts every letter of a match to fall within
// [range.start, range.start + range.length).
//
// Returns an array of { term, skip, start, indices: number[] }.
function searchELS(term, minSkip, maxSkip, range, limitPerSkipSign) {
  const text = TorahData.text;
  const N = text.length;
  const L = term.length;
  if (L < 2) return [];

  const first = term[0];
  const results = [];

  // Anchor on every occurrence of the term's first letter, then check the
  // remaining letters at the given skip distance. This avoids building any
  // O(N) intermediate strings per skip value.
  const candidates = [];
  for (let i = 0; i < N; i++) {
    if (text[i] === first) candidates.push(i);
  }

  const inRange = (idx) => {
    if (!range) return true;
    return idx >= range.start && idx < range.start + range.length;
  };

  for (let g = minSkip; g <= maxSkip; g++) {
    let posCount = 0;
    let negCount = 0;
    for (let ci = 0; ci < candidates.length; ci++) {
      const i = candidates[ci];

      // Positive skip: i, i+g, i+2g, ...
      if (i + (L - 1) * g < N) {
        let ok = true;
        for (let k = 1; k < L; k++) {
          if (text[i + k * g] !== term[k]) { ok = false; break; }
        }
        if (ok) {
          const indices = new Array(L);
          let okRange = true;
          for (let k = 0; k < L; k++) {
            indices[k] = i + k * g;
            if (!inRange(indices[k])) { okRange = false; break; }
          }
          if (okRange) {
            results.push({ term, skip: g, start: i, indices });
            posCount++;
          }
        }
      }

      // Negative skip: i, i-g, i-2g, ... (reverse reading direction)
      if (g > 0 && i - (L - 1) * g >= 0) {
        let ok = true;
        for (let k = 1; k < L; k++) {
          if (text[i - k * g] !== term[k]) { ok = false; break; }
        }
        if (ok) {
          const indices = new Array(L);
          let okRange = true;
          for (let k = 0; k < L; k++) {
            indices[k] = i - k * g;
            if (!inRange(indices[k])) { okRange = false; break; }
          }
          if (okRange) {
            results.push({ term, skip: -g, start: i, indices });
            negCount++;
          }
        }
      }

      if (limitPerSkipSign && posCount >= limitPerSkipSign && negCount >= limitPerSkipSign) {
        break;
      }
    }
  }

  return results;
}

// Build a rows x cols matrix of {idx, char} centered on `centerIdx`.
function buildGrid(centerIdx, rows, cols) {
  const text = TorahData.text;
  const halfRows = Math.floor(rows / 2);
  const halfCols = Math.floor(cols / 2);
  const topLeft = centerIdx - halfRows * cols - halfCols;
  const grid = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const idx = topLeft + r * cols + c;
      row.push({ idx, char: (idx >= 0 && idx < text.length) ? text[idx] : '' });
    }
    grid.push(row);
  }
  return grid;
}

// The center index of a match: the middle letter along its ELS line.
function centerOfMatch(match) {
  const mid = Math.floor((match.indices.length - 1) / 2);
  return match.indices[mid];
}

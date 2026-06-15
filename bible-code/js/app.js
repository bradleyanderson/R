const COLORS = ['#c9a227', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#e67e22', '#16a085', '#d35400'];
const MAX_TERMS = 30;

function colorForIndex(i) {
  if (i < COLORS.length) return COLORS[i];
  const hue = (i * 47) % 360;
  return `hsl(${hue}, 60%, 42%)`;
}

const HEB_LETTERS = [
  'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל',
  'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
];
const HEB_FINALS = ['ך', 'ם', 'ן', 'ף', 'ץ'];

const PRESETS = [
  { label: 'תורה (Torah)', value: 'תורה' },
  { label: 'משה (Moses)', value: 'משה' },
  { label: 'דוד (David)', value: 'דוד' },
  { label: 'ישראל (Israel)', value: 'ישראל' },
  { label: 'ירושלים (Jerusalem)', value: 'ירושלים' },
  { label: 'שבת (Sabbath)', value: 'שבת' },
  { label: 'אור (Light)', value: 'אור' },
  { label: 'אהבה (Love)', value: 'אהבה' },
];

const state = {
  allResults: [],   // [{ term, label, color, matches }]
  flatMatches: [],  // every match across all terms
  currentCenter: undefined,
};

let activeInput = null;

document.addEventListener('focusin', (e) => {
  if (e.target.classList && e.target.classList.contains('heb-input')) {
    activeInput = e.target;
  }
});

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function sanitizeHebrew(input) {
  const cleaned = input.value.replace(/[^א-ת]/g, '');
  if (cleaned !== input.value) input.value = cleaned;
}

function insertAtCursor(input, char) {
  const start = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? input.value.length;
  input.value = input.value.slice(0, start) + char + input.value.slice(end);
  const pos = start + char.length;
  input.setSelectionRange(pos, pos);
  input.focus();
  sanitizeHebrew(input);
}

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function rangeForBook(name) {
  const b = TorahData.books.find((b) => b.name === name);
  return b ? { start: b.start, length: b.length } : null;
}

// ---------------------------------------------------------------------
// Drawers
// ---------------------------------------------------------------------
const DRAWER_TOGGLES = {
  'theme-drawer': 'toggle-theme-btn',
  'results-drawer': 'toggle-results-btn',
  'search-drawer': 'toggle-search-btn',
};

function setDrawerOpen(id, open) {
  document.getElementById(id).classList.toggle('open', open);
  const btnId = DRAWER_TOGGLES[id];
  if (btnId) document.getElementById(btnId).classList.toggle('active', open);
}

function openDrawer(id) {
  setDrawerOpen(id, true);
}

function closeDrawer(id) {
  setDrawerOpen(id, false);
}

function toggleDrawer(id) {
  setDrawerOpen(id, !document.getElementById(id).classList.contains('open'));
}

function setupDrawers() {
  document.getElementById('toggle-theme-btn').addEventListener('click', () => toggleDrawer('theme-drawer'));
  document.getElementById('toggle-results-btn').addEventListener('click', () => toggleDrawer('results-drawer'));
  document.getElementById('toggle-search-btn').addEventListener('click', () => toggleDrawer('search-drawer'));
  document.getElementById('search-drawer-handle').addEventListener('click', () => toggleDrawer('search-drawer'));

  document.querySelectorAll('.drawer-close').forEach((btn) => {
    btn.addEventListener('click', () => closeDrawer(btn.dataset.close));
  });

  // Search drawer starts open so first-time users see the controls.
  openDrawer('search-drawer');
}

// Measure the top bar's real height (it wraps onto two lines on narrow
// screens) so the side drawers can start below it instead of covering it.
function setupLayout() {
  const topBar = document.getElementById('top-bar');
  const update = () => {
    document.documentElement.style.setProperty('--topbar-h', `${topBar.getBoundingClientRect().height}px`);
  };
  update();
  window.addEventListener('resize', update);
}

// ---------------------------------------------------------------------
// Theme customizer
// ---------------------------------------------------------------------
const THEME_VARS = {
  'theme-bg': '--parchment',
  'theme-panel': '--parchment-dark',
  'theme-ink': '--ink',
  'theme-accent': '--gold-dark',
};
const THEME_STORAGE_KEY = 'bibleCodeTheme';

function applyTheme(theme) {
  for (const [inputId, varName] of Object.entries(THEME_VARS)) {
    if (theme[inputId]) document.documentElement.style.setProperty(varName, theme[inputId]);
  }
}

function loadSavedTheme() {
  try {
    return JSON.parse(localStorage.getItem(THEME_STORAGE_KEY) || '{}');
  } catch (err) {
    return {};
  }
}

function setupTheme() {
  const defaults = {};
  for (const [inputId, varName] of Object.entries(THEME_VARS)) {
    defaults[inputId] = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  const saved = loadSavedTheme();
  const current = { ...defaults, ...saved };
  applyTheme(current);

  for (const inputId of Object.keys(THEME_VARS)) {
    const input = document.getElementById(inputId);
    input.value = current[inputId];
    input.addEventListener('input', () => {
      const theme = loadSavedTheme();
      theme[inputId] = input.value;
      localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify(theme));
      applyTheme({ [inputId]: input.value });
    });
  }

  document.getElementById('theme-reset').addEventListener('click', () => {
    localStorage.removeItem(THEME_STORAGE_KEY);
    for (const [inputId, varName] of Object.entries(THEME_VARS)) {
      document.documentElement.style.removeProperty(varName);
      document.getElementById(inputId).value = defaults[inputId];
    }
  });
}

// ---------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------
function setupKeyboard() {
  const board = document.getElementById('keyboard');
  const makeKey = (ch, extraClass) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'key' + (extraClass ? ' ' + extraClass : '');
    btn.textContent = ch;
    btn.addEventListener('click', () => {
      const target = activeInput || document.querySelector('.heb-input');
      if (target) insertAtCursor(target, ch);
    });
    return btn;
  };

  HEB_LETTERS.forEach((ch) => board.appendChild(makeKey(ch)));
  HEB_FINALS.forEach((ch) => board.appendChild(makeKey(ch, 'final')));

  const back = document.createElement('button');
  back.type = 'button';
  back.className = 'key wide';
  back.textContent = '⌫';
  back.addEventListener('click', () => {
    const target = activeInput || document.querySelector('.heb-input');
    if (!target) return;
    const start = target.selectionStart ?? target.value.length;
    const end = target.selectionEnd ?? target.value.length;
    if (start === end && start > 0) {
      target.value = target.value.slice(0, start - 1) + target.value.slice(end);
      target.setSelectionRange(start - 1, start - 1);
    } else {
      target.value = target.value.slice(0, start) + target.value.slice(end);
      target.setSelectionRange(start, start);
    }
    target.focus();
  });
  board.appendChild(back);

  const clear = document.createElement('button');
  clear.type = 'button';
  clear.className = 'key wide';
  clear.textContent = 'Clear';
  clear.addEventListener('click', () => {
    const target = activeInput || document.querySelector('.heb-input');
    if (!target) return;
    target.value = '';
    delete target.dataset.label;
    target.focus();
  });
  board.appendChild(clear);
}

// ---------------------------------------------------------------------
// Term rows
// ---------------------------------------------------------------------
function refreshTermRowColors() {
  const rows = document.querySelectorAll('.term-row');
  rows.forEach((row, i) => {
    const swatch = row.querySelector('.swatch');
    swatch.style.background = colorForIndex(i);
    const removeBtn = row.querySelector('.remove-term');
    removeBtn.disabled = rows.length <= 1;
  });
  const addBtn = document.getElementById('add-term-btn');
  addBtn.disabled = rows.length >= MAX_TERMS;
}

function addTermRow(prefillValue, label) {
  const rows = document.querySelectorAll('.term-row');
  if (rows.length >= MAX_TERMS) return null;

  const container = document.getElementById('term-rows');
  const row = document.createElement('div');
  row.className = 'term-row';

  const swatch = document.createElement('span');
  swatch.className = 'swatch';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'heb-input';
  input.dir = 'rtl';
  input.maxLength = 12;
  input.placeholder = 'הקלד מילה בעברית';
  input.setAttribute('aria-label', 'Hebrew search word');
  input.addEventListener('input', () => {
    sanitizeHebrew(input);
    delete input.dataset.label;
  });
  if (prefillValue) input.value = prefillValue;
  if (label) input.dataset.label = label;

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'remove-term';
  removeBtn.title = 'Remove this word';
  removeBtn.textContent = '✕';
  removeBtn.addEventListener('click', () => {
    row.remove();
    refreshTermRowColors();
  });

  row.appendChild(swatch);
  row.appendChild(input);
  row.appendChild(removeBtn);
  container.appendChild(row);
  refreshTermRowColors();
  if (!activeInput) activeInput = input;
  return input;
}

function setupTermRows() {
  addTermRow('תורה', 'Torah');
  document.getElementById('add-term-btn').addEventListener('click', () => addTermRow());
}

function setupPresets() {
  const container = document.getElementById('presets');
  PRESETS.forEach((p) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset';
    btn.textContent = p.label;
    btn.addEventListener('click', () => {
      const target = activeInput || document.querySelector('.heb-input');
      if (target) {
        target.value = p.value;
        const m = p.label.match(/\(([^)]+)\)/);
        target.dataset.label = m ? m[1] : p.label;
        target.focus();
      }
    });
    container.appendChild(btn);
  });
}

// ---------------------------------------------------------------------
// Translation
// ---------------------------------------------------------------------
const labelCache = new Map(); // Hebrew term -> label in another language

async function translateText(text, sl, tl) {
  const url = 'https://translate.googleapis.com/translate_a/single'
    + `?client=gtx&sl=${sl}&tl=${tl}&dt=t&q=` + encodeURIComponent(text);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return {
    text: (data[0] || []).map((seg) => seg[0]).join(''),
    detected: data[2] || null,
  };
}

// Best-effort label for a Hebrew search term, used to tag its line in the
// grid. Falls back to the Hebrew term itself if translation is unavailable.
async function labelForTerm(term) {
  if (labelCache.has(term)) return labelCache.get(term);
  try {
    const { text } = await translateText(term, 'iw', 'en');
    const label = text.trim() || term;
    labelCache.set(term, label);
    return label;
  } catch (err) {
    labelCache.set(term, term);
    return term;
  }
}

function setupTranslation() {
  const input = document.getElementById('translate-input');
  const btn = document.getElementById('translate-btn');
  const output = document.getElementById('translate-output');
  const useBtn = document.getElementById('use-translation-btn');
  const status = document.getElementById('translate-status');

  let lastHebrew = '';
  let lastOriginal = '';

  async function doTranslate() {
    const text = input.value.trim();
    if (!text) {
      status.textContent = 'Type a word or phrase first.';
      return;
    }

    btn.disabled = true;
    useBtn.disabled = true;
    output.textContent = '';
    lastHebrew = '';
    lastOriginal = '';
    status.textContent = 'Translating…';

    try {
      const { text: translated, detected } = await translateText(text, 'auto', 'iw');
      const detectedLabel = detected ? ` (detected language: ${detected})` : '';

      output.textContent = translated || '—';

      const hebrewOnly = translated.replace(/[^א-ת]/g, '').slice(0, 12);
      if (hebrewOnly.length >= 2) {
        lastHebrew = hebrewOnly;
        lastOriginal = text;
        useBtn.disabled = false;
        status.textContent = `Translated${detectedLabel}. Hebrew letters for searching: "${hebrewOnly}".`;
      } else {
        status.textContent = `Translated${detectedLabel}, but found fewer than 2 Hebrew letters in the result.`;
      }
    } catch (err) {
      console.error(err);
      status.textContent = 'Translation failed — check your internet connection and try again.';
    } finally {
      btn.disabled = false;
    }
  }

  btn.addEventListener('click', doTranslate);
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      doTranslate();
    }
  });

  useBtn.addEventListener('click', () => {
    if (!lastHebrew) return;
    const target = activeInput || document.querySelector('.heb-input');
    if (target) {
      target.value = lastHebrew;
      target.dataset.label = lastOriginal;
      labelCache.set(lastHebrew, lastOriginal);
      target.focus();
    }
  });
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------
async function onSearch() {
  const termInputs = Array.from(document.querySelectorAll('.heb-input'));
  const entries = termInputs
    .map((input) => ({ term: input.value.trim(), label: input.dataset.label }))
    .filter((e) => e.term);

  if (entries.length === 0) {
    setStatus('Enter at least one Hebrew word (2+ letters) to search for.');
    return;
  }
  for (const e of entries) {
    if (e.term.length < 2) {
      setStatus(`"${e.term}" is too short — a word needs at least 2 letters.`);
      return;
    }
  }

  const minSkip = clamp(parseInt(document.getElementById('min-skip').value, 10) || 2, 1, 2000);
  let maxSkip = clamp(parseInt(document.getElementById('max-skip').value, 10) || 50, minSkip, 2000);
  document.getElementById('min-skip').value = minSkip;
  document.getElementById('max-skip').value = maxSkip;

  const bookSel = document.getElementById('book-select').value;
  const range = bookSel === 'all' ? null : rangeForBook(bookSel);

  const searchBtn = document.getElementById('search-btn');
  searchBtn.disabled = true;
  document.getElementById('results').innerHTML = '';
  document.getElementById('overlap-status').textContent = '';
  clearGrid();

  setStatus('Looking up word labels…');
  await Promise.all(entries.map(async (e) => {
    if (!e.label) e.label = await labelForTerm(e.term);
  }));

  setStatus('Searching…');
  await new Promise((r) => setTimeout(r, 0));

  const t0 = performance.now();
  const allResults = [];
  for (let ti = 0; ti < entries.length; ti++) {
    const { term, label } = entries[ti];
    const matches = searchELS(term, minSkip, maxSkip, range, 1000);
    matches.sort((a, b) => Math.abs(a.skip) - Math.abs(b.skip) || a.start - b.start);
    const color = colorForIndex(ti);
    matches.forEach((m) => { m.color = color; m.term = term; m.label = label; });
    allResults.push({ term, label, color, matches });
    await new Promise((r) => setTimeout(r, 0));
  }
  const elapsed = (performance.now() - t0).toFixed(0);

  state.allResults = allResults;
  state.flatMatches = allResults.flatMap((r) => r.matches);

  renderResults(allResults, minSkip, maxSkip);

  const totalFound = allResults.reduce((s, r) => s + r.matches.length, 0);
  setStatus(
    `Search complete in ${elapsed} ms — skip range ${minSkip}–${maxSkip}, ` +
    `${totalFound.toLocaleString()} total ELS occurrence(s) found.`
  );
  searchBtn.disabled = false;

  openDrawer('results-drawer');
}

function renderResults(allResults, minSkip, maxSkip) {
  const container = document.getElementById('results');
  container.innerHTML = '';

  allResults.forEach((r) => {
    const section = document.createElement('div');
    section.className = 'result-section';

    const heading = document.createElement('h3');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = r.color;
    heading.appendChild(swatch);
    heading.appendChild(document.createTextNode(` ${r.term} `));
    if (r.label && r.label !== r.term) {
      const tag = document.createElement('span');
      tag.className = 'muted';
      tag.textContent = `"${r.label}"`;
      heading.appendChild(tag);
    }
    heading.appendChild(document.createTextNode(` — ${r.matches.length.toLocaleString()} match(es)`));
    section.appendChild(heading);

    if (r.matches.length === 0) {
      const p = document.createElement('p');
      p.className = 'muted';
      p.textContent = `Not found at any skip between ${minSkip} and ${maxSkip}. Try a wider skip range or a shorter word.`;
      section.appendChild(p);
    } else {
      const minSkipMatch = r.matches[0];
      const summary = document.createElement('p');
      summary.className = 'muted';
      summary.textContent =
        `Smallest skip found: ${minSkipMatch.skip > 0 ? '+' : ''}${minSkipMatch.skip} ` +
        `(near ${refForIndex(centerOfMatch(minSkipMatch))}).`;
      section.appendChild(summary);

      const list = document.createElement('div');
      list.className = 'match-list';
      const shown = r.matches.slice(0, 30);
      shown.forEach((m) => {
        const item = document.createElement('div');
        item.className = 'match-item';

        const label = document.createElement('span');
        label.className = 'match-label';
        const startRef = refForIndex(Math.min(...m.indices));
        const endRef = refForIndex(Math.max(...m.indices));
        const skipLabel = `${m.skip > 0 ? '+' : ''}${m.skip}`;
        label.textContent = startRef === endRef
          ? `skip ${skipLabel} — ${startRef}`
          : `skip ${skipLabel} — ${startRef} → ${endRef}`;

        const viewBtn = document.createElement('button');
        viewBtn.type = 'button';
        viewBtn.className = 'view-btn';
        viewBtn.textContent = 'View grid';
        viewBtn.addEventListener('click', () => showGrid(m));

        item.appendChild(label);
        item.appendChild(viewBtn);
        list.appendChild(item);
      });
      section.appendChild(list);

      if (r.matches.length > shown.length) {
        const more = document.createElement('p');
        more.className = 'muted';
        more.textContent = `… and ${(r.matches.length - shown.length).toLocaleString()} more, sorted by smallest skip.`;
        section.appendChild(more);
      }
    }

    container.appendChild(section);
  });
}

// ---------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------
function clearGrid() {
  state.currentCenter = undefined;
  document.getElementById('grid-container').innerHTML =
    '<p class="muted" id="grid-placeholder">Open "Search" below, run a search, then choose "View grid" on a result to explore the letter grid here.</p>';
  document.getElementById('grid-info').textContent = '';
}

function showGrid(match) {
  const center = centerOfMatch(match);
  const suggested = clamp(Math.abs(match.skip) || 10, 5, 80);
  document.getElementById('grid-width').value = suggested;
  document.getElementById('grid-height').value = suggested;
  state.currentCenter = center;
  renderGrid();
}

function renderGrid() {
  if (state.currentCenter === undefined) return;
  const cols = clamp(parseInt(document.getElementById('grid-width').value, 10) || 10, 4, 100);
  const rows = clamp(parseInt(document.getElementById('grid-height').value, 10) || 10, 4, 100);
  document.getElementById('grid-width').value = cols;
  document.getElementById('grid-height').value = rows;

  const grid = buildGrid(state.currentCenter, rows, cols);
  const topLeftIdx = grid[0][0].idx;
  const bottomRightIdx = grid[rows - 1][cols - 1].idx;

  const highlightMap = new Map(); // idx -> Set of colors
  const linePaths = []; // { color, label, points: [[x, y], ...] }
  for (const m of state.flatMatches) {
    const pts = [];
    for (const idx of m.indices) {
      if (idx >= topLeftIdx && idx <= bottomRightIdx) {
        if (!highlightMap.has(idx)) highlightMap.set(idx, new Set());
        highlightMap.get(idx).add(m.color);

        const rel = idx - topLeftIdx;
        const r = Math.floor(rel / cols);
        const c = rel % cols;
        pts.push([cols - c - 0.5, r + 0.5]); // RTL columns: 0 is rightmost
      }
    }
    if (pts.length >= 2) linePaths.push({ color: m.color, label: m.label, points: pts });
  }

  const table = document.createElement('table');
  table.className = 'els-grid';
  for (const row of grid) {
    const tr = document.createElement('tr');
    for (const cell of row) {
      const td = document.createElement('td');
      td.textContent = cell.char;
      if (cell.char) {
        td.title = refForIndex(cell.idx);
      }
      if (highlightMap.has(cell.idx)) {
        const colors = Array.from(highlightMap.get(cell.idx));
        td.classList.add('hl');
        td.style.background = colors.length === 1
          ? colors[0]
          : `linear-gradient(135deg, ${colors.join(', ')})`;
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  const wrap = document.createElement('div');
  wrap.className = 'grid-wrap';
  wrap.appendChild(table);
  wrap.appendChild(buildLinesSVG(rows, cols, linePaths));

  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  container.appendChild(wrap);

  const firstIdx = clamp(topLeftIdx, 0, TorahData.text.length - 1);
  const lastIdx = clamp(bottomRightIdx, 0, TorahData.text.length - 1);
  document.getElementById('grid-info').textContent =
    `${rows}×${cols} grid — covers ${refForIndex(firstIdx)} … ${refForIndex(lastIdx)}. ` +
    `Highlighted cells show every searched word that falls inside this view, including ones that cross.`;
}

// Build an SVG overlay with one polyline + start marker + label per match,
// connecting its letters from first to last in the grid's cell coordinates.
const SVG_NS = 'http://www.w3.org/2000/svg';
function buildLinesSVG(rows, cols, linePaths) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'els-lines');
  svg.setAttribute('viewBox', `0 0 ${cols} ${rows}`);
  svg.setAttribute('preserveAspectRatio', 'none');

  for (const { color, label, points } of linePaths) {
    const pointsAttr = points.map((p) => p.join(',')).join(' ');

    const halo = document.createElementNS(SVG_NS, 'polyline');
    halo.setAttribute('points', pointsAttr);
    halo.setAttribute('class', 'els-line-halo');
    svg.appendChild(halo);

    const line = document.createElementNS(SVG_NS, 'polyline');
    line.setAttribute('points', pointsAttr);
    line.setAttribute('class', 'els-line');
    line.setAttribute('stroke', color);
    svg.appendChild(line);

    const start = document.createElementNS(SVG_NS, 'circle');
    start.setAttribute('cx', points[0][0]);
    start.setAttribute('cy', points[0][1]);
    start.setAttribute('r', 0.16);
    start.setAttribute('class', 'els-line-start');
    start.setAttribute('fill', color);
    svg.appendChild(start);

    if (label) {
      const mid = points[Math.floor(points.length / 2)];
      const [x1, y1] = points[0];
      const [x2, y2] = points[points.length - 1];
      let angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
      if (angle > 90) angle -= 180;
      if (angle < -90) angle += 180;

      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', mid[0]);
      text.setAttribute('y', mid[1]);
      text.setAttribute('font-size', '0.4');
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('dominant-baseline', 'central');
      text.setAttribute('transform', `rotate(${angle.toFixed(1)} ${mid[0]} ${mid[1]})`);
      text.setAttribute('class', 'els-line-label');
      text.setAttribute('fill', color);
      text.textContent = label;
      svg.appendChild(text);
    }
  }

  return svg;
}

// ---------------------------------------------------------------------
// Overlap finder — jump the grid to where two or more searched words
// cross (share a letter), or come closest together if none cross.
// ---------------------------------------------------------------------
function findOverlapView() {
  const status = document.getElementById('overlap-status');
  const matches = state.flatMatches;

  if (matches.length === 0) {
    status.textContent = 'Run a search first.';
    return;
  }

  const distinctTerms = new Set(matches.map((m) => m.term));
  if (distinctTerms.size < 2) {
    status.textContent = 'Add a second search word and run the search again to look for overlaps.';
    return;
  }

  // idx -> set of terms whose ELS line passes through this letter
  const idxToTerms = new Map();
  for (const m of matches) {
    for (const idx of m.indices) {
      let set = idxToTerms.get(idx);
      if (!set) { set = new Set(); idxToTerms.set(idx, set); }
      set.add(m.term);
    }
  }

  let bestIdx = null;
  let bestCount = 1;
  for (const [idx, terms] of idxToTerms) {
    if (terms.size > bestCount) {
      bestCount = terms.size;
      bestIdx = idx;
    }
  }

  let center, size, message;

  if (bestIdx !== null) {
    const here = matches.filter((m) => m.indices.includes(bestIdx));
    let lo = Infinity, hi = -Infinity;
    for (const m of here) {
      for (const idx of m.indices) {
        if (idx < lo) lo = idx;
        if (idx > hi) hi = idx;
      }
    }
    const span = hi - lo;
    center = Math.round((lo + hi) / 2);
    size = clamp(Math.ceil(Math.sqrt(span + 1)) + 2, 4, 100);
    const labels = Array.from(new Set(here.map((m) => m.label || m.term)));
    message = `Found ${bestCount} words crossing at the same letter near ${refForIndex(bestIdx)}: "${labels.join('", "')}".`;
  } else {
    // No shared letters — find the closest pair of matches from different terms.
    const sorted = matches.slice().sort((a, b) => centerOfMatch(a) - centerOfMatch(b));
    const lastSeen = new Map(); // term -> { center, match }
    let bestDist = Infinity;
    let bestPair = null;
    for (const m of sorted) {
      const c = centerOfMatch(m);
      for (const [term, prev] of lastSeen) {
        if (term === m.term) continue;
        const dist = Math.abs(c - prev.center);
        if (dist < bestDist) {
          bestDist = dist;
          bestPair = [prev.match, m];
        }
      }
      lastSeen.set(m.term, { center: c, match: m });
    }

    const [m1, m2] = bestPair;
    const allIdx = [...m1.indices, ...m2.indices];
    const lo = Math.min(...allIdx);
    const hi = Math.max(...allIdx);
    const span = hi - lo;
    center = Math.round((lo + hi) / 2);
    size = clamp(Math.ceil(Math.sqrt(span + 1)) + 2, 4, 100);
    message = `"${m1.label || m1.term}" and "${m2.label || m2.term}" don't share a letter, ` +
      `but their closest occurrences come together near ${refForIndex(center)}.`;
  }

  document.getElementById('grid-width').value = size;
  document.getElementById('grid-height').value = size;
  state.currentCenter = center;
  renderGrid();
  status.textContent = message;
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
async function init() {
  setupLayout();
  setupDrawers();
  setupTheme();
  setupKeyboard();
  setupTermRows();
  setupPresets();
  setupTranslation();
  clearGrid();

  document.getElementById('search-btn').addEventListener('click', onSearch);
  document.getElementById('grid-width').addEventListener('input', renderGrid);
  document.getElementById('grid-height').addEventListener('input', renderGrid);
  document.getElementById('find-overlap-btn').addEventListener('click', findOverlapView);

  setStatus('Loading Torah text…');
  try {
    await loadTorahData();
    setStatus(`Ready — ${TorahData.text.length.toLocaleString()} letters of the Torah loaded (Genesis through Deuteronomy).`);
    document.getElementById('letter-count').textContent = TorahData.text.length.toLocaleString();
  } catch (err) {
    setStatus('Failed to load Torah text data. Please reload the page.');
    console.error(err);
  }
}

document.addEventListener('DOMContentLoaded', init);

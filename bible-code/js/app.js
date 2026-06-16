const COLORS = ['#c9a227', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#e67e22', '#16a085', '#d35400'];
const MAX_TERMS = 30;

// Converts an HSL color to a #rrggbb hex string, so generated colors can be
// used as <input type="color"> values (which require hex).
function hslToHex(h, s, l) {
  s /= 100;
  l /= 100;
  const k = (n) => (n + h / 30) % 12;
  const a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  const toHex = (x) => Math.round(255 * x).toString(16).padStart(2, '0');
  return `#${toHex(f(0))}${toHex(f(8))}${toHex(f(4))}`;
}

function colorForIndex(i) {
  if (i < COLORS.length) return COLORS[i];
  const hue = (i * 47) % 360;
  return hslToHex(hue, 60, 42);
}

const state = {
  allResults: [],       // [{ term, label, color, matches }]
  flatMatches: [],      // every match across all terms
  currentCenter: undefined,
  favorites: [],        // pinned matches, persisted to localStorage
  gridHistory: [],      // recently viewed grids, persisted to localStorage
  overlayMatches: [],   // [{match, color, label}] accumulated for multi-grid overlay
  crosswordWords: [],   // [{word, color, opacity, cells}] from open search
  gridViewMode: 'overlay', // 'overlay' | 'side-by-side'
  gridBounds: null,     // { topLeftIdx, bottomRightIdx, cols } for current grid view
};

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function sanitizeHebrew(input) {
  const cleaned = input.value.replace(/[^א-ת]/g, '');
  if (cleaned !== input.value) input.value = cleaned;
}

// ---------------------------------------------------------------------
// Gematria (Hebrew letter numeric values, Mispar Hechrachi)
// ---------------------------------------------------------------------
const GEMATRIA_VALUES = {
  'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9, 'י': 10,
  'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80, 'צ': 90,
  'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
  'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
};

function letterValue(ch) {
  return GEMATRIA_VALUES[ch] || 0;
}

function gematriaValue(word) {
  let total = 0;
  for (const ch of word) total += letterValue(ch);
  return total;
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
  'favorites-drawer': 'toggle-favorites-btn',
  'search-drawer': 'toggle-search-btn',
};

// Right-side drawers share the same slot, so opening one closes the other.
const DRAWER_EXCLUSIVE = {
  'results-drawer': 'favorites-drawer',
  'favorites-drawer': 'results-drawer',
};

function setDrawerOpen(id, open) {
  document.getElementById(id).classList.toggle('open', open);
  const btnId = DRAWER_TOGGLES[id];
  if (btnId) document.getElementById(btnId).classList.toggle('active', open);
  if (open && DRAWER_EXCLUSIVE[id]) setDrawerOpen(DRAWER_EXCLUSIVE[id], false);
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
  document.getElementById('toggle-favorites-btn').addEventListener('click', () => toggleDrawer('favorites-drawer'));

  // Closing the search drawer runs a search with whatever words are tagged.
  const toggleSearchDrawer = () => {
    const wasOpen = document.getElementById('search-drawer').classList.contains('open');
    toggleDrawer('search-drawer');
    if (wasOpen) onSearch({ silent: true });
  };
  document.getElementById('toggle-search-btn').addEventListener('click', toggleSearchDrawer);
  document.getElementById('search-drawer-handle').addEventListener('click', toggleSearchDrawer);

  document.querySelectorAll('.drawer-close').forEach((btn) => {
    btn.addEventListener('click', () => closeDrawer(btn.dataset.close));
  });

  // Search drawer starts open so first-time users see the controls.
  openDrawer('search-drawer');
}

// ---------------------------------------------------------------------
// Swipe gestures — close drawers by swiping, open from edges
// ---------------------------------------------------------------------
function setupSwipeDrawers() {
  const DIST = 52, VEL = 0.22;

  function addSwipe(el, onEnd) {
    let sx, sy, st;
    el.addEventListener('touchstart', (e) => {
      sx = e.touches[0].clientX; sy = e.touches[0].clientY; st = Date.now();
    }, { passive: true });
    el.addEventListener('touchend', (e) => {
      if (sx == null) return;
      const dx = e.changedTouches[0].clientX - sx;
      const dy = e.changedTouches[0].clientY - sy;
      const dt = Math.max(1, Date.now() - st);
      sx = sy = st = null;
      onEnd(dx, dy, dt);
    }, { passive: true });
  }

  function isSwipe(delta, perp, dt) {
    if (Math.abs(perp) > Math.abs(delta)) return false;
    return Math.abs(delta) > DIST || Math.abs(delta) / dt > VEL;
  }

  // Left drawer: swipe left to close
  addSwipe(document.getElementById('theme-drawer'), (dx, dy, dt) => {
    if (isSwipe(dx, dy, dt) && dx < 0) closeDrawer('theme-drawer');
  });

  // Right drawers: swipe right to close
  ['results-drawer', 'favorites-drawer'].forEach((id) => {
    addSwipe(document.getElementById(id), (dx, dy, dt) => {
      if (isSwipe(dx, dy, dt) && dx > 0) closeDrawer(id);
    });
  });

  // Bottom drawer: swipe down to close, swipe up on the handle to open
  addSwipe(document.getElementById('search-drawer'), (dx, dy, dt) => {
    if (isSwipe(dy, dx, dt) && dy > 0) closeDrawer('search-drawer');
  });

  // Grid stage: edge swipes to open side drawers
  let ex, ey, et;
  const stage = document.getElementById('grid-stage');
  stage.addEventListener('touchstart', (e) => {
    ex = e.touches[0].clientX; ey = e.touches[0].clientY; et = Date.now();
  }, { passive: true });
  stage.addEventListener('touchend', (e) => {
    if (ex == null) return;
    const dx = e.changedTouches[0].clientX - ex;
    const dy = e.changedTouches[0].clientY - ey;
    const dt = Math.max(1, Date.now() - et);
    const startX = ex;
    ex = ey = et = null;
    if (!isSwipe(dx, dy, dt)) return;
    if (dx > 0 && startX < 32) openDrawer('theme-drawer');
    if (dx < 0 && startX > window.innerWidth - 32) openDrawer('results-drawer');
  }, { passive: true });
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
// Grid zoom
// ---------------------------------------------------------------------
const ZOOM_LEVELS = [0.5, 0.6, 0.75, 0.9, 1, 1.15, 1.3, 1.5, 1.75, 2, 2.5, 3];

function setupZoom() {
  let zoomIndex = ZOOM_LEVELS.indexOf(1);
  const levelEl = document.getElementById('zoom-level');
  const outBtn = document.getElementById('zoom-out-btn');
  const inBtn = document.getElementById('zoom-in-btn');

  const apply = () => {
    const zoom = ZOOM_LEVELS[zoomIndex];
    document.documentElement.style.setProperty('--cell-zoom', zoom);
    levelEl.textContent = `${Math.round(zoom * 100)}%`;
    outBtn.disabled = zoomIndex === 0;
    inBtn.disabled = zoomIndex === ZOOM_LEVELS.length - 1;
  };

  outBtn.addEventListener('click', () => {
    zoomIndex = clamp(zoomIndex - 1, 0, ZOOM_LEVELS.length - 1);
    apply();
  });
  inBtn.addEventListener('click', () => {
    zoomIndex = clamp(zoomIndex + 1, 0, ZOOM_LEVELS.length - 1);
    apply();
  });

  apply();
}

// ---------------------------------------------------------------------
// Line thickness
// ---------------------------------------------------------------------
function setupLineThickness() {
  const slider = document.getElementById('line-thickness');
  const levelEl = document.getElementById('thickness-level');

  const apply = () => {
    const value = clamp(parseInt(slider.value, 10) || 1, 1, 100);
    document.documentElement.style.setProperty('--line-thickness', value / 100);
    levelEl.textContent = `${value}%`;
  };

  slider.addEventListener('input', apply);
  apply();
}

// ---------------------------------------------------------------------
// Full screen
// ---------------------------------------------------------------------
function setupFullscreen() {
  const btn = document.getElementById('fullscreen-btn');

  btn.addEventListener('click', () => {
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  });

  document.addEventListener('fullscreenchange', () => {
    const isFull = !!document.fullscreenElement;
    btn.classList.toggle('active', isFull);
    const label = isFull ? 'Exit full screen' : 'Enter full screen';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  });
}

// ---------------------------------------------------------------------
// Theme customizer
// ---------------------------------------------------------------------
const THEME_VARS = {
  'theme-bg': '--parchment',
  'theme-panel': '--parchment-dark',
  'theme-grid-bg': '--grid-bg',
  'theme-ink': '--ink',
  'theme-accent': '--gold-dark',
  'theme-slider': '--slider-accent',
  'theme-btn-bg': '--btn-bg',
  'theme-btn-text': '--btn-text',
  'theme-btn-active': '--btn-active-bg',
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
// Term rows
// ---------------------------------------------------------------------
function refreshTermRowState() {
  const rows = document.querySelectorAll('.term-row');
  rows.forEach((row) => {
    const removeBtn = row.querySelector('.remove-term');
    removeBtn.disabled = rows.length <= 1;
  });
  const addBtn = document.getElementById('add-term-btn');
  addBtn.disabled = rows.length >= MAX_TERMS;
}

// Refreshes a row's gematria badge from its resolved Hebrew term.
function updateTermGematria(row, gematria) {
  const hebrew = row.dataset.hebrew || '';
  gematria.textContent = hebrew.length >= 2 ? String(gematriaValue(hebrew)) : '';
}

// Translates whatever the user typed (any language) to Hebrew, shows the
// result as faded "ghost" text inside the box, and stores it as the row's
// search term. Falls back gracefully if translation is unavailable.
async function translateRowInput(row, input, ghost, gematria, raw) {
  try {
    const { text: translated } = await translateText(raw, 'auto', 'iw');
    if (input.value.trim() !== raw) return; // user kept typing — ignore stale result
    const hebrewOnly = translated.replace(/[^א-ת]/g, '').slice(0, 40);
    if (hebrewOnly.length >= 2) {
      row.dataset.hebrew = hebrewOnly;
      row.dataset.label = raw;
      labelCache.set(hebrewOnly, raw);
      ghost.textContent = hebrewOnly;
    } else {
      row.dataset.hebrew = '';
      ghost.textContent = '⚠ no Hebrew translation found';
    }
  } catch (err) {
    if (input.value.trim() !== raw) return;
    row.dataset.hebrew = '';
    ghost.textContent = '⚠ translation unavailable';
  }
  updateTermGematria(row, gematria);
}

// Wires up a term row's input: typing in any language auto-translates to
// Hebrew (shown as ghost text), typing Hebrew directly uses it as-is, and
// Enter runs a search with everything entered so far and opens a new row.
function setupTermInput(row, input, ghost, gematria) {
  let debounceTimer = null;

  input.addEventListener('input', () => {
    delete row.dataset.label;
    clearTimeout(debounceTimer);
    const raw = input.value.trim();

    if (!raw) {
      row.dataset.hebrew = '';
      ghost.textContent = '';
      gematria.textContent = '';
      return;
    }

    if (/^[א-ת]+$/.test(raw)) {
      row.dataset.hebrew = raw;
      ghost.textContent = '';
      updateTermGematria(row, gematria);
      return;
    }

    ghost.textContent = '…';
    debounceTimer = setTimeout(() => translateRowInput(row, input, ghost, gematria, raw), 500);
  });

  input.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter') return;
    e.preventDefault();
    if (!row.dataset.hebrew || row.dataset.hebrew.length < 2) {
      setStatus('Type a word (2+ letters once translated to Hebrew) before pressing Enter.');
      return;
    }
    onSearch();
    const next = addTermRow();
    if (next) next.focus();
  });
}

function addTermRow(prefillHebrew, label) {
  const rows = document.querySelectorAll('.term-row');
  if (rows.length >= MAX_TERMS) return null;

  const container = document.getElementById('term-rows');
  const row = document.createElement('div');
  row.className = 'term-row';
  row.dataset.hebrew = '';

  const swatch = document.createElement('input');
  swatch.type = 'color';
  swatch.className = 'swatch term-color';
  swatch.value = colorForIndex(rows.length);
  swatch.title = 'Line color for this word';
  swatch.setAttribute('aria-label', 'Line color for this word');

  const inputWrap = document.createElement('div');
  inputWrap.className = 'term-input-wrap';

  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'term-input';
  input.maxLength = 40;
  input.placeholder = 'Type a word in any language…';
  input.setAttribute('aria-label', 'Search word, any language');

  const ghost = document.createElement('span');
  ghost.className = 'term-ghost';
  ghost.setAttribute('aria-hidden', 'true');

  inputWrap.appendChild(input);
  inputWrap.appendChild(ghost);

  const gematria = document.createElement('span');
  gematria.className = 'term-gematria';
  gematria.title = 'Gematria value';

  const removeBtn = document.createElement('button');
  removeBtn.type = 'button';
  removeBtn.className = 'remove-term';
  removeBtn.title = 'Remove this word';
  removeBtn.textContent = '✕';
  removeBtn.addEventListener('click', () => {
    row.remove();
    refreshTermRowState();
  });

  row.appendChild(swatch);
  row.appendChild(inputWrap);
  row.appendChild(gematria);
  row.appendChild(removeBtn);
  container.appendChild(row);

  setupTermInput(row, input, ghost, gematria);

  if (prefillHebrew) {
    input.value = prefillHebrew;
    row.dataset.hebrew = prefillHebrew;
    if (label) row.dataset.label = label;
    updateTermGematria(row, gematria);
  }

  refreshTermRowState();
  return input;
}

function setupTermRows() {
  addTermRow('תורה', 'Torah');
  document.getElementById('add-term-btn').addEventListener('click', () => {
    const next = addTermRow();
    if (next) next.focus();
  });
}

// ---------------------------------------------------------------------
// Gematria calculator
// ---------------------------------------------------------------------
function setupGematriaCalculator() {
  const input = document.getElementById('gematria-input');
  const result = document.getElementById('gematria-result');

  const update = () => {
    sanitizeHebrew(input);
    const word = input.value;
    if (!word) { result.innerHTML = ''; return; }
    const parts = word.split('').map((ch) => `${ch}(${letterValue(ch)})`);
    result.innerHTML =
      `<span class="total">${gematriaValue(word)}</span>` +
      `<span class="gematria-breakdown">${parts.join(' + ')}</span>`;
  };

  input.addEventListener('input', update);
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

// The user's browser language (base code, e.g. "en"), used to translate
// crossword dictionary words into something the user can read.
function getUserLang() {
  return (navigator.language || 'en').split('-')[0].toLowerCase();
}

const crosswordLabelCache = new Map(); // `${hebrewWord}|${lang}` -> translation or null

// Best-effort translation of a Hebrew dictionary word into the given
// language. Returns null (and caches the failure) if unavailable.
async function translateHebrewWord(word, lang) {
  if (lang === 'he' || lang === 'iw') return null;
  const key = `${word}|${lang}`;
  if (crosswordLabelCache.has(key)) return crosswordLabelCache.get(key);
  try {
    const { text } = await translateText(word, 'iw', lang);
    const translated = text.trim() || null;
    crosswordLabelCache.set(key, translated);
    return translated;
  } catch (err) {
    crosswordLabelCache.set(key, null);
    return null;
  }
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------
async function onSearch(opts = {}) {
  const rows = Array.from(document.querySelectorAll('.term-row'));
  const entries = rows
    .map((row) => ({
      term: row.dataset.hebrew || '',
      label: row.dataset.label,
      color: row.querySelector('.term-color').value,
    }))
    .filter((e) => e.term.length >= 2);

  if (entries.length === 0) {
    if (!opts.silent) setStatus('Type at least one word (2+ Hebrew letters once translated) to search for.');
    return;
  }

  const minSkip = clamp(parseInt(document.getElementById('min-skip').value, 10) || 2, 1, 2000);
  let maxSkip = clamp(parseInt(document.getElementById('max-skip').value, 10) || 200, minSkip, 2000);
  document.getElementById('min-skip').value = minSkip;
  document.getElementById('max-skip').value = maxSkip;

  const bookSel = document.getElementById('book-select').value;
  const range = bookSel === 'all' ? null : rangeForBook(bookSel);

  document.getElementById('results').innerHTML = '';
  document.getElementById('overlap-status').textContent = '';
  state.overlayMatches = [];
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
    const { term, label, color } = entries[ti];
    const matches = searchELS(term, minSkip, maxSkip, range, 1000);
    matches.sort((a, b) => Math.abs(a.skip) - Math.abs(b.skip) || a.start - b.start);
    matches.forEach((m) => { m.term = term; m.label = label; m.termIndex = ti; });
    allResults.push({ term, label, color, matches, opacity: 1 });
    await new Promise((r) => setTimeout(r, 0));
  }
  const elapsed = (performance.now() - t0).toFixed(0);

  state.allResults = allResults;
  state.flatMatches = allResults.flatMap((r) => r.matches);
  state.recentLayers = [];

  renderResults(allResults, minSkip, maxSkip);
  renderLayerControls();

  const totalFound = allResults.reduce((s, r) => s + r.matches.length, 0);
  setStatus(
    `Search complete in ${elapsed} ms — skip range ${minSkip}–${maxSkip}, ` +
    `${totalFound.toLocaleString()} total ELS occurrence(s) found.`
  );

  openDrawer('results-drawer');
  updatePieChart();
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
        viewBtn.addEventListener('click', () => {
          showGrid(m, r.color);
          closeDrawer('results-drawer');
        });

        const overlayBtn = document.createElement('button');
        overlayBtn.type = 'button';
        overlayBtn.className = 'view-btn';
        overlayBtn.textContent = '＋ Overlay';
        overlayBtn.title = 'Add this result to the grid overlay';
        overlayBtn.addEventListener('click', () => {
          addOverlayMatch(m, r.color, r.label || r.term);
          closeDrawer('results-drawer');
        });

        const pinBtn = document.createElement('button');
        pinBtn.type = 'button';
        pinBtn.className = 'pin-btn';
        const setPinState = (pinned) => {
          pinBtn.textContent = pinned ? '★' : '☆';
          pinBtn.title = pinned ? 'Remove from favorites' : 'Pin to favorites';
          pinBtn.setAttribute('aria-label', pinBtn.title);
        };
        setPinState(isFavorited(m));
        pinBtn.addEventListener('click', () => {
          if (isFavorited(m)) {
            removeFavoriteByMatch(m);
            setPinState(false);
          } else {
            addFavorite(m, r.label, r.color);
            setPinState(true);
          }
        });

        const actions = document.createElement('div');
        actions.className = 'match-actions';
        actions.appendChild(viewBtn);
        actions.appendChild(overlayBtn);
        actions.appendChild(pinBtn);

        item.appendChild(label);
        item.appendChild(actions);
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
  document.getElementById('grid-container').className = 'grid-scroll';
  document.getElementById('grid-container').innerHTML =
    '<p class="muted" id="grid-placeholder">Open "Search" below, run a search, then choose "View grid" on a result to explore the letter grid here.</p>';
  document.getElementById('grid-info').textContent = '';
  document.getElementById('download-grid-btn').hidden = true;
  state.gridBounds = null;
  clearCrosswordResults();
  clearGridWordsMenu();
}

function showGrid(match, color) {
  const center = centerOfMatch(match);
  const suggested = clamp(Math.abs(match.skip) || 10, 5, 80);
  document.getElementById('grid-width').value = suggested;
  document.getElementById('grid-height').value = suggested;
  state.currentCenter = center;
  renderGrid();
  addGridHistory(match, color);
  document.getElementById('grid-view-btn').hidden = false;
  renderGridViewMenu();
}

function buildSingleGrid(center, rows, cols, allMatches, viewState) {
  const grid = buildGrid(center, rows, cols);
  const topLeftIdx = grid[0][0].idx;
  const bottomRightIdx = grid[rows - 1][cols - 1].idx;

  const highlightMap = new Map();
  const linePaths = [];

  // ELS matches from search results
  for (const m of allMatches) {
    const ri = viewState ? viewState.allResults[m.termIndex] : state.allResults[m.termIndex];
    if (!ri) continue;
    const color = ri.color;
    const pts = [];
    for (const idx of m.indices) {
      if (idx >= topLeftIdx && idx <= bottomRightIdx) {
        if (!highlightMap.has(idx)) highlightMap.set(idx, new Map());
        highlightMap.get(idx).set(m.termIndex, color);
        const rel = idx - topLeftIdx;
        const r = Math.floor(rel / cols);
        const c = rel % cols;
        pts.push([cols - c - 0.5, r + 0.5]);
      }
    }
    if (pts.length >= 2) linePaths.push({ color, label: m.label, points: pts, termIndex: m.termIndex });
  }

  // Extra overlay matches (from state.overlayMatches)
  for (let oi = 0; oi < state.overlayMatches.length; oi++) {
    const { match: om, color, label } = state.overlayMatches[oi];
    const pts = [];
    for (const idx of om.indices) {
      if (idx >= topLeftIdx && idx <= bottomRightIdx) {
        const syntheticTi = 1000 + oi;
        if (!highlightMap.has(idx)) highlightMap.set(idx, new Map());
        highlightMap.get(idx).set(syntheticTi, color);
        const rel = idx - topLeftIdx;
        const r = Math.floor(rel / cols);
        const c = rel % cols;
        pts.push([cols - c - 0.5, r + 0.5]);
      }
    }
    if (pts.length >= 2) linePaths.push({ color, label, points: pts, termIndex: 1000 + oi });
  }

  const table = document.createElement('table');
  table.className = 'els-grid';
  for (const row of grid) {
    const tr = document.createElement('tr');
    for (const cell of row) {
      const td = document.createElement('td');
      if (cell.char) {
        const content = document.createElement('div');
        content.className = 'cell-content';
        const letter = document.createElement('span');
        letter.className = 'cell-letter';
        letter.textContent = cell.char;
        const num = document.createElement('span');
        num.className = 'cell-num';
        num.textContent = String(letterValue(cell.char));
        content.appendChild(letter);
        content.appendChild(num);
        td.appendChild(content);
        td.title = refForIndex(cell.idx);
      }
      if (highlightMap.has(cell.idx)) {
        const entries = Array.from(highlightMap.get(cell.idx).entries());
        td.classList.add('hl');
        td.dataset.terms = entries.map(([ti]) => ti).join(',');
        // For overlay cells we manually mix colors
        const layers = entries.map(([ti, c]) => {
          const opacity = (ti < 1000 && state.allResults[ti]) ? (state.allResults[ti].opacity ?? 1) : 1;
          return colorWithOpacity(c, opacity);
        });
        td.style.background = layers.length === 1 ? layers[0] : `linear-gradient(135deg, ${layers.join(', ')})`;
      }
      tr.appendChild(td);
    }
    table.appendChild(tr);
  }

  // Build SVG for ELS lines (pass only real term lines; overlay lines use special handling below)
  const realLinePaths = linePaths.filter((lp) => lp.termIndex < 1000);
  const overlayLinePaths = linePaths.filter((lp) => lp.termIndex >= 1000);

  const svg = buildLinesSVG(rows, cols, realLinePaths);

  // Draw overlay lines as dashed with slightly different style
  for (const { color, label, points } of overlayLinePaths) {
    const pointsAttr = points.map((p) => p.join(',')).join(' ');
    const halo = document.createElementNS(SVG_NS, 'polyline');
    halo.setAttribute('points', pointsAttr);
    halo.setAttribute('class', 'els-line-halo');
    svg.appendChild(halo);
    const line = document.createElementNS(SVG_NS, 'polyline');
    line.setAttribute('points', pointsAttr);
    line.setAttribute('class', 'els-line');
    line.setAttribute('stroke', color);
    line.style.strokeDasharray = '0.25 0.1';
    svg.appendChild(line);
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

  const wrap = document.createElement('div');
  wrap.className = 'grid-wrap';
  wrap.appendChild(table);
  wrap.appendChild(svg);
  return { wrap, topLeftIdx, bottomRightIdx };
}

function renderGrid() {
  if (state.currentCenter === undefined) return;
  clearCrosswordResults();
  clearGridWordsMenu();
  const cols = clamp(parseInt(document.getElementById('grid-width').value, 10) || 10, 4, 100);
  const rows = clamp(parseInt(document.getElementById('grid-height').value, 10) || 10, 4, 100);
  document.getElementById('grid-width').value = cols;
  document.getElementById('grid-height').value = rows;

  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  container.className = 'grid-scroll';

  if (state.gridViewMode === 'side-by-side' && state.overlayMatches.length > 0) {
    // Side-by-side: show one grid per overlay match + primary
    container.className = 'grid-side-by-side';
    const allPanels = [
      { center: state.currentCenter, matches: state.flatMatches, label: 'Primary' },
      ...state.overlayMatches.map(({ match, label }) => ({
        center: centerOfMatch(match),
        matches: [...state.flatMatches, match],
        label,
      })),
    ];
    for (const panel of allPanels) {
      const { wrap, topLeftIdx, bottomRightIdx } = buildSingleGrid(
        panel.center, rows, cols, panel.matches, null
      );
      const panelDiv = document.createElement('div');
      panelDiv.style.display = 'flex';
      panelDiv.style.flexDirection = 'column';
      panelDiv.style.alignItems = 'center';
      panelDiv.style.gap = '0.25rem';
      const lbl = document.createElement('div');
      lbl.textContent = panel.label;
      lbl.style.fontSize = '0.72rem';
      lbl.style.color = 'var(--ink-light)';
      panelDiv.appendChild(lbl);
      panelDiv.appendChild(wrap);
      container.appendChild(panelDiv);
    }
    document.getElementById('grid-info').textContent =
      `Side-by-side: ${allPanels.length} grid(s), ${rows}×${cols} each.`;
  } else {
    // Overlay mode (default): single grid showing everything
    const { wrap, topLeftIdx, bottomRightIdx } = buildSingleGrid(
      state.currentCenter, rows, cols, state.flatMatches, null
    );
    state.gridBounds = { topLeftIdx, bottomRightIdx, cols };
    container.appendChild(wrap);
    const firstIdx = clamp(topLeftIdx, 0, TorahData.text.length - 1);
    const lastIdx = clamp(bottomRightIdx, 0, TorahData.text.length - 1);
    document.getElementById('grid-info').textContent =
      `${rows}×${cols} grid — covers ${refForIndex(firstIdx)} … ${refForIndex(lastIdx)}. ` +
      `Highlighted cells show every searched word that falls inside this view, including ones that cross.`;
  }

  // Show download button when grid is visible
  document.getElementById('download-grid-btn').hidden = false;
  updatePieChart();
}

// Build an SVG overlay with one polyline + start marker + label per match,
// connecting its letters from first to last in the grid's cell coordinates.
const SVG_NS = 'http://www.w3.org/2000/svg';
function buildLinesSVG(rows, cols, linePaths) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('class', 'els-lines');
  svg.setAttribute('viewBox', `0 0 ${cols} ${rows}`);
  svg.setAttribute('preserveAspectRatio', 'none');

  for (const { color, label, points, termIndex } of linePaths) {
    const pointsAttr = points.map((p) => p.join(',')).join(' ');
    const opacity = state.allResults[termIndex]?.opacity ?? 1;

    const halo = document.createElementNS(SVG_NS, 'polyline');
    halo.setAttribute('points', pointsAttr);
    halo.setAttribute('class', 'els-line-halo');
    halo.dataset.termIndex = String(termIndex);
    halo.style.opacity = opacity;
    svg.appendChild(halo);

    const line = document.createElementNS(SVG_NS, 'polyline');
    line.setAttribute('points', pointsAttr);
    line.setAttribute('class', 'els-line');
    line.setAttribute('stroke', color);
    line.dataset.termIndex = String(termIndex);
    line.style.opacity = opacity;
    svg.appendChild(line);

    const start = document.createElementNS(SVG_NS, 'circle');
    start.setAttribute('cx', points[0][0]);
    start.setAttribute('cy', points[0][1]);
    start.setAttribute('r', 0.16);
    start.setAttribute('class', 'els-line-start');
    start.setAttribute('fill', color);
    start.dataset.termIndex = String(termIndex);
    start.style.opacity = opacity;
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
      text.dataset.termIndex = String(termIndex);
      text.style.opacity = opacity;
      text.textContent = label;
      svg.appendChild(text);
    }
  }

  return svg;
}

// ---------------------------------------------------------------------
// Layer opacity controls (per-word, shown in the top bar)
// ---------------------------------------------------------------------
function colorWithOpacity(color, opacity) {
  const pct = Math.round(clamp(opacity, 0, 1) * 100);
  return `color-mix(in srgb, ${color} ${pct}%, transparent)`;
}

function applyCellBackground(td, entries) {
  const layers = entries.map(([ti, color]) => colorWithOpacity(color, state.allResults[ti]?.opacity ?? 1));
  td.style.background = layers.length === 1 ? layers[0] : `linear-gradient(135deg, ${layers.join(', ')})`;
}

function applyLineOpacity(termIndex, opacity) {
  document.querySelectorAll(`[data-term-index="${termIndex}"]`).forEach((el) => {
    el.style.opacity = opacity;
  });
}

// Recolors a term's line, start marker, and label in the current grid view.
function applyLineColor(termIndex, color) {
  document.querySelectorAll(`.els-line[data-term-index="${termIndex}"]`).forEach((el) => {
    el.setAttribute('stroke', color);
  });
  document.querySelectorAll(`.els-line-start[data-term-index="${termIndex}"], .els-line-label[data-term-index="${termIndex}"]`).forEach((el) => {
    el.setAttribute('fill', color);
  });
}

function refreshCellsForTerm(termIndex) {
  document.querySelectorAll('.els-grid td.hl').forEach((td) => {
    const terms = (td.dataset.terms || '').split(',').filter(Boolean).map(Number);
    if (!terms.includes(termIndex)) return;
    const entries = terms.map((ti) => [ti, state.allResults[ti].color]);
    applyCellBackground(td, entries);
  });
}

const MAX_PINNED_LAYERS = 4;

// Builds one swatch + label + opacity slider for a single searched word's
// layer controls. In the dropdown ("pinnable"), a pin button also lets the
// user promote that word's controls into the toolbar's pinned set.
function createLayerChip(i, r, pinnable) {
  const chip = document.createElement('div');
  chip.className = 'layer-chip';
  chip.dataset.chipIndex = String(i);

  const colorInput = document.createElement('input');
  colorInput.type = 'color';
  colorInput.className = 'swatch layer-color';
  colorInput.value = r.color;
  colorInput.title = `${r.label || r.term} line color`;
  colorInput.setAttribute('aria-label', `${r.label || r.term} line color`);
  colorInput.addEventListener('input', () => {
    r.color = colorInput.value;
    applyLineColor(i, r.color);
    refreshCellsForTerm(i);
    syncLayerChip(i, r);
  });

  const label = document.createElement('span');
  label.className = 'layer-label';
  label.textContent = r.label || r.term;
  label.title = r.label || r.term;

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.value = String(Math.round((r.opacity ?? 1) * 100));
  slider.className = 'layer-opacity';
  slider.setAttribute('aria-label', `${r.label || r.term} opacity`);
  slider.addEventListener('input', () => {
    const opacity = clamp(parseInt(slider.value, 10), 0, 100) / 100;
    r.opacity = opacity;
    applyLineOpacity(i, opacity);
    refreshCellsForTerm(i);
    syncLayerChip(i, r);
  });

  chip.appendChild(colorInput);
  chip.appendChild(label);
  chip.appendChild(slider);

  if (pinnable) {
    const pinBtn = document.createElement('button');
    pinBtn.type = 'button';
    pinBtn.className = 'layer-pin';
    pinBtn.title = 'Show in toolbar';
    pinBtn.setAttribute('aria-label', `Show ${r.label || r.term} in toolbar`);
    pinBtn.textContent = '📌';
    pinBtn.addEventListener('click', () => {
      state.recentLayers = [i, ...state.recentLayers.filter((x) => x !== i)].slice(0, MAX_PINNED_LAYERS);
      renderLayerControls();
    });
    chip.appendChild(pinBtn);
  }

  return chip;
}

// Keeps a term's toolbar chip and dropdown chip in sync when either is edited.
function syncLayerChip(i, r) {
  document.querySelectorAll(`[data-chip-index="${i}"]`).forEach((chip) => {
    const colorInput = chip.querySelector('.layer-color');
    const slider = chip.querySelector('.layer-opacity');
    if (colorInput.value !== r.color) colorInput.value = r.color;
    const sliderVal = String(Math.round((r.opacity ?? 1) * 100));
    if (slider.value !== sliderVal) slider.value = sliderVal;
  });
}

// Renders the pinned toolbar chips (up to 4, most recently selected first)
// plus a dropdown menu with controls for every searched word with results.
function renderLayerControls() {
  const container = document.getElementById('layer-controls');
  const menu = document.getElementById('layer-menu');
  const menuBtn = document.getElementById('layer-menu-btn');
  container.innerHTML = '';
  menu.innerHTML = '';

  const visible = state.allResults
    .map((r, i) => ({ r, i }))
    .filter(({ r }) => r.matches.length > 0);

  container.hidden = visible.length === 0;
  menuBtn.hidden = visible.length === 0;
  if (visible.length === 0) {
    closeLayerMenu();
    return;
  }

  const visibleIndices = visible.map(({ i }) => i);
  state.recentLayers = (state.recentLayers || []).filter((i) => visibleIndices.includes(i));
  for (const i of visibleIndices.slice(-MAX_PINNED_LAYERS).reverse()) {
    if (state.recentLayers.length >= MAX_PINNED_LAYERS) break;
    if (!state.recentLayers.includes(i)) state.recentLayers.push(i);
  }

  state.recentLayers.forEach((i) => {
    container.appendChild(createLayerChip(i, state.allResults[i], false));
  });

  visible.forEach(({ r, i }) => {
    menu.appendChild(createLayerChip(i, r, true));
  });
}

// ---------------------------------------------------------------------
// "All words" layer dropdown menu
// ---------------------------------------------------------------------
function closeLayerMenu() {
  document.getElementById('layer-menu').classList.remove('open');
  document.getElementById('layer-menu-btn').classList.remove('active');
}

function setupLayerMenu() {
  const btn = document.getElementById('layer-menu-btn');
  const menu = document.getElementById('layer-menu');

  btn.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    btn.classList.toggle('active', open);
  });

  document.addEventListener('click', (e) => {
    if (!menu.classList.contains('open')) return;
    if (menu.contains(e.target) || btn.contains(e.target)) return;
    if (!document.body.contains(e.target)) return;
    closeLayerMenu();
  });
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
// Crossing Hebrew words — scans the visible grid in all 8 directions for
// dictionary words (not tied to any search tag) that cross a cell from one
// of the user's searched ELS lines.
// ---------------------------------------------------------------------
const HEBREW_WORDS = [
  'תורה', 'משה', 'אהרן', 'דוד', 'שלמה', 'ישראל', 'יעקב', 'יצחק', 'אברהם', 'שרה',
  'רבקה', 'רחל', 'לאה', 'יהוה', 'אלהים', 'שדי', 'אדני', 'שמים', 'ארץ', 'מים',
  'אור', 'חשך', 'יום', 'לילה', 'שמש', 'ירח', 'כוכב', 'אדם', 'אשה', 'איש',
  'מלך', 'גוי', 'עיר', 'בית', 'דרך', 'ספר', 'דבר', 'נפש', 'רוח', 'פרי',
  'זרע', 'חיים', 'מות', 'שלום', 'אהבה', 'שנאה', 'אמונה', 'תקוה', 'חכמה', 'צדק',
  'משפט', 'חטא', 'קדוש', 'טהור', 'טמא', 'ברכה', 'קללה', 'ברית', 'מצוה', 'עולם',
  'ארון', 'מקדש', 'כהן', 'קרבן', 'מזבח', 'שבת', 'פסח', 'מדבר', 'נהר', 'מלאך',
  'נביא', 'חלום', 'קול', 'עין', 'רגל', 'ראש', 'פנים', 'לחם', 'חרב', 'קשת',
  'עבד', 'אדון', 'נחש', 'אהל', 'מחנה', 'צבא', 'מלחמה', 'שלל', 'אויב', 'גבור',
  'חיל', 'עצם', 'בשר', 'עור', 'ראה', 'שמע', 'אמר', 'הלך', 'ישב', 'קום',
  'בוא', 'יצא', 'עלה', 'ירד', 'נתן', 'לקח', 'עשה', 'ברא', 'ידע', 'אכל',
  'שתה', 'כתב', 'קרא', 'שלח', 'מצא', 'בנה', 'ילד', 'חיה', 'מלא', 'חדש',
  'ישן', 'גדול', 'קטן', 'טוב', 'יפה', 'חזק', 'חלש', 'עני', 'עשיר', 'ארבע',
  'חמש', 'שבע', 'שמנה', 'תשע', 'עשר', 'מאה', 'אלף',
];

const CROSS_DIRECTIONS = [
  [0, 1], [0, -1], [1, 0], [-1, 0],
  [1, 1], [1, -1], [-1, 1], [-1, -1],
];

function clearCrosswordResults() {
  document.getElementById('crossword-results').innerHTML = '';
  document.getElementById('crossword-status').textContent = '';
}

// Scans every cell of the currently rendered grid for Hebrew dictionary words
// in all 8 directions (regardless of whether they cross any ELS-tagged cell).
function runCrosswordSearch() {
  const status = document.getElementById('crossword-status');
  const resultsEl = document.getElementById('crossword-results');
  resultsEl.innerHTML = '';

  const table = document.querySelector('.els-grid');
  if (state.currentCenter === undefined || !table) {
    status.textContent = 'Run a search and open a grid view first.';
    return;
  }

  const cols = clamp(parseInt(document.getElementById('grid-width').value, 10) || 10, 4, 100);
  const rows = clamp(parseInt(document.getElementById('grid-height').value, 10) || 10, 4, 100);
  const grid = buildGrid(state.currentCenter, rows, cols);
  const topLeftIdx = grid[0][0].idx;

  document.querySelectorAll('.els-grid td.cross').forEach((td) => td.classList.remove('cross'));
  document.querySelectorAll('.els-line-cross, .els-line-cross-label').forEach((el) => el.remove());

  const taggedTerms = new Set(state.allResults.map((r) => r.term));
  const found = [];
  const seenKeys = new Set();

  for (const word of HEBREW_WORDS) {
    if (taggedTerms.has(word)) continue;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (grid[r][c].char !== word[0]) continue;
        for (const [dr, dc] of CROSS_DIRECTIONS) {
          const endR = r + dr * (word.length - 1);
          const endC = c + dc * (word.length - 1);
          if (endR < 0 || endR >= rows || endC < 0 || endC >= cols) continue;

          const cells = [];
          let ok = true;
          for (let k = 0; k < word.length; k++) {
            const cell = grid[r + dr * k][c + dc * k];
            if (cell.char !== word[k]) { ok = false; break; }
            cells.push(cell);
          }
          if (!ok) continue;

          const key = cells.map((cell) => cell.idx).join(',');
          const revKey = cells.slice().reverse().map((cell) => cell.idx).join(',');
          if (seenKeys.has(key) || seenKeys.has(revKey)) continue;
          seenKeys.add(key);
          found.push({ word, cells });
        }
      }
    }
  }

  if (found.length === 0) {
    status.textContent = 'No Hebrew dictionary words found in this grid view.';
    state.crosswordWords = [];
    renderGridWordsMenu();
    updatePieChart();
    return;
  }

  found.sort((a, b) => b.word.length - a.word.length);
  const shown = found.slice(0, 40);

  // Assign a default color per word using existing palette
  state.crosswordWords = shown.map((fw, i) => ({
    word: fw.word,
    cells: fw.cells,
    color: colorForIndex(state.allResults.length + i),
    opacity: 0.8,
    translated: null,
  }));

  const svg = document.querySelector('.els-lines');
  const userLang = getUserLang();

  state.crosswordWords.forEach((cw, i) => {
    const { word, cells } = cw;
    const points = [];
    for (const cell of cells) {
      const rel = cell.idx - topLeftIdx;
      const r = Math.floor(rel / cols);
      const c = rel % cols;
      const td = table.rows[r]?.cells[c];
      if (td) td.classList.add('cross');
      points.push([cols - c - 0.5, r + 0.5]);
    }

    const line = document.createElementNS(SVG_NS, 'polyline');
    line.setAttribute('points', points.map((p) => p.join(',')).join(' '));
    line.setAttribute('class', 'els-line-cross');
    line.setAttribute('stroke', cw.color);
    line.style.opacity = cw.opacity;
    line.dataset.cwIndex = String(i);
    if (svg) svg.appendChild(line);

    const mid = points[Math.floor(points.length / 2)];
    const [x1, y1] = points[0];
    const [x2, y2] = points[points.length - 1];
    let angle = Math.atan2(y2 - y1, x2 - x1) * (180 / Math.PI);
    if (angle > 90) angle -= 180;
    if (angle < -90) angle += 180;

    const text = document.createElementNS(SVG_NS, 'text');
    text.setAttribute('x', mid[0]);
    text.setAttribute('y', mid[1]);
    text.setAttribute('font-size', '0.32');
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('dominant-baseline', 'central');
    text.setAttribute('transform', `rotate(${angle.toFixed(1)} ${mid[0]} ${mid[1]})`);
    text.setAttribute('class', 'els-line-cross-label');
    text.setAttribute('fill', cw.color);
    text.style.opacity = cw.opacity;
    text.dataset.cwIndex = String(i);
    text.textContent = `${word} (${gematriaValue(word)})`;
    if (svg) svg.appendChild(text);

    const chip = document.createElement('span');
    chip.className = 'preset';
    chip.title = refForIndex(cells[0].idx);
    chip.textContent = `${word} (${gematriaValue(word)})`;
    chip.style.borderColor = cw.color;
    resultsEl.appendChild(chip);

    translateHebrewWord(word, userLang).then((translated) => {
      if (!translated) return;
      cw.translated = translated;
      chip.textContent = `${word} (${gematriaValue(word)}) — ${translated}`;
      text.textContent = `${translated} (${gematriaValue(word)})`;
    });
  });

  const more = found.length > shown.length ? ` (showing top ${shown.length} of ${found.length} by word length)` : '';
  status.textContent = `Found ${found.length} word(s)${more} in this grid.`;

  renderGridWordsMenu();
  updatePieChart();
}

function setupCrossword() {
  document.getElementById('crossword-btn').addEventListener('click', runCrosswordSearch);
}

// ---------------------------------------------------------------------
// Grid Words dropdown — controls color/opacity of each crossword word
// ---------------------------------------------------------------------
function clearGridWordsMenu() {
  state.crosswordWords = [];
  const btn = document.getElementById('grid-words-btn');
  btn.hidden = true;
  btn.classList.remove('active');
  document.getElementById('grid-words-menu').classList.remove('open');
  document.getElementById('grid-words-menu').innerHTML = '';
}

function applyGridWordStyle(i) {
  const cw = state.crosswordWords[i];
  document.querySelectorAll(`[data-cw-index="${i}"]`).forEach((el) => {
    el.style.opacity = cw.opacity;
    if (el.tagName === 'polyline' || el.getAttribute('class') === 'els-line-cross') {
      el.setAttribute('stroke', cw.color);
    }
    if (el.getAttribute('class') === 'els-line-cross-label') {
      el.setAttribute('fill', cw.color);
    }
  });
}

function renderGridWordsMenu() {
  const menu = document.getElementById('grid-words-menu');
  const btn = document.getElementById('grid-words-btn');
  menu.innerHTML = '';

  if (state.crosswordWords.length === 0) {
    btn.hidden = true;
    return;
  }
  btn.hidden = false;

  // Master opacity slider at the top
  const masterWrap = document.createElement('div');
  masterWrap.className = 'grid-words-master';
  const masterLabel = document.createElement('label');
  masterLabel.textContent = 'All';
  const masterSlider = document.createElement('input');
  masterSlider.type = 'range';
  masterSlider.min = '0';
  masterSlider.max = '100';
  masterSlider.value = '80';
  masterSlider.addEventListener('input', () => {
    const op = parseInt(masterSlider.value, 10) / 100;
    state.crosswordWords.forEach((cw, i) => {
      cw.opacity = op;
      applyGridWordStyle(i);
    });
    // sync individual sliders
    menu.querySelectorAll('.cw-opacity-slider').forEach((sl) => { sl.value = masterSlider.value; });
  });
  masterWrap.appendChild(masterLabel);
  masterWrap.appendChild(masterSlider);
  menu.appendChild(masterWrap);

  // One row per word
  state.crosswordWords.forEach((cw, i) => {
    const chip = document.createElement('div');
    chip.className = 'layer-chip';

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.className = 'swatch layer-color';
    colorInput.value = cw.color;
    colorInput.title = cw.word;
    colorInput.addEventListener('input', () => {
      cw.color = colorInput.value;
      applyGridWordStyle(i);
    });

    const lbl = document.createElement('span');
    lbl.className = 'layer-label';
    lbl.textContent = cw.translated ? `${cw.word} — ${cw.translated}` : cw.word;
    lbl.title = `${cw.word} (${gematriaValue(cw.word)})`;

    const opSlider = document.createElement('input');
    opSlider.type = 'range';
    opSlider.min = '0';
    opSlider.max = '100';
    opSlider.value = String(Math.round(cw.opacity * 100));
    opSlider.className = 'layer-opacity cw-opacity-slider';
    opSlider.addEventListener('input', () => {
      cw.opacity = parseInt(opSlider.value, 10) / 100;
      applyGridWordStyle(i);
    });

    chip.appendChild(colorInput);
    chip.appendChild(lbl);
    chip.appendChild(opSlider);
    menu.appendChild(chip);
  });
}

function setupGridWordsMenu() {
  const btn = document.getElementById('grid-words-btn');
  const menu = document.getElementById('grid-words-menu');

  btn.addEventListener('click', () => {
    const open = menu.classList.toggle('open');
    btn.classList.toggle('active', open);
  });

  document.addEventListener('click', (e) => {
    if (!menu.classList.contains('open')) return;
    if (menu.contains(e.target) || btn.contains(e.target)) return;
    if (!document.body.contains(e.target)) return;
    menu.classList.remove('open');
    btn.classList.remove('active');
  });
}

// ---------------------------------------------------------------------
// Grid View dropdown — overlay / side-by-side modes + overlay list
// ---------------------------------------------------------------------
function addOverlayMatch(match, color, label) {
  state.overlayMatches.push({ match, color, label: label || match.label || match.term });
  // Re-center on the overlay match so its letters fall within the grid bounds
  state.currentCenter = centerOfMatch(match);
  renderGrid();
  document.getElementById('grid-view-btn').hidden = false;
  renderGridViewMenu();
}

function removeOverlayAt(index) {
  state.overlayMatches.splice(index, 1);
  if (state.currentCenter !== undefined) renderGrid();
  renderGridViewMenu();
}

function clearOverlays() {
  state.overlayMatches = [];
  if (state.currentCenter !== undefined) renderGrid();
  renderGridViewMenu();
}

function renderGridViewMenu() {
  const menu = document.getElementById('grid-view-menu');
  const btn = document.getElementById('grid-view-btn');
  menu.innerHTML = '';

  const hasGrid = state.currentCenter !== undefined;
  btn.hidden = !hasGrid && state.overlayMatches.length === 0;

  const modes = [
    { key: 'overlay', icon: '⊞', label: 'Overlay (default)', desc: 'All results shown on one grid' },
    { key: 'side-by-side', icon: '⊟', label: 'Side by side', desc: 'Each result in its own panel' },
  ];

  modes.forEach(({ key, icon, label, desc }) => {
    const row = document.createElement('div');
    row.className = `grid-view-option${state.gridViewMode === key ? ' active-mode' : ''}`;
    row.title = desc;
    row.innerHTML = `<span>${icon}</span><span>${label}</span>`;
    row.addEventListener('click', () => {
      state.gridViewMode = key;
      if (state.currentCenter !== undefined) renderGrid();
      renderGridViewMenu();
      menu.classList.remove('open');
      btn.classList.remove('active');
    });
    menu.appendChild(row);
  });

  if (state.overlayMatches.length > 0) {
    const listDiv = document.createElement('div');
    listDiv.className = 'overlay-list';
    const title = document.createElement('div');
    title.className = 'overlay-list-title';
    title.textContent = `Overlaid grids (${state.overlayMatches.length})`;
    listDiv.appendChild(title);

    state.overlayMatches.forEach(({ color, label }, i) => {
      const item = document.createElement('div');
      item.className = 'overlay-item';
      const swatch = document.createElement('div');
      swatch.className = 'overlay-item-swatch';
      swatch.style.background = color;
      const lbl = document.createElement('div');
      lbl.className = 'overlay-item-label';
      lbl.textContent = label;
      const rem = document.createElement('button');
      rem.type = 'button';
      rem.className = 'overlay-item-remove';
      rem.textContent = '✕';
      rem.title = 'Remove from overlay';
      rem.addEventListener('click', (e) => { e.stopPropagation(); removeOverlayAt(i); });
      item.appendChild(swatch);
      item.appendChild(lbl);
      item.appendChild(rem);
      listDiv.appendChild(item);
    });

    const clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.className = 'secondary';
    clearBtn.textContent = 'Clear all overlays';
    clearBtn.style.marginTop = '0.4rem';
    clearBtn.addEventListener('click', clearOverlays);
    listDiv.appendChild(clearBtn);
    menu.appendChild(listDiv);
  }
}

function setupGridViewMenu() {
  const btn = document.getElementById('grid-view-btn');
  const menu = document.getElementById('grid-view-menu');

  btn.addEventListener('click', () => {
    renderGridViewMenu();
    const open = menu.classList.toggle('open');
    btn.classList.toggle('active', open);
  });

  document.addEventListener('click', (e) => {
    if (!menu.classList.contains('open')) return;
    if (menu.contains(e.target) || btn.contains(e.target)) return;
    if (!document.body.contains(e.target)) return;
    menu.classList.remove('open');
    btn.classList.remove('active');
  });
}

// ---------------------------------------------------------------------
// Download grid as JPG
// ---------------------------------------------------------------------
function downloadGridAsJPG() {
  const wrap = document.querySelector('#grid-container .grid-wrap, #grid-container .grid-side-by-side');
  if (!wrap) {
    setStatus('No grid to download — view a grid first.');
    return;
  }
  const table = wrap.querySelector('.els-grid') || document.querySelector('.els-grid');
  if (!table) return;

  const cols = clamp(parseInt(document.getElementById('grid-width').value, 10) || 10, 4, 100);
  const rows = clamp(parseInt(document.getElementById('grid-height').value, 10) || 10, 4, 100);
  const CELL = 56;
  const W = cols * CELL;
  const H = rows * CELL;
  const BG = getComputedStyle(document.documentElement).getPropertyValue('--grid-bg').trim() || '#07090c';
  const BORDER = getComputedStyle(document.documentElement).getPropertyValue('--border').trim() || '#262b33';
  const INK_LIGHT = getComputedStyle(document.documentElement).getPropertyValue('--ink-light').trim() || '#8a93a3';

  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  const allCells = table.querySelectorAll('td');
  allCells.forEach((td, idx) => {
    const r = Math.floor(idx / cols);
    const c = idx % cols;
    const x = c * CELL;
    const y = r * CELL;

    if (td.style.background) {
      ctx.fillStyle = td.style.background;
      ctx.fillRect(x + 1, y + 1, CELL - 1, CELL - 1);
    }

    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 0.5;
    ctx.strokeRect(x + 0.25, y + 0.25, CELL, CELL);

    const letter = td.querySelector('.cell-letter');
    if (letter) {
      ctx.fillStyle = td.classList.contains('hl') ? '#ffffff' : INK_LIGHT;
      ctx.font = `${td.classList.contains('hl') ? 'bold' : ''} ${Math.round(CELL * 0.48)}px serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(letter.textContent, x + CELL / 2, y + CELL * 0.42);
    }

    const num = td.querySelector('.cell-num');
    if (num) {
      ctx.fillStyle = INK_LIGHT;
      ctx.font = `${Math.round(CELL * 0.2)}px sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(num.textContent, x + CELL / 2, y + CELL * 0.77);
    }
  });

  // Render SVG lines on top
  const svgEl = wrap.querySelector('.els-lines');
  if (svgEl) {
    const svgClone = svgEl.cloneNode(true);
    svgClone.setAttribute('xmlns', SVG_NS);
    svgClone.setAttribute('width', W);
    svgClone.setAttribute('height', H);
    const svgStr = new XMLSerializer().serializeToString(svgClone);
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, W, H);
      URL.revokeObjectURL(url);
      finishDownload(canvas);
    };
    img.onerror = () => { URL.revokeObjectURL(url); finishDownload(canvas); };
    img.src = url;
  } else {
    finishDownload(canvas);
  }
}

function finishDownload(canvas) {
  canvas.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `bible-code-grid-${Date.now()}.jpg`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }, 'image/jpeg', 0.92);
}

function setupDownloadBtn() {
  document.getElementById('download-grid-btn').addEventListener('click', downloadGridAsJPG);
}

// ---------------------------------------------------------------------
// Pie chart — shows result counts as a donut chart
// ---------------------------------------------------------------------
function updatePieChart() {
  const wrap = document.getElementById('pie-wrap');
  if (wrap.hidden) return;
  drawPieChart();
}

function drawPieChart() {
  const canvas = document.getElementById('pie-canvas');
  const legend = document.getElementById('pie-legend');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const segments = [];
  const gb = state.gridBounds;

  if (gb) {
    // Grid is visible — count cells per term within the current grid bounds
    const { topLeftIdx, bottomRightIdx, cols } = gb;

    const rows = Math.round((bottomRightIdx - topLeftIdx + 1) / cols);
    const inGrid = (idx) => {
      if (idx < topLeftIdx || idx > bottomRightIdx) return false;
      const rel = idx - topLeftIdx;
      return Math.floor(rel / cols) < rows;
    };

    for (const r of state.allResults) {
      let cellCount = 0;
      for (const m of r.matches) {
        for (const idx of m.indices) {
          if (inGrid(idx)) cellCount++;
        }
      }
      if (cellCount > 0) {
        segments.push({ label: r.label || r.term, count: cellCount, color: r.color });
      }
    }

    // Each crossword word as its own segment
    for (const cw of state.crosswordWords) {
      const cellCount = (cw.cells || []).filter((c) => inGrid(c.idx)).length;
      if (cellCount > 0) {
        segments.push({ label: cw.translated || cw.word, count: cellCount, color: cw.color });
      }
    }
  } else {
    // No grid — fall back to total match counts
    for (const r of state.allResults) {
      if (r.matches.length > 0) {
        segments.push({ label: r.label || r.term, count: r.matches.length, color: r.color });
      }
    }
    for (const cw of state.crosswordWords) {
      segments.push({ label: cw.translated || cw.word, count: (cw.cells || []).length, color: cw.color });
    }
  }

  if (segments.length === 0) {
    ctx.fillStyle = 'rgba(255,255,255,0.12)';
    ctx.beginPath();
    ctx.arc(W / 2, H / 2, W / 2 - 4, 0, Math.PI * 2);
    ctx.fill();
    legend.innerHTML = '<div class="pie-legend-item" style="justify-content:center">' +
      (gb ? 'No results in this grid' : 'Run a search first') + '</div>';
    return;
  }

  const total = segments.reduce((s, seg) => s + seg.count, 0);
  let startAngle = -Math.PI / 2;
  const cx = W / 2, cy = H / 2;
  const outerR = W / 2 - 4;
  const innerR = outerR * 0.48;

  legend.innerHTML = '';

  segments.forEach((seg) => {
    const slice = (seg.count / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(cx + innerR * Math.cos(startAngle), cy + innerR * Math.sin(startAngle));
    ctx.arc(cx, cy, outerR, startAngle, startAngle + slice);
    ctx.arc(cx, cy, innerR, startAngle + slice, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = seg.color;
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    startAngle += slice;

    const pct = Math.round((seg.count / total) * 100);
    const item = document.createElement('div');
    item.className = 'pie-legend-item';
    const sw = document.createElement('div');
    sw.className = 'pie-legend-swatch';
    sw.style.background = seg.color;
    const txt = document.createElement('div');
    txt.className = 'pie-legend-text';
    txt.textContent = `${seg.label}: ${seg.count.toLocaleString()} (${pct}%)`;
    txt.title = txt.textContent;
    item.appendChild(sw);
    item.appendChild(txt);
    legend.appendChild(item);
  });
}

function setupPieChart() {
  const toggleBtn = document.getElementById('pie-toggle-btn');
  const wrap = document.getElementById('pie-wrap');
  const closeBtn = document.getElementById('pie-close-btn');

  toggleBtn.addEventListener('click', () => {
    const show = wrap.hidden;
    wrap.hidden = !show;
    toggleBtn.classList.toggle('active', show);
    if (show) drawPieChart();
  });

  closeBtn.addEventListener('click', () => {
    wrap.hidden = true;
    toggleBtn.classList.remove('active');
  });
}

// ---------------------------------------------------------------------
// Favorites
// ---------------------------------------------------------------------
const FAVORITES_STORAGE_KEY = 'bibleCodeFavorites';

function loadFavorites() {
  try {
    const parsed = JSON.parse(localStorage.getItem(FAVORITES_STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    return [];
  }
}

function saveFavorites() {
  try {
    localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(state.favorites));
  } catch (err) {
    // Storage unavailable (e.g. private browsing) — favorites just won't persist.
  }
}

function favoriteKey(m) {
  return `${m.term}|${m.skip}|${m.start}`;
}

function isFavorited(m) {
  return state.favorites.some((f) => favoriteKey(f) === favoriteKey(m));
}

function addFavorite(m, label, color) {
  if (isFavorited(m)) return;
  state.favorites.push({
    term: m.term,
    label: label || m.label || m.term,
    color,
    skip: m.skip,
    start: m.start,
    indices: m.indices,
    savedAt: Date.now(),
  });
  saveFavorites();
  renderFavorites();
}

function removeFavoriteByMatch(m) {
  state.favorites = state.favorites.filter((f) => favoriteKey(f) !== favoriteKey(m));
  saveFavorites();
  renderFavorites();
}

function removeFavoriteAt(index) {
  state.favorites.splice(index, 1);
  saveFavorites();
  renderFavorites();
}

function viewFavorite(fav) {
  showGrid(fav, fav.color);
  closeDrawer('favorites-drawer');
}

function renderFavorites() {
  const list = document.getElementById('favorites-list');
  list.innerHTML = '';

  if (state.favorites.length === 0) {
    const p = document.createElement('p');
    p.className = 'muted';
    p.textContent = 'No favorites yet — pin a result from the Results list to save it here.';
    list.appendChild(p);
    return;
  }

  state.favorites.forEach((fav, index) => {
    const item = document.createElement('div');
    item.className = 'favorite-item';

    const label = document.createElement('span');
    label.className = 'match-label';
    const startRef = refForIndex(Math.min(...fav.indices));
    const endRef = refForIndex(Math.max(...fav.indices));
    const skipLabel = `${fav.skip > 0 ? '+' : ''}${fav.skip}`;
    const rangeLabel = startRef === endRef
      ? `skip ${skipLabel} — ${startRef}`
      : `skip ${skipLabel} — ${startRef} → ${endRef}`;
    label.textContent = `${fav.label || fav.term} — ${rangeLabel}`;

    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'view-btn';
    viewBtn.textContent = 'View grid';
    viewBtn.addEventListener('click', () => viewFavorite(fav));

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'pin-btn';
    removeBtn.textContent = '★';
    removeBtn.title = 'Remove from favorites';
    removeBtn.setAttribute('aria-label', removeBtn.title);
    removeBtn.addEventListener('click', () => removeFavoriteAt(index));

    const actions = document.createElement('div');
    actions.className = 'match-actions';
    actions.appendChild(viewBtn);
    actions.appendChild(removeBtn);

    item.appendChild(label);
    item.appendChild(actions);
    list.appendChild(item);
  });
}

function setupFavorites() {
  state.favorites = loadFavorites();
  renderFavorites();
}

// ---------------------------------------------------------------------
// Grid history — every grid the user views is recorded here (most recent
// first) so they can jump back to it later, persisted to localStorage.
// ---------------------------------------------------------------------
const GRID_HISTORY_STORAGE_KEY = 'bibleCodeGridHistory';
const MAX_GRID_HISTORY = 25;

function loadGridHistory() {
  try {
    const parsed = JSON.parse(localStorage.getItem(GRID_HISTORY_STORAGE_KEY) || '[]');
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    return [];
  }
}

function saveGridHistory() {
  try {
    localStorage.setItem(GRID_HISTORY_STORAGE_KEY, JSON.stringify(state.gridHistory));
  } catch (err) {
    // Storage unavailable (e.g. private browsing) — history just won't persist.
  }
}

function historyKey(entry) {
  return `${entry.term}|${entry.skip}|${entry.start}`;
}

function addGridHistory(match, color) {
  const entry = {
    term: match.term,
    label: match.label || match.term,
    color,
    skip: match.skip,
    start: match.start,
    indices: match.indices,
    viewedAt: Date.now(),
  };
  state.gridHistory = state.gridHistory.filter((h) => historyKey(h) !== historyKey(entry));
  state.gridHistory.unshift(entry);
  if (state.gridHistory.length > MAX_GRID_HISTORY) state.gridHistory.length = MAX_GRID_HISTORY;
  saveGridHistory();
  renderGridHistory();
}

function clearGridHistory() {
  state.gridHistory = [];
  saveGridHistory();
  renderGridHistory();
}

function renderGridHistory() {
  const list = document.getElementById('history-list');
  const countEl = document.getElementById('history-count');
  const clearBtn = document.getElementById('clear-history-btn');
  list.innerHTML = '';
  countEl.textContent = state.gridHistory.length ? `(${state.gridHistory.length})` : '';
  clearBtn.hidden = state.gridHistory.length === 0;

  if (state.gridHistory.length === 0) {
    const p = document.createElement('p');
    p.className = 'muted';
    p.textContent = 'Grids you view will be listed here for quick access later.';
    list.appendChild(p);
    return;
  }

  state.gridHistory.forEach((entry) => {
    const item = document.createElement('div');
    item.className = 'favorite-item';

    const label = document.createElement('span');
    label.className = 'match-label';
    const startRef = refForIndex(Math.min(...entry.indices));
    const endRef = refForIndex(Math.max(...entry.indices));
    const skipLabel = `${entry.skip > 0 ? '+' : ''}${entry.skip}`;
    const rangeLabel = startRef === endRef
      ? `skip ${skipLabel} — ${startRef}`
      : `skip ${skipLabel} — ${startRef} → ${endRef}`;
    label.textContent = `${entry.label || entry.term} — ${rangeLabel}`;

    const viewBtn = document.createElement('button');
    viewBtn.type = 'button';
    viewBtn.className = 'view-btn';
    viewBtn.textContent = 'View grid';
    viewBtn.addEventListener('click', () => {
      showGrid(entry, entry.color);
      closeDrawer('search-drawer');
    });

    item.appendChild(label);
    item.appendChild(viewBtn);
    list.appendChild(item);
  });
}

function setupGridHistory() {
  state.gridHistory = loadGridHistory();
  renderGridHistory();
  document.getElementById('clear-history-btn').addEventListener('click', clearGridHistory);
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
async function init() {
  setupLayout();
  setupDrawers();
  setupSwipeDrawers();
  setupZoom();
  setupFullscreen();
  setupLineThickness();
  setupLayerMenu();
  setupGridWordsMenu();
  setupGridViewMenu();
  setupTheme();
  setupTermRows();
  setupGematriaCalculator();
  setupCrossword();
  setupFavorites();
  setupGridHistory();
  setupDownloadBtn();
  setupPieChart();
  clearGrid();

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

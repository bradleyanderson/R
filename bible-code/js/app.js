const COLORS = ['#c9a227', '#c0392b', '#2980b9', '#27ae60', '#8e44ad', '#e67e22'];

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
  allResults: [],   // [{ term, color, matches }]
  flatMatches: [],  // every match across all terms
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
    swatch.style.background = COLORS[i % COLORS.length];
    const removeBtn = row.querySelector('.remove-term');
    removeBtn.disabled = rows.length <= 1;
  });
}

function addTermRow(prefillValue) {
  const rows = document.querySelectorAll('.term-row');
  if (rows.length >= 4) return;

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
  input.addEventListener('input', () => sanitizeHebrew(input));
  if (prefillValue) input.value = prefillValue;

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
}

function setupTermRows() {
  addTermRow('תורה');
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
        target.focus();
      }
    });
    container.appendChild(btn);
  });
}

// ---------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------
async function onSearch() {
  const termInputs = Array.from(document.querySelectorAll('.heb-input'));
  const terms = termInputs.map((i) => i.value.trim()).filter(Boolean);

  if (terms.length === 0) {
    setStatus('Enter at least one Hebrew word (2+ letters) to search for.');
    return;
  }
  for (const t of terms) {
    if (t.length < 2) {
      setStatus(`"${t}" is too short — a word needs at least 2 letters.`);
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
  setStatus('Searching…');
  document.getElementById('results').innerHTML = '';
  document.getElementById('grid-container').innerHTML = '';
  document.getElementById('grid-info').textContent = '';

  // Yield so the "Searching..." status paints before the heavy loop.
  await new Promise((r) => setTimeout(r, 0));

  const t0 = performance.now();
  const allResults = [];
  for (let ti = 0; ti < terms.length; ti++) {
    const term = terms[ti];
    const matches = searchELS(term, minSkip, maxSkip, range, 1000);
    matches.sort((a, b) => Math.abs(a.skip) - Math.abs(b.skip) || a.start - b.start);
    const color = COLORS[ti % COLORS.length];
    matches.forEach((m) => { m.color = color; m.term = term; });
    allResults.push({ term, color, matches });
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
    heading.appendChild(document.createTextNode(` ${r.term} — ${r.matches.length.toLocaleString()} match(es)`));
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
function showGrid(match) {
  const center = centerOfMatch(match);
  const suggested = clamp(Math.abs(match.skip) || 10, 5, 60);
  document.getElementById('grid-width').value = suggested;
  document.getElementById('grid-height').value = suggested;
  state.currentCenter = center;
  renderGrid();
  document.getElementById('grid-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderGrid() {
  if (state.currentCenter === undefined) return;
  const cols = clamp(parseInt(document.getElementById('grid-width').value, 10) || 10, 4, 80);
  const rows = clamp(parseInt(document.getElementById('grid-height').value, 10) || 10, 4, 80);
  document.getElementById('grid-width').value = cols;
  document.getElementById('grid-height').value = rows;

  const grid = buildGrid(state.currentCenter, rows, cols);
  const topLeftIdx = grid[0][0].idx;
  const bottomRightIdx = grid[rows - 1][cols - 1].idx;

  const highlightMap = new Map(); // idx -> Set of colors
  for (const m of state.flatMatches) {
    for (const idx of m.indices) {
      if (idx >= topLeftIdx && idx <= bottomRightIdx) {
        if (!highlightMap.has(idx)) highlightMap.set(idx, new Set());
        highlightMap.get(idx).add(m.color);
      }
    }
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

  const container = document.getElementById('grid-container');
  container.innerHTML = '';
  container.appendChild(table);

  const firstIdx = clamp(topLeftIdx, 0, TorahData.text.length - 1);
  const lastIdx = clamp(bottomRightIdx, 0, TorahData.text.length - 1);
  document.getElementById('grid-info').textContent =
    `${rows}×${cols} grid — covers ${refForIndex(firstIdx)} … ${refForIndex(lastIdx)}. ` +
    `Highlighted cells show every searched word that falls inside this view, including ones that cross.`;
}

// ---------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------
async function init() {
  setupKeyboard();
  setupTermRows();
  setupPresets();

  document.getElementById('search-btn').addEventListener('click', onSearch);
  document.getElementById('grid-width').addEventListener('input', renderGrid);
  document.getElementById('grid-height').addEventListener('input', renderGrid);

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

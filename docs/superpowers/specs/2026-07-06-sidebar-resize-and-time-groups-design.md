# Sidebar Resize + Time Category Grouping — Design Spec

**Date**: 2026-07-06
**Status**: Draft, pending user review
**Iteration goal**: Two small frontend-only refinements to the conversation sidebar: (1) let the user drag the right edge of the sidebar to resize its width between 200px and 600px; (2) group conversations by recency using static time-category headers (`TODAY`, `YESTERDAY`, weekday names, and absolute dates for older items).

Both changes are scoped entirely to the existing frontend (`frontend/index.html`, `frontend/static/styles.css`, `frontend/static/app.js`). No backend changes.

---

## 1. Goals & Non-Goals

### Goals

1. **Resizable sidebar**: a 4px-wide hit zone on the right edge of the sidebar can be dragged horizontally to change the sidebar's width. Default 280px, clamped to `[200px, 600px]`. Cursor turns into `col-resize` on hover; subtle cyan glow on the hit zone matches the existing `--accent-cyan` palette.
2. **Time category grouping**: while rendering the conversation list, group items by their `updated_at` value with the labels `TODAY`, `YESTERDAY`, weekday name (last 7 days), `<Month> <Day>` (same year, older), `<Month> <Day>, <Year>` (different year). Items without `updated_at` fall into an `OLDER` bucket.
3. **Group headers always render for the buckets they belong to** — even an empty TODAY bucket is shown (it never actually is empty, but the helper guarantees no-empty-header bugs from the surrounding loop).
4. **Headers are static, non-interactive labels** — small caps, muted, with a top border separator. No collapse, no caret, no click handlers.
5. **No new dependencies**, no new endpoint, no backend model changes.

### Non-Goals

- No width persistence across page reloads (explicit user decision — reset to 280px on reload).
- No `created_at` exposure in the API (we use the already-returned `updated_at` field).
- No collapsible / sticky group headers.
- No date-fns or other library — date formatting uses native `Date.toLocaleDateString`.
- No new frontend test framework. Manual verification only.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Frontend                                   │
│                                                                         │
│  Sidebar (width: 280px default, drag-resizable 200-600px)               │
│  ├── .sidebar-tabs           (Conversations / Library tabs)              │
│  ├── .sidebar-header         (New Chat / batch controls)                │
│  ├── .conversation-list      (scrollable)                               │
│  │   └── TODAY             ← group header (NEW)                         │
│  │       ├── .conversation-item                                         │
│  │       └── .conversation-item                                         │
│  │   └── YESTERDAY         ← group header (NEW)                         │
│  │       └── .conversation-item                                         │
│  │   └── MONDAY                                                    NEW  │
│  │       └── ...                                                        │
│  │   └── MARCH 10                                                  NEW  │
│  │       └── ...                                                        │
│  └── .library-view (unchanged)                                           │
│                                                                         │
│  .sidebar-resizer  ← 4px wide, sits over the sidebar's right border;    │
│                      pointerdown initiates drag, clamped width updated  │
│                      on pointermove, released on pointerup.              │
└─────────────────────────────────────────────────────────────────────────┘
              │                            │
              ▼                            ▼
   GET /api/chat/conversations        (no change — already returns
                                       updated_at on every summary item)
```

---

## 3. Detailed Design

### 3.1 Resizable Sidebar

#### HTML

`frontend/index.html` — after the closing `</div>` of `.sidebar` (currently line 24):

```html
<div class="sidebar-resizer" id="sidebarResizer" aria-label="Resize sidebar"
     role="separator" hidden></div>
```

The `hidden` attribute starts it hidden. It becomes visible once the sidebar is uncollapsed (handled in CSS — `.sidebar:not(.collapsed) .sidebar-resizer { display: block; }`).

#### CSS

`frontend/static/styles.css`:

```css
/* Sidebar min/max bounds — applied together with the existing
   width: 280px default. */
.sidebar {
    min-width: 200px;
    max-width: 600px;
    /* existing rules unchanged: width: 280px, transition, etc. */
}

.sidebar-resizer {
    position: absolute;
    /* Anchor to the sidebar's right edge. Sidebar already has
       position: relative, so absolute positions inside its box. */
    right: -2px;
    top: 0;
    bottom: 0;
    width: 4px;
    cursor: col-resize;
    z-index: 11;        /* above sidebar contents (z=10) */
    border-radius: 2px;
    transition: background 0.15s ease;
}
.sidebar-resizer:hover,
.sidebar-resizer.dragging {
    background: linear-gradient(90deg,
        transparent 0%,
        var(--accent-cyan) 50%,
        transparent 100%);
}
.sidebar.collapsed .sidebar-resizer {
    display: none;       /* hidden until sidebar is uncollapsed */
}
/* Suppress text selection while dragging the handle. */
body.sidebar-dragging {
    cursor: col-resize;
    user-select: none;
}
```

#### JavaScript

`frontend/static/app.js` — new top-level function, called once at module init:

```js
function setupSidebarResizer() {
    const sidebar = document.getElementById('sidebar');
    const handle = document.getElementById('sidebarResizer');
    if (!sidebar || !handle) return;

    // Visible only when the sidebar is not collapsed (CSS hides it
    // otherwise). JS just makes sure the `hidden` attribute matches.
    const sync = () => {
        if (sidebar.classList.contains('collapsed')) {
            handle.setAttribute('hidden', '');
        } else {
            handle.removeAttribute('hidden');
        }
    };
    sync();
    // Existing toggle code adds/removes 'collapsed' — observe instead of
    // patching the toggle function.
    new MutationObserver(sync).observe(sidebar, {
        attributes: true,
        attributeFilter: ['class'],
    });

    let startX = 0, startWidth = 0, dragging = false;

    handle.addEventListener('pointerdown', (e) => {
        startX = e.clientX;
        startWidth = sidebar.getBoundingClientRect().width;
        dragging = true;
        handle.setPointerCapture(e.pointerId);
        handle.classList.add('dragging');
        document.body.classList.add('sidebar-dragging');
    });

    handle.addEventListener('pointermove', (e) => {
        if (!dragging) return;
        const next = Math.min(600, Math.max(200, startWidth + (e.clientX - startX)));
        sidebar.style.width = `${next}px`;
    });

    function endDrag(e) {
        if (!dragging) return;
        dragging = false;
        handle.classList.remove('dragging');
        document.body.classList.remove('sidebar-dragging');
        try { handle.releasePointerCapture(e.pointerId); } catch {}
    }
    handle.addEventListener('pointerup', endDrag);
    handle.addEventListener('pointercancel', endDrag);
}
```

The function is idempotent — safe to call once. Width is not persisted (explicit user decision). On reload, `sidebar.style.width` is empty so the CSS default (`width: 280px`) takes effect.

#### Touchscreen / Pointer Events

PointerEvents handle mouse + touch + pen with one listener set. `setPointerCapture` keeps the drag continuous even if the cursor leaves the 4px-wide handle mid-drag.

### 3.2 Time Category Grouping

#### Bucketing Function

`frontend/static/app.js` — pure helper, no DOM access:

```js
function bucketLabel(iso, now = new Date()) {
    if (!iso) return 'OLDER';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return 'OLDER';
    const dayStart = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate());
    const days = Math.round((dayStart(now) - dayStart(d)) / 86_400_000);
    if (days === 0) return 'TODAY';
    if (days === 1) return 'YESTERDAY';
    if (days > 1 && days <= 7) {
        return d.toLocaleDateString(undefined, { weekday: 'long' }).toUpperCase();
    }
    if (d.getFullYear() === now.getFullYear()) {
        return d.toLocaleDateString(undefined, { month: 'long', day: 'numeric' })
                 .toUpperCase();
    }
    return d.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
            .toUpperCase();
}
```

Bucket coverage:
- `days === 0` → `TODAY`
- `days === 1` → `YESTERDAY`
- `days 2..7` → weekday name (`MONDAY`, `TUESDAY`, etc.)
- `days > 7`, same year → `<Month> <Day>` (`MARCH 10`)
- `days > 7`, different year → `<Month> <Day>, <Year>` (`MARCH 10, 2025`)
- missing / unparseable `updated_at` → `OLDER` (also rendered for any oldest-bucket fallback)

#### Render Loop Change

`loadPairsList()` (currently `frontend/static/app.js:756-803`) ends with:

```js
Object.values(pairs).sort(...).forEach(addPairToList);
```

Replace with a bucketing loop:

```js
const sortedPairs = Object.values(pairs).sort(
    (a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')
);
let currentLabel = null;
for (const pair of sortedPairs) {
    const label = bucketLabel(pair.updated_at);
    if (label !== currentLabel) {
        currentLabel = label;
        const hdr = document.createElement('div');
        hdr.className = 'time-group-header';
        hdr.textContent = label;
        conversationList.appendChild(hdr);
    }
    addPairToList(pair);
}
```

The existing sort already orders by `updated_at` descending, so group headers come out in chronological order naturally — `TODAY` first, `OLDER` last. No re-sorting needed.

#### CSS

`frontend/static/styles.css`:

```css
.time-group-header {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    opacity: 0.7;
    padding: 12px 12px 6px 12px;
    margin-top: 4px;
    border-top: 1px solid var(--border-color);
    text-transform: uppercase;
}
.time-group-header:first-child {
    border-top: none;
    margin-top: 0;
}
```

`text-transform` is set in CSS (not on the JS string) so any future human-readable labels stay uppercased — but we keep the JS uppercase for non-CSS-aware consumers and explicit intent.

#### Edge Cases

- **Empty conversation list**: loop never enters — no headers render.
- **No `updated_at`**: bucket returns `OLDER`, conversations cluster under one `OLDER` header.
- **Multiple conversations same minute in TODAY**: they all live under a single `TODAY` header — `currentLabel` does not change.
- **Conversations strictly between yesterday and 7 days ago**: get a weekday header — e.g. on Wednesday, a Monday conversation shows `MONDAY`.
- **Year boundary**: `Jan 1 2026` and `Dec 31 2025` get different header formats (`JANUARY 1` vs `DECEMBER 31, 2025`).
- **Time zone**: `Date` is parsed as UTC (ISO 8601 with `Z`) then formatted in browser local time. The `dayStart` helper strips time-of-day so day boundaries align with the user's wall clock.

### 3.3 No Backend Changes

- `GET /api/chat/conversations` (already returns `updated_at`) — untouched.
- `ConversationSummary` Pydantic model — untouched.
- `file_storage.get_conversation_list()` — untouched.
- No new dependency in `requirements.txt`.

---

## 4. Testing

This iteration is frontend-only. The project has no JS test framework (only `pytest` for backend, per `pyproject.toml` `testpaths = ["backend/tests", "scripts/tests"]`). We rely on **manual browser verification** with a checklist.

### Resize — manual checklist

| Step | Expected |
|------|----------|
| 1. Load page | Sidebar visible at default 280px width. Resizer handle visible on right edge with hover glow. |
| 2. Drag right ~150px | Sidebar grows to ~430px; cursor stays `col-resize` during drag; text on page is not selected. |
| 3. Continue dragging past 320px more | Width clamps at 600px; further drag right has no effect. |
| 4. Drag left past 80px | Width clamps at 200px. |
| 5. Release mouse | Width sticks at the dropped position. `body.sidebar-dragging` class removed. |
| 6. Reload page | Width resets to 280px (no persistence). |
| 7. Click `#toggleSidebarMain` to collapse | Sidebar slides off-screen; resizer hides. |
| 8. Click toggle again to expand | Sidebar returns; resizer re-appears; previous dragged width is lost (resets to 280px on reload). |

### Time grouping — manual checklist

Use conversations with varied `updated_at` to exercise each bucket:
1. Start a fresh chat → `TODAY`.
2. Use `Touch` on `conversations.json` to backdate one conversation to `2026-07-05T10:00:00Z` (yesterday) → `YESTERDAY`.
3. Backdate one to `2026-07-02T10:00:00Z` (Thursday, assuming today is Friday `2026-07-06`) → `THURSDAY`.
4. Backdate one to `2026-03-10T10:00:00Z` (same year) → `MARCH 10`.
5. Backdate one to `2025-12-15T10:00:00Z` → `DECEMBER 15, 2025`.
6. Backdate one to a date without `updated_at` (or remove the field) → `OLDER`.

Reload sidebar; verify the headers appear in order `TODAY → YESTERDAY → THURSDAY → MARCH 10 → DECEMBER 15, 2025 → OLDER`, with all six conversations under their correct header.

### Regression

After both features ship:

- `pytest backend/tests/ -v` — confirm all 175 tests still pass. (No backend code touched, so this should pass by construction.)
- The `ConversationSummary` model shape is unchanged; frontend code that depends on `pair.updated_at` is unchanged.

---

## 5. Files Touched

| File | Change | Approx. size |
|------|--------|--------------|
| `frontend/index.html` | Add `<div class="sidebar-resizer" id="sidebarResizer" hidden>` after the sidebar div | +1 element |
| `frontend/static/styles.css` | Add `.sidebar-resizer` + `.sidebar-dragging` styles; add min/max on `.sidebar`; add `.time-group-header` styles | +35 lines |
| `frontend/static/app.js` | Add `bucketLabel()` helper + modify `loadPairsList()` to emit headers; add `setupSidebarResizer()` and call at init | +55 lines |

Total: ~90 lines added. No deletions. No file renamed. No migration needed.

---

## 6. Open Questions

None — all decisions resolved during brainstorming:
- Width persistence: NO (resets on reload).
- Width bounds: 200–600px.
- Time key: `updated_at`.
- Bucket labels: TODAY / YESTERDAY / weekday / `<Month> <Day>` / `<Month> <Day>, <Year>` / `OLDER`.
- Header style: static, non-interactive.
- Empty buckets: render their header when seen.
- Tests: manual UI checklist; no new test framework.

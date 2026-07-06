# Sidebar Resize + Time Category Grouping — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the conversation sidebar drag-resizable (200–600px, no persistence across reloads) and group conversations by recency into static time-category headers (`TODAY`, `YESTERDAY`, weekday name, absolute dates).

**Architecture:** Two frontend-only refinements. A 4px-wide `.sidebar-resizer` div sits on the right edge of the sidebar; PointerEvents drag it to update `sidebar.style.width`, clamped to `[200, 600]`. The existing `loadPairsList()` render loop is split into a bucketing loop that emits a `.time-group-header` between groups derived from each pair's `updated_at`. No backend, no new dependencies, no new test framework — verification is a manual browser checklist captured in the spec.

**Tech Stack:** Plain JS (PointerEvents), native CSS, native `Date.toLocaleDateString` for time-bucket labels. No new libraries.

---

## File Map

- **Modify:** `frontend/index.html` — add resizer `<div>` after the sidebar (one element).
- **Modify:** `frontend/static/styles.css` — resizer + body-dragging classes, sidebar min/max bounds, `.time-group-header` rules.
- **Modify:** `frontend/static/app.js` — `bucketLabel()` helper, modified `loadPairsList()` render loop, new `setupSidebarResizer()` invoked at module init.

Total: ~90 lines added across the three files. No deletions. No file rename.

---

## Task 1: Add the sidebar resizer hit zone in `index.html`

**Files:**
- Modify: `frontend/index.html:24` (append a single `<div>` after the `.sidebar` closing tag).

- [ ] **Step 1: Add the resizer div**

Open `frontend/index.html` and locate the closing `</div>` of the `.sidebar` element. The block currently looks like (with the comment stripped):

```html
    <div class="sidebar" id="sidebar">
        <div class="sidebar-tabs" id="sidebarTabs">
            <button class="sidebar-tab active" data-tab="conversations">Conversations</button>
            <button class="sidebar-tab" data-tab="library" id="libraryTabBtn" hidden>Library</button>
        </div>
        <div class="sidebar-header" id="sidebarHeader"></div>
        <div class="conversation-list" id="conversationList"></div>
        <div class="library-view" id="libraryView" hidden></div>
    </div>
```

Add the resizer element right after the closing `</div>` on line 24:

```html
    <div class="sidebar-resizer" id="sidebarResizer" aria-label="Resize sidebar"
         role="separator" hidden></div>
```

The full block becomes:

```html
    <div class="sidebar" id="sidebar">
        <div class="sidebar-tabs" id="sidebarTabs">
            <button class="sidebar-tab active" data-tab="conversations">Conversations</button>
            <button class="sidebar-tab" data-tab="library" id="libraryTabBtn" hidden>Library</button>
        </div>
        <div class="sidebar-header" id="sidebarHeader"></div>
        <div class="conversation-list" id="conversationList"></div>
        <div class="library-view" id="libraryView" hidden></div>
    </div>
    <div class="sidebar-resizer" id="sidebarResizer" aria-label="Resize sidebar"
         role="separator" hidden></div>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add sidebar resizer hit zone"
```

---

## Task 2: Style the resizer + sidebar bounds in `styles.css`

**Files:**
- Modify: `frontend/static/styles.css`
  - The existing `.sidebar` block lives at lines 46–55.
  - Append new rules near the end of the sidebar CSS section.

- [ ] **Step 1: Add min/max bounds to `.sidebar`**

Open `frontend/static/styles.css` and find the `.sidebar` rule at line 46. Extend it to include bounds. The full updated rule reads:

```css
        .sidebar {
            width: 280px;
            min-width: 200px;
            max-width: 600px;
            background: var(--bg-secondary);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            transition: margin-left 0.3s;
            position: relative;
            z-index: 10;
        }
```

Only `min-width: 200px;` and `max-width: 600px;` were added. Everything else is unchanged (the existing rule already declares `position: relative`, so the absolute-positioned resizer anchors correctly).

- [ ] **Step 2: Append the resizer + dragging styles**

At the end of the sidebar CSS block (right after the existing `.sidebar.collapsed { margin-left: -280px; }` rule), append:

```css
        .sidebar-resizer {
            position: absolute;
            right: -2px;
            top: 0;
            bottom: 0;
            width: 4px;
            cursor: col-resize;
            z-index: 11;
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
            display: none;
        }
        body.sidebar-dragging {
            cursor: col-resize;
            user-select: none;
        }
```

Note that `.collapsed` already has the rule `.sidebar.collapsed { margin-left: -280px; }`; we do not redefine that. We only hide the resizer while collapsed.

`hidden` attribute (set in HTML) is honored when JS hasn't yet cleared it; once `setupSidebarResizer()` runs, the `MutationObserver` in Task 4 keeps `hidden` synchronized with the `collapsed` class anyway.

- [ ] **Step 3: Commit**

```bash
git add frontend/static/styles.css
git commit -m "feat(frontend): style sidebar resizer + clamp sidebar width"
```

---

## Task 3: Add `.time-group-header` CSS

**Files:**
- Modify: `frontend/static/styles.css` (append after the resizer block from Task 2).

- [ ] **Step 1: Append the time-group-header rule set**

Append immediately after the `body.sidebar-dragging` rule:

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

The `:first-child` selector suppresses the top border on the very first header (there's nothing above it to separate from). All other headers get a thin top border in `var(--border-color)` for visual separation between groups.

- [ ] **Step 2: Commit**

```bash
git add frontend/static/styles.css
git commit -m "feat(frontend): style .time-group-header"
```

---

## Task 4: Add `bucketLabel()` helper in `app.js`

**Files:**
- Modify: `frontend/static/app.js` — insert the helper as a top-level function near the top of the file (above the first `document.getElementById('sidebarTabs')` / first DOMContentLoaded block).

- [ ] **Step 1: Locate a good insertion point**

Open `frontend/static/app.js`. The file is module-style (`<script type="module">`). The current top of the file has header comments at lines 4–18, then a `// DOM elements ---...` block at line 19 with `const sidebar = document.getElementById('sidebar');` at line 20.

Insert `bucketLabel` *before* the DOM elements block — right after the closing block comment on line 18 and before `// DOM elements ---------------------------------------------------------------` on line 19. That keeps the helper close to the top of the file, where pure helpers belong, and well-separated from the DOM-touching code below.

- [ ] **Step 2: Add the helper**

Insert the following function verbatim:

```javascript
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

Why each branch:
- `!iso || isNaN(d.getTime())` — defensive: covers `null`, `undefined`, or garbage strings. Returns `'OLDER'` so corrupt records don't crash rendering.
- `days === 0` / `=== 1` — same day / day-before boundary. `Math.round` absorbs DST transitions (~23h or ~25h days) so the boundary doesn't drift one hour.
- `days > 1 && <= 7` — weekday label like `MONDAY` (uppercased to match the other bucket labels' visual style).
- Same year, older than 7 days — `MARCH 10` style. `<Month> <Day>` literal with the day number, no comma.
- Different year — `MARCH 10, 2025` with year appended. `toLocaleDateString` adds the comma for the year form.

The `now` parameter has a default so callers don't need to pass it — but it is injectable, which makes the function devtools-testable.

- [ ] **Step 3: Commit**

```bash
git add frontend/static/app.js
git commit -m "feat(frontend): add bucketLabel helper for time grouping"
```

---

## Task 5: Modify `loadPairsList()` to emit group headers

**Files:**
- Modify: `frontend/static/app.js` — within `loadPairsList()` (currently around lines 756–803).

- [ ] **Step 1: Find the existing render loop**

Read `frontend/static/app.js` around line 793–799. The tail of `loadPairsList()` ends with:

```javascript
    Object.values(pairs).sort((a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')).forEach(addPairToList);
```

(Or two statements — `const sorted = Object.values(pairs).sort(...); sorted.forEach(addPairToList);` — depending on which refactor is in place. The exact shape doesn't matter for this task; the next step handles both.)

- [ ] **Step 2: Replace the trailing sort-foreach with the bucketing loop**

Replace the existing tail of `loadPairsList()` so the final two statements read:

```javascript
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

Use the same `conversationList` reference the existing code uses (it's already in scope inside `loadPairsList`). If the existing code defined a local `const conversationList = document.getElementById('conversationList');` near the top of the function, reuse that name; if it queried via `document.getElementById(...)` inline, the new code does the same.

`addPairToList` is unchanged — it appends a `.conversation-item` div. The new code interleaves a `.time-group-header` div between groups.

- [ ] **Step 3: Verify the loop handles empty and missing-timestamp inputs**

After the edit, the function should still:
- Render nothing if `sortedPairs` is empty (the `for...of` simply doesn't iterate).
- Group conversations with `null`/`undefined`/`garbage` `updated_at` under a single `OLDER` header at the bottom (because `''.localeCompare('')` returns `0`, they all sort to the end, and `bucketLabel` returns `'OLDER'` for them).

Read the function once end-to-end to confirm; do not edit anything else.

- [ ] **Step 4: Commit**

```bash
git add frontend/static/app.js
git commit -m "feat(frontend): group conversations by time category in sidebar"
```

---

## Task 6: Add `setupSidebarResizer()` and call it at init

**Files:**
- Modify: `frontend/static/app.js` — append the function definition and invoke it once at module init.

- [ ] **Step 1: Append the resizer function right after `bucketLabel`**

Insert the following verbatim into `frontend/static/app.js`, immediately after the closing brace of `bucketLabel` from Task 4. Both functions are top-level and live in the "pure helpers" zone of the file (above the `// DOM elements ---` block).

```javascript
function setupSidebarResizer() {
    const sidebar = document.getElementById('sidebar');
    const handle = document.getElementById('sidebarResizer');
    if (!sidebar || !handle) return;

    const sync = () => {
        if (sidebar.classList.contains('collapsed')) {
            handle.setAttribute('hidden', '');
        } else {
            handle.removeAttribute('hidden');
        }
    };
    sync();
    new MutationObserver(sync).observe(sidebar, {
        attributes: true,
        attributeFilter: ['class'],
    });

    let startX = 0;
    let startWidth = 0;
    let dragging = false;

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
        try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
    }
    handle.addEventListener('pointerup', endDrag);
    handle.addEventListener('pointercancel', endDrag);
}
```

Notes:
- `setupSidebarResizer()` is idempotent but only runs once. Calling it twice would add duplicate `pointerdown` listeners; that's why we only call it from the init block, not from anywhere else.
- The early `return` if either element is missing handles the case where the script is loaded on a page that hasn't yet rendered the sidebar (e.g. test harnesses).
- `setPointerCapture` keeps the drag continuous even if the cursor moves outside the 4px hit zone. `releasePointerCapture` can throw if the pointer isn't captured (e.g. on `pointercancel`); the `try/catch` swallows that.
- The MutationObserver is the cheapest place to hook into the `collapsed` toggle — the existing toggle code in app.js adds/removes the `collapsed` class on `#sidebar`, but we don't have to find or patch it.

- [ ] **Step 2: Invoke `setupSidebarResizer()` once at module scope**

The file runs top-down (no `DOMContentLoaded` wrapper, no async IIFE init). One-time setup happens at file scope, similar to the existing `showSourcesToggle` block near the bottom of the file. Append a single line right after `setupSidebarResizer`'s closing brace:

```javascript
setupSidebarResizer();
```

That call attaches the listeners. Place it after the function definition (i.e. immediately after the closing `}` you just pasted), separated by a blank line for readability.

- [ ] **Step 3: Commit**

```bash
git add frontend/static/app.js
git commit -m "feat(frontend): wire up sidebar drag-resize"
```

---

## Task 7: Manual browser verification

No automated tests ship with this iteration — the spec calls out a manual checklist. Use Chrome DevTools (or any browser with a console) at `http://localhost:8080`.

**Files:**
- Read: `docs/superpowers/specs/2026-07-06-sidebar-resize-and-time-groups-design.md` §4 (testing checklists).

- [ ] **Step 1: Verify the resize feature**

With the dev server running (`uvicorn backend.main:app --reload --port 8080`):

1. Load `http://localhost:8080`. Sidebar visible at 280px.
2. Hover over the right edge → cyan glow appears.
3. Drag right by ~150px → sidebar should reach ~430px. Cursor stays `col-resize` during drag. No text is selected on the page.
4. Continue dragging right → sidebar caps at 600px.
5. Drag left → caps at 200px.
6. Release mouse. Width sticks at dropped position.
7. Reload (`F5`) → sidebar is back at 280px (no persistence — by design).
8. Click the `☰` toggle (`#toggleSidebarMain`) to collapse → sidebar slides off-screen, resizer disappears.
9. Click again to expand → sidebar returns at 280px (reload reset), resizer reappears.

If steps 1–9 all pass, the resize feature is done.

- [ ] **Step 2: Verify the time grouping feature**

The simplest way to exercise every bucket is to seed conversations.json with varied timestamps. With the dev server stopped (or the file untouched while reload reads it), open `storage/conversations.json` and ensure it has at least one conversation per bucket:

- One with `updated_at = <today's ISO>`. If you just sent a message, you already have one.
- One with `updated_at = <yesterday's ISO>`. Backdate by editing the file: change `"updated_at": "<now>"` to `"updated_at": "<yesterday 14:00>"`.
- One with `updated_at = "2026-07-02T14:00:00Z"` (a Thursday, 4 days before today `2026-07-06`) → expect `THURSDAY`.
- One with `updated_at = "2026-03-10T14:00:00Z"` (same year, older than 7 days) → expect `MARCH 10`.
- One with `updated_at = "2025-12-15T14:00:00Z"` (different year) → expect `DECEMBER 15, 2025`.
- (Optional) one with the `updated_at` field removed → expect `OLDER`.

Reload the page. Confirm headers render in chronological order: `TODAY → YESTERDAY → THURSDAY → MARCH 10 → DECEMBER 15, 2025 → OLDER`, with the conversations correctly grouped under each. Confirm headers are static (no click handler), muted, and have a thin top border separating them.

- [ ] **Step 3: Run the backend test suite as a regression check**

Even though no backend code changed, the iter-9 expansions rely on a green suite. From the repo root:

```bash
PYTHONPATH=. pytest backend/tests/ -q
```

Expected: 175 passed (the same count as before this iteration). Backend should not be affected, so any change in test count is a signal something leaked across module boundaries.

- [ ] **Step 4: Final commit (no-op if no leftover changes)**

If `git status --short` shows any leftover scratch work (e.g. an unbackdated conversation, a stray console.log), reset those before committing. If the repo is clean, there's nothing to commit for Task 7.

If anything was modified while verifying (e.g. seeded conversations you don't want to keep), revert with `git checkout storage/conversations.json`. Don't commit test fixtures into git.

---

## Done

When all six tasks plus the verification checklist pass, this iteration is complete. Sidebar resizes cleanly between 200 and 600 px; conversations are visibly grouped by recency; the backend suite still shows 175 green; nothing else regressed.

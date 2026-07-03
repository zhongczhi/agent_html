import * as smd from "https://cdn.jsdelivr.net/npm/streaming-markdown/smd.min.js";
import { cache } from './cache.js';

// Side-by-side comparison layout (RAG iteration 7). The chat UI has two
// message columns that share a single input box. One Send click fires
// parallel POSTs: vanilla column with retrieval=null, RAG column (when
// enabled) with retrieval={library, uploads, top_k}. Both columns stream
// their responses concurrently, so the user can directly compare them on
// the same question.
//
// Storage & sidebar model: each comparison pair has a base UUID. The two
// columns map to `<base>-0` (vanilla) and `<base>-1` (rag) inside the
// backend's `conversations.json`. The SIDEBAR shows ONE row per pair (the
// base UUID, no column suffix) — clicking a pair loads both columns.
// Storage grouping happens client-side by stripping the `-0`/`-1` suffix.
// (User feedback: "one conversation in the conversation list, in which
// there will be two panels".)

// DOM elements ---------------------------------------------------------------
const sidebar = document.getElementById('sidebar');
const sidebarHeader = document.getElementById('sidebarHeader');
const conversationList = document.getElementById('conversationList');
const toggleSidebarMain = document.getElementById('toggleSidebarMain');
const sendButton = document.getElementById('sendButton');
const messageInput = document.getElementById('messageInput');
const compareGrid = document.getElementById('compareGrid');
const vanillaMessagesEl = document.getElementById('vanillaMessages');
const ragMessagesEl = document.getElementById('ragMessages');
const ragColumn = document.getElementById('ragColumn');

// Modal elements
const confirmModal = document.getElementById('confirmModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalCancelBtn = document.getElementById('modalCancelBtn');
const modalConfirmBtn = document.getElementById('modalConfirmBtn');

// Base UUID + per-column conversation IDs ----------------------------------
// The "active pair" — the two columns visible side-by-side. Both map to
// sub-conversation IDs `<activePairId>-0` (vanilla) and `<activePairId>-1`
// (rag). The active pair is persisted so the user returns to the same
// pair after a reload.
let activePairId = cache.getBaseConversationId();
if (!activePairId) {
    activePairId = crypto.randomUUID();
    cache.setBaseConversationId(activePairId);
    // Seed empty histories for both columns so a refresh shows the pair
    // immediately, even before the user sends anything.
    cache.setHistory(`${activePairId}-0`, []);
    cache.setHistory(`${activePairId}-1`, []);
}

// Sidebar / selection state -------------------------------------------------
// `pairs` is keyed by base UUID, derived from `conversations` (which is
// keyed by full conversation_id) by stripping the -0/-1 suffix.
let conversations = {};
let pairs = {};
let selectionMode = false;
// In pair-selection mode, the selected set holds BASE PAIR IDs, not full
// conversation_ids. Operations translate to both sub-IDs as needed.
let selectedPairIds = new Set();

// Per-column runtime state --------------------------------------------------
// Each column gets its own AbortController so cancelling one stream does
// not affect the other. The columns run their SSE pipelines in parallel.
// `conversationId` is recomputed whenever the active pair changes.
const columns = {
    vanilla: {
        el: vanillaMessagesEl,
        get conversationId() { return `${activePairId}-0`; },
        abortController: null,
        streaming: false,
        active: true,
        pendingInlineFiles: [],
    },
    rag: {
        el: ragMessagesEl,
        get conversationId() { return `${activePairId}-1`; },
        abortController: null,
        streaming: false,
        active: false,  // set by RAG_ENABLED probe
        pendingInlineFiles: [],
    },
};

// Threshold (raw bytes) for the inline-vs-indexed upload decision. Read
// from /api/rag/stats at init so client and server agree.
let inlineContextThresholdBytes = 8192;

// Track rendered equation elements per message div to avoid O(N²) re-rendering
const renderedEquations = new WeakMap();

// Helpers for pair <-> sub-conversation_id mapping ---------------------------
function subIdsFor(pairId) {
    return [`${pairId}-0`, `${pairId}-1`];
}
function baseIdFromSub(subId) {
    if (!subId) return null;
    const m = String(subId).match(/^(.+)-[01]$/);
    return m ? m[1] : null;
}

// Initialize ----------------------------------------------------------------
async function init() {
    // Probe the server to see if RAG is enabled.
    try {
        const resp = await fetch('/api/rag/stats');
        if (resp.ok) {
            const data = await resp.json();
            cache.setRagEnabled(true);
            columns.rag.active = true;
            ragColumn.classList.remove('rag-disabled');
            // Show the Library tab button (it was hidden by default).
            const libBtn = document.getElementById('libraryTabBtn');
            if (libBtn) libBtn.hidden = false;
            if (typeof data.inline_context_threshold_bytes === 'number') {
                inlineContextThresholdBytes = data.inline_context_threshold_bytes;
            }
        } else {
            cache.setRagEnabled(false);
            columns.rag.active = false;
            ragColumn.classList.add('rag-disabled');
            compareGrid.classList.add('rag-disabled');
        }
    } catch {
        cache.setRagEnabled(false);
        columns.rag.active = false;
        ragColumn.classList.add('rag-disabled');
        compareGrid.classList.add('rag-disabled');
    }

    // Render the sidebar first. loadColumn may block for the full duration
    // of an in-flight stream (it awaits the SSE resume), so doing the
    // sidebar first matches the old single-panel order and keeps the
    // conversation list visible during streaming.
    await loadPairsList();
    await Promise.all([
        loadColumn('vanilla'),
        columns.rag.active ? loadColumn('rag') : Promise.resolve(),
    ]);
}

async function loadColumn(channel) {
    const col = columns[channel];
    if (!col.active) return;

    const convId = col.conversationId;
    const cached = cache.getHistory(convId);
    if (cached) {
        renderMessagesFromCache(col, cached);
    } else {
        try {
            const resp = await fetch(`/api/chat/history/${convId}`);
            const data = await resp.json();
            cache.setHistory(convId, data.messages);
            renderMessagesFromCache(col, data.messages);
        } catch (e) {
            console.error(`Failed to load ${channel} history:`, e);
        }
    }
    if (cache.isStreaming(convId)) {
        // Streaming: keep the chunks cache so renderCachedChunks inside
        // resumeStreamFromPosition can replay the partial assistant message
        // that was already streamed before the page reload. processStreamResponse
        // clears chunks on data.end, so we don't lose the cleanup.
        await checkStreamStatus(col);
    } else {
        cache.clearChunks(convId);
    }
}

function renderMessagesFromCache(col, messages) {
    col.el.innerHTML = '';
    if (!messages || messages.length === 0) {
        col.el.classList.add('empty');
        const placeholder = document.createElement('div');
        placeholder.className = 'empty-placeholder';
        placeholder.textContent = 'No messages yet';
        col.el.appendChild(placeholder);
        return;
    }
    col.el.classList.remove('empty');

    for (const msg of messages) {
        if (msg.role === 'assistant') {
            const messageDiv = addMessage(col, msg.role, '');
            if (msg.thinking) {
                const thinkingContent = messageDiv.querySelector('.thinking-content');
                if (thinkingContent) {
                    thinkingContent.textContent = msg.thinking;
                    updateThinkingDisplay(messageDiv);
                }
            }
            const contentDiv = messageDiv.querySelector('.message-content');
            if (contentDiv) {
                const renderer = smd.default_renderer(contentDiv);
                const parser = smd.parser(renderer);
                DOMPurify.sanitize(msg.content);
                smd.parser_write(parser, msg.content);
                smd.parser_end(parser);
                applyLaTeX(contentDiv);
            }
        } else {
            addMessage(col, msg.role, msg.content);
        }
    }
}

// Stream resume -------------------------------------------------------------
async function checkStreamStatus(col) {
    const convId = col.conversationId;
    if (!convId || cache.getStreaming(convId) === 'false') return;

    try {
        const response = await fetch(`/api/chat/stream/status/${convId}`);
        const status = await response.json();
        if (status.status === 'active' || cache.isStreaming(convId)) {
            showStreamingBadge(true);
            return await resumeStreamFromPosition(col, cache.getConsumed(convId));
        } else if (status.status === 'completed' || status.status === 'failed' || status.status === 'none') {
            // Backend is the source of truth: the stream is over. Clear
            // both the local streaming flag AND the history cache — the
            // history may be stale (missing the assistant message) if
            // the original fetch was aborted mid-stream and we never
            // received `data.end`. The backend has the full history;
            // loadColumn will refetch on its next call.
            cache.setStreaming(convId, false);
            cache.clearHistory(convId);
            showStreamingBadge(false);
        }
    } catch (error) {
        console.error('Failed to check stream status:', error);
    }
}

function renderCachedChunks(col) {
    const convId = col.conversationId;
    const chunksCache = cache.getChunks(convId);
    let assistantMessage = null;
    let rawContent = '';
    let renderer = null, parser = null;

    for (const data of chunksCache) {
        if (data.type === 'user') continue;
        if (data.chunk) {
            if (!assistantMessage) {
                assistantMessage = addMessage(col, 'assistant', '');
            }
            if (data.type === 'thinking') {
                const thinkingElement = assistantMessage.querySelector('.thinking-content');
                if (thinkingElement) {
                    thinkingElement.textContent += data.chunk;
                    updateThinkingDisplay(assistantMessage);
                }
            } else if (data.type === 'token') {
                rawContent += data.chunk;
                const contentDiv = assistantMessage.querySelector('.message-content');
                if (contentDiv) {
                    const result = renderContent(contentDiv, data.chunk, renderer, parser);
                    if (result) [renderer, parser] = result;
                }
            }
        }
    }
    col.el.scrollTop = col.el.scrollHeight;
    return { assistantMessage, rawContent, renderer, parser };
}

async function resumeStreamFromPosition(col, consumedCount) {
    const convId = col.conversationId;

    const cached = cache.getHistory(convId);
    if (cached) renderMessagesFromCache(col, cached);
    const { assistantMessage: cachedAssistant, rawContent: cachedRawContent, renderer: cachedRenderer, parser: cachedParser } = renderCachedChunks(col);

    cache.setStreaming(convId, true);
    showStreamingBadge(true);
    col.abortController = new AbortController();

    try {
        const url = `/api/chat/stream/${convId}?from_pointer=${consumedCount}`;
        const response = await fetch(url, { signal: col.abortController.signal });
        if (!response.ok) {
            throw new Error(`Stream request failed: ${response.status}`);
        }
        await processStreamResponse(col, response, cachedAssistant, cachedRawContent, cachedRenderer, cachedParser);
    } catch (error) {
        console.error('Stream error:', error);
        return false;
    }
}

function showStreamingBadge(show) {
    const pairId = activePairId;
    const activeItem = document.querySelector(`.conversation-item[data-id="${pairId}"]`);
    if (!activeItem) return;
    let badge = activeItem.querySelector('.streaming-badge');
    if (show && !badge) {
        badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'Streaming';
        activeItem.appendChild(badge);
    } else if (!show && badge) {
        // Two-panel: vanilla and RAG columns can finish at different times.
        // Before removing the badge, confirm that no sub-conversation of
        // the active pair is still streaming — otherwise a fast-finishing
        // column would clear the badge out from under a still-streaming one.
        const anyStreaming = subIdsFor(pairId).some(id => cache.isStreaming(id));
        if (!anyStreaming) badge.remove();
    }
}

// Send message --------------------------------------------------------------
async function sendMessage() {
    if (selectionMode) return;

    const message = messageInput.value.trim();
    if (!message) return;

    const activeColumns = Object.values(columns).filter(c => c.active);
    if (activeColumns.some(c => c.streaming || cache.isStreaming(c.conversationId))) return;

    messageInput.value = '';
    messageInput.style.height = '';
    sendButton.disabled = true;

    await Promise.all(activeColumns.map(col => sendToColumn(col, message)));

    await loadPairsList();

    sendButton.disabled = false;
    messageInput.focus();
}

async function sendToColumn(col, message) {
    const convId = col.conversationId;
    const isRag = col === columns.rag;

    addMessage(col, 'user', message);
    cache.appendToHistory(convId, { role: 'user', content: message });
    const assistantMessage = addAssistantPlaceholder(col);
    col.el.scrollTop = col.el.scrollHeight;

    cache.appendToChunks(convId, { type: 'user', content: message });
    cache.setConsumed(convId, 0);
    cache.setStreaming(convId, true);
    showStreamingBadge(true);
    col.streaming = true;

    const inlineFiles = col.pendingInlineFiles.splice(0);

    col.abortController = new AbortController();

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                conversation_id: convId,
                retrieval: isRag ? { library: true, uploads: true, top_k: 4 } : null,
                uploaded_files: inlineFiles.length > 0 ? inlineFiles : null,
            }),
            signal: col.abortController.signal,
        });
        if (!response.ok) throw new Error('Failed to get response');

        // Refresh sidebar now that the backend has the conversation in
        // storage. Done before processStreamResponse so the new pair shows
        // up immediately rather than after the full stream finishes. Both
        // columns run in parallel; whichever fetch returns first triggers
        // the refresh, the second is a harmless re-render of the same data.
        await loadPairsList();

        await processStreamResponse(col, response, assistantMessage);
    } catch (error) {
        if (error.name !== 'AbortError') {
            if (assistantMessage && assistantMessage.parentNode) {
                assistantMessage.remove();
            }
            addMessage(col, 'assistant', 'Sorry, an error occurred. Please try again.');
            console.error('Error:', error);
        } else {
            if (assistantMessage && assistantMessage.parentNode) {
                assistantMessage.remove();
            }
        }
    } finally {
        col.streaming = false;
        showStreamingBadge(false);
    }
}

async function processStreamResponse(col, response, existingMessage = null, existingRawContent = '', existingRenderer = null, existingParser = null) {
    const convId = col.conversationId;
    if (!convId) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = existingMessage;
    if (!assistantMessage) {
        assistantMessage = addMessage(col, 'assistant', '');
    }
    const thinkingElement = assistantMessage.querySelector('.thinking-content');
    let consumedCount = cache.getConsumed(convId);
    let rawContent = existingRawContent;
    let thinkingContent = thinkingElement ? thinkingElement.textContent : '';
    let sseBuffer = '';
    let renderer = existingRenderer, parser = existingParser;

    function isScrolledToBottom(threshold = 50) {
        return col.el.scrollHeight - col.el.scrollTop - col.el.clientHeight <= threshold;
    }

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        sseBuffer += chunk;
        const events = sseBuffer.split('\n\n');
        sseBuffer = events.pop() || '';

        for (const event of events) {
            if (!event.startsWith('data: ')) continue;
            try {
                const data = JSON.parse(event.slice(6));

                if (data.chunk) {
                    if (data.type === 'thinking') {
                        if (thinkingElement) {
                            thinkingElement.textContent += data.chunk;
                            thinkingContent += data.chunk;
                            updateThinkingDisplay(assistantMessage);
                        }
                    } else if (data.type === 'sources') {
                        try {
                            const ev = JSON.parse(data.chunk);
                            renderSourcesBlock(assistantMessage, ev.sources || []);
                        } catch (e) {
                            console.warn('Failed to parse sources chunk', e);
                        }
                    } else if (data.type === 'token') {
                        const wasPinnedToBottom = isScrolledToBottom();
                        const contentDiv = assistantMessage.querySelector('.message-content');
                        if (contentDiv) {
                            if (contentDiv.classList.contains('loading')) {
                                contentDiv.innerHTML = '';
                            }
                            contentDiv.classList.remove('loading');
                        }
                        rawContent += data.chunk;
                        if (contentDiv) {
                            [renderer, parser] = renderContent(contentDiv, data.chunk, renderer, parser);
                        }
                        if (wasPinnedToBottom) {
                            col.el.scrollTop = col.el.scrollHeight;
                        }
                    }
                    cache.appendToChunks(convId, data);
                    consumedCount++;
                    cache.setConsumed(convId, consumedCount);
                } else if (data.end) {
                    cache.clearConsumed(convId);
                    cache.setStreaming(convId, false);
                    if (assistantMessage) {
                        const contentDiv = assistantMessage.querySelector('.message-content');
                        const thinkingSection = assistantMessage.querySelector('.thinking-section');
                        const thinkingContentEl = thinkingSection?.querySelector('.thinking-content');
                        if (contentDiv) {
                            const finalContent = rawContent.trim();
                            if (finalContent) {
                                end_parser(parser);
                            } else {
                                contentDiv.innerHTML = '';
                            }
                        }
                        if (thinkingContentEl && thinkingSection) {
                            const thinkingText = thinkingContentEl.textContent.trim();
                            if (!thinkingText) thinkingSection.style.display = 'none';
                        }
                    }
                    cache.appendToHistory(convId, {
                        role: 'assistant',
                        content: rawContent.trim(),
                        thinking: thinkingContent || ''
                    });
                    cache.clearChunks(convId);
                    return true;
                }
            } catch (e) {
                console.error('Chunk processing error:', e, 'Event was:', event);
            }
        }
    }
}

// Message rendering helpers -------------------------------------------------
function applyLaTeX(div) {
    const rendered = renderedEquations.get(div) || new Set();
    div.querySelectorAll('equation-inline').forEach(el => {
        if (!rendered.has(el)) {
            katex.render(el.textContent, el, { displayMode: false, throwOnError: false });
            rendered.add(el);
        }
    });
    div.querySelectorAll('equation-block').forEach(el => {
        if (!rendered.has(el)) {
            katex.render(el.textContent, el, { displayMode: true, throwOnError: false });
            rendered.add(el);
        }
    });
    renderedEquations.set(div, rendered);
}

function renderContent(div, content, renderer = null, parser = null) {
    if (!content || !content.trim()) return '';
    if (!(renderer && parser)) {
        renderer = smd.default_renderer(div);
        parser = smd.parser(renderer);
    }
    DOMPurify(content);
    smd.parser_write(parser, content);
    applyLaTeX(div);
    return [renderer, parser];
}

function end_parser(parser) {
    if (parser) smd.parser_end(parser);
}

function addMessage(col, role, content) {
    const wrapperDiv = document.createElement('div');
    wrapperDiv.className = `message-wrapper ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (role === 'assistant') {
        avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="2" y="8" width="20" height="12" rx="2"/><circle cx="8" cy="14" r="1.5"/><circle cx="16" cy="14" r="1.5"/><path d="M9 18h6"/><path d="M12 2v2"/></svg>`;
    } else {
        avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cx="8" r="4"/><path d="M4 20c0-4 4-6 8-6s8 2 8 6"/></svg>`;
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'message-body';

    if (role === 'assistant') {
        const thinkingSection = document.createElement('div');
        thinkingSection.className = 'thinking-section';
        const thinkingContent = document.createElement('div');
        thinkingContent.className = 'thinking-content';
        thinkingContent.textContent = '';
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'thinking-toggle';
        toggleBtn.textContent = 'Show more';
        toggleBtn.style.display = 'none';
        thinkingSection.appendChild(thinkingContent);
        thinkingSection.appendChild(toggleBtn);

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        bodyDiv.appendChild(thinkingSection);
        bodyDiv.appendChild(contentDiv);
    } else {
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        bodyDiv.appendChild(contentDiv);
    }

    messageDiv.appendChild(bodyDiv);
    wrapperDiv.appendChild(avatar);
    wrapperDiv.appendChild(messageDiv);
    col.el.appendChild(wrapperDiv);
    col.el.scrollTop = col.el.scrollHeight;
    col.el.classList.remove('empty');
    const placeholder = col.el.querySelector('.empty-placeholder');
    if (placeholder) placeholder.remove();

    if (role === 'assistant') {
        setupScrollbarAutoHide(messageDiv);
    }
    return messageDiv;
}

function addAssistantPlaceholder(col) {
    const messageDiv = addMessage(col, 'assistant', '');
    const contentDiv = messageDiv.querySelector('.message-content');
    if (contentDiv) {
        contentDiv.classList.add('loading');
        contentDiv.innerHTML = '<span>Thinking</span><div class="loading-dots"><span></span><span></span><span></span></div>';
    }
    return messageDiv;
}

function renderSourcesBlock(assistantMessageEl, sources) {
    if (!assistantMessageEl || !sources || sources.length === 0) return;
    // Show-sources toggle (iter-8 Phase F): if OFF, don't add the block to
    // the DOM at all. Pre-existing blocks (from earlier chunks in this
    // message) are hidden via applySourcesVisibility() instead.
    if (!cache.getShowSources()) return;
    const existing = assistantMessageEl.parentElement?.querySelector('.sources-block[data-for="assistant"]');
    if (existing) existing.remove();

    const block = document.createElement('div');
    block.className = 'sources-block';
    block.dataset.for = 'assistant';
    const header = document.createElement('div');
    header.className = 'sources-header';
    header.textContent = `Sources (${sources.length})`;
    block.appendChild(header);
    for (const s of sources) {
        const row = document.createElement('div');
        row.className = 'src-row';
        const scope = document.createElement('span');
        scope.className = 'src-scope';
        scope.textContent = `[${s.scope || '?'}]`;
        row.appendChild(scope);
        const filename = document.createElement('span');
        filename.textContent = ` ${s.filename || '?'}`;
        row.appendChild(filename);
        if (s.excerpt) {
            const excerpt = document.createElement('span');
            excerpt.textContent = ` — ${s.excerpt.slice(0, 200)}${s.excerpt.length > 200 ? '…' : ''}`;
            row.appendChild(excerpt);
        }
        block.appendChild(row);
    }
    assistantMessageEl.parentElement?.insertBefore(block, assistantMessageEl);
}

function updateThinkingDisplay(messageElement) {
    const thinkingSection = messageElement.querySelector('.thinking-section');
    if (!thinkingSection) return;
    const thinkingContent = thinkingSection.querySelector('.thinking-content');
    const toggleBtn = thinkingSection.querySelector('.thinking-toggle');
    if (!thinkingContent || !toggleBtn) return;

    const lines = countLines(thinkingContent);
    const shouldCollapse = lines > 3;
    if (shouldCollapse) {
        thinkingSection.classList.add('thinking-collapsed');
        toggleBtn.style.display = 'block';
        toggleBtn.textContent = 'Show more';
    } else {
        thinkingSection.classList.remove('thinking-collapsed');
        toggleBtn.style.display = 'none';
    }
    toggleBtn.onclick = () => {
        const isCollapsed = thinkingSection.classList.contains('thinking-collapsed');
        if (isCollapsed) {
            thinkingSection.classList.remove('thinking-collapsed');
            toggleBtn.textContent = 'Show less';
        } else {
            thinkingSection.classList.add('thinking-collapsed');
            toggleBtn.textContent = 'Show more';
        }
    };
}

function countLines(element) {
    const style = window.getComputedStyle(element);
    const lineHeight = parseFloat(style.lineHeight);
    return Math.round(element.scrollHeight / lineHeight);
}

function setupScrollbarAutoHide(messageElement) {
    if (!messageElement) return;
    const hasOverflow = messageElement.scrollHeight > messageElement.clientHeight;
    if (!hasOverflow) return;
    messageElement.addEventListener('wheel', function() {
        this.classList.add('scrollbar-visible');
        clearTimeout(this._hideTimer);
        this._hideTimer = setTimeout(() => this.classList.remove('scrollbar-visible'), 3000);
    }, { passive: true });
}

// Sidebar / pair management -------------------------------------------------
// Pairs are derived from the flat conversation list by stripping the -0/-1
// suffix. The pair's title and message_count are aggregated from both
// sub-conversations (vanilla is the visible title; rag contributes to the
// count so the badge reflects total activity).
async function loadPairsList() {
    try {
        const response = await fetch('/api/chat/conversations');
        const data = await response.json();
        conversations = {};
        pairs = {};
        conversationList.innerHTML = '';

        // 1. Index by full conversation_id
        for (const conv of data.conversations) {
            conversations[conv.conversation_id] = conv;
        }

        // 2. Group by base pair id
        for (const conv of data.conversations) {
            const baseId = baseIdFromSub(conv.conversation_id);
            if (!baseId) continue;  // skip non-pair conversations (legacy)
            if (!pairs[baseId]) {
                pairs[baseId] = {
                    pair_id: baseId,
                    title: conv.title || 'New chat',
                    message_count: 0,
                    updated_at: conv.updated_at,
                    sub_ids: new Set(),
                };
            }
            const p = pairs[baseId];
            p.sub_ids.add(conv.conversation_id);
            p.message_count += conv.message_count || 0;
            // Use the most recently updated sub's title; its first message
            // is the most natural title for the pair as a whole.
            if ((conv.updated_at || '') > (p.updated_at || '')) {
                p.updated_at = conv.updated_at;
                p.title = conv.title || p.title;
            }
        }

        // 3. Render pair list, sorted by updated_at desc
        const sortedPairs = Object.values(pairs).sort(
            (a, b) => (b.updated_at || '').localeCompare(a.updated_at || '')
        );
        for (const p of sortedPairs) {
            addPairToList(p);
        }
    } catch (error) {
        console.error('Failed to load pairs:', error);
    }
}

function addPairToList(pair) {
    const div = document.createElement('div');
    let classes = 'conversation-item';
    if (pair.pair_id === activePairId) classes += ' active';
    if (selectionMode && selectedPairIds.has(pair.pair_id)) classes += ' selected';
    div.className = classes;
    div.dataset.id = pair.pair_id;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'title';
    titleSpan.textContent = pair.title || 'New chat';
    div.appendChild(titleSpan);

    if (!selectionMode) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deletePair(pair.pair_id);
        };
        div.appendChild(deleteBtn);
    }

    // Streaming badge: show if either sub-conversation is mid-stream
    const anyStreaming = Array.from(pair.sub_ids).some(id => cache.isStreaming(id));
    if (anyStreaming) {
        const badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'Streaming';
        div.appendChild(badge);
    }

    if (selectionMode) {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'select-checkbox';
        checkbox.checked = selectedPairIds.has(pair.pair_id);
        checkbox.addEventListener('click', (e) => e.stopPropagation());
        checkbox.addEventListener('change', () => toggleSelection(pair.pair_id));
        div.appendChild(checkbox);
    }

    div.onclick = () => {
        if (selectionMode) {
            toggleSelection(pair.pair_id);
        } else {
            switchToPair(pair.pair_id);
        }
    };

    conversationList.appendChild(div);
}

function renderSidebarHeader() {
    sidebarHeader.innerHTML = '';
    if (selectionMode) {
        const wrap = document.createElement('div');
        wrap.className = 'selection-header';
        const count = document.createElement('div');
        count.className = 'selection-count';
        count.textContent = `${selectedPairIds.size} selected`;
        wrap.appendChild(count);
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'cancel-btn';
        cancelBtn.dataset.action = 'cancel-selection';
        cancelBtn.textContent = 'Cancel';
        wrap.appendChild(cancelBtn);
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn-confirm';
        deleteBtn.dataset.action = 'confirm-batch-delete';
        deleteBtn.textContent = selectedPairIds.size > 0 ? `Delete (${selectedPairIds.size})` : 'Delete';
        deleteBtn.disabled = selectedPairIds.size === 0;
        wrap.appendChild(deleteBtn);
        sidebarHeader.appendChild(wrap);
    } else {
        const newChatBtn = document.createElement('button');
        newChatBtn.className = 'new-chat-btn';
        newChatBtn.dataset.action = 'new-chat';
        newChatBtn.textContent = '+ New Chat';
        sidebarHeader.appendChild(newChatBtn);
        const batchBtn = document.createElement('button');
        batchBtn.className = 'batch-delete-btn';
        batchBtn.dataset.action = 'batch-delete';
        batchBtn.title = 'Batch delete conversations';
        batchBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path><path d="M9 6V4a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2v2"></path></svg>';
        sidebarHeader.appendChild(batchBtn);
    }
}

function enterSelectionMode() {
    selectionMode = true;
    selectedPairIds = new Set();
    renderSidebarHeader();
    loadPairsList();
}

function exitSelectionMode() {
    selectionMode = false;
    selectedPairIds = new Set();
    renderSidebarHeader();
    loadPairsList();
}

function toggleSelection(pairId) {
    if (selectedPairIds.has(pairId)) selectedPairIds.delete(pairId);
    else selectedPairIds.add(pairId);
    renderSidebarHeader();
    loadPairsList();
}

function showConfirmModal({ title, message, confirmText = 'Confirm', cancelText = 'Cancel', danger = false }) {
    modalTitle.textContent = title;
    modalMessage.innerHTML = '';
    if (typeof message === 'string') {
        modalMessage.textContent = message;
    } else {
        const text = document.createElement('div');
        text.textContent = message.text;
        modalMessage.appendChild(text);
        if (message.list && message.list.length) {
            const ul = document.createElement('ul');
            ul.className = 'modal-list';
            for (const item of message.list) {
                const li = document.createElement('li');
                li.textContent = item;
                ul.appendChild(li);
            }
            modalMessage.appendChild(ul);
        }
    }
    modalCancelBtn.textContent = cancelText;
    modalConfirmBtn.textContent = confirmText;
    modalConfirmBtn.classList.toggle('danger', !!danger);
    confirmModal.hidden = false;

    const focusTarget = danger ? modalCancelBtn : modalConfirmBtn;
    setTimeout(() => focusTarget.focus(), 0);

    return new Promise((resolve) => {
        const cleanup = (result) => {
            confirmModal.hidden = true;
            modalConfirmBtn.removeEventListener('click', onConfirm);
            modalCancelBtn.removeEventListener('click', onCancel);
            confirmModal.removeEventListener('click', onBackdrop);
            document.removeEventListener('keydown', onKeydown);
            resolve(result);
        };
        const onConfirm = () => cleanup(true);
        const onCancel = () => cleanup(false);
        const onBackdrop = (e) => { if (e.target === confirmModal) cleanup(false); };
        const onKeydown = (e) => { if (e.key === 'Escape') cleanup(false); };
        modalConfirmBtn.addEventListener('click', onConfirm);
        modalCancelBtn.addEventListener('click', onCancel);
        confirmModal.addEventListener('click', onBackdrop);
        document.addEventListener('keydown', onKeydown);
    });
}

// Switch the active pair. Both columns reload their histories from the
// new pair's sub-conversations.
async function switchToPair(pairId) {
    if (pairId === activePairId) return;

    // Abort any in-flight streams so they don't keep writing into the
    // columns we're about to reload.
    for (const c of Object.values(columns)) {
        if (c.abortController) c.abortController.abort();
        c.abortController = null;
        c.streaming = false;
    }

    activePairId = pairId;
    cache.setBaseConversationId(pairId);

    // For each sub-conv locally flagged as streaming, probe the backend
    // and reconcile in a single round-trip. checkStreamStatus will:
    //   - resume the stream if the backend still has it active
    //   - clear the flag and history cache if the backend says done/none
    // This MUST run before loadPairsList so the sidebar badge reflects
    // truth. We only probe subs that are flagged as streaming — non-
    // streaming subs skip the round-trip entirely.
    await Promise.all(
        Object.values(columns)
            .filter(c => c.active && cache.isStreaming(c.conversationId))
            .map(c => checkStreamStatus(c))
    );

    document.querySelectorAll('.conversation-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === pairId);
    });
    // Re-render the sidebar so the badge reflects the reconciled state.
    await loadPairsList();

    await Promise.all([
        loadColumn('vanilla'),
        columns.rag.active ? loadColumn('rag') : Promise.resolve(),
    ]);
}

async function startNewChat() {
    // Allocate a fresh base UUID → new pair. Columns reload from empty history.
    for (const c of Object.values(columns)) {
        if (c.abortController) c.abortController.abort();
        c.abortController = null;
        c.streaming = false;
    }
    if (selectionMode) {
        selectionMode = false;
        selectedPairIds = new Set();
        renderSidebarHeader();
    }

    activePairId = crypto.randomUUID();
    cache.setBaseConversationId(activePairId);
    cache.setHistory(`${activePairId}-0`, []);
    cache.setHistory(`${activePairId}-1`, []);

    for (const c of Object.values(columns)) {
        c.el.innerHTML = '';
        c.el.classList.add('empty');
    }
    document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));

    sendButton.disabled = false;
    messageInput.value = '';
    messageInput.style.height = '';
    messageInput.focus();
}

// Delete a pair: both sub-conversations are removed from backend and the
// local state. If we deleted the active pair, create a fresh pair so the
// UI is in a clean state.
async function deletePair(pairId) {
    const pair = pairs[pairId];
    const title = pair?.title || 'this conversation';
    const ok = await showConfirmModal({
        title: 'Delete conversation pair?',
        message: `“${title}” and its side-by-side comparison will be permanently deleted. This cannot be undone.`,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
    });
    if (!ok) return;

    const subIds = Array.from(pair?.sub_ids || subIdsFor(pairId));
    let failed = 0;
    for (const id of subIds) {
        try {
            cache.clearHistory(id);
            cache.clearChunks(id);
            cache.clearConsumed(id);
            cache.clearStreaming(id);
            await fetch(`/api/chat/conversation/${id}`, { method: 'DELETE' });
            delete conversations[id];
        } catch (error) {
            console.error('Failed to delete sub-conversation', id, error);
            failed++;
        }
    }

    // Refresh the sidebar list
    await loadPairsList();

    // If we deleted the active pair, start a fresh one so the UI isn't
    // stuck on a deleted conversation.
    if (pairId === activePairId) {
        await startNewChat();
    }
    if (failed > 0) console.warn(`${failed} sub-conversation(s) could not be deleted`);
}

// Batch delete: each selected entry is a pair; delete both subs per pair.
async function confirmBatchDelete() {
    const ids = Array.from(selectedPairIds);
    if (ids.length === 0) return;

    const titles = ids.map(id => pairs[id]?.title || 'Untitled');
    const message = ids.length <= 5
        ? { text: 'The following conversation pairs will be permanently deleted:', list: titles }
        : { text: `${ids.length} conversation pairs will be permanently deleted. This cannot be undone.` };

    const ok = await showConfirmModal({
        title: `Delete ${ids.length} conversation pair${ids.length === 1 ? '' : 's'}?`,
        message,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
    });
    if (!ok) return;

    let failed = 0;
    for (const pairId of ids) {
        const pair = pairs[pairId];
        const subIds = Array.from(pair?.sub_ids || subIdsFor(pairId));
        for (const id of subIds) {
            try {
                cache.clearHistory(id);
                cache.clearChunks(id);
                cache.clearConsumed(id);
                cache.clearStreaming(id);
                await fetch(`/api/chat/conversation/${id}`, { method: 'DELETE' });
                delete conversations[id];
            } catch (error) {
                console.error('Failed to delete sub-conversation', id, error);
                failed++;
            }
        }
    }

    selectionMode = false;
    selectedPairIds = new Set();
    renderSidebarHeader();
    await loadPairsList();

    if (ids.includes(activePairId)) {
        await startNewChat();
    }
    if (failed > 0) console.warn(`${failed} sub-conversation(s) could not be deleted`);
}

// Upload (per-column) -------------------------------------------------------
document.querySelectorAll('.column-header .upload-input').forEach((input) => {
    input.addEventListener('change', async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        const ch = input.dataset.column;
        await uploadFile(ch, file);
        input.value = '';
    });
});

async function uploadFile(channel, file) {
    const col = columns[channel];
    if (!col || !col.active) return;
    const convId = col.conversationId;

    const allowed = ['.md', '.txt', '.pdf', '.html'];
    const dotIdx = file.name.lastIndexOf('.');
    const ext = dotIdx >= 0 ? file.name.slice(dotIdx).toLowerCase() : '';
    if (!allowed.includes(ext)) {
        const status = document.createElement('div');
        status.className = 'upload-status error';
        status.textContent = `Unsupported file type '${ext || '(none)'}'. Allowed: ${allowed.join(', ')}`;
        col.el.appendChild(status);
        col.el.scrollTop = col.el.scrollHeight;
        return;
    }

    const status = document.createElement('div');
    status.className = 'upload-status';
    status.textContent = `Uploading ${file.name} (${formatBytes(file.size)})…`;
    col.el.appendChild(status);
    col.el.scrollTop = col.el.scrollHeight;

    try {
        const form = new FormData();
        form.append('conversation_id', convId);
        form.append('file', file);
        const resp = await fetch('/api/rag/upload', { method: 'POST', body: form });
        if (!resp.ok) {
            let detail = `HTTP ${resp.status}`;
            try {
                const errBody = await resp.json();
                if (errBody && errBody.detail) detail = errBody.detail;
            } catch { /* response wasn't JSON */ }
            status.classList.add('error');
            status.textContent = `Upload failed: ${detail}`;
            return;
        }
        const body = await resp.json();
        if (body.mode === 'inline') {
            col.pendingInlineFiles.push({ filename: body.filename, content: body.content });
            status.classList.add('ok');
            status.textContent = `Attached ${body.filename} (${formatBytes(body.bytes)}) — will send with your next message.`;
        } else {
            status.classList.add('ok');
            status.textContent = `Indexed ${body.filename} (${formatBytes(body.bytes)}, ${body.chunks_added} chunks).`;
        }
    } catch (e) {
        status.classList.add('error');
        status.textContent = `Upload error: ${e}`;
    }
}

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// Event wiring --------------------------------------------------------------
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
messageInput.addEventListener('input', function() { autoResizeInput(this); });
toggleSidebarMain.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

sidebarHeader.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === 'new-chat') startNewChat();
    else if (action === 'batch-delete') enterSelectionMode();
    else if (action === 'cancel-selection') exitSelectionMode();
    else if (action === 'confirm-batch-delete') confirmBatchDelete();
});

function autoResizeInput(textarea) {
    if (typeof CSS !== 'undefined' && CSS.supports('field-sizing', 'content')) return;
    const minHeight = 120;
    const clone = textarea.cloneNode();
    clone.style.position = 'absolute';
    clone.style.visibility = 'hidden';
    clone.style.width = textarea.offsetWidth + 'px';
    clone.style.height = 'auto';
    clone.style.minHeight = '0';
    document.body.appendChild(clone);
    const newHeight = Math.max(minHeight, clone.scrollHeight);
    document.body.removeChild(clone);
    textarea.style.height = newHeight + 'px';
}

renderSidebarHeader();
init();

// ── Library tab (iter-8 Phase E) ───────────────────────────────────────────

const libraryView = document.getElementById('libraryView');
const sidebarTabsEl = document.getElementById('sidebarTabs');

// State for the library view. Rebuilt on every renderLibraryView() call.
let libraryFilesCache = [];
let libraryStatsCache = null;

function switchSidebarTab(tab) {
    cache.setCurrentSidebarTab(tab);
    applyActiveTab();
}

function applyActiveTab() {
    const tab = cache.getCurrentSidebarTab();
    // Update tab button active states
    sidebarTabsEl.querySelectorAll('.sidebar-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });
    // Show/hide the two content panels
    const isLib = tab === 'library';
    libraryView.hidden = !isLib;
    conversationList.hidden = isLib;
    // The selection-header and conversation pair list both belong to the
    // Conversations tab. When switching to Library, exit selection mode
    // (otherwise the selection UI would be confusing without its target).
    if (isLib && selectionMode) {
        exitSelectionMode();
    }
    if (isLib) {
        renderLibraryView();
    } else {
        // Re-render conversation list when switching back so it isn't stale.
        loadPairsList();
    }
}

async function renderLibraryView() {
    libraryView.innerHTML = '';
    if (!cache.getRagEnabled()) {
        libraryView.textContent = 'Library unavailable: RAG is not enabled.';
        return;
    }
    // Fetch files + stats in parallel
    const [filesResp, statsResp] = await Promise.all([
        fetch('/api/rag/library/files'),
        fetch('/api/rag/stats'),
    ]);
    libraryFilesCache = filesResp.ok ? (await filesResp.json()).files : [];
    libraryStatsCache = statsResp.ok ? await statsResp.json() : null;

    // Header: Upload + Reindex buttons + inline error placeholder
    const header = document.createElement('div');
    header.className = 'library-header';

    const uploadLabel = document.createElement('label');
    uploadLabel.className = 'library-upload-btn';
    uploadLabel.title = 'Upload a file to the library';
    uploadLabel.textContent = 'Upload';
    const uploadInput = document.createElement('input');
    uploadInput.type = 'file';
    uploadInput.accept = '.md,.txt,.pdf,.html,.docx,.csv';
    uploadInput.style.display = 'none';
    uploadInput.addEventListener('change', handleLibraryUpload);
    uploadLabel.appendChild(uploadInput);
    header.appendChild(uploadLabel);

    const reindexBtn = document.createElement('button');
    reindexBtn.className = 'library-reindex-btn';
    reindexBtn.textContent = 'Reindex';
    reindexBtn.addEventListener('click', handleLibraryReindex);
    header.appendChild(reindexBtn);

    const errorMsg = document.createElement('div');
    errorMsg.className = 'library-error';
    errorMsg.id = 'libraryErrorMsg';
    header.appendChild(errorMsg);

    libraryView.appendChild(header);

    // File list
    if (libraryFilesCache.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'library-empty';
        empty.textContent = 'No files in library. Click Upload to add one.';
        libraryView.appendChild(empty);
    } else {
        const list = document.createElement('div');
        list.className = 'library-list';
        for (const f of libraryFilesCache) {
            const row = document.createElement('div');
            row.className = 'library-row';
            row.dataset.filename = f.filename;

            const name = document.createElement('span');
            name.className = 'library-filename';
            name.textContent = f.filename;
            row.appendChild(name);

            const meta = document.createElement('span');
            meta.className = 'library-meta';
            meta.textContent = `${formatBytes(f.size)} · ${formatRelativeTime(f.modified_at)}`;
            row.appendChild(meta);

            const delBtn = document.createElement('button');
            delBtn.className = 'library-delete-btn';
            delBtn.textContent = '×';
            delBtn.title = `Delete ${f.filename}`;
            delBtn.addEventListener('click', () => handleLibraryDelete(f.filename));
            row.appendChild(delBtn);

            list.appendChild(row);
        }
        libraryView.appendChild(list);
    }

    // Stats footer
    const footer = document.createElement('div');
    footer.className = 'library-footer';
    if (libraryStatsCache) {
        const chunks = libraryStatsCache.library_chunks ?? 0;
        const fileCount = libraryStatsCache.library_files ?? libraryFilesCache.length;
        footer.textContent = `${chunks} chunk${chunks === 1 ? '' : 's'} from ${fileCount} file${fileCount === 1 ? '' : 's'}`;
    }
    libraryView.appendChild(footer);
}

function setLibraryError(msg) {
    const el = document.getElementById('libraryErrorMsg');
    if (el) el.textContent = msg || '';
}

async function handleLibraryUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    setLibraryError('');
    const fd = new FormData();
    fd.append('file', file);
    try {
        const resp = await fetch('/api/rag/library/upload', { method: 'POST', body: fd });
        if (!resp.ok) {
            const detail = (await resp.json()).detail || resp.statusText;
            setLibraryError(`Upload failed: ${detail}`);
            return;
        }
    } catch (e) {
        setLibraryError(`Upload failed: ${e.message}`);
        return;
    } finally {
        event.target.value = '';  // allow re-uploading the same file
    }
    await renderLibraryView();
}

async function handleLibraryReindex() {
    setLibraryError('');
    try {
        const resp = await fetch('/api/rag/library/reindex', { method: 'POST' });
        if (!resp.ok) {
            const detail = (await resp.json()).detail || resp.statusText;
            setLibraryError(`Reindex failed: ${detail}`);
            return;
        }
    } catch (e) {
        setLibraryError(`Reindex failed: ${e.message}`);
        return;
    }
    await renderLibraryView();
}

async function handleLibraryDelete(filename) {
    const confirmed = await showConfirmModal({
        title: 'Delete from library?',
        message: `"${filename}" will be removed from the library and the FAISS index will be rebuilt.`,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
    });
    if (!confirmed) return;
    setLibraryError('');
    try {
        const resp = await fetch(`/api/rag/library/file/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        if (!resp.ok) {
            const detail = (await resp.json()).detail || resp.statusText;
            setLibraryError(`Delete failed: ${detail}`);
            return;
        }
    } catch (e) {
        setLibraryError(`Delete failed: ${e.message}`);
        return;
    }
    await renderLibraryView();
}

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRelativeTime(unixSeconds) {
    const t = typeof unixSeconds === 'number' ? unixSeconds * 1000 : Date.now();
    const diff = Date.now() - t;
    if (diff < 60_000) return 'just now';
    if (diff < 3600_000) return `${Math.floor(diff / 60_000)}m ago`;
    if (diff < 86_400_000) return `${Math.floor(diff / 3600_000)}h ago`;
    return `${Math.floor(diff / 86_400_000)}d ago`;
}

// Tab button click handlers
sidebarTabsEl.addEventListener('click', (e) => {
    const btn = e.target.closest('.sidebar-tab');
    if (!btn) return;
    switchSidebarTab(btn.dataset.tab);
});

// Apply the persisted tab on initial render
applyActiveTab();


// ── Show-sources toggle (iter-8 Phase F) ───────────────────────────────────

const showSourcesToggle = document.getElementById('showSourcesToggle');

function applySourcesVisibility() {
    const show = cache.getShowSources();
    // Affect every .sources-block on the page (both columns — the toggle is
    // RAG-only visually, but vanilla column sources are also gated by it).
    document.querySelectorAll('.sources-block').forEach(block => {
        block.style.display = show ? '' : 'none';
    });
}

if (showSourcesToggle) {
    // Initialize from cache (default ON).
    showSourcesToggle.checked = cache.getShowSources();
    showSourcesToggle.addEventListener('change', (e) => {
        cache.setShowSources(e.target.checked);
        applySourcesVisibility();
    });
}
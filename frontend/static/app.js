import * as smd from "https://cdn.jsdelivr.net/npm/streaming-markdown/smd.min.js";
import { cache } from './cache.js';

const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendButton = document.getElementById('sendButton');
const sidebar = document.getElementById('sidebar');
const sidebarHeader = document.getElementById('sidebarHeader');
const conversationList = document.getElementById('conversationList');
const toggleSidebarMain = document.getElementById('toggleSidebarMain');
const channelSwitcher = document.getElementById('channelSwitcher');
const ragUploadBtn = document.getElementById('ragUploadBtn');
const ragUploadInput = document.getElementById('ragUploadInput');

// Modal elements
const confirmModal = document.getElementById('confirmModal');
const modalTitle = document.getElementById('modalTitle');
const modalMessage = document.getElementById('modalMessage');
const modalCancelBtn = document.getElementById('modalCancelBtn');
const modalConfirmBtn = document.getElementById('modalConfirmBtn');

// Channel state (RAG iteration 7). The two channels share a base UUID
// and each derive a distinct conversation_id: `${base}-0` for vanilla,
// `${base}-1` for RAG. The current channel is persisted in cache so the
// user returns to the last channel they were on after a page reload.
let currentChannel = cache.getCurrentChannel();  // 'vanilla' | 'rag'
let baseConversationId = cache.getBaseConversationId();
if (!baseConversationId) {
    baseConversationId = crypto.randomUUID();
    cache.setBaseConversationId(baseConversationId);
    // Initialize both channel conversations so they appear in the sidebar
    // immediately. Without this, only the active channel would have a
    // conversation_id at all, and switching to the inactive channel would
    // create it on the fly (which works, but loses the "easy to identify as
    // a pair" property from the spec).
    cache.setHistory(`${baseConversationId}-0`, []);
    cache.setHistory(`${baseConversationId}-1`, []);
}
const conversationIdFor = (channel) => `${baseConversationId}-${channel === 'rag' ? '1' : '0'}`;

// currentConversationId is the existing global; keep its semantics so the
// rest of app.js (loadConversation, sendMessage, processStreamResponse,
// etc.) continues to work unchanged. It always points at the active
// channel's conversation_id.
let currentConversationId = conversationIdFor(currentChannel);

let conversations = {};
let currentAbortController = null;  // For cancelling ongoing streams
let selectionMode = false;
let selectedConvIds = new Set();

// Track rendered equation elements per div to avoid O(N²) re-rendering
const renderedEquations = new WeakMap();

function retrievalForChannel(channel) {
    if (channel === 'rag') {
        return { library: true, uploads: true, top_k: 4 };
    }
    return null;
}

function setChannel(channel) {
    if (channel !== 'vanilla' && channel !== 'rag') return;
    if (channel === currentChannel) return;
    currentChannel = channel;
    cache.setCurrentChannel(channel);
    currentConversationId = conversationIdFor(channel);
    cache.setCurrentConversationId(currentConversationId);

    // Update the toggle button visuals
    if (channelSwitcher) {
        channelSwitcher.querySelectorAll('.channel-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.channel === channel);
        });
    }
    // Show/hide the upload button (RAG only)
    if (ragUploadBtn) {
        ragUploadBtn.classList.toggle('rag-disabled', channel !== 'rag');
    }

    // Reload the conversation's history and re-render. The existing
    // loadConversation() handles history fetching, SSE-resume check, and
    // empty-state — we reuse it.
    loadConversation(currentConversationId).then(() => {
        loadConversationList();
    });
}

function applyLaTeX(div) {
    const rendered = renderedEquations.get(div) || new Set();
    // Manually render inline math in <equation-inline> tags
    div.querySelectorAll('equation-inline').forEach(el => {
        if (!rendered.has(el)) {
            katex.render(el.textContent, el, {
                displayMode: false,  // Inline mode
                throwOnError: false  // Optional: Don't throw errors on invalid LaTeX
            });
            rendered.add(el);
        }
    });
    // Manually render block math in <equation-block> tags
    div.querySelectorAll('equation-block').forEach(el => {
        if (!rendered.has(el)) {
            katex.render(el.textContent, el, {
                displayMode: true,   // Display (block) mode
                throwOnError: false  // Optional: Don't throw errors on invalid LaTeX
            });
            rendered.add(el);
        }
    });
    renderedEquations.set(div, rendered);
}

function renderContent(div, content, renderer=null, parser=null){
    if (!content || !content.trim()) return '';
    if (!(renderer && parser)){
        renderer = smd.default_renderer(div);
        parser = smd.parser(renderer);
    }
    // not used for now
    // TODO(U2 — known issue): this DOMPurify() call is a no-op — the return
    // value is discarded, so no sanitization is applied to the streamed HTML.
    // The markdown content is written straight into the DOM via smd. If the
    // LLM ever returns untrusted HTML it would render as-is. Either remove
    // this line (it's misleading) or wire up sanitization properly, e.g. by
    // wrapping the smd output through DOMPurify.sanitize() before insertion.
    DOMPurify(content);
    smd.parser_write(parser, content);
    applyLaTeX(div);
    return [renderer, parser];
}

function end_parser(parser){
    if (parser) smd.parser_end(parser);
}

// Initialize
async function init() {
    // Probe the server to see if RAG is enabled. If yes, show the channel
    // switcher and upload button. If no, hide them — the iteration-6 UI
    // is preserved exactly. The probe is fire-and-forget; failure to
    // reach the server is treated as "RAG disabled".
    try {
        const resp = await fetch('/api/rag/stats');
        if (resp.ok) {
            cache.setRagEnabled(true);
            // RAG is enabled: remove the rag-disabled class so the toggle
            // and upload button are visible.
            if (channelSwitcher) channelSwitcher.classList.remove('rag-disabled');
            if (ragUploadBtn) ragUploadBtn.classList.remove('rag-disabled');
            // Active button reflects the persisted channel
            if (channelSwitcher) {
                channelSwitcher.querySelectorAll('.channel-btn').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.channel === currentChannel);
                });
            }
            if (ragUploadBtn) {
                ragUploadBtn.classList.toggle('rag-disabled', currentChannel !== 'rag');
            }
        } else {
            // RAG disabled — apply the rag-disabled class to hide the
            // toggle and upload button. The persisted currentChannel is
            // left alone; the server ignores the retrieval field when
            // RAG is disabled.
            cache.setRagEnabled(false);
            if (channelSwitcher) channelSwitcher.classList.add('rag-disabled');
            if (ragUploadBtn) ragUploadBtn.classList.add('rag-disabled');
        }
    } catch {
        cache.setRagEnabled(false);
        if (channelSwitcher) channelSwitcher.classList.add('rag-disabled');
        if (ragUploadBtn) ragUploadBtn.classList.add('rag-disabled');
    }

    await loadConversationList();

    if (currentConversationId) {
        // If the streaming flag is set, attempt to resume. checkStreamStatus
        // calls resumeStreamFromPosition which renders cached chunks into
        // the DOM. Do NOT fall back to loadConversation afterwards — its
        // renderMessagesFromCache call would wipe the cached chunks from
        // the DOM (the partial assistant message disappears on every
        // refresh during streaming if we fall back here).
        if (cache.isStreaming(currentConversationId)) {
            await checkStreamStatus();
            return;
        }
        // Streaming flag not set — load from history cache.
        await loadConversation(currentConversationId);
    } else {
        messagesContainer.classList.add('empty');
    }
}

// Load conversation list
async function loadConversationList() {
    try {
        const response = await fetch('/api/chat/conversations');
        const data = await response.json();
        conversations = {};
        conversationList.innerHTML = '';

        for (const conv of data.conversations) {
            conversations[conv.conversation_id] = conv;
            addConversationToList(conv);
        }
    } catch (error) {
        console.error('Failed to load conversations:', error);
    }
}

// Add conversation to sidebar list
function addConversationToList(conv) {
    const div = document.createElement('div');
    let classes = 'conversation-item';
    if (conv.conversation_id === currentConversationId) classes += ' active';
    if (selectionMode && selectedConvIds.has(conv.conversation_id)) classes += ' selected';
    div.className = classes;
    div.dataset.id = conv.conversation_id;

    // Layout: title + (× button if not in selection mode) + (streaming
    // badge if streaming) + (checkbox at the back if in selection mode).
    // The per-item × is hidden in selection mode so the user uses the
    // batch Delete (N) button instead.
    const titleSpan = document.createElement('span');
    titleSpan.className = 'title';
    titleSpan.textContent = conv.title || 'New conversation';
    div.appendChild(titleSpan);

    if (!selectionMode) {
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.textContent = '×';
        deleteBtn.onclick = (e) => {
            e.stopPropagation();
            deleteConversation(conv.conversation_id);
        };
        div.appendChild(deleteBtn);
    }

    // Derive streaming badge from the cache so the badge is applied
    // on every sidebar render — including the new-conversation case
    // where the item didn't exist when showStreamingBadge was first called.
    if (cache.isStreaming(conv.conversation_id)) {
        const badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'Streaming';
        div.appendChild(badge);
    }

    if (selectionMode) {
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'select-checkbox';
        checkbox.checked = selectedConvIds.has(conv.conversation_id);
        checkbox.addEventListener('click', (e) => e.stopPropagation());
        checkbox.addEventListener('change', () => toggleSelection(conv.conversation_id));
        div.appendChild(checkbox);
    }

    div.onclick = () => {
        if (selectionMode) {
            toggleSelection(conv.conversation_id);
        } else {
            switchConversation(conv.conversation_id);
        }
    };

    conversationList.appendChild(div);
}

// Render the sidebar header based on selectionMode + selection count.
// Two layouts: normal (New Chat + Batch Delete) or selection (count + Delete + Cancel).
function renderSidebarHeader() {
    sidebarHeader.innerHTML = '';
    if (selectionMode) {
        const wrap = document.createElement('div');
        wrap.className = 'selection-header';

        const count = document.createElement('div');
        count.className = 'selection-count';
        count.textContent = `${selectedConvIds.size} selected`;
        wrap.appendChild(count);

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'cancel-btn';
        cancelBtn.dataset.action = 'cancel-selection';
        cancelBtn.textContent = 'Cancel';
        wrap.appendChild(cancelBtn);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn-confirm';
        deleteBtn.dataset.action = 'confirm-batch-delete';
        deleteBtn.textContent = selectedConvIds.size > 0 ? `Delete (${selectedConvIds.size})` : 'Delete';
        deleteBtn.disabled = selectedConvIds.size === 0;
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
    selectedConvIds = new Set();
    renderSidebarHeader();
    loadConversationList();
}

function exitSelectionMode() {
    selectionMode = false;
    selectedConvIds = new Set();
    renderSidebarHeader();
    loadConversationList();
}

function toggleSelection(convId) {
    if (selectedConvIds.has(convId)) {
        selectedConvIds.delete(convId);
    } else {
        selectedConvIds.add(convId);
    }
    // Re-render the list (to update checkbox + highlight) and header (count).
    renderSidebarHeader();
    loadConversationList();
}

// Show a themed confirmation modal. Returns a Promise<boolean>.
function showConfirmModal({ title, message, confirmText = 'Confirm', cancelText = 'Cancel', danger = false }) {
    modalTitle.textContent = title;
    // message can be a string or { text, list } where list is an array of items.
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

    // Focus: danger actions focus Cancel so Enter doesn't accidentally confirm.
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

// Switch to a conversation
async function switchConversation(convId) {
    if (convId === currentConversationId) return;

    // Abort any ongoing stream for current conversation before switching
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    // Clear displayed messages first
    messagesContainer.innerHTML = '';

    currentConversationId = convId;
    cache.setCurrentConversationId(convId);

    // Update active state in list
    document.querySelectorAll('.conversation-item').forEach(el => {
        el.classList.toggle('active', el.dataset.id === convId);
    });

    let status = await checkStreamStatus();
    // if the conversation is resuming, skip the history logic
    if (status === true) {
        return;
    }
    await loadConversation(convId);
}

// Load conversation history
async function loadConversation(convId) {
    try {
        // Check cache first
        const cached = cache.getHistory(convId);
        if (cached) {
            // Render from cache — no fetch needed
            renderMessagesFromCache(cached);
            return;
        }

        // Fetch from backend
        const response = await fetch(`/api/chat/history/${convId}`);
        const data = await response.json();

        // Store in cache
        cache.setHistory(convId, data.messages);

        // Render
        renderMessagesFromCache(data.messages);

        // remove chunk cache if exist
        cache.clearChunks(convId);
    } catch (error) {
        console.error('Failed to load conversation:', error);
    }
}

// Render messages from cache (or fetched data)
function renderMessagesFromCache(messages) {
    messagesContainer.innerHTML = '';

    for (const msg of messages) {
        if (msg.role === 'assistant') {
            const messageDiv = addMessage(msg.role, '');

            // If there's thinking content, add it
            if (msg.thinking) {
                const thinkingContent = messageDiv.querySelector('.thinking-content');
                if (thinkingContent) {
                    thinkingContent.textContent = msg.thinking;
                    updateThinkingDisplay(messageDiv);
                }
            }

            // Render markdown for content
            const contentDiv = messageDiv.querySelector('.message-content');
            let content = msg.content;
            if (contentDiv) {
                // contentDiv.innerHTML = renderMarkdown(contentDiv.textContent);
                const renderer = smd.default_renderer(contentDiv);
                const parser = smd.parser(renderer);
                // TODO(U2 — known issue): DOMPurify.sanitize(content) is a
                // no-op here too — the return value is discarded, so the
                // cached history's content is rendered without sanitization.
                // Same fix as in renderContent(): wire up sanitization or
                // remove the misleading call.
                DOMPurify.sanitize(content);
                smd.parser_write(parser, content);
                smd.parser_end(parser);
                applyLaTeX(contentDiv);
            }
        } else {
            addMessage(msg.role, msg.content);
        }
    }
}

// Check stream status and resume if needed
async function checkStreamStatus() {
    // if no currentConversationId or streaming == false
    // note: when streaming is not set, should consider when the cache is cleared
    if (!currentConversationId || cache.getStreaming(currentConversationId) === 'false') return;

    try {
        const response = await fetch(`/api/chat/stream/status/${currentConversationId}`);
        const status = await response.json();

        // Resume if stream is active OR if cache is incomplete (streaming flag still true but backend stream done)
        if (status.status === 'active' || cache.isStreaming(currentConversationId)) {
            showStreamingBadge(true);
            return await resumeStreamFromPosition(cache.getConsumed(currentConversationId));
        } else if (status.status === 'completed') {
            showStreamingBadge(false);
        }
    } catch (error) {
        console.error('Failed to check stream status:', error);
    }
}

// Render cached chunks from the cache before resuming stream
// Remerber to render the history message before the cached chunks
function renderCachedChunks(convId) {
    const chunksCache = cache.getChunks(convId);

    let assistantMessage = null;
    let rawContent = '';
    let hasReceivedTokens = false;
    let renderer = null, parser = null;

    for (const data of chunksCache) {
        if (data.type === 'user') {
            // Render user message
            // addMessage('user', data.content);
            continue;
        } else if (data.chunk) {
            if (!assistantMessage) {
                assistantMessage = addMessage('assistant', '');
            }
            if (data.type === 'thinking') {
                const thinkingElement = assistantMessage.querySelector('.thinking-content');
                if (thinkingElement) {
                    thinkingElement.textContent += data.chunk;
                    updateThinkingDisplay(assistantMessage);
                }
            } else if (data.type === 'token') {
                hasReceivedTokens = true;
                rawContent += data.chunk;
                const contentDiv = assistantMessage.querySelector('.message-content');
                if (contentDiv) {
                    const result = renderContent(contentDiv, data.chunk, renderer, parser);
                    if (result) {
                        [renderer, parser] = result;
                    }
                }
            }
        }
    }

    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    return {assistantMessage, rawContent, renderer, parser};
}

// Resume stream from specific position
// case 1: switch off and back to the running conversation
// case 2: refresh the page
async function resumeStreamFromPosition(consumedCount) {
    if (!currentConversationId) return;

    // Load the cache history first
    const cached = cache.getHistory(currentConversationId);
        if (cached) {
            // Render from cache — no fetch needed
            renderMessagesFromCache(cached);
        }

    // First, render cached chunks from the cache
    const { assistantMessage: cachedAssistant, rawContent: cachedRawContent, renderer: cachedRenderer, parser: cachedParser } = renderCachedChunks(currentConversationId);

    // Mark this conversation as streaming (for safety check in processStreamResponse)
    cache.setStreaming(currentConversationId, true);
    showStreamingBadge(true);

    // Create abort controller for this resume stream
    currentAbortController = new AbortController();
    let result = null;

    try {
        // Call stream with single pointer parameter
        const url = `/api/chat/stream/${currentConversationId}?from_pointer=${consumedCount}`;
        const response = await fetch(url, { signal: currentAbortController.signal });
        if (!response.ok) {
            throw new Error(`Stream request failed: ${response.status} ${response.statusText}`);
        }
        // Pass cached assistant message, raw content, and parser/renderer to continue from where we left off
        // so the markdown parser state (tables, code blocks) carries across the cache-replay → live-stream boundary.
        result = await processStreamResponse(response, true, cachedAssistant, cachedRawContent, cachedRenderer, cachedParser);
    } catch (error) {
        // The streaming flag must survive transient fetch failures so the
        // next refresh can retry the resume. In Chromium, a fetch aborted by
        // page navigation surfaces as `TypeError: network error`, NOT
        // `AbortError`, so we cannot distinguish it from a real failure here.
        // Either way: do not clear the flag or badge — only `data.end` and
        // explicit user actions should clear them.
        console.error('Stream error:', error);
        return false;
    }
    if (result === true){
        // Clear streaming state when done
        cache.setStreaming(currentConversationId, false);
        showStreamingBadge(false);
        cache.clearChunks(currentConversationId);
        return true;
    }
}

// Show/hide streaming badge on conversation item
function showStreamingBadge(show) {
    const activeItem = document.querySelector(`.conversation-item[data-id="${currentConversationId}"]`);
    if (!activeItem) return;

    let badge = activeItem.querySelector('.streaming-badge');
    if (show && !badge) {
        badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'Streaming';
        activeItem.appendChild(badge);
    } else if (!show && badge) {
        badge.remove();
    }
}

// Start new chat
async function startNewChat() {
    // Abort any ongoing stream for the current conversation so the user
    // is not stuck waiting for it before they can send in the new chat.
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }

    // Exit selection mode if active
    if (selectionMode) {
        selectionMode = false;
        selectedConvIds = new Set();
        renderSidebarHeader();
    }

    currentConversationId = null;
    cache.setCurrentConversationId(null);
    messagesContainer.innerHTML = '';
    messagesContainer.classList.add('empty');
    document.querySelectorAll('.conversation-item').forEach(el => el.classList.remove('active'));

    // Reset UI state so the user can immediately send in the new conversation
    sendButton.disabled = false;
    messageInput.value = '';
    messageInput.style.height = '';

    messageInput.focus();
}

// Delete conversation
async function deleteConversation(convId) {
    const conv = conversations[convId];
    const title = conv?.title || 'this conversation';
    const ok = await showConfirmModal({
        title: 'Delete conversation?',
        message: `“${title}” will be permanently deleted. This cannot be undone.`,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
    });
    if (!ok) return;

    try {
        // Clear all caches for this conversation
        cache.clearHistory(convId);
        cache.clearChunks(convId);
        cache.clearConsumed(convId);
        cache.clearStreaming(convId);

        await fetch(`/api/chat/conversation/${convId}`, { method: 'DELETE' });

        delete conversations[convId];
        const item = document.querySelector(`.conversation-item[data-id="${convId}"]`);
        if (item) item.remove();

        if (convId === currentConversationId) {
            await startNewChat();
        }
    } catch (error) {
        console.error('Failed to delete conversation:', error);
    }
}

// Confirm and execute batch deletion of selected conversations
async function confirmBatchDelete() {
    const ids = Array.from(selectedConvIds);
    if (ids.length === 0) return;

    const titles = ids.map(id => {
        const c = conversations[id];
        return c?.title || 'Untitled';
    });

    const message = ids.length <= 5
        ? { text: 'The following conversations will be permanently deleted:', list: titles }
        : { text: `${ids.length} conversations will be permanently deleted. This cannot be undone.` };

    const ok = await showConfirmModal({
        title: `Delete ${ids.length} conversation${ids.length === 1 ? '' : 's'}?`,
        message,
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true,
    });
    if (!ok) return;

    const activeDeleted = ids.includes(currentConversationId);
    let failed = 0;
    for (const id of ids) {
        try {
            cache.clearHistory(id);
            cache.clearChunks(id);
            cache.clearConsumed(id);
            cache.clearStreaming(id);
            await fetch(`/api/chat/conversation/${id}`, { method: 'DELETE' });
            delete conversations[id];
        } catch (error) {
            console.error('Failed to delete conversation', id, error);
            failed++;
        }
    }

    // Exit selection mode and refresh the sidebar
    selectionMode = false;
    selectedConvIds = new Set();
    renderSidebarHeader();
    await loadConversationList();

    if (activeDeleted) {
        await startNewChat();
    }
    if (failed > 0) {
        console.warn(`${failed} conversation(s) could not be deleted`);
    }
}

// Send message
async function sendMessage() {
    // Sending messages is not allowed while in batch deletion selection
    // mode — bail without disturbing the user's pending selection.
    // Covers both the Send button click and Enter keypress (both route
    // through sendMessage()).
    if (selectionMode) return;

    const message = messageInput.value.trim();
    if (!message) return;

    if (!currentConversationId) {
        currentConversationId = conversationIdFor(currentChannel);
        cache.setCurrentConversationId(currentConversationId);
    }

    // Check if this conversation is already streaming
    if (cache.isStreaming(currentConversationId)) return;

    messageInput.value = '';
    messageInput.style.height = '';
    sendButton.disabled = true;

    addMessage('user', message);

    // Cache user message in history cache
    cache.appendToHistory(currentConversationId, { role: 'user', content: message });

    // Create assistant message placeholder immediately
    const assistantMessage = addAssistantPlaceholder();
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Cache user message
    cache.appendToChunks(currentConversationId, { type: 'user', content: message });

    // Clear pointer for new stream
    cache.setConsumed(currentConversationId, 0);

    // Mark this conversation as streaming
    cache.setStreaming(currentConversationId, true);
    showStreamingBadge(true);
    let result = null;

    // Create abort controller for this stream
    currentAbortController = new AbortController();

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                conversation_id: currentConversationId,
                // Channel-aware retrieval config: null for vanilla, full
                // config for RAG. The server resolves "retrieval is set
                // AND rag_service is set" to decide whether to retrieve.
                retrieval: retrievalForChannel(currentChannel),
            }),
            signal: currentAbortController.signal
        });

        if (!response.ok) {
            throw new Error('Failed to get response');
        }

        await loadConversationList();
        result = await processStreamResponse(response, false, assistantMessage);

    } catch (error) {
        if (error.name !== 'AbortError') {
            // Remove placeholder on error
            if (assistantMessage && assistantMessage.parentNode) {
                assistantMessage.remove();
            }
            addMessage('assistant', 'Sorry, an error occurred. Please try again.');
            console.error('Error:', error);
        } else {
            if (assistantMessage && assistantMessage.parentNode) {
                assistantMessage.remove();
            }
            result = true;
        }
    }
    if (result === true){
        // Clear streaming state when done (normal completion or error)
        cache.setStreaming(currentConversationId, false);
        showStreamingBadge(false);
        sendButton.disabled = false;
        messageInput.focus();
    }
}

// Process SSE stream response
async function processStreamResponse(response, isResume, existingMessage = null, existingRawContent = '', existingRenderer = null, existingParser = null) {
    // Capture conversationId at start - only process if this conversation is still current
    const convId = currentConversationId;
    if (!convId) {
        return;
    }

    // For non-resume streams, check if this conversation is still marked as streaming
    // For resume, we always process (resume is explicitly requested)
    if (!isResume && !cache.isStreaming(convId)) {
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = existingMessage;
    if (!assistantMessage) {
        assistantMessage = addMessage('assistant', '');
    }
    const thinkingElement = assistantMessage.querySelector('.thinking-content');
    let consumedCount = 0;
    let rawContent = existingRawContent;
    let thinkingContent = thinkingElement ? thinkingElement.textContent : '';
    let sseBuffer = '';  // Accumulator for incomplete SSE events
    // note: this renderer should not be ended during one continuous streaming (no switch or refresh)
    // If a parser/renderer was already created (e.g. by renderCachedChunks during resume),
    // reuse it so markdown state (tables, code blocks) carries across the cache/new boundary.
    let renderer = existingRenderer, parser = existingParser;

    // Returns true if the user is at (or within `threshold` px of) the bottom of the
    // messages container. Used by the per-chunk scroll to decide whether to pin to
    // the latest line during streaming: if the user scrolled up to read earlier content,
    // we leave their scroll alone.
    function isScrolledToBottom(threshold = 50) {
        return messagesContainer.scrollHeight - messagesContainer.scrollTop - messagesContainer.clientHeight <= threshold;
    }

    // Get current consumed count from the cache
    consumedCount = cache.getConsumed(convId);

    while (true) {
        // Check if we're still on the same conversation (user may have switched)
        if (convId !== currentConversationId) {
            // Conversation changed, stop processing this stream but keep it cached for later resume
            return;
        }

        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        sseBuffer += chunk;

        // SSE events are delimited by \n\n (double newline)
        // Split and process complete events, keep remainder in buffer
        const events = sseBuffer.split('\n\n');
        sseBuffer = events.pop() || '';  // Keep incomplete event in buffer

        for (const event of events) {
            if (!event.startsWith('data: ')) continue;
            try {
                const data = JSON.parse(event.slice(6));

                if (data.chunk) {
                    if (data.type === 'thinking') {
                        // Update thinking section
                        if (thinkingElement) {
                            thinkingElement.textContent += data.chunk;
                            thinkingContent += data.chunk;
                            updateThinkingDisplay(assistantMessage);
                        }
                    } else if (data.type === 'sources') {
                        // RAG channel: the server emits a one-time 'sources'
                        // chunk before the first token. The chunk's text is a
                        // JSON string: {"sources": [{filename, excerpt, scope}, ...]}.
                        // Render a Sources block immediately above the
                        // assistant message, before any tokens.
                        try {
                            const ev = JSON.parse(data.chunk);
                            renderSourcesBlock(assistantMessage, ev.sources || []);
                        } catch (e) {
                            console.warn('Failed to parse sources chunk', e);
                        }
                    } else if (data.type === 'token') {
                        // Capture pinned-to-bottom state BEFORE the DOM update. The
                        // per-chunk height increase can exceed our threshold, which would
                        // cause a post-update check to incorrectly report "not pinned".
                        const wasPinnedToBottom = isScrolledToBottom();

                        // Remove loading indicator when receiving tokens
                        const contentDiv = assistantMessage.querySelector('.message-content');
                        if (contentDiv) {
                            // Clear loading innerHTML only once on first token
                            if (contentDiv.classList.contains('loading')) {
                                contentDiv.innerHTML = '';
                            }
                            contentDiv.classList.remove('loading');
                        }
                        // Accumulate raw content during streaming
                        rawContent += data.chunk;
                        // Use insertAdjacentText for efficient text appending during streaming
                        // (following Chrome best practices - avoids re-parsing entire content)
                        if (contentDiv) {
                            //contentDiv.insertAdjacentText('beforeend', data.chunk);
                            [renderer, parser] = renderContent(contentDiv, data.chunk, renderer, parser);
                        }
                        // Only auto-scroll to the newest line if the user was already
                        // pinned to the bottom before this chunk arrived. If they had
                        // scrolled up to read earlier content, leave their scroll alone.
                        // Scrolling back to the bottom re-pins on the next chunk.
                        if (wasPinnedToBottom) {
                            messagesContainer.scrollTop = messagesContainer.scrollHeight;
                        }
                    }

                    // Cache chunk and increment pointer using captured convId
                    cache.appendToChunks(convId, data);
                    consumedCount++;
                    cache.setConsumed(convId, consumedCount);
                } else if (data.end) {
                    // End of stream - clear pointer and streaming state (keep chunks cached)
                    cache.clearConsumed(convId);
                    cache.setStreaming(convId, false);
                    // Final markdown parse and cleanup
                    if (assistantMessage) {
                        const contentDiv = assistantMessage.querySelector('.message-content');
                        const thinkingSection = assistantMessage.querySelector('.thinking-section');
                        const thinkingContentEl = thinkingSection.querySelector('.thinking-content');

                        if (contentDiv) {
                            // contentDiv.classList.remove('loading');
                            // Filter out any thinking content that might have leaked into rawContent
                            const finalContent = rawContent.trim();
                            if (finalContent) {
                                // contentDiv.innerHTML = renderMarkdown(finalContent);
                                end_parser(parser);
                            } else {
                                contentDiv.innerHTML = '';
                            }
                        }

                        // Hide thinking section if it only contains the thinking indicator text
                        if (thinkingContentEl && thinkingSection) {
                            const thinkingText = thinkingContentEl.textContent.trim();
                            // If thinking content is empty or just placeholder, hide the section
                            if (!thinkingText) {
                                thinkingSection.style.display = 'none';
                            }
                        }
                    }
                    // Append assistant message to history cache. The chunk cache is
                    // cleared below — the history cache and the server-side
                    // conversation storage converge on the same final message, so
                    // a page refresh after this point will load via /api/chat/history
                    // instead of resuming from chunks.
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

function addMessage(role, content) {
    // Create wrapper div for avatar + message bubble
    const wrapperDiv = document.createElement('div');
    wrapperDiv.className = `message-wrapper ${role}`;

    // Avatar SVG icon
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    if (role === 'assistant') {
        avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect x="2" y="8" width="20" height="12" rx="2"/><circle cx="8" cy="14" r="1.5"/><circle cx="16" cy="14" r="1.5"/><path d="M9 18h6"/><path d="M12 2v2"/></svg>`;
    } else {
        avatar.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M4 20c0-4 4-6 8-6s8 2 8 6"/></svg>`;
    }

    // Message bubble (the actual content container)
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'message-body';

    if (role === 'assistant') {
        // Create thinking section
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

        // Create message content
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

    // Wrapper contains avatar and bubble as siblings
    wrapperDiv.appendChild(avatar);
    wrapperDiv.appendChild(messageDiv);

    messagesContainer.appendChild(wrapperDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Remove empty class when messages exist
    messagesContainer.classList.remove('empty');

    // Setup scrollbar auto-hide for assistant messages
    if (role === 'assistant') {
        setupScrollbarAutoHide(messageDiv);
    }

    return messageDiv;
}

// Create assistant message placeholder with loading indicator
function addAssistantPlaceholder() {
    const messageDiv = addMessage('assistant', '');
    const contentDiv = messageDiv.querySelector('.message-content');
    if (contentDiv) {
        contentDiv.classList.add('loading');
        contentDiv.innerHTML = '<span>Thinking</span><div class="loading-dots"><span></span><span></span><span></span></div>';
    }
    return messageDiv;
}

// Insert a "Sources" block immediately above the assistant message element.
// Called when the server emits a 'sources' SSE chunk (RAG channel only).
// Idempotent: if a sources block is already present for this assistant
// message, replace it (handles resume-from-cache where sources may replay).
function renderSourcesBlock(assistantMessageEl, sources) {
    if (!assistantMessageEl || !sources || sources.length === 0) return;
    // Remove any existing block (idempotency for cache replay)
    const existing = assistantMessageEl.parentElement?.querySelector(
        '.sources-block[data-for="assistant"]'
    );
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

    // Check if content exceeds 3 lines
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

    // Setup toggle button handler
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
    const height = element.scrollHeight;
    return Math.round(height / lineHeight);
}

function setupScrollbarAutoHide(messageElement) {
    if (!messageElement) return;

    // Check if already has scrollbar (content overflows)
    const hasOverflow = messageElement.scrollHeight > messageElement.clientHeight;
    if (!hasOverflow) return;

    messageElement.addEventListener('wheel', function() {
        this.classList.add('scrollbar-visible');
        clearTimeout(this._hideTimer);
        this._hideTimer = setTimeout(() => {
            this.classList.remove('scrollbar-visible');
        }, 3000);
    }, { passive: true });
}

// Event listeners
sendButton.addEventListener('click', sendMessage);
messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
messageInput.addEventListener('input', function() {
    autoResizeInput(this);
});
toggleSidebarMain.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

// Channel switcher (RAG iteration 7). Click a button to switch the active
// channel; the conversation_id changes, history reloads.
if (channelSwitcher) {
    channelSwitcher.querySelectorAll('.channel-btn').forEach(btn => {
        btn.addEventListener('click', () => setChannel(btn.dataset.channel));
    });
}
// Upload button: only meaningful on the RAG channel. The button label
// triggers the hidden file input.
if (ragUploadBtn && ragUploadInput) {
    ragUploadBtn.addEventListener('click', () => ragUploadInput.click());
    ragUploadInput.addEventListener('change', async (e) => {
        const file = e.target.files?.[0];
        if (!file) return;
        await uploadForRagChannel(file);
        ragUploadInput.value = '';
    });
}

// Upload a file to the current RAG channel's conversation. Posts to
// /api/rag/upload with the active conversation_id. Status messages are
// rendered as small italic lines in the chat (not stored in history).
async function uploadForRagChannel(file) {
    if (currentChannel !== 'rag') return;
    const status = document.createElement('div');
    status.className = 'upload-status';
    status.textContent = `Uploading ${file.name}…`;
    messagesContainer.appendChild(status);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    try {
        const form = new FormData();
        form.append('conversation_id', currentConversationId);
        form.append('file', file);
        const resp = await fetch('/api/rag/upload', { method: 'POST', body: form });
        if (!resp.ok) {
            status.classList.add('error');
            status.textContent = `Upload failed: HTTP ${resp.status}`;
            return;
        }
        const body = await resp.json();
        status.classList.add('ok');
        status.textContent = `Indexed ${body.chunks_added} chunks from ${body.filename}.`;
    } catch (e) {
        status.classList.add('error');
        status.textContent = `Upload error: ${e}`;
    }
}

// Sidebar header: event delegation for the dynamically-rendered buttons
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
    // Check if field-sizing is supported
    if (typeof CSS !== 'undefined' && CSS.supports('field-sizing', 'content')) {
        return;
    }

    // Fallback: adjust height based on content
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

// Initialize
renderSidebarHeader();
init();

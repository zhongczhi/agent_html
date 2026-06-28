// Compare-page JS — two chat panels sharing one input. Independent histories.
// Vanilla panel: no retrieval. RAG panel: full retrieval + sources + upload.
//
// Conversation ID convention: a single random base on page load, with `-0`
// for the vanilla panel and `-1` for the RAG panel. This makes the pair easy
// to identify and delete together.

const SHARED_INPUT = document.getElementById('sharedInput');
const SHARED_SEND = document.getElementById('sharedSend');
const VANILLA_MESSAGES = document.getElementById('vanillaMessages');
const RAG_MESSAGES = document.getElementById('ragMessages');
const VANILLA_CONV_LABEL = document.getElementById('vanillaConvIdLabel');
const RAG_CONV_LABEL = document.getElementById('ragConvIdLabel');
const SHOW_SOURCES = document.getElementById('showSources');
const RAG_UPLOAD = document.getElementById('ragUpload');

const BASE_ID = crypto.randomUUID();
const VANILLA_CONV_ID = `${BASE_ID}-0`;
const RAG_CONV_ID = `${BASE_ID}-1`;

VANILLA_CONV_LABEL.textContent = VANILLA_CONV_ID;
RAG_CONV_LABEL.textContent = RAG_CONV_ID;

const panels = {
    vanilla: {
        convId: VANILLA_CONV_ID,
        messagesEl: VANILLA_MESSAGES,
        retrieval: null,
    },
    rag: {
        convId: RAG_CONV_ID,
        messagesEl: RAG_MESSAGES,
        retrieval: { library: true, uploads: true, top_k: 4 },
    },
};

function appendMessage(panelKey, role, text, opts = {}) {
    const panel = panels[panelKey];
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    panel.messagesEl.appendChild(div);
    panel.messagesEl.scrollTop = panel.messagesEl.scrollHeight;
    return div;
}

function appendSources(panelKey, sources) {
    if (!SHOW_SOURCES.checked) return;
    const panel = panels[panelKey];
    const block = document.createElement('div');
    block.className = 'msg sources-block';
    block.innerHTML = '<strong>Sources:</strong>';
    for (const s of sources) {
        const row = document.createElement('div');
        row.className = 'src-row';
        const scope = document.createElement('span');
        scope.className = 'src-scope';
        scope.textContent = `[${s.scope || '?'}]`;
        row.appendChild(scope);
        row.appendChild(document.createTextNode(` ${s.filename || '?'}: ${s.excerpt || ''}`));
        block.appendChild(row);
    }
    panel.messagesEl.appendChild(block);
    panel.messagesEl.scrollTop = panel.messagesEl.scrollHeight;
}

async function loadHistory(panelKey) {
    const panel = panels[panelKey];
    try {
        const resp = await fetch(`/api/chat/history/${encodeURIComponent(panel.convId)}`);
        if (!resp.ok) return;
        const body = await resp.json();
        panel.messagesEl.innerHTML = '';
        for (const m of (body.messages || [])) {
            appendMessage(panelKey, m.role, m.content);
        }
    } catch (e) {
        // ignore — empty panel is fine
    }
}

async function sendToPanel(panelKey, text) {
    const panel = panels[panelKey];
    appendMessage(panelKey, 'user', text);

    let assistantDiv = null;
    let fullText = '';

    try {
        const resp = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                conversation_id: panel.convId,
                retrieval: panel.retrieval,
            }),
        });
        if (!resp.ok) {
            appendMessage(panelKey, 'error', `HTTP ${resp.status}: ${await resp.text()}`);
            return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        for (;;) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            // SSE: events are separated by blank lines; data is `data: <json>\n\n`.
            let idx;
            while ((idx = buf.indexOf('\n\n')) !== -1) {
                const event = buf.slice(0, idx);
                buf = buf.slice(idx + 2);
                const dataLine = event.split('\n').find(l => l.startsWith('data: '));
                if (!dataLine) continue;
                let payload;
                try { payload = JSON.parse(dataLine.slice(6)); } catch { continue; }
                // SSE payload shape: { chunk: <text>, type: "token"|"thinking"|"sources", message_id: "..." }
                if (payload.chunk !== undefined) {
                    const text = payload.chunk;
                    const type = payload.type;
                    if (type === 'token') {
                        if (!assistantDiv) {
                            assistantDiv = appendMessage(panelKey, 'assistant', '');
                        }
                        fullText += text;
                        // Plain-text rendering only (no smd streaming-markdown here —
                        // kept simple for the compare page; can be added later).
                        assistantDiv.textContent = fullText;
                        panel.messagesEl.scrollTop = panel.messagesEl.scrollHeight;
                    } else if (type === 'sources') {
                        // text is a JSON string of { sources: [...] }
                        try {
                            const ev = JSON.parse(text);
                            appendSources(panelKey, ev.sources || []);
                        } catch { /* ignore malformed sources */ }
                    } else if (type === 'thinking') {
                        const t = document.createElement('div');
                        t.className = 'msg thinking';
                        t.textContent = `💭 ${text}`;
                        panel.messagesEl.appendChild(t);
                        panel.messagesEl.scrollTop = panel.messagesEl.scrollHeight;
                    }
                } else if (payload.end) {
                    return;
                }
            }
        }
    } catch (e) {
        appendMessage(panelKey, 'error', String(e));
    }
}

async function sendBoth() {
    const text = SHARED_INPUT.value.trim();
    if (!text) return;
    SHARED_INPUT.value = '';
    SHARED_SEND.disabled = true;
    try {
        // Fire both in parallel — independent streams, independent histories.
        await Promise.all([
            sendToPanel('vanilla', text),
            sendToPanel('rag', text),
        ]);
    } finally {
        SHARED_SEND.disabled = false;
        SHARED_INPUT.focus();
    }
}

SHARED_SEND.addEventListener('click', sendBoth);
SHARED_INPUT.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendBoth();
    }
});

RAG_UPLOAD.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('conversation_id', RAG_CONV_ID);
    form.append('file', file);
    appendMessage('rag', 'thinking', `Uploading ${file.name}…`);
    try {
        const resp = await fetch('/api/rag/upload', { method: 'POST', body: form });
        if (!resp.ok) {
            appendMessage('rag', 'error', `Upload failed: HTTP ${resp.status}`);
            return;
        }
        const body = await resp.json();
        appendMessage('rag', 'thinking', `Indexed ${body.chunks_added} chunks from ${body.filename}.`);
    } catch (err) {
        appendMessage('rag', 'error', String(err));
    } finally {
        RAG_UPLOAD.value = '';
    }
});

// On load: pull history for both panels (empty for new IDs)
loadHistory('vanilla');
loadHistory('rag');
SHARED_INPUT.focus();

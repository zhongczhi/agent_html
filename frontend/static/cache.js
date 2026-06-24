// Owns every localStorage access in the frontend. Callers (app.js) see
// structured data (arrays, numbers, booleans) — never raw strings, never
// JSON.parse / JSON.stringify, never the localStorage keys themselves.
//
// This is the only module that touches localStorage. To migrate to a
// different backing store later (IndexedDB, etc.), change this file.

const KEYS = {
    chunks: (convId) => `chunks_${convId}`,
    consumed: (convId) => `consumed_${convId}`,
    streaming: (convId) => `streaming_${convId}`,
    history: (convId) => `history_${convId}`,
    currentConversationId: () => 'currentConversationId',
};

function readJSON(key) {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    try {
        return JSON.parse(raw);
    } catch {
        return null;
    }
}

function writeJSON(key, value) {
    localStorage.setItem(key, JSON.stringify(value));
}

function readString(key) {
    return localStorage.getItem(key);
}

function writeString(key, value) {
    localStorage.setItem(key, value);
}

function remove(key) {
    localStorage.removeItem(key);
}

export const cache = {
    // --- current conversation (global, plain string) -------------------

    getCurrentConversationId() {
        return readString(KEYS.currentConversationId());
    },

    setCurrentConversationId(convId) {
        if (convId) {
            writeString(KEYS.currentConversationId(), convId);
        } else {
            remove(KEYS.currentConversationId());
        }
    },

    // --- history (per-conversation, JSON array) ------------------------

    getHistory(convId) {
        return readJSON(KEYS.history(convId));
    },

    setHistory(convId, messages) {
        writeJSON(KEYS.history(convId), messages);
    },

    appendToHistory(convId, message) {
        const history = this.getHistory(convId) || [];
        history.push(message);
        this.setHistory(convId, history);
    },

    clearHistory(convId) {
        remove(KEYS.history(convId));
    },

    // --- chunks (per-conversation, JSON array) -------------------------

    getChunks(convId) {
        return readJSON(KEYS.chunks(convId)) || [];
    },

    setChunks(convId, chunks) {
        writeJSON(KEYS.chunks(convId), chunks);
    },

    appendToChunks(convId, chunk) {
        const chunks = this.getChunks(convId);
        chunks.push(chunk);
        this.setChunks(convId, chunks);
    },

    clearChunks(convId) {
        remove(KEYS.chunks(convId));
    },

    // --- consumed pointer (per-conversation, plain string) --------------

    getConsumed(convId) {
        const raw = readString(KEYS.consumed(convId));
        const n = raw === null ? 0 : parseInt(raw, 10);
        return Number.isNaN(n) ? 0 : n;
    },

    setConsumed(convId, count) {
        writeString(KEYS.consumed(convId), count.toString());
    },

    clearConsumed(convId) {
        remove(KEYS.consumed(convId));
    },

    // --- streaming flag (per-conversation, plain string "true"/"false") --

    isStreaming(convId) {
        return readString(KEYS.streaming(convId)) === 'true';
    },

    getStreaming(convId) {
        return readString(KEYS.streaming(convId));
    },

    setStreaming(convId, value) {
        if (!convId) return;
        writeString(KEYS.streaming(convId), value ? 'true' : 'false');
    },

    clearStreaming(convId) {
        remove(KEYS.streaming(convId));
    },
};

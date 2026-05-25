# Multi-Conversation & Streaming Persistence - Requirements Specification

## Overview

Adding multi-conversation support with seamless streaming when switching conversations or refreshing the page.

---

## Functional Requirements

### FR-1: Multi-Conversation Management

| ID | Requirement |
|----|-------------|
| FR-1.1 | Users can create a new conversation at any time |
| FR-1.2 | Users can switch between existing conversations by clicking |
| FR-1.3 | Users can delete any conversation |
| FR-1.4 | Conversation list shows title (first message preview), last updated time |

### FR-2: Streaming Persistence

| ID | Requirement |
|----|-------------|
| FR-2.1 | When a user sends a message, the response streams in real-time |
| FR-2.2 | When user switches away from an active stream, streaming continues server-side |
| FR-2.3 | When user switches back to a conversation with active stream, streaming resumes from current position |
| FR-2.4 | When stream completes, the complete message is saved and displayed |
| FR-2.5 | On page refresh, active conversation is restored with current stream state |

### FR-3: Chat History

| ID | Requirement |
|----|-------------|
| FR-3.1 | Each conversation maintains its full message history |
| FR-3.2 | Users can view the complete history of any conversation |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat/stream` | Send message, receive streaming response |
| `GET` | `/api/chat/history/{id}` | Get conversation history |
| `GET` | `/api/chat/conversations` | List all conversations |
| `DELETE` | `/api/chat/conversation/{id}` | Delete a conversation |
| `GET` | `/api/chat/stream/status/{id}` | Check stream status for a conversation |

---

## Data Requirements

### Conversation Storage
- Each conversation has: `conversation_id`, `messages[]`, `created_at`, `updated_at`
- Messages are arrays of `{role: "user"|"assistant", content: string}`

### Stream Status
- Stream status includes: `streaming` (bool), `status`, `tokens_count`, `is_complete`, `partial_content`

---

## Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Switching conversations must not interrupt or restart an in-progress stream |
| NFR-2 | Page refresh must restore the user's current conversation and stream state |
| NFR-3 | No duplicate LLM calls for the same active stream |
| NFR-4 | Completed streams show complete message when user returns |

---

## Out of Scope (Future)

- Conversation search/filter
- Conversation renaming
- Sync across browser tabs
- Stream backpressure handling

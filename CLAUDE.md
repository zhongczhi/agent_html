# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular chatbot application using FastAPI backend + plain HTML/JS frontend with streaming responses via SSE. LLM: Anthropic Claude via MiniMax endpoint.

## Commands

### Run the server
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```
Frontend served at `/`, API at `/api/chat/stream`

### Run tests
```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

## Architecture

```
frontend/          # Plain HTML/JS (no framework), served by FastAPI
backend/
├── main.py        # FastAPI app entry point
├── config.py      # Pydantic settings from env vars
├── chat/          # Chat domain (routes, chain, service)
└── storage/       # Storage domain (JSON file operations)
```

### Key Patterns
- **SSE Streaming**: `POST /api/chat/stream` returns `StreamingResponse` with SSE format
- **LangChain LCEL**: Chat logic in `chat/chain.py` using LangChain Expression Language
- **Domain structure**: Each domain has `routes.py`, `chain.py`, `service.py`
- **Conversation history**: File-based JSON in `storage/conversations.json`

## Environment Variables

```env
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1
ANTHROPIC_API_KEY=your-api-key-here
```

## Testing

Tests mock LLM calls via `langchain.anthropic.ChatModel` - no real API calls. Use temporary directories for storage tests. HTTP tests via `httpx.AsyncClient` against FastAPI TestClient.

## Adding Features

- **RAG**: Create `backend/rag/` domain, integrate into `chat/chain.py`
- **File upload**: Create `backend/files/` domain
- **Auth**: Create `backend/auth/` domain with FastAPI dependency

## Iterative Development Workflow

This project uses iterative, incremental development. Each development cycle follows this loop:

```
idea → SPEC.md → DESI.md → implementation → testing → assessment → merge or refine
```

### Documentation Structure

| File | Purpose |
|------|---------|
| `SPEC.md` | Full project specification (all features, past and future, explains what the system does, nothing more) |
| `SPEC_focus.md` | Current iteration's spec (active working document) |
| `DESI.md` | Full project detailed design (all decisions, explains how the system implements it, nothing more) |
| `DESI_focus.md` | Current iteration's design (active working document) |

### Iteration Process

1. **Plan**: Define new feature/modification in `./document/SPEC_focus.md` and `./document/DESI_focus.md`
2. **Implement**: Work on code based on `_focus.md` files only
3. **Test**: Verify implementation against `_focus.md` specifications
4. **Full Test**: Run full test suite to ensure existing functionality is not broken
5. **Merge**: On success, merge `_focus.md` changes into main `SPEC.md`/`DESI.md`
6. **Archive**: Rename completed `_focus.md` as `SPEC_focus_v{N}.md` / `DESI_focus_v{N}.md` and move to `./document/history`, delete the old files.

### Key Principle

- ALWAYS ask the user to determine whether to proceed to the next step. Specifically. for bug-fixing, step 1: figure out the problem and locate, present possible solution for user to determine. step 2: proceed on implementing

- The `_focus.md` files are the **working documents** during an development iteration. The main `SPEC.md`/`DESI.md` files represent the "done" state. This separation prevents distraction from future backlog items during active development.

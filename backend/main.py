# backend/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.chat.routes import router as chat_router
from backend.rag.config import RagSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_settings = RagSettings()
    if rag_settings.rag_enabled:
        # Build the service eagerly at startup so model load + index load
        # are part of server startup time, not first-request latency.
        from backend.rag.service import RagService
        from backend.rag import routes as rag_routes
        from backend.storage import file_storage

        rag = RagService.from_settings()
        app.state.rag = rag

        # Wire the upload/del callback. We monkey-patch the module-level
        # function (file_storage.delete_conversation) so existing call sites
        # in chat/routes.py automatically trigger RAG cleanup. The patch
        # uses functools.partial to set the callback.
        from functools import partial
        original_delete = file_storage.delete_conversation
        patched = partial(original_delete, on_delete=rag.purge_uploads)
        # Replace the function in the module's namespace. The chat routes
        # import the name (not the function object), so this works.
        file_storage.delete_conversation = patched

        # Tell the rag routes module how to find the service.
        rag_routes.get_rag_service = lambda: rag

        # Mount the RAG routes.
        app.include_router(rag_routes.router)
        logger.info("RAG enabled: embedding=%s, top_k=%d", rag_settings.rag_embedding_backend, rag_settings.rag_top_k)

    yield

    if hasattr(app.state, "rag"):
        app.state.rag.persist_all()


app = FastAPI(lifespan=lifespan)

app.include_router(chat_router)

frontend_path = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    # Iteration 7: landing page is the side-by-side compare UI.
    # The original single-pane chat is preserved at /single for users
    # who prefer it.
    return FileResponse(frontend_path / "compare.html")


@app.get("/single")
async def single():
    return FileResponse(frontend_path / "index.html")


if (frontend_path / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

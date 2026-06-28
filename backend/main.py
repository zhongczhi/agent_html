# backend/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.chat.routes import router as chat_router
from backend.rag.config import RagSettings

logging.basicConfig(level=logging.INFO)
# Silence noisy INFO-level chatter from external libraries so the operator
# log only shows what's actually relevant to this app.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)
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

        # Inject the rag service into the chat service. The chat routes
        # module has a set_rag_service() helper that we call here so any
        # chat service constructed later (lazy-init) will pick it up.
        from backend.chat import routes as chat_routes
        chat_routes.set_rag_service(rag)

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


# Disable HTTP caching for static files and the index. Without this, a
# browser may keep serving stale index.html / app.js / styles.css even
# after the user restarts the server with new code, hiding new features.
class _NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(_NoCacheStaticMiddleware)


@app.get("/")
async def root():
    return FileResponse(frontend_path / "index.html")


if (frontend_path / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path / "static")), name="static")

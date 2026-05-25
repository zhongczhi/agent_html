# backend/main.py
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.chat.routes import router as chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.include_router(chat_router)

frontend_path = Path(__file__).parent.parent / "frontend"


@app.get("/")
async def root():
    return FileResponse(frontend_path / "index.html")


@app.get("/index.html")
async def index_html():
    return FileResponse(frontend_path / "index.html")


if (frontend_path / "index.html").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

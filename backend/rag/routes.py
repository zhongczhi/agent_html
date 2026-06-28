import logging
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import backend.storage.file_storage as file_storage
from backend.rag.service import RagService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])


def get_rag_service() -> RagService | None:
    """Module-level getter; main.py monkey-patches this to return the
    long-lived RagService. Returns None when RAG is disabled — routes
    surface this as 503.
    """
    return None


def _service_or_503() -> RagService:
    svc = get_rag_service()
    if svc is None:
        raise HTTPException(status_code=503, detail="RAG is not enabled")
    return svc


@router.get("/stats")
def stats():
    svc = _service_or_503()
    return svc.stats()


@router.post("/library/reindex")
def library_reindex():
    svc = _service_or_503()
    return svc.reindex_library()


@router.post("/upload")
def upload(
    conversation_id: str = Form(...),
    file: UploadFile = File(...),
):
    svc = _service_or_503()
    # Idempotent — guarantees the conversation is visible in the sidebar
    # immediately after upload, even if the user hasn't sent any message.
    file_storage.create_conversation(conversation_id)

    # Save the upload to a temp file then ingest by path. service.ingest_file
    # copies the file into the per-conversation uploads dir.
    import tempfile
    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = Path(tmp.name)
    try:
        chunk_ids = svc.ingest_file(conversation_id, tmp_path)
    except Exception as e:
        logger.exception("Upload ingestion failed: %s", e)
        # File may be in storage/uploads/<conv_id>/ already; leave it for retry.
        raise HTTPException(status_code=500, detail=f"Embedding/indexing failed: {e}")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return {
        "filename": file.filename,
        "chunks_added": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }

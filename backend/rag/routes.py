import logging
from pathlib import Path
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import backend.storage.file_storage as file_storage
from backend.rag.config import RagSettings
from backend.rag.service import RagService
from backend.rag.loaders import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])


# Re-exported for tests that import `ALLOWED_EXTENSIONS` from this module.
# Single source of truth lives in `backend.rag.loaders`.
__all__ = ["router", "ALLOWED_EXTENSIONS"]


def _check_extension(filename: str | None) -> str:
    """Return the lowercased extension (with dot) for `filename`, or raise
    HTTP 400 if it's outside the allowlist."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )
    return suffix


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
    data = svc.stats()
    # Surface the threshold so the client can apply the same boundary
    # without a separate env var. (FR-12.5)
    data["inline_context_threshold_bytes"] = RagSettings().rag_inline_context_threshold_bytes
    # library_files is already in svc.stats() (FR-28.8)
    return data


# ── Library management (iter-8 Phase D) ─────────────────────────────────────

@router.get("/library/files")
def library_files():
    svc = _service_or_503()
    return {"files": svc.list_library_files()}


@router.post("/library/upload")
def library_upload(file: UploadFile = File(...)):
    svc = _service_or_503()
    if not file.filename or "/" in file.filename or "\\" in file.filename or file.filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename")
    _check_extension(file.filename)
    if (svc.library_dir / file.filename).exists():
        raise HTTPException(
            status_code=409,
            detail=f"'{file.filename}' is already in the library; delete first",
        )
    content = file.file.read()
    try:
        path = svc.save_library_file(file.filename, content)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    return {"filename": file.filename, "size": path.stat().st_size, "saved": True}


@router.delete("/library/file/{filename:path}")
def library_file_delete(filename: str):
    svc = _service_or_503()
    # Iter-9: filenames may now contain '/' (subpaths like
    # `hotpotqa/<id>.md`). Traversal is blocked at the service layer by
    # _safe_library_path, which raises ValueError for escape attempts.
    # We translate that to 400 here so the route doesn't 500.
    _check_extension(filename)
    try:
        deleted = svc.delete_library_file(filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not deleted:
        raise HTTPException(status_code=404, detail="Not found")
    return {"deleted": True, "filename": filename}


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
    # FR-12.1: reject disallowed file types up front, before any IO.
    # Raises HTTP 400 if the extension isn't in the allowlist.
    suffix = _check_extension(file.filename)

    # Idempotent — guarantees the conversation is visible in the sidebar
    # immediately after upload, even if the user hasn't sent any message.
    file_storage.create_conversation(conversation_id)

    threshold = RagSettings().rag_inline_context_threshold_bytes

    # Read the raw upload into memory. We need its size + content to decide
    # which path to take; reading it twice would double-IO for large files.
    raw = file.file.read()
    size = len(raw)

    if size <= threshold:
        # ── Inline path ──────────────────────────────────────────────
        # Small file: decode as UTF-8 text, return content. The client will
        # include this in the next chat request's `uploaded_files` field.
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            # Binary file that happens to be small — fall through to FAISS.
            logger.info("Small upload %s is not UTF-8; falling back to FAISS", file.filename)
            content = None
        if content is not None:
            return {
                "filename": file.filename,
                "mode": "inline",
                "bytes": size,
                "content": content,
            }

    # ── FAISS path ──────────────────────────────────────────────────
    # Either the file is large OR it's small but binary. Either way: save
    # to a temp file then ingest by path (service.ingest_file copies into
    # the per-conversation uploads dir). `suffix` was already validated
    # above; reuse it for the temp file name.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        chunk_ids = svc.ingest_file(conversation_id, tmp_path)
    except Exception as e:
        logger.exception("Upload ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Embedding/indexing failed: {e}")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
    return {
        "filename": file.filename,
        "mode": "indexed",
        "bytes": size,
        "chunks_added": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }

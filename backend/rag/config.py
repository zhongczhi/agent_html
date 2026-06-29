from pydantic_settings import BaseSettings, SettingsConfigDict


class RagSettings(BaseSettings):
    """RAG module settings. Field names match the RAG_* env var convention
    used throughout this project (e.g., ANTHROPIC_BASE_URL for the global
    config). Independent of backend.config so RAG is fully opt-in."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    rag_enabled: bool = False
    rag_embedding_backend: str = "sentence-transformers"
    rag_sentence_transformers_model: str = "all-MiniLM-L6-v2"
    rag_library_dir: str = "storage/library"
    rag_uploads_dir: str = "storage/uploads"
    rag_index_dir: str = "storage/rag"
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 200
    rag_top_k: int = 4
    rag_inline_context_threshold_bytes: int = 8192

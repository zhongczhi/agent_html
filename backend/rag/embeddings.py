from typing import Any
from langchain_core.embeddings import Embeddings


def _build_huggingface(model_name: str) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)


class MiniMaxEmbeddings(Embeddings):
    """Embeddings via the MiniMax /embeddings endpoint (same vendor as the
    chat model). Reuses ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY for convenience
    since the project's .env already has them.

    If the endpoint is unavailable, the import or the first call will raise
    and the user must fall back to EMBEDDING_BACKEND=sentence-transformers.
    """
    def __init__(self, api_key: str, base_url: str, model: str = "minimax-3"):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model

    def _embed(self, text: str) -> list[float]:
        import httpx
        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


def _build_minimax(api_key: str, base_url: str) -> Embeddings:
    return MiniMaxEmbeddings(api_key=api_key, base_url=base_url)


def make_embeddings(
    backend: str,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    api_key: str = "",
    base_url: str = "https://api.minimax.chat/v1",
) -> Embeddings:
    if backend == "sentence-transformers":
        return _build_huggingface(model_name)
    if backend == "minimax":
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return _build_minimax(api_key, base_url)
    raise ValueError(f"Unknown embedding backend: {backend!r}")

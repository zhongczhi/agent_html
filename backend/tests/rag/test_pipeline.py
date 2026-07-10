"""Tests for backend.rag.pipeline. No real API calls; no real FAISS."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from backend.rag.pipeline import (
    AnthropicLLM,
    CrossEncoderReranker,
    DefaultPromptBuilder,
    DenseRetriever,
    ExtractSpanPromptBuilder,
    NoOpReranker,
    PRESETS,
    PipelineConfig,
    RagPipeline,
    build_pipeline,
    build_prompt_builder,
    build_reranker,
    build_retriever,
    list_presets,
)


# ---- PipelineConfig ------------------------------------------------------

def test_config_defaults():
    """Default config matches naive_dense behavior (baseline)."""
    cfg = PipelineConfig(name="test")
    assert cfg.embedding_backend == "sentence-transformers"
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.retriever == "dense"
    assert cfg.reranker is None
    assert cfg.top_k == 4
    assert cfg.prompt_template == "default"


def test_config_is_frozen():
    """Frozen dataclass — assignment should raise."""
    cfg = PipelineConfig(name="test")
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.top_k = 8  # type: ignore[misc]


# ---- DenseRetriever ------------------------------------------------------

class FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k):
        return self._docs[:k]


def test_dense_retriever_passes_through():
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(5)]
    vs = FakeVectorStore(docs)
    r = DenseRetriever(vs)
    out = r.retrieve("query", k=3)
    assert len(out) == 3
    assert [d.page_content for d in out] == ["d0", "d1", "d2"]


# ---- NoOpReranker --------------------------------------------------------

def test_noop_reranker_slices_to_top_k():
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(10)]
    r = NoOpReranker()
    out = r.rerank("q", docs, top_k=3)
    assert [d.page_content for d in out] == ["d0", "d1", "d2"]


def test_noop_reranker_handles_empty():
    r = NoOpReranker()
    assert r.rerank("q", [], top_k=5) == []


# ---- CrossEncoderReranker ------------------------------------------------

def test_cross_encoder_reranker_ranks_by_score():
    """Mock the underlying CrossEncoder.predict to verify the reranker
    sorts by score descending and slices to top_k."""
    import numpy as np

    r = CrossEncoderReranker()
    # Inject a fake model so we don't hit HuggingFace.
    fake_model = MagicMock()
    # Scores: d0=0.1, d1=0.9, d2=0.5, d3=0.2 — expect order d1, d2, d3 (top-3).
    fake_model.predict = MagicMock(return_value=np.array([0.1, 0.9, 0.5, 0.2]))
    r._model = fake_model

    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(4)]
    out = r.rerank("q", docs, top_k=3)
    assert [d.page_content for d in out] == ["d1", "d2", "d3"]


def test_cross_encoder_reranker_handles_empty():
    r = CrossEncoderReranker()
    assert r.rerank("q", [], top_k=5) == []


# ---- DefaultPromptBuilder -----------------------------------------------

def test_default_prompt_builder_with_context_has_system_and_user():
    b = DefaultPromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "<context>" in msgs[1]["content"]


def test_default_prompt_builder_without_context_user_only():
    b = DefaultPromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


# ---- ExtractSpanPromptBuilder --------------------------------------------

def test_extract_span_builder_appends_instruction():
    b = ExtractSpanPromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    assert len(msgs) == 2
    assert "extract" in msgs[0]["content"].lower()
    assert "verbatim" in msgs[0]["content"].lower()


def test_extract_span_builder_user_only_when_no_context():
    b = ExtractSpanPromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1


# ---- Factory: build_reranker --------------------------------------------

def test_build_reranker_none():
    cfg = PipelineConfig(name="x", reranker=None)
    assert build_reranker(cfg) is None


def test_build_reranker_cross_encoder():
    cfg = PipelineConfig(name="x", reranker="cross_encoder")
    r = build_reranker(cfg)
    assert isinstance(r, CrossEncoderReranker)


def test_build_reranker_unknown_raises():
    cfg = PipelineConfig(name="x", reranker="bogus")
    with pytest.raises(ValueError, match="Unknown reranker"):
        build_reranker(cfg)


# ---- Factory: build_prompt_builder --------------------------------------

def test_build_prompt_builder_default():
    cfg = PipelineConfig(name="x", prompt_template="default")
    assert isinstance(build_prompt_builder(cfg), DefaultPromptBuilder)


def test_build_prompt_builder_extract_span():
    cfg = PipelineConfig(name="x", prompt_template="extract_span")
    assert isinstance(build_prompt_builder(cfg), ExtractSpanPromptBuilder)


def test_build_prompt_builder_unknown_raises():
    cfg = PipelineConfig(name="x", prompt_template="bogus")
    with pytest.raises(ValueError, match="Unknown prompt_template"):
        build_prompt_builder(cfg)


# ---- Factory: build_retriever -------------------------------------------

def test_build_retriever_dense():
    cfg = PipelineConfig(name="x", retriever="dense")
    vs = FakeVectorStore([Document(page_content="d")])
    r = build_retriever(cfg, vs)
    assert isinstance(r, DenseRetriever)


def test_build_retriever_unknown_raises():
    cfg = PipelineConfig(name="x", retriever="bogus")
    with pytest.raises(ValueError, match="Unknown retriever"):
        build_retriever(cfg, FakeVectorStore([]))


# ---- AnthropicLLM (delegates to qa_judge.ask_llm) ----------------------

@pytest.mark.asyncio
async def test_anthropic_llm_delegates():
    """AnthropicLLM.ask should call qa_judge.ask_llm with the right args."""
    fake_client = MagicMock()
    cfg = PipelineConfig(name="x", llm_model="custom-model")
    llm = AnthropicLLM(client=fake_client, model=cfg.llm_model)
    # Patch qa_judge.ask_llm so we don't make a real call.
    with patch("backend.eval.qa_judge.ask_llm", new_callable=AsyncMock) as mock_ask:
        mock_ask.return_value = "answer"
        out = await llm.ask([{"role": "user", "content": "q"}])
    assert out == "answer"
    mock_ask.assert_called_once()
    args, kwargs = mock_ask.call_args
    assert args[0] is fake_client
    assert args[1] == "custom-model"


# ---- RagPipeline.run -----------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_run_no_reranker_slices_top_k():
    cfg = PipelineConfig(name="x", top_k=3, reranker=None)
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(5)]
    # Use a MagicMock retriever so we can assert the call args.
    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=docs)
    prompt_builder = MagicMock()
    prompt_builder.build = MagicMock(return_value=[{"role": "user", "content": "x"}])
    llm = MagicMock()
    llm.ask = AsyncMock(return_value="A")

    p = RagPipeline(cfg, retriever, None, prompt_builder, llm)
    out = await p.run("q")
    assert out == "A"
    # Retriever called with top_k (3), not rerank_top_k (50 default), because no reranker.
    retriever.retrieve.assert_called_once_with("q", k=3)
    # Prompt builder got the sliced 3 docs (positional args).
    prompt_builder.build.assert_called_once()
    pb_args = prompt_builder.build.call_args.args
    assert len(pb_args[1]) == 3  # second positional arg is context_docs


@pytest.mark.asyncio
async def test_pipeline_run_with_reranker_retrieves_more_then_reranks():
    cfg = PipelineConfig(name="x", top_k=3, reranker="cross_encoder", rerank_top_k=10)
    docs = [Document(page_content=f"d{i}", metadata={}) for i in range(10)]

    retriever = MagicMock()
    retriever.retrieve = MagicMock(return_value=docs)  # returns all 10 when asked for 10
    reranker = MagicMock()
    reranker.rerank = MagicMock(return_value=docs[:3])
    prompt_builder = MagicMock()
    prompt_builder.build = MagicMock(return_value=[{"role": "user", "content": "x"}])
    llm = MagicMock()
    llm.ask = AsyncMock(return_value="B")

    p = RagPipeline(cfg, retriever, reranker, prompt_builder, llm)
    out = await p.run("q")
    assert out == "B"
    # Retriever was called with rerank_top_k (10), not top_k (3).
    retriever.retrieve.assert_called_once_with("q", k=10)
    # Reranker received all 10 candidates and was asked for top_k=3.
    reranker.rerank.assert_called_once()
    rk_args = reranker.rerank.call_args.args
    rk_kwargs = reranker.rerank.call_args.kwargs
    # candidates can be positional or keyword depending on caller; check both
    candidates = rk_kwargs.get("candidates", rk_args[1] if len(rk_args) > 1 else None)
    assert len(candidates) == 10
    assert rk_kwargs["top_k"] == 3
    # Prompt builder got 3 reranked docs.
    prompt_builder.build.assert_called_once()
    pb_args = prompt_builder.build.call_args.args
    assert len(pb_args[1]) == 3  # second positional arg is context_docs


# ---- Presets ------------------------------------------------------------

def test_presets_have_expected_keys():
    assert "naive_dense" in PRESETS
    assert "large_dense" in PRESETS
    assert "dense_then_ce" in PRESETS
    assert "extract_span_prompt" in PRESETS


def test_naive_dense_uses_default_embedding():
    cfg = PRESETS["naive_dense"]
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.reranker is None
    assert cfg.top_k == 4
    assert cfg.prompt_template == "default"


def test_large_dense_uses_mpnet():
    cfg = PRESETS["large_dense"]
    assert cfg.embedding_model == "all-mpnet-base-v2"
    assert cfg.reranker is None


def test_dense_then_ce_has_reranker():
    cfg = PRESETS["dense_then_ce"]
    assert cfg.reranker == "cross_encoder"
    assert cfg.rerank_top_k > cfg.top_k  # must retrieve more than top_k


def test_extract_span_prompt_uses_extract_template():
    cfg = PRESETS["extract_span_prompt"]
    assert cfg.prompt_template == "extract_span"


def test_list_presets_returns_sorted():
    names = list_presets()
    assert names == sorted(names)
    assert len(names) >= 4


# ---- build_pipeline (the one-switch API) --------------------------------

def test_build_pipeline_composes_all_components():
    """The top-level factory should wire up retriever, reranker, prompt, llm
    from a single config. Verified by checking that each component is the
    right class."""
    cfg = PRESETS["naive_dense"]
    fake_client = MagicMock()
    vs = FakeVectorStore([Document(page_content="d")])

    pipeline = build_pipeline(cfg, vectorstore=vs, llm_client=fake_client)
    assert isinstance(pipeline, RagPipeline)
    assert pipeline.config is cfg
    # Each component was built correctly.
    assert isinstance(pipeline._retriever, DenseRetriever)
    # naive_dense has no reranker; build_reranker returned None.
    assert pipeline._reranker is None
    assert isinstance(pipeline._prompt_builder, DefaultPromptBuilder)
    assert isinstance(pipeline._llm, AnthropicLLM)


def test_build_pipeline_with_reranker():
    cfg = PRESETS["dense_then_ce"]
    fake_client = MagicMock()
    vs = FakeVectorStore([Document(page_content="d")])

    pipeline = build_pipeline(cfg, vectorstore=vs, llm_client=fake_client)
    assert pipeline._reranker is not None
    assert isinstance(pipeline._reranker, CrossEncoderReranker)
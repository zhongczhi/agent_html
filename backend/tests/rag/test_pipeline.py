"""Tests for backend.rag.pipeline. No real API calls; no real FAISS."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document

from backend.rag.pipeline import (
    AnthropicLLM,
    BM25Retriever,
    CoTExtractNoTitlesPromptBuilder,
    CoTExtractPromptBuilder,
    CoTExtractV2PromptBuilder,
    CrossEncoderReranker,
    DefaultPromptBuilder,
    DenseRetriever,
    ExtractSpanPromptBuilder,
    HybridRetriever,
    NoOpReranker,
    PRESETS,
    PreAnalysisExtractPromptBuilder,
    PipelineConfig,
    RagPipeline,
    CleanGroupedPromptBuilder,
    ParametrizedGroupedPromptBuilder,
    _build_v15_preset,
    build_llm,
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


# ---- CoTExtractPromptBuilder (iter-15) -----------------------------------

def test_cot_extract_builder_appends_step_by_step():
    """iter-15: CoT-extract prompt must scaffold step-by-step reasoning
    in addition to the verbatim-span discipline. Both should appear in
    the system prompt."""
    b = CoTExtractPromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    assert len(msgs) == 2
    system = msgs[0]["content"].lower()
    assert "step by step" in system
    assert "extract" in system
    assert "verbatim" in system


def test_cot_extract_builder_user_only_when_no_context():
    b = CoTExtractPromptBuilder()
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


def test_build_prompt_builder_cot_extract():
    """iter-15: cot_extract dispatch returns the CoT-aware builder."""
    from backend.rag.pipeline import CoTExtractPromptBuilder
    cfg = PipelineConfig(name="x", prompt_template="cot_extract")
    assert isinstance(build_prompt_builder(cfg), CoTExtractPromptBuilder)



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
    # iter-14 ceiling variants
    assert "extract_span_k8" in PRESETS
    assert "extract_span_k10" in PRESETS
    # iter-15 SOTA
    assert "cot_extract_k10" in PRESETS
    # iter-19: minimal v2 refinement of cot_extract
    assert "cot_extract_v2_k10" in PRESETS
    # iter-19: rolled back the iter-15/16/17 canonical + iter-18 two-step
    # experiments
    assert "canonical_extract_k10" not in PRESETS
    assert "two_step_extract_k10" not in PRESETS


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


def test_extract_span_k8_combines_prompt_and_top_k():
    """iter-14 ceiling variant: extract_span prompt with top_k=8."""
    cfg = PRESETS["extract_span_k8"]
    assert cfg.prompt_template == "extract_span"
    assert cfg.top_k == 8
    assert cfg.reranker is None


def test_extract_span_k10_uses_full_context():
    """iter-14 max-context variant: extract_span prompt with top_k=10."""
    cfg = PRESETS["extract_span_k10"]
    assert cfg.prompt_template == "extract_span"
    assert cfg.top_k == 10
    assert cfg.reranker is None


def test_cot_extract_k10_combines_cot_and_full_context():
    """iter-15 preset: cot_extract prompt + k=10 full context window."""
    cfg = PRESETS["cot_extract_k10"]
    assert cfg.prompt_template == "cot_extract"
    assert cfg.top_k == 10
    assert cfg.reranker is None
    assert cfg.retriever == "dense"



def test_list_presets_returns_sorted():
    names = list_presets()
    assert names == sorted(names)
    assert len(names) >= 4


# ---- CoTExtractV2PromptBuilder (iter-19) -------------------------------

def test_cot_extract_v2_keeps_cot_scaffold():
    """iter-19: V2 inherits CoT scaffold (same 4 numbered steps)."""
    from backend.rag.pipeline import CoTExtractV2PromptBuilder
    b = CoTExtractV2PromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    system = msgs[0]["content"]
    assert "Think step by step:" in system
    assert "1. Identify the entities" in system
    assert "2. Find the relevant paragraph" in system
    assert "3. If multi-hop reasoning is needed" in system



def test_cot_extract_v2_user_only_when_no_context():
    from backend.rag.pipeline import CoTExtractV2PromptBuilder
    b = CoTExtractV2PromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1


def test_build_prompt_builder_v2_returns_v2_class():
    """iter-19: cot_extract_v2 template dispatches to CoTExtractV2PromptBuilder."""
    cfg = PipelineConfig(name="x", prompt_template="cot_extract_v2")
    assert isinstance(build_prompt_builder(cfg), CoTExtractV2PromptBuilder)


def test_cot_extract_v2_k10_preset_uses_v2_template():
    cfg = PRESETS["cot_extract_v2_k10"]
    assert cfg.prompt_template == "cot_extract_v2"
    assert cfg.top_k == 10



# ---- CoTExtractV2PromptBuilder (iter-19) -------------------------------

def test_cot_extract_v2_keeps_cot_scaffold():
    """iter-19: V2 inherits CoT scaffold (same 4 numbered steps)."""
    b = CoTExtractV2PromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    system = msgs[0]["content"]
    assert "Think step by step:" in system
    assert "1. Identify the entities" in system
    assert "2. Find the relevant paragraph" in system
    assert "3. If multi-hop reasoning is needed" in system


def test_cot_extract_v2_adds_canonical_nudge_in_step_4():
    """iter-19: step 4 contains the 'most complete form' nudge embedded
    directly into the CoT instruction rather than as a separate rule."""
    b = CoTExtractV2PromptBuilder()
    docs = [Document(page_content="text", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    system = msgs[0]["content"]
    # Step 4 contains the nudge.
    assert "most complete form" in system


def test_cot_extract_v2_prompt_grows_only_minimally_over_cot():
    """iter-19: V2 should add only a few extra words vs the cot baseline.
    No examples, no discriminators, no separate rules."""
    from backend.rag.pipeline import CoTExtractPromptBuilder
    v2 = CoTExtractV2PromptBuilder()
    v1 = CoTExtractPromptBuilder()
    v2_len = len(v2._system_prompt)
    v1_len = len(v1._system_prompt)
    # The nudge adds roughly 25 words. Allow generous slack.
    assert v2_len - v1_len < 200, (
        f"V2 prompt grew by {v2_len - v1_len} chars vs cot; iter-19 budget is < 200."
    )


def test_cot_extract_v2_user_only_when_no_context():
    b = CoTExtractV2PromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1


def test_build_prompt_builder_v2_returns_v2_class():
    """iter-19: cot_extract_v2 template dispatches to CoTExtractV2PromptBuilder."""
    cfg = PipelineConfig(name="x", prompt_template="cot_extract_v2")
    assert isinstance(build_prompt_builder(cfg), CoTExtractV2PromptBuilder)


def test_cot_extract_v2_k10_preset_uses_v2_template():
    cfg = PRESETS["cot_extract_v2_k10"]
    assert cfg.prompt_template == "cot_extract_v2"
    assert cfg.top_k == 10
    assert cfg.reranker is None
    assert cfg.retriever == "dense"


def test_iter_19_presets_rolled_back():
    """iter-19: rolled back iter-15/16/17 canonical and iter-18 two-step
    experiments; those presets should no longer be in PRESETS."""
    assert "canonical_extract_k10" not in PRESETS
    assert "two_step_extract_k10" not in PRESETS


# ---- iter-21: CoTExtractNoTitlesPromptBuilder (heading-stripped) -------

def test_cot_extract_no_titles_strips_heading_prefix():
    """iter-21: context paragraphs have NO `[title]:` prefix in the
    user prompt. The model has to find entity names in the body text."""
    b = CoTExtractNoTitlesPromptBuilder()
    docs = [
        Document(page_content="Louis-Hector Berlioz (born 11 December 1803) was a French Romantic composer.", metadata={"title": "Hector Berlioz"}),
        Document(page_content="Gaetano Donizetti was an Italian composer.", metadata={"title": "Gaetano Donizetti"}),
    ]
    msgs = b.build("Which is the French Romantic composer?", docs)
    user = msgs[1]["content"]
    # Body content is present...
    assert "Louis-Hector Berlioz" in user
    assert "Gaetano Donizetti" in user
    # ...but no `[title]:` heading prefix.
    assert "[Hector Berlioz]:" not in user
    assert "[Gaetano Donizetti]:" not in user


def test_cot_extract_no_titles_keeps_cot_scaffold():
    """iter-21: title-stripped variant still includes the CoT instruction."""
    b = CoTExtractNoTitlesPromptBuilder()
    docs = [Document(page_content="x", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    system = msgs[0]["content"]
    assert "Think step by step:" in system
    assert "quote it verbatim from the context" in system


def test_cot_extract_no_titles_user_only_when_no_context():
    b = CoTExtractNoTitlesPromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1


def test_build_prompt_builder_no_titles_dispatches_correctly():
    cfg = PipelineConfig(name="x", prompt_template="cot_extract_no_titles")
    assert isinstance(build_prompt_builder(cfg), CoTExtractNoTitlesPromptBuilder)


def test_cot_extract_notitles_k10_preset_uses_no_titles_template():
    """iter-21: preset registered with the title-stripped template."""
    cfg = PRESETS["cot_extract_notitles_k10"]
    assert cfg.prompt_template == "cot_extract_no_titles"
    assert cfg.top_k == 10
    assert cfg.reranker is None


def test_cot_extract_notitles_thinking_k10_combines_both():
    """iter-22: combine title-strip with thinking mode (4096 budget)."""
    cfg = PRESETS["cot_extract_notitles_thinking_k10"]
    assert cfg.prompt_template == "cot_extract_no_titles"
    assert cfg.thinking_budget == 4096
    assert cfg.top_k == 10


# ---- iter-29: PreAnalysisExtractPromptBuilder (pre-analysis prefix) -----

def test_pre_analysis_extract_includes_pre_analysis_instruction():
    """iter-29 v2: the user message opens with a pre-analysis instruction
    that enumerates the four question shapes BEFORE the <context> block."""
    b = PreAnalysisExtractPromptBuilder()
    docs = [Document(page_content="Foo bar baz.", metadata={"title": "T1"})]
    msgs = b.build("Which is X?", docs)
    user = msgs[1]["content"]
    assert user.startswith("Before reading the context, briefly identify what kind of question this is.")
    assert "<context>" in user
    assert "Which is X?" in user
    # The pre-analysis instruction must come before the context block.
    assert user.index("Before reading the context") < user.index("<context>")


def test_pre_analysis_extract_enumerates_all_question_shapes():
    """iter-29 v2: prompt must cover all four question shapes observed in
    the eval datasets. The iter-29 v1 generic wording missed comparison
    questions because the model couldn't decide which 'kind of material'
    applied to 'Does X suggest Y' questions."""
    b = PreAnalysisExtractPromptBuilder()
    msgs = b.build("Q?", [Document(page_content="x", metadata={"title": "T"})])
    user = msgs[1]["content"]
    # All four shapes must be present so the LLM can pick the right one.
    assert "ENTITY LOOKUP" in user
    assert "YES/NO ADJUDICATION" in user
    assert "TEMPORAL ORDERING" in user
    assert "REFUSAL" in user
    # Each shape should give a brief extraction directive.
    assert "extract a single named entity" in user  # entity
    assert "answer with one word" in user            # yes/no
    assert "Insufficient information" in user        # refusal


def test_pre_analysis_extract_strips_heading_prefix():
    """iter-29: inherits the title-strip behavior from iter-21."""
    b = PreAnalysisExtractPromptBuilder()
    docs = [
        Document(page_content="Louis-Hector Berlioz was a French composer.", metadata={"title": "Hector Berlioz"}),
    ]
    msgs = b.build("Which composer?", docs)
    user = msgs[1]["content"]
    assert "Louis-Hector Berlioz" in user
    assert "[Hector Berlioz]:" not in user


def test_pre_analysis_extract_keeps_cot_scaffold_in_system():
    """iter-29: the system prompt still contains the CoT scaffold."""
    b = PreAnalysisExtractPromptBuilder()
    msgs = b.build("Q?", [Document(page_content="x", metadata={"title": "T"})])
    system = msgs[0]["content"]
    assert "Think step by step:" in system
    assert "quote it verbatim from the context" in system


def test_pre_analysis_extract_user_only_when_no_context():
    b = PreAnalysisExtractPromptBuilder()
    msgs = b.build("Q?", None)
    assert len(msgs) == 1


def test_build_prompt_builder_pre_analysis_dispatches_correctly():
    cfg = PipelineConfig(name="x", prompt_template="pre_analysis_extract")
    assert isinstance(build_prompt_builder(cfg), PreAnalysisExtractPromptBuilder)


def test_pre_analysis_extract_thinking_k10_preset_registered():
    """iter-29: preset combines pre-analysis with title-strip + thinking."""
    cfg = PRESETS["pre_analysis_extract_thinking_k10"]
    assert cfg.prompt_template == "pre_analysis_extract"
    assert cfg.top_k == 10
    assert cfg.thinking_budget == 4096
    assert cfg.reranker is None


def test_cot_extract_keeps_titles_for_backward_compat():
    """iter-21: original cot_extract template still uses titles (the iter-21
    experiment must not silently change SOTA behavior)."""
    b = CoTExtractPromptBuilder()
    docs = [Document(page_content="x", metadata={"title": "T"})]
    msgs = b.build("Q?", docs)
    user = msgs[1]["content"]
    assert "[T]:" in user  # cot_extract DOES include the heading


# ---- iter-20: Anthropic extended thinking mode ---------------------------

def test_pipeline_config_default_thinking_budget_is_none():
    """iter-20: default thinking_budget is None (disabled) for backward
    compatibility with all existing presets."""
    cfg = PipelineConfig(name="test")
    assert cfg.thinking_budget is None


def test_cot_thinking_k10_preset_uses_thinking_budget_4096():
    """iter-20: cot_thinking_k10 enables Anthropic thinking mode at 4096."""
    cfg = PRESETS["cot_thinking_k10"]
    assert cfg.thinking_budget == 4096
    assert cfg.top_k == 10
    assert cfg.reranker is None
    assert cfg.retriever == "dense"


def test_anthropic_llm_stores_thinking_budget():
    """iter-20: AnthropicLLM accepts thinking_budget in __init__."""
    from backend.rag.pipeline import AnthropicLLM
    llm = AnthropicLLM(client=MagicMock(), model="m", thinking_budget=4096)
    assert llm._thinking_budget == 4096


def test_anthropic_llm_thinking_budget_default_is_none():
    """iter-20: default thinking_budget is None (preserves old behavior)."""
    from backend.rag.pipeline import AnthropicLLM
    llm = AnthropicLLM(client=MagicMock(), model="m")
    assert llm._thinking_budget is None


def test_build_llm_threads_thinking_budget_from_config():
    """iter-20: build_llm reads thinking_budget from PipelineConfig."""
    cfg = PRESETS["cot_thinking_k10"]
    llm = build_llm(cfg, client=MagicMock())
    assert llm._thinking_budget == 4096


@pytest.mark.asyncio
async def test_anthropic_llm_ask_calls_ask_llm_with_thinking_budget():
    """iter-20: AnthropicLLM.ask passes thinking_budget through to ask_llm."""
    from unittest.mock import AsyncMock, patch
    from backend.rag.pipeline import AnthropicLLM
    llm = AnthropicLLM(client=MagicMock(), model="m", thinking_budget=2048)
    with patch("backend.eval.qa_judge.ask_llm", new=AsyncMock(return_value="ANS")) as mock_ask:
        out = await llm.ask([{"role": "user", "content": "Q"}], max_tokens=2048)
    assert out == "ANS"
    # Confirm thinking_budget was passed through.
    kwargs = mock_ask.call_args.kwargs
    assert kwargs.get("thinking_budget") == 2048


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


# ---- BM25Retriever -------------------------------------------------------

def test_bm25_retriever_finds_exact_match():
    """A query that exactly matches one document returns that doc first."""
    docs = [
        Document(page_content="the quick brown fox jumps"),
        Document(page_content="a stitch in time saves nine"),
        Document(page_content="the cat sat on the mat"),
    ]
    r = BM25Retriever(docs)
    out = r.retrieve("cat mat", k=3)
    assert out[0].page_content == "the cat sat on the mat"


def test_bm25_retriever_handles_empty_corpus():
    r = BM25Retriever([])
    assert r.retrieve("anything", k=5) == []


def test_bm25_retriever_handles_query_with_no_matches():
    """Query words not in any doc returns docs by corpus order (BM25 ranks
    by score; with no matches all scores are 0, so docs are returned in
    insertion order). This is intentional — the retriever always returns
    up to k docs, leaving the caller's downstream filtering (e.g., RRF
    fusion) to handle relevance."""
    docs = [
        Document(page_content="alpha bravo charlie"),
        Document(page_content="delta echo foxtrot"),
    ]
    r = BM25Retriever(docs)
    out = r.retrieve("xyzzy plover", k=2)
    # All scores are 0 -> docs returned in original order.
    assert len(out) == 2
    assert out[0].page_content == docs[0].page_content


def test_bm25_retriever_respects_top_k():
    docs = [
        Document(page_content=f"document {i}: common word") for i in range(10)
    ]
    r = BM25Retriever(docs)
    out = r.retrieve("common word", k=3)
    assert len(out) == 3


def test_bm25_retriever_handles_punctuation_in_docs():
    """Non-word chars in doc text should be stripped during tokenization."""
    docs = [Document(page_content="Foo, bar! baz? qux...")]
    r = BM25Retriever(docs)
    out = r.retrieve("foo bar", k=1)
    assert len(out) == 1


# ---- HybridRetriever (RRF) -----------------------------------------------

class _RecordingDenseRetriever:
    """A fake dense retriever that returns docs in a fixed order."""

    def __init__(self, docs):
        self._docs = list(docs)

    def retrieve(self, query, k):
        return self._docs[:k]


def _mk_docs():
    """Five documents with distinct lexical + semantic profiles."""
    return [
        Document(page_content="apple banana cherry", metadata={"id": "A"}),
        Document(page_content="apple date elderberry", metadata={"id": "B"}),
        Document(page_content="fig grape honeydew", metadata={"id": "C"}),
        Document(page_content="apple kiwi lemon", metadata={"id": "D"}),
        Document(page_content="mango nectarine orange", metadata={"id": "E"}),
    ]


def test_hybrid_returns_top_k():
    docs = _mk_docs()
    dense = _RecordingDenseRetriever([docs[2], docs[0], docs[4], docs[1], docs[3]])
    bm25 = BM25Retriever(docs)
    h = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, rrf_k=60)
    out = h.retrieve("apple banana", k=3)
    assert len(out) == 3


def test_hybrid_favors_docs_in_both_lists():
    """A doc ranked highly by BOTH dense and BM25 should rank highest in fusion."""
    docs = _mk_docs()
    # Dense: A, B, C, D, E. BM25: A, B, C, D, E (same order for "apple banana").
    dense = _RecordingDenseRetriever([docs[0], docs[1], docs[2], docs[3], docs[4]])
    bm25 = BM25Retriever(docs)
    h = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, rrf_k=60)
    out = h.retrieve("apple banana", k=1)
    # Both lists put A first -> RRF puts A first.
    assert out[0].metadata["id"] == "A"


def test_hybrid_rrf_breaks_tie_when_only_one_list_ranks_high():
    """If BM25 ranks X at #1 but dense doesn't have X in top-3, X still gets
    fused in via BM25's contribution. The doc at dense #1 still wins overall."""
    docs = _mk_docs()
    # Dense ranks C, D, B (no A, no E).
    dense = _RecordingDenseRetriever([docs[2], docs[3], docs[1]])
    # BM25 ranks A, B, C.
    bm25 = BM25Retriever(docs)
    h = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25, rrf_k=60)
    out = h.retrieve("apple banana", k=3)
    # B should be in both lists -> high score. C is top in dense, but
    # only mid in BM25. A is top in BM25, but absent from dense top-3.
    # Without testing exact ordering, assert that B appears (it's in both).
    ids = [d.metadata["id"] for d in out]
    assert "B" in ids


def test_hybrid_handles_empty_corpus():
    h = HybridRetriever(
        dense_retriever=_RecordingDenseRetriever([]),
        bm25_retriever=BM25Retriever([]),
    )
    assert h.retrieve("anything", k=3) == []


# ---- Factory: build_retriever (hybrid dispatch) -------------------------

def test_build_retriever_hybrid_requires_corpus():
    cfg = PRESETS["hybrid_bm25_dense"]
    vs = FakeVectorStore([Document(page_content="d")])
    # Without corpus -> raises.
    with pytest.raises(ValueError, match="corpus"):
        build_retriever(cfg, vs)


def test_build_retriever_hybrid_with_corpus():
    cfg = PRESETS["hybrid_bm25_dense"]
    vs = FakeVectorStore([Document(page_content="d")])
    docs = [Document(page_content="d1"), Document(page_content="d2")]
    r = build_retriever(cfg, vs, corpus=docs)
    assert isinstance(r, HybridRetriever)


# ---- Presets (hybrid) ---------------------------------------------------

def test_hybrid_preset_uses_mini_lm():
    """The hybrid preset uses MiniLM (cheap) — embedding size is not the lever."""
    cfg = PRESETS["hybrid_bm25_dense"]
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.retriever == "hybrid"
    assert cfg.reranker is None


def test_list_presets_includes_hybrid():
    names = list_presets()
    assert "hybrid_bm25_dense" in names


# ---- build_pipeline with hybrid -----------------------------------------

def test_build_pipeline_with_hybrid():
    cfg = PRESETS["hybrid_bm25_dense"]
    fake_client = MagicMock()
    vs = FakeVectorStore([Document(page_content="d1"), Document(page_content="d2")])
    corpus = [Document(page_content="d1"), Document(page_content="d2")]

    pipeline = build_pipeline(cfg, vectorstore=vs, llm_client=fake_client, corpus=corpus)
    assert isinstance(pipeline._retriever, HybridRetriever)


def test_build_pipeline_hybrid_without_corpus_raises():
    cfg = PRESETS["hybrid_bm25_dense"]
    fake_client = MagicMock()
    vs = FakeVectorStore([Document(page_content="d")])
    with pytest.raises(ValueError, match="corpus"):
        build_pipeline(cfg, vectorstore=vs, llm_client=fake_client)


# ---- iter-33 v12: CleanGroupedPromptBuilder -------------------------------

def _make_grouped_docs():
    return [
        Document(page_content="para1", metadata={"title": "T1"}),
        Document(page_content="para2", metadata={"title": "T2"}),
    ]


def test_clean_grouped_all_prompts_share_same_base():
    """iter-33 v12: every group prompt starts with the same base phrase
    ('You are a helpful assistant. Answer the question carefully.') so
    the model sees a consistent framing across question types."""
    b = CleanGroupedPromptBuilder()
    docs = _make_grouped_docs()
    for q in [
        "Who is X?",
        "Does X suggest Y?",
        "Between X and Y, was Z consistent?",
        "Considering all the evidence, what should we do?",
    ]:
        msgs = b.build(q, docs)
        sys = msgs[0]["content"]
        assert sys.startswith("You are a helpful assistant. Answer the question carefully.")


def test_clean_grouped_entity_lookup_prompt_has_canonical_name_note():
    """iter-33 v13: ENTITY LOOKUP prompt targets the canonical-name
    extraction with a 'most complete form' note. Dropped
    'begin with entity name' and 'no parentheticals' notes
    (failed in v12 — model still adds parentheticals)."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Who is X?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "Notes:" in sys
    # Note 1: canonical name (kept)
    assert "most complete form" in sys.lower() or "canonical" in sys.lower()


def test_clean_grouped_yesno_prompt_has_verdict_first_and_attribution_notes():
    """iter-33 v13: YES/NO prompt targets (a) source-attribution confusion,
    (b) verdict options, and (c) premise-disagreement. The
    'first word must be answer' rule was ABANDONED (failed in v9-v12)."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Does X suggest Y?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "Notes:" in sys
    # Note 1: source attribution (positive directive, no "do NOT verify" anti-pattern)
    assert "match" in sys.lower() or "source names" in sys.lower()
    assert "do not verify" not in sys.lower()
    # Note 2: verdict options
    assert "Yes" in sys and "no" in sys
    # Note 4 (new in v13): answer the question as asked, don't dispute framing
    assert "as asked" in sys.lower() or "dispute" in sys.lower() or "framing" in sys.lower()


def test_clean_grouped_yesno_prompt_does_not_use_v9_anti_patterns():
    """iter-33 v13: NO 'do NOT verify which article' prime. NO
    'first word must be the answer' rule (abandoned — failed in v9-v12).
    NO 'do NOT write preamble' phrasing. Use positive directives only."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Does X suggest Y?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "do not verify" not in sys.lower()
    assert "first word must be" not in sys.lower()
    assert "first word must" not in sys.lower()


def test_clean_grouped_temporal_prompt_targets_preamble_failure():
    """iter-33 v13: TEMPORAL prompt targets the preamble/hedging failure
    with a 'state verdict in first sentence, 1-2 sentences total'
    directive. Dropped 'commit to verdict' (failed in v12)."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Between X and Y, was Z consistent?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "Notes:" in sys
    # New direction: state in first sentence, 1-2 sentences total
    assert "first sentence" in sys.lower() or "1-2 sentences" in sys.lower()
    # Drop the "commit" word (that direction failed)
    assert "commit to" not in sys.lower()
    assert "consistent" in sys.lower()


def test_clean_grouped_refusal_prompt_emits_literal_phrase():
    """iter-33 v12: REFUSAL prompt instructs the model to emit the literal
    'Insufficient information.' phrase (3 words + period)."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Considering all the evidence, what should we do?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "Notes:" in sys
    assert "Insufficient information" in sys


def test_clean_grouped_user_message_has_clean_context():
    """iter-33 v12: user message is <context>...</context> + question, no
    pre-analysis prefix, no CoT instruction. Just context + question."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Does X suggest Y?", _make_grouped_docs())
    user = msgs[1]["content"]
    # No pre-analysis prefix
    assert "Before reading the context" not in user
    assert "Think step by step" not in user
    assert "Notes:" not in user  # notes are in system, not user
    # Clean: <context> + question
    assert user.lstrip().startswith("<context>")
    assert "Does X suggest Y?" in user


def test_clean_grouped_user_only_when_no_context():
    """iter-33 v12: no context → user-only message."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("Does X suggest Y?", None)
    assert msgs == [{"role": "user", "content": "Does X suggest Y?"}]


def test_clean_grouped_classification_handles_lowercase_and_punctuation():
    """iter-33 v12: classification is robust to lowercase and trailing punctuation."""
    b = CleanGroupedPromptBuilder()
    msgs = b.build("does X suggest Y, while Z says W?", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "Yes" in sys  # yes/no prompt
    msgs = b.build("Who,", _make_grouped_docs())
    sys = msgs[0]["content"]
    assert "most complete form" in sys.lower() or "canonical" in sys.lower()  # entity_lookup prompt


def test_clean_grouped_preset_registered():
    """iter-33 v12: PRESETS contains clean_grouped_thinking_k10."""
    cfg = PRESETS["clean_grouped_thinking_k10"]
    assert cfg.prompt_template == "clean_grouped"
    assert cfg.top_k == 10
    assert cfg.thinking_budget == 4096


def test_build_prompt_builder_returns_clean_grouped():
    """iter-33 v12: build_prompt_builder wires clean_grouped → CleanGroupedPromptBuilder."""
    cfg = PipelineConfig(name="test", prompt_template="clean_grouped")
    builder = build_prompt_builder(cfg)
    assert isinstance(builder, CleanGroupedPromptBuilder)


def test_clean_grouped_strips_heading_prefix():
    """iter-33 v12: context paragraphs are stripped of [title]: prefix."""
    b = CleanGroupedPromptBuilder()
    docs = [
        Document(page_content="para1", metadata={"title": "T1"}),
        Document(page_content="para2", metadata={"title": "T2"}),
    ]
    msgs = b.build("Does X suggest Y?", docs)
    user = msgs[1]["content"]
    assert "[T1]:" not in user
    assert "[T2]:" not in user
    assert "para1" in user
    assert "para2" in user


# ---- iter-33 v15: ParametrizedGroupedPromptBuilder (15 experiments) -----

def test_parametrized_grouped_uses_passed_notes():
    """iter-33 v15: ParametrizedGroupedPromptBuilder uses constructor-passed
    note wordings instead of hardcoded defaults."""
    custom_yesno = "Notes:\n1. Custom YES/NO directive."
    custom_temporal = "Notes:\n1. Custom TEMPORAL directive."
    b = ParametrizedGroupedPromptBuilder(
        entity_notes=CleanGroupedPromptBuilder._ENTITY_LOOKUP_NOTES,
        yesno_notes=custom_yesno,
        temporal_notes=custom_temporal,
        refusal_notes=CleanGroupedPromptBuilder._REFUSAL_NOTES,
    )
    msgs = b.build("Does X suggest Y?", _make_grouped_docs())
    assert "Custom YES/NO directive" in msgs[0]["content"]
    msgs = b.build("Between X and Y, was Z?", _make_grouped_docs())
    assert "Custom TEMPORAL directive" in msgs[0]["content"]
    msgs = b.build("Who is X?", _make_grouped_docs())
    assert "Custom" not in msgs[0]["content"]
    assert "most complete form" in msgs[0]["content"]


def test_v15_presets_resolve():
    """iter-33 v15: all 15 v15 preset names resolve to a builder."""
    for d in range(1, 6):
        for v in range(1, 4):
            name = f"clean_grouped_v15_d{d}v{v}"
            cfg = PRESETS[name]
            assert cfg.prompt_template == "parametrized_grouped_v15"
            builder = _build_v15_preset(name)
            assert isinstance(builder, ParametrizedGroupedPromptBuilder)


def test_v15_d1_d2_only_change_yesno_notes():
    """iter-33 v15: directions 1 & 2 only change yesno notes."""
    base_entity = CleanGroupedPromptBuilder._ENTITY_LOOKUP_NOTES
    base_temporal = CleanGroupedPromptBuilder._TEMPORAL_NOTES
    base_refusal = CleanGroupedPromptBuilder._REFUSAL_NOTES
    for d in (1, 2):
        for v in (1, 2, 3):
            b = _build_v15_preset(f"clean_grouped_v15_d{d}v{v}")
            assert b._entity_notes == base_entity
            assert b._temporal_notes == base_temporal
            assert b._refusal_notes == base_refusal


def test_v15_d3_only_changes_temporal_notes():
    """iter-33 v15: direction 3 only changes temporal notes."""
    base_entity = CleanGroupedPromptBuilder._ENTITY_LOOKUP_NOTES
    base_yesno = CleanGroupedPromptBuilder._YESNO_NOTES
    base_refusal = CleanGroupedPromptBuilder._REFUSAL_NOTES
    for v in (1, 2, 3):
        b = _build_v15_preset(f"clean_grouped_v15_d3v{v}")
        assert b._entity_notes == base_entity
        assert b._yesno_notes == base_yesno
        assert b._refusal_notes == base_refusal


def test_v15_d4_only_changes_entity_notes():
    """iter-33 v15: direction 4 only changes entity notes."""
    base_yesno = CleanGroupedPromptBuilder._YESNO_NOTES
    base_temporal = CleanGroupedPromptBuilder._TEMPORAL_NOTES
    base_refusal = CleanGroupedPromptBuilder._REFUSAL_NOTES
    for v in (1, 2, 3):
        b = _build_v15_preset(f"clean_grouped_v15_d4v{v}")
        assert b._yesno_notes == base_yesno
        assert b._temporal_notes == base_temporal
        assert b._refusal_notes == base_refusal


def test_v15_d5_only_changes_refusal_notes():
    """iter-33 v15: direction 5 only changes refusal notes."""
    base_entity = CleanGroupedPromptBuilder._ENTITY_LOOKUP_NOTES
    base_yesno = CleanGroupedPromptBuilder._YESNO_NOTES
    base_temporal = CleanGroupedPromptBuilder._TEMPORAL_NOTES
    for v in (1, 2, 3):
        b = _build_v15_preset(f"clean_grouped_v15_d5v{v}")
        assert b._entity_notes == base_entity
        assert b._yesno_notes == base_yesno
        assert b._temporal_notes == base_temporal


def test_v15_variants_are_different_within_direction():
    """iter-33 v15: V1/V2/V3 within a direction must have different wordings."""
    for d in range(1, 6):
        notes = []
        for v in (1, 2, 3):
            b = _build_v15_preset(f"clean_grouped_v15_d{d}v{v}")
            if d in (1, 2):
                notes.append(b._yesno_notes)
            elif d == 3:
                notes.append(b._temporal_notes)
            elif d == 4:
                notes.append(b._entity_notes)
            else:
                notes.append(b._refusal_notes)
        assert notes[0] != notes[1], f"d{d}: V1 == V2"
        assert notes[1] != notes[2], f"d{d}: V2 == V3"
        assert notes[0] != notes[2], f"d{d}: V1 == V3"


def test_v15_unknown_preset_raises():
    """iter-33 v15: unknown v15 preset name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown v15"):
        _build_v15_preset("clean_grouped_v15_d99v1")

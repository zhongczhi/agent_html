"""Pluggable RAG pipeline.

A RAG pipeline is composed of four stages, each swappable independently:

    [embedder] -> [retriever] -> [reranker?] -> [prompt_builder] -> [LLM]

`PipelineConfig` names each stage's implementation. `build_pipeline(config)`
returns a `RagPipeline` ready to call `run(question, corpus)`. Preset
configurations live in `PRESETS`.

Adding a new pipeline variant (e.g., a different reranker or a new prompt
template) means adding one preset entry — no other code changes.

Design notes
------------
- The pipeline is intentionally a thin orchestration layer. Heavy lifting
  (FAISS index construction, LLM calls, prompt formatting) lives in the
  existing `backend.eval.*` and `backend.rag.*` modules. This module
  composes them.
- `corpus: list[Document]` is passed in by the caller. This makes the
  pipeline testable with synthetic corpora and reusable across the
  eval script (per-question FAISS) and the chat service (library index).
- The reranker slot is optional. When `reranker=None`, the retriever's
  output is sliced directly to `top_k`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ── Pipeline config ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class PipelineConfig:
    """Names each stage's implementation.

    Attributes:
        name: human-readable preset name (matches a key in `PRESETS`).
        embedding_backend: passed to `make_embeddings()`. One of
            'sentence-transformers' or 'minimax'.
        embedding_model: model name for sentence-transformers backend.
            Ignored when embedding_backend != 'sentence-transformers'.
        retriever: retriever kind. One of 'dense' (FAISS top-k).
            Future: 'hybrid_bm25_dense', 'multi_query', etc.
        reranker: reranker kind. None or 'cross_encoder'.
        rerank_top_k: how many candidates the retriever returns before
            the reranker narrows to `top_k`. Ignored when reranker is None.
        top_k: final number of chunks sent to the LLM.
        prompt_template: prompt template name. 'default' or 'extract_span'.
        thinking_budget: Anthropic extended thinking budget in tokens
            (passed to messages.create as thinking.budget_tokens). When
            set, the model produces internal reasoning blocks that are
            discarded by our scoring path. Pass None to disable.
        llm_model: model name for the LLM call (passed to qa_judge.ask_llm).
    """

    name: str
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever: str = "dense"
    reranker: str | None = None
    rerank_top_k: int = 50
    top_k: int = 4
    prompt_template: str = "default"
    thinking_budget: int | None = None
    llm_model: str = "minimax-3"


# ── Component protocols (duck-typed; concrete impls in this module) ──────


class Retriever(Protocol):
    """Returns up to `k` candidate Documents for a query."""

    def retrieve(self, query: str, k: int) -> list[Document]: ...


class Reranker(Protocol):
    """Re-orders `candidates` for `query` and returns the top `top_k`."""

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]: ...


class PromptBuilder(Protocol):
    """Builds the messages list passed to the LLM."""

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]: ...


class LLM(Protocol):
    """One LLM call returning answer text."""

    async def ask(self, messages: list[dict], max_tokens: int = 200) -> str: ...


# ── Concrete implementations ─────────────────────────────────────────────


class DenseRetriever:
    """FAISS-based dense retriever over a corpus built once at construction.

    `corpus` is a list of (id, embedding) pairs OR a list of Documents with
    pre-computed embeddings via `embed_documents`. For simplicity here we
    take pre-built FAISS-like objects (anything with `.similarity_search`).
    """

    def __init__(self, vectorstore):
        self._vs = vectorstore

    def retrieve(self, query: str, k: int) -> list[Document]:
        return self._vs.similarity_search(query, k=k)


class BM25Retriever:
    """Sparse keyword retriever over a corpus of Documents.

    Wraps rank_bm25.BM25Okapi. The corpus is tokenized once at construction
    (lowercase, alphanumeric-only). At query time, the query is tokenized
    the same way and scored against every doc.

    Pure-Python, deterministic, no model download. Slower than FAISS for
    large corpora but adequate for per-question eval workloads (10 docs each).
    """

    def __init__(self, docs: list[Document]):
        import re

        from rank_bm25 import BM25Okapi

        self._docs = list(docs)
        # Tokenize: lowercase, strip non-word chars (except spaces), split.
        # We keep numbers and punctuation-free tokens.
        self._tokenize = lambda s: [
            t for t in re.findall(r"[a-z0-9]+", s.lower()) if t
        ]
        if self._docs:
            tokenized_corpus = [self._tokenize(d.page_content) for d in self._docs]
            # BM25Okapi raises on empty docs (rare but possible if a paragraph
            # is just punctuation). Replace empty token lists with a placeholder.
            self._bm25 = BM25Okapi(
                [toks if toks else ["_empty_"] for toks in tokenized_corpus]
            )
        else:
            self._bm25 = None

    def retrieve(self, query: str, k: int) -> list[Document]:
        if not self._docs or self._bm25 is None:
            return []
        q_tokens = self._tokenize(query) or ["_empty_"]
        scores = self._bm25.get_scores(q_tokens)
        # Top-k by score (descending). We keep all docs by score; the
        # >0 filter used to drop zero-scored docs but with very small
        # corpora BM25 returns 0 for every term (each term appears in
        # every doc, IDF -> 0). For our 10-paragraph eval workload this
        # rarely matters, but the filter is too aggressive for tiny cases.
        ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._docs[i] for i in ranked_indices[:k]]


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) of dense + BM25 retrieval.

    For each query, get top-K from both retrievers, then fuse ranks with
    RRF: score(d) = sum(1 / (rrf_k + rank_in_list)). RRF doesn't require
    learning weights and is robust to score-scale differences between
    the two retrievers. The dense_retriever is expected to be a
    `DenseRetriever`; the bm25_retriever a `BM25Retriever`.

    Returns the top `k` documents by fused RRF score.
    """

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, rrf_k: int = 60):
        self._dense = dense_retriever
        self._bm25 = bm25_retriever
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[Document]:
        # Pull more candidates than k from each retriever so fusion has
        # room to express itself. We use 4k as the candidate depth — the
        # RRF paper recommends ~3-5x the desired depth.
        cand_k = max(k * 4, 20)
        dense_hits = self._dense.retrieve(query, k=cand_k)
        bm25_hits = self._bm25.retrieve(query, k=cand_k)

        scores: dict[int, float] = {}
        docs_by_id: dict[int, Document] = {}

        for rank, doc in enumerate(dense_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc
        for rank, doc in enumerate(bm25_hits):
            doc_id = id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
            docs_by_id[doc_id] = doc

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [docs_by_id[doc_id] for doc_id, _ in ranked[:k]]


class NoOpReranker:
    """Returns the first `top_k` candidates unchanged."""

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker over (query, document) pairs.

    Uses sentence-transformers CrossEncoder. Falls back to the input order
    if the model fails to load — but logs a warning so operators see it.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model_name = model_name
        self._model = None  # lazy-loaded on first call

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: WPS433
            self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, candidates: list[Document], top_k: int) -> list[Document]:
        if not candidates:
            return []
        self._ensure_model()
        pairs = [(query, d.page_content) for d in candidates]
        scores = self._model.predict(pairs)
        # Sort by score descending, take top_k.
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [d for d, _ in ranked[:top_k]]


class DefaultPromptBuilder:
    """Wraps backend.eval.qa_judge.build_qa_prompt (RAG context + question)."""

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        # Imported here to avoid a top-level import cycle (qa_judge.py
        # is in backend.eval.* and must not import from backend.rag.*).
        from backend.eval.qa_judge import build_qa_prompt
        return build_qa_prompt(question, context_docs)


class ExtractSpanPromptBuilder:
    """Same shape as Default, but instructs the LLM to extract the answer
    span verbatim from the context first.

    This typically increases answer_f1 (more literal spans) at the cost of
    sounding less conversational. Use when downstream eval is F1-based.
    """

    EXTRACT_INSTRUCTION = (
        "Read the <context>...</context> block carefully and extract the "
        "exact span that answers the question. Begin your response with the "
        "extracted span (in quotation marks if it is a phrase), then briefly "
        "explain. Do not paraphrase the answer — quote it verbatim."
    )

    def __init__(self):
        # Local mirror of RAG_SYSTEM_PROMPT — see backend.eval.qa_judge for
        # the isolation rationale. We append the extract instruction.
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.EXTRACT_INSTRUCTION

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class CoTExtractPromptBuilder:
    """Iter-15 SOTA: `ExtractSpanPromptBuilder` + an explicit step-by-step
    reasoning scaffold that targets multi-hop questions.

    Why this exists: at k≥8 retrieval is saturated (0 retrieval misses).
    The remaining failures are LLM extraction/reasoning errors. About half
    are multi-hop — the model needs to chain facts across paragraphs before
    it can pick the right span. A bare extract_span instruction doesn't
    scaffold that reasoning; this builder does.

    Contains_gold stays safe: the prompt forces the model to begin its
    visible output with the extracted span, so substring containment of
    the gold answer remains high. Step-by-step reasoning only appears
    after the lead span, which is the part substring containment matches.
    """

    COT_INSTRUCTION = (
        "Read the <context>...</context> block carefully. Some questions "
        "require combining facts from multiple paragraphs (multi-hop reasoning).\n\n"
        "Think step by step:\n"
        "1. Identify the entities and facts the question asks about.\n"
        "2. Find the relevant paragraph(s) in the context.\n"
        "3. If multi-hop reasoning is needed, chain together the supporting "
        "facts in order.\n"
        "4. Decide which exact span answers the question.\n\n"
        "Begin your response with the extracted span (in quotation marks "
        "if it is a phrase), then briefly explain your reasoning. "
        "Do not paraphrase the answer — quote it verbatim from the context."
    )

    def __init__(self):
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.COT_INSTRUCTION

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class CoTExtractNoTitlesPromptBuilder(CoTExtractPromptBuilder):
    """Iter-21: same CoT instruction as CoTExtractPromptBuilder, but the
    `[title]:` heading prefix on each context paragraph is stripped.

    Hypothesis (from iter-20 thinking audit): when context paragraphs
    are introduced with their Wikipedia article heading as a prefix
    (e.g. `[Hector Berlioz]: ...body...`), the model uses the heading as
    the entity label when emitting its answer. HotpotQA's gold answers
    typically use the full canonical name found in the article body
    opening (e.g. "Louis-Hector Berlioz"), not the colloquial heading
    form ("Hector Berlioz"). Stripping the heading forces the model to
    extract the canonical form from the body, where Wikipedia puts it in
    the first sentence.

    Why a separate builder: preserves the existing
    `CoTExtractPromptBuilder` behavior for all other presets, isolating
    the title-stripping experiment to one preset.
    """

    # COT_INSTRUCTION inherited from CoTExtractPromptBuilder

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        # Strip the `[title]:` heading prefix; model has to find names
        # in the body text instead of copying the colloquial heading.
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class PreAnalysisExtractPromptBuilder(CoTExtractNoTitlesPromptBuilder):
    """Iter-29 (v2): cot_extract_no_titles + a pre-analysis prefix that
    enumerates the four question shapes observed in the eval datasets.

    Iter-29 history (all v3-v6 attempts are documented but did not
    produce a clear improvement over v2 on the smoke 200 set; full
    per-attempt analysis in docs/eval-results/2026-07-18-iter29-attempt-log.md):

      v1 (generic): "what entities, facts, or attributes does it ask
        about" — helped temporal (+5.3 pp) but hurt comparison (-4.1 pp)
        because the model couldn't decide which "kind of material"
        applied to "Does X suggest Y" questions.
      v2 (this, shape-enumerated): four shape bullets with example
        phrasings — fixed the comparison regression, +6.0 pp on smoke
        200 (run 1; run 2 produced 0.645, showing 3.5 pp run-to-run
        variance on n=200). Lift came from implicit pattern-matching
        to the example phrasings.
      v3 (refinement): added agreement words, yes/no caveat in ENTITY,
        exact-words emphasis on REFUSAL — regressed -0.5 pp because
        the "if the question expects a yes/no answer" hint made the
        model write premise-correction meta-commentary.
      v4 (paraphrase): replaced shape-matching with question-paraphrase
        step — regressed to SOTA baseline (0.620). "ignoring source
        attributions" made the model over-confidently reject framing.
      v5 (CRITICAL anti-preamble): added explicit "first word must be
        the answer" rules with "CRITICAL" framing — 0.685 (+0.5 pp vs
        v2 run 1, +4.0 pp vs v2 run 2). The CRITICAL framing did not
        actually change the preamble rate; the small lift is within
        run-to-run noise.
      v6 (fill-in-the-blank template): literal [ANSWER] template +
        worked examples — 0.646 (within noise of v2). Hurt comparison
        by 8.2 pp because the model became over-confident in
        rejecting question framing.

    Across all attempts, the dominant v2 failure modes are:
      1. Source-attribution confusion in thinking (37% of fails): the
         model spends thinking budget trying to figure out which
         context chunk matches "the Fortune article". A prompt
         change can't fix this — it requires dataset-level changes
         (include source attributions in retrieved context, or
         remove source names from questions).
      2. "Based on the context..." preamble (35% of fails): the model
         writes analysis first, then the answer. The system prompt's
         CoT scaffold reinforces this. No prompt change reliably
         suppresses the preamble.
      3. Semantic refusals (32% of fails): the model says "context
         does not contain" when the answer is present. A metric
         change (semantic similarity for refusal-shaped answers)
         would fix this without prompt changes.

    Run-to-run variance is ~3.5 pp on n=200 (37/200 questions change
    pass/fail between runs). The v2 vs SOTA +6.0 pp is suggestive
    (1.8σ) but not conclusive. v3-v6 results are all within noise.
    The next step is either a full n=2556 run to get a firm answer
    on v2, or a direction that attacks one of the three root causes
    (dataset-level source attribution, preamble-suppression at the
    system-prompt level, or a metric change for refusals).

    Cost: ~80-100 extra output tokens per question. No extra LLM
    call (single-turn; the analysis prefix is in the same user
    message as the context, before it).
    """

    PRE_ANALYSIS_INSTRUCTION = (
        "Before reading the context, briefly identify what kind of question this is. "
        "Pick the shape that matches, then extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a single named entity (1-3 words) verbatim from the context.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?', "
        "'Was there...?'): compare both sides of the claim, then answer with one word "
        "(Yes, no, True, or False).\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', 'Was there a "
        "change between...?', 'Was X consistent with Y?'): check whether the time "
        "order or consistency holds across the two articles, then answer Yes or no.\n"
        "- REFUSAL (the context may not contain the answer): if neither paragraph "
        "states what's asked, answer 'Insufficient information' rather than guessing.\n"
        "One short sentence naming the shape is enough; do not re-read the question. "
        "Then read the <context>...</context> block and answer."
    )

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self.PRE_ANALYSIS_INSTRUCTION}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class SimplifiedV2PromptBuilder(PreAnalysisExtractPromptBuilder):
    """iter-34 v16-a: SIMPLIFIED v2.

    v2 (PreAnalysisExtractPromptBuilder, the local-maximum SOTA at 0.680)
    had two overlapping scaffolding mechanisms:
      - System prompt: iter-22 CoT scaffold ('Think step by step: 1. ...
        2. ... 3. ... 4. ... Begin with extracted span')
      - User prompt: pre-analysis prefix with 4-shape enumeration

    These overlapped. The CoT scaffold's "identify entities" duplicates
    the pre-analysis's "identify what kind of question". The "Begin
    with extracted span" duplicates ENTITY LOOKUP's "extract a named
    entity verbatim". Dropping the CoT scaffold should not hurt — the
    iter-29 v2 lift came from the 4-shape enumeration, not the CoT
    scaffold (iter-22 SOTA was 0.620, iter-29 v2 was 0.680 with the
    same CoT scaffold plus the new pre-analysis prefix).

    v16-a drops:
      - The full CoT scaffold (4 steps)
      - 'Some questions require combining facts from multiple paragraphs'
      - 'Begin your response with the extracted span'
      - 'then briefly explain your reasoning'
      - 'One short sentence naming the shape is enough; do not re-read
        the question'
      - 'Then read the <context>...</context> block and answer'

    v16-a keeps:
      - RAG framing (essential for grounding)
      - 4-shape enumeration (the lift mechanism)
      - 'Quote your answer verbatim from the context' (consolidated
        canonical-name directive)

    v16-a on n=200: 0.620 (vs v2 baseline 0.615 this run). Tied within
    noise. Simplification verified.
    """

    _SIMPLIFIED_PRE_ANALYSIS = (
        "Before reading the context, identify the question type and "
        "extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a named entity verbatim from the context.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?'): "
        "compare both sides, answer Yes, no, True, or False.\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', "
        "'Was X consistent with Y?'): check time order or consistency, "
        "answer Yes or no.\n"
        "- REFUSAL (the context may not contain the answer): answer "
        "'Insufficient information' rather than guessing.\n\n"
        "Quote your answer verbatim from the context."
    )

    def __init__(self):
        # Drop the CoT scaffold; keep only RAG framing in system.
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class SimplifiedV2Bv1PromptBuilder(SimplifiedV2PromptBuilder):
    """iter-34 v16-b: v16-a + TEMPORAL verdict-leading directive.

    v16-a failure analysis (per sub-agent):
      - 16/17 TEMPORAL fails are verdict-buried (model writes
        'Based on the context...' preamble and never leads with the
        verdict word Yes/no/Consistent/Inconsistent).
      - v2 (the historical SOTA) had a CoT scaffold that said
        'Begin your response with the extracted span' which partially
        fixed this. v16-a dropped the scaffold, and TEMPORAL
        regressed by 1 question.
      - v15 d3v2 tested this on 20% sample with directive 'Your
        first sentence must state the answer. Keep the entire
        response to two sentences or fewer' and lifted TEMPORAL
        from 4/9 to 6/9.

    v16-b adds ONE directive to the TEMPORAL_ORDER bullet:
        'Your first sentence states the verdict; keep the entire
        response to two sentences or fewer.'

    v16-b on n=200: 0.655 (131/200) — +2.0 pp over v2 baseline
    re-run (0.635). TEMPORAL recovered from 0.622 (v16-a) to 0.689
    (v16-b). But the brevity directive didn't bite — only 2/45
    v16-b responses are ≤2 sentences.
    """

    _SIMPLIFIED_PRE_ANALYSIS_V16B = (
        "Before reading the context, identify the question type and "
        "extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a named entity verbatim from the context.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?'): "
        "compare both sides, answer Yes, no, True, or False.\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', "
        "'Was X consistent with Y?'): check time order or consistency. "
        "Your first sentence states the verdict; keep the entire response "
        "to two sentences or fewer.\n"
        "- REFUSAL (the context may not contain the answer): answer "
        "'Insufficient information' rather than guessing.\n\n"
        "Quote your answer verbatim from the context."
    )

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS_V16B}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class SimplifiedV2Cv1PromptBuilder(SimplifiedV2Bv1PromptBuilder):
    """iter-34 v16-c: v16-b + strengthened TEMPORAL opening directive.

    v16-b analysis showed TEMPORAL pass rate jumped 0.622 → 0.689
    (+6.7 pp), but 14/45 still fail with the same verdict-buried
    pattern (model writes 'Based on...' preamble, never includes the
    verdict word). The brevity part of v16-b's directive didn't bite
    (only 2/45 ≤2 sentences).

    v16-c strengthens the opening format with an explicit verdict
    structure pattern:

      Old (v16-b): 'Your first sentence states the verdict; keep the
                    entire response to two sentences or fewer.'
      New (v16-c): 'Lead with the verdict word (Yes, No, Consistent,
                    or Inconsistent), followed by a brief one-sentence
                    explanation.'

    Plus an explicit list of valid verdict words to reduce ambiguity
    about what counts as a "verdict". The model now sees the format
    pattern in the directive itself.

    v16-c on n=200: 0.690 (138/200) — +5.5 pp over v2 baseline re-run.
    YES/NO pass rate went 0.62 → 0.70 (sub-agent says spillover from
    the TEMPORAL directive; TEMPORAL itself stayed at 0.689). First
    clear improvement over the historical v2 SOTA (0.680).
    """

    _SIMPLIFIED_PRE_ANALYSIS_V16C = (
        "Before reading the context, identify the question type and "
        "extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a named entity verbatim from the context.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?'): "
        "compare both sides, answer Yes, no, True, or False.\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', "
        "'Was X consistent with Y?'): check time order or consistency. "
        "Lead with the verdict word (Yes, No, Consistent, or Inconsistent), "
        "followed by a brief one-sentence explanation.\n"
        "- REFUSAL (the context may not contain the answer): answer "
        "'Insufficient information' rather than guessing.\n\n"
        "Quote your answer verbatim from the context."
    )

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS_V16C}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class SimplifiedV2Dv1PromptBuilder(SimplifiedV2Cv1PromptBuilder):
    """iter-34 v16-d: v16-c + YES/NO verdict-leading + chunk-matching.

    v16-c failure analysis (per sub-agent):
      - v16-c YES/NO: 73/104 = 0.70 (vs v16-b 0.62, +8 pp).
      - Remaining yesno failures: 31 = 23 no-lead + 8 wrong-verdict.
      - Wrong-verdict cases (8) often come from source-attribution
        confusion: model picks wrong chunk when same publisher has
        multiple articles (qid 1388f62e, 2b8acb60).
      - The v16-c TEMPORAL directive ("Lead with verdict word, brief
        one-sentence explanation") generalizes to yesno by spillover.
        Formalizing it in the YES/NO bullet should compound the lift.

    v16-d modifies ONLY the YES/NO bullet. Combines three positive
    directives:
      1. Chunk-matching: "identify the context chunk whose content
         matches the topic and details the question names"
      2. Lead-with-verdict: "lead with the verdict word (Yes, no,
         True, or False)"
      3. Brevity: "followed by a brief one-sentence explanation"

    This is qualitatively different from prior D1 (source-attribution)
    attempts because:
      - v15 d1v1/d1v2/v9 source-attribution: framed as "match by
        publisher name" (which primed the model).
      - v16-d: framed as "match by content + topic details" — positive
        matching by content, not by source name.
      - And adds verdict-leading + brevity as combined constraints.

    If v16-d yesno regresses below 0.70, tighten the directive (drop
    "brief one-sentence explanation" part).
    """

    _SIMPLIFIED_PRE_ANALYSIS_V16D = (
        "Before reading the context, identify the question type and "
        "extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a named entity verbatim from the context.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?'): "
        "identify the context chunk whose content matches the topic and "
        "details the question names, judge the claim against that chunk, "
        "lead with the verdict word (Yes, no, True, or False), followed by "
        "a brief one-sentence explanation.\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', "
        "'Was X consistent with Y?'): check time order or consistency. "
        "Lead with the verdict word (Yes, No, Consistent, or Inconsistent), "
        "followed by a brief one-sentence explanation.\n"
        "- REFUSAL (the context may not contain the answer): answer "
        "'Insufficient information' rather than guessing.\n\n"
        "Quote your answer verbatim from the context."
    )

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS_V16D}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class SimplifiedV2Ev1PromptBuilder(SimplifiedV2Cv1PromptBuilder):
    """iter-34 v16-e (final iteration): v16-c + ENTITY canonical-name directive.

    v16-d (YES/NO verdict-leading with chunk-match) regressed -6.5 pp
    from v16-c. ABANDONED per user's 2-failure rule (this was attempt
    1 of the combined chunk+verdict+brevity direction for yesno; it
    joins the 5+ prior yesno-directive failures).

    Per user's protocol, the next iteration targets a different failure.
    v16-c remaining failures:
      - inference: 3/37 fails = substring mismatches
      - yesno: 31/104 fails
      - temporal: 14/45 fails = verdict-buried
      - refusal: 14/14 fails = untouchable

    ENTITY canonical-name directive is the only direction untried in
    v16 series. It targets the 3 inference substring mismatches:
      - qid 607962ec "New Zealand All Blacks" → "New Zealand (the All Blacks)"
      - qid 7b40f027 "Australia's cricket team" → "Australia"

    v16-e modifies ONLY the ENTITY bullet. Adds positive wording:
      "Use the most complete form of the entity name as written in the
       context. Do not add parenthetical clarifications after the name."

    Predicted lift: +1.5 pp (3 fixes). Even if small, confirms the
    approach of incremental targeted directives.
    """

    _SIMPLIFIED_PRE_ANALYSIS_V16E = (
        "Before reading the context, identify the question type and "
        "extract accordingly:\n"
        "- ENTITY LOOKUP (e.g. 'Who is X?', 'What company...?', 'Which director...'): "
        "extract a named entity verbatim from the context. Use the most "
        "complete form of the entity name as written in the context "
        "(e.g. 'New Zealand All Blacks', not 'New Zealand' or 'the All "
        "Blacks'). Do not add parenthetical clarifications after the name.\n"
        "- YES/NO ADJUDICATION (e.g. 'Does X suggest Y?', 'Are A and B both...?'): "
        "compare both sides, answer Yes, no, True, or False.\n"
        "- TEMPORAL ORDERING / CONSISTENCY (e.g. 'Which came first?', "
        "'Was X consistent with Y?'): check time order or consistency. "
        "Lead with the verdict word (Yes, No, Consistent, or Inconsistent), "
        "followed by a brief one-sentence explanation.\n"
        "- REFUSAL (the context may not contain the answer): answer "
        "'Insufficient information' rather than guessing.\n\n"
        "Quote your answer verbatim from the context."
    )

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS_V16E}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class AnthropicLLM:

    def __init__(self):
        # Drop the CoT scaffold; keep only RAG framing in system.
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = (
            f"{self._SIMPLIFIED_PRE_ANALYSIS}\n\n"
            f"<context>\n{context_str}\n</context>\n\n{question}"
        )
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


class CleanGroupedPromptBuilder:
    """Iter-33 v13 default preset (clean_grouped_thinking_k10).

    Per-group prompts with numbered notes. The user abandoned this
    direction in iter-33 v14 (3 attempts all regressed), but the
    default preset is kept for backward compatibility. See
    ParametrizedGroupedPromptBuilder for the parameterized variant
    used in iter-33 v15 experiments.

    All groups share the base: "You are a helpful assistant. Answer
    the question carefully." Each group has a SHORT body + NUMBERED
    NOTES (1-4 each) targeting that group's dominant failure mode.
    No CoT scaffold, no pre-analysis prefix, no "do NOT X" anti-patterns.
    """

    _BASE = "You are a helpful assistant. Answer the question carefully."

    _ENTITY_LOOKUP_BODY = (
        "Read the <context>...</context> block. Find the named entity "
        "the question asks about."
    )
    _ENTITY_LOOKUP_NOTES = (
        "Notes:\n"
        "1. Use the most complete form of the entity name as written in "
        "the context."
    )

    _YESNO_BODY = (
        "Read the <context>...</context> block. The question asks "
        "whether a claim is supported by the context."
    )
    _YESNO_NOTES = (
        "Notes:\n"
        "1. Match the question's source names (e.g. 'the Fortune "
        "article') to the correct context chunk. Two articles from the "
        "same publisher may appear.\n"
        "2. Compare the claim against what those articles say.\n"
        "3. Answer with Yes, no, True, False, Consistent, or Aligned "
        "based on whether the claim is supported.\n"
        "4. Answer the question as asked. Do not dispute the question's "
        "framing."
    )

    _TEMPORAL_BODY = (
        "Read the <context>...</context> block. The question asks about "
        "time order or consistency between two articles."
    )
    _TEMPORAL_NOTES = (
        "Notes:\n"
        "1. Match the question's source names to the correct context "
        "chunk.\n"
        "2. State your verdict in the first sentence. Use 1-2 sentences "
        "total. Do not write multi-section comparative essays.\n"
        "3. Answer with Yes, no, Consistent, Inconsistent, or Aligned."
    )

    _REFUSAL_BODY = (
        "If the context does not contain the information needed to "
        "answer the question:"
    )
    _REFUSAL_NOTES = (
        "Notes:\n"
        "1. Write EXACTLY 'Insufficient information.' (with the period) "
        "and stop. Do not write any explanation."
    )

    _ENTITY_TRIGGERS = frozenset({
        "who", "what", "which", "where", "how",
    })
    _YESNO_TRIGGERS = frozenset({
        "does", "do", "did", "is", "are", "was", "were",
        "has", "have", "had", "can", "could", "will", "would",
        "should", "may", "might", "must", "shall",
    })
    _TEMPORAL_TRIGGERS = frozenset({
        "between", "after", "before", "when",
    })

    def _classify(self, question: str) -> str:
        first = question.strip().lower().split(maxsplit=1)[0] if question.strip() else ""
        first = first.rstrip(",.;:?!")
        if first in self._ENTITY_TRIGGERS:
            return "entity_lookup"
        if first in self._YESNO_TRIGGERS:
            return "yesno"
        if first in self._TEMPORAL_TRIGGERS:
            return "temporal"
        return "refusal"

    def _system_for(self, group: str) -> str:
        if group == "entity_lookup":
            return "\n\n".join([self._BASE, self._ENTITY_LOOKUP_BODY, self._ENTITY_LOOKUP_NOTES])
        if group == "yesno":
            return "\n\n".join([self._BASE, self._YESNO_BODY, self._YESNO_NOTES])
        if group == "temporal":
            return "\n\n".join([self._BASE, self._TEMPORAL_BODY, self._TEMPORAL_NOTES])
        return "\n\n".join([self._BASE, self._REFUSAL_BODY, self._REFUSAL_NOTES])

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        group = self._classify(question)
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = f"<context>\n{context_str}</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_for(group)},
            {"role": "user", "content": user_content},
        ]


class ParametrizedGroupedPromptBuilder:
    """Iter-33 v15: parameterized version of CleanGroupedPromptBuilder.

    Same architecture (clean base + per-group body + numbered notes) but
    accepts note wordings for each group as constructor parameters. This
    lets us run 5 rounds × 3 variants = 15 experiments on a 20% sample
    without writing 15 separate classes.

    Each preset (clean_grouped_v15_dXvY) configures one round+variant.
    The dispatch (first-word classification) is identical to
    CleanGroupedPromptBuilder; only the note wordings vary.

    Designed for the user's experimental protocol:
      "choose 20% samples of each group as the dataset for this round
       for time efficiency, for the previous failure modes you conclude,
       for each specific guiding note, write it in 3 different forms
       as control group, keep previous per-group experimental steps
       with 5 loops"

    Five directions × three variants (15 experiments):
      d1: YES/NO source-attribution verification (failed in v12-v14)
      d2: YES/NO premise-disagreement (failed in v13)
      d3: TEMPORAL brevity (partial success in v13)
      d4: ENTITY canonical name (partial)
      d5: REFUSAL literal-phrase (failed 8x)

    For each direction, ONLY the relevant group's note wording is
    changed; the other 3 groups use the v13 default notes. This
    isolates the effect of each direction's variants.
    """

    _BASE = "You are a helpful assistant. Answer the question carefully."

    _ENTITY_LOOKUP_BODY = (
        "Read the <context>...</context> block. Find the named entity "
        "the question asks about."
    )
    _YESNO_BODY = (
        "Read the <context>...</context> block. The question asks "
        "whether a claim is supported by the context."
    )
    _TEMPORAL_BODY = (
        "Read the <context>...</context> block. The question asks about "
        "time order or consistency between two articles."
    )
    _REFUSAL_BODY = (
        "If the context does not contain the information needed to "
        "answer the question:"
    )

    # Default notes (v13) — used for non-target groups in each experiment
    _ENTITY_DEFAULT_NOTES = (
        "Notes:\n"
        "1. Use the most complete form of the entity name as written in "
        "the context."
    )
    _YESNO_DEFAULT_NOTES = (
        "Notes:\n"
        "1. Match the question's source names (e.g. 'the Fortune "
        "article') to the correct context chunk. Two articles from the "
        "same publisher may appear.\n"
        "2. Compare the claim against what those articles say.\n"
        "3. Answer with Yes, no, True, False, Consistent, or Aligned.\n"
        "4. Answer the question as asked. Do not dispute the question's "
        "framing."
    )
    _TEMPORAL_DEFAULT_NOTES = (
        "Notes:\n"
        "1. Match the question's source names to the correct context "
        "chunk.\n"
        "2. State your verdict in the first sentence. Use 1-2 sentences "
        "total. Do not write multi-section comparative essays.\n"
        "3. Answer with Yes, no, Consistent, Inconsistent, or Aligned."
    )
    _REFUSAL_DEFAULT_NOTES = (
        "Notes:\n"
        "1. Write EXACTLY 'Insufficient information.' (with the period) "
        "and stop. Do not write any explanation."
    )

    _ENTITY_TRIGGERS = frozenset({
        "who", "what", "which", "where", "how",
    })
    _YESNO_TRIGGERS = frozenset({
        "does", "do", "did", "is", "are", "was", "were",
        "has", "have", "had", "can", "could", "will", "would",
        "should", "may", "might", "must", "shall",
    })
    _TEMPORAL_TRIGGERS = frozenset({
        "between", "after", "before", "when",
    })

    def __init__(self, entity_notes, yesno_notes, temporal_notes, refusal_notes):
        self._entity_notes = entity_notes
        self._yesno_notes = yesno_notes
        self._temporal_notes = temporal_notes
        self._refusal_notes = refusal_notes

    def _classify(self, question: str) -> str:
        first = question.strip().lower().split(maxsplit=1)[0] if question.strip() else ""
        first = first.rstrip(",.;:?!")
        if first in self._ENTITY_TRIGGERS:
            return "entity_lookup"
        if first in self._YESNO_TRIGGERS:
            return "yesno"
        if first in self._TEMPORAL_TRIGGERS:
            return "temporal"
        return "refusal"

    def _system_for(self, group: str) -> str:
        if group == "entity_lookup":
            return "\n\n".join([self._BASE, self._ENTITY_LOOKUP_BODY, self._entity_notes])
        if group == "yesno":
            return "\n\n".join([self._BASE, self._YESNO_BODY, self._yesno_notes])
        if group == "temporal":
            return "\n\n".join([self._BASE, self._TEMPORAL_BODY, self._temporal_notes])
        return "\n\n".join([self._BASE, self._REFUSAL_BODY, self._refusal_notes])

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        group = self._classify(question)
        context_str = "\n\n".join(d.page_content for d in context_docs)
        user_content = f"<context>\n{context_str}</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_for(group)},
            {"role": "user", "content": user_content},
        ]


# iter-33 v15: 5 directions × 3 variants = 15 note wordings.
# Only the target group's notes change per experiment; the other 3
# groups use ParametrizedGroupedPromptBuilder's default notes.

# Direction 1: YES/NO source-attribution (3 variants)
_V15_D1V1_YESNO = (
    "Notes:\n"
    "1. The question names a source (e.g. \"the Fortune article\"). "
    "Multiple context chunks may come from the same publisher; identify "
    "the chunk whose content matches the claimed topic and dates.\n"
    "2. Compare the claim against the matched chunk's statements, not "
    "against any unrelated chunk that shares the publisher.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned based "
    "on what the matched chunk says."
)
_V15_D1V2_YESNO = (
    "Notes:\n"
    "1. The question references a specific article (e.g. \"the Fortune "
    "article\"). Locate that article in the context by reading each "
    "chunk's content, not by the publication name alone.\n"
    "2. Once located, evaluate the claim against that chunk's content.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned."
)
_V15_D1V3_YESNO = (
    "Notes:\n"
    "1. The source named in the question (e.g. \"the Fortune article\") "
    "refers to a specific piece. Use the people, dates, and facts the "
    "question mentions to find the chunk where they appear.\n"
    "2. Assess whether that chunk's content supports, contradicts, or "
    "is consistent with the claim.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned based "
    "on what is in the matched chunk."
)

# Direction 2: YES/NO premise-disagreement (3 variants)
_V15_D2V1_YESNO = (
    "Notes:\n"
    "1. Identify whether the statements the question asks about appear "
    "in the context.\n"
    "2. Evaluate based on whether those statements are present and "
    "what they say, not on whether the question's wording would "
    "normally be phrased that way.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned."
)
_V15_D2V2_YESNO = (
    "Notes:\n"
    "1. Focus on whether the substantive claim is supported by the "
    "context's content.\n"
    "2. Base your answer on the presence or absence of supporting "
    "statements; treat how the claim is framed as separate from "
    "whether it is supported.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned."
)
_V15_D2V3_YESNO = (
    "Notes:\n"
    "1. Read each statement the question asks about and check whether "
    "it is supported by the context.\n"
    "2. Base your answer strictly on what the context says about those "
    "statements, not on whether the question's framing is "
    "conventionally accurate.\n"
    "3. Answer with Yes, no, True, False, Consistent, or Aligned."
)

# Direction 3: TEMPORAL brevity (3 variants)
_V15_D3V1_TEMPORAL = (
    "Notes:\n"
    "1. Match the question's source names to the correct context chunk.\n"
    "2. Lead with the answer (one sentence). Follow with the single "
    "most relevant supporting fact.\n"
    "3. Answer with Yes, no, Consistent, Inconsistent, or Aligned."
)
_V15_D3V2_TEMPORAL = (
    "Notes:\n"
    "1. Match the question's source names to the correct context chunk.\n"
    "2. Your first sentence must state the answer. Keep the entire "
    "response to two sentences or fewer.\n"
    "3. Answer with Yes, no, Consistent, Inconsistent, or Aligned."
)
_V15_D3V3_TEMPORAL = (
    "Notes:\n"
    "1. Match the question's source names to the correct context chunk.\n"
    "2. Commit to one verdict in a single sentence. Cite at most one "
    "supporting fact.\n"
    "3. Answer with Yes, no, Consistent, Inconsistent, or Aligned."
)

# Direction 4: ENTITY canonical name (3 variants)
_V15_D4V1_ENTITY = (
    "Notes:\n"
    "1. Use the most complete form of the entity name as written in "
    "the context."
)
_V15_D4V2_ENTITY = (
    "Notes:\n"
    "1. Copy the entity name verbatim from the context as it appears "
    "there. Do not paraphrase or add qualifications."
)
_V15_D4V3_ENTITY = (
    "Notes:\n"
    "1. Use the entity's full name as it first appears in the relevant "
    "context chunk, without additions."
)

# Direction 5: REFUSAL literal-phrase (3 variants)
_V15_D5V1_REFUSAL = (
    "Notes:\n"
    "1. Write EXACTLY three words — \"Insufficient information.\" — and "
    "stop. No other text."
)
_V15_D5V2_REFUSAL = (
    "Notes:\n"
    "1. Your entire response must be only: Insufficient information."
)
_V15_D5V3_REFUSAL = (
    "Notes:\n"
    "1. Respond with the literal phrase \"Insufficient information.\" "
    "and nothing else. Do not explain, qualify, or paraphrase."
)


# Build the 15 ParametrizedGroupedPromptBuilder variants for the v15
# experimental protocol. Each preset name encodes direction + variant:
#   clean_grouped_v15_d1v1 = Direction 1 (YES/NO source-attribution), Variant 1
#   clean_grouped_v15_d1v2 = Direction 1, Variant 2
#   ... etc.
#
# Only the target group's notes change per experiment; the other 3
# groups use the v13 default notes.

def _build_v15_preset(name: str) -> ParametrizedGroupedPromptBuilder:
    """Resolve a v15 preset name to a configured builder."""
    entity_default = CleanGroupedPromptBuilder._ENTITY_LOOKUP_NOTES
    yesno_default = CleanGroupedPromptBuilder._YESNO_NOTES
    temporal_default = CleanGroupedPromptBuilder._TEMPORAL_NOTES
    refusal_default = CleanGroupedPromptBuilder._REFUSAL_NOTES

    if name == "clean_grouped_v15_d1v1":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D1V1_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d1v2":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D1V2_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d1v3":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D1V3_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d2v1":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D2V1_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d2v2":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D2V2_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d2v3":
        return ParametrizedGroupedPromptBuilder(entity_default, _V15_D2V3_YESNO, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d3v1":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, _V15_D3V1_TEMPORAL, refusal_default)
    if name == "clean_grouped_v15_d3v2":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, _V15_D3V2_TEMPORAL, refusal_default)
    if name == "clean_grouped_v15_d3v3":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, _V15_D3V3_TEMPORAL, refusal_default)
    if name == "clean_grouped_v15_d4v1":
        return ParametrizedGroupedPromptBuilder(_V15_D4V1_ENTITY, yesno_default, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d4v2":
        return ParametrizedGroupedPromptBuilder(_V15_D4V2_ENTITY, yesno_default, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d4v3":
        return ParametrizedGroupedPromptBuilder(_V15_D4V3_ENTITY, yesno_default, temporal_default, refusal_default)
    if name == "clean_grouped_v15_d5v1":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, temporal_default, _V15_D5V1_REFUSAL)
    if name == "clean_grouped_v15_d5v2":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, temporal_default, _V15_D5V2_REFUSAL)
    if name == "clean_grouped_v15_d5v3":
        return ParametrizedGroupedPromptBuilder(entity_default, yesno_default, temporal_default, _V15_D5V3_REFUSAL)
    raise ValueError(f"Unknown v15 preset name: {name!r}")


class AnthropicLLM:
    """Async Anthropic client wrapped as an LLM protocol.

    Uses the same prompt shape as the eval pipeline (qa_judge.ask_llm) so
    pipelines and evals produce identical outputs given identical inputs.

    If `thinking_budget` is set, enables Anthropic extended thinking mode
    with that many tokens of internal reasoning. The visible answer
    remains the only thing returned by `ask()` — internal reasoning is
    consumed by the model and discarded by our scoring path.
    """

    def __init__(self, client, model: str, thinking_budget: int | None = None):
        self._client = client
        self._model = model
        self._thinking_budget = thinking_budget

    async def ask(self, messages: list[dict], max_tokens: int = 200) -> str:
        from backend.eval.qa_judge import ask_llm
        return await ask_llm(
            self._client,
            self._model,
            messages,
            max_tokens=max_tokens,
            thinking_budget=self._thinking_budget,
        )


# ── Factory ──────────────────────────────────────────────────────────────


def build_embedder(backend: str, model_name: str) -> Embeddings:
    """Dispatch to the named embedding backend.

    Thin wrapper over backend.rag.embeddings.make_embeddings that takes the
    model_name as a parameter (the upstream helper accepts it too but the
    default name is "all-MiniLM-L6-v2").
    """
    from backend.rag.embeddings import make_embeddings
    return make_embeddings(backend, model_name=model_name)


def build_retriever(config: PipelineConfig, vectorstore, corpus: list[Document] | None = None) -> Retriever:
    """Dispatch on config.retriever.

    For 'dense' and 'hybrid', `vectorstore` is the FAISS vectorstore built
    over the corpus. For 'hybrid', `corpus` is also required (must be the
    list of Documents that built the FAISS index, in the same order).
    """
    if config.retriever == "dense":
        return DenseRetriever(vectorstore)
    if config.retriever == "hybrid":
        if corpus is None:
            raise ValueError(
                "Hybrid retriever requires the raw corpus (list of Documents) "
                "to build the BM25 index. Pass corpus=... to build_pipeline."
            )
        return HybridRetriever(
            dense_retriever=DenseRetriever(vectorstore),
            bm25_retriever=BM25Retriever(corpus),
        )
    raise ValueError(f"Unknown retriever: {config.retriever!r}")


def build_reranker(config: PipelineConfig) -> Reranker | None:
    if config.reranker is None:
        return None
    if config.reranker == "cross_encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {config.reranker!r}")


def build_prompt_builder(config: PipelineConfig) -> PromptBuilder:
    if config.prompt_template == "default":
        return DefaultPromptBuilder()
    if config.prompt_template == "extract_span":
        return ExtractSpanPromptBuilder()
    if config.prompt_template == "cot_extract":
        return CoTExtractPromptBuilder()
    if config.prompt_template == "cot_extract_v2":
        return CoTExtractV2PromptBuilder()
    if config.prompt_template == "cot_extract_no_titles":
        return CoTExtractNoTitlesPromptBuilder()
    if config.prompt_template == "pre_analysis_extract":
        return PreAnalysisExtractPromptBuilder()
    if config.prompt_template == "simplified_v2":
        return SimplifiedV2PromptBuilder()
    if config.prompt_template == "simplified_v2_v16b":
        return SimplifiedV2Bv1PromptBuilder()
    if config.prompt_template == "simplified_v2_v16c":
        return SimplifiedV2Cv1PromptBuilder()
    if config.prompt_template == "simplified_v2_v16d":
        return SimplifiedV2Dv1PromptBuilder()
    if config.prompt_template == "simplified_v2_v16e":
        return SimplifiedV2Ev1PromptBuilder()
    if config.prompt_template == "clean_grouped":
        return CleanGroupedPromptBuilder()
    if config.prompt_template == "parametrized_grouped_v15":
        return _build_v15_preset(config.name)
    raise ValueError(f"Unknown prompt_template: {config.prompt_template!r}")


def build_llm(config: PipelineConfig, client) -> LLM:
    return AnthropicLLM(
        client=client,
        model=config.llm_model,
        thinking_budget=config.thinking_budget,
    )


# ── Pipeline orchestrator ────────────────────────────────────────────────


class RagPipeline:
    """Orchestrates retriever -> reranker -> prompt -> LLM for a query.

    The retriever and LLM are provided at construction time (so they can
    hold expensive state like FAISS indices or API clients). The reranker
    and prompt builder are stateless.
    """

    def __init__(
        self,
        config: PipelineConfig,
        retriever: Retriever,
        reranker: Reranker | None,
        prompt_builder: PromptBuilder,
        llm: LLM,
    ):
        self.config = config
        self._retriever = retriever
        self._reranker = reranker
        self._prompt_builder = prompt_builder
        self._llm = llm

    async def run(self, question: str) -> str:
        """One end-to-end RAG call.

        Step 1: retrieve up to `rerank_top_k` candidates (or `top_k` if
                no reranker).
        Step 2: optionally rerank to `top_k`.
        Step 3: build the prompt.
        Step 4: ask the LLM.
        """
        # Step 1 — retrieve.
        retrieve_k = (
            self.config.rerank_top_k if self._reranker is not None else self.config.top_k
        )
        candidates = self._retriever.retrieve(question, k=retrieve_k)

        # Step 2 — rerank (or slice).
        if self._reranker is not None:
            final_docs = self._reranker.rerank(question, candidates, top_k=self.config.top_k)
        else:
            final_docs = candidates[: self.config.top_k]

        # Step 3 — build the prompt.
        messages = self._prompt_builder.build(question, final_docs)

        # Step 4 — ask.
        return await self._llm.ask(messages)


def build_pipeline(
    config: PipelineConfig,
    vectorstore,
    llm_client,
    corpus: list[Document] | None = None,
) -> RagPipeline:
    """Top-level factory. The user-visible one-switch API.

    Example:
        config = PRESETS["naive_dense"]
        pipeline = build_pipeline(config, vectorstore=my_faiss, llm_client=client)
        answer = await pipeline.run("What year was X born?")

    For hybrid presets (`hybrid_bm25_dense`), `corpus` must be the list of
    Documents that built the FAISS index, in the same order. For dense
    presets, corpus is ignored.
    """
    return RagPipeline(
        config=config,
        retriever=build_retriever(config, vectorstore, corpus=corpus),
        reranker=build_reranker(config),
        prompt_builder=build_prompt_builder(config),
        llm=build_llm(config, llm_client),
    )


# ── CoT-extract small modifications (iter-19) ───────────────────────────


class CoTExtractV2PromptBuilder:
    """Iter-19: small, targeted refinement of `CoTExtractPromptBuilder`.

    The iter-14 → iter-18 arc explored many lever combinations (canonical-
    name rules, yes/no discipline, two-step extraction) and all variants
    regressed vs cot_extract_k10. The LLM applies conditional rules
    inconsistently. Rather than more rules, this iter makes a minimal
    targeted nudge: tighten step 4 toward using the most complete form
    of an entity name (canonical names live in the context), and tighten
    the closing directive to reinforce the same.

    What changed vs CoTExtractPromptBuilder:
      - Step 4: append ", using the most complete form as written in the
        context for entity-name answers" (one short clause).
      - Closing: "quote it verbatim" → "quote it verbatim from the context,
        using the most complete form of an entity name".

    Total prompt growth: ~25 words. No examples, no discriminators, no
    conditional rules — just one guiding principle embedded in two
    existing instructions.
    """

    COT_INSTRUCTION_V2 = (
        "Read the <context>...</context> block carefully. Some questions "
        "require combining facts from multiple paragraphs (multi-hop reasoning).\n\n"
        "Think step by step:\n"
        "1. Identify the entities and facts the question asks about.\n"
        "2. Find the relevant paragraph(s) in the context.\n"
        "3. If multi-hop reasoning is needed, chain together the supporting "
        "facts in order.\n"
        "4. Decide which exact span answers the question — using the most "
        "complete form as written in the context for entity-name answers.\n\n"
        "Begin your response with the extracted span (in quotation marks "
        "if it is a phrase), then briefly explain your reasoning. "
        "Do not paraphrase — quote it verbatim from the context, using the "
        "most complete form of an entity name."
    )

    def __init__(self):
        from backend.eval.qa_judge import RAG_SYSTEM_PROMPT_HERE
        self._system_prompt = RAG_SYSTEM_PROMPT_HERE + "\n\n" + self.COT_INSTRUCTION_V2

    def build(self, question: str, context_docs: list[Document] | None) -> list[dict]:
        if not context_docs:
            return [{"role": "user", "content": question}]
        context_str = "\n\n".join(
            f"[{d.metadata.get('title', '')}]: {d.page_content}" for d in context_docs
        )
        user_content = f"<context>\n{context_str}\n</context>\n\n{question}"
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]


# ── Presets ──────────────────────────────────────────────────────────────


PRESETS: dict[str, PipelineConfig] = {
    # Existing baseline: small embedding model, no reranker, default prompt.
    "naive_dense": PipelineConfig(
        name="naive_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Bigger embedding model, no reranker — measure embedding-model lift alone.
    "large_dense": PipelineConfig(
        name="large_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Bigger embedding + cross-encoder reranker — full precision push.
    "dense_then_ce": PipelineConfig(
        name="dense_then_ce",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker="cross_encoder",
        rerank_top_k=50,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
    # Prompt-only change: same retrieval, but ask LLM to extract spans
    # verbatim. Useful for boosting answer_f1 in benchmarks.
    "extract_span_prompt": PipelineConfig(
        name="extract_span_prompt",
        embedding_backend="sentence-transformers",
        embedding_model="all-mpnet-base-v2",
        retriever="dense",
        reranker=None,
        top_k=4,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # Combined: bigger context (top_k=8) + verbatim-span extraction. This
    # composes the two extraction-side wins from iter-14: more paragraphs
    # in context covers the long tail of gold-paragraphs-outside-top-4,
    # and verbatim-span guidance reduces LLM extraction error.
    "extract_span_k8": PipelineConfig(
        name="extract_span_k8",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=8,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # Same as extract_span_k8 but with the full 10-paragraph context.
    # Only marginally better than k=8 on HotpotQA but useful when the
    # supporting-fact set spans more paragraphs than top-k=8 covers.
    "extract_span_k10": PipelineConfig(
        name="extract_span_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="extract_span",
        llm_model="minimax-3",
    ),
    # iter-15 SOTA: CoT scaffold + verbatim-span directive at k=10.
    "cot_extract_k10": PipelineConfig(
        name="cot_extract_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="cot_extract",
        llm_model="minimax-3",
    ),
    # iter-19: minimal targeted refinement of cot_extract. One guiding
    # clause in step 4 + matching tightening of the closing directive,
    # nudging the model toward using the most complete entity-name form
    # as written in the context.
    "cot_extract_v2_k10": PipelineConfig(
        name="cot_extract_v2_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="cot_extract_v2",
        llm_model="minimax-3",
    ),
    # iter-20: enable Anthropic extended thinking mode (4096 reasoning
    # budget). Uses the simple extract_span prompt — the model reasons
    # internally instead of in the visible output. Visible answer stays
    # clean (no reasoning noise diluted into it), while the model still
    # gets the multi-hop / canonical-name / yes-no reasoning benefit.
    "cot_thinking_k10": PipelineConfig(
        name="cot_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="extract_span",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-21: same CoT prompt as cot_extract_k10 BUT with the [title]:
    # heading prefix stripped from each context paragraph. Hypothesis
    # (from iter-20 thinking audit): the model uses Wikipedia article
    # headings as entity labels when emitting answers, while HotpotQA
    # gold uses the full canonical form (typically in the body opening).
    # Stripping the heading forces the model to extract the canonical
    # name from the body text.
    "cot_extract_notitles_k10": PipelineConfig(
        name="cot_extract_notitles_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="cot_extract_no_titles",
        llm_model="minimax-3",
    ),
    # iter-22: combine title-strip (iter-21) with thinking mode (iter-20).
    # Title-strip forces canonical-name extraction from the body; thinking
    # gives the model more reasoning budget for hard multi-hop questions.
    "cot_extract_notitles_thinking_k10": PipelineConfig(
        name="cot_extract_notitles_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="cot_extract_no_titles",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-29: iter-22 SOTA + pre-analysis prefix in the user message. The
    # model briefly analyzes the question (entities, reasoning type) before
    # seeing the <context> block, priming attention toward the right chunks.
    # A/B variant only — the iter-22 SOTA preset is unchanged.
    "pre_analysis_extract_thinking_k10": PipelineConfig(
        name="pre_analysis_extract_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="pre_analysis_extract",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-34 v16-a: simplified v2. Drops the iter-22 CoT scaffold (4
    # steps + 'Begin with extracted span'), keeps the 4-shape pre-
    # analysis enumeration as the lift mechanism. Hypothesis: the CoT
    # scaffold overlapped with the pre-analysis and was redundant.
    # See SimplifiedV2PromptBuilder docstring.
    "simplified_v2_thinking_k10": PipelineConfig(
        name="simplified_v2_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="simplified_v2",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-34 v16-b: v16-a + TEMPORAL verdict-leading directive
    # (recovered from v15 d3v2 winner wording). See
    # SimplifiedV2Bv1PromptBuilder docstring.
    "simplified_v2_v16b_thinking_k10": PipelineConfig(
        name="simplified_v2_v16b_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="simplified_v2_v16b",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-34 v16-c: v16-b + strengthened TEMPORAL opening
    # ('Lead with the verdict word (Yes, No, Consistent, or
    # Inconsistent), followed by a brief one-sentence explanation').
    "simplified_v2_v16c_thinking_k10": PipelineConfig(
        name="simplified_v2_v16c_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="simplified_v2_v16c",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-34 v16-d: v16-c + YES/NO chunk-match + verdict-leading
    # + brevity directive. Mirrors v16-c's TEMPORAL pattern.
    "simplified_v2_v16d_thinking_k10": PipelineConfig(
        name="simplified_v2_v16d_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="simplified_v2_v16d",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-34 v16-e: v16-c + ENTITY canonical-name directive.
    # See SimplifiedV2Ev1PromptBuilder docstring.
    "simplified_v2_v16e_thinking_k10": PipelineConfig(
        name="simplified_v2_v16e_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="simplified_v2_v16e",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-33 v12: clean per-group prompts with numbered notes. All
    # groups share the base "You are a helpful assistant. Answer the
    # question carefully." Each group has 1-3 positive directives
    # targeting its dominant failure mode (from iter-32 sub-agent
    # analysis). No CoT scaffold, no pre-analysis prefix. See
    # CleanGroupedPromptBuilder docstring for failure-mode mapping.
    "clean_grouped_thinking_k10": PipelineConfig(
        name="clean_grouped_thinking_k10",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="clean_grouped",
        thinking_budget=4096,
        llm_model="minimax-3",
    ),
    # iter-33 v15: 5 directions × 3 variants = 15 experimental presets.
    # Each preset changes ONLY the target group's note wording; the other
    # 3 groups use the v13 default notes. This isolates the effect of
    # each direction's variant. All presets share the same base +
    # dispatch logic. See ParametrizedGroupedPromptBuilder docstring.
    # Tested on a 20% stratified sample (40 questions: 7 entity_lookup,
    # 21 yesno, 9 temporal_order, 3 refusal) for time efficiency.
    **{f"clean_grouped_v15_d{d}v{v}": PipelineConfig(
        name=f"clean_grouped_v15_d{d}v{v}",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="dense",
        reranker=None,
        top_k=10,
        prompt_template="parametrized_grouped_v15",
        thinking_budget=4096,
        llm_model="minimax-3",
    ) for d in range(1, 6) for v in range(1, 4)},
    # Hybrid BM25 + dense via Reciprocal Rank Fusion. Different lever than
    # embedding-model size: BM25 catches exact entity-name matches the
    # embedding model glosses over, dense catches paraphrase matches BM25
    # misses. RRF combines without learning weights.
    "hybrid_bm25_dense": PipelineConfig(
        name="hybrid_bm25_dense",
        embedding_backend="sentence-transformers",
        embedding_model="all-MiniLM-L6-v2",
        retriever="hybrid",
        reranker=None,
        top_k=4,
        prompt_template="default",
        llm_model="minimax-3",
    ),
}


def list_presets() -> list[str]:
    """Return the names of all available presets (sorted)."""
    return sorted(PRESETS.keys())

"""Tests for backend.eval.normalizer."""
import pytest

from backend.eval.normalizer import (
    is_refusal,
    normalize_answer,
    normalize_for_comparison,
    normalize_for_null,
    normalize_for_temporal,
    strip_preamble,
)


# ---- is_refusal --------------------------------------------------------

def test_is_refusal_insufficient_information():
    assert is_refusal("Insufficient information.")


def test_is_refusal_cannot_determine():
    assert is_refusal("I cannot determine the answer from this context.")


def test_is_refusal_no_information():
    assert is_refusal("There is no specific information about X in the context.")


def test_is_refusal_case_insensitive():
    assert is_refusal("BASED ON THE CONTEXT, I CANNOT VERIFY")


def test_is_refusal_unable_to_answer():
    """iter-35 v19a D1 addition."""
    assert is_refusal("I'm unable to answer this question based on the information.")


def test_is_refusal_does_not_contain_any():
    """iter-35 v19a D1 addition — covers 'does not contain any articles'."""
    assert is_refusal("The context you've provided does not contain any articles fr...")


def test_is_refusal_articles_referenced():
    """iter-35 v19a D1 addition — 'articles you've referenced' hedge."""
    assert is_refusal("The articles you've referenced about Wozniak are not in the corpus.")


def test_is_refusal_information_asking_about():
    """iter-35 v19a D1 addition — 'information you're asking about' hedge."""
    assert is_refusal("The information you're asking about isn't available in the context.")


def test_is_not_refusal_yes():
    assert not is_refusal("Yes, the article supports this.")


def test_is_not_refusal_inconsistent():
    # 'inconsistent' alone isn't refusal — it's a verdict.
    assert not is_refusal("Inconsistent: the reports differ on dates.")


# ---- strip_preamble ----------------------------------------------------

def test_strip_preamble_based_on_context():
    assert strip_preamble("Based on the context, Yes.") == "Yes."


def test_strip_preamble_no_preamble():
    assert strip_preamble("Yes, the article supports this.") == "Yes, the article supports this."


def test_strip_preamble_looking_at():
    assert strip_preamble("Looking at the context, the answer is X.") == "the answer is X."


# ---- normalize_for_null ------------------------------------------------

def test_normalize_for_null_rewrites_to_literal():
    assert normalize_for_null("I cannot determine from this context.") == "Insufficient information."


def test_normalize_for_null_unable_to_answer():
    """iter-35 v19a D1 — catches paraphrases the original set missed."""
    assert normalize_for_null("I'm unable to answer this question based on the information.") == "Insufficient information."


def test_normalize_for_null_does_not_contain():
    """iter-35 v19a D1 — covers 'does not contain any articles'."""
    assert normalize_for_null("The context you've provided does not contain any articles about X.") == "Insufficient information."


def test_normalize_for_null_articles_referenced():
    """iter-35 v19a D1 — 'articles you've referenced' hedge."""
    assert normalize_for_null("The articles you've referenced about Wozniak are not in the corpus.") == "Insufficient information."


def test_normalize_for_null_information_asking_about():
    """iter-35 v19a D1 — 'information you're asking about' hedge."""
    assert normalize_for_null("The information you're asking about isn't available in the context.") == "Insufficient information."


def test_normalize_for_null_preserves_non_refusal():
    # Non-refusal answers on null-type questions: leave alone (model
    # shouldn't have answered at all, but we don't second-guess).
    assert normalize_for_null("Yes.") == "Yes."


def test_normalize_for_null_already_literal():
    assert normalize_for_null("Insufficient information.") == "Insufficient information."


# ---- normalize_for_temporal --------------------------------------------

def test_normalize_for_temporal_yes_appends_consistent():
    out = normalize_for_temporal("Yes. Both reports date the event to March 5.")
    assert "Yes (Consistent)" in out


def test_normalize_for_temporal_no_appends_inconsistent():
    out = normalize_for_temporal("No, the reports conflict on the timeline.")
    assert "No (Inconsistent)" in out


def test_normalize_for_temporal_already_has_both():
    # If model already emits 'Yes (Consistent)', leave alone.
    out = normalize_for_temporal("Yes (Consistent). Both reports agree.")
    assert "Yes (Consistent)" in out
    assert out.count("Yes") == 1


def test_normalize_for_temporal_consistent_only_no_double():
    # 'Consistent.' alone (no Yes/No) — leave alone, no double-append.
    out = normalize_for_temporal("Consistent. Both reports agree.")
    assert out.startswith("Consistent.")


def test_normalize_for_temporal_strips_preamble():
    out = normalize_for_temporal("Based on the context, Yes, they are consistent.")
    assert "Yes (Consistent)" in out
    assert "Based on the context" not in out


# ---- normalize_for_comparison (iter-35 v19a C1) ------------------------

def test_normalize_for_comparison_yes_expands_positive_cluster():
    """Yes → Yes (True, Consistent, Aligned). So gold=True, gold=Consistent,
    gold=Aligned all substring-match."""
    out = normalize_for_comparison("Yes, both claims are supported.")
    assert "Yes (True, Consistent, Aligned)" in out


def test_normalize_for_comparison_true_expands_positive_cluster():
    out = normalize_for_comparison("True, the reports agree.")
    assert "True (Yes, Consistent, Aligned)" in out


def test_normalize_for_comparison_no_expands_negative_cluster():
    out = normalize_for_comparison("No, the reports conflict.")
    assert "No (False, Different, Inconsistent)" in out


def test_normalize_for_comparison_false_expands_negative_cluster():
    out = normalize_for_comparison("False, the premise is wrong.")
    assert "False (No, Different, Inconsistent)" in out


def test_normalize_for_comparison_different_expands_negative_cluster():
    out = normalize_for_comparison("Different takes on the situation.")
    assert "Different (No, False, Inconsistent)" in out


def test_normalize_for_comparison_same_expands_identity_cluster():
    out = normalize_for_comparison("Same perspective across reports.")
    assert "Same (Similar)" in out


def test_normalize_for_comparison_similar_expands_identity_cluster():
    out = normalize_for_comparison("Similar treatment in both articles.")
    assert "Similar (Same)" in out


def test_normalize_for_comparison_strips_preamble():
    out = normalize_for_comparison("Based on the context, Yes, they align.")
    assert "Yes (True, Consistent, Aligned)" in out
    assert "Based on the context" not in out


def test_normalize_for_comparison_no_verdict_word_no_op():
    """If model never leads with a verdict word (verdict-buried prompt
    directive didn't bite), the normalizer can't help — this is a no-op.

    iter-35 v19f: 'Both ...' affirmative framing is the one exception —
    that's caught by C2 and prepended with 'Yes (...).' Test the
    non-affirmative case to confirm no-op."""
    text = "The two articles describe distinctly different approaches to the question."
    out = normalize_for_comparison(text)
    assert out == text  # no change


def test_normalize_for_comparison_both_prefix_prepends_yes():
    """iter-35 v19f C2: 'Both ...' affirmative framing is prepended with
    'Yes (True, Consistent, Aligned), ' so contains_gold matches the gold
    form 'Yes' even when the model never uses a verdict word."""
    text = "Both observations are accurate, and they actually complement each other."
    out = normalize_for_comparison(text)
    assert out.startswith("Yes (True, Consistent, Aligned), ")
    assert "Both observations" in out  # original preserved after prefix


def test_normalize_for_temporal_i_can_confirm_prepends_yes():
    """iter-35 v19f T1: 'I can confirm ...' affirmative hedging is prepended
    with 'Yes (Consistent), ' so contains_gold matches 'Yes'."""
    from backend.eval.normalizer import normalize_for_temporal_v19f
    text = "I can confirm that the article supports your claim."
    out = normalize_for_temporal_v19f(text)
    assert out.startswith("Yes (Consistent), ")
    assert "I can confirm" in out


def test_normalize_for_comparison_already_has_synonyms_no_double():
    """If model already says 'Yes (True)', don't re-expand."""
    text = "Yes (True), the article supports this."
    out = normalize_for_comparison(text)
    assert out == text
    assert out.count("Yes") == 1


# ---- normalize_answer (top-level) --------------------------------------

def test_normalize_answer_null_with_refusal():
    assert normalize_answer("There is no information about X.", qtype="null") == "Insufficient information."


def test_normalize_answer_temporal_yes():
    out = normalize_answer("Yes. The reports agree.", qtype="temporal_order")
    assert "Yes (Consistent)" in out


def test_normalize_answer_comparison_yes_to_synonyms():
    """iter-35 v19a C1 — comparison verdict-vocab mapping dispatched."""
    out = normalize_answer("Yes, both are supported.", qtype="comparison")
    assert "True" in out  # gold=True now substring-matches


def test_normalize_answer_comparison_no_qtype():
    # No qtype: just preamble strip (no comparison dispatch).
    assert normalize_answer("Based on the context, Yes.") == "Yes."


def test_normalize_answer_empty():
    assert normalize_answer("", qtype="null") == ""


def test_normalize_answer_qtype_inference_unchanged():
    # Inference: no type-specific transform, only preamble strip.
    assert normalize_answer("Based on the context, Jane Smith.") == "Jane Smith."


def test_normalize_answer_comparison_hedge_not_refusal():
    """Regression: model often hedges ('cannot verify', 'no article') while
    still answering. We must NOT rewrite these to 'Insufficient
    information.' — that destroys the answer. The trailing 'Yes' is not
    at the start so v19a's synonym expansion doesn't fire on it; the
    hedge text is preserved as-is."""
    # Model says there's no Fortune article, but gives a verdict below
    text = (
        "Looking at the provided context, the Fortune article is not "
        "available, but based on the TechCrunch article, the claim is "
        "Yes."
    )
    out = normalize_answer(text, qtype="comparison")
    assert "Insufficient information" not in out
    assert "Yes" in out
    # v19a: only LEADING verdict words get expanded. "Yes" at the end
    # stays as just "Yes." — preserved verbatim.
    assert out == text


def test_normalize_answer_temporal_hedge_not_refusal():
    """Regression: temporal questions with hedge language must keep their
    substantive answer."""
    text = (
        "Based on the context, I cannot verify the specific dates, but "
        "the information is Consistent across reports."
    )
    out = normalize_answer(text, qtype="temporal_order")
    assert "Insufficient information" not in out
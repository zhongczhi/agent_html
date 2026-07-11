import pytest

from backend.eval.metrics import (
    answer_coverage_at_k,
    answer_f1,
    exact_match,
    gold_paragraph_in_top_k,
    paragraph_recall_at_k,
    supporting_fact_metrics,
)


def test_paragraph_recall_at_k_empty_gold_vacuous():
    assert paragraph_recall_at_k([], set()) == 1.0
    assert paragraph_recall_at_k(["a", "b"], set()) == 1.0


def test_paragraph_recall_at_k_all_retrieved():
    assert paragraph_recall_at_k(["a", "b"], {"a", "b"}) == 1.0


def test_paragraph_recall_at_k_none_retrieved():
    assert paragraph_recall_at_k(["x", "y"], {"a", "b"}) == 0.0


def test_paragraph_recall_at_k_partial_overlap():
    # gold size 2, retrieved ["a","x"] -> 1 hit, recall = 1/2 = 0.5
    assert paragraph_recall_at_k(["a", "x"], {"a", "b"}) == 0.5


def test_paragraph_recall_at_k_capped_at_one():
    # gold size 1, retrieved hits 2 (duplicate), capped at 1/1 = 1.0
    assert paragraph_recall_at_k(["a", "a"], {"a"}) == 1.0


def test_sf_metrics_empty_empty():
    assert supporting_fact_metrics([], set()) == (1.0, 1.0, 1.0, 1.0)


def test_sf_metrics_empty_gold_nonempty_retrieved():
    # vacuous recall=1, precision=0 (nothing valid), f1=0, em=0
    assert supporting_fact_metrics(["x"], set()) == (0.0, 1.0, 0.0, 0.0)


def test_sf_metrics_retrieved_empty_nonempty_gold():
    assert supporting_fact_metrics([], {"a"}) == (0.0, 0.0, 0.0, 0.0)


def test_sf_metrics_partial_overlap():
    # gold={a,b}, retrieved=[a,c] -> tp=1, p=1/2=0.5, r=1/2=0.5, f1=0.5, em=0
    sp, sr, sf, em = supporting_fact_metrics(["a", "c"], {"a", "b"})
    assert (sp, sr, sf, em) == pytest.approx((0.5, 0.5, 0.5, 0.0))


def test_sf_metrics_em_one_iff_exact_set_match():
    assert supporting_fact_metrics(["a", "b"], {"a", "b"})[3] == 1.0
    # Order-insensitive
    assert supporting_fact_metrics(["b", "a"], {"a", "b"})[3] == 1.0
    # Extra retrieved -> em=0
    assert supporting_fact_metrics(["a", "b", "c"], {"a", "b"})[3] == 0.0


def test_answer_coverage_empty_gold_vacuous():
    # Vacuous: an empty gold answer is treated as "already covered" so
    # we don't divide by zero and so degenerate items don't tank the mean.
    assert answer_coverage_at_k([], "") == 1.0
    assert answer_coverage_at_k(["anything"], "") == 1.0


def test_answer_coverage_empty_retrieved_zero():
    assert answer_coverage_at_k([], "yes") == 0.0


def test_answer_coverage_exact_match():
    assert answer_coverage_at_k(["The answer is yes."], "yes") == 1.0


def test_answer_coverage_case_insensitive():
    assert answer_coverage_at_k(["...YES..."], "yes") == 1.0


def test_answer_coverage_punctuation_stripped():
    # Gold has a comma; doc doesn't — should still match.
    assert answer_coverage_at_k(["Born in 1968"], "1968,") == 1.0


def test_answer_coverage_no_match():
    assert answer_coverage_at_k(["foo bar baz"], "qux") == 0.0


def test_answer_coverage_across_multiple_docs():
    # Gold answer is a multi-word phrase that spans two retrieved paragraphs.
    assert answer_coverage_at_k(
        ["The composer was", "born in 1968 in Berlin."],
        "1968 in Berlin",
    ) == 1.0


def test_answer_coverage_multiline_inside_one_doc():
    # Paragraphs retrieved from FAISS may contain embedded newlines from
    # the original Wikipedia source.
    assert answer_coverage_at_k(
        ["line one\nline two contains YES in it\nline three"],
        "yes",
    ) == 1.0


def test_gold_paragraph_in_top_k_empty_gold_vacuous():
    # Vacuous: no gold means no miss possible.
    assert gold_paragraph_in_top_k([], set()) is True
    assert gold_paragraph_in_top_k(["a", "b"], set()) is True


def test_gold_paragraph_in_top_k_empty_retrieved_false():
    assert gold_paragraph_in_top_k([], {"a"}) is False


def test_gold_paragraph_in_top_k_hit():
    assert gold_paragraph_in_top_k(["a", "b"], {"a"}) is True
    # Hit even if it's not the first one.
    assert gold_paragraph_in_top_k(["x", "y", "a"], {"a"}) is True


def test_gold_paragraph_in_top_k_miss():
    assert gold_paragraph_in_top_k(["x", "y"], {"a"}) is False


def test_gold_paragraph_in_top_k_subset_partial_hit():
    # Multi-fact HotpotQA items often have 2+ gold titles; one in top-k is enough.
    assert gold_paragraph_in_top_k(["a", "x"], {"a", "b"}) is True


# ---- answer_f1 + exact_match (FR-40) ---------------------------------------

def test_answer_f1_exact_match():
    assert answer_f1("The Godfather", "The Godfather") == 1.0


def test_answer_f1_case_insensitive():
    assert answer_f1("the godfather", "The Godfather") == 1.0


def test_answer_f1_strips_punctuation():
    assert answer_f1("The Godfather.", "The Godfather") == 1.0


def test_answer_f1_strips_articles():
    # 'The' is removed from both sides; 'cat' vs 'cat' matches.
    assert answer_f1("a cat", "cat") == 1.0


def test_answer_f1_partial_overlap():
    # pred="The Godfather Part II" -> tokens ['godfather', 'part', 'ii'] (3)
    # gold="The Godfather"          -> tokens ['godfather'] (1)
    # common = {'godfather': min(1,1)=1}; num_same=1
    # precision = 1/3, recall = 1/1 = 1.0, F1 = 2*(1/3)*1 / (1/3+1) = 0.5
    assert answer_f1("The Godfather Part II", "The Godfather") == pytest.approx(0.5)


def test_answer_f1_empty_predicted():
    assert answer_f1("", "yes") == 0.0


def test_answer_f1_empty_gold():
    assert answer_f1("anything", "") == 0.0


def test_answer_f1_no_overlap():
    assert answer_f1("apple", "banana") == 0.0


def test_answer_f1_handles_duplicates():
    # pred="yes yes yes" -> ['yes','yes','yes'] (3)
    # gold="yes"          -> ['yes'] (1)
    # common = Counter({'yes': 1}); num_same=1
    # precision = 1/3, recall = 1/1, F1 = 2*(1/3)*1 / (1/3+1) = 0.5
    assert answer_f1("yes yes yes", "yes") == pytest.approx(0.5)


def test_exact_match_identical_tokens():
    assert exact_match("The Godfather", "The Godfather") is True


def test_exact_match_after_normalization():
    # 'the' stripped on both sides -> ['godfather'] vs ['godfather'] -> equal
    assert exact_match("the godfather", "The Godfather") is True


def test_exact_match_strips_articles_asymmetrically():
    # pred="The cat" -> ['cat']; gold="cat" -> ['cat'] -> equal
    assert exact_match("The cat", "cat") is True


def test_exact_match_partial():
    assert exact_match("The Godfather II", "The Godfather") is False


def test_exact_match_empty_predicted():
    assert exact_match("", "yes") is False


def test_exact_match_empty_gold():
    assert exact_match("anything", "") is False


def test_exact_match_different_after_normalization():
    # 'a cat' -> ['cat']; 'a dog' -> ['dog'] -> different
    assert exact_match("a cat", "a dog") is False

import pytest

from backend.eval.metrics import paragraph_recall_at_k, supporting_fact_metrics


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

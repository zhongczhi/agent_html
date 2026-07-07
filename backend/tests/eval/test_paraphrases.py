import json
from pathlib import Path

import pytest

from backend.eval.paraphrases import (
    load_paraphrases,
    lookup,
    required_styles,
    validate_paraphrase,
)


# ---- validate_paraphrase ---------------------------------------------------

def test_validate_accepts_unrelated_paraphrase():
    # 'Paris' is not in the paraphrase; no leak.
    assert validate_paraphrase(
        "Which composer wrote the piece?",
        "Paris",
    ) is True


def test_validate_rejects_answer_leak_full():
    # Gold is 'Paris'; paraphrase contains 'Paris' verbatim -> reject.
    assert validate_paraphrase(
        "The composer lived in Paris, when?",
        "Paris",
    ) is False


def test_validate_rejects_answer_leak_partial_high_overlap():
    # Gold has 2 tokens ['john', 'smith']; paraphrase has both -> 100% overlap.
    assert validate_paraphrase(
        "Who is John Smith?",
        "John Smith",
    ) is False


def test_validate_threshold_is_80_percent():
    # Gold has 5 tokens; paraphrase contains 4 of them -> 80% overlap,
    # which is NOT strictly less than 80% -> rejected.
    # Token overlap test: gold=['a','b','c','d','e'],
    # paraphrase contains {'a','b','c','d','x'} -> 4/5 = 0.80 -> reject.
    assert validate_paraphrase(
        "x x x x a b c d",
        "a b c d e",
    ) is False


def test_validate_threshold_just_under_80_passes():
    # 3 of 5 = 60% overlap -> below threshold -> pass.
    assert validate_paraphrase(
        "x y z a b c",
        "a b c d e",
    ) is True


def test_validate_rejects_empty_paraphrase():
    assert validate_paraphrase("", "yes") is False


def test_validate_rejects_empty_gold():
    assert validate_paraphrase("anything", "") is False


# ---- load_paraphrases ------------------------------------------------------

def test_load_returns_items_dict(tmp_path: Path):
    p = tmp_path / "para.json"
    p.write_text(
        json.dumps(
            {
                "dataset_sha": "abc",
                "schema_version": 1,
                "items": {"q1": {"paraphrases": {"lexical": "x"}}},
            }
        ),
        encoding="utf-8",
    )
    out = load_paraphrases(p)
    assert out == {"q1": {"paraphrases": {"lexical": "x"}}}


def test_load_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_paraphrases(tmp_path / "absent.json")


def test_load_corrupt_json_raises(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_paraphrases(p)


def test_load_missing_items_key_returns_empty(tmp_path: Path):
    p = tmp_path / "noitems.json"
    p.write_text(json.dumps({"dataset_sha": "x"}), encoding="utf-8")
    assert load_paraphrases(p) == {}


# ---- lookup + required_styles ---------------------------------------------

def test_lookup_returns_paraphrases_when_present():
    data = {"q1": {"paraphrases": {"lexical": "L", "structural": "S", "casual": "C"}}}
    assert lookup(data, "q1") == {"lexical": "L", "structural": "S", "casual": "C"}


def test_lookup_returns_none_when_absent():
    assert lookup({"q1": {}}, "q2") is None


def test_required_styles_is_lexical_structural_casual():
    assert required_styles() == ("lexical", "structural", "casual")
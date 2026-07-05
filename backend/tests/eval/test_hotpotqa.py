from pathlib import Path

from backend.eval.hotpotqa import (
    dataset_sha,
    gold_paragraph_titles,
    load,
    sample,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hotpot.json"


def test_load_count_and_fields():
    items = load(FIXTURE)
    assert len(items) == 3
    assert items[0].id == "aaa111"
    assert items[0].type == "bridge"
    assert items[0].level == "easy"
    assert len(items[0].context) == 3
    assert items[0].context[0] == ("Title A", ["s0", "s1"])
    assert items[0].supporting_facts == [("Title A", 0), ("Title A", 1)]


def test_dataset_sha_changes_on_file_change():
    sha1 = dataset_sha(FIXTURE)
    original = FIXTURE.read_bytes()
    try:
        modified = original.replace(b"aaa111", b"aaa999")
        FIXTURE.write_bytes(modified)
        sha2 = dataset_sha(FIXTURE)
    finally:
        FIXTURE.write_bytes(original)
    assert sha1 != sha2
    assert len(sha1) == 16


def test_gold_paragraph_titles_dedup():
    items = load(FIXTURE)
    # aaa111's supporting facts both reference "Title A" - must dedupe to 1.
    assert gold_paragraph_titles(items[0]) == {"Title A"}
    assert gold_paragraph_titles(items[1]) == {"Title X"}
    assert gold_paragraph_titles(items[2]) == {"Title M", "Title N"}


def test_sample_deterministic():
    items = load(FIXTURE)
    s1 = sample(items, 12)
    s2 = sample(items, 12)
    assert [i.id for i in s1] == [i.id for i in s2]


def test_sample_rejects_too_small():
    import pytest

    items = load(FIXTURE)
    with pytest.raises(ValueError):
        sample(items, 0)
    with pytest.raises(ValueError):
        sample(items, 1)

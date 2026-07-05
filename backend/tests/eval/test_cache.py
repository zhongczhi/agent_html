import shutil
from pathlib import Path

import pytest
from langchain_community.embeddings.fake import FakeEmbeddings

from backend.eval.cache import EVAL_CACHE_ROOT, _build_index, load_or_build
from backend.eval.hotpotqa import load as load_items

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_hotpot.json"


@pytest.fixture
def tmp_cache_root(tmp_path, monkeypatch):
    """Point EVAL_CACHE_ROOT at a tmp dir for the duration of the test."""
    new_root = tmp_path / "eval_cache"
    monkeypatch.setattr("backend.eval.cache.EVAL_CACHE_ROOT", new_root)
    yield new_root
    shutil.rmtree(new_root, ignore_errors=True)


def _embedding_factory():
    # Deterministic so we don't download anything; size=64 keeps memory low.
    return FakeEmbeddings(size=64)


def test_build_index_has_all_paragraphs(tmp_cache_root):
    items = load_items(FIXTURE)
    index = _build_index(items[0], _embedding_factory())
    n = sum(
        1 for d in index.docstore._dict.values() if not d.metadata.get("_placeholder")
    )
    assert n == 3  # items[0].context has 3 paragraphs


def test_load_or_build_first_call_misses(tmp_cache_root):
    items = load_items(FIXTURE)
    _index, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is False


def test_load_or_build_second_call_hits(tmp_cache_root):
    items = load_items(FIXTURE)
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    _index, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is True


def test_load_or_build_no_cache_forces_rebuild(tmp_cache_root):
    items = load_items(FIXTURE)
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    _index, hit = load_or_build(
        items[0], "deadbeef00000000", _embedding_factory(), no_cache=True
    )
    assert hit is False


def test_load_or_build_different_sha_separates_dirs(tmp_cache_root):
    items = load_items(FIXTURE)
    _, hit_a = load_or_build(items[0], "aaaaaaaa00000000", _embedding_factory())
    _, hit_b = load_or_build(items[0], "bbbbbbbb00000000", _embedding_factory())
    assert hit_a is False and hit_b is False
    assert (tmp_cache_root / "aaaaaaaa00000000").exists()
    assert (tmp_cache_root / "bbbbbbbb00000000").exists()


def test_load_or_build_recovers_from_corruption(tmp_cache_root):
    items = load_items(FIXTURE)
    cache_dir = tmp_cache_root / "deadbeef00000000" / items[0].id
    # First call writes a valid cache.
    load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    # Corrupt the cache by overwriting index.faiss with garbage.
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "index.faiss").write_bytes(b"not a real faiss index")
    # Second call must rebuild cleanly.
    _index, hit = load_or_build(items[0], "deadbeef00000000", _embedding_factory())
    assert hit is False  # treated as miss because we rebuilt
    assert _index is not None

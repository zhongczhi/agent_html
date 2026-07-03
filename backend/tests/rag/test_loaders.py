"""Loader unit tests. One assertion per loader + edge cases (empty file,
malformed content). Binary fixtures are generated on demand by conftest.
"""
from pathlib import Path

import pytest

from backend.rag.loaders import (
    ALLOWED_EXTENSIONS,
    REGISTRY,
    RawDocument,
    UnsupportedFormatError,
    load,
)


# ── Registry sanity ─────────────────────────────────────────────────────────

def test_registry_has_all_six_supported_formats():
    expected = {".md", ".txt", ".pdf", ".html", ".docx", ".csv"}
    assert expected.issubset(REGISTRY.keys())


def test_allowed_extensions_matches_registry_keys():
    assert ALLOWED_EXTENSIONS == frozenset(REGISTRY.keys())


def test_load_raises_for_unsupported_extension(tmp_path):
    f = tmp_path / "bad.xyz"
    f.write_text("anything")
    with pytest.raises(UnsupportedFormatError):
        list(load(f, "upload"))


# ── .txt ─────────────────────────────────────────────────────────────────────

def test_txt_loader_yields_one_raw_document_with_format(sample_txt: Path):
    raws = list(load(sample_txt, "upload"))
    assert len(raws) == 1
    assert raws[0].metadata["format"] == ".txt"
    assert "plain text" in raws[0].text


# ── .md ─────────────────────────────────────────────────────────────────────

def test_md_loader_yields_one_raw_document(sample_md: Path):
    raws = list(load(sample_md, "upload"))
    assert len(raws) == 1
    assert raws[0].metadata["format"] == ".md"
    # The whole file is one RawDocument — header_path is computed later
    # by the splitter, not by the loader.
    assert "## Setup" in raws[0].text


# ── .pdf ─────────────────────────────────────────────────────────────────────

def test_pdf_loader_yields_one_raw_document_per_page(sample_pdf: Path):
    raws = list(load(sample_pdf, "upload"))
    # The fixture has 3 pages; even blank pages yield a RawDocument with
    # empty text. The splitter filters empties, not the loader.
    assert len(raws) == 3
    for i, raw in enumerate(raws, start=1):
        assert raw.metadata["format"] == ".pdf"
        assert raw.metadata["page_number"] == i
        assert raw.metadata["total_pages"] == 3


# ── .html ────────────────────────────────────────────────────────────────────

def test_html_loader_strips_script_and_style(sample_html: Path):
    raws = list(load(sample_html, "upload"))
    assert len(raws) == 1
    text = raws[0].text
    assert "console.log" not in text, "script content should be stripped"
    assert "color: red" not in text, "style content should be stripped"
    assert "Visible Heading" in text
    assert "nested" in text


def test_html_loader_captures_title(sample_html: Path):
    raws = list(load(sample_html, "upload"))
    assert raws[0].metadata["title"] == "Sample Document"


# ── .docx ────────────────────────────────────────────────────────────────────

def test_docx_loader_yields_one_raw_document_per_paragraph(sample_docx: Path):
    raws = list(load(sample_docx, "upload"))
    # 3 paragraphs in the fixture, but the empty one is filtered out.
    assert len(raws) == 2
    assert raws[0].metadata["paragraph_number"] == 1
    assert raws[0].metadata["style"] == "Normal"
    assert raws[1].metadata["paragraph_number"] == 2
    assert raws[1].metadata["style"] == "Heading 2"


# ── .csv ─────────────────────────────────────────────────────────────────────

def test_csv_loader_yields_one_raw_document_per_data_row(sample_csv: Path):
    raws = list(load(sample_csv, "upload"))
    assert len(raws) == 3
    assert raws[0].metadata["row_number"] == 1
    assert raws[0].metadata["headers"] == ["name", "role", "team"]
    # Each row is formatted as "header: value\n..." — the first row's text
    # contains all three fields.
    assert "name: Alice" in raws[0].text
    assert "role: Engineer" in raws[0].text
    assert "team: Frontend" in raws[0].text


def test_csv_loader_handles_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("", encoding="utf-8")
    assert list(load(p, "upload")) == []


def test_csv_loader_handles_header_only(tmp_path):
    p = tmp_path / "header_only.csv"
    p.write_text("a,b,c\n", encoding="utf-8")
    assert list(load(p, "upload")) == []
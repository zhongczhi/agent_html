"""Loader for .csv files. Emits one RawDocument per data row formatted as
"header1: value1\nheader2: value2\n..." with row_number and headers
metadata. The first row is treated as headers (no RawDocument emitted
for it).
"""
import csv
from pathlib import Path
from typing import Iterator

from backend.rag.loaders import RawDocument, register


@register(".csv")
def load_csv(path: Path, source: str) -> Iterator[RawDocument]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    headers = rows[0]
    for i, row in enumerate(rows[1:], start=1):
        cells = []
        for h, v in zip(headers, row):
            h = h.strip() or "(unnamed)"
            cells.append(f"{h}: {v.strip()}")
        yield RawDocument(
            text="\n".join(cells),
            metadata={
                "format": ".csv",
                "row_number": i,
                "headers": headers,
            },
        )
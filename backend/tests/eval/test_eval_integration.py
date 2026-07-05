import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "eval_hotpotqa.py"
FIXTURE = ROOT / "backend" / "tests" / "eval" / "fixtures" / "integration_hotpot.json"
CACHE_ROOT = ROOT / "storage" / "eval" / "hotpotqa" / "cache"


@pytest.fixture(autouse=True)
def clean_cache():
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT, ignore_errors=False)
    yield
    # Leave cache intact by default; tests can override via fixture.


def _run():
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--fixture", str(FIXTURE), "--k", "4"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
        timeout=60,
    )


def test_eval_first_run_cold_cache():
    p = _run()
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    for label in ("paragraph_recall@4", "sf_precision", "sf_recall", "sf_f1", "sf_em"):
        assert label in p.stdout, f"{label!r} missing from:\n{p.stdout}"
    m = re.search(r"cache hits / builds\s+:\s+(\d+)\s+/\s+(\d+)", p.stdout)
    assert m is not None, p.stdout
    assert m.group(2) == "5", f"expected 5 builds, got {p.stdout}"


def test_eval_second_run_warm_cache():
    _run()
    p = _run()
    assert p.returncode == 0, p.stdout + "\n" + p.stderr
    m = re.search(r"cache hits / builds\s+:\s+(\d+)\s+/\s+(\d+)", p.stdout)
    assert m is not None
    assert m.group(1) == "5" and m.group(2) == "0", p.stdout

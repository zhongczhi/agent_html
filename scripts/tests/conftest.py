"""Make `scripts/ingest_hotpotqa.py` importable from tests without installing
the repo as a package. Appends `scripts/` to sys.path the first time a test
imports it.
"""
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

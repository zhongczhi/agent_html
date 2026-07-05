"""Regression tests for scripts/ingest_hotpotqa.py::download_or_use_cache.

The Wayback Machine's TLS chain fails strict RFC 5280 verification under
urllib's default ssl context ("Basic Constraints of CA cert not marked
critical"). This test pins the fix: download_or_use_cache MUST use
`requests.get` (with certifi's bundle), NOT urllib.request.urlopen, so
the same fetch that works in Chrome also works in the script.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import ingest_hotpotqa


@pytest.fixture
def redirected_cache(tmp_path, monkeypatch):
    """Point CACHE_PATH at a temp file so tests never touch the real
    `scripts/.cache/` directory."""
    cache = tmp_path / "hotpot.json"
    monkeypatch.setattr(ingest_hotpotqa, "CACHE_PATH", cache)
    return cache


@pytest.fixture
def no_sleep(monkeypatch):
    """The function sleeps 5s between failed attempts; tests don't."""
    monkeypatch.setattr(ingest_hotpotqa.time, "sleep", lambda _s: None)


def _fake_resp(body: bytes) -> MagicMock:
    r = MagicMock()
    r.content = body
    r.raise_for_status = lambda: None
    return r


def test_download_uses_requests_not_urllib(redirected_cache):
    """Pin the dependency: requests.get, with a User-Agent header and
    timeout. Wayback is strict about missing UAs and a hung connection."""
    fake = _fake_resp(b'[{"_id":"a"}]')
    with patch("requests.get", return_value=fake) as mock_get:
        result = ingest_hotpotqa.download_or_use_cache(
            force=True, url="http://example.test/x.json"
        )
    assert result == redirected_cache
    assert redirected_cache.read_bytes() == b'[{"_id":"a"}]'
    # The single positional arg is the URL.
    assert mock_get.call_args.args[0] == "http://example.test/x.json"
    # Headers must include the agent ident.
    headers = mock_get.call_args.kwargs["headers"]
    assert "User-Agent" in headers
    # Timeout is set so a hung connection doesn't block the CLI forever.
    assert mock_get.call_args.kwargs["timeout"] == 60


def test_download_skips_when_cache_exists_and_no_force(redirected_cache):
    """Cache short-circuit: no HTTP call, cached bytes returned untouched."""
    redirected_cache.write_bytes(b"cached bytes")
    with patch("requests.get") as mock_get:
        result = ingest_hotpotqa.download_or_use_cache(
            force=False, url="http://example.test/x.json"
        )
    mock_get.assert_not_called()
    assert result == redirected_cache
    assert redirected_cache.read_bytes() == b"cached bytes"


def test_download_retries_once_then_exits_on_two_failures(
    redirected_cache, no_sleep, capsys
):
    """Two failures → sys.exit(1) and a stderr hint that points at
    --from-local so the user has an actionable next step."""
    with patch("requests.get", side_effect=Exception("ssl boom")):
        with pytest.raises(SystemExit) as exc_info:
            ingest_hotpotqa.download_or_use_cache(
                force=True, url="http://example.test/x.json"
            )
    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Download failed after 2 attempts" in err
    assert "--from-local" in err


def test_download_recovers_on_second_attempt(redirected_cache, no_sleep):
    """First request raises, second returns the file. retry logic is intact."""
    success = _fake_resp(b'{"ok": true}')
    with patch(
        "requests.get", side_effect=[Exception("transient"), success]
    ) as mock_get:
        result = ingest_hotpotqa.download_or_use_cache(
            force=True, url="http://example.test/x.json"
        )
    assert mock_get.call_count == 2
    assert result == redirected_cache
    assert redirected_cache.read_bytes() == b'{"ok": true}'


def test_download_raises_for_status(redirected_cache, no_sleep, capsys):
    """HTTP errors (raise_for_status) are treated like any other failure
    and surface the --from-local hint after the retry budget is exhausted."""
    bad = MagicMock()
    bad.raise_for_status = lambda: (_ for _ in ()).throw(
        requests.HTTPError("500 Server Error")
    )
    with patch("requests.get", return_value=bad):
        with pytest.raises(SystemExit) as exc_info:
            ingest_hotpotqa.download_or_use_cache(
                force=True, url="http://example.test/x.json"
            )
    assert exc_info.value.code == 1
    assert "--from-local" in capsys.readouterr().err

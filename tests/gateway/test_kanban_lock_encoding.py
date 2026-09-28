"""BOM handling for the singleton dispatcher lock (``kanban/.dispatcher.lock``).

``scripts/check-windows-footguns.py`` (blocking CI job `Python lints /
Windows footguns`) flagged the two lock-file opens in
``gateway/kanban_watchers_common.py``: both READ the lease record and hand it
to ``json.loads``, so a BOM-prefixed file — Windows tooling (PowerShell
Set-Content/Out-File, some editors) BOMs files it touches — would make the
parse fail and silently degrade the record to ``{}``.

The policy is one-sided in each direction: READS must be ``utf-8-sig``
(BOM'd and BOM-less files parse alike), WRITES must be ``utf-8`` (never emit
a BOM ourselves). ``_dispatcher_takeover_challenge`` does both through a
single ``r+`` handle, so this module pins both halves:

* a BOM'd lease still parses (the fix), and
* rewriting the lease leaves no BOM behind (the invariant the fix must not
  break — a BOM in the lock file is exactly what plain-utf-8 readers,
  including ``jq`` during an incident, choke on).
"""

from __future__ import annotations

import json
from pathlib import Path

from gateway.kanban_watchers_common import (
    _dispatcher_takeover_challenge,
    _read_dispatcher_lease,
)

BOM = b"\xef\xbb\xbf"


def _write_lock(path, record: dict, *, bom: bool) -> None:
    path.write_bytes((BOM if bom else b"") + json.dumps(record).encode("utf-8"))


def _lock(tmp_path, *, bom: bool) -> Path:
    path = tmp_path / ".dispatcher.lock"
    _write_lock(path, {"pid": 4242, "profile": "default"}, bom=bom)
    return path


def test_reads_a_bom_prefixed_lease(tmp_path) -> None:
    """utf-8-sig on the read side: a BOM'd lock file still parses."""
    path = _lock(tmp_path, bom=True)
    assert _read_dispatcher_lease(path) == {"pid": 4242, "profile": "default"}


def test_reads_a_bom_less_lease(tmp_path) -> None:
    """utf-8-sig must also read the file our own writers produce."""
    path = _lock(tmp_path, bom=False)
    assert _read_dispatcher_lease(path) == {"pid": 4242, "profile": "default"}


def test_missing_or_garbage_lock_is_empty(tmp_path) -> None:
    """Never raises: no file, empty file, non-dict JSON, non-JSON."""
    missing = tmp_path / "nope.lock"
    assert _read_dispatcher_lease(missing) == {}

    for payload in (b"", b"   ", b"[]", b"{not json", BOM + b"{not json"):
        path = tmp_path / f"case-{len(payload)}.lock"
        path.write_bytes(payload)
        assert _read_dispatcher_lease(path) == {}, payload


def test_challenge_reads_bom_lease_and_keeps_it_bom_free(tmp_path) -> None:
    """Read-modify-write through one ``r+`` handle: tolerant read, clean write."""
    path = _lock(tmp_path, bom=True)

    _dispatcher_takeover_challenge(path, "profile not dispatch-eligible", "other")

    raw = path.read_bytes()
    assert not raw.startswith(BOM), "the challenge writer must never emit a BOM"
    record = json.loads(raw.decode("utf-8"))  # plain utf-8 parse — no BOM
    assert record["pid"] == 4242  # the existing lease survived the rewrite
    assert record["challenge"]["by"] == "other"
    assert record["challenge"]["reason"] == "profile not dispatch-eligible"
    assert isinstance(record["challenge"]["at"], int)


def test_challenge_leaves_a_bom_less_lock_bom_less(tmp_path) -> None:
    """The common case (our own writer produced the file): no BOM appears."""
    path = _lock(tmp_path, bom=False)

    _dispatcher_takeover_challenge(path, "stale holder", "other")

    raw = path.read_bytes()
    assert not raw.startswith(BOM)
    assert json.loads(raw.decode("utf-8"))["challenge"]["reason"] == "stale holder"


def test_challenge_never_creates_the_lock_file(tmp_path) -> None:
    """The lock file must already exist: r+ (not a+) keeps creation exclusive."""
    path = tmp_path / "sub" / ".dispatcher.lock"
    _dispatcher_takeover_challenge(path, "stale holder", "other")
    assert not path.exists()

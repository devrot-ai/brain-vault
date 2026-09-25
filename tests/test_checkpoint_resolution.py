"""Regression tests for canonical checkpoint resolution.

Guards the single resolve_checkpoint() implementation in brainvuln.config:
no filesystem discovery, no mtime selection, loud failure on a missing
checkpoint, deterministic SHA256, and identical identity across app.py
and predict.py.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainvuln.config import (  # noqa: E402
    checkpoint_identity,
    resolve_checkpoint,
    sha256_file,
)


def _write_config(tmp_path: Path, checkpoint: str) -> Path:
    cfg = {
        "run": {"name": "test", "seed": 42, "output_dir": "results"},
        "ahba": {"missing": None},
        "atlases": {"primary": "dk68_tianS1", "dk68_tianS1": {"kind": "union"}},
        "inference": {"checkpoint": {"path": checkpoint}},
    }
    cfg_path = tmp_path / "analysis_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


def _fake_checkpoint(tmp_path: Path, name: str, payload: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_configured_checkpoint_wins_over_newer_file(tmp_path):
    a = _fake_checkpoint(tmp_path, "a.pt", b"checkpoint-A")
    b = _fake_checkpoint(tmp_path, "b.pt", b"checkpoint-B")
    old = 1_000_000_000
    os.utime(a, (old, old))
    os.utime(b, (old + 9999, old + 9999))  # b much newer than a
    cfg = _write_config(tmp_path, str(a))
    resolved = resolve_checkpoint(config_path=cfg)
    assert resolved == a.resolve()
    identity = checkpoint_identity(config_path=cfg)
    assert identity["filename"] == "a.pt"
    assert identity["size_bytes"] == len(b"checkpoint-A")


def test_missing_checkpoint_fails_loudly(tmp_path):
    cfg = _write_config(tmp_path, str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_checkpoint(config_path=cfg)


def test_sha256_deterministic(tmp_path):
    p = _fake_checkpoint(tmp_path, "c.pt", b"deterministic-bytes")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert h1 == hashlib.sha256(b"deterministic-bytes").hexdigest()


def test_app_and_cli_share_checkpoint_identity():
    import app
    import predict
    cli_path = predict._resolve_model_path(None)
    cli_identity = checkpoint_identity(str(cli_path))
    model, app_path, threshold, app_identity = app._get_model()
    assert app_path.resolve() == cli_path.resolve()
    assert app_identity["sha256"] == cli_identity["sha256"]
    assert app_identity["size_bytes"] == cli_identity["size_bytes"]
    assert threshold > 0


def test_no_mtime_checkpoint_discovery_anywhere():
    import brainvuln.config as cfg_mod
    src = inspect.getsource(cfg_mod.resolve_checkpoint)
    assert "glob" not in src
    assert "st_mtime" not in src
    for fname in ["app.py", "predict.py", "scripts/validate_external.py"]:
        text = (PROJECT_ROOT / fname).read_text(encoding="utf-8")
        assert "st_mtime" not in text, fname
        assert ".glob(" not in text, fname

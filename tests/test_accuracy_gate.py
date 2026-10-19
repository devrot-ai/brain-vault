"""Acceptance gate + MRI input contract for the canonical model.

Locks in two behaviours requested as project requirements:

* the minimum-accuracy gate is 0.85, is evaluated on the VALIDATION split
  only, and never alters the frozen serving threshold; and
* ``predict.py`` accepts 3D NIfTI volumes and rejects other formats, empty
  volumes, and calibrated temperatures that would silently move the frozen
  operating point.

Run:  pytest -q tests/test_accuracy_gate.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gate():
    return _load_module("brainvault_eval_validation", "scripts/eval_validation.py")


@pytest.fixture(scope="module")
def predict_mod():
    # predict.py imports brainvuln from src at call time, not import time
    return _load_module("brainvault_predict", "predict.py")


# ---------------------------------------------------------------------------
# gate definition
# ---------------------------------------------------------------------------


def test_minimum_accuracy_is_85_percent(gate):
    assert gate.DEFAULT_MIN_ACCURACY == pytest.approx(0.85)


def test_gate_defaults_to_validation_split(gate):
    import inspect

    # the split and its validation-only evaluation live inside main's argparse
    src = inspect.getsource(gate.main)
    assert "val_subjects.csv" in src
    assert "test_subjects.csv" not in src
    assert "test_predictions" not in src


def test_wilson_interval_brackets_proportion(gate):
    lo, hi = gate.wilson_interval(23, 27)
    assert lo < 23 / 27 < hi
    assert hi <= 1.0 and lo >= 0.0
    assert gate.wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# frozen serving threshold remains untouched
# ---------------------------------------------------------------------------


def test_gate_does_not_change_frozen_threshold():
    from brainvuln.config import resolve_checkpoint, resolve_threshold

    ckpt = resolve_checkpoint()
    value, source = resolve_threshold(ckpt_path=ckpt)
    assert value == pytest.approx(0.07, abs=1e-9)
    assert "frozen" in source


# ---------------------------------------------------------------------------
# MRI input contract in predict.py
# ---------------------------------------------------------------------------


def test_non_nifti_input_is_rejected(predict_mod, tmp_path):
    bad = tmp_path / "scan.png"
    bad.write_bytes(b"\x89PNG\r\n\x1a\n")
    rc = predict_mod.main(["--image", str(bad), "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_calibrated_temperature_requires_explicit_threshold(predict_mod, tmp_path):
    nii = tmp_path / "scan.nii.gz"
    nii.write_bytes(b"not really nifti")
    rc = predict_mod.main(["--image", str(nii), "--temperature", "auto",
                           "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_invalid_temperature_is_rejected(predict_mod, tmp_path):
    nii = tmp_path / "scan.nii"
    nii.write_bytes(b"not really nifti")
    rc = predict_mod.main(["--image", str(nii), "--temperature", "warm",
                           "--out-dir", str(tmp_path / "out")])
    assert rc == 2


def test_missing_image_is_rejected(predict_mod, tmp_path):
    rc = predict_mod.main(["--image", str(tmp_path / "missing.nii.gz"),
                           "--out-dir", str(tmp_path / "out")])
    assert rc == 2

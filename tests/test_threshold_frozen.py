"""REAL MODEL EVALUATION regression tests: the frozen operating threshold.

The threshold was selected on VALIDATION subjects only (Youden's J; see
audit_artifacts/threshold_selection.md) and frozen before any test-set use.
These tests pin three sources to the same number so inference can never
silently drift to a different operating point:

* ``inference.threshold.value`` in config/analysis.yaml (the freeze);
* the run's frozen metrics file (metrics/test_metrics.json), written by the
  tested train/eval path after validation-only selection;
* the value actually resolved by the serving path (predict.py / app.py)
  through brainvuln.config.resolve_threshold().
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainvuln.config import (  # noqa: E402
    ConfigError,
    load_analysis_config,
    resolve_checkpoint,
    resolve_threshold,
)

FROZEN_THRESHOLD = 0.07  # validation-selected Youden operating point


def _approx(value: float) -> bool:
    """The Youden grid stores 0.06999999999999999; YAML writes 0.07.

    Same number to well within 1e-9 (the tolerance used in the
    threshold_selection.json agreement record).
    """
    return abs(value - FROZEN_THRESHOLD) < 1e-9


def _config_value() -> float:
    cfg = load_analysis_config()
    assert cfg.inference.threshold_value is not None, (
        "inference.threshold.value missing from config/analysis.yaml — "
        "the threshold freeze was removed")
    return cfg.inference.threshold_value


def _metrics_value() -> float:
    metrics = (resolve_checkpoint().parents[1] / "metrics" / "test_metrics.json")
    assert metrics.exists(), f"frozen metrics file missing: {metrics}"
    return float(json.loads(metrics.read_text(encoding="utf-8"))["threshold"])


def test_config_threshold_matches_frozen_metrics_file():
    assert _approx(_config_value())
    assert _approx(_metrics_value())
    assert abs(_config_value() - _metrics_value()) < 1e-12


def test_resolve_threshold_uses_frozen_metrics_file():
    value, source = resolve_threshold()
    assert _approx(value)
    assert "frozen validation threshold" in source


def test_explicit_override_is_labeled_and_wins():
    value, source = resolve_threshold(override=0.35)
    assert value == 0.35
    assert "override" in source


def test_config_fallback_and_labeled_default(tmp_path):
    """Without a metrics file: config freeze, else an explicit 0.5 fallback."""
    ckpt = tmp_path / "run" / "checkpoints" / "best.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"placeholder")
    assert not (ckpt.parents[1] / "metrics" / "test_metrics.json").exists()

    # config freeze is honored when no metrics file exists
    cfg = {
        "run": {"name": "test", "seed": 42, "output_dir": "results"},
        "ahba": {"missing": None},
        "atlases": {"primary": "dk68_tianS1", "dk68_tianS1": {"kind": "union"}},
        "inference": {"checkpoint": {"path": str(ckpt)}, "threshold": {"value": 0.07}},
    }
    cfg_path = tmp_path / "analysis_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    value, source = resolve_threshold(ckpt_path=ckpt, config_path=cfg_path)
    assert _approx(value)
    assert "config" in source

    # nothing frozen anywhere: loud, labeled 0.5 fallback (never silent)
    cfg["inference"]["threshold"] = {"value": None}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    value, source = resolve_threshold(ckpt_path=ckpt, config_path=cfg_path)
    assert value == 0.5
    assert "no frozen threshold" in source

    # a metrics file with no usable threshold fails loudly
    mdir = ckpt.parents[1] / "metrics"
    mdir.mkdir()
    (mdir / "test_metrics.json").write_text(json.dumps({"test": {}}))
    try:
        import pytest
        with pytest.raises(ConfigError):
            resolve_threshold(ckpt_path=ckpt, config_path=cfg_path)
    finally:
        (mdir / "test_metrics.json").unlink()


def test_serving_path_resolves_the_frozen_threshold():
    """The production app path resolves exactly the frozen value."""
    import app
    _model, _ckpt_path, threshold, _identity = app._get_model()
    assert _approx(threshold)
    assert _approx(resolve_threshold()[0])


def test_predict_py_goes_through_resolve_threshold():
    """predict.py must use the shared resolver (no ad-hoc threshold logic)."""
    src = (PROJECT_ROOT / "predict.py").read_text(encoding="utf-8")
    assert "resolve_threshold(" in src
    assert "default 0.5 (no metrics file)" not in src  # old silent fallback gone
    # the old inline precedence logic is gone from predict.py (it owns none of it)
    assert "metrics_path.exists()" not in src

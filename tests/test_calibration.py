"""Tests for brainvuln.mri.calibration — Item 1 of the improvement plan."""

from __future__ import annotations

import numpy as np

from brainvuln.mri.calibration import (
    apply_temperature,
    cost_optimal_threshold,
    fit_temperature,
    youden_threshold,
)


def test_fit_temperature_returns_reasonable_value():
    rng = np.random.default_rng(0)
    y = (rng.random(40) > 0.5).astype(float)
    # raw logits already reasonably scaled: T should stay near 1
    z = (y - 0.5) * 4.0
    T = fit_temperature(z, y)
    assert isinstance(T, float)
    assert 0.5 <= T <= 10.0


def test_temperature_pushes_probability_toward_half():
    z = np.array([2.0, -2.0])
    t0 = apply_temperature(z, 1.0)
    t2 = apply_temperature(z, 2.0)
    # larger T -> probabilities closer to 0.5
    assert abs(t2[0] - 0.5) < abs(t0[0] - 0.5)
    assert abs(t2[1] - 0.5) < abs(t0[1] - 0.5)


def test_cost_optimal_threshold_selects_sensible_cut():
    # 5 positives (high p) + 5 negatives (low p)
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.15, 0.2, 0.25, 0.3])
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    t, cost = cost_optimal_threshold(p, y)
    assert 0.3 < t < 0.6
    assert cost < 0.1


def test_cost_optimal_fn_cost_shifts_threshold_down():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.15, 0.2, 0.25, 0.3])
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    t_low_fn, _ = cost_optimal_threshold(p, y, cost_fn=1.0)
    t_high_fn, _ = cost_optimal_threshold(p, y, cost_fn=10.0)
    # weighing FN more -> lower threshold (fewer missed positives)
    assert t_high_fn <= t_low_fn


def test_youden_threshold_matches_recognized_baseline():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.1, 0.15, 0.2, 0.25, 0.3])
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    t = youden_threshold(p, y)
    assert 0.3 < t < 0.6


def test_rejects_mismatched_shapes():
    p = np.array([0.5, 0.6])
    y = np.array([1, 0, 0])
    try:
        cost_optimal_threshold(p, y)
        assert False, "expected ValueError"
    except ValueError:
        pass

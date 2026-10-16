from __future__ import annotations

from brainvuln.monitor import cohort_drift_report


def test_no_drift_on_training_like_cohort():
    r = cohort_drift_report(
        new_ages=[76.0, 77.0, 75.0, 78.0],
        new_sex_pct_female=59.0,
        train_age_mean=76.76,
        train_age_sd=7.12,
        train_sex_pct_female=59.0,
    )
    assert all(not v["warning"] for v in r.values())


def test_drift_flagged_on_extreme_age():
    r = cohort_drift_report(
        new_ages=[90.0, 92.0, 93.0],
        new_sex_pct_female=60.0,
        train_age_mean=76.76,
        train_age_sd=7.12,
        train_sex_pct_female=59.0,
    )
    assert r["age"]["warning"] is True

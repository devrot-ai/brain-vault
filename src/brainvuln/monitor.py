"""Serving-time cohort-drift detection (Item 7)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def standardised_mean_diff(a: float, b: float, pooled_sd: float) -> float:
    if pooled_sd <= 0:
        return float("nan")
    return (a - b) / pooled_sd


def cohort_drift_report(
    new_ages: Optional[list[float]],
    new_sex_pct_female: Optional[float],
    train_age_mean: float,
    train_age_sd: float,
    train_sex_pct_female: float,
    age_smd_threshold: float = 0.2,
    sex_smd_threshold: float = 0.15,
) -> dict[str, dict[str, float]]:
    new_age_mean = float(np.mean(new_ages)) if new_ages else train_age_mean
    age_sd = float(np.std(new_ages)) if new_ages and len(new_ages) > 1 else train_age_sd
    pooled = math.sqrt((train_age_sd**2 + age_sd**2) / 2.0) if (train_age_sd > 0 and age_sd > 0) else train_age_sd
    sex_smd = standardised_mean_diff(
        new_sex_pct_female if new_sex_pct_female is not None else train_sex_pct_female,
        train_sex_pct_female,
        100.0,
    )
    return {
        "age": {
            "mean_new": new_age_mean,
            "mean_train": train_age_mean,
            "smd": standardised_mean_diff(new_age_mean, train_age_mean, pooled),
            "threshold": age_smd_threshold,
            "warning": abs(standardised_mean_diff(new_age_mean, train_age_mean, pooled)) > age_smd_threshold,
        },
        "sex": {
            "pct_female_new": (
                new_sex_pct_female if new_sex_pct_female is not None else train_sex_pct_female
            ),
            "pct_female_train": train_sex_pct_female,
            "smd": sex_smd,
            "threshold": sex_smd_threshold,
            "warning": abs(sex_smd) > sex_smd_threshold,
        },
    }

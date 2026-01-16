"""Tests for the data simulation module."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.simulator import generate_raw_data, prepare_cohort_data


class TestGenerateRawData:
    """Tests for generate_raw_data."""

    def test_output_shape(self) -> None:
        raw = generate_raw_data(num_users=100, num_months=6, seed=0)
        assert len(raw) == 100 * 6

    def test_columns(self) -> None:
        raw = generate_raw_data(num_users=50, num_months=4, seed=0)
        expected = {"user_id", "signup_month", "month", "spend", "treatment"}
        assert set(raw.columns) == expected

    def test_treatment_matches_signup(self) -> None:
        raw = generate_raw_data(num_users=200, num_months=6, seed=42)
        assert (raw["treatment"] == (raw["signup_month"] > 0)).all()

    def test_month_range(self) -> None:
        raw = generate_raw_data(num_users=50, num_months=12, seed=1)
        assert raw["month"].min() == 1
        assert raw["month"].max() == 12

    def test_seed_reproducibility(self) -> None:
        a = generate_raw_data(num_users=100, num_months=6, seed=99)
        b = generate_raw_data(num_users=100, num_months=6, seed=99)
        assert a.equals(b)


class TestPrepareCohortData:
    """Tests for prepare_cohort_data."""

    def test_output_columns(self) -> None:
        raw = generate_raw_data(num_users=200, num_months=12, seed=0)
        cohort = prepare_cohort_data(raw, signup_month=3)
        expected = {"user_id", "signup_month", "treatment", "pre_spends", "post_spends"}
        assert set(cohort.columns) == expected

    def test_only_relevant_cohorts(self) -> None:
        raw = generate_raw_data(num_users=500, num_months=12, seed=0)
        cohort = prepare_cohort_data(raw, signup_month=3)
        assert set(cohort["signup_month"].unique()).issubset({0, 3})

    def test_no_empty_result(self) -> None:
        raw = generate_raw_data(num_users=1000, num_months=12, seed=0)
        cohort = prepare_cohort_data(raw, signup_month=3)
        assert len(cohort) > 0

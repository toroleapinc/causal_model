"""Simulated loyalty program data generation.

Generates synthetic observational data mimicking a loyalty program where
customers may sign up in a given month and exhibit changed spending behavior
post-enrollment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_raw_data(
    num_users: int = 10_000,
    num_months: int = 12,
    base_spend_lambda: int = 500,
    month_decay_rate: int = 10,
    treatment_effect: int = 100,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate raw panel data for the loyalty program simulation.

    Each user is observed across *num_months* months.  Approximately half of
    users sign up for the program in a random month; the rest never enroll.
    Spending follows a Poisson baseline that decays linearly over the year,
    with a fixed additive treatment effect applied **after** the signup month.

    Args:
        num_users: Number of unique customers.
        num_months: Number of observation months (1-indexed).
        base_spend_lambda: Poisson λ for baseline monthly spend.
        month_decay_rate: Linear spend decay per month.
        treatment_effect: Additive spend lift after enrollment.
        seed: Optional random seed for reproducibility.

    Returns:
        DataFrame with columns ``user_id``, ``signup_month``, ``month``,
        ``spend``, and ``treatment``.
    """
    rng = np.random.default_rng(seed)

    signup_months = (
        rng.choice(np.arange(1, num_months), num_users)
        * rng.integers(0, 2, size=num_users)
    )

    df = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(num_users), num_months),
            "signup_month": np.repeat(signup_months, num_months),
            "month": np.tile(np.arange(1, num_months + 1), num_users),
            "spend": rng.poisson(base_spend_lambda, num_users * num_months),
        }
    )

    df["treatment"] = df["signup_month"] > 0
    df["spend"] = df["spend"] - df["month"] * month_decay_rate

    after_signup = (df["signup_month"] < df["month"]) & df["treatment"]
    df.loc[after_signup, "spend"] = df.loc[after_signup, "spend"] + treatment_effect

    return df


def prepare_cohort_data(
    df: pd.DataFrame,
    signup_month: int = 3,
) -> pd.DataFrame:
    """Aggregate panel data into a pre/post cohort for a single signup month.

    Filters to users who either signed up in *signup_month* or never signed up,
    then computes mean spend before and after that month per user.

    Args:
        df: Raw panel DataFrame from :func:`generate_raw_data`.
        signup_month: The enrollment month to isolate.

    Returns:
        DataFrame with columns ``user_id``, ``signup_month``, ``treatment``,
        ``pre_spends``, and ``post_spends``.
    """
    cohort = df[df.signup_month.isin([0, signup_month])].copy()

    aggregated = (
        cohort.groupby(["user_id", "signup_month", "treatment"])
        .apply(
            lambda x: pd.Series(
                {
                    "pre_spends": x.loc[x.month < signup_month, "spend"].mean(),
                    "post_spends": x.loc[x.month > signup_month, "spend"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    return aggregated

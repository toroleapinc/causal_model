"""Visualization helpers for causal analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_dag(model, output_path: str | Path | None = None) -> None:
    """Render the causal DAG from a DoWhy CausalModel.

    Args:
        model: A ``dowhy.CausalModel`` instance.
        output_path: Optional file path to save the rendered graph.
    """
    model.view_model()
    if output_path is not None:
        import shutil
        src = Path("causal_model.png")
        if src.exists():
            shutil.move(str(src), str(output_path))


def plot_treatment_effect(
    data: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "post_spends",
    title: str = "Post-Enrollment Spend Distribution",
) -> plt.Figure:
    """Box plot comparing outcome distributions across treatment groups.

    Args:
        data: Cohort DataFrame.
        treatment_col: Column indicating treatment assignment.
        outcome_col: Column with the outcome variable.
        title: Plot title.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [
        data.loc[data[treatment_col] == False, outcome_col].dropna(),  # noqa: E712
        data.loc[data[treatment_col] == True, outcome_col].dropna(),   # noqa: E712
    ]
    ax.boxplot(groups, labels=["Control", "Treatment"])
    ax.set_ylabel(outcome_col)
    ax.set_title(title)
    plt.tight_layout()
    return fig

"""DoWhy causal model wrapper for loyalty program analysis.

Encapsulates the full causal inference pipeline: graph specification,
effect identification, estimation via propensity score matching, and
refutation testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dowhy
import pandas as pd


CAUSAL_GRAPH: str = """digraph {
    treatment[label="Program Signup in month i"];
    pre_spends;
    post_spends;
    Z -> treatment;
    pre_spends -> treatment;
    treatment -> post_spends;
    signup_month -> post_spends;
    signup_month -> treatment;
}"""


@dataclass
class RefutationResult:
    """Container for a single refutation test outcome."""

    name: str
    estimated_effect: float
    new_effect: float
    p_value: float | None = None


@dataclass
class AnalysisResult:
    """Container for the full causal analysis output."""

    estimand: Any = None
    estimate: Any = None
    ate: float = 0.0
    refutations: list[RefutationResult] = field(default_factory=list)


class CausalAnalysis:
    """End-to-end causal inference pipeline using DoWhy.

    Args:
        data: Cohort-level DataFrame with ``treatment``, ``post_spends``,
              ``pre_spends``, and ``signup_month`` columns.
        treatment: Name of the treatment column.
        outcome: Name of the outcome column.
        graph: DOT-format causal graph string.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment: str = "treatment",
        outcome: str = "post_spends",
        graph: str = CAUSAL_GRAPH,
    ) -> None:
        self._data = data
        self._treatment = treatment
        self._outcome = outcome
        self._graph = graph.replace("\n", " ")
        self._model = dowhy.CausalModel(
            data=self._data,
            graph=self._graph,
            treatment=self._treatment,
            outcome=self._outcome,
        )
        self._result = AnalysisResult()

    @property
    def model(self) -> dowhy.CausalModel:
        """Return the underlying DoWhy CausalModel."""
        return self._model

    @property
    def result(self) -> AnalysisResult:
        """Return the latest analysis result."""
        return self._result

    def identify(self) -> Any:
        """Identify the causal estimand from the graph.

        Returns:
            The identified estimand object.
        """
        self._result.estimand = self._model.identify_effect(
            proceed_when_unidentifiable=True,
        )
        return self._result.estimand

    def estimate(
        self,
        method_name: str = "iv.propensity_score_matching",
        target_units: str = "att",
    ) -> Any:
        """Estimate the causal effect.

        Args:
            method_name: DoWhy estimation method identifier.
            target_units: Target population for the estimate (``att``, ``ate``, ``atc``).

        Returns:
            The estimate object.

        Raises:
            RuntimeError: If :meth:`identify` has not been called.
        """
        if self._result.estimand is None:
            raise RuntimeError("Call identify() before estimate().")

        self._result.estimate = self._model.estimate_effect(
            self._result.estimand,
            method_name=method_name,
            target_units=target_units,
        )
        self._result.ate = self._result.estimate.value
        return self._result.estimate

    def refute(
        self,
        placebo: bool = True,
        random_common_cause: bool = True,
        data_subset: bool = True,
        subset_fraction: float = 0.9,
    ) -> list[RefutationResult]:
        """Run refutation tests on the estimated effect.

        Args:
            placebo: Run placebo treatment refuter.
            random_common_cause: Run random common cause refuter.
            data_subset: Run data subset refuter.
            subset_fraction: Fraction of data to use for subset test.

        Returns:
            List of :class:`RefutationResult` objects.

        Raises:
            RuntimeError: If :meth:`estimate` has not been called.
        """
        if self._result.estimate is None:
            raise RuntimeError("Call estimate() before refute().")

        refutations: list[RefutationResult] = []
        refuters = []

        if placebo:
            refuters.append(("Placebo Treatment", "placebo_treatment_refuter"))
        if random_common_cause:
            refuters.append(("Random Common Cause", "random_common_cause"))
        if data_subset:
            refuters.append(("Data Subset", "data_subset_refuter"))

        for name, method in refuters:
            kwargs: dict[str, Any] = {}
            if method == "data_subset_refuter":
                kwargs["subset_fraction"] = subset_fraction

            ref = self._model.refute_estimate(
                self._result.estimand,
                self._result.estimate,
                method_name=method,
                **kwargs,
            )
            refutations.append(
                RefutationResult(
                    name=name,
                    estimated_effect=ref.estimated_effect,
                    new_effect=ref.new_effect,
                    p_value=getattr(ref, "refutation_result", {}).get("p_value")
                    if isinstance(getattr(ref, "refutation_result", None), dict)
                    else None,
                )
            )

        self._result.refutations = refutations
        return refutations

    def run(
        self,
        method_name: str = "iv.propensity_score_matching",
        target_units: str = "att",
        refute: bool = True,
    ) -> AnalysisResult:
        """Execute the full pipeline: identify → estimate → refute.

        Args:
            method_name: DoWhy estimation method.
            target_units: Target population.
            refute: Whether to run refutation tests.

        Returns:
            :class:`AnalysisResult` with all outputs populated.
        """
        self.identify()
        self.estimate(method_name=method_name, target_units=target_units)
        if refute:
            self.refute()
        return self._result

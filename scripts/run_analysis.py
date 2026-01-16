#!/usr/bin/env python3
"""CLI entry point for the causal inference pipeline.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --config config/default.yaml --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.simulator import generate_raw_data, prepare_cohort_data
from src.models.causal import CausalAnalysis
from src.utils import load_config, print_section


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run causal inference analysis on simulated loyalty program data.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-refute",
        action="store_true",
        help="Skip refutation tests.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full causal analysis pipeline."""
    args = parse_args()
    cfg = load_config(args.config)
    sim_cfg = cfg["simulation"]
    model_cfg = cfg["model"]

    # --- Data Generation ---
    print_section("Data Generation")
    raw = generate_raw_data(
        num_users=sim_cfg["num_users"],
        num_months=sim_cfg["num_months"],
        base_spend_lambda=sim_cfg["base_spend_lambda"],
        month_decay_rate=sim_cfg["month_decay_rate"],
        treatment_effect=sim_cfg["treatment_effect"],
        seed=args.seed,
    )
    print(f"Generated {len(raw):,} observations for {sim_cfg['num_users']:,} users.")

    cohort = prepare_cohort_data(raw, signup_month=sim_cfg["signup_month"])
    print(f"Cohort data: {len(cohort):,} users (month={sim_cfg['signup_month']}).")
    print(f"  Treatment: {cohort.treatment.sum():,}  |  Control: {(~cohort.treatment).sum():,}")

    # --- Causal Analysis ---
    print_section("Causal Identification & Estimation")
    analysis = CausalAnalysis(
        data=cohort,
        treatment=model_cfg["treatment"],
        outcome=model_cfg["outcome"],
    )
    analysis.identify()
    print(analysis.result.estimand)

    analysis.estimate(
        method_name=model_cfg["estimation_method"],
        target_units=model_cfg["target_units"],
    )
    print(f"\nEstimated ATE: {analysis.result.ate:.4f}")
    print(f"(True treatment effect: {sim_cfg['treatment_effect']})")

    # --- Refutation ---
    if not args.no_refute:
        print_section("Refutation Tests")
        refutations = analysis.refute()
        for ref in refutations:
            print(f"  {ref.name}:")
            print(f"    Original effect : {ref.estimated_effect:.4f}")
            print(f"    New effect      : {ref.new_effect:.4f}")
            if ref.p_value is not None:
                print(f"    p-value         : {ref.p_value:.4f}")
            print()

    print_section("Done")


if __name__ == "__main__":
    main()

# Causal Inference: Loyalty Program Impact Analysis

Estimating the causal effect of a loyalty program on customer spending using [DoWhy](https://github.com/py-why/dowhy) and propensity score matching.

## Overview

This project demonstrates a rigorous causal inference workflow to answer: **Does enrolling in a loyalty program cause customers to spend more?**

Using simulated observational data, we construct a causal DAG, apply propensity score matching to control for confounders, estimate the Average Treatment Effect on the Treated (ATT), and validate with refutation tests.

![Causal DAG](assets/causal_dag.png)

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run the full pipeline (CLI)
python scripts/run_analysis.py

# Or with a fixed seed
python scripts/run_analysis.py --seed 42

# Run tests
pytest tests/ -v

# Launch the demo notebook
jupyter notebook notebooks/causal_inference_demo.ipynb
```

Or use the Makefile:

```bash
make install
make run
make test
make notebook
```

## Methodology

### Causal Framework

1. **Causal Graph (DAG)** — Encodes the assumed data-generating process: signup timing, pre-enrollment spending, and unobserved confounders influence both treatment assignment and the outcome.

2. **Identification** — The backdoor criterion is applied to the DAG to identify valid adjustment sets.

3. **Estimation** — Propensity score matching creates comparable treated/control groups, then estimates the ATT.

4. **Refutation** — Three robustness checks validate the estimate:
   - **Placebo treatment** — Replace treatment with a random variable; effect should vanish.
   - **Random common cause** — Add a random confounder; estimate should remain stable.
   - **Data subset** — Re-estimate on a 90% subset; estimate should hold.

## Architecture

```
├── config/
│   └── default.yaml                 # Simulation & model parameters
├── src/
│   ├── data/
│   │   └── simulator.py             # Synthetic data generation
│   ├── models/
│   │   └── causal.py                # DoWhy pipeline wrapper
│   ├── visualization/
│   │   └── plots.py                 # DAG & treatment effect plots
│   └── utils.py                     # Config loading, helpers
├── notebooks/
│   └── causal_inference_demo.ipynb   # Interactive demo (imports from src/)
├── scripts/
│   └── run_analysis.py              # CLI entry point
├── assets/
│   └── causal_dag.png               # DAG visualization
├── tests/
│   └── test_simulator.py            # Data generation tests
├── pyproject.toml
├── Makefile
├── requirements.txt
└── LICENSE
```

## Data Schema

The simulation generates panel data with:

| Column | Description |
|---|---|
| `user_id` | Unique customer identifier |
| `signup_month` | Month of loyalty program enrollment (0 = never enrolled) |
| `month` | Observation month (1–12) |
| `spend` | Monthly spending (Poisson baseline with seasonal decay) |
| `treatment` | Whether the customer enrolled (boolean) |

The cohort aggregation step computes per-user `pre_spends` and `post_spends` relative to a target signup month.

## Configuration

All simulation and model parameters are defined in `config/default.yaml`:

```yaml
simulation:
  num_users: 10000
  num_months: 12
  treatment_effect: 100
  signup_month: 3

model:
  estimation_method: iv.propensity_score_matching
  target_units: att
```

## License

MIT — see [LICENSE](LICENSE).

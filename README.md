# Causal Model 🔬

**Causal inference with DoWhy — estimating treatment effects from observational data.**

A worked example demonstrating causal effect estimation using Microsoft's [DoWhy](https://github.com/py-why/dowhy) library, with simulated customer spend data.

## What's Inside

Simulates a loyalty program signup scenario with 10,000 users over 12 months:

- **Causal Graph Definition** — DAG specifying the assumed causal structure (treatment → outcome, with confounders)
- **Effect Identification** — Automated identification of estimands from the causal graph
- **Effect Estimation** — Propensity score matching (IV method) to estimate the Average Treatment Effect on the Treated (ATT)

## Key Concepts Demonstrated

- Defining causal graphs as DOT diagrams
- Pre/post treatment spend comparison
- Handling confounders (signup month, pre-treatment spending)
- DoWhy's identify → estimate → (refute) workflow

## Tech Stack

- **Causal Inference:** DoWhy
- **Data:** pandas, NumPy
- **Visualization:** DAG rendering via graphviz

## Quick Start

```bash
pip install dowhy pandas numpy
jupyter notebook causal_model/Untitled.ipynb
```

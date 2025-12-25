# Causal Inference: Loyalty Program Impact Analysis

Estimating the causal effect of a loyalty program on customer spending using DoWhy and propensity score matching.

## Overview

This project demonstrates a causal inference workflow to answer the question: **Does enrolling in a loyalty program cause customers to spend more?**

Using simulated observational data, we construct a causal graph (DAG), apply propensity score matching to control for confounders, and estimate the Average Treatment Effect (ATE) of program enrollment on monthly spend.

![Causal DAG](causal_model.png)

## Methodology

### Causal Framework

1. **Causal Graph (DAG)** — Define the assumed data-generating process: signup timing and other covariates influence both treatment assignment (loyalty program enrollment) and the outcome (spend).

2. **Identification** — Use the DAG to identify valid adjustment sets via the backdoor criterion.

3. **Estimation** — Apply **propensity score matching** to create comparable treated and control groups, then estimate the ATE.

4. **Refutation** — Run robustness checks (placebo treatment, random common cause, data subset) to validate the estimate.

### Tools

- [**DoWhy**](https://github.com/py-why/dowhy) — End-to-end causal inference library that handles identification, estimation, and refutation.

## Data

The dataset is **simulated** within the notebook and contains:

| Column | Description |
|---|---|
| `user_id` | Unique customer identifier |
| `signup_month` | Month the customer signed up |
| `month` | Observation month |
| `spend` | Monthly spending amount |
| `treatment` | Whether the customer enrolled in the loyalty program (0/1) |

## Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run

Open and execute the notebook:

```bash
jupyter notebook causal_inference_loyalty_program.ipynb
```

## Project Structure

```
├── causal_inference_loyalty_program.ipynb   # Full analysis notebook
├── causal_model.png                         # Causal DAG visualization
├── requirements.txt
├── LICENSE
└── README.md
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

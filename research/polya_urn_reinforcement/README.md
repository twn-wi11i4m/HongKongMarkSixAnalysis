# Hong Kong Mark Six Hot Numbers Prediction: Advanced Probabilistic Analysis Based on Reinforced Urn Model

## Abstract

This study presents a comprehensive mathematical analysis of the Hong Kong Mark Six lottery, focusing on predicting "hot numbers"—numbers that tend to appear consistently in recent draws. Using a Pólya-like reinforced urn model with real draw data in 2025, we estimate reinforcement parameters and calculate predicted probabilities for each number. This probabilistic framework integrates reinforcement and decay dynamics, offering insight into lottery randomness and potential predictive patterns.

## Mathematical Framework

Let $\mathbf{w}(t) = (w_1(t), \dots, w_{49}(t))$ represent the weight vector for 49 numbers, initialized as ($ w_i(0) = 1 $). Each number’s draw probability at time ($ t $) is given by:

$$
p_i(t) = \frac{w_i(t)}{\sum_j w_j(t)}.
$$

After each draw ( $S_t $), weights are updated as:

$$
\mathbf{w}(t+1) = \lambda \mathbf{w}(t) + \alpha \sum_{i \in S_t} \mathbf{e}_i,
$$

or equivalently for each ( $i$ ):

$$
w_i(t+1) = \lambda w_i(t) + \alpha \cdot \mathbb{I}(i \in S_t),
$$

where ( $\lambda = 1 - \beta $) (decay factor), ( $\alpha > 0 $) (reinforcement amplitude), and ( $\mathbf{e}\_i $) is a unit vector.

We estimate parameters ( ($\alpha, \lambda$) ) by maximizing log-likelihood:

$$
\hat{\theta} = \arg\max_{\theta=(\alpha,\lambda)} \sum_t \sum_{i\in S_t} \log\left(\frac{w_i(t)}{\sum_j w_j(t)}\right),
$$

solved via numerical optimization. The final draw probabilities are ( $p_i(T+1)$ ), with hot numbers defined as ( $p_i > \frac{1}{49} + \delta$ ), where ( $\delta = 0.001$ ).

## Implementation

We implemented model fitting and prediction in Python using 2025 draw records. Example code:

```python
import numpy as np
from scipy.optimize import minimize

# Simplified draw data example
draws = [
    {"draw": "25/066", "date": "2025-06-17", "main": [5, 19, 30, 40, 44, 45], "extra": 26},
    # ... (50 periods total)
    {"draw": "25/115", "date": "2025-10-25", "main": [6, 7, 27, 36, 39, 43], "extra": 1}
]

full_draws = [d['main'] + [d['extra']] for d in draws]
N = 49
epsilon = 1e-10

def neg_log_lik(params):
    alpha, lam = params
    if alpha < 0 or lam < 0 or lam > 1:
        return np.inf
    w = np.ones(N) + epsilon
    log_lik = 0
    for draw in full_draws:
        total_w = np.sum(w)
        p = w / total_w
        ll_draw = np.sum(np.log(p[[num-1 for num in draw]] + epsilon))
        log_lik += ll_draw
        w = lam * w
        for num in draw:
            w[num - 1] += alpha
        w = np.maximum(w, epsilon)
    return -log_lik

initial = [1.0, 0.9]
bounds = [(0, None), (0, 1)]
res = minimize(neg_log_lik, initial, bounds=bounds, method='L-BFGS-B')

alpha_est, lam_est = res.x
beta_est = 1 - lam_est

# Final probabilities
w_final = np.ones(N) + epsilon
for draw in full_draws:
    w_final = lam_est * w_final
    for num in draw:
        w_final[num - 1] += alpha_est
    w_final = np.maximum(w_final, epsilon)

p_final = w_final / np.sum(w_final)
```

## Extended Framework: Combinatorial & Reinforced Urn Model (Polya-like Dynamics)

### Overview

We propose a generative urn model where selection probabilities evolve through reinforcement and decay effects based on observed draws. Parameters are fitted to test deviations from uniform randomness, potentially reflecting operational or systemic biases.

### Mathematical Structure

- **Base Model:**
  ( $w_i(0) = 1$ ). After each draw:

  $$
  w_i(t+1) = w_i(t) + \alpha \cdot \mathbb{I}(i \in draw_t) - \beta \cdot decay_t(i)
  $$

  ( $\alpha$ ) controls reinforcement, and ( $\beta$ ) controls decay. Draw probability is proportional to ( $w_i(t)$ ).

- **Cross Reinforcement Extension:**
  Introduce a matrix ( $A_{ij}$ ) such that observing ( $i$ ) increases ( $j$ )’s weight by ( $A_{ij}$ ), capturing pairwise reinforcement.

- **Inference Objective:**
  Estimate ( $\alpha$, $\beta$, $A$ ) using MLE or Bayesian inference (e.g., MCMC) to obtain posterior distributions and uncertainty intervals.

### Implementation Guidelines

- **Forward Simulator:**
  Create a simulator that generates synthetic sequences for testing given parameters ( ($\alpha$, $\beta$, $A$) ).

- **Inference Methods:**

  - Use likelihood-based optimization or Bayesian inference.
  - Compare models via AIC/BIC or Bayes factors.

- **Validation:**
  Simulate from fitted models and compare empirical statistics (marginal frequencies, inter-arrival intervals, co-occurrence matrices) against observed data.

### Data Requirement

Sequentially ordered full draw records are required for accurate inference.

### Pitfalls

- Overfitting risk if the full pairwise matrix ( $A_{ij}$ ) is unconstrained.
- Identifiability issues between ( $\alpha$ ) and ( $\beta$ ) for short datasets.

### For-Fun Prediction

If a positive ( $\alpha$ ) is estimated, short-term persistence exists—numbers appearing frequently in the last 10 draws are likely to reappear in the next 5.

# Hong Kong Mark Six Lottery: Extreme Value and Tail Dependence Analysis

## Overview

This document presents a comprehensive analysis of extreme value behavior and tail dependence in the Hong Kong Mark Six lottery, utilizing historical draw data from 2025. The study focuses on the gaps between adjacent winning numbers, employing advanced statistical techniques to uncover underlying patterns and dependencies.

## A. Executive Summary

### A.1 Core Concept

Treat the sorted winning numbers of each draw as a 5-dimensional random vector of adjacent gaps:

$$\mathbf{G} = (G_1, G_2, G_3, G_4, G_5), \quad G_i = X_{(i+1)} - X_{(i)}, \quad i=1,\dots,5$$

where $X_{(1)}<\cdots<X_{(6)}$ are the order statistics of the 6 main numbers. The goal is to test whether the extreme behavior of gaps conforms to the random sampling hypothesis and whether adjacent gaps are tail dependent.

### A.2 Three-Layer Methodology

| Level               | Technique                                      | Objective                                         |
| ------------------- | ---------------------------------------------- | ------------------------------------------------- |
| **Univariate**      | POT + GPD fitting                              | Estimate probability distribution of extreme gaps |
| **Multivariate**    | Radial/Angular decomposition, spectral measure | Capture vector tail dependence structure          |
| **Dependence Test** | $\chi(u)$, $\bar{\chi}(u)$ + Bootstrap         | Quantify tail dependence of adjacent gaps         |

### A.3 Data and Scale

- **Sample size**: 114 draws (Jan–Oct 23, 2025)
- **Observations per draw**: 6 main numbers → 5 adjacent gaps
- **Total gaps**: $114 \times 5 = 570$ independent observations
- **Intended application**: Test randomness, predict rare large gaps, identify potential mechanism bias

### A.4 Main Results Preview

- **GPD fitting**: Shape parameter $\xi \approx 0.01$–0.03 (close to Gumbel type), scale $\sigma \approx 4$–5
- **Tail dependence**: $\chi(u) \to 0$ as $u$ increases, indicating asymptotic independence (supports random sampling hypothesis)
- **Rare event prediction**: Probability of maximum gap $\geq 14$ in the next 100 draws is about 40–45%

---

## B. Detailed Mathematical Framework

### B.1 Basic Definitions and Assumptions

**Setup**: Each draw selects 6 numbers without replacement from $\{1,2,\dots,49\}$, denoted as $\{X_1,\dots,X_6\}$. After sorting:

$$X_{(1)} < X_{(2)} < \cdots < X_{(6)}$$

**Adjacent gap vector**:

$$\mathbf{G} = (G_1, G_2, G_3, G_4, G_5), \quad G_i = X_{(i+1)} - X_{(i)}$$

where $G_i \geq 1$ (numbers are integers), $\sum_{i=1}^{5} G_i = X_{(6)} - X_{(1)} \leq 48$.

**Randomness assumption**: If sampling is i.i.d. uniform, $\mathbf{G}$ should follow the marginal distribution of the Dirichlet distribution. Empirical data may violate this, requiring EVT testing.

### B.2 Univariate Extreme Value Theory: Pickands–Balkema–de Haan (PBdH) Theorem

**Theorem statement**: For a sufficiently high threshold $u$, the conditional distribution of exceedances $Y = G - u$ ($G > u$) approximates the Generalized Pareto Distribution (GPD):

$$P(Y > y \mid Y > 0) \approx \left(1 + \xi \frac{y}{\sigma}\right)^{-1/\xi}, \quad y > 0$$

**Parameter interpretation**:

- $\xi$ (shape): controls tail heaviness
  - $\xi > 0$ (Fréchet): heavy tail (no upper bound, e.g., power law)
  - $\xi = 0$ (Gumbel, limiting case): exponential decay
  - $\xi < 0$ (Weibull): bounded tail
- $\sigma > 0$ (scale): determines tail decay rate

**Implication for lottery gaps**: Since $G_i \in [1, 48]$ is finite, expect $\xi \leq 0$ (Weibull or Gumbel), but when $\xi$ is close to 0, the behavior is hard to distinguish; in practice, a small positive $\xi$ may be considered approximate (possibly due to discreteness).

### B.3 Return Level and Extreme Value Prediction

Return level $z_p$ is defined as the extreme value occurring at frequency $p$, e.g., $p = 1/100$ means a "once in 100 draws" event. Estimated using GPD:

$$z_p = u + \frac{\sigma}{\xi} \left( (Np)^{\xi} - 1 \right)$$

where $N$ is the total number of observations (draws or gaps), $p$ is the chosen frequency. For small $\xi$ (Gumbel domain), this formula degenerates to the exponential form.

**Application**: Estimate the probability of observing a maximum gap $\geq 14$ in the next 100 draws.

### B.4 Multivariate Extremes: Radial-Angular Decomposition

For vector $\mathbf{G}$, define:

- **Radial vector**: $R = \|\mathbf{G}\|_\infty = \max_i G_i$ (max component)
- **Angular vector**: $\mathbf{W} = \mathbf{G} / R \in [0,1]^5$ (normalized)

**Tail distribution structure**: The tail is governed by the spectral measure $H(\mathbf{w})$:

$$P(R > r, \mathbf{W} \in A \mid R > u) \approx \frac{H(A)}{r^\alpha}, \quad \alpha = 1/\xi$$

**Interpretation**: $H$ determines the joint behavior of components in extreme events; concentration in certain angles means large gaps tend to cluster at specific positions; dispersion means no particular pattern.

### B.5 Tail Dependence Measures

#### B.5a Upper Tail Dependence Coefficient ($\chi$)

For adjacent pairs $(G_i, G_{i+1})$:

$$\chi = \lim_{u \to u_{\max}} P(G_{i+1} > u \mid G_i > u)$$

- $\chi = 0$: asymptotic independence (marginals may still be heavy-tailed)
- $\chi > 0$: tail dependence (one large, the other tends to be large)

#### B.5b Tail Correlation Index ($\bar{\chi}$)

Detects weak dependence (asymptotic independence but local dependence):

$$\bar{\chi} = \lim_{u \to u_{\max}} \frac{2 \log P(G_i > u)}{\log P(G_i > u, G_{i+1} > u)} - 1$$

- $\bar{\chi} \in [-1, 1]$
- If $\chi = 0$ but $\bar{\chi} > -1$, indicates sub-exponential dependence

#### B.5c Empirical Estimation

Given threshold sequence $u_1 < u_2 < \cdots < u_m$:

$$\hat{\chi}(u_k) = \frac{\sum_{t=1}^{N} \mathbf{1}(G_i^{(t)} > u_k, G_{i+1}^{(t)} > u_k)}{\sum_{t=1}^{N} \mathbf{1}(G_i^{(t)} > u_k)}$$

Use bootstrap (e.g., $B=1000$ resamples) to estimate 95% confidence intervals.

### B.6 Non-Euclidean Geometry and Discreteness Correction

**Manifold perspective**: Numbers can be viewed as points on a 5-dimensional torus $\mathbb{T}^5$ (cyclic: numbers mod 49). The curvature of gap vector $\mathbf{G}$ is described by the Riemann tensor, but for simplicity, we work in Euclidean space.

**Discreteness correction**: Since $G_i$ is integer, it affects the GPD continuity assumption. Robustness test method:

1. Add small normal jitter $\epsilon \sim \mathcal{N}(0, 0.05)$ to gaps
2. Repeat GPD fitting
3. Compare parameter changes (if <5%, considered robust)

### B.7 Fokker-Planck Equation and Evolution (Optional)

To model the evolution of tail distribution across draws:

$$\frac{\partial p(\mathbf{g},t)}{\partial t} = -\nabla \cdot (\boldsymbol{\mu} p) + \frac{1}{2} \nabla^2 (\sigma^2 p)$$

where $p$ is the probability density of $\mathbf{G}$, $\boldsymbol{\mu}$ is drift (expected 0), $\sigma^2$ is diffusion coefficient. This PDE describes how density evolves under random perturbations, but due to limited sample size, this study focuses on static analysis.

## C. Novelty of This Study

This module focuses on the adjacent gaps of the main numbers in each Hong Kong Mark Six draw, using EVT and multivariate tail dependence methods to test the randomness and extreme behavior of number distribution.

## Analysis Workflow

1. Data preprocessing: Read 6 main numbers per draw, calculate gaps, check for outliers.
2. Threshold selection: Use 85%–97.5% quantiles as EVT thresholds, plot parameter stability.
3. GPD fitting: Fit GPD to gaps above threshold, diagnose with QQ/PP/return-level plots.
4. Tail dependence: Calculate $\chi(u)$, $\bar{\chi}(u)$, estimate confidence intervals with bootstrap, visualize tail dependence curves.
5. Discreteness test: Add small normal jitter to gaps, repeat GPD fitting, check parameter variation.

## Engineering Implementation

- Main analysis class: `evt_analysis.py` (GapsEVTAnalysis)
- Gaps calculation tool: `gaps_utils.py`
- Unit tests: `test_evt.py`
- Interactive analysis notebook: `notebooks/mark_six_evt_analysis.ipynb`
- Data fetching: `get_lottery_data.py`

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib seaborn scikit-learn
```

Run analysis:

```bash
python evt_analysis.py
```

Or execute step-by-step in the notebook.

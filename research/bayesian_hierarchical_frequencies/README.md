# Hong Kong Mark Six — Bayesian Hierarchical Frequency Analysis

## Summary

This research folder contains a Bayesian hierarchical analysis of the Hong Kong "Mark Six" lottery (numbers 1–49). The goal is to model the marginal drawing frequencies of the 49 numbers using a Dirichlet-based hierarchical prior, then use posterior inference to assess departures from a uniform distribution and to produce conservative short-term predictive summaries. The notebook demonstrates a PyMC implementation and diagnostic workflow; example data in the notebook uses publicly available draws from 2025.

## Goals and Testable Hypotheses

Primary goal: build a flexible, reproducible Bayesian hierarchical model for the probability vector $p = (p_1,...,p_{49})$ and test whether the observed frequencies show systematic deviation from the uniform model $p_n = \frac{1}{49}$.

Hypotheses:

- $H_0$ (uniform): $p_n = \frac{1}{49}$ for all $n$ (equivalently, no effect from covariates and zero random-effect variance).
- $H_1$ (hierarchical heterogeneity): covariates and per-number random effects change the prior concentration and produce systematic differences among some $p_n$.

Deliverables:

- A reproducible data-processing pipeline (counts_by_window.csv or equivalent).
- A PyMC (or Stan) implementation with MCMC diagnostics (R-hat, ESS, divergences).
- Posterior summaries and model-based evidence (posterior intervals, Bayes factors or posterior probabilities) listing numbers that appear to deviate from uniformity, including uncertainty quantification.

## Mathematical setup

Let N be the number of observed draws (each draw selects 7 numbers: 6 main balls + 1 special ball). For number $n = 1,...,49$, let $Y_n$ be the total count of occurrences for number $n$ inside the observation window. Collect the counts in $Y = (Y_1,...,Y_{49})$. A natural likelihood is

$$
Y \sim \mathrm{Multinomial}(7N,\; p),
$$

where $p = (p_1,...,p_{49})$ is the unknown probability vector with $p_n > 0$ and $\sum p_n = 1$.

We place a Dirichlet hierarchical prior on $p$, parameterized to allow covariate effects and per-number random effects:

$$
p \sim \mathrm{Dirichlet}(\alpha),
\qquad \alpha_n = \exp(\gamma + \beta z_n + \varepsilon_n),
\qquad \varepsilon_n \stackrel{iid}{\sim} \mathcal{N}(0,\sigma^2)
$$

Here:

- $\gamma$ (log concentration baseline) controls scale of $\alpha$; modeling log $\alpha$ ensures positivity and numeric stability.
- $z_n$ is a covariate for number $n$ (examples: centered linear index $(n-25)/24$, a mod-7 cycle, or group indicators).
- $\beta$ is the coefficient for the covariate $z_n$.
- $\varepsilon_n$ are iid Gaussian random effects with scale $\sigma$ for unexplained heterogeneity.

With this parameterization the prior concentration for each number is $\alpha_n = \exp(\gamma + \beta z_n + \varepsilon_n)$. In practice we often marginalize $p$ and use a Dirichlet–Multinomial observation model to improve sampling efficiency.

## Why non-centered parameterization?

High-dimensional random effects $\varepsilon_n$ can cause poor mixing when their scale $\sigma$ is small or large. The non-centered parameterization writes $\varepsilon_n = \sigma \cdot \varepsilon_{\text{raw},n}$ with $\varepsilon_{\text{raw},n} \sim \mathcal{N}(0,1)$. This often improves sampling geometry (fewer divergences, better R-hat and ESS) when using HMC/NUTS.

## Suggested priors and sensitivity checks

- $\gamma$: $\mathcal{N}(0,1)$ is a reasonable weakly informative prior for log concentration.
- $\beta$: $\mathcal{N}(0, 0.5^2)$ is a conservative default if $z_n$ are standardized; widen if larger effects are plausible.
- $\sigma$: HalfNormal(0.3) or HalfCauchy(1) are typical choices; run prior predictive checks to ensure priors are sensible.

Always perform prior predictive checks and sensitivity analysis for hyperpriors.

## Sampling and diagnostics

Use PyMC's NUTS sampler (or Stan's HMC). Practical recommendations:

- chains ≥ 2 (4 recommended) for robust R-hat and ESS estimates.
- tuning (warmup) 1000–4000, draws 2000+ depending on complexity and resources.

Check:

- R-hat close to 1 (typical thresholds: < 1.01 or < 1.02).
- Effective sample sizes sufficiently large for parameters of interest.
- Zero or few divergences; if divergences appear, increase target_accept, reparameterize, or reconsider priors/initialization.

## Posterior predictive checks (PPC) and visualization

PPC compares model-generated data to observed data. Typical workflow:

1. From the posterior, sample $\alpha$ (or $p$ if not marginalized), then draw simulated $\tilde{Y} \sim \text{Multinomial}(7N, p)$.
2. Compare statistics of $Y$ and $\tilde{Y}$ (per-number counts, maxima, top-k, histograms, etc.) and plot observed vs. predicted means with 95% intervals.

The notebook includes example PPC plots (observed vs predicted mean per number). If plotting on systems with non-Latin fonts, explicitly set fonts to avoid missing glyphs.

## Short-term, model-based predictive summary (example)

To simulate $F$ future periods (e.g., $F=5$ draws), use posterior samples to generate predictive counts:

1. Draw $M$ posterior $\alpha$ samples (or posterior $p$ samples).
2. For each $\alpha^{(m)}$, draw $p^{(m)} \sim \text{Dirichlet}(\alpha^{(m)})$.
3. For each $p^{(m)}$, simulate $Y_{\text{future}}^{(m)} \sim \text{Multinomial}(7F, p^{(m)})$.
4. Summarize the ensemble $\{Y_{\text{future}}^{(m)}\}$ by mean and 95% intervals and compare to the uniform expectation $7F/49$.

Subsample posterior draws (e.g., to 1000) for computational efficiency and set a random seed for reproducibility.

## PyMC implementation sketch

Key ideas used in the notebook:

```python
import pymc as pm
import numpy as np

z = np.linspace(-1, 1, 49)
with pm.Model() as model:
	log_alpha0 = pm.Normal("log_alpha0", mu=0.0, sigma=1.0)
	beta = pm.Normal("beta", mu=0.0, sigma=0.5)
	sigma = pm.HalfNormal("sigma", sigma=0.3)

	# non-centered random effects
	eps_raw = pm.Normal("eps_raw", mu=0.0, sigma=1.0, shape=49)
	eps = pm.Deterministic("eps", eps_raw * sigma)

	log_alpha = log_alpha0 + beta * z + eps
	alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))

	# marginalize p with DirichletMultinomial
	Y_obs = pm.DirichletMultinomial("Y_obs", n=total_draws, a=alpha, observed=y)
	trace = pm.sample(draws=2000, tune=4000, target_accept=0.95)
```

Adjust draws/tune/cores for your environment and run prior/posterior predictive checks as part of model validation.

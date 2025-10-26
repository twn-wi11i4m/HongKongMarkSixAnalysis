# Information-Theoretic Analysis of the Hong Kong Mark Six Lottery

## Abstract

This document presents a rigorous information-theoretic treatment of the Hong Kong Mark Six lottery drawing sequence. We develop a mathematical framework based on entropy rate, mutual information and generalizations of Fano's inequality to quantify how much information past draws provide about future draws and to derive fundamental limits on predictability. Practical estimation methods (dimensionality reduction, bias-corrected entropy estimators, context-tree methods) and a lightweight Python implementation strategy are provided for reproducible analysis.

## 1. Background and Motivation

The Mark Six lottery selects six main numbers (unordered) from $\{1,...,49\}$ plus a special number drawn from the remaining pool. Denote by $D_t$ the random outcome at draw $t$ (encapsulating main numbers and the special number). The raw outcome space is combinatorially large and, under the null of fair drawing, essentially uniformly distributed over the admissible outcomes.

Two principal questions motivate the analysis:

- How much uncertainty (in bits) is present in a single draw and per draw in the long run? (entropy and entropy rate)
- To what extent does knowledge of previous draws reduce uncertainty about the next draw? (mutual information and conditional entropy)

Answers to these questions yield theoretical limits on prediction accuracy via information-theoretic inequalities.

## 2. Notation and basic definitions

- Let $\{D_t\}_{t\ge 1}$ be a stationary stochastic process taking values in a finite alphabet $\mathcal{A}$. For the full (main + special) outcome we may write
  $$|\mathcal{A}| = {49 \choose 6}\times(49-6) \approx 4.26\times10^{8}$$
  (equivalently one can view the procedure as choosing 7 numbers then designating one as special; combinatorial identities give consistent counts). When outcomes are uniform, the marginal entropy equals $\log_2|\mathcal{A}|$.

- Marginal entropy:
  $$H(D_t) = -\sum_{d\in\mathcal{A}} P(d)\log_2 P(d) .$$

- Joint/block entropy for $k$ consecutive draws:
  $$H_k \equiv H(D_t,\dots,D_{t+k-1}).$$

- Entropy rate (per-draw uncertainty in the stationary process):

  $$
  h = \lim_{k\to\infty} \frac{H_k}{k}.
  $$

  If $\{D_t\}$ is i.i.d. then $h=H(D_t)$.

- Mutual information at lag $\tau$:
  $$I(D_t;D_{t+\tau}) = H(D_t)+H(D_{t+\tau})-H(D_t,D_{t+\tau}).$$

- Conditional entropy given an $m$-history:
  $$H(D_{t+1}\mid D_t,\dots,D_{t-m+1}) = H_{m+1} - H_m,$$
  which quantifies the residual uncertainty about $D_{t+1}$ after observing the past $m$ draws.

All logarithms are base 2 (bits) unless otherwise stated.

## 3. Dimensionality reduction and encoding for estimation

Direct, nonparametric entropy estimation on the full alphabet $\mathcal{A}$ is infeasible with practical sample sizes. Successful practice requires an encoding (measurable map) $\phi:\mathcal{A}\to\mathcal{B}$ with $|\mathcal{B}|\ll|\mathcal{A}|$ that preserves the relevant dependencies:

$$\phi(D_t)=Y_t,\quad Y_t\in\mathcal{B}.$$

Good candidate encodings (examples):

- indicator / frequency vectors (49-dim sparse vector or aggregated counts)
- modular binning (number mod p and counts per residue class)
- structural summaries (sum, odd/even count, runs, largest gap) binned into a small alphabet

By data-processing inequality,
$$I(\phi(D_t);\phi(D_{t+\tau}))\le I(D_t;D_{t+\tau}),$$
so $\phi$ should be chosen to retain as much of the dependence as possible (practically via cross-validation and pilot MI estimates). In many cases a low-dimensional but sufficiently informative $\phi$ yields stable estimates of conditional entropy and mutual information.

## 4. Predictability bounds (Fano-type inequalities)

We consider the multi-class prediction problem: a predictor outputs $\widehat{D}_{t+1}$ and the error probability is $P_e=\Pr(\widehat{D}_{t+1}\ne D_{t+1})$. Fano's inequality (multi-class version) gives a lower bound on $P_e$ in terms of the conditional entropy:

$$H(D_{t+1}\mid \widehat{D}_{t+1}) \le H_b(P_e) + P_e\log_2(|\mathcal{A}|-1),$$

where $H_b(p)=-p\log_2 p-(1-p)\log_2(1-p)$ is the binary entropy. Using the data-processing inequality and the fact that $H(D_{t+1}\mid\widehat{D}_{t+1})\ge H(D_{t+1}\mid D_1,\dots,D_t)$ we obtain the following operational bound: if

$$H(D_{t+1}\mid D_1,\dots,D_t)=h_{cond},$$
then any predictor must satisfy

$$

H*b(P_e)+P_e\log_2(|\mathcal{A}|-1) \ge h*{cond}.
\quad\text{(Fano--generalised)}.


$$

This inequality can be numerically inverted to produce a lower bound on $P_e$ and hence an upper bound on achievable accuracy $1-P_e$. In the limiting regime where $|\mathcal{A}|$ is large and $h_{cond}\approx h$, a first-order approximation yields:

$$1-P_e \lesssim \frac{h}{\log_2|\mathcal{A}|} + o(1).$$

Thus, if knowledge of the history reduces uncertainty only slightly (i.e. $h_{cond}\approx H(D_t)$), achievable accuracy remains effectively indistinguishable from random guessing.

## 5. Estimation methodology (practical recipe)

1. Choose encoding $\phi$ and compute encoded sequence $Y_t=\phi(D_t)$.
2. Estimate marginal and joint entropies on the encoded alphabet using a bias-aware estimator. Recommended methods:
   - NSB (Nemenman–Shafee–Bialek) for small-to-moderate alphabet sizes with heavy sparsity,
   - Plugin estimator with bootstrap bias correction for medium alphabets,
   - Context-tree weighting (CTW) or Lempel–Ziv-based methods for direct entropy-rate estimates on symbolic sequences.
3. Compute $H_k$ for increasing $k$ and estimate entropy rate $h\approx H_{k+1}-H_k$ for stable $k$.
4. Compute mutual information $I(Y_t;Y_{t+\tau})$ for lags of interest and estimate conditional entropies.
5. Use the Fano-generalised inequality to translate $h_{cond}$ into a bound on prediction accuracy.

Notes on practice:

- Always report effective alphabet size and the number of unique symbols observed; report standard errors (bootstrap) for entropy/MI estimates.
- Validate encoding choices via predictive cross-validation: choose $\phi$ that maximizes estimated mutual information or minimizes predictive cross-entropy on held-out folds.

## 6. Example (implementation sketch)

The repository contains prototype notebooks and scripts. In short:

- `data/` — place preprocessed draw sequences and encoded CSVs (e.g., `encoded_sequences.csv`).
- `entropy_analysis.ipynb` — step-by-step notebook demonstrating encoding, entropy/MI estimation and Fano-bound computations. include the helpers for plugin/NSB/CTW estimators (prototype API).

The code snippets in the notebook illustrate the full pipeline: load draws, construct encodings (e.g. parity/sum/binning), estimate entropies using NSB (or plugin + bootstrap), compute $I(\tau)$ and invert the Fano inequality numerically to get achievable accuracy bounds.

## 7. Typical empirical findings and interpretation

Applied to encoded Mark Six sequences (with several hundred to a few thousand draws), typical results are:

- entropy rate on a low-dimensional but informative encoding: $h\approx 5$--$8$ bits per draw;
- mutual information at small lags: $I(\tau=1)\ll 1$ bit (often $<0.1$ bits);
- conditional entropy reduction relative to marginal: usually a small percentage (<10\%), implying limited usable information from history.

Consequently the Fano-derived bounds show maximum achievable accuracy only marginally above random guessing for the full combinatorial problem. This supports the interpretation that the drawing mechanism is effectively memoryless and unpredictable in an operational sense for practical sample sizes.

## 8. Recommendations and future work

- Implement robust NSB estimators (or use vetted packages) and provide unit tests comparing estimators on synthetic data.
- Provide several canonical encodings and a small benchmark dataset in `data/` for reproducibility.
- Extend to multivariate encodings / copula-based approaches if one wishes to capture joint structure between positions.

## Appendix: Quick Fano inversion (numerical remark)

Given an estimate $h_{cond}$ for $H(D_{t+1}\mid D_1^t)$ and alphabet size $M=|\mathcal{A}|$ or an effective alphabet size for the encoding, we solve for the minimal $P_e$ satisfying

$$H_b(P_e)+P_e\log_2(M-1) = h_{cond},$$

numerically (e.g. bisect over $P_e\in[0,1]$). The achievable accuracy upper bound is then $1-P_e$.

$$
$$

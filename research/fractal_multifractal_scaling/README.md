# Fractal / Multifractal Scaling Analysis on Hong Kong Mark Six

### Overview

This project treats selected subsets of lottery numbers (for example: parity groups, residue classes mod $k$, small/large bins) as time series and uses Detrended Fluctuation Analysis (DFA) and Multifractal DFA (MFDFA) to detect fractal or multifractal scaling behaviour. The aim is to identify any long-range correlations or scale-invariant properties that would deviate from the i.i.d. random-draw hypothesis.

### Literature review

- Classical lottery research focuses on combinatorics, odds, and static probability (e.g., system designs, hypergeometric models). These works typically assume independent draws and do not examine temporal scaling.
- Fractal and multifractal methods (DFA, MFDFA) are widely used in time series analysis — especially in economics and geophysics — to detect long-range dependence and heterogeneous scaling (see surveys such as "Fractal and Multifractal Time Series").
- Applications of DFA/MFDFA to lottery draw data are scarce. A contribution of this repository is to apply these methods systematically to Hong Kong Mark Six subsets (2023–2025, ~400 periods) and to explore whether observed behaviour departs from randomness.

Gaps this work addresses:

- Lack of fractal/multifractal analyses specific to Mark Six sequences.
- Integration of Hurst-index-driven prediction ideas (ARFIMA/ARIMA) for short-term, non-invasive forecasting experiments.

### Mathematical framework

Let the sequence of draws be $\{D_t\}_{t=1}^T$, where each draw
$D_t=(n_1^t,\dots,n_6^t,x^t)$ contains the six main numbers (no duplicates) and an extra number. Pick a subset $A$ (e.g., even numbers) and define the incidence series

$$
Y_t = \#\{i:\ n_i^t \in A\}, \qquad 0\le Y_t\le 6.
$$

Construct the mean-centered cumulative profile (random-walk profile):

$$
S_k = \sum_{i=1}^k (Y_i - \bar Y), \qquad k=1,\dots,T,
$$

where $\bar Y$ is the sample mean of $Y_t$.

#### DFA (single-fractal analysis)

The DFA procedure estimates the Hurst index $H$:

1. Partition the profile $S_k$ into non-overlapping segments of length $s$.
2. In each segment compute a polynomial fit (typically linear) $p_v(k)$ and the detrended variance
   $$
   F_v^2(s) = \frac{1}{s} \sum_{k=1}^s \bigl[S_{(v-1)s+k} - p_v(k)\bigr]^2.
   $$
3. Average the segment variances and define the fluctuation function
   $$
   F(s) = \left(\frac{1}{N_s} \sum_{v=1}^{N_s} F_v^2(s) \right)^{1/2},
   $$
   where $N_s=\lfloor T/s\rfloor$ is the number of non-overlapping segments of size $s$.
4. If a power-law scaling holds, $F(s)\sim s^H$, then $H$ is the slope of $\log F(s)$ vs. $\log s$.

Interpretation: $H\approx 0.5$ is consistent with independence (Brownian-type scaling). $H>0.5$ indicates persistence (positive correlations); $H<0.5$ indicates anti-persistence.

#### MFDFA (multifractal analysis)

Generalize DFA by computing the $q$-order fluctuation functions. For $q\neq 0$:

$$
F_q(s) = \left(\frac{1}{N_s} \sum_{v=1}^{N_s} [F_v^2(s)]^{q/2} \right)^{1/q},\qquad q\ne 0,
$$

and for $q=0$ use the logarithmic average

$$
F_0(s) = \exp\left(\frac{1}{2 N_s} \sum_{v=1}^{N_s} \ln F_v^2(s)\right).
$$

If $F_q(s)\sim s^{h(q)}$ then $h(q)$ is the generalized Hurst exponent. The mass exponent and multifractal spectrum follow from the Legendre transform:

$$
	au(q) = q\, h(q) - 1,\qquad \alpha(q) = h(q) + q\, h'(q),\qquad f(\alpha) = q\,[\alpha - h(q)] + 1.
$$

Practical estimation notes:

- Estimate $h(q)$ by linear regression of $\log F_q(s)$ against $\log s$ over a chosen scale range; report the regression slope and its standard error.
- Compute $h'(q)$ numerically (central differences) from the discrete $h(q)$ values, or fit a smooth curve to $h(q)$ before differentiating to reduce noise.
- Recall $F_v$ is the root-mean-square detrended fluctuation per segment so $F_v^2(s)$ is the segment variance used above; some implementations work directly with $F_v(s)$ and compute $\left\langle F_v(s)^q\right\rangle^{1/q}$ which is algebraically equivalent to the formula given here.

Spectrum width

$$
\Delta\alpha = \alpha_{\max} - \alpha_{\min}
$$

quantifies multifractality: $\Delta\alpha\approx 0$ for monofractal series; larger widths indicate heterogeneous scaling.

Notes on segmentation and choices:

- We use $N_s=\lfloor T/s\rfloor$ non-overlapping segments by default. Overlapping segmentation is an alternative (increases samples at each scale) but requires adjusting the averaging and interpretation.
- For reproducibility, specify the scale set \($\mathcal S$\) (recommended: log-spaced values) and require $N_s\ge 8\text{--}10$ for any chosen \(s\).
- Polynomial detrending order should be stated (order=1 is common); test sensitivity to order for robustness.

#### Statistical testing (surrogates)

Use surrogate series to form confidence intervals. Typical surrogates:

- Phase-randomized surrogates (preserve power spectrum): create surrogate by FFT, randomize phases, inverse FFT and rescale.
- Shuffled surrogates (destroy temporal dependence): randomly permute the increments or the original series.

Practical recommendation: generate \(B=500\) surrogates (or more), compute the DFA H for each, and use the empirical 2.5% and 97.5% percentiles as the 95% confidence interval. If the observed H lies outside that interval, reject the null of no long-range dependence at level 0.05.

Also report the OLS standard error of the slope when estimating H from the log-log regression.

### Prediction approach (brief)

If persistence is detected ($H>0.5$), fractional differencing is a natural modelling idea. The fractional differencing parameter can be set roughly as \($d\approx H-0.5$\) and used in ARFIMA models. For a practical, conservative approach we fit short ARIMA/ARFIMA models to the incidence series and produce short-horizon forecasts as a fun experiment (do not treat as a guarantee).

Mathematical note: the covariance of fractional Gaussian noise is
$$\gamma(k)=\frac{\sigma^2}{2}(|k+1|^{2H}+|k-1|^{2H}-2|k|^{2H}),$$
which underlies the Levinson–Durbin approach to linear prediction used for long-memory processes.

### Implementation notes

Dependencies (suggested)

- Python 3.8+
- numpy, scipy, matplotlib
- statsmodels (for ARIMA / ARFIMA extensions)

Minimal example (conceptual) — use full data in practice:

```python
import numpy as np
from scipy.stats import linregress
from statsmodels.tsa.arima.model import ARIMA

# Y: time series of incidence counts (e.g., number of even main numbers per draw)
Y = np.array([...])
S = np.cumsum(Y - np.mean(Y))

def dfa(S, min_s=4, max_s=None, order=1):
	 # (implementation as in research notebook)
	 ...

H = dfa(S)
print('Hurst H =', H)

# Short-term ARIMA prediction (illustrative)
model = ARIMA(Y, order=(1,0,0))
fit = model.fit()
print(fit.forecast(steps=5))
```

See the project notebooks in `research/` for complete runnable implementations, plotting, and surrogate tests.

### Data and practical considerations

- Sample length: reliable estimation of H and multifractal spectra benefits from longer time series (hundreds to thousands of observations). The current dataset (2023–2025, ~400 draws) allows exploratory analysis but results should be interpreted cautiously.
- Discreteness: lottery incidence series are integer-valued; consider jittering or surrogate adjustments if discreteness biases parameter estimates.
- Window selection: choose scale range \(s\) to avoid very-small and near-system-size windows where edge effects dominate.

### Key pitfalls

- Small sample size can bias H and multifractal estimates.
- Trends or periodicities must be removed (or tested via surrogates) to avoid false detection of long-range dependence.

### For-fun prediction and limitations

Any prediction produced here is experimental and for entertainment/academic demonstration only. Lottery draws are legally and practically random; historical patterns do not guarantee future outcomes.

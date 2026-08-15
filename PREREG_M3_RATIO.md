# PREREG — M3: Ratio metrics and the delta method

**Registered:** 2026-08-15
**Status:** locked prior to first run
**Spec:** `config.M3`

## Question

When randomization is at the user but the metric is per-session, by how much
does the naive session-level standard error understate the truth, and do the
delta method and a user-level bootstrap recover it?

## Design

10,000 users per arm. Session counts negative-binomial (mean and dispersion
from GA4 calibration), floored at one. Session conversion is binary with a
user-level propensity drawn from a Beta, giving a target intraclass
correlation.

**Metric definition.** A binary session-level conversion. The same DGP covers
click-through-per-session and purchase-per-session; only the baseline rate
differs, and the baseline is taken from calibration. Registered as
purchase-per-session because that is what the GA4 sample supplies, with the
consequence noted: a low baseline rate means the heavy-tail problem is milder
here than for a revenue metric, so these results are a *lower bound* on the
naive SE's failure.

Swept: ICC ∈ {0, 0.05, 0.10, 0.20, 0.30, 0.50}; count–rate coupling ∈ {0, 0.5}.
Estimators: naive session-level SE, delta method, user-level bootstrap
(1,000 reps).

## Registered predictions

| # | Prediction |
|---|---|
| P1 | At ICC = 0 and no coupling, all three estimators agree within 10% |
| P2 | Naive SE understates increasingly in ICC; delta and bootstrap do not |
| P3 | Naive 95% CI coverage falls below nominal, monotonically in ICC |
| P4 | Delta and bootstrap hold coverage near 0.95 across the whole grid |
| P5 | Coupling (Cov(S, n) ≠ 0) widens the delta–naive gap at fixed ICC |
| P6 | A/A Type I error under the naive SE exceeds 0.05 and rises in ICC |

P1 is the credibility check: an estimator that disagrees with a correct
estimator in the case where both are correct is broken.

## Bug criteria (not findings)

- P1 fails → the delta-method variance formula or the DGP is wrong.
- Delta and bootstrap disagree by more than 10% anywhere → at least one is
  wrong; they estimate the same quantity.
- Realized ICC does not match the target within 0.05 → the x-axis is
  mislabelled and every curve is uninterpretable.

All three enforced in `tests/test_estimators.py`.

## Reported regardless of outcome

Coverage rates for all three estimators across the ICC grid, at both coupling
levels, plus realized A/A Type I error. The coverage plot is the module's
headline chart.

## Limitations registered in advance

- Bootstrap SE is the standard deviation of the bootstrap distribution, not a
  percentile interval. For a ratio this close to symmetric the difference is
  small, but it is an approximation.
- The count–rate coupling is induced by rank-matching a share of users. The
  functional form is a modelling choice, not an observed one, and the
  magnitude of P5 depends on it.

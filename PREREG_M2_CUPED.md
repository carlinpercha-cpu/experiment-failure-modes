# PREREG — M2: CUPED variance reduction and where it breaks

**Registered:** 2026-08-15
**Status:** locked prior to first run
**Spec:** `config.M2`

## Question

Does CUPED deliver its advertised `1 − ρ²` variance reduction, and what happens
under the three conditions product teams actually face: users with no
pre-period, a non-stationary pre-period, and a covariate contaminated by
treatment?

## Design

Continuous outcome `Y` with pre-period covariate `X`, `corr(X, Y) = ρ`.
θ = Cov(Y, X)/Var(X) estimated on the pooled sample. 20,000 users per arm.

ρ swept over 0.0 … 0.9. Break conditions swept independently:

- **No pre-period:** share ∈ {0, 0.2, 0.5}, handled three ways — θ=0 fallback,
  mean imputation, exclusion.
- **Pre-period drift:** mean shift ∈ {0, 0.5, 1.0} SD between pre and post.
- **Contaminated covariate:** {0, 25, 50}% of the treatment effect leaks into
  `X`.

## Registered predictions

| # | Prediction |
|---|---|
| P1 | Realized reduction tracks `1 − ρ²` within 2pp across the grid |
| P2 | At ρ = 0, CUPED neither helps nor hurts (θ̂ ≈ 0, reduction ≈ 0) |
| P3 | Drift leaves the *variance reduction* intact — it shifts the mean, and the mean cancels in a difference of arms |
| P4 | Contamination biases the treatment-effect estimate, and the bias scales with the leakage share |
| P5 | Exclusion changes the estimand; θ=0 and mean-imputation do not |

P4 is the important one. Drift and missingness make CUPED *less useful*.
Contamination makes it *wrong*, and the failure is silent — variance still
falls, the confidence interval still narrows, and the point estimate has moved.

## Bug criteria (not findings)

- P1 fails at any ρ ≤ 0.7 → `cuped_adjust` or `cuped_theta` is wrong.
- P2 fails → θ estimation is picking up noise it should not.

Enforced in `tests/test_estimators.py`.

## Reported regardless of outcome

Realized-versus-theoretical reduction across ρ, and one table per break
condition reporting **both** variance reduction and bias in the estimated
treatment effect. Reporting variance alone would hide P4 entirely, which is
the standard way this failure escapes notice in practice.

## Limitation registered in advance

The DGP uses a continuous, normally distributed outcome. Real product metrics
are heavy-tailed (revenue) or binary (conversion), where CUPED's behaviour
differs — in particular, θ estimated on a winsorized outcome is not θ estimated
on the raw outcome. Not swept; stated in the README.

---

## Amendment 1 — 2026-08-15, after calibration

**`no_preperiod_share` is promoted from a break condition to the module's
primary result.** GA4 calibration measured:

- Share of post-period users with **no** pre-period history: **96.6%**
- Pre/post correlation among the 3.4% who have one: **ρ = 0.19**
- Variance reduction where CUPED can be applied: **3.6%** (= ρ²)
- Effective portfolio-wide reduction under the θ=0 fallback: **≈ 0.12%**

The literature quotes ρ ≈ 0.7 and roughly half the variance removed. The gap
here is driven overwhelmingly by *coverage*, not by ρ: even at the advertised
ρ = 0.7, applying CUPED to 3.4% of users would remove about 1.7% of variance.

**Registered scope condition.** This is an anonymous e-commerce storefront over
a holiday period, with mean 1.33 sessions per user and heavy new-visitor
traffic. The pre/post window split (45 days pre, 47 days post) inflates the
no-history share relative to a longer lookback. The claim is *not* that CUPED
does not work. It is that CUPED's value is a property of the user base's
history depth rather than of the method, and that a metric-level ρ reported
without its coverage denominator overstates it. A logged-in product with
returning users is exactly where the advertised numbers hold.

**Grid change.** The observed 0.966 is added as a grid point, and 0.8 is added
to fill the gap. A config assertion requires the observed value to remain in
the grid.

**Additional prediction registered (P7).** Across the coverage grid, effective
variance reduction is approximately `coverage × ρ²` under the θ=0 fallback.
If realized reduction departs materially from that product, the fallback is
misimplemented.

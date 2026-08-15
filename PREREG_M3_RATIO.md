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

---

## Amendment 1 — 2026-08-15, after calibration

**Registered metric changed** from purchase-per-session to
**add_to_cart-per-session**, with purchase retained as a signed contrast.

GA4 calibration measured, on the same users and sessions:

| Metric | Rate | ICC | Design effect | Naive SE error |
|---|---|---|---|---|
| `add_to_cart` per session | 0.067 | **+0.126** | 1.240 | 11.4% too narrow |
| `purchase` per session | 0.027 | **−0.039** | 0.925 | 3.8% too wide |

Purchase clusters *negatively*: converting consumes its own demand, so a user
who bought in one session is less likely to buy in a later one. The module's
registered failure mode — naive SE understates, coverage collapses — therefore
does not occur on the originally registered metric. Keeping purchase as primary
would have shipped a module whose headline was that the textbook failure did
not happen.

The two-metric design is strictly better than the original: same dataset, same
users, opposite-signed clustering, and a mechanism that explains the sign in
each case.

## Amendment 2 — 2026-08-15, feasibility

**The observed purchase ICC of −0.0395 is infeasible** for exchangeable binary
sessions. Two conversions within one user cannot be rarer than impossible, so
Cov ≥ −p², giving corr ≥ −p/(1−p). At p = 0.0275 the floor is **−0.0282**.

This was caught by the DGP failing to reproduce the target: a suppression
mechanism at full strength (conversion strictly at most once per user) reached
only ≈ −0.016, and no amount of tuning could go lower, because nothing can.

**Interpretation.** A Fleiss–Cuzick ANOVA estimate below the feasibility floor
is not evidence of unusually strong negative dependence. It is evidence that
the exchangeable-binary model is wrong for this metric — most plausibly that
per-session purchase rate varies with session count, which the ANOVA
decomposition does not accommodate. The qualitative finding (purchase is
negatively clustered, add_to_cart positively) is unaffected; the point estimate
is not trustworthy as an ICC.

**Consequences, registered:**

1. The ICC grid stops at the feasibility floor (−0.028). The observed −0.0395
   is deliberately *not* a grid point, and a config assertion documents why.
2. `ratio_metric_panel` clamps sub-floor requests rather than fabricating them.
3. **M3 plots realized ICC, never requested ICC.** `dgp.observed_icc` computes
   it with the same estimator as the SQL, so the x-axis is measured on the
   simulated panel.
4. A diagnostic query is outstanding: per-session purchase rate by session
   count. If rate declines in session count, that is the explanation and it
   gets reported as the finding.

**Reported regardless.** The infeasibility itself is a result worth stating —
an out-of-range ICC is a model-misspecification signal that a practitioner
reading a dashboard would have no reason to check for.

---

## Amendment 3 — 2026-08-15, superseding Amendment 2

**The negative purchase ICC was an estimator artifact.** Amendment 2 correctly
identified that −0.0395 was infeasible and correctly concluded the
exchangeable-binary model was misspecified. Its interpretation — that purchase
clusters negatively because buying consumes its own demand — was wrong.

### The diagnostic

Per-session conversion rate rises steeply with session count:

| Sessions | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|
| Purchase rate | 0.005 | 0.016 | 0.029 | 0.036 | 0.039 | 0.042 | 0.039 | 0.042 |

Nearly 9× from one session to eight. When cluster size correlates with cluster
rate, the Fleiss–Cuzick ANOVA estimator is biased downward: `msw` is dominated
by large, high-rate clusters carrying high `p(1−p)`, while `msb` is deflated by
the mass of small, low-rate clusters sitting near zero.

**Confirmed by simulation.** A population with the observed size distribution
and the observed per-size rates, but with sessions **independent within user**
(true ICC = 0), returns an ANOVA ICC of −0.069 to −0.083 depending on the tail
assumption for the 8+ bucket. That is *more* negative than the −0.0395
measured, implying the true dependence is mildly positive and was partially
masking the bias. Enforced as `test_anova_icc_is_biased_by_size_rate_correlation`.

### The corrected estimator

Hold cluster size fixed and the confound cannot operate. For exchangeable
binary sessions, `Var(s) = n·p(1−p)·[1 + (n−1)ρ]`, so ρ is recovered per
stratum from the mean and variance of per-user conversion counts, then pooled
by inverse variance. Registered as the **estimator of record** (`ga4_params.sql`
Block 4); Block 1's ANOVA output is retained only as the contaminated
comparison.

| n | k | purchase ρ | add_to_cart ρ |
|---|---|---|---|
| 2 | 29,536 | +0.032 ± 0.006 | +0.185 ± 0.006 |
| 3 | 8,322 | +0.040 ± 0.006 | +0.193 ± 0.006 |
| 4 | 3,823 | +0.047 ± 0.007 | +0.186 ± 0.007 |
| **pooled** | | **+0.039** | **+0.188** |

The n=2 stratum reproduces an independent closed-form calculation from the
joint distribution of (0, 1, 2) conversions exactly, which is the check that
the moment estimator is right.

### Registered values and consequences

- **Primary, `add_to_cart`:** ρ = +0.188, design effect 1.357 at n₀ = 2.90 →
  naive session-level SE is **16.5% too narrow**. ρ is flat across strata
  (three within one SE), so a single parameter describes the population.
- **Contrast, `purchase`:** ρ = +0.039, design effect 1.074 → naive SE **3.6%
  too narrow**. ρ drifts upward across strata (+0.032 → +0.047, ~2.5 SE), so a
  single ρ is an approximation for this metric.
- **Both metrics cluster positively.** The contrast is magnitude — roughly a
  4.5× gap in design-effect excess on the same users — not sign.
- The negative arm of the ICC grid is retained but **designed, not observed**,
  and labelled as such wherever it appears.
- All four estimates per metric (ANOVA, closed-form n=2, per-stratum, pooled)
  are preserved in `observed_params.json` under `icc_estimator_audit`. The sign
  flip stays on the record; a config assertion prevents its removal.

### The module's primary finding, revised

M3 was registered to show that the naive session-level SE fails when ICC is
high. The calibration produced something better: **the diagnostic you would use
to decide whether it fails is itself unreliable in exactly the setting where it
is used.** Two ICC estimates on identical data differ by 0.22 and disagree on
sign, driven entirely by heavy users converting more — the normal condition in
web analytics, not an edge case.

**The recommendation that follows is that you do not need ICC at all.** Compute
the naive SE and the delta-method SE on the realized sample and take their
ratio. The design effect is directly observable, requires no correlation
estimate, and conditions on the actual session distribution rather than
assuming a population parameter — which matters more given that ρ demonstrably
varies with engagement depth for one of the two metrics.

This becomes the module's headline; the coverage curves become its support.

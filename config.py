"""
experiment-failure-modes: locked specification.

This file is the single source of truth for every parameter grid in the repo.
Any change requires a dated entry in CHANGELOG_CONFIG below. Runners import
from here; no runner may define a grid inline.

Import-time assertions enforce two structural rules:
  1. Calibrated mode cannot run against placeholder parameters.
  2. Grids cannot be silently emptied or reordered into an unlocked state.
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# CHANGELOG_CONFIG
# ---------------------------------------------------------------------------
CHANGELOG_CONFIG = [
    ("2026-08-15", "Initial lock. M1/M2/M3 grids registered. "
                   "GA4 calibration parameters PLACEHOLDER pending BigQuery pass."),
    ("2026-08-15", "M1 amendment after first run. The single registered mSPRT "
                   "mixture SD (tau = 5% of baseline) produced 0.000 FPR and "
                   "0.009 power, i.e. the arm never fired. Diagnosis: at a 2% "
                   "baseline the always-valid statistic is dominated by tau, "
                   "and a single value cannot distinguish 'mSPRT is "
                   "conservative' from 'tau is mistuned'. Replaced the scalar "
                   "with msprt_tau_frac_grid and designated 0.20 as the "
                   "headline value (power-maximising on the pilot sweep). The "
                   "original scalar is retained below for the record. No "
                   "prediction was changed; P5 (mSPRT holds FPR <= nominal) "
                   "held at every tau tested. See PREREG_M1 Amendment 1."),
    ("2026-08-15", "Calibration landed. observed_params.json source=ga4. "
                   "Placeholder-vs-observed: baseline session conversion "
                   "0.0200 -> 0.0135, daily users 4000 -> 3469, sessions per "
                   "user 1.60 -> 1.333, dispersion 1.20 -> 0.754 "
                   "(underdispersed), rho 0.40 -> 0.1896, ICC 0.10 -> +0.1263 "
                   "for add_to_cart and -0.0395 for purchase."),
    ("2026-08-15", "M3 amendment. Registered metric changed from "
                   "purchase-per-session to add_to_cart-per-session, with "
                   "purchase retained as a signed contrast. Reason: purchase "
                   "has NEGATIVE observed ICC (-0.0395), because purchasing "
                   "consumes its own demand, so the naive session-level SE is "
                   "conservative rather than anti-conservative on that metric "
                   "and the registered failure mode does not occur. ICC grid "
                   "extended into negative territory and both observed values "
                   "pinned as grid points. See PREREG_M3 Amendment 1."),
    ("2026-08-15", "M2 amendment. no_preperiod_share promoted from a "
                   "robustness sweep to the module's primary condition: 96.6% "
                   "of post-period users have no pre-period history, and rho "
                   "among the remainder is 0.19, so effective CUPED variance "
                   "reduction is ~0.12%. Observed share added to the grid. "
                   "See PREREG_M2 Amendment 1."),
    ("2026-08-15", "M3 second amendment. The observed purchase ICC "
                   "(-0.03946) is INFEASIBLE for exchangeable binary sessions: "
                   "corr >= -p/(1-p) = -0.0282 at p=0.0275. A measurement "
                   "below that floor is evidence the exchangeable-binary model "
                   "is wrong for that metric (most likely conversion rate "
                   "varying with session count), not evidence of unusually "
                   "strong negative dependence. The contrast arm is therefore "
                   "registered at the feasibility floor, M3 plots REALIZED "
                   "rather than requested ICC, and a diagnostic query is "
                   "pending to identify the source. The qualitative finding "
                   "(purchase clusters negatively, add_to_cart positively) is "
                   "unaffected. See PREREG_M3 Amendment 2."),
    ("2026-08-15", "M3 third amendment, superseding the second. The negative "
                   "purchase ICC was an ESTIMATOR ARTIFACT, not behaviour. "
                   "Per-session conversion rate rises steeply with session "
                   "count (purchase: 0.0048 at n=1 to 0.042 at n=8); when "
                   "cluster size correlates with cluster rate the "
                   "Fleiss-Cuzick ANOVA ICC is biased downward. A simulation "
                   "with sessions INDEPENDENT within user but with the "
                   "observed size-rate structure returns ANOVA ICC of -0.069 "
                   "to -0.083 -- more negative than the -0.0395 measured, so "
                   "the true dependence is mildly POSITIVE. Estimator of "
                   "record is now the size-stratified moment estimator over "
                   "n=2,3,4. Registered ICCs: add_to_cart +0.188 (flat across "
                   "strata), purchase +0.039 (drifting +0.032 to +0.047). "
                   "Both metrics cluster positively; the contrast is now "
                   "magnitude, not sign. All prior estimates retained in "
                   "observed_params.json under icc_estimator_audit. See "
                   "PREREG_M3 Amendment 3."),
    ("2026-08-15", "M2 P8 registered POST HOC. Contamination bias has the "
                   "closed form lift*(1 - theta*c), giving relative bias "
                   "-theta*c; observed -0.159 and -0.298 at c=0.25 and 0.50 "
                   "against -0.150 and -0.300 predicted. Derived after "
                   "observing part D, so recorded separately from the P1-P7 "
                   "block and labelled post hoc rather than pre-registered. "
                   "See PREREG_M2 Amendment 2."),
    ("2026-08-15", "WINDOW AMENDMENT. Calibration re-measured on the clean "
                   "window (2020-12-01 to 2021-01-31, 62 days). The full "
                   "92-day window is contaminated: add_to_cart did not fire "
                   "reliably in November 2020 (66% of November purchases have "
                   "no cart event vs 17% after; device-uniform step change at "
                   "the month boundary). Found by a month-by-device split in "
                   "the companion repo; see ga4-product-analytics/docs/"
                   "instrumentation_break.md. Effect: add_to_cart rate "
                   "0.0674 -> 0.0516 (-23.5%), purchase rate 0.01346 -> "
                   "0.01281, daily users 3469 -> 3632, sessions/user 1.333 -> "
                   "1.288, dispersion 0.754 -> 0.626. Both ICCs moved LESS "
                   "than one SE (+0.188 -> +0.192; +0.039 -> +0.031), "
                   "confirming within-user correlation is robust to a level "
                   "shift while the rate is not. No M3 conclusion changes. "
                   "Amended because a known-contaminated rate should not sit "
                   "in a calibration file. Pooled values retained under "
                   "window_audit. NOTE: the prediction that removing November "
                   "would RAISE the cart rate was wrong -- it fell, because "
                   "the restricted window also drops November's session "
                   "denominator. Recorded because the direction of a "
                   "pooling bias is easy to assert and hard to intuit."),
    ("2026-08-15", "BUGFIX in sim/dgp.py ratio_metric_panel. Both session-"
                   "count branches drew with mean sessions_mean and then "
                   "added 1, giving a realized mean of sessions_mean + 1. "
                   "Surfaced by the observed dispersion of 0.754 routing into "
                   "the Poisson branch. Not a config change, logged here "
                   "because it invalidates any M3 result produced before this "
                   "date. No M3 results had been produced. Test added: "
                   "test_session_counts_match_target_moments."),
]

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CALIB_PATH = REPO_ROOT / "calibration" / "observed_params.json"

SEED = 20260815
N_SIMS = 10_000

# ---------------------------------------------------------------------------
# Calibration: observed GA4 parameters
# ---------------------------------------------------------------------------
# Written by calibration/ga4_params.py after the BigQuery pass. Until then the
# file carries source="placeholder" and CALIBRATED_MODE stays False.

with open(CALIB_PATH) as fh:
    OBSERVED = json.load(fh)

CALIBRATED_MODE = OBSERVED.get("source") == "ga4"

BASELINE_CONVERSION = OBSERVED["baseline_session_conversion"]
DAILY_USERS = OBSERVED["daily_users"]
SESSIONS_PER_USER_MEAN = OBSERVED["sessions_per_user_mean"]
SESSIONS_PER_USER_DISPERSION = OBSERVED["sessions_per_user_dispersion"]
OBSERVED_ICC = OBSERVED["session_conversion_icc"]
OBSERVED_RHO = OBSERVED["cuped_pre_post_correlation"]

PRIMARY_METRIC = OBSERVED.get("primary_metric", "session_conversion")
PRIMARY_METRIC_RATE = OBSERVED.get("primary_metric_rate", BASELINE_CONVERSION)
CONTRAST_METRIC = OBSERVED.get("contrast_metric")
CONTRAST_METRIC_ICC = OBSERVED.get("contrast_metric_icc")
CONTRAST_METRIC_RATE = OBSERVED.get("contrast_metric_rate")
NO_PREPERIOD_SHARE = OBSERVED.get("no_preperiod_share", 0.0)

# ---------------------------------------------------------------------------
# M1 — peeking inflates Type I error
# ---------------------------------------------------------------------------
M1 = dict(
    horizon_days=14,
    alpha=0.05,
    # (label, peek on these day indices, 1-based; horizon day always included)
    schedules={
        "fixed_horizon": [14],
        "weekly": [7, 14],
        "every_3_days": [3, 6, 9, 12, 14],
        "daily": list(range(1, 15)),
        "twice_daily": [d + h for d in range(0, 14) for h in (0.5, 1.0)],
    },
    # mSPRT mixture SD, expressed as a fraction of the baseline rate. This is
    # the "effect size you are tuned to detect"; too small kills power, too
    # large kills early stopping. See CHANGELOG_CONFIG 2026-08-15 amendment.
    msprt_tau_frac=0.20,          # headline value
    msprt_tau_frac_original=0.05,  # originally registered; retained for record
    msprt_tau_frac_grid=[0.05, 0.10, 0.20, 0.50, 1.00, 2.00],
    correction_arms=["uncorrected", "obrien_fleming", "msprt"],
)

# ---------------------------------------------------------------------------
# M2 — CUPED variance reduction
# ---------------------------------------------------------------------------
M2 = dict(
    rho_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    n_per_arm=20_000,
    true_lift=0.0,          # A/A for coverage; A/B leg uses true_lift_ab
    true_lift_ab=0.02,
    # Break conditions. The observed GA4 share (0.966) is the module's primary
    # condition, not an extreme case: see CHANGELOG_CONFIG 2026-08-15.
    no_preperiod_share=[0.0, 0.2, 0.5, 0.8, round(NO_PREPERIOD_SHARE, 3)],
    no_preperiod_handling=["theta_zero", "mean_impute", "exclude"],
    preperiod_drift=[0.0, 0.5, 1.0],   # SD units of drift between pre and post
    contaminated_covariate=[0.0, 0.25, 0.5],  # share of treatment effect
                                              # leaking into the covariate
)

# ---------------------------------------------------------------------------
# M3 — ratio metrics and the delta method
# ---------------------------------------------------------------------------
# Metric: binary session-level conversion under user-level randomization.
# Parameterised so the same DGP covers click-through-per-session (high
# baseline) and purchase-per-session (low baseline); only the rate changes.
M3 = dict(
    n_users_per_arm=10_000,
    # Both observed ICCs are pinned as grid points. The negative arm exists
    # because purchase-per-session measured -0.0395 on GA4, which is the
    # opposite sign from what every treatment of this topic assumes.
    # Both observed ICCs are positive and are pinned as grid points. The
    # negative arm is retained but is DESIGNED, not observed: no metric in the
    # calibration clusters negatively, and the feasibility floor -p/(1-p)
    # bounds how negative a low-rate binary metric can get. Kept because the
    # asymmetry of the coverage curve either side of zero is worth showing.
    icc_grid=sorted({-0.015, 0.0, round(CONTRAST_METRIC_ICC, 4), 0.10,
                     round(OBSERVED_ICC, 4), 0.30, 0.50}),
    metrics={
        "primary": dict(name=PRIMARY_METRIC, rate=PRIMARY_METRIC_RATE,
                        icc=OBSERVED_ICC),
        "contrast": dict(name=CONTRAST_METRIC, rate=CONTRAST_METRIC_RATE,
                         icc=CONTRAST_METRIC_ICC),
    },
    # Correlation between a user's session count and their conversion
    # propensity. Non-zero is what makes the Cov(S, n) term in the delta
    # method bite; zero is the textbook case.
    count_rate_coupling=[0.0, 0.5],
    estimators=["naive_session", "delta_method", "user_bootstrap"],
    bootstrap_reps=1_000,
    nominal_coverage=0.95,
)

# ---------------------------------------------------------------------------
# Structural assertions
# ---------------------------------------------------------------------------
assert M1["horizon_days"] == max(M1["schedules"]["daily"]), \
    "daily schedule must span the full horizon"
assert M1["schedules"]["fixed_horizon"] == [M1["horizon_days"]], \
    "fixed-horizon arm must peek exactly once, at the horizon"
assert M1["msprt_tau_frac"] in M1["msprt_tau_frac_grid"], \
    "the headline tau must be one of the swept values"
assert M1["msprt_tau_frac_original"] in M1["msprt_tau_frac_grid"], \
    "the originally registered tau must stay in the sweep, so the amendment " \
    "is auditable rather than a silent retune"
assert all(0.0 <= r <= 0.99 for r in M2["rho_grid"]), "rho out of range"
assert M2["rho_grid"] == sorted(M2["rho_grid"]), "rho grid must stay ordered"
assert M3["icc_grid"] == sorted(M3["icc_grid"]), "ICC grid must stay ordered"
assert all(-1.0 < i < 1.0 for i in M3["icc_grid"]), "ICC out of feasible range"
assert any(i < 0 for i in M3["icc_grid"]), \
    "the negative-ICC arm is the point of M3's amendment; do not drop it"
assert round(OBSERVED_ICC, 4) in M3["icc_grid"], "observed ICC must be a grid point"
assert round(CONTRAST_METRIC_ICC, 4) in M3["icc_grid"], \
    "observed contrast ICC must be a grid point"
assert CONTRAST_METRIC_ICC > 0 and OBSERVED_ICC > 0, \
    "both calibrated metrics cluster positively; the negative arm is designed"
assert OBSERVED_ICC > CONTRAST_METRIC_ICC, \
    "primary metric must be the stronger-clustering one, or the contrast " \
    "loses its point"
assert "icc_estimator_audit" in OBSERVED, \
    "the ANOVA-to-stratified correction must stay auditable"
assert round(NO_PREPERIOD_SHARE, 3) in M2["no_preperiod_share"], \
    "observed no-preperiod share must be a grid point"
assert 0.0 < BASELINE_CONVERSION < 1.0, "baseline conversion out of range"
assert len(CHANGELOG_CONFIG) >= 1, "config changes require a changelog entry"


def require_calibrated(module: str) -> None:
    """Guard for runs that claim GA4-calibrated parameters.

    Structural enforcement: a figure captioned 'calibrated to GA4' cannot be
    produced from placeholder numbers. Runners call this when invoked with
    --calibrated.
    """
    if not CALIBRATED_MODE:
        raise RuntimeError(
            f"{module}: calibrated mode requested but "
            f"calibration/observed_params.json has source="
            f"{OBSERVED.get('source')!r}. Run calibration/ga4_params.py first, "
            f"or drop --calibrated to run on placeholder parameters."
        )

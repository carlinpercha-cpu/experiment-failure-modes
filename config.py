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
    # Break conditions
    no_preperiod_share=[0.0, 0.2, 0.5],
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
    icc_grid=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50],
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

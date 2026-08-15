"""Estimators under test. Each is written to be checkable against a known
truth in tests/, because in a simulation study a deviation from theory is a
bug before it is a finding.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Fixed-horizon and sequential tests (M1)
# ---------------------------------------------------------------------------


def two_proportion_z(conv_c: np.ndarray, conv_t: np.ndarray,
                     n_per_arm: np.ndarray) -> np.ndarray:
    """Pooled two-proportion z statistic. Broadcasts over any leading shape."""
    p_c = conv_c / n_per_arm
    p_t = conv_t / n_per_arm
    p_pool = (conv_c + conv_t) / (2.0 * n_per_arm)
    se = np.sqrt(2.0 * p_pool * (1.0 - p_pool) / n_per_arm)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p_t - p_c) / se, 0.0)
    return z


def two_sided_p(z: np.ndarray) -> np.ndarray:
    return 2.0 * stats.norm.sf(np.abs(z))


def obrien_fleming_alphas(info_fractions: np.ndarray,
                          alpha: float = 0.05) -> np.ndarray:
    """Per-look nominal alphas from the Lan-DeMets O'Brien-Fleming spending
    function, alpha*(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t))).

    We take the per-look budget as the increment of the spending function.
    This ignores the positive correlation between successive looks, so the
    realized family-wise error is *below* the nominal alpha: the arm is
    conservative, not exact. Pre-registered as such; the exact recursive
    boundary would require numerical multivariate-normal integration and is
    out of scope.
    """
    t = np.asarray(info_fractions, dtype=float)
    if np.any(t <= 0) or np.any(t > 1.0 + 1e-9):
        raise ValueError("information fractions must lie in (0, 1]")
    z_half = stats.norm.isf(alpha / 2.0)
    spent = 2.0 * stats.norm.sf(z_half / np.sqrt(t))
    return np.diff(np.concatenate([[0.0], spent]))


def msprt_always_valid_p(conv_c: np.ndarray, conv_t: np.ndarray,
                         n_per_arm: np.ndarray, tau: float) -> np.ndarray:
    """Always-valid p-values from a normal-mixture SPRT (Johari et al.).

    For a difference of means d with per-arm n and common variance sigma^2,
    the mixture likelihood ratio under theta ~ N(0, tau^2) is

        Lambda = sqrt(V / (V + tau^2)) * exp(d^2 tau^2 / (2 V (V + tau^2)))

    with V = 2 sigma^2 / n. The always-valid p-value is the running minimum
    of 1 / Lambda, capped at 1.

    Expects arrays shaped (..., n_looks); the running minimum is taken over
    the final axis.
    """
    p_c = conv_c / n_per_arm
    p_t = conv_t / n_per_arm
    p_pool = (conv_c + conv_t) / (2.0 * n_per_arm)
    sigma2 = p_pool * (1.0 - p_pool)
    d = p_t - p_c

    v = 2.0 * sigma2 / n_per_arm
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lambda = 0.5 * np.log(v / (v + tau ** 2)) + \
            (d ** 2 * tau ** 2) / (2.0 * v * (v + tau ** 2))
    log_lambda = np.nan_to_num(log_lambda, nan=0.0, posinf=700.0, neginf=-700.0)

    p = np.minimum(1.0, np.exp(-log_lambda))
    return np.minimum.accumulate(p, axis=-1)


def first_crossing(p_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """True where any look crosses its threshold. Shape (..., n_looks) in,
    (...,) out."""
    return np.any(p_values <= thresholds, axis=-1)


# ---------------------------------------------------------------------------
# CUPED (M2)
# ---------------------------------------------------------------------------


def cuped_theta(y: np.ndarray, x: np.ndarray) -> float:
    """theta = Cov(Y, X) / Var(X), estimated on the pooled sample."""
    ok = np.isfinite(x) & np.isfinite(y)
    var_x = np.var(x[ok], ddof=1)
    if var_x <= 0:
        return 0.0
    return float(np.cov(y[ok], x[ok], ddof=1)[0, 1] / var_x)


def cuped_adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None,
                 missing_policy: str = "theta_zero"
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Return adjusted outcomes and the mask of rows retained.

    missing_policy:
      theta_zero  -- users with no pre-period keep the raw outcome
      mean_impute -- missing X set to the observed mean (adjustment is a no-op
                     for those users, but they stay in the sample)
      exclude     -- users with no pre-period are dropped entirely, which
                     changes the estimand
    """
    if theta is None:
        theta = cuped_theta(y, x)

    has_x = np.isfinite(x)
    keep = np.ones_like(has_x, dtype=bool)
    x_use = x.copy()

    if missing_policy == "exclude":
        keep = has_x
    elif missing_policy == "mean_impute":
        x_use[~has_x] = np.nanmean(x)
    elif missing_policy == "theta_zero":
        x_use[~has_x] = np.nanmean(x)
    else:
        raise ValueError(f"unknown missing_policy {missing_policy!r}")

    x_centered = x_use - np.nanmean(x_use[keep])
    y_adj = y - theta * x_centered
    return y_adj[keep], keep


# ---------------------------------------------------------------------------
# Ratio metrics (M3)
# ---------------------------------------------------------------------------


def ratio_estimate(successes: np.ndarray, sessions: np.ndarray) -> float:
    return float(successes.sum() / sessions.sum())


def naive_session_se(successes: np.ndarray, sessions: np.ndarray) -> float:
    """Treats every session as an independent Bernoulli draw. Wrong whenever
    sessions within a user are correlated."""
    n = sessions.sum()
    p = successes.sum() / n
    return float(np.sqrt(p * (1.0 - p) / n))


def delta_method_se(successes: np.ndarray, sessions: np.ndarray) -> float:
    """SE of a ratio of means, users as the independent unit.

    Var(R) ~= (1 / (N * mean_n^2)) * [Var(S) - 2 R Cov(S, n) + R^2 Var(n)]
    """
    n_users = len(sessions)
    mean_n = sessions.mean()
    r = successes.sum() / sessions.sum()

    var_s = np.var(successes, ddof=1)
    var_n = np.var(sessions, ddof=1)
    cov_sn = np.cov(successes, sessions, ddof=1)[0, 1]

    var_r = (var_s - 2.0 * r * cov_sn + r ** 2 * var_n) / (n_users * mean_n ** 2)
    return float(np.sqrt(max(var_r, 0.0)))


def user_bootstrap_se(successes: np.ndarray, sessions: np.ndarray,
                      reps: int, rng: np.random.Generator) -> float:
    """Resample users with replacement; recompute the ratio."""
    n = len(sessions)
    idx = rng.integers(0, n, size=(reps, n))
    boot = successes[idx].sum(axis=1) / sessions[idx].sum(axis=1)
    return float(boot.std(ddof=1))

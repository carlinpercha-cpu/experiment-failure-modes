"""Data-generating processes. Ground truth is known by construction in all of
these; any deviation of an estimator from theory is a defect in the estimator
until proven otherwise.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# M1 — daily arrival stream of binary conversions
# ---------------------------------------------------------------------------


def daily_conversion_stream(
    rng: np.random.Generator,
    n_sims: int,
    n_days: int,
    users_per_day_per_arm: int,
    p_control: float,
    lift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate cumulative conversions and exposures for two arms.

    Returns
    -------
    conv : (n_sims, n_days, 2) cumulative conversions, columns = (control, treat)
    expo : (n_days,) cumulative exposures per arm

    lift is a *relative* lift on the control rate. lift=0 gives an A/A test.
    """
    p_treat = p_control * (1.0 + lift)
    if not 0.0 < p_treat < 1.0:
        raise ValueError(f"treatment rate {p_treat} out of range")

    daily = np.empty((n_sims, n_days, 2), dtype=np.int64)
    daily[:, :, 0] = rng.binomial(users_per_day_per_arm, p_control,
                                  size=(n_sims, n_days))
    daily[:, :, 1] = rng.binomial(users_per_day_per_arm, p_treat,
                                  size=(n_sims, n_days))

    conv = np.cumsum(daily, axis=1)
    expo = np.arange(1, n_days + 1, dtype=np.int64) * users_per_day_per_arm
    return conv, expo


# ---------------------------------------------------------------------------
# M2 — correlated pre-period covariate
# ---------------------------------------------------------------------------


def cuped_panel(
    rng: np.random.Generator,
    n_per_arm: int,
    rho: float,
    true_lift: float = 0.0,
    drift: float = 0.0,
    contamination: float = 0.0,
    no_pre_share: float = 0.0,
) -> dict[str, np.ndarray]:
    """Continuous outcome Y with pre-period covariate X, corr(X, Y) = rho.

    drift          : mean shift (SD units) applied to the post period only,
                     breaking the stationarity CUPED assumes.
    contamination  : share of the treatment effect that leaks into X. X is
                     nominally pre-experiment, so any non-zero value means the
                     covariate is post-treatment and CUPED becomes biased.
    no_pre_share   : share of users with no pre-period history (X missing).
    """
    n = 2 * n_per_arm
    treat = np.zeros(n, dtype=bool)
    treat[n_per_arm:] = True

    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    y = rho * x + np.sqrt(max(0.0, 1.0 - rho ** 2)) * eps
    y = y + drift
    y = y + true_lift * treat

    if contamination > 0.0:
        x = x + contamination * true_lift * treat

    missing = np.zeros(n, dtype=bool)
    if no_pre_share > 0.0:
        missing = rng.random(n) < no_pre_share
        x = x.copy()
        x[missing] = np.nan

    return dict(y=y, x=x, treat=treat, missing=missing)


# ---------------------------------------------------------------------------
# M3 — user-level randomization, session-level metric
# ---------------------------------------------------------------------------


def _beta_params(mean_rate: float, icc: float) -> tuple[float, float]:
    """Beta(a, b) with the given mean and intraclass correlation.

    For a beta-binomial, ICC = 1 / (a + b + 1).
    """
    if icc <= 0.0:
        raise ValueError("use icc > 0; icc == 0 is handled by the caller")
    total = 1.0 / icc - 1.0
    return mean_rate * total, (1.0 - mean_rate) * total


def ratio_metric_panel(
    rng: np.random.Generator,
    n_users_per_arm: int,
    sessions_mean: float,
    sessions_dispersion: float,
    p_control: float,
    icc: float,
    coupling: float = 0.0,
    lift: float = 0.0,
) -> dict[str, np.ndarray]:
    """Users get sessions; sessions convert. Randomization is at the user.

    icc      : within-user correlation of session conversion. 0 gives
               independent sessions, where the naive session-level SE is
               correct.
    coupling : correlation between a user's session count and their conversion
               propensity, induced by rank-matching. Non-zero makes the
               Cov(S, n) term of the delta method non-negligible.

    Returns per-user session counts and conversion counts for both arms.
    """
    out = {}
    for arm, rate in (("control", p_control),
                      ("treat", p_control * (1.0 + lift))):
        n = n_users_per_arm

        # Session counts: negative binomial, at least one session.
        var = sessions_mean * sessions_dispersion
        if var > sessions_mean:
            p_nb = sessions_mean / var
            r_nb = sessions_mean * p_nb / (1.0 - p_nb)
            counts = rng.negative_binomial(r_nb, p_nb, size=n) + 1
        else:
            counts = rng.poisson(sessions_mean, size=n) + 1

        # User-level conversion propensity.
        if icc <= 0.0:
            rates = np.full(n, rate, dtype=float)
        else:
            a, b = _beta_params(rate, icc)
            rates = rng.beta(a, b, size=n)

        if coupling > 0.0 and icc > 0.0:
            # Rank-couple counts and rates for a share of users, so heavier
            # users are systematically higher- (or lower-) converting.
            k = int(round(coupling * n))
            if k > 1:
                idx = rng.choice(n, size=k, replace=False)
                counts_sub = np.sort(counts[idx])
                rates_sub = np.sort(rates[idx])
                order = rng.permutation(k)
                counts[idx] = counts_sub[order]
                rates[idx] = rates_sub[order]

        successes = rng.binomial(counts, rates)
        out[f"{arm}_sessions"] = counts.astype(np.int64)
        out[f"{arm}_successes"] = successes.astype(np.int64)

    return out

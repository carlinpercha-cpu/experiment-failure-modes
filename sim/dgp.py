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


def icc_feasible_floor(rate: float) -> float:
    """Most negative ICC attainable for exchangeable binary sessions.

    Two conversions in one user cannot be rarer than impossible, so
    Cov >= -p^2 and corr >= -p/(1-p). Any measurement below this floor is not
    a within-user correlation: it indicates the exchangeable-binary model is
    wrong for that metric (e.g. conversion rate varying with session count),
    not that dependence is unusually strong.
    """
    return -rate / (1.0 - rate)


def _negative_icc_successes(rng: np.random.Generator, counts: np.ndarray,
                            rate: float, icc: float) -> np.ndarray:
    """Session outcomes with negative within-user dependence.

    Mechanism, chosen to match the substantive story rather than to hit a
    number: converting suppresses later conversion within the same user. A
    user converts with probability p0 in each session, reduced by a factor
    (1 - supp) in every session after their first conversion. supp = 1 makes
    conversion strictly at-most-once per user, which is the maximally
    negative case.

    p0 is calibrated by fixed point so the marginal session-level rate matches
    the target, and supp by bisection so the realized ICC matches the request.
    Requests below icc_feasible_floor(rate) are clamped to the floor and the
    realized value is returned to the caller via the panel, because M3 plots
    realized ICC, never requested ICC.

    A Beta mixture cannot be used here: mixing independent Bernoullis induces
    only non-negative correlation.
    """
    floor = icc_feasible_floor(rate)
    target = max(icc, floor)

    def draw(p0: float, supp: float) -> np.ndarray:
        n = len(counts)
        s = np.zeros(n, dtype=np.int64)
        had = np.zeros(n, dtype=bool)
        for j in range(int(counts.max())):
            active = counts > j
            prob = np.where(had, p0 * (1.0 - supp), p0)
            hit = (rng.random(n) < prob) & active
            s += hit
            had |= hit
        return s

    def draw_calibrated(supp: float) -> np.ndarray:
        p0 = rate
        for _ in range(6):
            s = draw(p0, supp)
            realized = s.sum() / counts.sum()
            if realized <= 0:
                break
            p0 *= rate / realized
            p0 = min(p0, 0.999)
        return s

    lo, hi = 0.0, 1.0
    best = draw_calibrated(1.0)
    if observed_icc(best, counts) > target:
        return best  # even full suppression is not negative enough

    for _ in range(10):
        mid = 0.5 * (lo + hi)
        s = draw_calibrated(mid)
        if observed_icc(s, counts) > target:
            lo = mid
        else:
            hi = mid
        best = s
    return best


def observed_icc(successes: np.ndarray, sessions: np.ndarray) -> float:
    """Fleiss-Cuzick ANOVA ICC on multi-session users, matching the SQL.

    Exposed because M3 reports the realized ICC of each simulated panel rather
    than the value it requested.
    """
    m = sessions >= 2
    if m.sum() < 2:
        return float("nan")
    s = successes[m].astype(float)
    n = sessions[m].astype(float)
    k, big_n = len(n), n.sum()
    p = s.sum() / big_n
    if p <= 0 or p >= 1:
        return float("nan")
    msb = np.sum(n * (s / n - p) ** 2) / (k - 1)
    msw = np.sum(s * (1 - s / n)) / (big_n - k)
    n0 = (big_n - np.sum(n ** 2) / big_n) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


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

        # Session counts: every user has at least one session, so the count is
        # 1 + a non-negative draw. The shift moves the mean, so the underlying
        # draw must target (sessions_mean - 1), not sessions_mean.
        #
        # BUGFIX 2026-08-15: both branches previously drew with mean
        # sessions_mean and then added 1, producing a realized mean of
        # sessions_mean + 1. Undetected because no test asserted the realized
        # count distribution matched its target. See CHANGELOG_CONFIG.
        var_target = sessions_mean * sessions_dispersion
        shifted_mean = sessions_mean - 1.0
        if shifted_mean < 0.0:
            raise ValueError("mean sessions per user cannot be below 1")

        if shifted_mean == 0.0:
            counts = np.ones(n, dtype=np.int64)
        elif var_target > shifted_mean:
            p_nb = shifted_mean / var_target
            r_nb = shifted_mean * p_nb / (1.0 - p_nb)
            counts = rng.negative_binomial(r_nb, p_nb, size=n) + 1
        else:
            counts = rng.poisson(shifted_mean, size=n) + 1

        # User-level conversion propensity.
        #
        # icc > 0: Beta-distributed propensities give positive within-user
        #          correlation (some users convert more than others).
        # icc = 0: constant propensity, independent sessions.
        # icc < 0: negative within-user dependence. Cannot be produced by a
        #          mixture of independent Bernoullis, so it is imposed
        #          directly on the counts (see _negative_icc_successes).
        if icc == 0.0:
            rates = np.full(n, rate, dtype=float)
        elif icc > 0.0:
            a, b = _beta_params(rate, icc)
            rates = rng.beta(a, b, size=n)
        else:
            rates = np.full(n, rate, dtype=float)

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

        if icc < 0.0:
            successes = _negative_icc_successes(rng, counts, rate, icc)
        else:
            successes = rng.binomial(counts, rates)

        out[f"{arm}_sessions"] = counts.astype(np.int64)
        out[f"{arm}_successes"] = successes.astype(np.int64)

    return out

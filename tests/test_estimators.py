"""Estimator correctness under known truth.

The premise of this repo is that ground truth is known by construction, so a
gap between an estimator and its theory is a bug, not a result. These tests
are the mechanism that keeps that claim honest.

Run: pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

import config as cfg
from sim import dgp, estimators as est


# ---------------------------------------------------------------------------
# Config integrity
# ---------------------------------------------------------------------------


def test_changelog_present():
    assert cfg.CHANGELOG_CONFIG, "config changes require a changelog entry"


def test_calibrated_guard_blocks_placeholder():
    if cfg.CALIBRATED_MODE:
        pytest.skip("calibration already run")
    with pytest.raises(RuntimeError):
        cfg.require_calibrated("test")


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------


def test_fixed_horizon_is_nominal():
    """A single look at alpha=0.05 must reject ~5% of the time under the null."""
    rng = np.random.default_rng(7)
    conv, expo = dgp.daily_conversion_stream(rng, 20_000, 1, 5_000, 0.05, 0.0)
    z = est.two_proportion_z(conv[:, -1, 0], conv[:, -1, 1], expo[-1])
    p = est.two_sided_p(z)
    assert 0.043 < (p <= 0.05).mean() < 0.058


def test_obrien_fleming_spends_full_budget():
    info = np.linspace(0.1, 1.0, 10)
    a = est.obrien_fleming_alphas(info, alpha=0.05)
    assert np.all(a >= 0)
    assert a[0] < a[-1], "OBF must be strict early and permissive late"
    assert np.isclose(a.sum(), 0.05, atol=1e-6)


def test_msprt_p_is_monotone_and_bounded():
    rng = np.random.default_rng(11)
    conv, expo = dgp.daily_conversion_stream(rng, 200, 14, 2_000, 0.05, 0.0)
    n = expo.astype(float)[None, :]
    p = est.msprt_always_valid_p(conv[:, :, 0].astype(float),
                                conv[:, :, 1].astype(float), n, tau=0.0025)
    assert np.all(p <= 1.0) and np.all(p >= 0.0)
    assert np.all(np.diff(p, axis=-1) <= 1e-12), "running minimum must not rise"


# ---------------------------------------------------------------------------
# M2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_cuped_matches_one_minus_rho_squared(rho):
    """Realized variance reduction must track 1 - rho^2 within 2pp.

    Pre-registered tolerance. A breach is a defect in cuped_adjust, not a
    finding about CUPED.
    """
    rng = np.random.default_rng(3)
    d = dgp.cuped_panel(rng, n_per_arm=200_000, rho=rho)
    theta = est.cuped_theta(d["y"], d["x"])
    y_adj, keep = est.cuped_adjust(d["y"], d["x"], theta)

    realized = 1.0 - np.var(y_adj, ddof=1) / np.var(d["y"], ddof=1)
    theoretical = rho ** 2
    assert abs(realized - theoretical) < 0.02, \
        f"rho={rho}: realized {realized:.4f} vs theoretical {theoretical:.4f}"


def test_cuped_theta_is_zero_when_uncorrelated():
    rng = np.random.default_rng(5)
    d = dgp.cuped_panel(rng, n_per_arm=100_000, rho=0.0)
    assert abs(est.cuped_theta(d["y"], d["x"])) < 0.02


def test_contaminated_covariate_biases_the_estimate():
    """A covariate touched by treatment makes CUPED biased, not just noisy."""
    rng = np.random.default_rng(13)
    lift = 0.20
    d = dgp.cuped_panel(rng, n_per_arm=200_000, rho=0.6,
                        true_lift=lift, contamination=0.5)
    theta = est.cuped_theta(d["y"], d["x"])
    y_adj, keep = est.cuped_adjust(d["y"], d["x"], theta)
    t = d["treat"][keep]
    biased = y_adj[t].mean() - y_adj[~t].mean()
    assert abs(biased - lift) > 0.01, "contamination should move the estimate"


# ---------------------------------------------------------------------------
# M3
# ---------------------------------------------------------------------------


def test_naive_and_delta_agree_when_icc_zero():
    """With independent sessions and no count coupling, the naive session-level
    SE is correct and the delta method must reproduce it."""
    rng = np.random.default_rng(17)
    d = dgp.ratio_metric_panel(rng, 40_000, 1.6, 1.2, 0.05, icc=0.0)
    s = d["control_successes"]
    n = d["control_sessions"]
    naive = est.naive_session_se(s, n)
    delta = est.delta_method_se(s, n)
    assert abs(delta - naive) / naive < 0.10


def test_naive_understates_when_icc_positive():
    rng = np.random.default_rng(19)
    d = dgp.ratio_metric_panel(rng, 40_000, 1.6, 1.2, 0.05, icc=0.30)
    s = d["control_successes"]
    n = d["control_sessions"]
    assert est.delta_method_se(s, n) > est.naive_session_se(s, n)


def test_delta_matches_bootstrap():
    """Two correct estimators of the same quantity must agree."""
    rng = np.random.default_rng(23)
    d = dgp.ratio_metric_panel(rng, 20_000, 1.6, 1.2, 0.05, icc=0.20)
    s = d["control_successes"]
    n = d["control_sessions"]
    delta = est.delta_method_se(s, n)
    boot = est.user_bootstrap_se(s, n, reps=500, rng=rng)
    assert abs(delta - boot) / delta < 0.10


def test_dgp_hits_target_rate_and_icc():
    """The DGP must produce the ICC it was asked for, or M3's x-axis is a lie."""
    rng = np.random.default_rng(29)
    target = 0.20
    d = dgp.ratio_metric_panel(rng, 60_000, 3.0, 1.5, 0.10, icc=target)
    s = d["control_successes"].astype(float)
    n = d["control_sessions"].astype(float)

    assert abs(s.sum() / n.sum() - 0.10) < 0.005

    # Method-of-moments ICC for a beta-binomial with unequal cluster sizes.
    p = s.sum() / n.sum()
    msw = np.sum(s * (1 - s / n)) / (n.sum() - len(n))
    icc_hat = 1.0 - msw / (p * (1 - p))
    assert abs(icc_hat - target) < 0.05, f"realized ICC {icc_hat:.3f}"


# ---------------------------------------------------------------------------
# Added 2026-08-15 with the calibration pass
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mean,disp", [(1.333, 0.754), (2.5, 1.6), (1.0, 0.5)])
def test_session_counts_match_target_moments(mean, disp):
    """Realized mean session count must match the target.

    This is the test that was missing when ratio_metric_panel drew with mean
    `sessions_mean` and then added 1, giving a realized mean of
    `sessions_mean + 1`. The bug survived because the existing tests checked
    the conversion rate and the ICC but never the count distribution.
    """
    rng = np.random.default_rng(101)
    d = dgp.ratio_metric_panel(rng, 200_000, mean, disp, 0.05, icc=0.0)
    realized = d["control_sessions"].mean()
    assert abs(realized - mean) < 0.02, \
        f"target mean {mean}, realized {realized:.4f}"
    assert d["control_sessions"].min() >= 1, "every user needs >= 1 session"


def test_calibration_is_loaded():
    """The repo should now be running on measured, not invented, parameters."""
    assert cfg.CALIBRATED_MODE, "expected observed_params.json source=ga4"
    assert cfg.OBSERVED["bigquery_bytes_scanned"] is not None


def test_observed_values_are_pinned_to_grids():
    """A calibrated value that is not a grid point is a value nothing uses."""
    assert round(cfg.OBSERVED_ICC, 4) in cfg.M3["icc_grid"]
    assert round(cfg.CONTRAST_METRIC_ICC, 4) in cfg.M3["icc_grid"]
    assert round(cfg.NO_PREPERIOD_SHARE, 3) in cfg.M2["no_preperiod_share"]


def _anova_icc(successes, sessions):
    """Fleiss-Cuzick ANOVA ICC, matching the estimator used in the SQL."""
    m = sessions >= 2
    s, n = successes[m].astype(float), sessions[m].astype(float)
    k, bigN = len(n), n.sum()
    p = s.sum() / bigN
    msb = np.sum(n * (s / n - p) ** 2) / (k - 1)
    msw = np.sum(s * (1 - s / n)) / (bigN - k)
    n0 = (bigN - np.sum(n ** 2) / bigN) / (k - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


@pytest.mark.parametrize("target", [0.05, 0.1263, 0.30])
def test_positive_icc_is_recovered(target):
    rng = np.random.default_rng(103)
    d = dgp.ratio_metric_panel(rng, 150_000, 3.0, 2.0, 0.0674, icc=target)
    icc_hat = _anova_icc(d["control_successes"], d["control_sessions"])
    assert abs(icc_hat - target) < 0.03, f"target {target}, realized {icc_hat:.4f}"


def test_negative_icc_floor_is_enforced():
    """Requests below the feasibility floor must be clamped, not fabricated.

    For exchangeable binary sessions, corr >= -p/(1-p). The GA4 purchase
    estimate (-0.0395 at p=0.0275) sits below its floor of -0.0282, so it
    cannot be a pure within-user correlation and the DGP must not pretend to
    reproduce it.
    """
    rate = 0.0275
    floor = dgp.icc_feasible_floor(rate)
    assert floor == pytest.approx(-0.0283, abs=1e-3)

    rng = np.random.default_rng(107)
    d = dgp.ratio_metric_panel(rng, 120_000, 3.0, 2.0, rate, icc=-0.50)
    realized = dgp.observed_icc(d["control_successes"], d["control_sessions"])
    assert realized >= floor - 0.01, \
        f"realized {realized:.4f} below feasibility floor {floor:.4f}"
    assert realized < 0.0


def test_negative_icc_preserves_marginal_rate():
    rng = np.random.default_rng(113)
    rate = 0.0275
    d = dgp.ratio_metric_panel(rng, 120_000, 3.0, 2.0, rate, icc=-0.02)
    s, n = d["control_successes"], d["control_sessions"]
    assert abs(s.sum() / n.sum() - rate) / rate < 0.10, "marginal rate drifted"
    assert dgp.observed_icc(s, n) < 0.02


def test_naive_overstates_when_icc_negative():
    """The finding that flips M3: with negative ICC the naive session-level SE
    is conservative, not anti-conservative."""
    rng = np.random.default_rng(109)
    d = dgp.ratio_metric_panel(rng, 120_000, 3.0, 2.0, 0.0275, icc=-0.025)
    s, n = d["control_successes"], d["control_sessions"]
    assert est.delta_method_se(s, n) < est.naive_session_se(s, n)



def test_anova_icc_is_biased_by_size_rate_correlation():
    """The estimator finding, as an executable claim.

    Sessions are INDEPENDENT within user (true ICC = 0), but per-session rate
    rises with session count, matching the observed GA4 structure. The
    Fleiss-Cuzick ANOVA estimator must return a materially negative value --
    which is why the -0.0395 measured on purchase was an artifact rather than
    evidence of negative dependence.
    """
    rng = np.random.default_rng(211)
    sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    users = [222790, 29536, 8322, 3823, 2102, 1238, 791, 1552]
    rates = [0.00484, 0.01588, 0.02936, 0.03623, 0.03892,
             0.04187, 0.03883, 0.04197]

    counts = np.concatenate([np.full(u, n) for n, u in zip(sizes, users)])
    p = np.concatenate([np.full(u, r) for u, r in zip(users, rates)])
    successes = rng.binomial(counts, p)

    icc_hat = dgp.observed_icc(successes, counts)
    assert icc_hat < -0.03, \
        f"expected substantial downward bias, got {icc_hat:.4f}"


def test_stratified_icc_is_unbiased_under_the_same_structure():
    """The corrected estimator: hold cluster size fixed, and the size-rate
    confound cannot operate. Must recover ~0 where ANOVA reports ~-0.07."""
    rng = np.random.default_rng(213)
    est_by_n = []
    for n, k, rate in ((2, 60_000, 0.01588), (3, 40_000, 0.02936),
                       (4, 30_000, 0.03623)):
        s = rng.binomial(n, rate, size=k)
        p_hat = s.mean() / n
        rho = (s.var(ddof=1) / (n * p_hat * (1 - p_hat)) - 1) / (n - 1)
        est_by_n.append(rho)
    assert all(abs(r) < 0.03 for r in est_by_n), \
        f"stratified estimator should be near zero, got {est_by_n}"


def test_calibrated_iccs_are_positive_and_ordered():
    assert cfg.OBSERVED_ICC > cfg.CONTRAST_METRIC_ICC > 0
    audit = cfg.OBSERVED["icc_estimator_audit"]
    assert audit["purchase"]["anova_fleiss_cuzick"] < 0 < \
        audit["purchase"]["pooled"], "the sign flip must stay on the record"


def test_negative_icc_calibration_is_deterministic():
    """The (p0, supp) pair must not depend on which draw warmed the cache.

    It did: the pair was calibrated against the caller's own panel, and since
    the bisection target is an ICC estimated on only the users with 2+
    sessions, that noise leaked into the between-replicate variance of the
    ratio -- inflating the true sampling SD at negative ICC above its icc=0
    value, which is backwards. Now calibrated on a fixed 400k-user pilot.
    """
    pairs = []
    for seed in (1, 2, 3):
        dgp._NEG_ICC_CACHE.clear()
        rng = np.random.default_rng(seed)
        dgp.ratio_metric_panel(rng, 5_000, 1.333, 0.754, 0.0674, icc=-0.015)
        pairs.append(list(dgp._NEG_ICC_CACHE.values())[0])
    assert len({(round(p, 8), round(s, 8)) for p, s in pairs}) == 1, \
        f"calibration varied across seeds: {pairs}"


def test_design_effect_identity_holds():
    """The module's recommendation, as an executable claim.

    (delta SE / naive SE)^2 must equal 1 + (n0 - 1) * ICC, where n0 is the
    SESSION-WEIGHTED mean cluster size sum(n^2)/sum(n) of the sample in hand.
    This is why the SE ratio is preferable to plugging a published ICC into
    the formula: it needs neither an ICC estimate nor the right n0.
    """
    rng = np.random.default_rng(307)
    for icc in (0.10, 0.188, 0.30):
        d = dgp.ratio_metric_panel(rng, 150_000, 1.333, 0.754, 0.0674, icc=icc)
        s, n = d["control_successes"], d["control_sessions"].astype(float)
        observed = (est.delta_method_se(s, n) / est.naive_session_se(s, n)) ** 2
        n0 = (n ** 2).sum() / n.sum()
        implied = 1 + (n0 - 1) * dgp.observed_icc(s, n.astype(np.int64))
        assert abs(observed - implied) / implied < 0.03, \
            f"icc={icc}: observed {observed:.4f} vs implied {implied:.4f}"


@pytest.mark.parametrize("c", [0.25, 0.50])
def test_contamination_bias_matches_closed_form(c):
    """M2 P8: a covariate carrying share c of the treatment effect biases the
    CUPED estimate to lift*(1 - theta*c), i.e. relative bias -theta*c.

    Registered post hoc; see PREREG_M2_CUPED Amendment 2.
    """
    rng = np.random.default_rng(401)
    rho, lift = 0.6, 0.02

    effects, thetas = [], []
    for _ in range(60):
        d = dgp.cuped_panel(rng, n_per_arm=40_000, rho=rho,
                            true_lift=lift, contamination=c)
        theta = est.cuped_theta(d["y"], d["x"])
        y_adj, keep = est.cuped_adjust(d["y"], d["x"], theta)
        t = d["treat"][keep]
        effects.append(y_adj[t].mean() - y_adj[~t].mean())
        thetas.append(theta)

    relative_bias = (np.mean(effects) - lift) / lift
    predicted = -np.mean(thetas) * c
    assert abs(relative_bias - predicted) < 0.02, \
        f"c={c}: observed {relative_bias:.4f}, closed form {predicted:.4f}"


def test_contamination_leaves_variance_reduction_intact():
    """The failure is silent: variance still falls while the estimate moves."""
    rng = np.random.default_rng(403)
    reductions = []
    for c in (0.0, 0.5):
        d = dgp.cuped_panel(rng, n_per_arm=40_000, rho=0.6,
                            true_lift=0.02, contamination=c)
        theta = est.cuped_theta(d["y"], d["x"])
        y_adj, keep = est.cuped_adjust(d["y"], d["x"], theta)
        reductions.append(1 - np.var(y_adj, ddof=1) /
                          np.var(d["y"][keep], ddof=1))
    assert abs(reductions[0] - reductions[1]) < 0.01, \
        "contamination should not show up in the variance reduction"

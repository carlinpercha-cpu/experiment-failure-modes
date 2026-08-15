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

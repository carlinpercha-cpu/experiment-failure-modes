"""M2 — CUPED variance reduction and where it breaks.

Four parts:
  A  rho sweep: realized vs theoretical 1 - rho^2
  B  coverage sweep: effective reduction when most users have no pre-period,
     across three handling policies. This is the module's primary result --
     96.6% of GA4 post-period users have no history.
  C  pre-period drift
  D  covariate contaminated by treatment, reporting BIAS not variance

Usage:
    python -m sim.run_m2
    python -m sim.run_m2 --calibrated
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config as cfg
from sim import dgp, estimators as est

N_SIMS = 400
N_PER_ARM = 20_000


def _effect(y, treat):
    return float(y[treat].mean() - y[~treat].mean())


def _run_cell(rng, rho, lift, drift, contamination, no_pre, policy, n_sims):
    raw_var, adj_var, effects, thetas = [], [], [], []
    for _ in range(n_sims):
        d = dgp.cuped_panel(rng, N_PER_ARM, rho, true_lift=lift, drift=drift,
                            contamination=contamination, no_pre_share=no_pre)
        theta = est.cuped_theta(d["y"], d["x"])
        y_adj, keep = est.cuped_adjust(d["y"], d["x"], theta,
                                       missing_policy=policy)
        raw_var.append(np.var(d["y"][keep], ddof=1))
        adj_var.append(np.var(y_adj, ddof=1))
        effects.append(_effect(y_adj, d["treat"][keep]))
        thetas.append(theta)
    return dict(
        variance_reduction=1.0 - np.mean(adj_var) / np.mean(raw_var),
        theta=float(np.mean(thetas)),
        effect=float(np.mean(effects)),
        bias=float(np.mean(effects) - lift),
    )


def part_a(rng, n_sims):
    rows = []
    for rho in cfg.M2["rho_grid"]:
        r = _run_cell(rng, rho, 0.0, 0.0, 0.0, 0.0, "theta_zero", n_sims)
        rows.append(dict(rho=rho, realized=r["variance_reduction"],
                         theoretical=rho ** 2,
                         gap=r["variance_reduction"] - rho ** 2))
    return pd.DataFrame(rows)


def part_b(rng, n_sims):
    rho = cfg.OBSERVED_RHO
    rows = []
    for share in cfg.M2["no_preperiod_share"]:
        for policy in cfg.M2["no_preperiod_handling"]:
            r = _run_cell(rng, rho, cfg.M2["true_lift_ab"], 0.0, 0.0,
                          share, policy, n_sims)
            rows.append(dict(no_preperiod_share=share, policy=policy,
                             coverage=1.0 - share,
                             variance_reduction=r["variance_reduction"],
                             predicted=(1.0 - share) * rho ** 2,
                             bias=r["bias"]))
    return pd.DataFrame(rows)


def part_c(rng, n_sims):
    rows = []
    for drift in cfg.M2["preperiod_drift"]:
        r = _run_cell(rng, 0.6, cfg.M2["true_lift_ab"], drift, 0.0, 0.0,
                      "theta_zero", n_sims)
        rows.append(dict(drift=drift,
                         variance_reduction=r["variance_reduction"],
                         bias=r["bias"]))
    return pd.DataFrame(rows)


def part_d(rng, n_sims):
    rows = []
    for c in cfg.M2["contaminated_covariate"]:
        r = _run_cell(rng, 0.6, cfg.M2["true_lift_ab"], 0.0, c, 0.0,
                      "theta_zero", n_sims)
        rows.append(dict(contamination=c,
                         variance_reduction=r["variance_reduction"],
                         estimated_effect=r["effect"],
                         true_effect=cfg.M2["true_lift_ab"],
                         bias=r["bias"],
                         relative_bias=r["bias"] / cfg.M2["true_lift_ab"]))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=N_SIMS)
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    if args.calibrated:
        cfg.require_calibrated("M2")

    rng = np.random.default_rng(cfg.SEED)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)

    a = part_a(rng, args.n_sims)
    b = part_b(rng, args.n_sims)
    c = part_c(rng, args.n_sims)
    d = part_d(rng, args.n_sims)

    for name, df in (("a_rho", a), ("b_coverage", b), ("c_drift", c),
                     ("d_contamination", d)):
        df.to_csv(cfg.RESULTS_DIR / f"m2_{name}.csv", index=False)

    print("\n=== A. Realized vs theoretical variance reduction ===")
    print(a.round(4).to_string(index=False))

    print(f"\n=== B. Coverage: effective reduction at observed rho = "
          f"{cfg.OBSERVED_RHO:.3f} ===")
    print(b.round(5).to_string(index=False))

    print("\n=== C. Pre-period drift ===")
    print(c.round(4).to_string(index=False))

    print("\n=== D. Contaminated covariate (bias, not variance) ===")
    print(d.round(4).to_string(index=False))

    obs = b[(b.no_preperiod_share == round(cfg.NO_PREPERIOD_SHARE, 3)) &
            (b.policy == "theta_zero")]
    if len(obs):
        print(f"\nAt the observed GA4 coverage "
              f"({(1 - cfg.NO_PREPERIOD_SHARE) * 100:.1f}% of users have a "
              f"pre-period), CUPED removes "
              f"{obs.variance_reduction.iloc[0] * 100:.3f}% of variance.")

    _plot(a, b)


def _plot(a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(a.rho, a.theoretical, "k--", lw=1, label=r"theoretical $\rho^2$")
    axes[0].plot(a.rho, a.realized, "o-", label="realized")
    axes[0].axvline(cfg.OBSERVED_RHO, color="C3", ls=":", lw=1.5,
                    label=f"GA4 observed $\\rho$ = {cfg.OBSERVED_RHO:.2f}")
    axes[0].set_xlabel(r"pre/post correlation $\rho$")
    axes[0].set_ylabel("variance reduction")
    axes[0].set_title("CUPED delivers its theory")
    axes[0].legend(frameon=False, fontsize=8)

    sub = b[b.policy == "theta_zero"].sort_values("coverage")
    axes[1].plot(sub.coverage * 100, sub.variance_reduction * 100, "o-",
                 label="realized")
    axes[1].plot(sub.coverage * 100, sub.predicted * 100, "k--", lw=1,
                 label=r"coverage $\times\ \rho^2$")
    axes[1].axvline((1 - cfg.NO_PREPERIOD_SHARE) * 100, color="C3", ls=":",
                    lw=1.5, label="GA4 observed coverage (3.4%)")
    axes[1].set_xlabel("% of users with a pre-period")
    axes[1].set_ylabel("variance reduction (%)")
    axes[1].set_title("...and coverage takes it away")
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "m2_cuped.png", dpi=150)
    print(f"\nwrote {cfg.FIGURES_DIR / 'm2_cuped.png'}")


if __name__ == "__main__":
    main()

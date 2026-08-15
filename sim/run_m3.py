"""M3 — ratio metrics and the delta method.

Three parts:
  A  A/A coverage and Type I error, naive session-level SE vs delta method,
     across the ICC grid at both calibrated metrics
  B  delta method vs user-level bootstrap (agreement check; two correct
     estimators of the same quantity must agree)
  C  the recommendation: the observed SE ratio recovers the design effect
     without estimating an ICC at all

Part B uses fewer replicates than A by design: it is an agreement check, not
a coverage estimate, and the bootstrap is expensive.

Usage:
    python -m sim.run_m3
    python -m sim.run_m3 --calibrated
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

import config as cfg
from sim import dgp, estimators as est

N_SIMS_A = 600
N_SIMS_B = 40
BOOT_REPS = 400
Z = stats.norm.isf((1 - 0.95) / 2)


def _panel(rng, icc, rate, coupling, lift=0.0):
    return dgp.ratio_metric_panel(
        rng,
        n_users_per_arm=cfg.M3["n_users_per_arm"],
        sessions_mean=cfg.SESSIONS_PER_USER_MEAN,
        sessions_dispersion=cfg.SESSIONS_PER_USER_DISPERSION,
        p_control=rate, icc=icc, coupling=coupling, lift=lift,
    )


def _arm_ses(d, arm, rng=None, boot=False):
    s, n = d[f"{arm}_successes"], d[f"{arm}_sessions"]
    out = dict(naive=est.naive_session_se(s, n),
               delta=est.delta_method_se(s, n))
    if boot:
        out["bootstrap"] = est.user_bootstrap_se(s, n, BOOT_REPS, rng)
    return out


def part_a(rng, n_sims):
    rows = []
    for label, meta in cfg.M3["metrics"].items():
        for icc in cfg.M3["icc_grid"]:
            for coupling in cfg.M3["count_rate_coupling"]:
                cov = {"naive": 0, "delta": 0}
                realized_icc, ratios, panel_n0 = [], [], []
                for _ in range(n_sims):
                    d = _panel(rng, icc, meta["rate"], coupling)
                    c = _arm_ses(d, "control")
                    t = _arm_ses(d, "treat")
                    diff = (est.ratio_estimate(d["treat_successes"],
                                               d["treat_sessions"]) -
                            est.ratio_estimate(d["control_successes"],
                                               d["control_sessions"]))
                    for k in ("naive", "delta"):
                        se = np.sqrt(c[k] ** 2 + t[k] ** 2)
                        cov[k] += abs(diff) <= Z * se
                    realized_icc.append(
                        dgp.observed_icc(d["control_successes"],
                                         d["control_sessions"]))
                    ratios.append(c["delta"] / c["naive"])
                    nn = d["control_sessions"].astype(float)
                    panel_n0.append(float((nn ** 2).sum() / nn.sum()))
                rows.append(dict(
                    metric=meta["name"], metric_role=label,
                    requested_icc=icc,
                    realized_icc=float(np.nanmean(realized_icc)),
                    coupling=coupling,
                    coverage_naive=cov["naive"] / n_sims,
                    coverage_delta=cov["delta"] / n_sims,
                    type1_naive=1 - cov["naive"] / n_sims,
                    type1_delta=1 - cov["delta"] / n_sims,
                    se_ratio_delta_over_naive=float(np.mean(ratios)),
                    panel_n0=float(np.mean(panel_n0)),
                ))
    return pd.DataFrame(rows)


def part_b(rng, n_sims):
    rows = []
    meta = cfg.M3["metrics"]["primary"]
    for icc in [0.0, round(cfg.CONTRAST_METRIC_ICC, 4),
                round(cfg.OBSERVED_ICC, 4), 0.30]:
        deltas, boots = [], []
        for _ in range(n_sims):
            d = _panel(rng, icc, meta["rate"], 0.0)
            r = _arm_ses(d, "control", rng=rng, boot=True)
            deltas.append(r["delta"])
            boots.append(r["bootstrap"])
        dm, bm = float(np.mean(deltas)), float(np.mean(boots))
        rows.append(dict(icc=icc, delta_se=dm, bootstrap_se=bm,
                         relative_gap=(bm - dm) / dm))
    return pd.DataFrame(rows)


def part_c(a: pd.DataFrame) -> pd.DataFrame:
    """The recommendation: SE ratio squared recovers the design effect."""
    sub = a[a.coupling == 0.0].copy()
    sub["observed_design_effect"] = sub.se_ratio_delta_over_naive ** 2

    # The design-effect formula needs the SESSION-WEIGHTED mean cluster size,
    # sum(n^2)/sum(n), computed on the sample actually in the experiment.
    # GA4's published n0 = 2.90 is measured on multi-session users only and
    # overstates it for a panel that includes singletons -- which is exactly
    # the trap the SE-ratio approach avoids.
    n0_ga4 = cfg.OBSERVED["mean_cluster_size_n0"]
    sub["formula_correct_n0"] = 1 + (sub.panel_n0 - 1) * sub.realized_icc
    sub["formula_ga4_n0"] = 1 + (n0_ga4 - 1) * sub.realized_icc
    return sub[["metric", "requested_icc", "realized_icc", "panel_n0",
                "observed_design_effect", "formula_correct_n0",
                "formula_ga4_n0"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=N_SIMS_A)
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()
    if args.calibrated:
        cfg.require_calibrated("M3")

    rng = np.random.default_rng(cfg.SEED)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)

    a = part_a(rng, args.n_sims)
    b = part_b(rng, N_SIMS_B)
    c = part_c(a)
    a.to_csv(cfg.RESULTS_DIR / "m3_coverage.csv", index=False)
    b.to_csv(cfg.RESULTS_DIR / "m3_bootstrap.csv", index=False)
    c.to_csv(cfg.RESULTS_DIR / "m3_design_effect.csv", index=False)

    print("\n=== A. A/A coverage, nominal 95% (no count-rate coupling) ===")
    show = a[a.coupling == 0.0][["metric", "requested_icc", "realized_icc",
                                 "coverage_naive", "coverage_delta",
                                 "type1_naive"]]
    print(show.round(4).to_string(index=False))

    print("\n=== A2. Effect of count-rate coupling on naive coverage ===")
    piv = a.pivot_table(index=["metric", "requested_icc"], columns="coupling",
                        values="coverage_naive")
    print(piv.round(4).to_string())

    print("\n=== B. Delta method vs user-level bootstrap ===")
    print(b.round(6).to_string(index=False))

    print("\n=== C. Observed SE ratio recovers the design effect ===")
    print(c.round(4).to_string(index=False))

    _plot(a)


def _plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sub = a[a.coupling == 0.0]

    for (name, grp), color in zip(sub.groupby("metric"), ("C0", "C1")):
        g = grp.sort_values("realized_icc")
        ax.plot(g.realized_icc, g.coverage_naive, "o-", color=color,
                label=f"{name} — naive session SE")
        ax.plot(g.realized_icc, g.coverage_delta, "s--", color=color,
                alpha=0.55, label=f"{name} — delta method")

    ax.axhline(0.95, ls="--", lw=1, color="k", label="nominal 95%")
    for icc, lbl in ((cfg.OBSERVED_ICC, "add_to_cart"),
                     (cfg.CONTRAST_METRIC_ICC, "purchase")):
        ax.axvline(icc, color="C3", ls=":", lw=1.2)
        ax.text(icc, 0.775, f" GA4 {lbl}", rotation=90, fontsize=7, color="C3")

    ax.set_xlabel("realized intraclass correlation")
    ax.set_ylabel("A/A coverage of the 95% CI")
    ax.set_title("Naive session-level SE loses coverage as sessions cluster")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(cfg.FIGURES_DIR / "m3_coverage.png", dpi=150)
    print(f"\nwrote {cfg.FIGURES_DIR / 'm3_coverage.png'}")


if __name__ == "__main__":
    main()

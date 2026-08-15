"""M1 — peeking inflates Type I error.

Runs A/A tests under five peeking schedules, uncorrected and under two
corrections, and reports the empirical false-positive rate for each. Then
repeats under a true lift to price the power each correction costs.

Usage:
    python -m sim.run_m1
    python -m sim.run_m1 --calibrated
    python -m sim.run_m1 --n-sims 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from sim import dgp, estimators as est


def _peek_days(schedule: list[float]) -> np.ndarray:
    """Map a schedule to integer day indices (0-based) into the daily grid.

    Twice-daily is approximated by peeking on each half-day; because the DGP
    is generated at daily granularity, the two looks within a day see the same
    data, so the schedule is collapsed to its distinct daily looks. Documented
    rather than silently dropped: the honest reading is that this arm is a
    daily schedule, and the twice-daily row is reported as such.
    """
    days = sorted({int(np.ceil(d)) for d in schedule})
    return np.array(days, dtype=int) - 1


def run_arm(rng: np.random.Generator, n_sims: int, lift: float) -> pd.DataFrame:
    m1 = cfg.M1
    n_days = m1["horizon_days"]
    users_per_day = int(cfg.DAILY_USERS // 2)

    conv, expo = dgp.daily_conversion_stream(
        rng=rng,
        n_sims=n_sims,
        n_days=n_days,
        users_per_day_per_arm=users_per_day,
        p_control=cfg.BASELINE_CONVERSION,
        lift=lift,
    )

    rows = []
    for label, schedule in m1["schedules"].items():
        looks = _peek_days(schedule)
        n_looks = len(looks)

        c = conv[:, looks, 0].astype(float)
        t = conv[:, looks, 1].astype(float)
        n = expo[looks].astype(float)[None, :]

        z = est.two_proportion_z(c, t, n)
        p = est.two_sided_p(z)

        # Uncorrected: fixed alpha at every look.
        thresh_unc = np.full(n_looks, m1["alpha"])
        rej_unc = est.first_crossing(p, thresh_unc)

        # O'Brien-Fleming alpha spending over the same looks.
        info = n.ravel() / n.ravel()[-1]
        thresh_obf = est.obrien_fleming_alphas(info, alpha=m1["alpha"])
        rej_obf = est.first_crossing(p, thresh_obf)

        arms = [("uncorrected", rej_unc, np.nan),
                ("obrien_fleming", rej_obf, np.nan)]

        # mSPRT always-valid p-values, swept over the mixture SD.
        for frac in m1["msprt_tau_frac_grid"]:
            tau = frac * cfg.BASELINE_CONVERSION
            p_av = est.msprt_always_valid_p(c, t, n, tau=tau)
            rej = est.first_crossing(p_av, np.full(n_looks, m1["alpha"]))
            name = "msprt" if frac == m1["msprt_tau_frac"] \
                else f"msprt_tau{frac:g}"
            arms.append((name, rej, frac))

        for arm, rej, frac in arms:
            rate = float(rej.mean())
            se = float(np.sqrt(rate * (1 - rate) / n_sims))
            rows.append(dict(
                schedule=label,
                n_looks=n_looks,
                correction=arm,
                msprt_tau_frac=frac,
                lift=lift,
                rejection_rate=rate,
                mc_se=se,
            ))

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=cfg.N_SIMS)
    ap.add_argument("--calibrated", action="store_true")
    args = ap.parse_args()

    if args.calibrated:
        cfg.require_calibrated("M1")

    rng = np.random.default_rng(cfg.SEED)

    aa = run_arm(rng, args.n_sims, lift=0.0)
    ab = run_arm(rng, args.n_sims, lift=0.10)

    out = pd.concat([aa, ab], ignore_index=True)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(cfg.RESULTS_DIR / "m1_peeking.csv", index=False)

    meta = dict(
        n_sims=args.n_sims,
        calibrated=bool(cfg.CALIBRATED_MODE and args.calibrated),
        calibration_source=cfg.OBSERVED.get("source"),
        baseline_conversion=cfg.BASELINE_CONVERSION,
        daily_users=cfg.DAILY_USERS,
        seed=cfg.SEED,
    )
    (cfg.RESULTS_DIR / "m1_meta.json").write_text(json.dumps(meta, indent=2))

    headline = ["uncorrected", "obrien_fleming", "msprt"]

    def show(df, title):
        piv = (df[df.correction.isin(headline)]
               .pivot(index="schedule", columns="correction",
                      values="rejection_rate")
               .reindex(list(cfg.M1["schedules"]))[headline])
        print(f"\n=== {title} ===")
        print(piv.round(4).to_string())

    show(aa, "A/A: false-positive rate (nominal alpha = 0.05)")
    show(ab, "A/B (10% relative lift): power")

    print("\n=== mSPRT sensitivity to the mixture SD (daily peeking) ===")
    sweep = pd.concat([aa, ab])
    sweep = sweep[(sweep.schedule == "daily") & sweep.msprt_tau_frac.notna()]
    piv3 = sweep.pivot(index="msprt_tau_frac", columns="lift",
                       values="rejection_rate")
    piv3.columns = ["A/A false-positive rate", "A/B power"]
    print(piv3.round(4).to_string())

    _plot(aa)


def _plot(aa: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg.FIGURES_DIR.mkdir(exist_ok=True)
    order = [s for s in cfg.M1["schedules"]]
    labels = {"uncorrected": "Uncorrected",
              "obrien_fleming": "O'Brien-Fleming spending",
              "msprt": "mSPRT (always valid)"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    for arm, marker in (("uncorrected", "o"),
                        ("obrien_fleming", "s"),
                        ("msprt", "^")):
        sub = aa[aa.correction == arm].set_index("schedule").reindex(order)
        ax.plot(x, sub.rejection_rate, marker=marker, label=labels[arm])

    ax.axhline(cfg.M1["alpha"], ls="--", lw=1, color="k",
               label="Nominal alpha = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in order], rotation=20)
    ax.set_ylabel("Empirical false-positive rate")
    ax.set_title("A/A false-positive rate by peeking frequency")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cfg.FIGURES_DIR / "m1_peeking_fpr.png", dpi=150)
    print(f"\nwrote {cfg.FIGURES_DIR / 'm1_peeking_fpr.png'}")


if __name__ == "__main__":
    main()

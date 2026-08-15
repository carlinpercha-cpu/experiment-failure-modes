# PREREG — M1: Peeking inflates Type I error

**Registered:** 2026-08-15
**Status:** locked prior to first run
**Spec:** `config.M1`

## Question

By how much does checking an experiment repeatedly and stopping at
significance inflate the false-positive rate, and do the two standard fixes
restore nominal α — at what cost in power?

## Design

A/A tests (no true effect) and A/B tests (10% relative lift on the control
rate), two arms, daily arrivals, 14-day horizon. Baseline conversion and daily
traffic come from `calibration/observed_params.json`. 10,000 simulations per
cell, seed 20260815.

Five peeking schedules: fixed horizon, weekly, every three days, daily, twice
daily. Three decision rules: uncorrected α=0.05 at every look; O'Brien–Fleming
alpha spending; mSPRT always-valid p-values with mixture SD equal to 5% of the
baseline rate.

## Registered predictions

| # | Prediction | Direction |
|---|---|---|
| P1 | Fixed horizon, uncorrected, recovers nominal α | 0.043 – 0.058 |
| P2 | FPR rises monotonically in the number of looks | monotone increasing |
| P3 | Daily peeking over 14 looks lands materially above nominal | > 0.12 |
| P4 | O'Brien–Fleming holds FPR at or below nominal at every schedule | ≤ 0.05 |
| P5 | mSPRT holds FPR at or below nominal at every schedule | ≤ 0.05 |
| P6 | Both corrections cost power relative to a fixed horizon at equal N | power lower |

## Bug criteria (not findings)

This is a simulation with known ground truth. The following are defects in the
implementation and must be fixed rather than reported:

- P1 fails → the base test or the DGP is wrong; nothing downstream is
  interpretable.
- P4 or P5 fails → the correction is misimplemented. A correction that does
  not control α is a bug, not a result about the method.
- mSPRT p-values rise at any look → the running minimum is broken.
- O'Brien–Fleming per-look alphas do not sum to the total budget → the
  spending function is misparameterised.

Each is enforced in `tests/test_estimators.py`.

## Known approximations, registered in advance

1. **O'Brien–Fleming boundaries are approximate.** Per-look budgets are taken
   as increments of the Lan–DeMets spending function, ignoring the positive
   correlation between successive looks. The realized family-wise error is
   therefore *below* nominal: this arm is conservative, not exact. The exact
   recursive boundary requires numerical multivariate-normal integration and is
   out of scope. Reported as a limitation, not corrected silently.

2. **Twice-daily collapses to daily.** The DGP generates data at daily
   granularity, so two looks within one day see identical data. The twice-daily
   row is reported with its true number of *distinct* looks. It is retained in
   the grid because the collapse is itself informative: peeking more often than
   your data refreshes buys nothing and costs nothing.

3. **The mSPRT mixture SD is a tuning choice.** τ = 5% of baseline. Too small
   destroys power, too large destroys early stopping. No sweep is registered;
   sensitivity is noted as a limitation.

## Reported regardless of outcome

The full FPR table for all fifteen cells, the power table, and the
FPR-vs-peek-frequency figure. If any registered prediction fails and the cause
is *not* traced to a bug, it is reported as a failure with the diagnosis.

---

## Amendment 1 — 2026-08-15, after first run

**What happened.** The registered mSPRT mixture SD (τ = 5% of baseline, i.e.
τ = 0.001 at a 2% baseline rate) produced a 0.000 false-positive rate and 0.012
power under daily peeking. The arm never fired in either direction, so it
carried no information about whether mSPRT controls α.

**Diagnosis.** At a low baseline rate the always-valid statistic is dominated
by τ, which sets the effect size the mixture is tuned for. τ = 0.001 against a
true absolute effect of 0.002 is tuned for an effect half the size of the one
present, and the always-valid guarantee then costs essentially all remaining
power. This is a property of the method's parameterisation, not a coding error:
`tests/test_estimators.py` confirms the always-valid p-values are bounded and
monotone.

**Change.** `msprt_tau_frac` replaced by `msprt_tau_frac_grid` =
[0.05, 0.10, 0.20, 0.50, 1.00, 2.00], with 0.20 designated the headline value
on the pilot sweep. The originally registered 0.05 is retained in the grid and
an import-time assertion in `config.py` prevents its removal, so the amendment
stays auditable rather than becoming a silent retune.

**What did not change.** No prediction was altered. P5 (mSPRT holds FPR at or
below nominal) held at every τ tested — maximum observed 0.0120. The reason for
the change was uninformativeness, not an unwelcome result.

**Reported.** The full τ sweep appears in the results, alongside the headline
table, so a reader can see the sensitivity rather than only the tuned value.

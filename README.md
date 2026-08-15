# experiment-failure-modes

Simulation study of when standard A/B test analysis breaks, and by how much.
Ground truth is known by construction in every module, so the error is measured
rather than argued about.

Each module is pre-registered before it runs — hypotheses, parameter grid, and
the conditions under which a deviation is a **bug in my code** rather than a
finding about the method. In a simulation study that distinction is the whole
game, and it is enforced by a test suite rather than by good intentions.

---

## Findings

**Daily peeking at α = 0.05 produced a 22.0% false-positive rate against a
nominal 5%.** Over a 14-day horizon with 14 looks, a rule of "check daily, stop
when significant" rejected a true null in more than one A/A test in five. A
single look at the horizon recovered nominal α exactly (4.8%).

**Uncorrected peeking looks like it buys power, and does not.** Against a true
10% lift, daily peeking "detected" the effect 55.1% of the time versus 37.9%
for a fixed horizon. That 17-point gain is bought with a 17-point increase in
false positives. It is the same rejections, not more true ones.

**Both corrections work, and both are expensive.** O'Brien–Fleming alpha
spending held the false-positive rate at or below nominal under every schedule
(1.8%–4.8%) at a cost of 45% of the power under daily peeking (0.38 → 0.21).
mSPRT always-valid p-values were far more conservative still — never above 1.1%
false positives, but only 13.2% power against the fixed horizon's 37.9%.

**The mSPRT result is dominated by one tuning parameter, and I got it wrong on
the first pass.** The originally registered mixture SD (5% of baseline) gave a
0.0% false-positive rate and 1.2% power — an arm that never fires. Sweeping it
showed power peaks near 20% of baseline and falls off in both directions, while
α control holds everywhere. The amendment is recorded in `CHANGELOG_CONFIG` and
`PREREG_M1_PEEKING.md` rather than folded in silently, because "we retuned
until it worked" and "the method is robust" are different claims.

![False-positive rate by peeking frequency](figures/m1_peeking_fpr.png)

| Schedule | Looks | Uncorrected | O'Brien–Fleming | mSPRT |
|---|---|---|---|---|
| Fixed horizon | 1 | 0.048 | 0.048 | 0.002 |
| Weekly | 2 | 0.077 | 0.043 | 0.004 |
| Every 3 days | 5 | 0.138 | 0.029 | 0.007 |
| Daily | 14 | 0.220 | 0.018 | 0.011 |

*A/A false-positive rates, nominal α = 0.05, 10,000 simulations per cell.*

---

## Modules

| | Module | Status |
|---|---|---|
| M1 | Peeking and sequential correction | complete |
| M2 | CUPED variance reduction and its break conditions | scaffolded |
| M3 | Ratio metrics, delta method, user-level bootstrap | scaffolded |
| M4 | Interference and switchback designs | not started (optional) |
| M5 | Dilution and triggered analysis as a collider problem | not started (optional) |

M1–M3 are the complete repo. M4 and M5 ship only if M1–M3 are finished first.

---

## Calibration

Simulation parameters that could be invented are instead measured on the
Google Analytics 4 public sample in BigQuery
(`bigquery-public-data.ga4_obfuscated_sample_ecommerce`): baseline session
conversion, daily users, sessions per user and their dispersion, the
intraclass correlation of session conversion, and the pre/post user
correlation that determines whether CUPED is worth anything.

Queries are in `calibration/ga4_params.sql`. `config.py` refuses to run in
calibrated mode against placeholder values, so no figure can be captioned as
calibrated unless it is.

**The results above are on placeholder parameters** (2% session conversion,
4,000 daily users) and will be re-run once the BigQuery pass lands.

---

## Layout

```
config.py                  locked spec + CHANGELOG_CONFIG; runners define no grids
PREREG_M1_PEEKING.md       hypotheses, bug criteria, registered limitations
PREREG_M2_CUPED.md
PREREG_M3_RATIO.md
calibration/               GA4 queries -> observed_params.json
sim/dgp.py                 data-generating processes
sim/estimators.py          estimators under test
sim/run_m1.py              M1 runner
tests/                     estimator correctness against known truth
results/, figures/         outputs
```

```bash
pip install -r requirements.txt
pytest -q                  # 15 tests, all estimator-correctness checks
python -m sim.run_m1
```

---

## Limitations

- Simulation, not production data. Effect sizes are chosen, not observed.
- O'Brien–Fleming boundaries are the Lan–DeMets spending-function increments,
  which ignore correlation across looks. That arm is conservative rather than
  exact; the exact recursive boundary needs numerical multivariate-normal
  integration and is out of scope. Registered in advance, not discovered after.
- Twice-daily peeking collapses to daily because the DGP is generated at daily
  granularity. Reported with its true number of distinct looks.
- M3's metric is a low-rate binary conversion, so its results are a *lower
  bound* on how badly the naive standard error fails; a heavy-tailed revenue
  metric would be worse.
- The mSPRT power figures depend on a mixture SD with no principled default.
  The sweep is reported rather than a single tuned value.

## Related

- `mortgage-performance-sql` — competing-risks and delinquency analysis on
  Freddie Mac loan-level data
- `ga4-product-analytics` — funnel, cohort, and MDE analysis on the same GA4
  sample used to calibrate this repo

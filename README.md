# RAILCAST — working prototype

Problem Statement **SIH26028** · Dynamic Forecast of Expected Time of Arrival (ETA)
for Coaching Trains · Ministry of Railways.

This is a runnable implementation of the forecast loop the deck describes, on a
simulated New Delhi – Mumbai Central corridor. It produces the numbers, tables
and charts that go into the deck.

```bash
python run_prototype.py                # full run, ~450 simulated operating days
python run_prototype.py --quick        # 60 days, ~90 seconds, for iterating
```

Everything lands in `outputs/`:

| Path | What it is |
|---|---|
| `outputs/RESULTS.md` | Every headline number, written out ready to paste into slides |
| `outputs/charts/light/` | 15 charts on white — for print and light slides |
| `outputs/charts/dark/` | The same 15 on near-black — for dark slides |
| `outputs/data/*.csv` | The tables behind every chart |
| `outputs/data/metrics_summary.json` | Machine-readable summary of the whole run |

## What it actually does

The five stages in the deck's technical slide, each implemented:

| Stage | Deck | Code |
|---|---|---|
| — | Ground truth | `railcast/simulator.py` — event-driven movement over the block graph |
| A | Free-run time model | `models.fit_free_run` — LightGBM on observed unimpeded section runs |
| B | Conflict-resolution simulator | `simulator.forward_replay` — every train advanced over the block graph until it conflicts, precedence resolved by priority rules |
| C | Learned residual, direct multi-horizon | `models.ResidualModel` — LightGBM quantile loss; the horizon is a feature, so a 30-hour forecast is made in one shot and never chained |
| D | Calibrated window | `models.MondrianConformal` — conformalised quantile regression, calibrated per horizon bucket × zone × time of day |
| — | Decision surfaces | `railcast/analysis.py` — connection risk, costed precedence, serving cost |

The corridor is 35 stations and 1386 km, split into ~170 block sections, with
32 trains in five priority classes running both directions. The simulator
generates delay from causes the forecaster cannot see directly: run-time
friction correlated along a journey, winter fog on NR/NCR, monsoon rain on WR,
temporary speed restrictions, incidents, block-occupancy headway, and trains
looped for faster ones behind them.

## How it is evaluated

- **Temporal holdout.** Trained on the earliest ~70% of operating days,
  calibrated on the next 15%, scored on the last 15%. A random split would leak
  and flatter every number.
- **A stated baseline.** Carry-forward — the current delay repeated onto the
  timetable — computed on exactly the same rows. Both the plain and the
  padding-absorbing variant are computed; the one that scores better on the
  training days is used, so the baseline is not a straw man.
- **Error broken out by horizon**, not pooled. Carry-forward is perfectly good
  for the next station; it is the multi-hour horizon where it fails, and that is
  where the comparison is made honestly visible.
- **Coverage audited** per zone, per horizon, per time of day and per month
  against the nominal 80%.

## The honesty line

**Every number here comes from a simulator I wrote. No Indian Railways data was
used, downloaded, or consulted.** Station names, km posts and train numbers are
written from general knowledge of the route and are approximate. Timetables,
running times, delays, conflicts and weather are all generated.

The simulator's parameters were set by hand until corridor punctuality looked
plausible - roughly half of journeys arriving within 15 minutes, median about 15
minutes late - but they were **not fitted to published punctuality returns**.
That calibration is step 2 of the roadmap and has not been done.

What the results do show is that the *method* works on a problem with the right
shape: the forecaster never sees the noise that generates the delay, the
baseline is computed on exactly the same rows, and the split is temporal. What
they do not show is field performance. A simulator validating a method is not a
trial.

Every chart carries that caption in its footer. Keep it there. The deck already
frames its numbers as "targets we will measure against, not results claimed",
and that framing is worth more in the room than an unqualified number.

## Charts, and where each one belongs in the deck

| File | Slide it supports |
|---|---|
| `01_mae_by_horizon` | Impact — the ≥30% error-cut claim, broken out by horizon |
| `02_error_vs_lead` | Impact / Technical — where carry-forward starts to fail |
| `03_coverage_heatmap` | Technical — "coverage checked per horizon, zone and time of day" |
| `04_coverage_by_month` | Feasibility — the "coverage fails in the monsoon" risk, retired |
| `05_window_vs_lead` | Title slide — "it widens with distance" |
| `06_journey_forecast` | Title slide — the product in one picture |
| `07_window_closing` | Title slide — "narrows as the train runs" |
| `08_cascade_marey` | Problem — "cascades are invisible" |
| `09_precedence_cost` | Innovation — costed precedence, in network-minutes |
| `10_connection_risk` | Innovation / Impact — connection-make probability, and proof it is honest |
| `11_calibration_effect` | Technical — why calibration is a stage and not a footnote |
| `12_feature_importance` | Technical — attribution, what moves a forecast |
| `13_error_distribution` | Impact — error shape, and the corridor being learned on |
| `14_latency_scaling` | Feasibility — the 30 s re-forecast and national scale claim |
| `15_console_card` | Title / Technical — the passenger-facing card, real numbers in it |

## Layout

```
railcast/
  corridor.py    station list, km posts, zones, block sections, free-run physics
  trains.py      roster, priority classes, timetable construction with padding
  noise.py       pre-drawn stochastic terms, so a day is reproducible
  simulator.py   the event-driven block-graph simulator (truth and replay)
  dataset.py     snapshot the network, build one row per (train, time, station)
  models.py      stages A, C and D
  evaluate.py    scoring against the baseline, coverage audit
  analysis.py    precedence experiment, connection risk, latency benchmark
  charts.py      all 15 figures, light and dark
run_prototype.py end to end
```

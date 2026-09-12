# RAILCAST — working prototype

Problem Statement **SIH26028** · Dynamic Forecast of Expected Time of Arrival (ETA)
for Coaching Trains · Ministry of Railways.

This is a runnable implementation of the forecast loop the deck describes, on a
simulated New Delhi – Mumbai Central corridor. It produces the numbers, tables
and charts that go into the deck.

```bash
python -m railcast.feeds fetch         # real timetable data, ~99 MB, once
python run_prototype.py                # real corridor, real weather
python run_prototype.py --quick        # 60 days, for iterating
python run_prototype.py --source synthetic   # no network at all
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

## Where the data comes from

`--source real` (the default) builds the corridor from published data. Nothing
about the route, the roster or the weather is invented any more.

| Input | Source | Licence |
|---|---|---|
| 8,990 stations - code, name, coordinates | [datameet/railways](https://github.com/datameet/railways) | CC0 |
| 5,208 trains - number, name, type, route distance | same | CC0 |
| 417,080 scheduled calls - the working timetable | same | CC0 |
| Observed weather, every day, per corridor segment | [Open-Meteo](https://open-meteo.com) ERA5 reanalysis | CC BY 4.0 |
| Live running position | `RailRadarFeed` (needs a key) or the IR feed | - |

Three quantities that were guesses are now measured from the timetable:

- **km posts** - great-circle distance between real station coordinates, scaled
  so the total matches the route distance Indian Railways publishes. For
  12951 that is 1384 km, and the built corridor reproduces it exactly.
- **Sectional speed** - the speed implied by the fastest booked run over each
  section, so the limit is what the timetable actually allows.
- **Recovery padding** - booked run time minus free-run time, per section, per
  train. Previously a class constant; the real spread runs from 3.7% on a
  Rajdhani to 31% on a Janata Express, which is most of why the two behave
  nothing alike.

Weather is real and the seasonal pattern falls out of it rather than being coded
in: January fog costs up to 17% of section speed on Delhi-Kota and nothing at
all on the coast, while rain peaks on Surat-Mumbai in July.

## What is still not real

**There is no public archive of past Indian Railways arrivals.** NTES serves
current and near-future state only, and nobody publishes the history. So the
part of the simulator that *generates* delay cannot be fitted from published
data:

- run-time friction and its spread across days
- incident rate and severity
- block clearance time and how hard a controller clears a premier train's path
- temporary speed restrictions and engineering blocks

Every one of those lives in `railcast/simconfig.py`, in one object, so they can
be listed rather than buried. `SimConfig.provenance()` reports whether they were
fitted and against what, and that provenance is written into
`metrics_summary.json` on every run.

Two ways to fit them:

```bash
# collect running data - one call captures a whole journey so far
setx RAILRADAR_API_KEY "rr_..."            # railradar.in, free tier: 1,000/month
python -m railcast.feeds collect --trains 12951,12952,12953,12903,12904,12471

# check the simulator against what was actually collected
python -m railcast.feeds validate

# fit the delay parameters
python -m railcast.feeds calibrate --target-file my_target.json
```

`validate` is the test the project was missing: everything upstream can be
checked against a published source, but how delay arises could not be checked at
all until there were real arrivals to compare with. It compares station-level
delay on both sides - comparing mid-run delay against end-of-run delay would
flatter the simulator.

The first 241 observed arrivals say the simulator is wrong in a specific way:

| | observed | simulated |
|---|---|---|
| median delay | 16 min | 4 min |
| P90 | 57 min | 109 min |
| P99 | 65 min | 287 min |
| within 15 min | 47% | 64% |

Real delay is **tighter and later** than the model. The simulator has too many
trains running to time and too many running catastrophically late - bimodal
where the real distribution is unimodal.

`calibrate --from-observed` fits the delay parameters to that distribution
rather than to a stated assumption, matching station-level delay on both sides.
It halves the loss and it does not close the gap:

| | observed | simulated, after fitting |
|---|---|---|
| p25 | 6 min | -1 min |
| p50 | 14 min | 7 min |
| p75 | 23 min | 33 min |
| p90 | 56 min | 82 min |
| p95 | 61 min | 130 min |

The fit drives `max_hold_min` and `precedence_lookahead_min` to the bottom of
their ranges trying to kill the tail, and still cannot. **That is a structural
result, not a tuning one**: no setting of these five parameters reproduces the
real shape. Real delay is concentrated - most trains a little late, few
disastrously so - while the simulator spreads out, because conflicts either miss
a train entirely or cascade into it. The honest conclusion is that the conflict
model, not its parameters, is what needs work next.

Note that live responses mark some stations `upcoming` while still carrying an
"actual" time. That is the operator's own projection, not an observation.
`LiveRecord.observed` excludes them; fitting on them would be training on
another system's forecast and calling it ground truth. They are kept separately
as `is_incumbent_eta`, because the deployed system's ETA is the baseline worth
beating.

The shipped fit is against a **stated** target, not a measurement: roughly 75%
of Rajdhani journeys arriving within 15 minutes, down to 50% for passenger
services. That target is an assumption. The fit reaches it to within about 10
points for Rajdhani and Express and undershoots Superfast by 23 - those
residuals are recorded in the config rather than smoothed over.

## The honesty line

The corridor, the roster, the timetable, the padding and the weather are real.
**How delay arises is still modelled, not measured**, and the forecasting
results depend on it. What the numbers establish is that the method works on a
problem with the right shape: the forecaster never sees the noise that generates
the delay, the baseline is computed on exactly the same rows, and the split is
temporal. They are not a measurement of field performance, and the first real
running data collected will change them.

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
  feeds/
    base.py      the adapter interfaces, caching and HTTP
    datameet.py  real stations, trains and the working timetable
    openmeteo.py real observed weather, and the fog index derived from it
    corridor.py  build the block graph and roster from a feed
    live.py      live position adapters, the collector and the local store
    __main__.py  fetch / status / corridor / collect / calibrate
  simconfig.py   every parameter that is not measured, in one object
  calibrate.py   fit those parameters to data or to a stated target
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

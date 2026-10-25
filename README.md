# RAILCAST — get there by train, even when the direct one is full

For a trip in the next 48 hours, Railcast finds ways from A to B with up to two
changes — including change stations **off** the direct line, on other corridors
— and tells you the chance of actually making each connection. Options come back
grouped **Safe / Likely / Risky**, with three picks and a plan for what to book
first. It plans; it does not book.

The product is described in [`docs/PRD.md`](docs/PRD.md). The SIH26028 ETA
prototype this grew out of — deck, 15 charts, videos — is on the
[`sih26028`](https://github.com/TypeError-4-ever/Railcast/tree/sih26028) branch.

## Setup

```bash
brew install libomp                      # LightGBM needs it on macOS
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m railcast.feeds fetch  # the timetable, ~99 MB, once
```

## Ask it a question

```bash
python -m railcast plan JUC PNBE --date 2026-10-27 --class SL
python -m railcast plan CSMT PRYJ --class 3A --max-changes 2 --chart
python -m railcast plan NDLS HWH --after 18:00 --json
```

Stations can be today's codes (PRYJ, CSMT, KSR), the old ones in the timetable
snapshot, or a name prefix. Other stations in the same city count unless you pass
`--exact`. The response is printed, saved as JSON to `outputs/plans/`, and with
`--chart` drawn as one picture: each option's chance of reaching B, and the
weakest connection among the picks.

```
* Safest  -  SAFE, >99% chance you make every connection (seats not checked)
  Tue 07:20 JUC -> Wed 06:05 PNBE on the timetable; predicted arrival Wed 06:36 (80%: 06:08-07:38)
  22.8 h, direct, 1,265 km
  1. 12318 AKAL TAKHT Express           JUC Tue 07:20 -> PNBE Wed 06:05
  Book: 12318 (general quota open; tatkal opens Mon 26 Oct 11:00)

SAFE (21)
   >99%  Tue 01:05 -> Wed 02:09  24.5 h  18508 + 12332 via LDH
    99%  Tue 08:40 -> Wed 07:12  22.2 h  15708 + 12394 via DLI  [other corridor]
```

## How an answer is made

| Step | Module | What it does |
|---|---|---|
| Network | `planner/network.py` | The datameet timetable as a graph. Repairs pass-through stops, backwards times and renamed codes |
| Search | `planner/search.py` | 0–2 changes, change stations anywhere; detour caps; drops dominated and intra-city options |
| Delays | `planner/delays.py` | Delay at each station as 10th/50th/90th percentiles |
| Risk | `planner/risk.py` | P(make each connection) by Monte Carlo; P(reach B) = seats × connections |
| Seats | `planner/seats.py` | RailKit seat and fare checks for the top 10 options; "not checked" without a key |
| Rank | `planner/rank.py` | Safe ≥ 90%, Likely 70–90%, Risky below; Safest / Fastest / Cheapest |
| Booking | `planner/booking.py` | Tatkal (day before the train's start date, 10:00 AC / 11:00 SL), chart 8 h before departure; scarcest leg first |
| Response | `planner/respond.py` | Text, JSON, and the caveats every answer carries |

Thresholds, transfer times and booking rules are in `railcast/config.py`.

## Training on data

The planner's delay model is the part that gets better with data.

```bash
python -m railcast train-delays --source sim    # simulated runs - tests the pipeline
python -m railcast train-delays --source live   # collected running data - the real model
```

Both build the same rows (one per train, run date, station and event) and go
through one training path. Until a model is trained on **real** data, the
planner uses a stated prior rule scaled to the 241 arrivals observed so far, and
says so in every response. The simulated model is saved separately and used only
with `plan --delays sim`: the simulator's hand-set rules decide how each class of
train behaves, so it is not a national delay model.

Collecting the real data:

```bash
export RAILRADAR_API_KEY=rr_live_...           # railradar.in
python -m railcast.feeds collect --trains 12951,12952,12318,12332
python -m railcast.feeds status                 # what has been collected
```

Run `collect` on a schedule from a machine that stays on. Once there are a few
thousand observed station events, `train-delays --source live` becomes the
default model.

## Seats and fares

```bash
export RAILKIT_API_KEY=...                      # railkit.in dashboard
export RAILKIT_BASE_URL=https://...             # from the dashboard; not in the public docs
export RAILKIT_AUTH_HEADER=x-api-key            # only if yours differs
```

Seats are checked for the ten most promising options only (each check is an API
call). Every response is logged to `data/availability/` — the history a future
"will this leg have seats" model will be trained on. RailKit is a third-party
service; its live routes relay NTES / Where Is My Train data.

## The corridor forecaster

`railcast/forecast/` is the model built for SIH: a conflict simulator over the
block graph, a learned residual and a calibrated window, forecasting a running
train's arrival at every remaining station. It needs live positions, so the
planner will hand running trains to it for travel-day rerouting.

```bash
python -m railcast evaluate            # ~3 min; --quick for a smoke run
```

writes `outputs/RESULTS.md` and three charts: accuracy by horizon against
carry-forward, whether the connection probability can be trusted, and one
journey forecast. On the real NDLS–BCT corridor: 60% lower error than
carry-forward, 79% coverage of the 80% window, connection reliability error
0.008 — **scored on simulated delays**.

## What is not real yet

- **The timetable is an old snapshot.** New trains, retimings and cancellations
  are missing, and it has no running days — every train is treated as daily.
- **The planner's delay model is a prior rule** until real running data is
  collected and trained on.
- **Seats are not checked** without a RailKit key, so groups then reflect
  connection risk only.
- **How delay arises in the simulator is modelled, not measured.** The first
  241 observed arrivals say it is wrong in a specific way:

  | | observed | simulated |
  |---|---|---|
  | median delay | 16 min | 4 min |
  | P90 | 57 min | 109 min |
  | within 15 min | 47% | 64% |

  Real delay is tighter and later; the simulator has too many trains on time
  and too many catastrophically late. `calibrate --from-observed` halves the
  gap but cannot close it — the conflict model, not its parameters, is what
  needs work. Parameters live in `railcast/forecast/simconfig.py`.

## Layout

```
railcast/
  __main__.py      plan | train-delays | evaluate
  config.py        thresholds, transfer times, booking rules
  dist.py          distributions from three quantiles
  planner/         the trip planner (table above), plus train_delays.py and chart.py
  forecast/        the corridor forecaster: simulator, stages A-D, pipeline, 3 charts
  feeds/           datameet timetable, Open-Meteo weather, live collection
tests/             planner tests on a hand-built network - no download needed
docs/PRD.md        what the product is and why
```

"""
Train the planner's delay model.

    python -m railcast train-delays --source sim     simulated runs on the real corridor
    python -m railcast train-delays --source live    running data the collector gathered

Both sources become the same rows - one per train, run date, station and event -
and go through one training path, so switching to real data later changes
nothing but the flag. The model is saved to models/delays.pkl and the planner
picks it up from there.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import delays

MIN_LIVE_ROWS = 2000


def _row(run_day, klass, zone, sched, origin_dep, month, dow, halt, event, delay):
    k, z = delays.encode(klass, zone)
    return (run_day, k, (sched - origin_dep) / 60.0, z, int(sched // 60) % 24,
            month, dow, int(halt), event, float(delay))


COLUMNS = ["run_day", "klass", "hours_into_run", "zone", "hour", "month", "dow",
           "halt", "event", "delay"]


def records_from_sim(days: int, start_date: str, reference: str,
                     max_trains: int) -> pd.DataFrame:
    """Every booked halt of every simulated run, arrival and departure."""
    from ..forecast.pipeline import build_world
    import time

    args = argparse.Namespace(source="real", reference=reference,
                              max_trains=max_trains, start_date=start_date)
    world, _ = build_world(args, days, np.random.default_rng(29), time.time())
    world.simulate()
    rows = []
    for d, (env, truth) in world.days.items():
        month, dow = world.calendar(d)
        for tid, run in truth.items():
            t = world.sim.trains[tid]
            for s in {t.origin, t.dest, *t.halts}:
                zone = world.cor.stations[s].zone
                if s != t.origin and s in run.arr and s in t.sched_arr:
                    rows.append(_row(d, t.klass, zone, t.sched_arr[s], t.dep_minute,
                                     month, dow, True, 0, run.arr[s] - t.sched_arr[s]))
                if s != t.dest and s in run.dep and s in t.sched_dep:
                    rows.append(_row(d, t.klass, zone, t.sched_dep[s], t.dep_minute,
                                     month, dow, True, 1, run.dep[s] - t.sched_dep[s]))
    return pd.DataFrame(rows, columns=COLUMNS)


def records_from_live(store, net) -> pd.DataFrame:
    """Observed station events from the live store. Projections are skipped."""
    by_run: dict[tuple, list] = {}
    for r in store.read():
        if r.observed and r.start_date:
            by_run.setdefault((r.train_number, r.start_date), []).append(r)
    rows = []
    for (num, start), recs in by_run.items():
        origin_dep = min(r.scheduled for r in recs)
        info = net.trains.get(num)
        klass = info.klass if info else "EXPRESS"
        d0 = date.fromisoformat(start)
        for r in recs:
            st = net.stations.get(r.station_code)
            zone = (st.zone if st and st.zone else "") or (info.zone if info else "")
            day = d0 + timedelta(days=int(r.scheduled // 1440))
            delay = r.delay_min if r.delay_min is not None else r.actual - r.scheduled
            rows.append(_row(d0.toordinal(), klass, zone, r.scheduled, origin_dep,
                             day.month, day.weekday(), r.is_halt,
                             int(r.event == "departure"), delay))
    df = pd.DataFrame(rows, columns=COLUMNS)
    if len(df) < MIN_LIVE_ROWS:
        raise RuntimeError(
            f"only {len(df)} observed station events collected; need at least "
            f"{MIN_LIVE_ROWS} before a fit means anything. Keep the collector "
            "running:\n    python -m railcast.feeds collect --trains 12951,...")
    return df


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--source", choices=("sim", "live"), default="sim")
    ap.add_argument("--days", type=int, default=240, help="simulated days (sim only)")
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--reference", default="12951")
    ap.add_argument("--max-trains", type=int, default=48)


def run(args) -> dict:
    if args.source == "sim":
        rows = records_from_sim(args.days, args.start_date, args.reference, args.max_trains)
    else:
        from ..feeds.live import LiveStore
        from .network import Network
        rows = records_from_live(LiveStore(), Network.load())
    print(f"{len(rows):,} station events from {rows['run_day'].nunique()} run days")
    model, metrics = delays.fit(rows, args.source)
    delays.save(model)
    print(f"saved {delays.MODEL_PATH} - {model.label}")
    if "mae_q50_min" in metrics:
        print(f"  held-out latest days: median-delay error {metrics['mae_q50_min']:.1f} min "
              f"(prior rule {metrics['prior_mae_q50_min']:.1f} min), "
              f"80% window covers {metrics['coverage_80_pct']:.1f}%")
    return metrics

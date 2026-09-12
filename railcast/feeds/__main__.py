"""
    python -m railcast.feeds fetch       download the real timetable data
    python -m railcast.feeds status      what is available locally
    python -m railcast.feeds corridor    build a corridor and describe it
    python -m railcast.feeds collect     poll a live feed into the local store
    python -m railcast.feeds calibrate   fit the delay parameters
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(prog="railcast.feeds", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download the datameet timetable files")
    sub.add_parser("status", help="what data is present locally")

    c = sub.add_parser("corridor", help="build a corridor from the real timetable")
    c.add_argument("--reference", default="12951")
    c.add_argument("--max-trains", type=int, default=48)

    p = sub.add_parser("collect", help="poll a live feed and append to the store")
    p.add_argument("--trains", required=True, help="comma-separated numbers")
    p.add_argument("--source", default="railradar", choices=("railradar", "ntes"))

    k = sub.add_parser("calibrate", help="fit the delay parameters")
    k.add_argument("--reference", default="12951")
    k.add_argument("--max-trains", type=int, default=48)
    k.add_argument("--days", type=int, default=14)
    k.add_argument("--rounds", type=int, default=2)
    k.add_argument("--target-file", help="JSON: class -> fraction within 15 min")
    k.add_argument("--start-date", default="2024-01-01")

    v = sub.add_parser("validate", help="compare the simulator with collected data")
    v.add_argument("--reference", default="12951")
    v.add_argument("--max-trains", type=int, default=44)
    v.add_argument("--days", type=int, default=30)
    v.add_argument("--start-date", default="2024-01-01")

    a = ap.parse_args()

    if a.cmd == "fetch":
        from .datameet import DatameetFeed
        feed = DatameetFeed()
        feed.ensure()
        print(json.dumps(feed.summary(), indent=2))
        return

    if a.cmd == "status":
        from .base import RAW
        from .live import LiveStore
        from ..simconfig import SimConfig
        print("static files:")
        for f in ("stations.json", "trains.json", "schedules.json"):
            p_ = RAW / f
            print(f"  {f:16s} " + (f"{p_.stat().st_size / 1e6:7.1f} MB"
                                   if p_.exists() else "   missing"))
        print("collected live data:")
        print("  " + json.dumps(LiveStore().summary()))
        print("delay parameters:")
        print("  " + json.dumps(SimConfig.load().provenance()))
        return

    if a.cmd == "corridor":
        from .corridor import build_corridor, build_roster
        from .datameet import DatameetFeed
        feed = DatameetFeed()
        cor, meta = build_corridor(feed, a.reference)
        roster = build_roster(feed, cor, limit=a.max_trains)
        print(json.dumps(meta, indent=2))
        print(f"roster: {len(roster)} trains")
        for t in roster[:10]:
            run = (t.sched_arr[t.dest] - t.dep_minute) / 60
            print(f"  {t.number:6s} {t.name[:40]:40s} {t.klass:9s} "
                  f"{len(t.halts):3d} halts  {run:5.2f} h")
        return

    if a.cmd == "collect":
        from .live import LiveStore, NTESFeed, RailRadarFeed, collect
        import sys
        feed = RailRadarFeed() if a.source == "railradar" else NTESFeed()
        store = LiveStore()
        try:
            n = collect(feed, [x.strip() for x in a.trains.split(",")], store)
        except (RuntimeError, NotImplementedError) as exc:
            print(f"cannot collect: {exc}", file=sys.stderr)
            raise SystemExit(2)
        print(f"{n} position reports appended to {store.path()}")
        print(json.dumps(store.summary(), indent=2))
        if n == 0:
            print("collected nothing - check the train numbers and the feed",
                  file=sys.stderr)
            raise SystemExit(1)
        return

    if a.cmd == "validate":
        from .. import validate as val
        from ..simconfig import SimConfig
        from .corridor import build_corridor, build_roster
        from .datameet import DatameetFeed
        from .live import LiveStore
        from .openmeteo import build_environments
        feed = DatameetFeed()
        cor, _ = build_corridor(feed, a.reference)
        trains = build_roster(feed, cor, limit=a.max_trains)
        envs = build_environments(cor, feed, a.start_date, a.days,
                                  np.random.default_rng(1))
        obs = val.observed_delays(LiveStore())
        sim = val.simulated_delays(cor, trains, envs, SimConfig.load(), a.days)
        print(val.report(val.compare(obs, sim)))
        return

    if a.cmd == "calibrate":
        from .. import calibrate as cal
        from .corridor import build_corridor, build_roster
        from .datameet import DatameetFeed
        from .openmeteo import build_environments
        feed = DatameetFeed()
        cor, _ = build_corridor(feed, a.reference)
        trains = build_roster(feed, cor, limit=a.max_trains)
        envs = build_environments(cor, feed, a.start_date, max(a.days, 30),
                                  np.random.default_rng(1))
        cfg = cal.fit(cor, trains, envs, target=cal.load_target(a.target_file),
                      days=a.days, rounds=a.rounds)
        cfg.save()
        print(json.dumps(cfg.provenance(), indent=2))
        return


if __name__ == "__main__":
    main()

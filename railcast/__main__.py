"""
    python -m railcast plan FROM TO [--date 2026-10-27] [--after 18:00] [--class SL]
                                    [--max-changes 1] [--json] [--chart]
    python -m railcast train-delays --source sim|live
    python -m railcast evaluate [--quick] [--source real|synthetic]

plan          answer a trip query: Safe / Likely / Risky options, three picks,
              a booking plan. Saved to outputs/plans/.
train-delays  train the planner's delay model on simulated or collected running data
evaluate      train and score the corridor forecaster; writes outputs/RESULTS.md

Data feeds (timetable download, live collection) are under `python -m railcast.feeds`.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta


def _plan(a) -> None:
    from .planner.network import Network
    from .planner.respond import plan

    net = Network.load()
    day0 = date.fromisoformat(a.date) if a.date else date.today() + timedelta(days=1)
    try:
        resp = plan(net, a.origin, a.dest, day0, after=a.after, window_h=a.window,
                    klass=a.klass, max_changes=a.max_changes, nearby=not a.exact,
                    delays=a.delays, include_unreserved=a.unreserved)
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        print(f"cannot plan: {exc}", file=sys.stderr)
        raise SystemExit(2)
    path = resp.save()
    if a.json:
        print(json.dumps(resp.to_dict(), indent=2))
    else:
        print(resp.text(per_group=a.limit))
        print(f"\nfull response: {path}")
    if a.chart:
        from .planner.chart import render
        print(f"chart: {render(net, resp)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="railcast", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="answer a trip query")
    p.add_argument("origin", help="station code (current or old) or name")
    p.add_argument("dest")
    p.add_argument("--date", help="travel date, YYYY-MM-DD (default tomorrow)")
    p.add_argument("--after", default="00:00", help="earliest departure, HH:MM")
    p.add_argument("--window", type=float, default=24.0, help="hours to search from --after")
    p.add_argument("--class", dest="klass", help="SL, 3A, 2A, 1A, CC ... (needed for seat checks)")
    p.add_argument("--max-changes", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--exact", action="store_true",
                   help="only the stations given, not others in the same city")
    p.add_argument("--delays", default="auto", choices=("auto", "prior", "sim", "live"),
                   help="delay model: auto = trained on real data if present, else prior")
    p.add_argument("--unreserved", action="store_true",
                   help="include unreserved passenger and MEMU trains")
    p.add_argument("--limit", type=int, default=4, help="options listed per group")
    p.add_argument("--json", action="store_true", help="print the response as JSON")
    p.add_argument("--chart", action="store_true", help="draw the response chart")

    from .planner import train_delays
    t = sub.add_parser("train-delays", help="train the planner's delay model")
    train_delays.add_arguments(t)

    from .forecast import pipeline
    e = sub.add_parser("evaluate", help="train and score the corridor forecaster")
    pipeline.add_arguments(e)

    a = ap.parse_args(argv)
    if a.cmd == "plan":
        _plan(a)
    elif a.cmd == "train-delays":
        train_delays.run(a)
    elif a.cmd == "evaluate":
        pipeline.run(a)


if __name__ == "__main__":
    main()

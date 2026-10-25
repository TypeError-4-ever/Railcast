"""
The forecaster, end to end.

    python -m railcast evaluate [--days 450] [--quick] [--source real|synthetic]

Simulates the New Delhi - Mumbai Central corridor, trains the forecast stack on
a temporal holdout, calibrates the arrival window, scores it against the
carry-forward baseline, saves the trained stack to models/forecaster.pkl and
writes outputs/RESULTS.md plus the three evaluation charts.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from .dataset import World, build_dataset
from .models import (HORIZON_LABELS, MondrianConformal, ResidualModel,
                     enforce_monotone, fit_free_run, free_run_table)

OUT = Path("outputs")
DATA = OUT / "data"
MODELS = Path("models")


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def build_world(args, n_days: int, rng, t0: float):
    """The real corridor and weather, or the hand-built one with no network."""
    if args.source != "real":
        world = World(n_days, seed=11)
        return world, {"name": "synthetic", "corridor": "hand-built New Delhi - "
                       "Mumbai Central corridor (35 stations, 1386 km)"}

    from ..feeds import DatameetFeed, OpenMeteoFeed
    from ..feeds.corridor import build_corridor, build_roster
    from ..feeds.openmeteo import build_environments

    feed = DatameetFeed()
    cor, meta = build_corridor(feed, args.reference)
    trains = build_roster(feed, cor, min_span=0.5, limit=args.max_trains)
    log(f"corridor from real timetable: {meta['spine_stations']} stations, "
        f"{meta['route_km']:.0f} km, {len(trains)} trains", t0)
    envs = build_environments(cor, feed, args.start_date, n_days, rng)
    log(f"observed weather for {len(envs)} days from {args.start_date}", t0)
    world = World(n_days, seed=11, cor=cor, trains=trains, envs=envs)
    source = {
        "name": "real",
        "static_feed": feed.summary(),
        "weather_feed": {"source": OpenMeteoFeed.name,
                         "licence": OpenMeteoFeed.licence,
                         "start_date": args.start_date},
        "corridor": (f"{cor.code(0)} - {cor.code(cor.n_stations - 1)}, "
                     f"{meta['route_km']:.0f} km, {meta['spine_stations']} "
                     "stations, real working timetable"),
        "invented_inputs": ["temporary speed restrictions", "engineering blocks",
                            "run-time friction and incident rates",
                            "conflict outcomes (no live feed collected yet)"],
        "delay_parameters": world.config.provenance(),
    }
    return world, source


def write_results_md(summary, tables, conn) -> None:
    s = summary
    imp = tables["by_horizon"].set_index("horizon").reindex(
        [h for h in HORIZON_LABELS if h in set(tables["by_horizon"]["horizon"])])
    real = s["data_source"].get("name") == "real"
    lines = [
        "# Forecaster results",
        "",
        f"Corridor: **{s['corridor']}**, {s['trains']} trains, "
        f"{s['operating_days_simulated']} simulated operating days. "
        f"Trained on days {s['train_days'][0]}-{s['train_days'][1]}, calibrated on "
        f"{s['calib_days'][0]}-{s['calib_days'][1]}, scored on "
        f"{s['test_days'][0]}-{s['test_days'][1]} (temporal holdout).",
        "",
        ("**Scored on simulated delays.** The corridor, timetable and weather are "
         "real; how delay arises is still modelled, so these numbers show the "
         "method works on a problem of the right shape - not field accuracy. "
         "They become real once the simulator is fitted to collected running "
         "data (`python -m railcast.feeds calibrate --from-observed`)."
         if real else
         "**Everything here is simulated** - the hand-built corridor, timetable "
         "and weather included."),
        "",
        "| Metric | Carry-forward (today) | RAILCAST |",
        "|---|---|---|",
        f"| MAE | {s['baseline_mae_min']:.1f} min | {s['railcast_mae_min']:.1f} min "
        f"({s['mae_improvement_pct']:.0f}% lower) |",
        f"| P90 absolute error | {s['baseline_p90_min']:.1f} min "
        f"| {s['railcast_p90_min']:.1f} min |",
        f"| 80% arrival window | none | {s['coverage_pct']:.1f}% coverage, "
        f"{s['mean_window_min']:.0f} min wide |",
        f"| Connection probability | none | reliability error {conn['ece']:.3f} "
        f"over {conn['n']:,} forecasts |",
        "",
        "## By horizon",
        "",
        "| Horizon | n | Carry-forward | RAILCAST | Improvement | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for h, r in imp.iterrows():
        lines.append(f"| {h} | {r['n']:,.0f} | {r['baseline_mae']:.1f} min | "
                     f"{r['railcast_mae']:.1f} min | {r['improvement_pct']:.0f}% | "
                     f"{r['coverage_pct']:.1f}% |")
    sv = s["serving"]
    lines += [
        "",
        f"Serving: {sv['full_corridor_reforecast_s']:.2f} s to re-forecast the "
        f"corridor ({sv['station_forecasts_per_cycle']} station forecasts), one "
        "process, no GPU.",
        "",
        "Charts: `outputs/charts/` - accuracy by horizon, connection risk, one "
        "journey forecast.",
        "",
    ]
    (OUT / "RESULTS.md").write_text(chr(10).join(lines), encoding="utf-8")


def run(args) -> dict:
    n_days = 60 if args.quick else args.days
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)

    world, source = build_world(args, n_days, rng, t0)
    world.simulate()
    log(f"simulated {n_days} operating days, {len(world.trains)} trains", t0)

    # temporal holdout: train on earlier months, test on later ones
    tr_end, ca_end = int(n_days * 0.70), int(n_days * 0.85)
    train_days = range(0, tr_end)
    calib_days = range(tr_end, ca_end)
    test_days = range(ca_end, n_days)

    # stage A: free-run model
    sections = world.section_runtime_rows(train_days)
    fr_model, fr_stats = fit_free_run(sections)
    fr_table = free_run_table(fr_model, sections)
    log(f"stage A free-run model: MAE {fr_stats['mae_min']:.2f} min/section", t0)

    # stages B+C: replay and residual
    df = build_dataset(world, fr_table, rng, snapshots=args.snapshots)
    log(f"built {len(df):,} forecasting rows", t0)
    tr = df[df["day"].isin(train_days)]
    ca = df[df["day"].isin(calib_days)]
    te = df[df["day"].isin(test_days)]
    model = ResidualModel().fit(tr)
    log("stage C residual model fitted", t0)

    # stage D: Mondrian conformal calibration
    _, ca_lo, ca_hi = model.predict(ca)
    conf = MondrianConformal().fit(ca, ca_lo, ca_hi)
    log(f"stage D calibrated, {len(conf.q)} Mondrian groups", t0)

    # score on the test months
    base_col = ev.pick_baseline(tr)
    mid, lo, hi = model.predict(te)
    out = te.copy()
    out["railcast_arr"] = out["rules_arr"] + mid
    lo_c, hi_c = conf.apply(out, out["rules_arr"] + lo, out["rules_arr"] + hi)
    out["lo_arr"], out["hi_arr"] = lo_c, hi_c
    out["baseline"] = out[base_col]
    out["rules_only"] = out["rules_arr"]
    out = enforce_monotone(out, ["railcast_arr", "lo_arr", "hi_arr"])
    scored = ev.score(out)
    log(f"scored {len(scored):,} test forecasts, baseline = {base_col}", t0)

    from . import analysis as an
    conn = an.connection_analysis(scored)
    lat = an.latency_benchmark(world, model, fr_table, test_days)
    delays = an.final_delays(world, test_days)

    summary = ev.overall(scored)
    summary.update({
        "data_source": source,
        "corridor": source["corridor"],
        "trains": len(world.trains),
        "operating_days_simulated": n_days,
        "train_days": [0, tr_end - 1], "calib_days": [tr_end, ca_end - 1],
        "test_days": [ca_end, n_days - 1],
        "baseline_definition": base_col,
        "free_run_model": fr_stats,
        "connection_reliability_ece": round(conn["ece"], 4),
        "serving": {
            "full_corridor_reforecast_s": round(lat["cycle_s"], 3),
            "station_forecasts_per_cycle": int(lat["mean_rows"]),
        },
        "punctuality_simulated": {
            "median_delay_min": round(float(np.median(delays)), 1),
            "p90_delay_min": round(float(np.percentile(delays, 90)), 1),
        },
    })
    tables = {"by_horizon": ev.by(scored, "horizon")}
    tables["by_horizon"].to_csv(DATA / "by_horizon.csv", index=False)
    pd.DataFrame({"buffer_min": conn["buffer"], "p_make": conn["p_make"],
                  "observed_rate": conn["observed"]}).to_csv(
        DATA / "connection_risk.csv", index=False)
    (DATA / "metrics_summary.json").write_text(json.dumps(summary, indent=2))

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "forecaster.pkl", "wb") as fh:
        pickle.dump({"fr_table": fr_table, "model": model, "conf": conf,
                     "baseline": base_col, "source": source["name"]}, fh)
    log("trained stack saved to models/forecaster.pkl", t0)

    print(f"\n  MAE  carry-forward {summary['baseline_mae_min']:.1f} min  ->  "
          f"RAILCAST {summary['railcast_mae_min']:.1f} min  "
          f"({summary['mae_improvement_pct']:+.1f}%)")
    print(f"  80% window coverage {summary['coverage_pct']:.1f}%, connection "
          f"reliability error {conn['ece']:.3f}\n")

    if not args.no_charts:
        from . import charts
        # a typical late train with most of its run still ahead - not the worst case
        late = scored[(scored["hops"] >= 3) & scored["cur_delay"].between(20, 60)]
        pool = late if len(late) else scored[scored["hops"] >= 3]
        card = pool.sort_values("dist_remaining_km").iloc[-1]
        track = an.journey_track(world, model, conf, fr_table,
                                 int(card["day"]), str(card["train"]))
        n = charts.render_all(dict(world=world, tables=tables, summary=summary,
                                   connection=conn, track=track))
        log(f"{n} charts written to outputs/charts", t0)

    write_results_md(summary, tables, conn)
    log("outputs/RESULTS.md written", t0)
    return summary


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--days", type=int, default=450)
    ap.add_argument("--quick", action="store_true", help="60 days, for a smoke run")
    ap.add_argument("--source", choices=("real", "synthetic"), default="real",
                    help="real = Indian Railways timetable + ERA5 weather; "
                         "synthetic = the hand-built corridor, no network")
    ap.add_argument("--reference", default="12951",
                    help="train whose route defines the corridor spine")
    ap.add_argument("--start-date", default="2024-01-01",
                    help="first calendar day, for real weather lookup")
    ap.add_argument("--max-trains", type=int, default=48)
    ap.add_argument("--snapshots", type=int, default=4,
                    help="re-forecast snapshots sampled per operating day")
    ap.add_argument("--no-charts", action="store_true")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    run(p.parse_args())

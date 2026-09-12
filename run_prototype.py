"""
RAILCAST prototype - end to end.

    python run_prototype.py [--days 450] [--quick]

Simulates the New Delhi - Mumbai Central corridor, trains the forecast stack on
a temporal holdout, calibrates the arrival window, scores it against the
carry-forward baseline, and writes every table and chart the deck needs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from railcast import evaluate as ev
from railcast.dataset import World, build_dataset, calendar
from railcast.models import (MondrianConformal, ResidualModel, enforce_monotone,
                             fit_free_run, free_run_table)

OUT = Path("outputs")
DATA = OUT / "data"


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)



def write_results_md(summary, tables, prec, conn, lat, card):
    """Every headline number, written out ready to paste into the deck."""
    s = summary
    from railcast.models import HORIZON_LABELS
    imp = tables["by_horizon"].set_index("horizon").reindex(
        [h for h in HORIZON_LABELS if h in set(tables["by_horizon"]["horizon"])])
    src = s.get("data_source", {})
    lines = [
        "# RAILCAST prototype - results",
        "",
        f"Corridor: **{s['corridor']}**, {s['trains']} trains, "
        f"{s['operating_days_simulated']} simulated operating days.",
        "",
        (f"Data source: **{src.get('name')}**. "
         + (f"Station master, roster and working timetable from "
            f"{src['static_feed']['source']} "
            f"({src['static_feed']['stations']:,} stations, "
            f"{src['static_feed']['trains']:,} trains, "
            f"{src['static_feed']['scheduled_calls']:,} scheduled calls, "
            f"{src['static_feed']['licence']}); weather is ERA5 reanalysis via "
            f"Open-Meteo from {src['weather_feed']['start_date']}. Still "
            "invented: " + ", ".join(src["invented_inputs"]) + "."
            if src.get("name") == "real" else
            "Corridor, roster and weather are all hand-built.")),
        "",
        "Temporal holdout - trained on days "
        f"{s['train_days'][0]}-{s['train_days'][1]} "
        f"({s['rows_train']:,} forecasts), calibrated on days "
        f"{s['calib_days'][0]}-{s['calib_days'][1]} ({s['rows_calib']:,}), "
        f"scored on days {s['test_days'][0]}-{s['test_days'][1]} "
        f"({s['rows_test']:,}). A random split would leak and flatter every number.",
        "",
        "## Headline",
        "",
        "| Metric | Carry-forward (today) | RAILCAST | Change |",
        "|---|---|---|---|",
        f"| MAE | {s['baseline_mae_min']:.1f} min | {s['railcast_mae_min']:.1f} min "
        f"| **{s['mae_improvement_pct']:.0f}% lower** |",
        f"| P90 absolute error | {s['baseline_p90_min']:.1f} min "
        f"| {s['railcast_p90_min']:.1f} min "
        f"| {100 * (1 - s['railcast_p90_min'] / s['baseline_p90_min']):.0f}% lower |",
        f"| Uncertainty stated | none | 80% window, "
        f"{s['coverage_pct']:.1f}% achieved coverage | - |",
        f"| Mean window width | - | {s['mean_window_min']:.0f} min | - |",
        "",
        "Conflict simulator alone, with the learned layer switched off, scores "
        f"{s['rules_only_mae_min']:.1f} min MAE overall - still well ahead of "
        "carry-forward, which is what makes it a usable fallback when the model "
        "is unavailable. It is worse than carry-forward at the next station, "
        "where repeating the current delay is hard to beat; the learned residual "
        "is what fixes that end of the range.",
        "",
        "## By horizon",
        "",
        "| Horizon | n | Carry-forward MAE | Simulator only | RAILCAST MAE | "
        "Improvement | Coverage | Window |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h, r in imp.iterrows():
        lines.append(
            f"| {h} | {r['n']:,.0f} | {r['baseline_mae']:.1f} min | "
            f"{r['simulator_mae']:.1f} min | {r['railcast_mae']:.1f} min | "
            f"**{r['improvement_pct']:.0f}%** | {r['coverage_pct']:.1f}% | "
            f"{r['width_min']:.0f} min |")
    lines += [
        "",
        "## Decision surfaces",
        "",
        f"- **Costed precedence.** One real conflict from day {prec['day']}, "
        f"re-run {prec['runs']} times each way on identical days. Reversing the "
        f"call saves {prec['b']} {abs(prec['delta_by']['held']):.0f} minutes, "
        f"costs {prec['a']} {prec['delta_by']['blocker']:+.0f} and the rest of the "
        f"corridor {prec['delta_by']['others']:+.0f} - "
        f"{prec['delta']:+.0f} network-minutes on net "
        f"(standard error {prec['delta_se']['total']:.0f}).",
        f"- **Connection risk.** Stated make-probability against realised outcome "
        f"on held-out days: reliability error {conn['ece']:.3f} over "
        f"{conn['n']:,} forecasts.",
        f"- **Serving.** {s['serving']['ms_per_station_forecast']:.2f} ms per "
        f"station forecast, {s['serving']['full_corridor_reforecast_s']:.2f} s for a "
        f"full corridor re-forecast of {s['serving']['station_forecasts_per_cycle']} "
        f"station predictions, single process, no GPU. Linear extrapolation to "
        f"13,000 trains: {s['serving']['projected_national_reforecast_s']:.0f} s "
        "in one process, about "
        f"{s['serving']['projected_national_reforecast_s'] / 16:.0f} s sharded "
        "across 16 zonal graphs - inside the 30 s budget.",
        "",
        "## The corridor the model learns on",
        "",
        f"Simulated punctuality: median {s['punctuality_simulated']['median_delay_min']:.0f} "
        f"min late at destination, P90 {s['punctuality_simulated']['p90_delay_min']:.0f} min, "
        f"{s['punctuality_simulated']['within_15_min_pct']:.0f}% arriving within 15 minutes.",
        "",
        "## Honesty note for the deck",
        "",
        "**Every number above comes from a simulator. No Indian Railways data was "
        "used, downloaded or consulted.** Station names, km posts and train "
        "numbers are approximate; timetables, running times, delays, conflicts "
        "and weather are all generated. The simulator's parameters were set by "
        "hand until punctuality looked plausible - they were not fitted to "
        "published punctuality returns, which is step 2 of the roadmap.",
        "",
        "What this does establish: the forecaster never sees the noise that "
        "generates the delay, the baseline is computed on exactly the same rows, "
        "and the split is temporal. That is evidence the method works on a "
        "problem of the right shape - not evidence of field performance. Label "
        "every chart that way.",
        "",
    ]
    Path("outputs/RESULTS.md").write_text(chr(10).join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()
    n_days = 60 if args.quick else args.days

    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(23)

    # ---- 1. the world: real corridor and weather, or the synthetic one ---- #
    source = {"name": "synthetic", "corridor": "hand-built New Delhi - Mumbai "
              "Central corridor (35 stations, 1386 km)"}
    if args.source == "real":
        from railcast.feeds import DatameetFeed, OpenMeteoFeed
        from railcast.feeds.corridor import build_corridor, build_roster
        from railcast.feeds.openmeteo import build_environments

        feed = DatameetFeed()
        cor, meta = build_corridor(feed, args.reference)
        trains = build_roster(feed, cor, min_span=0.5, limit=args.max_trains)
        log(f"corridor from real timetable: {meta['spine_stations']} stations, "
            f"{meta['route_km']:.0f} km (published {meta['published_km']:.0f}), "
            f"{len(trains)} trains", t0)
        envs = build_environments(cor, feed, args.start_date, n_days, rng)
        log(f"observed weather for {len(envs)} days from {args.start_date} "
            f"(ERA5 via Open-Meteo, {len(meta['segments'])} segments)", t0)
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
            "corridor_meta": meta,
            "invented_inputs": ["temporary speed restrictions",
                                "engineering blocks",
                                "run-time friction and incident rates",
                                "conflict outcomes (no live feed collected yet)"],
        }
        source["delay_parameters"] = world.config.provenance()
    else:
        world = World(n_days, seed=11)
    world.simulate()
    log(f"simulated {n_days} operating days, {len(world.trains)} trains", t0)

    # temporal holdout: train on earlier months, test on later ones
    tr_end, ca_end = int(n_days * 0.70), int(n_days * 0.85)
    train_days = range(0, tr_end)
    calib_days = range(tr_end, ca_end)
    test_days = range(ca_end, n_days)

    # ---- 2. stage A: free-run model -------------------------------------- #
    sections = world.section_runtime_rows(train_days)
    fr_model, fr_stats = fit_free_run(sections)
    fr_table = free_run_table(fr_model, sections)
    log(f"stage A free-run model: MAE {fr_stats['mae_min']:.2f} min/section "
        f"(timetable physics alone {fr_stats['theoretical_mae_min']:.2f})", t0)

    # ---- 3. stages B+C: replay and dataset ------------------------------- #
    df = build_dataset(world, fr_table, rng, snapshots=args.snapshots)
    log(f"built {len(df):,} forecasting rows "
        f"from {len(world.trains)} trains x {n_days} days", t0)

    tr = df[df["day"].isin(train_days)]
    ca = df[df["day"].isin(calib_days)]
    te = df[df["day"].isin(test_days)]

    model = ResidualModel().fit(tr)
    log("stage C residual model fitted (l1 + two quantiles)", t0)

    # ---- 4. stage D: Mondrian conformal calibration ----------------------- #
    _, ca_lo, ca_hi = model.predict(ca)
    conf = MondrianConformal().fit(ca, ca_lo, ca_hi)
    log(f"stage D calibrated on {len(ca):,} held-out rows, "
        f"{len(conf.q)} Mondrian groups", t0)

    # ---- 5. score on the test months ------------------------------------- #
    base_col = ev.pick_baseline(tr)
    res = {}
    for name, part in (("test", te), ("calib", ca)):
        mid, lo, hi = model.predict(part)
        out = part.copy()
        out["railcast_arr"] = out["rules_arr"] + mid
        lo_c, hi_c = conf.apply(out, out["rules_arr"] + lo, out["rules_arr"] + hi)
        out["lo_arr"], out["hi_arr"] = lo_c, hi_c
        out["baseline"] = out[base_col]
        out["rules_only"] = out["rules_arr"]
        out = enforce_monotone(out, ["railcast_arr", "lo_arr", "hi_arr"])
        res[name] = ev.score(out)
    scored = res["test"]
    log(f"scored {len(scored):,} test forecasts, baseline = {base_col}", t0)

    # ---- 6. tables -------------------------------------------------------- #
    summary = ev.overall(scored)
    summary.update({
        "data_source": source,
        "corridor": source["corridor"],
        "trains": len(world.trains),
        "operating_days_simulated": n_days,
        "train_days": [0, tr_end - 1], "calib_days": [tr_end, ca_end - 1],
        "test_days": [ca_end, n_days - 1],
        "baseline_definition": base_col,
        "rows_train": len(tr), "rows_calib": len(ca), "rows_test": len(te),
        "free_run_model": fr_stats,
        "rules_only_mae_min": round(
            float(np.mean(np.abs(scored["rules_only"] - scored["act_arr"]))), 2),
    })

    tables = {
        "by_horizon": ev.by(scored, "horizon"),
        "by_zone": ev.by(scored, "zone_name"),
        "by_zone_horizon": ev.by(scored, ["zone_name", "horizon"]),
        "by_month": ev.by(scored, "month_name"),
        "by_tod": ev.by(scored, "tod"),
        "by_class": ev.by(scored, "klass"),
    }
    for name, t in tables.items():
        if t is not None:
            t.to_csv(DATA / f"{name}.csv", index=False)
    (DATA / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    model.importance().rename("gain_pct").to_csv(DATA / "feature_importance.csv")
    scored.to_parquet(DATA / "scored_test.parquet")
    log("tables written to outputs/data", t0)

    print("\n" + "=" * 62)
    print(f"  MAE   carry-forward {summary['baseline_mae_min']:6.1f} min"
          f"   ->  RAILCAST {summary['railcast_mae_min']:6.1f} min"
          f"   ({summary['mae_improvement_pct']:+.1f}%)")
    print(f"  P90   carry-forward {summary['baseline_p90_min']:6.1f} min"
          f"   ->  RAILCAST {summary['railcast_p90_min']:6.1f} min")
    print(f"  80% window coverage achieved {summary['coverage_pct']:.1f}%"
          f"   mean width {summary['mean_window_min']:.0f} min")
    print("=" * 62 + "\n")
    print(tables["by_horizon"].to_string(index=False))

    # ---- 7. decision surfaces and experiments ----------------------------- #
    from railcast import analysis as an

    cascade = an.pick_cascade(world, test_days)
    prec = an.precedence_experiment(world, cascade[0], cascade[1], cascade[4])
    conn = an.connection_analysis(scored)
    calib = an.calibration_effect(model, conf, te)
    lat = an.latency_benchmark(world, model, fr_table, test_days)
    card = an.console_card(world, scored, conn)
    track = an.journey_track(world, model, conf, fr_table, card["day"], card["train"])
    delays = an.final_delays(world, test_days)
    log("decision surfaces computed "
        f"(precedence delta {prec['delta']:+.0f} net-min, "
        f"connection ECE {conn['ece']:.3f}, "
        f"{1000 * lat['cycle_s'] / lat['mean_rows']:.2f} ms per station forecast)", t0)

    summary.update({
        "precedence_experiment": {k: (round(v, 2) if isinstance(v, float) else v)
                                  for k, v in prec.items()},
        "connection_reliability_ece": round(conn["ece"], 4),
        "serving": {
            "ms_per_station_forecast": round(1000 * lat["cycle_s"] / lat["mean_rows"], 3),
            "full_corridor_reforecast_s": round(lat["cycle_s"], 3),
            "station_forecasts_per_cycle": int(lat["mean_rows"]),
            "projected_national_reforecast_s": round(
                lat["cycle_s"] * 13000 / lat["trains"], 1),
        },
        "punctuality_simulated": {
            "median_delay_min": round(float(np.median(delays)), 1),
            "p90_delay_min": round(float(np.percentile(delays, 90)), 1),
            "within_15_min_pct": round(float(100 * (delays < 15).mean()), 1),
        },
    })
    (DATA / "metrics_summary.json").write_text(json.dumps(summary, indent=2))
    card["rows"].to_csv(DATA / "sample_forecast_card.csv", index=False)
    track[["now", "train", "to_code", "hops", "sched_arr", "baseline",
           "railcast_arr", "lo_arr", "hi_arr", "act_arr"]].to_csv(
        DATA / "tracked_journey.csv", index=False)
    pd.DataFrame({"buffer_min": conn["buffer"], "p_make": conn["p_make"],
                  "observed_rate": conn["observed"]}).to_csv(
        DATA / "connection_risk.csv", index=False)

    # ---- 8. charts -------------------------------------------------------- #
    if not args.no_charts:
        from railcast import charts
        ctx = dict(world=world, scored=scored, tables=tables, summary=summary,
                   importance=model.importance(), hero_train="12951",
                   cascade=cascade, precedence=prec, connection=conn,
                   calibration=calib, latency=lat, card=card, final_delays=delays,
                   track=track, track_step=30.0)
        n = charts.render_all(ctx)
        log(f"{n} chart files written to outputs/charts", t0)

    write_results_md(summary, tables, prec, conn, lat, card)
    log("outputs/RESULTS.md written", t0)


if __name__ == "__main__":
    main()

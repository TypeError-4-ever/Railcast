"""
Check the simulator against running data actually collected.

This is the test the whole project was missing. Everything upstream - corridor,
timetable, weather - comes from published sources and can be checked against
them. How delay *arises* could not be checked at all until there were real
arrivals to compare with.

The comparison is station-level on both sides: delay at each reported station,
not delay at the destination. Mixing the two flatters the simulator, because
mid-run delay is smaller than end-of-run delay.
"""
from __future__ import annotations

import numpy as np

from .simulator import Simulator

QUANTILES = (10, 25, 50, 75, 90, 95, 99)


def observed_delays(store, event: str = "arrival") -> np.ndarray:
    """Delay in minutes at every station a train was actually reported at."""
    vals = [r.delay_min for r in store.read()
            if r.observed and r.event == event and r.delay_min is not None]
    return np.array(sorted(vals), dtype=float)


def simulated_delays(cor, trains, envs, cfg, days: int, seed: int = 5,
                     halts_only: bool = True) -> np.ndarray:
    """The same quantity from the simulator: delay at each station call."""
    sim = Simulator(cor, trains, cfg)
    rng = np.random.default_rng(seed)
    out = []
    for d in range(days):
        env = envs[d] if isinstance(envs, dict) else envs[d % len(envs)]
        for tid, run in sim.run_day(env, rng).items():
            t = sim.trains[tid]
            for stn, actual in run.arr.items():
                if halts_only and stn not in t.halts:
                    continue
                sched = t.sched_arr.get(stn)
                if sched is not None:
                    out.append(actual - sched)
    return np.array(sorted(out), dtype=float)


def compare(obs: np.ndarray, sim: np.ndarray) -> dict:
    if obs.size == 0:
        raise RuntimeError(
            "No observed arrivals collected yet. Run\n"
            "    python -m railcast.feeds collect --trains 12951,12952,...\n"
            "on a schedule first - there is no archive to download.")
    rows = []
    for q in QUANTILES:
        o, s = float(np.percentile(obs, q)), float(np.percentile(sim, q))
        rows.append({"quantile": f"p{q}", "observed": o, "simulated": s,
                     "gap": s - o})
    within = lambda a, m: float(100 * (a < m).mean())     # noqa: E731
    return {
        "n_observed": int(obs.size), "n_simulated": int(sim.size),
        "quantiles": rows,
        "within_15_min": {"observed": within(obs, 15), "simulated": within(sim, 15)},
        "over_60_min": {"observed": 100 - within(obs, 60),
                        "simulated": 100 - within(sim, 60)},
        "median_gap_min": float(np.median(sim) - np.median(obs)),
        "tail_ratio_p90": float(np.percentile(sim, 90) /
                                max(np.percentile(obs, 90), 1e-6)),
    }


def _verdict(result: dict) -> str:
    r = result["tail_ratio_p90"]
    return "too fat" if r > 1.25 else "too thin" if r < 0.8 else "about right"


def report(result: dict) -> str:
    lines = [
        f"observed {result['n_observed']:,} station arrivals   "
        f"simulated {result['n_simulated']:,}",
        "",
        f"{'':>9} {'observed':>10} {'simulated':>10} {'gap':>8}",
    ]
    for r in result["quantiles"]:
        lines.append(f"{r['quantile']:>9} {r['observed']:>9.0f}m "
                     f"{r['simulated']:>9.0f}m {r['gap']:>+7.0f}m")
    w, o = result["within_15_min"], result["over_60_min"]
    lines += [
        "",
        f"within 15 min   observed {w['observed']:.0f}%   "
        f"simulated {w['simulated']:.0f}%",
        f"over 60 min     observed {o['observed']:.0f}%   "
        f"simulated {o['simulated']:.0f}%",
        "",
        f"median gap {result['median_gap_min']:+.0f} min, "
        f"P90 tail ratio {result['tail_ratio_p90']:.2f}x ({_verdict(result)})",
    ]
    if result["n_observed"] < 2000:
        lines += ["",
                  "NOTE: fewer than 2,000 observed arrivals. Read this as a "
                  "direction of travel, not a fit. Keep the collector running."]
    return "\n".join(lines)

"""
What the forecaster's output means for a passenger, checked against the outcome.

The connection probability is scored against what actually happened on held-out
days; the serving cost is timed, not estimated.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..dist import quantile_cdf


# --------------------------------------------------------------------------- #
def final_delays(world, days) -> np.ndarray:
    out = []
    for d in days:
        _, truth = world.days[d]
        for tid, run in truth.items():
            t = world.sim.trains[tid]
            if t.dest in run.arr:
                out.append(run.arr[t.dest] - t.sched_arr[t.dest])
    return np.array(out)


# --------------------------------------------------------------------------- #
def connection_analysis(scored: pd.DataFrame, buffers=(15, 20, 30, 40, 50, 60, 75, 90, 120)):
    d = scored[scored["hops"] >= 3]
    if len(d) > 40000:
        d = d.sample(40000, random_state=3)
    lo, mid, hi = d["lo_arr"].to_numpy(), d["railcast_arr"].to_numpy(), d["hi_arr"].to_numpy()
    sched, act = d["sched_arr"].to_numpy(), d["act_arr"].to_numpy()
    p_curve, obs_curve, allp, allo = [], [], [], []
    for B in buffers:
        p = np.clip(quantile_cdf(lo, mid, hi, sched + B), 0, 1)
        o = (act <= sched + B).astype(float)
        p_curve.append(float(p.mean()))
        obs_curve.append(float(o.mean()))
        allp.append(p)
        allo.append(o)
    allp, allo = np.concatenate(allp), np.concatenate(allo)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(allp, edges[1:-1]), 0, 9)
    pred_bin, obs_bin, wt = [], [], []
    for k in range(10):
        m = idx == k
        if m.sum() < 50:
            continue
        pred_bin.append(float(allp[m].mean()))
        obs_bin.append(float(allo[m].mean()))
        wt.append(m.sum())
    wt = np.array(wt, dtype=float)
    ece = float(np.sum(wt / wt.sum() * np.abs(np.array(pred_bin) - np.array(obs_bin))))
    return dict(buffer=list(buffers), p_make=p_curve, observed=obs_curve,
                pred_bin=pred_bin, obs_bin=obs_bin, ece=ece, n=int(len(d)))


# --------------------------------------------------------------------------- #
def latency_benchmark(world, model, fr_table, days, cycles: int = 120):
    """Time a full-corridor re-forecast: replay, features, inference."""
    from .dataset import COLUMNS
    rng = np.random.default_rng(77)
    rows_per, secs = [], []
    days = list(days)
    for _ in range(cycles):
        d = int(rng.choice(days))
        now = float(rng.uniform(300, 1300))
        t0 = time.perf_counter()
        rows = world.snapshot_rows(d, now, fr_table)
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=COLUMNS)
        model.predict(df)
        secs.append(time.perf_counter() - t0)
        rows_per.append(len(df))
    return dict(cycle_s=float(np.mean(secs)), mean_rows=float(np.mean(rows_per)),
                trains=len(world.trains), cycles=len(secs))


# --------------------------------------------------------------------------- #
def journey_track(world, model, conf, fr_table, day: int, tid: str,
                  step: float = 30.0):
    """Re-forecast one journey every `step` minutes, start to finish.

    Each new actual arrival updates the state and the rest of the run is
    forecast again.
    """
    from . import evaluate as ev
    from .dataset import COLUMNS
    from .models import enforce_monotone

    t = world.sim.trains[tid]
    run = world.days[day][1][tid]
    t0, t1 = run.dep[t.origin], run.arr[t.dest]
    rows = []
    for now in np.arange(t0 + 2.0, t1, step):
        rows.extend([r for r in world.snapshot_rows(day, float(now), fr_table)
                     if r[3] == tid])
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["residual"] = df["act_arr"] - df["rules_arr"]
    df["baseline_arr"] = df["sched_arr"] + df["cur_delay"]
    mid, lo, hi = model.predict(df)
    df["railcast_arr"] = df["rules_arr"] + mid
    lo_c, hi_c = conf.apply(df, df["rules_arr"] + lo, df["rules_arr"] + hi)
    df["lo_arr"], df["hi_arr"] = lo_c, hi_c
    df["baseline"] = df["baseline_arr"]
    df["rules_only"] = df["rules_arr"]
    df = enforce_monotone(df, ["railcast_arr", "lo_arr", "hi_arr"])
    # Serving-layer damping. A single deterministic replay is chaotic - a small
    # change of state can flip one precedence decision and move the whole tail of
    # the run. Successive forecasts of the same station are smoothed before they
    # are shown. This is display behaviour only: every accuracy and coverage
    # number in the evaluation is computed on unsmoothed independent snapshots.
    df = df.sort_values(["to_idx", "now"])
    for c in ("railcast_arr", "lo_arr", "hi_arr"):
        df[c] = df.groupby("to_idx", sort=False)[c].transform(
            lambda x: x.ewm(alpha=0.45, adjust=False).mean())
    return ev.score(df)

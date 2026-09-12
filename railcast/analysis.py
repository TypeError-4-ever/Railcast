"""
The decision surfaces and the experiments behind them.

Everything here is computed from the prototype, not asserted: the cost of a
precedence decision comes from re-running the same day both ways, and the
connection probability is checked against what actually happened.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .dataset import calendar
from .models import HORIZON_LABELS, horizon_bucket
from .noise import draw_day


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


def pick_cascade(world, days):
    """The clearest hold in the test period: one train looped for another."""
    best = None
    for d in days:
        _, truth = world.days[d]
        for tid, run in truth.items():
            for stn, blocker in run.held_by.items():
                m = run.conflict.get(stn, 0.0)
                if not blocker or blocker not in truth:
                    continue
                t = world.sim.trains[tid]
                if stn not in (t.origin, t.dest) and (best is None or m > best[3]):
                    best = (d, tid, stn, m, blocker)
    return best


# --------------------------------------------------------------------------- #
def precedence_experiment(world, day: int, held: str, blocker: str, runs: int = 24):
    """Re-run the same day with the precedence decision reversed.

    Paired Monte Carlo: both policies see identical run-time noise, incidents
    and origin delays, so the difference is the decision and nothing else. The
    cost is reported split three ways - the train that gives way, the train that
    goes first, and every other train on the corridor.
    """
    sim = world.sim
    env, _ = world.days[day]
    pair = frozenset((held, blocker))

    def split(truth):
        out = {"held": 0.0, "blocker": 0.0, "others": 0.0}
        for tid, run in truth.items():
            t = sim.trains[tid]
            if t.dest not in run.arr:
                continue
            d = max(0.0, run.arr[t.dest] - t.sched_arr[t.dest])
            out["held" if tid == held else
                "blocker" if tid == blocker else "others"] += d
        out["total"] = out["held"] + out["blocker"] + out["others"]
        return out

    cur, rev = [], []
    for k in range(runs):
        nz = draw_day(sim, np.random.default_rng(50_000 + k))
        sim.precedence_swap = set()
        cur.append(split(sim.run_day(env, noise=nz)))        # rules as they are
        sim.precedence_swap = {pair}
        rev.append(split(sim.run_day(env, noise=nz)))        # the other call
    sim.precedence_swap = set()
    keys = ("held", "blocker", "others", "total")
    C = {k: np.array([r[k] for r in cur]) for k in keys}
    R = {k: np.array([r[k] for r in rev]) for k in keys}
    delta = {k: R[k] - C[k] for k in keys}
    return dict(
        a=blocker, b=held, runs=runs, day=int(day),
        cost_hold_b=float(C["total"].mean()), cost_hold_a=float(R["total"].mean()),
        se_hold_b=float(C["total"].std(ddof=1) / np.sqrt(runs)),
        se_hold_a=float(R["total"].std(ddof=1) / np.sqrt(runs)),
        delta=float(delta["total"].mean()),
        delta_by={k: float(delta[k].mean()) for k in keys},
        delta_se={k: float(delta[k].std(ddof=1) / np.sqrt(runs)) for k in keys})


# --------------------------------------------------------------------------- #
def _cdf(lo, mid, hi, x):
    """Predictive CDF through the 10th, 50th and 90th percentiles."""
    lo, mid, hi, x = map(np.asarray, (lo, mid, hi, x))
    left = np.maximum(mid - lo, 1e-6)
    right = np.maximum(hi - mid, 1e-6)
    xs = np.stack([lo - 1.5 * left, lo, mid, hi, hi + 1.5 * right], axis=1)
    fs = np.array([0.01, 0.10, 0.50, 0.90, 0.99])
    out = np.empty(len(x))
    for i in range(len(x)):
        out[i] = np.interp(x[i], xs[i], fs)
    return out


def connection_analysis(scored: pd.DataFrame, buffers=(15, 20, 30, 40, 50, 60, 75, 90, 120)):
    d = scored[scored["hops"] >= 3]
    if len(d) > 40000:
        d = d.sample(40000, random_state=3)
    lo, mid, hi = d["lo_arr"].to_numpy(), d["railcast_arr"].to_numpy(), d["hi_arr"].to_numpy()
    sched, act = d["sched_arr"].to_numpy(), d["act_arr"].to_numpy()
    p_curve, obs_curve, allp, allo = [], [], [], []
    for B in buffers:
        p = np.clip(_cdf(lo, mid, hi, sched + B), 0, 1)
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
def calibration_effect(model, conf, test: pd.DataFrame):
    """Coverage before and after stage D, per horizon bucket."""
    _, lo, hi = model.predict(test)
    lo_raw, hi_raw = test["rules_arr"] + lo, test["rules_arr"] + hi
    lo_cal, hi_cal = conf.apply(test, lo_raw, hi_raw)
    act = test["act_arr"]
    raw = ((act >= lo_raw) & (act <= hi_raw)).to_numpy()
    cal = ((act >= lo_cal) & (act <= hi_cal)).to_numpy()
    hb = horizon_bucket(test["sched_lead"].to_numpy())
    groups, r, c = [], [], []
    for k, label in enumerate(HORIZON_LABELS):
        m = hb == k
        if m.sum() < 200:
            continue
        groups.append(label.replace(" (", "\n("))
        r.append(100 * raw[m].mean())
        c.append(100 * cal[m].mean())
    return dict(groups=groups, raw=r, cal=c)


# --------------------------------------------------------------------------- #
def latency_benchmark(world, model, fr_table, days, cycles: int = 120):
    """Time a full-network re-forecast: replay, features, inference."""
    from .dataset import COLUMNS, build_dataset  # noqa: F401
    rng = np.random.default_rng(77)
    per, rows_per, secs = [], [], []
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
        dt = time.perf_counter() - t0
        secs.append(dt)
        rows_per.append(len(df))
        per.append(dt * 1000.0 / len(df))
    return dict(per_forecast_ms=np.array(per), cycle_s=float(np.mean(secs)),
                mean_rows=float(np.mean(rows_per)), trains=len(world.trains),
                cycles=len(secs))


# --------------------------------------------------------------------------- #
def console_card(world, scored: pd.DataFrame, connection):
    """The numbers behind the passenger-facing card, from a real test forecast.

    Picks a forecast that is worth showing: the train is running, it is late,
    something downstream is holding it, and the connection is a genuine call
    rather than a certainty.
    """
    d = scored[(scored["hops"] >= 6) & (scored["started"] == 1)]
    key = d.groupby(["day", "train", "now"]).size()
    key = key[key >= 8]
    rng = np.random.default_rng(5)
    idx = list(key.index)
    if len(idx) > 400:
        idx = [idx[i] for i in rng.choice(len(idx), 400, replace=False)]

    best, best_score = None, -1e9
    for day, tid, now in idx:
        g = scored[(scored["day"] == day) & (scored["train"] == tid) &
                   (scored["now"] == now)].sort_values("hops")
        t = world.sim.trains[tid]
        _, truth = world.days[day]
        run = truth[tid]
        at = int(g["from_idx"].iloc[0])
        k = t.path.index(at)
        blocker, blocked_at = "", ""
        for stn in t.path[k + 1:]:
            who = run.held_by.get(stn)
            if who and run.conflict.get(stn, 0.0) > 4:
                blocker, blocked_at = who, world.cor.code(stn)
                break
        dest = g.iloc[-1]
        p = float(np.clip(_cdf([dest["lo_arr"]], [dest["railcast_arr"]],
                               [dest["hi_arr"]], [dest["sched_arr"] + 45])[0], 0, 1))
        score = (3.0 if blocker else 0.0) + (2.0 if 0.35 < p < 0.90 else 0.0)             + min(float(g["cur_delay"].iloc[0]), 45) / 30.0
        if score > best_score:
            best_score, best = score, (day, tid, now, g, at, blocker, blocked_at, p, dest)

    day, tid, now, g, at, blocker, blocked_at, p_conn, dest = best
    t = world.sim.trains[tid]
    nxt = g.iloc[0]
    return dict(train=tid, name=t.name, now=float(now), at=world.cor.code(at),
                cur_delay=float(g["cur_delay"].iloc[0]),
                momentum=float(g["delay_momentum"].iloc[0]),
                next_code=nxt["to_code"], next_eta=float(nxt["railcast_arr"]),
                next_lo=float(nxt["lo_arr"]), next_hi=float(nxt["hi_arr"]),
                dest_code=dest["to_code"], dest_eta=float(dest["railcast_arr"]),
                dest_lo=float(dest["lo_arr"]), dest_hi=float(dest["hi_arr"]),
                dest_sched=float(dest["sched_arr"]),
                blocker=blocker, blocked_at=blocked_at, p_connection=p_conn,
                rows=g[["to_code", "sched_arr", "baseline", "railcast_arr",
                        "lo_arr", "hi_arr", "act_arr"]].copy(),
                day=int(day))


# --------------------------------------------------------------------------- #
def journey_track(world, model, conf, fr_table, day: int, tid: str,
                  step: float = 30.0):
    """Re-forecast one journey every `step` minutes, start to finish.

    This is the 30-second loop running for real on a single train: each new
    actual arrival updates the state and the rest of the run is forecast again.
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

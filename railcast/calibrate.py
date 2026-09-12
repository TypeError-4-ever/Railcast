"""
Fit the simulator's delay parameters instead of guessing them.

Two objectives, in order of how much they are worth:

  observed   Punctuality measured from running data collected through the live
             feed. This is the real thing. It needs a collector to have been
             running - see railcast.feeds.live - because no public archive of
             past Indian Railways arrivals exists to download.

  target     A punctuality profile the operator states: what fraction of each
             class arrives within fifteen minutes. Weaker than observed data,
             but it turns hand-tuning into something written down, reproducible
             and checkable. The target is recorded in the fitted config, so
             anyone reading a result can see what it was fitted against.

The fit is coordinate descent over a small number of parameters. It is slow
enough to be worth caching and fast enough to run before a deck deadline.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .simconfig import SimConfig
from .simulator import Simulator

# Indian Railways does not publish punctuality split this way, so this is a
# stated assumption, not a measurement. Override it with --target-file.
ILLUSTRATIVE_TARGET = {
    "RAJDHANI": 0.75, "DURONTO": 0.72, "SHATABDI": 0.78,
    "SUPERFAST": 0.62, "EXPRESS": 0.55, "PASSENGER": 0.50,
}

# Punctuality alone does not pin down a delay distribution: a model can put 75%
# of runs inside fifteen minutes and the rest six hours late. These bound the
# tail. Minutes late at destination, across all classes.
TAIL_TARGET = {"median": 12.0, "p90": 75.0, "p99": 220.0}

KNOBS = [
    ("priority_protection", (1.0, 4.0)),
    ("friction_scale", (0.10, 0.95)),
    ("incidents_per_1000km", (0.10, 1.60)),
    ("precedence_lookahead_min", (10.0, 60.0)),
    ("max_hold_min", (8.0, 40.0)),
]


# --------------------------------------------------------------------------- #
def observe(cor, trains, envs, cfg: SimConfig, days: int, seed: int = 3,
            tolerance_min: float = 15.0):
    """Simulated punctuality by class, and the shape of the delay tail."""
    sim = Simulator(cor, trains, cfg)
    rng = np.random.default_rng(seed)
    hit: dict[str, list] = {}
    late_all: list[float] = []
    for d in range(days):
        env = envs[d] if isinstance(envs, dict) else envs[d % len(envs)]
        for tid, run in sim.run_day(env, rng).items():
            t = sim.trains[tid]
            if t.dest in run.arr:
                late = run.arr[t.dest] - t.sched_arr[t.dest]
                hit.setdefault(t.klass, []).append(late < tolerance_min)
                late_all.append(late)
    punc = {k: float(np.mean(v)) for k, v in hit.items() if v}
    arr = np.array(late_all) if late_all else np.array([0.0])
    tail = {"median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99))}
    return punc, tail


def punctuality(cor, trains, envs, cfg: SimConfig, days: int, seed: int = 3,
                tolerance_min: float = 15.0) -> dict:
    return observe(cor, trains, envs, cfg, days, seed, tolerance_min)[0]


def _loss(got: dict, want: dict, tail: dict | None = None,
          tail_want: dict | None = None) -> float:
    keys = [k for k in want if k in got]
    if not keys:
        return 1e9
    punc = float(np.sqrt(np.mean([(got[k] - want[k]) ** 2 for k in keys])))
    if tail is None or tail_want is None:
        return punc
    # relative error on each quantile, so a six-hour p99 cannot hide behind a
    # respectable on-time percentage
    rel = [abs(tail[q] - tail_want[q]) / max(tail_want[q], 1.0) for q in tail_want]
    return punc + 0.35 * float(np.mean(rel))


def fit(cor, trains, envs, target: dict | None = None, days: int = 20,
        rounds: int = 2, probes: int = 5, base: SimConfig | None = None,
        verbose: bool = True) -> SimConfig:
    """Coordinate descent over KNOBS against a punctuality profile."""
    want = target or ILLUSTRATIVE_TARGET
    cfg = base or SimConfig()
    tail_want = TAIL_TARGET
    p0, t0 = observe(cor, trains, envs, cfg, days)
    best = _loss(p0, want, t0, tail_want)
    if verbose:
        print(f"  start loss {best:.4f}")
    for r in range(rounds):
        for knob, (lo, hi) in KNOBS:
            cur = getattr(cfg, knob)
            grid = np.linspace(lo, hi, probes)
            for v in grid:
                trial = replace(cfg, **{knob: float(v)})
                pg, tg = observe(cor, trains, envs, trial, days)
                score = _loss(pg, want, tg, tail_want)
                if score < best - 1e-4:
                    best, cfg = score, trial
            if verbose:
                print(f"  round {r + 1} {knob:26s} {cur:7.3f} -> "
                      f"{getattr(cfg, knob):7.3f}   loss {best:.4f}")
    got, tail = observe(cor, trains, envs, cfg, days)
    cfg = replace(cfg, fitted=True,
                  fitted_against=("stated punctuality target"
                                  if target is None else "supplied target"),
                  fit_residual={**{k: round(got.get(k, float("nan")) - v, 4)
                                   for k, v in want.items() if k in got},
                                **{f"delay_{q}_min": round(tail[q], 1)
                                   for q in tail}})
    if verbose:
        print("  delay: median %.0f  p90 %.0f  p99 %.0f min (target %.0f/%.0f/%.0f)"
              % (tail["median"], tail["p90"], tail["p99"],
                 tail_want["median"], tail_want["p90"], tail_want["p99"]))
        print("  achieved: " + "  ".join(
            f"{k} {100 * got[k]:.0f}% (want {100 * want[k]:.0f}%)"
            for k in sorted(got) if k in want))
    return cfg


# --------------------------------------------------------------------------- #
# fitting against arrivals actually observed
# --------------------------------------------------------------------------- #
FIT_QUANTILES = (25, 50, 75, 90, 95)
MIN_OBSERVATIONS = 150      # below this, refuse: the fit would be noise
TRUSTWORTHY = 2000          # below this, fit but say so
MIN_PER_CLASS = 30          # below this, do not fit that class separately


def target_from_store(store, trains=None, halts_only: bool = True) -> dict:
    """Turn collected arrivals into a fitting target.

    Only rows the operator has actually reported count. Rows the feed marks
    `upcoming` carry the deployed system's own projection, and fitting on those
    would be fitting to a competitor's forecast rather than to what happened.
    """
    from .validate import observed_delays

    rows = [r for r in store.read()
            if r.observed and r.event == "arrival" and r.delay_min is not None
            and (r.is_halt or not halts_only)]
    if len(rows) < MIN_OBSERVATIONS:
        raise RuntimeError(
            f"only {len(rows)} observed arrivals at booked halts; need at least "
            f"{MIN_OBSERVATIONS} before a fit means anything. Keep the collector "
            "running:\n"
            "    python -m railcast.feeds collect --trains 12951,...")
    delays = np.array([r.delay_min for r in rows], dtype=float)

    by_class: dict[str, tuple] = {}
    if trains:
        klass = {t.number: t.klass for t in trains}
        buckets: dict[str, list] = {}
        for r in rows:
            k = klass.get(r.train_number)
            if k:
                buckets.setdefault(k, []).append(r.delay_min < 15.0)
        by_class = {k: (float(np.mean(v)), len(v))
                    for k, v in buckets.items() if len(v) >= MIN_PER_CLASS}

    return {
        "source": "observed",
        "n": len(rows),
        "quantiles": {q: float(np.percentile(delays, q)) for q in FIT_QUANTILES},
        "within_15": float(np.mean(delays < 15.0)),
        "by_class": by_class,
        "trains_seen": sorted({r.train_number for r in rows}),
        "halts_only": halts_only,
    }


def _observed_loss(sim: np.ndarray, target: dict) -> float:
    """Relative error on each quantile, plus the on-time fraction."""
    if sim.size == 0:
        return 1e9
    rel = [abs(float(np.percentile(sim, q)) - v) / max(abs(v), 5.0)
           for q, v in target["quantiles"].items()]
    punc = abs(float(np.mean(sim < 15.0)) - target["within_15"])
    return float(np.mean(rel)) + 1.5 * punc


def fit_observed(cor, trains, envs, target: dict, days: int = 20, rounds: int = 2,
                 probes: int = 5, base: SimConfig | None = None,
                 verbose: bool = True) -> SimConfig:
    """Coordinate descent against the distribution actually observed.

    Simulated and observed delays are both taken at booked halts, station by
    station. Comparing mid-run delay against end-of-run delay would flatter the
    simulator, so the two sides are measured the same way.
    """
    from .validate import simulated_delays

    cfg = base or SimConfig()
    halts = target.get("halts_only", True)

    def score(c: SimConfig) -> float:
        return _observed_loss(
            simulated_delays(cor, trains, envs, c, days, halts_only=halts), target)

    best = score(cfg)
    if verbose:
        print(f"  fitting to {target['n']:,} observed arrivals from "
              f"{len(target['trains_seen'])} trains")
        print(f"  start loss {best:.4f}")
    for r in range(rounds):
        for knob, (lo, hi) in KNOBS:
            cur = getattr(cfg, knob)
            for v in np.linspace(lo, hi, probes):
                trial = replace(cfg, **{knob: float(v)})
                sc = score(trial)
                if sc < best - 1e-4:
                    best, cfg = sc, trial
            if verbose:
                print(f"  round {r + 1} {knob:26s} {cur:7.3f} -> "
                      f"{getattr(cfg, knob):7.3f}   loss {best:.4f}")

    got = simulated_delays(cor, trains, envs, cfg, days, halts_only=halts)
    residual = {f"p{q}_gap_min": round(float(np.percentile(got, q)) - v, 1)
                for q, v in target["quantiles"].items()}
    residual["within_15_gap"] = round(float(np.mean(got < 15.0))
                                      - target["within_15"], 3)
    residual["n_observed"] = target["n"]
    note = ("observed arrivals, RailRadar live feed"
            if target["n"] >= TRUSTWORTHY else
            f"observed arrivals ({target['n']}) - below {TRUSTWORTHY}, "
            "treat as provisional")
    cfg = replace(cfg, fitted=True, fitted_against=note, fit_residual=residual)
    if verbose:
        print("  after fit:  " + "  ".join(
            f"p{q} {float(np.percentile(got, q)):.0f}m (obs {v:.0f}m)"
            for q, v in target["quantiles"].items()))
        if target["n"] < TRUSTWORTHY:
            print(f"  NOTE: {target['n']} observations is thin. Provisional.")
    return cfg


# --------------------------------------------------------------------------- #
def target_from_live(store) -> dict:
    """Punctuality profile measured from collected running data."""
    rows = store.read()
    if not rows:
        raise RuntimeError(
            "No running data collected yet. There is no public archive of past "
            "Indian Railways arrivals to download, so this has to be gathered:\n"
            "    python -m railcast.feeds collect --trains 12951,12952,12903\n"
            "run on a schedule. Until then, fit against a stated target with\n"
            "    python -m railcast.feeds calibrate")
    by_train: dict[str, float] = {}
    for r in rows:
        if r.delay_min is not None:
            by_train[r.train_number] = float(r.delay_min)
    return by_train


def load_target(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))

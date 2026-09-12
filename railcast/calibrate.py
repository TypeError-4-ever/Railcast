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

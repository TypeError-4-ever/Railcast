"""
Pre-drawn stochastic terms for one operating day.

Drawing every random term up front, keyed by train and by position, makes a day
reproducible independently of the order the event loop happens to visit trains.
That is what lets the precedence experiment compare two dispatching decisions on
the *same* day rather than on two different ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .simconfig import DEFAULT, SimConfig
from .trains import CLASSES


@dataclass
class DayNoise:
    origin_delay: dict = field(default_factory=dict)   # tid -> minutes
    friction: dict = field(default_factory=dict)       # tid -> run-time multiplier
    block: dict = field(default_factory=dict)          # tid -> array over block seq
    dwell: dict = field(default_factory=dict)          # tid -> array over path
    incident: dict = field(default_factory=dict)       # (tid, station) -> minutes


def _pad_fraction(t) -> float:
    """Recovery padding as a fraction of the train's own running time."""
    pad = sum(t.pad.values())
    run = t.sched_arr.get(t.dest, 0.0) - t.dep_minute
    if run <= 0 or pad <= 0:
        return CLASSES[t.klass]["pad"]
    return float(min(0.30, pad / max(run - pad, 1.0)))


def draw_day(sim, rng: np.random.Generator,
             cfg: SimConfig | None = None) -> DayNoise:
    cfg = cfg or DEFAULT
    n = DayNoise()
    for tid, t in sim.trains.items():
        n.origin_delay[tid] = float(max(0.0, rng.gamma(1.1, cfg.origin_delay_mean_min / 1.1) - 1.2))
        # one draw of crew, loco, load and permanent restrictions for the whole
        # run, so friction is correlated along the journey. The centre is a
        # fraction of the recovery padding this train actually carries, measured
        # from its own timetable where the corridor came from a feed.
        n.friction[tid] = float(np.exp(rng.normal(
            _pad_fraction(t) * cfg.friction_scale, cfg.friction_sd)))
        n.block[tid] = np.exp(rng.normal(0.0, cfg.block_noise_sd, len(sim.seqs[tid])))
        n.dwell[tid] = rng.gamma(1.3, cfg.dwell_overrun_mean_min / 1.3, len(t.path))
        # Incidents scale with distance run, not with how finely the route
        # happens to be divided into stations.
        path = t.path
        for a, b in zip(path[:-1], path[1:]):
            km = abs(sim.cor.stations[b].km - sim.cor.stations[a].km)
            if rng.random() < cfg.incidents_per_1000km * km / 1000.0:
                n.incident[(tid, b)] = float(rng.gamma(2.0, cfg.incident_mean_min / 2.0))
    return n

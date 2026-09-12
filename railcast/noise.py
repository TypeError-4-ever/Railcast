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

from .trains import CLASSES


@dataclass
class DayNoise:
    origin_delay: dict = field(default_factory=dict)   # tid -> minutes
    friction: dict = field(default_factory=dict)       # tid -> run-time multiplier
    block: dict = field(default_factory=dict)          # tid -> array over block seq
    dwell: dict = field(default_factory=dict)          # tid -> array over path
    incident: dict = field(default_factory=dict)       # (tid, station) -> minutes


def draw_day(sim, rng: np.random.Generator) -> DayNoise:
    n = DayNoise()
    for tid, t in sim.trains.items():
        n.origin_delay[tid] = float(max(0.0, rng.gamma(1.1, 3.4) - 1.2))
        # one draw of crew, loco, load and permanent restrictions for the whole
        # run, so friction is correlated along the journey
        n.friction[tid] = float(np.exp(rng.normal(CLASSES[t.klass]["pad"] * 0.45, 0.025)))
        n.block[tid] = np.exp(rng.normal(0.0, 0.08, len(sim.seqs[tid])))
        n.dwell[tid] = rng.gamma(1.3, 0.9, len(t.path))
        for k, s in enumerate(t.path):
            if rng.random() < 0.008:
                n.incident[(tid, s)] = float(rng.gamma(2.0, 6.0))
    return n

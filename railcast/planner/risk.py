"""
The chance of making each connection, and of reaching B at all.

    P(make)    = P(arrival at C + delay + transfer <= departure from C + delay)
    P(reach B) = product over legs of P(seat) x product over changes of P(make)

Delays come from the planner's delay model as three quantiles per station and
are sampled by Monte Carlo. A train is never assumed to leave early. Legs are
treated as independent - fog that delays one train often delays the next too,
so P(make) is somewhat optimistic in bad weather.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .. import config
from ..dist import quantile_sample
from . import delays as dl
from .network import Network
from .search import Itinerary


@dataclass
class Scored:
    it: Itinerary
    dep_q: list = field(default_factory=list)    # per leg: delay quantiles leaving
    arr_q: list = field(default_factory=list)    # per leg: delay quantiles arriving
    p_make: list = field(default_factory=list)   # per change
    seats: list = field(default_factory=list)    # per leg: SeatStatus or None
    p_seat: list = field(default_factory=list)   # per leg: float or None
    fare: float | None = None

    @property
    def seats_checked(self) -> bool:
        return bool(self.p_seat) and all(p is not None for p in self.p_seat)

    @property
    def p_connections(self) -> float:
        return float(np.prod(self.p_make)) if self.p_make else 1.0

    @property
    def p_reach(self) -> float:
        """Seats and connections when seats were checked; connections alone if not."""
        p = self.p_connections
        if self.seats_checked:
            p *= float(np.prod(self.p_seat))
        return p

    @property
    def arrival(self) -> tuple[float, float, float]:
        """Predicted arrival at B: 10th, 50th, 90th percentile, minutes."""
        lo, mid, hi = self.arr_q[-1]
        return self.it.arr + lo, self.it.arr + mid, self.it.arr + hi


def _features(net: Network, it: Itinerary, day0: date) -> list[tuple]:
    rows = []
    for leg in it.legs:
        info = net.trains[leg.train]
        seq = net.calls[leg.train]
        for idx, event in ((leg.i, 1), (leg.j, 0)):
            c = seq[idx]
            t = c.dep if event else c.arr
            st = net.stations.get(c.code)
            zone = (st.zone if st and st.zone else "") or info.zone
            k, z = dl.encode(info.klass, zone)
            when = day0 + timedelta(days=leg.k + int(t // 1440))
            rows.append((k, (t - seq[0].dep) / 60.0, z, dl.night(t),
                         when.month, when.weekday(), event))
    return rows


def score(net: Network, its: list[Itinerary], day0: date, model=None,
          seed: int = 7) -> list[Scored]:
    if not its:
        return []
    model = model or dl.load()
    rows, spans = [], []
    for it in its:
        r = _features(net, it, day0)
        spans.append((len(rows), len(r)))
        rows.extend(r)
    q10, q50, q90 = model.predict(pd.DataFrame(rows, columns=dl.FEATURES))
    rng = np.random.default_rng(seed)
    out = []
    for it, (s0, n) in zip(its, spans):
        qs = list(zip(q10[s0:s0 + n], q50[s0:s0 + n], q90[s0:s0 + n]))
        sc = Scored(it, dep_q=qs[0::2], arr_q=qs[1::2])
        for a in range(len(it.legs) - 1):
            sc.p_make.append(p_make(net, it.legs[a], it.legs[a + 1],
                                    sc.arr_q[a], sc.dep_q[a + 1], rng))
        out.append(sc)
    return out


def p_make(net: Network, arriving, departing, arr_q, dep_q, rng,
           n: int = config.MONTE_CARLO_DRAWS) -> float:
    """Chance of reaching the departing train in time, given both delay spreads."""
    transfer = net.transfer_min(arriving.to(net), departing.frm(net))
    d1 = quantile_sample(*([v] for v in arr_q), n, rng)[0]
    d2 = np.maximum(quantile_sample(*([v] for v in dep_q), n, rng)[0], 0.0)
    ready = arriving.arr(net) + d1 + transfer
    return float(np.mean(ready <= departing.dep(net) + d2))

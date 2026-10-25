"""
Itineraries from A to B with up to two changes.

The change station can be anywhere a passenger can get off - not only on the
direct A-B line. That is the point: when every train on the direct line is
full, a train on another corridor draws from a different crowd.

The search is a meet-in-the-middle over one timetable that repeats daily:

  forward    every station reachable from A on one train, with arrival time
  backward   every station from which one train reaches B
  1 change   forward meets backward at the same station, or across a city
  2 changes  each train is scanned once: boarded where forward reaches it,
             left where backward picks up

Times are minutes from midnight of the query date. A train's start date is the
query date plus `k` days; trains are assumed to run daily.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .. import config
from .network import Network


@dataclass(frozen=True)
class Leg:
    train: str
    k: int               # start date = query date + k days
    i: int               # boarding call index
    j: int               # alighting call index

    def dep(self, net: Network) -> float:
        return net.calls[self.train][self.i].dep + 1440.0 * self.k

    def arr(self, net: Network) -> float:
        return net.calls[self.train][self.j].arr + 1440.0 * self.k

    def frm(self, net: Network) -> str:
        return net.calls[self.train][self.i].code

    def to(self, net: Network) -> str:
        return net.calls[self.train][self.j].code

    def km(self, net: Network) -> float:
        c = net.calls[self.train]
        return c[self.j].km - c[self.i].km


@dataclass
class Itinerary:
    legs: list[Leg]
    dep: float
    arr: float
    km: float
    layovers: list[float] = field(default_factory=list)
    off_line: bool = False   # changes at a station no direct A-B train serves

    @property
    def changes(self) -> int:
        return len(self.legs) - 1

    @property
    def duration(self) -> float:
        return self.arr - self.dep

    @property
    def key(self) -> tuple:
        return tuple((l.train, l.k) for l in self.legs)


@dataclass
class Query:
    origin: str
    dest: str
    t0: float = 0.0                       # earliest departure, minutes
    window: float = config.DEFAULT_WINDOW_H * 60
    max_changes: int = 1
    klass: str | None = None              # "SL", "3A" ... None = any
    nearby: bool = True                   # include other stations in the same city
    include_unreserved: bool = False


# --------------------------------------------------------------------------- #
class Search:
    def __init__(self, net: Network, q: Query):
        self.net, self.q = net, q
        self.origins = self._expand(q.origin)
        self.dests = self._expand(q.dest)
        a, b = net.stations[q.origin], net.stations[q.dest]
        self.ab = net.km_between(q.origin, q.dest)
        self._a, self._b = (a.lat, a.lon), (b.lat, b.lon)
        self._geo: dict[str, bool] = {}

    def _expand(self, code: str) -> set[str]:
        return {code, *self.net.city_mates(code)} if self.q.nearby else {code}

    def _same_city(self, a: str, b: str) -> bool:
        """A train ride inside one city is a transfer, not a leg."""
        ca = self.net.city_of.get(a)
        return ca is not None and ca == self.net.city_of.get(b)

    def _ok_train(self, num: str) -> bool:
        t = self.net.trains[num]
        if not t.reserved and not self.q.include_unreserved:
            return False
        if self.q.klass and t.classes and self.q.klass not in t.classes:
            return False
        return True

    def _near_line(self, code: str) -> bool:
        """Is a change at `code` a plausible detour between A and B?"""
        hit = self._geo.get(code)
        if hit is None:
            s = self.net.stations.get(code)
            if s is None:
                hit = False
            else:
                from ..feeds.base import haversine_km
                p = (s.lat, s.lon)
                hit = (haversine_km(self._a, p) + haversine_km(p, self._b)
                       <= config.DETOUR_GEO * max(self.ab, 50.0))
            self._geo[code] = hit
        return hit

    # ------------------------------------------------------------------ #
    def forward(self):
        """Station -> legs from A that alight there, sorted by arrival."""
        net, q = self.net, self.q
        fwd: dict[str, list] = {}
        for o in self.origins:
            for num, i in net.halts_at.get(o, []):
                if not self._ok_train(num):
                    continue
                seq = net.calls[num]
                dep = seq[i].dep
                for k in range(math.ceil((q.t0 - dep) / 1440),
                               math.floor((q.t0 + q.window - dep) / 1440) + 1):
                    for j in range(i + 1, len(seq)):
                        if seq[j].halt and not self._same_city(o, seq[j].code):
                            leg = Leg(num, k, i, j)
                            fwd.setdefault(seq[j].code, []).append((leg.arr(net), leg))
        for v in fwd.values():
            v.sort(key=lambda x: x[0])
        return fwd

    def backward(self):
        """Station -> (departure time of day, train, i, j) of trains reaching B."""
        net = self.net
        bwd: dict[str, list] = {}
        for d in self.dests:
            for num, j in net.halts_at.get(d, []):
                if not self._ok_train(num):
                    continue
                seq = net.calls[num]
                for i in range(j):
                    if (seq[i].halt and seq[i].code not in self.dests
                            and not self._same_city(seq[i].code, d)):
                        bwd.setdefault(seq[i].code, []).append(
                            (seq[i].dep % 1440, num, i, j))
        for v in bwd.values():
            v.sort(key=lambda x: x[0])
        return bwd

    def _departures(self, entries, earliest: float, latest: float):
        """Backward entries departing in [earliest, latest], with their k."""
        if not entries or latest < earliest:
            return
        tods = [e[0] for e in entries]
        lo = earliest % 1440
        span = latest - earliest
        start = bisect.bisect_left(tods, lo)
        n = len(entries)
        for step in range(n):
            e = entries[(start + step) % n]
            ahead = (e[0] - lo) % 1440
            if ahead > span:
                break
            when = earliest + ahead
            off = self.net.calls[e[1]][e[2]].dep
            yield e, int(round((when - off) / 1440.0)), when

    def _changes_at(self, code: str):
        """Where a passenger alighting at `code` can board next, and how long it takes."""
        yield code, self.net.transfer_min(code, code)
        for m in self.net.city_mates(code):
            yield m, self.net.transfer_min(code, m)

    # ------------------------------------------------------------------ #
    def run(self) -> list[Itinerary]:
        net, q = self.net, self.q
        fwd, bwd = self.forward(), self.backward()
        found: dict[tuple, Itinerary] = {}

        def keep(legs):
            it = _assemble(net, legs)
            old = found.get(it.key)
            if old is None or min(it.layovers, default=1e9) > min(old.layovers, default=1e9):
                found[it.key] = it

        # direct
        for d in self.dests:
            for _, leg in fwd.get(d, []):
                keep([leg])

        # one change
        if q.max_changes >= 1:
            for c, arrivals in fwd.items():
                if c in self.dests or not self._near_line(c):
                    continue
                for c2, tt in self._changes_at(c):
                    entries = bwd.get(c2)
                    if not entries:
                        continue
                    for arr, leg1 in arrivals:
                        for e, k2, _ in self._departures(
                                entries, arr + tt, arr + config.MAX_LAYOVER):
                            if e[1] != leg1.train:
                                keep([leg1, Leg(e[1], k2, e[2], e[3])])

        # two changes
        if q.max_changes >= 2:
            self._two_changes(fwd, bwd, keep)

        return self._cap(_pareto(list(found.values())))

    def _two_changes(self, fwd, bwd, keep):
        net = self.net
        # where a forward leg can hand over, keyed by the station it hands over at
        board: dict[str, list] = {}
        for c, arrivals in fwd.items():
            if c in self.dests or not self._near_line(c):
                continue
            for c2, tt in self._changes_at(c):
                board.setdefault(c2, []).extend((arr + tt, arr, leg) for arr, leg in arrivals)
        for v in board.values():
            v.sort(key=lambda x: x[0])
        for num, seq in net.calls.items():
            if not self._ok_train(num):
                continue
            for i, call in enumerate(seq):
                if not call.halt or call.code not in board:
                    continue
                for ready, arr1, leg1 in board[call.code]:
                    if leg1.train == num:
                        continue
                    k = math.ceil((ready - call.dep) / 1440)
                    if call.dep + 1440 * k - arr1 > config.MAX_LAYOVER:
                        continue
                    for j in range(i + 1, len(seq)):
                        e2 = seq[j]
                        if (not e2.halt or not self._near_line(e2.code)
                                or self._same_city(call.code, e2.code)):
                            continue
                        arr2 = e2.arr + 1440 * k
                        for c3, tt in self._changes_at(e2.code):
                            entries = bwd.get(c3)
                            if not entries:
                                continue
                            for e, k3, _ in self._departures(
                                    entries, arr2 + tt, arr2 + config.MAX_LAYOVER):
                                if e[1] not in (num, leg1.train):
                                    keep([leg1, Leg(num, k, i, j), Leg(e[1], k3, e[2], e[3])])

    def _cap(self, its: list[Itinerary]) -> list[Itinerary]:
        """Detour caps, the off-line tag, and a bounded candidate list."""
        if not its:
            return []
        direct = [it for it in its if it.changes == 0]
        ref_km = min((it.km for it in direct), default=self.ab * 1.25)
        fastest = min(it.duration for it in its)
        max_dur = max(fastest * config.DETOUR_TIME, fastest + config.DETOUR_TIME_FLOOR)
        on_line = set()
        for it in direct:
            seq = self.net.calls[it.legs[0].train]
            on_line |= {c.code for c in seq[it.legs[0].i:it.legs[0].j + 1]}
        out = []
        for it in its:
            if it.changes and it.km > config.DETOUR_KM * ref_km:
                continue
            if it.duration > max_dur:
                continue
            if it.changes and direct:
                it.off_line = any(l.to(self.net) not in on_line for l in it.legs[:-1])
            out.append(it)
        out.sort(key=lambda it: (it.arr, it.changes, -min(it.layovers, default=1e9)))
        return out[:config.MAX_CANDIDATES]


def _pareto(its: list[Itinerary]) -> list[Itinerary]:
    """Drop an option when another leaves no earlier, arrives no later, changes
    no more often and has no tighter connection.

    The last test keeps a dominated option that is safer. It also means a feeder
    train to catch a train you could have boarded at A is dropped even though it
    might have seats on a different quota - a trade-off until seat checks feed
    back into the search.
    """
    def slack(it):
        return min(it.layovers, default=1e9)

    its = sorted(its, key=lambda it: (it.arr, it.changes, -it.dep, -slack(it)))
    kept: list[Itinerary] = []
    for it in its:
        if not any(k.dep >= it.dep and k.arr <= it.arr and k.changes <= it.changes
                   and slack(k) >= slack(it) for k in kept):
            kept.append(it)
    return kept


def _assemble(net: Network, legs: list[Leg]) -> Itinerary:
    lay = [b.dep(net) - a.arr(net) for a, b in zip(legs, legs[1:])]
    return Itinerary(legs=legs, dep=legs[0].dep(net), arr=legs[-1].arr(net),
                     km=sum(l.km(net) for l in legs), layovers=lay)


def search(net: Network, q: Query) -> list[Itinerary]:
    return Search(net, q).run()

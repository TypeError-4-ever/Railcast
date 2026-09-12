"""Train roster and timetable construction for the prototype corridor."""
from __future__ import annotations

from dataclasses import dataclass

from .corridor import CORRIDOR, Corridor

# priority: 1 highest. Used by the conflict simulator for precedence.
CLASSES = {
    "RAJDHANI":  dict(priority=1, max_speed=130.0, pad=0.075),
    "DURONTO":   dict(priority=1, max_speed=130.0, pad=0.085),
    "SHATABDI":  dict(priority=2, max_speed=130.0, pad=0.085),
    "SUPERFAST": dict(priority=3, max_speed=110.0, pad=0.12),
    "EXPRESS":   dict(priority=4, max_speed=100.0, pad=0.16),
    "PASSENGER": dict(priority=5, max_speed=75.0,  pad=0.20),
}

# (number, name, class, origin_idx, dest_idx, departure hh:mm, halt density)
# halt density: fraction of intermediate stations at which the train stops.
ROSTER_RAW = [
    ("12951", "Mumbai Rajdhani",        "RAJDHANI",  0, 34, "16:25", 0.18),
    ("12953", "August Kranti Rajdhani", "RAJDHANI",  1, 34, "17:15", 0.22),
    ("12909", "Garib Rath",             "SUPERFAST", 1, 33, "15:35", 0.30),
    ("12925", "Paschim Express",        "SUPERFAST", 0, 34, "11:05", 0.55),
    ("12471", "Swaraj Express",         "SUPERFAST", 0, 32, "20:40", 0.50),
    ("19020", "Dehradun Express",       "EXPRESS",   0, 34, "06:10", 0.80),
    ("12915", "Ashram Express",         "SUPERFAST", 0, 23, "15:20", 0.45),
    ("12903", "Golden Temple Mail",     "SUPERFAST", 0, 34, "07:30", 0.60),
    ("12264", "Duronto Express",        "DURONTO",   1, 34, "19:55", 0.12),
    ("12931", "Ahmedabad Duronto",      "DURONTO",   0, 23, "19:25", 0.15),
    ("59023", "Delhi-Kota Passenger",   "PASSENGER", 0, 12, "05:30", 0.95),
    ("59811", "Kota-Ratlam Passenger",  "PASSENGER", 12, 18, "09:10", 0.95),
    ("19038", "Avadh Express",          "EXPRESS",   6, 34, "04:45", 0.75),
    ("12danger", "_unused",             "EXPRESS",   0, 0,  "00:00", 0.0),
    ("12danger2", "_unused",            "EXPRESS",   0, 0,  "00:00", 0.0),
    ("22209", "Mumbai AC Superfast",    "SUPERFAST", 0, 34, "13:10", 0.28),
    ("12danger3", "_unused",            "EXPRESS",   0, 0,  "00:00", 0.0),
    ("19107", "Yoga Express",           "EXPRESS",   0, 34, "22:15", 0.85),
    ("12danger4", "_unused",            "EXPRESS",   0, 0,  "00:00", 0.0),
    ("14805", "Barmer AC Express",      "SUPERFAST", 0, 18, "09:50", 0.40),
]
ROSTER_RAW = [r for r in ROSTER_RAW if not r[1].startswith("_")]

# Down direction trains (Mumbai -> Delhi) are mirrored automatically.


@dataclass
class Train:
    number: str
    name: str
    klass: str
    up: bool                     # True = Delhi -> Mumbai
    origin: int
    dest: int
    dep_minute: float            # minutes after 00:00 on day 0
    halts: list[int]             # station indices where it stops
    sched_arr: dict[int, float]  # station idx -> scheduled arrival, minutes
    sched_dep: dict[int, float]
    pad: dict[int, float]        # recovery padding built into section ending at idx

    @property
    def priority(self) -> int:
        return CLASSES[self.klass]["priority"]

    @property
    def max_speed(self) -> float:
        return CLASSES[self.klass]["max_speed"]

    @property
    def path(self) -> list[int]:
        step = 1 if self.up else -1
        return list(range(self.origin, self.dest + step, step))


def _hhmm(s: str) -> float:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _dwell(cor: Corridor, idx: int, klass: str) -> float:
    st = cor.stations[idx]
    base = 2.0
    if st.is_junction:
        base = 5.0
    if st.code in ("KOTA", "RTM", "BRC", "ST", "MTJ", "NAD"):
        base = 8.0 if CLASSES[klass]["priority"] > 2 else 5.0
    return base


def build_timetable(cor: Corridor, number: str, name: str, klass: str,
                    origin: int, dest: int, dep: float, halt_density: float,
                    rng) -> Train:
    up = dest > origin
    step = 1 if up else -1
    path = list(range(origin, dest + step, step))
    spec = CLASSES[klass]

    halts = {origin, dest}
    for i in path[1:-1]:
        st = cor.stations[i]
        p = halt_density * (1.35 if st.is_junction else 0.85)
        if rng.random() < min(p, 0.97):
            halts.add(i)
    halts = sorted(halts, reverse=not up)

    sched_arr: dict[int, float] = {}
    sched_dep: dict[int, float] = {origin: dep}
    pad: dict[int, float] = {}
    t = dep
    for a, b in zip(path[:-1], path[1:]):
        lo, hi = (a, b) if up else (b, a)
        stopping = b in halts
        fr = cor.free_run_minutes(lo, hi, spec["max_speed"], stopping=stopping)
        padding = fr * spec["pad"]
        # extra recovery time before the terminal and at big junctions
        if b == dest:
            padding += 3.0
        elif cor.stations[b].is_junction:
            padding += 2.0
        pad[b] = padding
        t += fr + padding
        sched_arr[b] = t
        if b in halts and b != dest:
            t += _dwell(cor, b, klass)
        sched_dep[b] = t
    return Train(number, name, klass, up, origin, dest, dep, list(halts),
                 sched_arr, sched_dep, pad)


def build_roster(cor: Corridor = CORRIDOR, rng=None) -> list[Train]:
    import numpy as np
    rng = rng or np.random.default_rng(7)
    trains: list[Train] = []
    for num, name, klass, o, d, dep, hd in ROSTER_RAW:
        trains.append(build_timetable(cor, num, name, klass, o, d, _hhmm(dep), hd, rng))
        # mirrored down-direction service
        dnum = str(int(num) + 1) if num.isdigit() else num + "D"
        ddep = (_hhmm(dep) + 60 * int(rng.integers(3, 10))) % 1440
        trains.append(build_timetable(cor, dnum, name + " (Down)", klass,
                                      d, o, ddep, hd, rng))
    return trains

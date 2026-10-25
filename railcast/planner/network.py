"""
The national timetable as a searchable network.

Built once from the datameet working timetable and cached. Three things the raw
data gets wrong are repaired here:

  halts         datameet lists every station a train passes, with arrival equal
                to departure. Only a stop with dwell time (or the origin and the
                destination) is somewhere a passenger can board or get off.
  times         343 trains have booked times that run backwards, mostly a wrong
                `day` field. A drop of more than 12 hours is a missed midnight;
                anything smaller is clamped.
  codes         the snapshot predates several renamings (PRYJ was ALD, DDU was
                MGS, CSMT was CSTM ...). Current codes are mapped to the ones in
                the data. KSR in the data is Kasturi, not Bengaluru.

What the data cannot tell us: which days a train runs. Every train is treated as
daily, and every response says so.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import cached_property

from .. import config
from ..feeds.base import RAW, haversine_km
from ..feeds.datameet import TYPE_TO_CLASS, DatameetFeed

CACHE = RAW.parent / "cache" / "network.pkl"
CACHE_VERSION = 1

# current code -> code in the datameet snapshot
ALIASES = {
    "PRYJ": "ALD", "DDU": "MGS", "VGLJ": "JHS", "RKMP": "HBJ",
    "CSMT": "CSTM", "KSR": "SBC", "AYC": "FD",
}

# Stations in one city a passenger can change between by road or metro.
CITY_GROUPS = {
    "Delhi": ["NDLS", "DLI", "NZM", "HNZM", "ANVT", "DEE", "DSA", "DEC"],
    "Mumbai": ["CSTM", "BCT", "DR", "DDR", "LTT", "BDTS"],
    "Kolkata": ["HWH", "SDAH", "KOAA", "SHM", "SRC"],
    "Chennai": ["MAS", "MS", "PER"],
    "Bengaluru": ["SBC", "YPR", "BNC", "KJM"],
    "Hyderabad": ["SC", "HYB", "KCG"],
    "Patna": ["PNBE", "RJPB", "PPTA", "DNR", "PNC"],
    "Lucknow": ["LKO", "LJN"],
    "Ahmedabad": ["ADI", "SBI"],
    "Pune": ["PUNE", "SVJR"],
}

# trains.json flags -> class codes
CLASS_FLAGS = {"first_ac": "1A", "second_ac": "2A", "third_ac": "3A",
               "sleeper": "SL", "chair_car": "CC", "first_class": "FC"}


@dataclass(frozen=True)
class Call:
    """One stop. Times are minutes from midnight of the train's start date."""
    code: str
    arr: float
    dep: float
    km: float            # great-circle km from origin, summed stop to stop
    halt: bool


@dataclass(frozen=True)
class TrainInfo:
    number: str
    name: str
    type: str
    klass: str           # priority class the delay models understand
    classes: frozenset   # coach classes, empty when the data does not say
    reserved: bool


@dataclass
class Network:
    stations: dict                      # code -> StationRecord
    trains: dict                        # number -> TrainInfo
    calls: dict                         # number -> list[Call]
    halts_at: dict = field(default_factory=dict)   # code -> list[(number, idx)]

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, feed: DatameetFeed | None = None, rebuild: bool = False) -> "Network":
        feed = feed or DatameetFeed(download=False)
        src = feed._path("schedules")
        if CACHE.exists() and not rebuild:
            with open(CACHE, "rb") as fh:
                blob = pickle.load(fh)
            if (blob.get("version") == CACHE_VERSION
                    and blob.get("mtime") == src.stat().st_mtime):
                return blob["network"]
        net = cls.build(feed)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump({"version": CACHE_VERSION, "mtime": src.stat().st_mtime,
                         "network": net}, fh)
        return net

    @classmethod
    def build(cls, feed: DatameetFeed) -> "Network":
        stations = feed.stations()
        raw = {f["properties"]["number"]: f["properties"]
               for f in feed._load("trains")["features"]}
        trains, calls = {}, {}
        for num, stops in feed.all_schedules().items():
            rec = feed.trains().get(num)
            if rec is None or len(stops) < 2:
                continue
            seq = _repair(stations, stops)
            if len(seq) < 2:
                continue
            props = raw.get(num, {})
            trains[num] = TrainInfo(
                number=num, name=rec.name, type=rec.type,
                klass=TYPE_TO_CLASS.get(rec.type, "EXPRESS"),
                classes=frozenset(c for k, c in CLASS_FLAGS.items() if props.get(k)),
                reserved=rec.type not in config.UNRESERVED_TYPES)
            calls[num] = seq
        net = cls(stations=stations, trains=trains, calls=calls)
        for num, seq in calls.items():
            for i, c in enumerate(seq):
                if c.halt:
                    net.halts_at.setdefault(c.code, []).append((num, i))
        return net

    # ------------------------------------------------------------------ #
    def resolve(self, text: str) -> str:
        """A station code, current or old, or a unique name prefix."""
        q = text.strip().upper()
        code = ALIASES.get(q, q)
        if code in self.stations and code in self.halts_at:
            return code
        hits = sorted(c for c, s in self.stations.items()
                      if s.name.upper().startswith(q) and c in self.halts_at)
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise KeyError(f"no station matches {text!r}")
        raise KeyError(f"{text!r} is ambiguous: " + ", ".join(
            f"{c} ({self.stations[c].name})" for c in hits[:8]))

    @cached_property
    def city_of(self) -> dict:
        return {c: city for city, codes in CITY_GROUPS.items() for c in codes
                if c in self.halts_at}

    def city_mates(self, code: str) -> list[str]:
        """Other stations in the same city, nearest first."""
        city = self.city_of.get(code)
        if city is None:
            return []
        mates = [c for c in CITY_GROUPS[city] if c != code and c in self.halts_at]
        return sorted(mates, key=lambda c: self.km_between(code, c))

    def km_between(self, a: str, b: str) -> float:
        sa, sb = self.stations.get(a), self.stations.get(b)
        if sa is None or sb is None:
            return 0.0
        return haversine_km((sa.lat, sa.lon), (sb.lat, sb.lon))

    def transfer_min(self, a: str, b: str) -> float:
        """Minimum time to change from station a to station b."""
        if a == b:
            return config.MIN_LAYOVER_SAME_STATION
        lo, hi = config.CROSS_CITY_CLAMP
        t = config.CROSS_CITY_BASE_MIN + config.CROSS_CITY_MIN_PER_KM * self.km_between(a, b)
        return min(hi, max(lo, t))

    def name(self, code: str) -> str:
        s = self.stations.get(code)
        return s.name.title() if s else code


def _repair(stations, stops) -> list[Call]:
    """Monotone times, halts marked, km from origin."""
    out, prev, wrap, km, last_pos = [], None, 0.0, 0.0, None
    n = len(stops)
    for k, s in enumerate(stops):
        a = s.arrival if s.arrival is not None else s.departure
        d = s.departure if s.departure is not None else s.arrival
        if a is None or d is None:
            continue
        a, d = a + wrap, d + wrap
        if prev is not None and a < prev - 720:      # a missed midnight
            wrap += 1440.0
            a, d = a + 1440.0, d + 1440.0
        if prev is not None and a < prev:
            a = prev
        d = max(d, a)
        prev = d
        st = stations.get(s.station_code)
        if st is not None:
            pos = (st.lat, st.lon)
            if last_pos is not None:
                km += haversine_km(last_pos, pos)
            last_pos = pos
        halt = k == 0 or k == n - 1 or s.is_halt
        out.append(Call(s.station_code, a, d, km, halt))
    return out

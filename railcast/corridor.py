"""
Block-section graph for the prototype corridor: New Delhi -> Mumbai Central.

Stations, km posts, zones, sanctioned speeds and line type are approximations of
the real Western Railway / WCR / NCR / NR route. They are close enough in
structure (spacing, gradient bands, junction density) for the conflict simulator
to behave like the real corridor; they are not an operating document.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# (code, name, km from NDLS, is_junction, has_loop)
STATIONS_RAW = [
    ("NDLS", "New Delhi",            0,   True,  True),
    ("NZM",  "Hazrat Nizamuddin",    8,   True,  True),
    ("FDB",  "Faridabad",            27,  False, True),
    ("BVH",  "Ballabgarh",           34,  False, True),
    ("PWL",  "Palwal",               57,  False, True),
    ("KSV",  "Kosi Kalan",           91,  False, True),
    ("MTJ",  "Mathura Jn",           141, True,  True),
    ("BTE",  "Bharatpur Jn",         176, True,  True),
    ("BXN",  "Bayana Jn",            217, True,  True),
    ("GGC",  "Gangapur City",        262, False, True),
    ("SWM",  "Sawai Madhopur Jn",    310, True,  True),
    ("IDG",  "Indargarh",            375, False, True),
    ("KOTA", "Kota Jn",              465, True,  True),
    ("RMA",  "Ramganj Mandi Jn",     528, True,  True),
    ("BWM",  "Bhawani Mandi",        561, False, True),
    ("SGZ",  "Shamgarh",             590, False, True),
    ("VMA",  "Vikramgarh Alot",      617, False, True),
    ("NAD",  "Nagda Jn",             683, True,  True),
    ("RTM",  "Ratlam Jn",            726, True,  True),
    ("MGN",  "Meghnagar",            789, False, True),
    ("DHD",  "Dahod",                828, False, True),
    ("GDA",  "Godhra Jn",            902, True,  True),
    ("BRC",  "Vadodara Jn",          975, True,  True),
    ("BH",   "Bharuch Jn",           1046, True, True),
    ("ST",   "Surat",                1105, True,  True),
    ("NVS",  "Navsari",              1135, False, True),
    ("BL",   "Valsad",               1180, False, True),
    ("VAPI", "Vapi",                 1206, False, True),
    ("BOR",  "Boisar",               1252, False, True),
    ("PLG",  "Palghar",              1264, False, True),
    ("VR",   "Virar",                1302, True,  True),
    ("BSR",  "Vasai Road",           1314, True,  True),
    ("BVI",  "Borivali",             1340, True,  True),
    ("BDTS", "Bandra Terminus",      1361, True,  True),
    ("MMCT", "Mumbai Central",       1386, True,  True),
]


def zone_of(km: float) -> str:
    """Railway zone the km post falls in (approximate divisional boundaries)."""
    if km < 60:
        return "NR"
    if km < 230:
        return "NCR"
    if km < 700:
        return "WCR"
    return "WR"


def sanctioned_speed(km: float) -> float:
    """Sanctioned sectional speed, km/h."""
    if km < 30:            # Delhi throat, heavy suburban interference
        return 90.0
    if km < 141:
        return 130.0
    if km < 465:
        return 130.0
    if km < 726:           # Kota - Ratlam
        return 120.0
    if km < 902:           # Ratlam - Godhra ghat, curves and gradients
        return 100.0
    if km < 1105:
        return 130.0
    if km < 1302:
        return 120.0
    return 80.0            # Mumbai suburban section, shared with locals


def gradient_factor(km: float) -> float:
    """Multiplier on achievable speed from ruling gradient / curvature."""
    if 726 <= km < 902:
        return 0.88        # ghat section
    if 310 <= km < 465:
        return 0.95        # Sawai Madhopur - Kota
    return 1.0


@dataclass(frozen=True)
class Station:
    idx: int
    code: str
    name: str
    km: float
    is_junction: bool
    has_loop: bool

    @property
    def zone(self) -> str:
        return zone_of(self.km)


@dataclass(frozen=True)
class Block:
    """One block section between two block posts (finer than station spacing)."""
    idx: int
    km_start: float
    km_end: float
    from_station: int      # index of the station at/behind km_start
    to_station: int        # index of the next station ahead

    @property
    def length(self) -> float:
        return self.km_end - self.km_start

    @property
    def mid_km(self) -> float:
        return (self.km_start + self.km_end) / 2.0

    @property
    def speed(self) -> float:
        return sanctioned_speed(self.mid_km) * gradient_factor(self.mid_km)

    @property
    def zone(self) -> str:
        return zone_of(self.mid_km)


@dataclass
class Corridor:
    stations: list[Station]
    blocks: list[Block]
    blocks_between: dict[tuple[int, int], list[int]] = field(default_factory=dict)

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def length_km(self) -> float:
        return self.stations[-1].km

    def code(self, i: int) -> str:
        return self.stations[i].code

    def section_blocks(self, i: int, j: int) -> list[int]:
        """Block indices for the run from station i to station i+1."""
        return self.blocks_between[(i, j)]

    def free_run_minutes(self, i: int, j: int, train_max_speed: float,
                         weather: float = 1.0, tsr: dict[int, float] | None = None,
                         stopping: bool = True) -> float:
        """Unimpeded running time station i -> j, minutes. The 'physics' term."""
        tsr = tsr or {}
        total = 0.0
        for b in self.section_blocks(i, j):
            blk = self.blocks[b]
            v = min(blk.speed, train_max_speed) * weather
            v = min(v, tsr.get(b, 1e9))
            total += blk.length / max(v, 15.0) * 60.0
        # acceleration / braking penalty at each end of the section
        total += 1.4 if stopping else 0.6
        return total


    def free_run_span(self, i: int, j: int, train_max_speed: float,
                      weather: float = 1.0, tsr: dict | None = None) -> float:
        """Unimpeded running time across every section between stations i and j."""
        lo, hi = (i, j) if i < j else (j, i)
        return sum(self.free_run_minutes(a, a + 1, train_max_speed, weather, tsr)
                   for a in range(lo, hi))


def build_corridor(block_len_km: float = 8.0) -> Corridor:
    stations = [
        Station(i, c, n, float(km), j, l)
        for i, (c, n, km, j, l) in enumerate(STATIONS_RAW)
    ]
    blocks: list[Block] = []
    between: dict[tuple[int, int], list[int]] = {}
    for i in range(len(stations) - 1):
        a, b = stations[i], stations[i + 1]
        span = b.km - a.km
        n = max(1, int(round(span / block_len_km)))
        ids = []
        for k in range(n):
            ks = a.km + span * k / n
            ke = a.km + span * (k + 1) / n
            blocks.append(Block(len(blocks), ks, ke, i, i + 1))
            ids.append(len(blocks) - 1)
        between[(i, i + 1)] = ids
    return Corridor(stations, blocks, between)


CORRIDOR = build_corridor()

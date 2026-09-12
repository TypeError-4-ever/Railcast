"""
Feed adapters.

The deck promises a swappable FeedAdapter so that the model does not care where
position, schedule and weather come from. This is that seam.

    StaticFeed    stations, trains, timetables, route geometry
    WeatherFeed   observed and forecast weather for a point and a day
    LiveFeed      current position and delay for a running train

Three implementations ship:

    datameet.DatameetFeed      real Indian Railways timetable and station data
    openmeteo.OpenMeteoFeed    real historical and forecast weather
    synthetic.SyntheticFeed    the simulator, for running with no network

Nothing above the adapter layer knows which one it is talking to.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

CACHE = Path("data/cache")
RAW = Path("data/raw")
USER_AGENT = "railcast-prototype/0.1 (SIH26028; research use)"


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StationRecord:
    code: str
    name: str
    lat: float
    lon: float
    zone: str
    state: str = ""

    @property
    def is_junction(self) -> bool:
        n = self.name.upper()
        return n.endswith(" JN") or " JN " in n or "JUNCTION" in n


@dataclass(frozen=True)
class TrainRecord:
    number: str
    name: str
    type: str                 # Raj, Drnt, Shtb, SF, Exp, Mail, Pass, MEMU, GR ...
    zone: str
    from_code: str
    to_code: str
    distance_km: float


@dataclass(frozen=True)
class StopRecord:
    """One scheduled call. Times are minutes from the train's day-1 midnight."""
    station_code: str
    arrival: float | None
    departure: float | None
    day: int

    @property
    def dwell(self) -> float:
        if self.arrival is None or self.departure is None:
            return 0.0
        return max(0.0, self.departure - self.arrival)

    @property
    def is_halt(self) -> bool:
        return self.dwell > 0.0


@dataclass(frozen=True)
class LiveRecord:
    """One observed station event for a running train.

    Times are minutes from midnight of the train's own start date, so a run that
    crosses midnight stays monotonic.
    """
    train_number: str
    station_code: str
    scheduled: float
    actual: float | None
    delay_min: float | None
    event: str                # "arrival" | "departure"
    observed_at: float        # unix seconds
    source: str
    start_date: str = ""      # the train's start date, not the observation date
    sequence: int = 0         # position along the route
    distance_km: float = 0.0  # cumulative km from origin, as the operator has it
    is_halt: bool = False
    speed_to_next_kmph: float = 0.0
    status: str = ""          # departed | at-station | upcoming

    @property
    def observed(self) -> bool:
        """True only if the train has actually been reported here.

        An 'upcoming' station can still carry an actual time - that is the
        operator's own projection, not an observation. Fitting on those would
        be training on another system's forecast and calling it ground truth.
        """
        return self.status in ("departed", "at-station") and self.actual is not None

    @property
    def is_incumbent_eta(self) -> bool:
        """The ETA the deployed system is showing right now - a real baseline."""
        return self.status == "upcoming" and self.actual is not None


# --------------------------------------------------------------------------- #
# interfaces
# --------------------------------------------------------------------------- #
class StaticFeed(ABC):
    """Stations, trains, timetables and route geometry."""

    name = "static"

    @abstractmethod
    def stations(self) -> dict[str, StationRecord]: ...

    @abstractmethod
    def trains(self) -> dict[str, TrainRecord]: ...

    @abstractmethod
    def schedule(self, number: str) -> list[StopRecord]: ...

    def route_geometry(self, number: str) -> list[tuple[float, float]]:
        """(lon, lat) along the train's path. Empty if the feed has none."""
        return []


class WeatherFeed(ABC):
    name = "weather"

    @abstractmethod
    def daily(self, lat: float, lon: float, start: str, end: str) -> dict:
        """Date -> {'rain_mm', 'fog_risk', 'tmin_c'} for [start, end] (ISO dates)."""


class LiveFeed(ABC):
    """Current running position.

    No public historical delay dataset for Indian Railways exists - NTES serves
    current and near-future state only. A live feed is therefore also a
    *collector*: poll it on a schedule and the history accumulates locally.
    """

    name = "live"

    def preflight(self) -> None:
        """Raise now if this feed cannot possibly work. Called before polling, so
        a scheduled job fails loudly instead of quietly collecting nothing."""

    @abstractmethod
    def running_status(self, number: str, start_date: str) -> list[LiveRecord]: ...


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between (lat, lon) pairs."""
    r = 6371.0088
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def hhmmss_to_min(v) -> float | None:
    """'16:40:00' -> 1000.0. Handles the literal string 'None' in the source."""
    if v is None or v == "None" or v == "":
        return None
    parts = str(v).split(":")
    try:
        h, m = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None
    s = int(parts[2]) if len(parts) > 2 else 0
    return h * 60 + m + s / 60.0


def fetch(url: str, dest: Path, *, force: bool = False, timeout: int = 300,
          retries: int = 3) -> Path:
    """Download to `dest`, reusing what is already there unless forced."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    while chunk := r.read(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def get_json(url: str, *, ttl_hours: float = 24.0, timeout: int = 60):
    """GET a JSON API with an on-disk cache keyed by URL."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = CACHE / f"{key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600:
        return json.loads(path.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload

"""
Seat availability, and what each status is worth.

    RailKitSeats   railkit.in REST API. Needs RAILKIT_API_KEY, RAILKIT_BASE_URL
                   and, if it is not x-api-key, RAILKIT_AUTH_HEADER. RailKit's
                   docs list the routes but not the base URL or header name -
                   take both from your dashboard.
    NoSeats        the default with no key. Seats are reported as not checked,
                   never guessed.

Every RailKit response is appended to data/availability/, the history a future
model of "will this leg have seats" will be trained on. Station codes are sent
as today's codes (PRYJ, not the ALD in the timetable snapshot).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from .. import config
from ..feeds.base import CACHE, USER_AGENT
from .network import current_code

LOG = Path("data/availability")
TTL_S = 300


@dataclass
class SeatStatus:
    status: str              # CNF | RAC | WL | REGRET | UNKNOWN
    count: int | None        # seats left, or waitlist position
    text: str                # exactly what the provider said
    fare: float | None
    quota: str
    source: str
    checked_at: float


_PATTERNS = [
    (re.compile(r"REGRET|NOT\s*AVAILABLE|TRAIN\s*CANCEL", re.I), "REGRET"),
    (re.compile(r"(?:CURR_)?(?:AVAILABLE|AVBL|AVL)[-\s]*0*(\d+)", re.I), "CNF"),
    (re.compile(r"\bAVAILABLE\b", re.I), "CNF"),
    (re.compile(r"\bRAC\s*0*(\d+)", re.I), "RAC"),
    (re.compile(r"(?:GN|PQ|RL|TQ|RS|CK)?WL\s*0*(\d+)(?!.*WL)", re.I), "WL"),
]


def parse_availability(text: str) -> tuple[str, int | None]:
    """'AVAILABLE-0023' -> ('CNF', 23); 'GNWL45/WL30' -> ('WL', 30)."""
    t = (text or "").strip()
    for rx, status in _PATTERNS:
        m = rx.search(t)
        if m:
            n = int(m.group(1)) if m.groups() and m.group(1) else None
            return status, n
    return "UNKNOWN", None


def p_seat(s: SeatStatus | None, hours_to_chart: float) -> float | None:
    """Chance of travelling on this leg. Judgement, see config.P_SEAT."""
    if s is None or s.status == "UNKNOWN":
        return None
    if s.status == "WL" and (s.count or 99) <= 5 and hours_to_chart > 24:
        return config.P_SEAT_WL_SHORT
    return config.P_SEAT[s.status]


# --------------------------------------------------------------------------- #
class NoSeats:
    name = "none"
    available = False

    def check(self, train, frm, to, day: date, klass: str, quota: str = "GN"):
        return None


class RailKitSeats:
    name = "railkit"
    available = True

    def __init__(self):
        self.key = os.environ.get("RAILKIT_API_KEY", "")
        self.base = os.environ.get("RAILKIT_BASE_URL", "").rstrip("/")
        self.header = os.environ.get("RAILKIT_AUTH_HEADER", "x-api-key")
        if not self.key or not self.base:
            raise RuntimeError(
                "RailKit needs RAILKIT_API_KEY and RAILKIT_BASE_URL in the "
                "environment (base URL and header name are on your railkit.in "
                "dashboard). Without them seats are reported as not checked.")

    def _get(self, path: str) -> dict:
        url = self.base + path
        cache = CACHE / "railkit" / (hashlib.sha256(url.encode()).hexdigest()[:20] + ".json")
        if cache.exists() and time.time() - cache.stat().st_mtime < TTL_S:
            return json.loads(cache.read_text(encoding="utf-8"))
        req = urllib.request.Request(url, headers={self.header: self.key,
                                                   "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            payload = json.loads(r.read().decode("utf-8"))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload), encoding="utf-8")
        LOG.mkdir(parents=True, exist_ok=True)
        with open(LOG / f"{date.today().isoformat()}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"url": path, "at": time.time(), "response": payload}) + "\n")
        return payload

    def check(self, train, frm, to, day: date, klass: str, quota: str = "GN"):
        d = day.strftime("%d-%m-%Y")
        frm, to = current_code(frm), current_code(to)
        try:
            p = self._get(f"/api/v1/seats/{train}/{frm}/{to}/{d}/{klass}/{quota}")
        except Exception as exc:                          # one bad leg is not fatal
            return SeatStatus("UNKNOWN", None, f"error: {exc}", None, quota,
                              self.name, time.time())
        data = p.get("data") or {}
        text = ""
        for row in data.get("availability") or []:
            if row.get("date") in (d, day.isoformat()) or not text:
                text = str(row.get("availabilityText") or "")
        status, n = parse_availability(text)
        fare = data.get("fare")
        if isinstance(fare, dict):
            fare = fare.get("totalFare") or fare.get("total")
        try:
            fare = float(fare) if fare is not None else None
        except (TypeError, ValueError):
            fare = None
        return SeatStatus(status, n, text, fare, quota, self.name, time.time())


def provider():
    """RailKit when it is configured, else NoSeats."""
    if os.environ.get("RAILKIT_API_KEY"):
        return RailKitSeats()
    return NoSeats()


def as_dict(s: SeatStatus | None) -> dict | None:
    return asdict(s) if s else None

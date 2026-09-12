"""
Live position: the adapter, a collector, and a replay feed.

There is no public historical delay dataset for Indian Railways. NTES serves
current and near-future running state only; nobody publishes an archive of past
arrivals. So the history this project needs has to be *accumulated*: poll a live
source on a schedule, append every station report to a local store, and replay
that store to fit the model.

    RailRadarFeed   real REST adapter. Needs RAILRADAR_API_KEY.
    NTESFeed        placeholder for the IR-provided feed the deck assumes.
    LiveStore       append-only JSONL of every position report seen.
    ReplayFeed      reads the store back, so collected history is a feed too.

Nothing here scrapes a site that does not offer an API. If you have an
agreement with CRIS for the NTES feed, implement NTESFeed.running_status and
every stage above this file keeps working unchanged.
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, datetime
from pathlib import Path

from .base import LiveFeed, LiveRecord, USER_AGENT, hhmmss_to_min

STORE = Path("data/live")


# --------------------------------------------------------------------------- #
# the local history that gets accumulated
# --------------------------------------------------------------------------- #
class LiveStore:
    """Append-only JSONL, one file per calendar day of observation."""

    def __init__(self, root: Path = STORE):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, day: str | None = None) -> Path:
        return self.root / f"{day or date.today().isoformat()}.jsonl"

    def append(self, records) -> int:
        n = 0
        with open(self.path(), "a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.__dict__) + "\n")
                n += 1
        return n

    def read(self, dedupe: bool = True) -> list[LiveRecord]:
        """Every record on disk, one per station event by default.

        The file stays append-only so the raw history is auditable, but polling
        the same train twice re-reports every station it has already passed.
        Left raw, a nightly cron would count an early station once per poll and
        weight it dozens of times in any fit. Keyed on the train's own run, the
        later observation wins - actual times firm up as a train progresses.
        """
        out = []
        for path in sorted(self.root.glob("*.jsonl")):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(LiveRecord(**json.loads(line)))
        if not dedupe:
            return out
        latest: dict[tuple, LiveRecord] = {}
        for r in out:
            key = (r.train_number, r.start_date, r.station_code, r.event)
            prev = latest.get(key)
            if prev is None or r.observed_at >= prev.observed_at:
                latest[key] = r
        return sorted(latest.values(),
                      key=lambda r: (r.train_number, r.start_date, r.sequence))

    def frame(self):
        import pandas as pd
        rows = [r.__dict__ for r in self.read()]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        rows = self.read()
        raw = len(self.read(dedupe=False))
        if not rows:
            return {"records": 0, "trains": 0, "days": 0,
                    "note": "nothing collected yet - run `python -m railcast.feeds collect`"}
        obs = [r for r in rows if r.observed]
        eta = [r for r in rows if r.is_incumbent_eta]
        delays = sorted(r.delay_min for r in obs if r.delay_min is not None)
        return {
            "records": len(rows),
            "rows_on_disk": raw,
            "duplicate_reports_collapsed": raw - len(rows),
            "observed_arrivals": len(obs),
            "incumbent_eta_rows": len(eta),
            "trains": len({r.train_number for r in rows}),
            "stations": len({r.station_code for r in rows}),
            "run_dates": len({r.start_date for r in rows if r.start_date}),
            "median_observed_delay_min": (delays[len(delays) // 2] if delays else None),
            "note": ("observed rows are ground truth; incumbent_eta rows are the "
                     "deployed system's own forecast, kept as a baseline"),
        }


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
class RailRadarFeed(LiveFeed):
    """https://api.railradar.in/v1 - commercial API, free tier available.

    Set RAILRADAR_API_KEY in the environment. Without it this raises rather
    than falling back to anything, so a run can never quietly become synthetic.
    """

    name = "railradar"
    base = "https://api.railradar.in/v1"

    def __init__(self, api_key: str | None = None, timeout: int = 30):
        self.api_key = api_key or os.environ.get("RAILRADAR_API_KEY", "")
        self.timeout = timeout

    def preflight(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "RAILRADAR_API_KEY is not set. Get a key at railradar.in, then:\n"
                "  PowerShell   $env:RAILRADAR_API_KEY = 'rr_live_...'\n"
                "  bash         export RAILRADAR_API_KEY=rr_live_...\n"
                "Use setx on Windows to make it stick across sessions.")

    def _get(self, path: str):
        import urllib.request
        self.preflight()
        req = urllib.request.Request(
            f"{self.base}{path}",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "User-Agent": USER_AGENT, "Accept": "application/json"})
        import urllib.request as u
        with u.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def running_status(self, number: str, start_date: str = "") -> list[LiveRecord]:
        """Every station event on today's run that has actually been reported.

        The live endpoint returns the whole route, with actual times filled in
        for stations already passed. One call therefore captures a journey's
        entire history so far, which is why polling once a day is enough.
        """
        payload = self._get(f"/trains/{number}/live")
        data = payload.get("data") or {}
        route = data.get("route") or []
        started = data.get("startDate") or start_date or ""
        base = _midnight(started)
        now = time.time()
        out = []
        for stop in route:
            code = stop.get("stationCode")
            if not code:
                continue
            common = dict(
                train_number=str(data.get("trainNumber") or number),
                station_code=str(code), source=self.name, observed_at=now,
                start_date=started, sequence=int(stop.get("sequence") or 0),
                distance_km=float(stop.get("distance") or 0.0),
                is_halt=bool(stop.get("isHalt")),
                speed_to_next_kmph=float(stop.get("speedToNextStationKmph") or 0.0),
                status=str(stop.get("status") or ""))
            for event, sk, ak, dk in (
                    ("arrival", "scheduledArrival", "actualArrival", "delayArrival"),
                    ("departure", "scheduledDeparture", "actualDeparture",
                     "delayDeparture")):
                sched = _iso_minutes(stop.get(sk), base)
                actual = _iso_minutes(stop.get(ak), base)
                if sched is None and actual is None:
                    continue
                delay = stop.get(dk)
                out.append(LiveRecord(
                    scheduled=sched if sched is not None else 0.0,
                    actual=actual,
                    delay_min=None if delay is None else float(delay),
                    event=event, **common))
        return out


class NTESFeed(LiveFeed):
    """The feed the deck assumes in production: NTES / FOIS, via CRIS.

    Not implemented. NTES publishes no open API, and this project will not
    scrape it. The interface is here so that the day an IR-provided feed exists,
    one method is all that changes.
    """

    name = "ntes"

    def running_status(self, number: str, start_date: str = "") -> list[LiveRecord]:
        raise NotImplementedError(
            "NTES has no public API. Options:\n"
            "  1. Obtain the CRIS/NTES feed through the Ministry and implement\n"
            "     this method - nothing else in the pipeline changes.\n"
            "  2. Use RailRadarFeed with an API key.\n"
            "  3. Run on SyntheticFeed, which needs no network.")


class ReplayFeed(LiveFeed):
    """Serves history already collected into the local store."""

    name = "replay"

    def __init__(self, store: LiveStore | None = None):
        self.store = store or LiveStore()
        self._by_train: dict[str, list] = {}
        for r in self.store.read():
            self._by_train.setdefault(r.train_number, []).append(r)

    def running_status(self, number: str, start_date: str = "") -> list[LiveRecord]:
        return self._by_train.get(str(number), [])

    def trains_seen(self) -> list[str]:
        return sorted(self._by_train)


# --------------------------------------------------------------------------- #
def collect(feed: LiveFeed, numbers, store: LiveStore | None = None,
            pause: float = 1.0, verbose: bool = True) -> int:
    """One polling pass. Run this on a cron; the history builds itself."""
    store = store or LiveStore()
    feed.preflight()                      # fail before touching the network
    total = 0
    for num in numbers:
        try:
            records = feed.running_status(str(num))
        except NotImplementedError:
            raise
        except Exception as exc:
            if verbose:
                print(f"  {num}: {type(exc).__name__}: {exc}")
            continue
        total += store.append(records)
        if verbose:
            print(f"  {num}: {len(records)} station reports")
        time.sleep(pause)
    return total


def _midnight(start_date: str) -> datetime | None:
    """Midnight of the train's start date, in the feed's own timezone."""
    if not start_date:
        return None
    try:
        return datetime.fromisoformat(start_date)
    except ValueError:
        return None


def _iso_minutes(stamp, base) -> float | None:
    """ISO-8601 timestamp -> minutes from the start date's midnight."""
    if not stamp or base is None:
        return None
    try:
        t = datetime.fromisoformat(str(stamp))
    except ValueError:
        return None
    if t.tzinfo is not None and base.tzinfo is None:
        base = base.replace(tzinfo=t.tzinfo)
    return (t - base).total_seconds() / 60.0


def _iter_stops(data):
    for key in ("route", "stations", "schedule", "stops", "previousStations"):
        v = data.get(key)
        if isinstance(v, list) and v:
            return v
    return []


def _first_minute(stop: dict, keys):
    for k in keys:
        if k in stop and stop[k] not in (None, "", "None"):
            v = hhmmss_to_min(stop[k])
            if v is not None:
                return v
    return None

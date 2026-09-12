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

    def read(self) -> list[LiveRecord]:
        out = []
        for p in sorted(self.root.glob("*.jsonl")):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        out.append(LiveRecord(**json.loads(line)))
        return out

    def frame(self):
        import pandas as pd
        rows = [r.__dict__ for r in self.read()]
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        rows = self.read()
        if not rows:
            return {"records": 0, "trains": 0, "days": 0,
                    "note": "nothing collected yet - run `python -m railcast.feeds collect`"}
        days = {datetime.utcfromtimestamp(r.observed_at).date().isoformat() for r in rows}
        delays = [r.delay_min for r in rows if r.delay_min is not None]
        return {
            "records": len(rows),
            "trains": len({r.train_number for r in rows}),
            "stations": len({r.station_code for r in rows}),
            "days": len(days),
            "median_delay_min": (sorted(delays)[len(delays) // 2] if delays else None),
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
        payload = self._get(f"/trains/{number}/live")
        data = payload.get("data") or {}
        now = time.time()
        out = []
        for stop in _iter_stops(data):
            sched = _first_minute(stop, ("scheduledArrival", "sta", "scheduled_arrival"))
            actual = _first_minute(stop, ("actualArrival", "ata", "actual_arrival"))
            delay = stop.get("delayArrival", stop.get("delay"))
            code = stop.get("stationCode") or stop.get("station_code") or stop.get("code")
            if not code:
                continue
            out.append(LiveRecord(
                train_number=str(number), station_code=str(code),
                scheduled=sched if sched is not None else 0.0, actual=actual,
                delay_min=None if delay is None else float(delay),
                event="arrival", observed_at=now, source=self.name))
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

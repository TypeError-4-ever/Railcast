"""
Real Indian Railways static data.

Source: github.com/datameet/railways, published CC0 by the DataMeet community
from Indian Railways public timetable and station listings.

    stations.json    8,990 stations - code, name, zone, state, coordinates
    trains.json      5,208 trains - number, name, type, zone, distance, route
    schedules.json   417,080 scheduled calls - arrival, departure, day

This is the published working timetable, not a live feed. It is enough to build
the block-section graph, the roster and the padding structure from real numbers
instead of estimates. Live position is a separate adapter - see live.py.
"""
from __future__ import annotations

import json
from functools import cached_property

from .base import (RAW, StaticFeed, StationRecord, StopRecord, TrainRecord,
                   fetch, hhmmss_to_min)

BASE = "https://raw.githubusercontent.com/datameet/railways/master"
FILES = {
    "stations": ("stations.json", 1.8),
    "trains": ("trains.json", 14.8),
    "schedules": ("schedules.json", 79.0),
}

# Indian Railways train type -> the priority class the simulator understands.
TYPE_TO_CLASS = {
    "Raj": "RAJDHANI", "Drnt": "DURONTO", "Shtb": "SHATABDI", "JShtb": "SHATABDI",
    "GR": "SUPERFAST", "SKr": "SUPERFAST", "SF": "SUPERFAST", "Suvidha": "SUPERFAST",
    "Mail": "EXPRESS", "Exp": "EXPRESS", "Hyd": "EXPRESS", "AC": "SUPERFAST",
    "Pass": "PASSENGER", "MEMU": "PASSENGER", "DEMU": "PASSENGER",
    "Toy": "PASSENGER", "Metro": "PASSENGER",
}


class DatameetFeed(StaticFeed):
    name = "datameet"
    licence = "CC0 1.0 (datameet/railways)"

    def __init__(self, root=RAW, download: bool = True):
        self.root = root
        self.download = download

    # -- fetching ----------------------------------------------------------- #
    def _path(self, key: str):
        fname, mb = FILES[key]
        dest = self.root / fname
        if not dest.exists():
            if not self.download:
                raise FileNotFoundError(
                    f"{dest} missing. Run `python -m railcast.feeds fetch` "
                    f"to download it ({mb:.0f} MB).")
            print(f"  downloading {fname} ({mb:.0f} MB) ...", flush=True)
            fetch(f"{BASE}/{fname}", dest)
        return dest

    def _load(self, key: str):
        with open(self._path(key), encoding="utf-8") as fh:
            return json.load(fh)

    def ensure(self) -> None:
        for key in FILES:
            self._path(key)

    # -- records ------------------------------------------------------------ #
    @cached_property
    def _stations(self) -> dict:
        out = {}
        for f in self._load("stations")["features"]:
            p = f["properties"]
            geom = f.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            if coords[0] is None or not p.get("code"):
                continue
            out[p["code"]] = StationRecord(
                code=p["code"], name=p.get("name") or p["code"],
                lat=float(coords[1]), lon=float(coords[0]),
                zone=p.get("zone") or "", state=p.get("state") or "")
        return out

    def stations(self) -> dict:
        return self._stations

    @cached_property
    def _trains(self) -> dict:
        out = {}
        for f in self._load("trains")["features"]:
            p = f["properties"]
            out[p["number"]] = TrainRecord(
                number=p["number"], name=p.get("name") or p["number"],
                type=p.get("type") or "Exp", zone=p.get("zone") or "",
                from_code=p.get("from_station_code") or "",
                to_code=p.get("to_station_code") or "",
                distance_km=float(p.get("distance") or 0.0))
        return out

    def trains(self) -> dict:
        return self._trains

    @cached_property
    def _geometry(self) -> dict:
        out = {}
        for f in self._load("trains")["features"]:
            g = f.get("geometry") or {}
            if g.get("type") == "LineString":
                out[f["properties"]["number"]] = [tuple(c) for c in g["coordinates"]]
        return out

    def route_geometry(self, number: str) -> list:
        return self._geometry.get(number, [])

    @cached_property
    def _schedules(self) -> dict:
        by_train: dict[str, list] = {}
        for r in self._load("schedules"):
            by_train.setdefault(r["train_number"], []).append(r)
        out = {}
        for num, rows in by_train.items():
            stops = []
            for r in rows:
                day = int(r.get("day") or 1)
                off = (day - 1) * 1440.0
                arr = hhmmss_to_min(r.get("arrival"))
                dep = hhmmss_to_min(r.get("departure"))
                stops.append(StopRecord(
                    station_code=r["station_code"],
                    arrival=None if arr is None else arr + off,
                    departure=None if dep is None else dep + off,
                    day=day))
            out[num] = stops
        return out

    def schedule(self, number: str) -> list:
        return self._schedules.get(number, [])

    def all_schedules(self) -> dict:
        return self._schedules

    def summary(self) -> dict:
        return {"source": "datameet/railways", "licence": self.licence,
                "stations": len(self.stations()), "trains": len(self.trains()),
                "scheduled_calls": sum(len(v) for v in self._schedules.values())}

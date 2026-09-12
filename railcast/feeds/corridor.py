"""
Build the block-section graph and the roster from real timetable data.

The spine of a corridor is the station sequence of a reference train that runs
it end to end. Every other train whose calls move monotonically along that spine
is on the corridor, and its path is the sub-sequence it touches.

Three things come out of the working timetable rather than out of an estimate:

  km posts          cumulative great-circle distance between real station
                    coordinates, scaled so the total matches the route distance
                    Indian Railways publishes for the reference train.
  sectional speed   the speed implied by the fastest booked run over each
                    section. This is not the sanctioned speed - it is the speed
                    the timetable actually allows, which is what a forecast
                    needs.
  recovery padding  booked run time minus free-run time, per section, per train.
                    Previously assumed as a percentage; now measured.
"""
from __future__ import annotations

import numpy as np

from ..corridor import Block, Corridor, Station, zone_id
from ..trains import CLASSES, Train
from .base import StaticFeed, haversine_km
from .datameet import TYPE_TO_CLASS

MAX_BLOCK_KM = 12.0
MIN_SECTION_MIN = 0.6


# --------------------------------------------------------------------------- #
def spine_codes(feed: StaticFeed, reference: str, reverse: bool = False):
    stops = feed.schedule(reference)
    codes = [s.station_code for s in stops]
    if reverse:
        codes = codes[::-1]
    known = feed.stations()
    seen, out = set(), []
    for c in codes:                      # a route may touch a station twice
        if c in known and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def corridor_trains(feed: StaticFeed, spine, min_stops: int = 8,
                    min_monotone: float = 0.9):
    """Trains whose calls run monotonically along the spine."""
    idx = {c: i for i, c in enumerate(spine)}
    out = {}
    for num, stops in feed.all_schedules().items():
        on = [(idx[s.station_code], s) for s in stops if s.station_code in idx]
        if len(on) < min_stops:
            continue
        pos = [p for p, _ in on]
        up = sum(1 for a, b in zip(pos, pos[1:]) if b > a)
        dn = sum(1 for a, b in zip(pos, pos[1:]) if b < a)
        if max(up, dn) / max(1, len(pos) - 1) < min_monotone:
            continue
        out[num] = (on, up >= dn)      # kept in travel order, not spine order
    return out


# --------------------------------------------------------------------------- #
def _km_posts(feed: StaticFeed, spine, total_km: float):
    st = feed.stations()
    km, run = [0.0], 0.0
    for a, b in zip(spine[:-1], spine[1:]):
        sa, sb = st[a], st[b]
        run += haversine_km((sa.lat, sa.lon), (sb.lat, sb.lon))
        km.append(run)
    if total_km > 0 and run > 0:
        scale = total_km / run          # great-circle is shorter than the rails
        km = [k * scale for k in km]
    return km


def _implied_speeds(feed: StaticFeed, spine, km, runners, quantile: float = 0.97):
    """Sectional speed the timetable allows, from every train that runs it.

    Near the maximum, not the average: the section limit has to let the fastest
    booked train keep its own booked time. Slower trains are held back by their
    own maximum permissible speed, not by this. A high quantile rather than the
    outright maximum keeps one mis-keyed timetable entry from setting the limit.
    """
    n = len(spine)
    samples: list[list[float]] = [[] for _ in range(n - 1)]
    for num, (on, _up) in runners.items():
        for (i, sa), (j, sb) in zip(on[:-1], on[1:]):
            if abs(j - i) != 1:
                continue
            lo = min(i, j)
            t0 = sa.departure if sa.departure is not None else sa.arrival
            t1 = sb.arrival if sb.arrival is not None else sb.departure
            if t0 is None or t1 is None:
                continue
            dt = t1 - t0
            if dt <= 0:
                dt += 1440.0            # crossed midnight without a day bump
            dist = abs(km[j] - km[i])
            if dt < MIN_SECTION_MIN or dist <= 0.05:
                continue
            samples[lo].append(dist / (dt / 60.0))
    speeds = []
    for i, vals in enumerate(samples):
        if vals:
            v = float(np.quantile(vals, quantile))
        else:
            v = float(np.median([s for s in speeds] or [80.0]))
        speeds.append(float(np.clip(v, 30.0, 160.0)))
    # a single mis-keyed timetable entry should not create a 160 km/h section
    # between two suburban halts: smooth over three sections.
    sm = np.convolve(np.array(speeds), np.ones(3) / 3.0, mode="same")
    sm[0], sm[-1] = speeds[0], speeds[-1]
    return [float(x) for x in sm]


def _segments(feed: StaticFeed, spine, reference: str):
    """Label each station with the corridor segment it sits in.

    The station master leaves the zone field blank for about half of all
    stations, so a zone label would be half invented. Segments are derived
    instead from the reference train's own commercial halts - real, complete,
    and close to the divisional boundaries a controller works to.
    """
    halts = [s.station_code for s in feed.schedule(reference) if s.is_halt]
    idx = {c: i for i, c in enumerate(spine)}
    anchors = sorted({idx[c] for c in halts if c in idx} | {0, len(spine) - 1})
    labels, a = [""] * len(spine), 0
    for lo, hi in zip(anchors[:-1], anchors[1:]):
        tag = f"{spine[lo]}-{spine[hi]}"
        for i in range(lo, hi + 1):
            if not labels[i]:
                labels[i] = tag
        a = hi
    for i, v in enumerate(labels):
        if not v:
            labels[i] = f"{spine[anchors[-2]]}-{spine[anchors[-1]]}"
    return labels


def build_corridor(feed: StaticFeed, reference: str = "12951",
                   reverse: bool = True) -> tuple[Corridor, dict]:
    """Corridor graph for the route the reference train runs."""
    spine = spine_codes(feed, reference, reverse=reverse)
    st = feed.stations()
    total = feed.trains()[reference].distance_km
    km = _km_posts(feed, spine, total)
    runners = corridor_trains(feed, spine)
    speeds = _implied_speeds(feed, spine, km, runners)
    segs = _segments(feed, spine, reference)

    stations = []
    for i, code in enumerate(spine):
        r = st[code]
        stations.append(Station(i, code, r.name, km[i], r.is_junction, True,
                                zone_code=segs[i]))
        zone_id(segs[i])
    blocks, between = [], {}
    for i in range(len(stations) - 1):
        span = km[i + 1] - km[i]
        n = max(1, int(np.ceil(span / MAX_BLOCK_KM))) if span > 0 else 1
        ids = []
        for k in range(n):
            ks = km[i] + span * k / n
            ke = km[i] + span * (k + 1) / n
            blocks.append(Block(len(blocks), ks, ke, i, i + 1,
                                speed_kmh=speeds[i],
                                zone_code=stations[i].zone_code))
            ids.append(len(blocks) - 1)
        between[(i, i + 1)] = ids
    # sectional speeds came from booked run times, which already include
    # station acceleration and braking
    cor = Corridor(stations, blocks, between, accel_stop=0.0, accel_pass=0.0)
    meta = {"reference_train": reference, "spine_stations": len(spine),
            "segments": sorted(set(segs), key=segs.index),
            "route_km": round(km[-1], 1), "published_km": total,
            "blocks": len(blocks), "trains_on_corridor": len(runners),
            "median_section_speed_kmh": round(float(np.median(speeds)), 1)}
    return cor, meta


# --------------------------------------------------------------------------- #
def build_roster(feed: StaticFeed, cor: Corridor, min_span: float = 0.5,
                 limit: int | None = None) -> list[Train]:
    """Real trains, real halts, real booked times, measured padding."""
    spine = [s.code for s in cor.stations]
    runners = corridor_trains(feed, spine)
    recs = feed.trains()
    idx = {c: i for i, c in enumerate(spine)}
    out = []
    for num, (on, up) in sorted(runners.items()):
        rec = recs.get(num)
        if rec is None:
            continue
        pos = [p for p, _ in on]
        if (max(pos) - min(pos)) / len(spine) < min_span:
            continue
        train = _to_train(num, rec, on, up, cor, idx)
        if train is not None:
            out.append(train)
    out.sort(key=lambda t: (-_span(t, cor), t.number))
    return out[:limit] if limit else out


def _span(t: Train, cor: Corridor) -> float:
    return abs(cor.stations[t.dest].km - cor.stations[t.origin].km)


def _to_train(num, rec, on, up, cor, idx) -> Train | None:
    """Map real booked times onto the spine, interpolating the stations the
    train passes without a booked call."""
    klass = TYPE_TO_CLASS.get(rec.type, "EXPRESS")
    spec = CLASSES[klass]
    called = [p for p, _ in on]
    stops = [s for _, s in on]
    if len(called) < 3:
        return None

    # absolute minutes, repaired: booked times may not run backwards
    times, prev = [], None
    for s in stops:
        a = s.arrival if s.arrival is not None else s.departure
        d = s.departure if s.departure is not None else s.arrival
        if a is None or d is None:
            return None
        if prev is not None:
            while a < prev - 1e-9:
                a += 1440.0
            while d < a - 1e-9:
                d += 1440.0
        prev = d
        times.append([a, d])

    origin, dest = called[0], called[-1]
    step = 1 if up else -1
    path = list(range(origin, dest + step, step))
    dep0 = times[0][1]
    base = dep0 % 1440.0

    # booked arrival/departure for every station on the path
    booked: dict[int, tuple[float, float]] = {}
    for p, (a, d) in zip(called, times):
        booked[p] = (a, d)
    for (p, q), (ta, tb) in zip(zip(called[:-1], called[1:]), zip(times[:-1], times[1:])):
        gap = [x for x in path[path.index(p) + 1:path.index(q)]]
        if not gap:
            continue
        k0, k1 = cor.stations[p].km, cor.stations[q].km
        t0, t1 = ta[1], tb[0]
        for g in gap:
            f = 0.0 if k1 == k0 else (cor.stations[g].km - k0) / (k1 - k0)
            t = t0 + (t1 - t0) * min(max(f, 0.0), 1.0)
            booked[g] = (t, t)

    sched_arr, sched_dep, pad = {}, {origin: base}, {}
    halts = {origin, dest}
    for k in range(1, len(path)):
        i, j = path[k - 1], path[k]
        a, d = booked[j]
        sched_arr[j] = base + (a - dep0)
        sched_dep[j] = base + (d - dep0)
        if d - a > 0.4:
            halts.add(j)
        lo, hi = (i, j) if up else (j, i)
        free = cor.free_run_minutes(lo, hi, spec["max_speed"],
                                    stopping=(d - a) > 0.4)
        run = sched_arr[j] - sched_dep[i]
        pad[j] = max(0.0, run - free)
    return Train(number=num, name=rec.name, klass=klass, up=up,
                 origin=origin, dest=dest, dep_minute=base,
                 halts=sorted(halts, reverse=not up),
                 sched_arr=sched_arr, sched_dep=sched_dep, pad=pad)

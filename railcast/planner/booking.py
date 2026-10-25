"""
When each leg can be booked, and in what order.

For a trip in the next 48 hours the general quota has been open for weeks and
is usually waitlisted. What is left:

    tatkal            opens the day before the train's start date - the date it
                      leaves its first station, not the date you board -
                      at 10:00 for AC classes, 11:00 for sleeper and seating
    current booking   leftover berths, after the chart is prepared about
                      8 hours before the train leaves its first station

Book the scarcest leg first, so nobody ends up holding leg 1 with no way on.
Rule times live in railcast.config.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .. import config
from .network import Network
from .search import Leg


@dataclass
class LegPlan:
    leg: int                 # index into the itinerary
    train: str
    start_date: date         # the train's own start date
    tatkal_opens: datetime
    chart_at: datetime
    departs: datetime        # from the boarding station
    window: str              # what can be done right now
    order: int = 0           # 1 = book this first


def _at(d: date, minutes: float) -> datetime:
    return datetime.combine(d, time()) + timedelta(minutes=float(minutes))


def leg_plan(net: Network, idx: int, leg: Leg, day0: date, klass: str | None,
             now: datetime) -> LegPlan:
    seq = net.calls[leg.train]
    start = day0 + timedelta(days=leg.k)
    hh, mm = (config.TATKAL_OPEN_AC if klass in config.AC_CLASSES
              else config.TATKAL_OPEN_NON_AC)
    tatkal = datetime.combine(start - timedelta(days=1), time(hh, mm))
    origin_dep = _at(start, seq[0].dep)
    chart = origin_dep - timedelta(hours=config.CHART_BEFORE_DEPARTURE_H)
    departs = _at(start, seq[leg.i].dep)
    if now >= departs:
        window = "departed"
    elif now >= chart:
        window = "chart prepared - current booking"
    elif now >= tatkal:
        window = "tatkal open"
    else:
        window = "general quota open; tatkal opens " + tatkal.strftime("%a %d %b %H:%M")
    return LegPlan(idx, leg.train, start, tatkal, chart, departs, window)


def plan(net: Network, legs: list[Leg], day0: date, klass: str | None,
         now: datetime, p_seat: list | None = None) -> list[LegPlan]:
    plans = [leg_plan(net, i, l, day0, klass, now) for i, l in enumerate(legs)]
    if p_seat and all(p is not None for p in p_seat):
        order = sorted(range(len(plans)), key=lambda i: p_seat[i])
    else:   # scarcity unknown: whatever opens first, then the longer leg
        order = sorted(range(len(plans)),
                       key=lambda i: (plans[i].tatkal_opens, -legs[i].km(net)))
    for rank, i in enumerate(order, 1):
        plans[i].order = rank
    return plans


def alerts(plans: list[LegPlan], now: datetime) -> list[tuple[datetime, str]]:
    """The moments a user should be pinged: a minute before tatkal opens, and
    when the chart frees up leftover berths."""
    out = []
    for p in plans:
        if p.tatkal_opens > now:
            out.append((p.tatkal_opens - timedelta(minutes=1),
                        f"tatkal for {p.train} opens in 1 minute"))
        if p.chart_at > now:
            out.append((p.chart_at, f"chart prepared for {p.train} - check current booking"))
    return sorted(out)

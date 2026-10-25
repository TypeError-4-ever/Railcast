"""
The answer to a trip query: search, score, check seats, rank, plan the booking.

    plan(...)       -> Response       the whole pipeline
    Response.text()                   what a person reads in the terminal
    Response.to_dict()                the same thing as JSON, for a bot or an app

Every response says what it does not know: the timetable is an old snapshot,
running days are not in it, seats may not have been checked, and the delay
model may be a stated rule rather than one learned from real running data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .. import config
from . import booking, rank, risk, seats as seats_mod
from . import delays as dl
from .network import Network, current_code
from .search import Query, search

OUT = Path("outputs/plans")


@dataclass
class LegView:
    train: str
    name: str
    frm: str
    frm_name: str
    to: str
    to_name: str
    dep: datetime
    arr: datetime
    km: float
    seat: dict | None
    p_seat: float | None


@dataclass
class ConnectionView:
    at: str
    at_name: str
    board_at: str             # differs from `at` when the change crosses the city
    board_at_name: str
    layover_min: float
    p_make: float


@dataclass
class Option:
    id: int
    group: str
    labels: list
    p_reach: float
    p_connections: float
    seats_checked: bool
    dep: datetime
    arr: datetime                       # timetable
    arr_predicted: tuple                # 10th, 50th, 90th percentile
    hours: float
    changes: int
    km: float
    extra_hours_vs_fastest: float
    off_line: bool
    fare: float | None
    legs: list = field(default_factory=list)
    connections: list = field(default_factory=list)
    booking: list = field(default_factory=list)


@dataclass
class Response:
    query: dict
    generated_at: datetime
    delay_model: str
    seats: str
    caveats: list
    picks: dict
    groups: dict
    options: list
    found: int

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        def conv(v):
            if isinstance(v, (datetime, date)):
                return v.isoformat(timespec="minutes") if isinstance(v, datetime) else v.isoformat()
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, float):
                return round(v, 4)
            return v
        return conv(asdict(self))

    def save(self) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        q = self.query
        path = OUT / f"{q['from']}-{q['to']}-{q['date']}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def text(self, per_group: int = 4) -> str:
        q = self.query
        lines = [
            f"{q['from_name']} ({q['from']}) -> {q['to_name']} ({q['to']})",
            f"{_day(q['date'])}, leaving after {q['after']} within {q['window_h']:.0f} h"
            f" - {q['class'] or 'any class'} - up to {q['max_changes']} change"
            f"{'s' if q['max_changes'] != 1 else ''}",
            "",
        ]
        if not self.options:
            lines += ["No itinerary found in that window.", ""]
            lines += ["Notes:"] + [f"  - {c}" for c in self.caveats]
            return "\n".join(lines)
        by_id = {o.id: o for o in self.options}
        shown = []
        for label, oid in self.picks.items():
            if oid not in shown:
                shown.append(oid)
        for oid in shown:
            lines += _detail(by_id[oid]) + [""]
        for g, ids in self.groups.items():
            if not ids:
                continue
            note = (" - seats not checked, % is connections only"
                    if g == rank.UNCHECKED else "")
            lines.append(f"{g.upper()} ({len(ids)}){note}")
            for oid in ids[:per_group]:
                lines.append("  " + _one_line(by_id[oid]))
            if len(ids) > per_group:
                lines.append(f"  ... {len(ids) - per_group} more in the JSON")
            lines.append("")
        lines.append(f"Delay model: {self.delay_model}")
        lines.append(f"Seats: {self.seats}")
        lines += ["Notes:"] + [f"  - {c}" for c in self.caveats]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _day(d) -> str:
    d = date.fromisoformat(d) if isinstance(d, str) else d
    return d.strftime("%a %d %b %Y")


def _t(dt: datetime) -> str:
    return dt.strftime("%a %H:%M")


def _pct(p: float) -> str:
    return ">99%" if p > 0.995 else f"{100 * p:.0f}%"


def _route(o: Option) -> str:
    via = " via " + ", ".join(c.at for c in o.connections) if o.connections else " direct"
    return " + ".join(l.train for l in o.legs) + via


def _one_line(o: Option) -> str:
    tags = []
    if o.off_line:
        tags.append("other corridor")
    if o.fare is not None:
        tags.append(f"Rs {o.fare:,.0f}")
    tag = f"  [{', '.join(tags)}]" if tags else ""
    return (f"{_pct(o.p_reach):>5}  {_t(o.dep)} -> {_t(o.arr_predicted[1])}  "
            f"{o.hours:4.1f} h  {_route(o)}{tag}")


def _detail(o: Option) -> list[str]:
    head = " / ".join(f"* {l}" for l in o.labels)
    what = "reach" if o.seats_checked else "make every connection (seats not checked)"
    lo, mid, hi = o.arr_predicted
    out = [
        f"{head}  -  {o.group.upper()}, {_pct(o.p_reach)} chance you {what}",
        f"  {_t(o.dep)} {o.legs[0].frm} -> {_t(o.arr)} {o.legs[-1].to} on the timetable; "
        f"predicted arrival {mid.strftime('%a %H:%M')} (80%: {lo.strftime('%H:%M')}-"
        f"{hi.strftime('%H:%M')})",
        f"  {o.hours:.1f} h, "
        + (f"{o.changes} change{'s' if o.changes != 1 else ''}" if o.changes else "direct")
        + f", {o.km:,.0f} km" + (f", Rs {o.fare:,.0f}" if o.fare is not None else "")
        + (", changes on another corridor" if o.off_line else ""),
    ]
    for k, l in enumerate(o.legs):
        seat = ""
        if l.seat:
            seat = f"  seat: {l.seat['text'] or l.seat['status']}"
        out.append(f"  {k + 1}. {l.train} {l.name[:28]:<28} {l.frm} {_t(l.dep)} -> "
                   f"{l.to} {_t(l.arr)}{seat}")
        if k < len(o.connections):
            c = o.connections[k]
            cross = f", then to {c.board_at_name}" if c.board_at != c.at else ""
            out.append(f"     change at {c.at_name}{cross}: {c.layover_min:.0f} min, "
                       f"{_pct(c.p_make)} chance you make it")
    order = sorted(o.booking, key=lambda b: b["order"])
    out.append("  Book: " + "; then ".join(
        f"{b['train']} ({b['window']})" for b in order))
    return out


# --------------------------------------------------------------------------- #
def plan(net: Network, origin: str, dest: str, day0: date, after: str = "00:00",
         window_h: float = config.DEFAULT_WINDOW_H, klass: str | None = None,
         max_changes: int = 1, nearby: bool = True, delays: str = "auto",
         seat_provider=None, now: datetime | None = None,
         include_unreserved: bool = False) -> Response:
    now = now or datetime.now()
    o, d = net.resolve(origin), net.resolve(dest)
    hh, mm = (int(x) for x in after.split(":"))
    q = Query(o, d, t0=hh * 60 + mm, window=window_h * 60, max_changes=max_changes,
              klass=klass, nearby=nearby, include_unreserved=include_unreserved)
    its = search(net, q)
    model = dl.load(delays)
    scored = risk.score(net, its, day0, model)
    provider = seat_provider if seat_provider is not None else seats_mod.provider()
    _check_seats(net, scored, day0, klass, provider, now)
    kept = rank.prune(scored)
    groups = rank.grouped(kept)
    picks = rank.picks(kept)

    base = datetime.combine(day0, time())
    fastest = min((s.it.duration for s in kept), default=0.0)
    options, ids = [], {}
    for g, members in groups.items():
        for s in members:
            ids[id(s)] = len(options)
            options.append(_option(net, s, len(options), g, base, fastest, day0,
                                   klass, now))
    for label, s in picks.items():
        options[ids[id(s)]].labels.append(label)

    caveats = [
        "Timetable: datameet snapshot of the Indian Railways working timetable "
        "(several years old). New trains, retimings and cancellations are missing.",
        "Running days are not in the data - every train is treated as daily. "
        "Confirm the train runs on your date before booking.",
        "Connection chances assume the two trains' delays are independent; in "
        "fog they are not, so treat winter-night chances as optimistic.",
    ]
    if not provider.available:
        caveats.append("Seats were not checked, so groups reflect connection risk "
                       "only. Set RAILKIT_API_KEY and RAILKIT_BASE_URL to check "
                       "seats and fares.")
    return Response(
        query={"from": current_code(o), "from_name": net.name(o),
               "to": current_code(d), "to_name": net.name(d),
               "date": day0.isoformat(), "after": after, "window_h": window_h,
               "class": klass, "max_changes": max_changes, "nearby": nearby},
        generated_at=now,
        delay_model=model.label,
        seats=(f"checked for the top {config.SEAT_CHECKS} options via {provider.name}"
               if provider.available else "not checked"),
        caveats=caveats,
        picks={label: ids[id(s)] for label, s in picks.items()},
        groups={g: [ids[id(s)] for s in members] for g, members in groups.items()},
        options=options,
        found=len(its),
    )


def _check_seats(net, scored, day0, klass, provider, now) -> None:
    """Seats for the most promising options only - each check costs an API call."""
    for s in scored:
        s.seats = [None] * len(s.it.legs)
        s.p_seat = [None] * len(s.it.legs)
    if not provider.available or not klass:
        return
    seen: dict[tuple, object] = {}
    top = sorted(scored, key=lambda s: (-round(s.p_connections, 2), s.it.changes,
                                        s.arrival[1]))[:config.SEAT_CHECKS]
    for s in top:
        fares = []
        for k, leg in enumerate(s.it.legs):
            lp = booking.leg_plan(net, k, leg, day0, klass, now)
            key = (leg.train, leg.frm(net), leg.to(net), lp.departs.date())
            if key not in seen:
                seen[key] = provider.check(leg.train, leg.frm(net), leg.to(net),
                                           lp.departs.date(), klass)
            st = seen[key]
            s.seats[k] = st
            s.p_seat[k] = seats_mod.p_seat(st, (lp.chart_at - now).total_seconds() / 3600)
            fares.append(st.fare if st else None)
        if fares and all(f is not None for f in fares):
            s.fare = float(sum(fares))


def _option(net, s, oid, g, base, fastest, day0, klass, now) -> Option:
    it = s.it
    cc = current_code
    legs = [LegView(l.train, net.trains[l.train].name, cc(l.frm(net)), net.name(l.frm(net)),
                    cc(l.to(net)), net.name(l.to(net)),
                    base + timedelta(minutes=l.dep(net)), base + timedelta(minutes=l.arr(net)),
                    round(l.km(net), 1), seats_mod.as_dict(st), p)
            for l, st, p in zip(it.legs, s.seats, s.p_seat)]
    conns = []
    for k, (a, b) in enumerate(zip(it.legs, it.legs[1:])):
        conns.append(ConnectionView(cc(a.to(net)), net.name(a.to(net)), cc(b.frm(net)),
                                    net.name(b.frm(net)), it.layovers[k], s.p_make[k]))
    plans = booking.plan(net, it.legs, day0, klass, now, s.p_seat)
    return Option(
        id=oid, group=g, labels=[], p_reach=s.p_reach, p_connections=s.p_connections,
        seats_checked=s.seats_checked,
        dep=base + timedelta(minutes=it.dep), arr=base + timedelta(minutes=it.arr),
        arr_predicted=tuple(base + timedelta(minutes=x) for x in s.arrival),
        hours=round(it.duration / 60, 2), changes=it.changes, km=round(it.km, 1),
        extra_hours_vs_fastest=round((it.duration - fastest) / 60, 2),
        off_line=it.off_line, fare=s.fare, legs=legs, connections=conns,
        booking=[{"train": p.train, "order": p.order, "start_date": p.start_date,
                  "tatkal_opens": p.tatkal_opens, "chart_at": p.chart_at,
                  "window": p.window}
                 for p in plans],
    )

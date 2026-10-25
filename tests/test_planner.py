"""
Planner tests on a tiny hand-built network - no timetable download needed.

    AAA ---- XXX ---- BBB      T1 runs the direct line
      \\              /
       \\--- CCC ----/          T2 to CCC, then T3a or T3b on to BBB

CCC is off the direct line, which is the case the product exists for.
"""
from __future__ import annotations

import json
from datetime import date, datetime

import numpy as np
import pytest

from railcast import config
from railcast.dist import quantile_cdf
from railcast.feeds.base import StationRecord
from railcast.planner import booking, rank, risk
from railcast.planner.network import Call, Network, TrainInfo
from railcast.planner.respond import plan
from railcast.planner.search import Query, search
from railcast.planner.seats import NoSeats, SeatStatus, p_seat, parse_availability


def _station(code, lat, lon, name=None):
    return StationRecord(code, name or code, lat, lon, "NR")


def _train(num, stops):
    """stops: (code, arr, dep, km)."""
    calls = [Call(c, a, d, km, True) for c, a, d, km in stops]
    info = TrainInfo(num, f"Train {num}", "SF", "SUPERFAST", "NR", frozenset({"SL", "3A"}), True)
    return info, calls


@pytest.fixture
def net():
    stations = {s.code: s for s in (
        _station("AAA", 28.0, 77.0), _station("XXX", 28.0, 78.5),
        _station("BBB", 28.0, 80.0), _station("CCC", 28.6, 78.5),
        _station("ALD", 25.4, 81.8, "ALLAHABAD JN"), _station("SBC", 12.9, 77.6, "BANGALORE CITY JN"),
        _station("KSR", 10.0, 78.0, "KASTURI"))}
    trains = dict([
        ("T1", _train("T1", [("AAA", 600, 600, 0), ("XXX", 700, 705, 150), ("BBB", 800, 800, 300)])),
        ("T2", _train("T2", [("AAA", 540, 540, 0), ("CCC", 620, 620, 160)])),
        ("T3a", _train("T3a", [("CCC", 650, 650, 0), ("BBB", 740, 740, 160)])),
        ("T3b", _train("T3b", [("CCC", 720, 720, 0), ("BBB", 790, 790, 160)])),
        ("T4", _train("T4", [("ALD", 100, 100, 0), ("SBC", 1500, 1500, 1800), ("KSR", 1700, 1700, 2000)])),
    ])
    n = Network(stations=stations, trains={k: v[0] for k, v in trains.items()},
                calls={k: v[1] for k, v in trains.items()})
    for num, seq in n.calls.items():
        for i, c in enumerate(seq):
            n.halts_at.setdefault(c.code, []).append((num, i))
    return n


# --------------------------------------------------------------------------- #
def test_current_codes_resolve_to_the_snapshot(net):
    assert net.resolve("PRYJ") == "ALD"
    assert net.resolve("allahabad") == "ALD"                  # name prefix
    # KSR is Bengaluru today; in the snapshot KSR is a different station
    assert net.resolve("KSR") == "SBC"
    assert net.name("ALD") == "Prayagraj Jn"


def test_search_finds_direct_and_off_line_connection(net):
    its = search(net, Query("AAA", "BBB", t0=0, window=1440, max_changes=1))
    routes = {tuple(l.train for l in it.legs): it for it in its}
    assert ("T1",) in routes
    via = routes[("T2", "T3a")]
    assert via.legs[0].to(net) == "CCC"
    assert via.off_line                       # CCC is not on T1's route
    assert via.layovers == [30.0]
    # T3b arrives later than T3a but with a longer layover - safer, so kept
    assert ("T2", "T3b") in routes


def test_no_changes_means_direct_only(net):
    its = search(net, Query("AAA", "BBB", t0=0, window=1440, max_changes=0))
    assert all(it.changes == 0 for it in its)


def test_bigger_buffer_means_better_chance(net):
    its = {tuple(l.train for l in it.legs): it
           for it in search(net, Query("AAA", "BBB", t0=0, window=1440, max_changes=1))}
    rng = np.random.default_rng(0)
    arr_q, dep_q = (0.0, 15.0, 45.0), (0.0, 0.0, 5.0)
    tight = its[("T2", "T3a")]
    loose = its[("T2", "T3b")]
    p_tight = risk.p_make(net, tight.legs[0], tight.legs[1], arr_q, dep_q, rng)
    p_loose = risk.p_make(net, loose.legs[0], loose.legs[1], arr_q, dep_q, rng)
    assert 0.0 < p_tight < p_loose <= 1.0


def test_group_thresholds():
    assert rank.group(config.SAFE) == "Safe"
    assert rank.group(config.SAFE - 0.01) == "Likely"
    assert rank.group(config.LIKELY) == "Likely"
    assert rank.group(config.LIKELY - 0.01) == "Risky"


@pytest.mark.parametrize("text,status,n", [
    ("AVAILABLE-0023", "CNF", 23), ("CURR_AVBL-0012", "CNF", 12), ("RAC 12", "RAC", 12),
    ("GNWL45/WL30", "WL", 30), ("PQWL12", "WL", 12), ("REGRET", "REGRET", None),
    ("NOT AVAILABLE", "REGRET", None), ("Charting Done", "UNKNOWN", None),
])
def test_parse_availability(text, status, n):
    assert parse_availability(text) == (status, n)


def test_seat_probability():
    def s(status, n=None):
        return SeatStatus(status, n, "", None, "GN", "test", 0.0)
    assert p_seat(None, 10) is None
    assert p_seat(s("RAC", 3), 10) == 1.0
    assert p_seat(s("WL", 3), 30) == config.P_SEAT_WL_SHORT
    assert p_seat(s("WL", 3), 5) == config.P_SEAT["WL"]
    assert p_seat(s("REGRET"), 30) == 0.0


def test_booking_windows(net):
    leg = search(net, Query("AAA", "BBB", t0=0, window=1440, max_changes=0))[0].legs[0]
    d = date(2026, 10, 27)
    now = datetime(2026, 10, 26, 9, 0)
    ac = booking.leg_plan(net, 0, leg, d, "3A", now)
    sl = booking.leg_plan(net, 0, leg, d, "SL", now)
    assert ac.tatkal_opens == datetime(2026, 10, 26, 10, 0)
    assert sl.tatkal_opens == datetime(2026, 10, 26, 11, 0)
    assert ac.chart_at == datetime(2026, 10, 27, 2, 0)          # 10:00 departure - 8 h
    assert ac.window.startswith("general quota open")


def test_quantile_cdf_hits_the_median():
    assert quantile_cdf([0.0], [10.0], [30.0], [10.0])[0] == pytest.approx(0.5)


def test_plan_end_to_end(net):
    resp = plan(net, "AAA", "BBB", date(2026, 10, 27), klass="SL", max_changes=1,
                delays="prior", seat_provider=NoSeats(), now=datetime(2026, 10, 26, 9, 0))
    assert resp.options and "Safest" in resp.picks
    assert set(resp.groups) == set(rank.GROUPS)
    assert resp.seats == "not checked"
    text = resp.text()
    assert "Seats were not checked" in text and "AAA" in text
    json.dumps(resp.to_dict())                                # serialisable

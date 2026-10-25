"""
Safe / Likely / Risky, and the three picks.

Priority, from the PRD: the chance of reaching B, then predicted real travel
time, then fare. Options are grouped by that chance rather than sorted strictly
by it - a strict sort puts a confirmed 38-hour trip with two changes above an
18-hour one at 95%, and nobody wants that.
"""
from __future__ import annotations

from .. import config
from .risk import Scored

GROUPS = ("Safe", "Likely", "Risky")


def group(p: float) -> str:
    if p >= config.SAFE:
        return "Safe"
    if p >= config.LIKELY:
        return "Likely"
    return "Risky"


def _sort_key(s: Scored):
    return (s.arrival[1], s.fare if s.fare is not None else float("inf"),
            s.it.changes)


def prune(scored: list[Scored]) -> list[Scored]:
    """Drop an option another beats or ties on every count: chance of reaching
    B, predicted arrival, departure (later is better), changes and fare."""
    def fare(s):
        return s.fare if s.fare is not None else float("inf")

    ordered = sorted(scored, key=lambda s: (-s.p_reach, s.arrival[1]))
    kept: list[Scored] = []
    for s in ordered:
        if not any(k.p_reach >= s.p_reach and k.arrival[1] <= s.arrival[1]
                   and k.it.dep >= s.it.dep and k.it.changes <= s.it.changes
                   and fare(k) <= fare(s) for k in kept):
            kept.append(s)
    return kept


def grouped(scored: list[Scored]) -> dict[str, list[Scored]]:
    out = {g: [] for g in GROUPS}
    for s in scored:
        out[group(s.p_reach)].append(s)
    for g in out:
        out[g].sort(key=_sort_key)
    return out


def picks(scored: list[Scored]) -> dict[str, Scored]:
    """Safest, Fastest (among options at least Likely, if any), Cheapest
    (only when fares are known)."""
    if not scored:
        return {}
    # to the percent: at equal chance, fewer changes is the safer trip
    out = {"Safest": max(scored, key=lambda s: (round(s.p_reach, 2), -s.it.changes,
                                                -s.arrival[1]))}
    good = [s for s in scored if s.p_reach >= config.LIKELY] or scored
    out["Fastest"] = min(good, key=lambda s: (s.arrival[1], -s.p_reach))
    priced = [s for s in good if s.fare is not None]
    if priced:
        out["Cheapest"] = min(priced, key=lambda s: (s.fare, s.arrival[1]))
    return out

"""
Build the forecasting dataset.

One row = one (train, snapshot time, downstream station) forecasting problem:
the network state as it was observable at `now`, and what actually happened at
that station later. This is the direct multi-horizon formulation - the horizon
is a feature, so a 30-hour forecast is made in one shot and never chained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .corridor import CORRIDOR
from .simulator import Simulator, make_environment
from .trains import CLASSES, build_roster

ZONE_ID = {"NR": 0, "NCR": 1, "WCR": 2, "WR": 3}
KLASS_ID = {k: i for i, k in enumerate(CLASSES)}
SNAPSHOTS_PER_DAY = 6


def calendar(day: int) -> tuple[int, int]:
    """Simulation day -> (calendar month 1-12, day of week 0-6)."""
    return (day // 30) % 12 + 1, day % 7


class World:
    """Simulates operating days and keeps the ground truth for each."""

    def __init__(self, n_days: int, seed: int = 11):
        self.rng = np.random.default_rng(seed)
        self.cor = CORRIDOR
        self.trains = build_roster()
        self.sim = Simulator(self.cor, self.trains)
        self.n_days = n_days
        self.days: dict[int, tuple] = {}

    def simulate(self) -> None:
        for d in range(self.n_days):
            env = make_environment(d, self.rng, self.cor)
            truth = self.sim.run_day(env, self.rng)
            self.days[d] = (env, truth)

    # ------------------------------------------------------------------ #
    # stage A training data: observed section run times
    # ------------------------------------------------------------------ #
    def section_runtime_rows(self, days) -> pd.DataFrame:
        rows = []
        for d in days:
            env, truth = self.days[d]
            fc = env.forecast()
            month, _ = calendar(d)
            for tid, run in truth.items():
                t = self.sim.trains[tid]
                for a, b in zip(t.path[:-1], t.path[1:]):
                    if a not in run.dep or b not in run.arr:
                        continue
                    lo, hi = (a, b) if t.up else (b, a)
                    ids = self.cor.section_blocks(lo, hi)
                    theo = self.cor.free_run_minutes(
                        lo, hi, t.max_speed, stopping=b in t.halts)
                    obs = run.arr[b] - run.dep[a]
                    conflict = run.conflict.get(b, 0.0) + run.incident.get(b, 0.0)
                    blk = self.cor.blocks[ids[len(ids) // 2]]
                    rows.append((
                        theo, obs, conflict,
                        CLASSES[t.klass]["priority"], t.max_speed,
                        ZONE_ID[blk.zone], int((run.dep[a] / 60) % 24),
                        self.cor.stations[hi].km - self.cor.stations[lo].km,
                        blk.speed, fc.factor(blk.zone, run.dep[a]),
                        int(b in t.halts), month,
                        min((env.tsr.get(x, 200.0) for x in ids), default=200.0),
                    ))
        return pd.DataFrame(rows, columns=[
            "theo_min", "obs_min", "conflict_min", "priority", "max_speed",
            "zone", "hour", "length_km", "sect_speed", "weather_fc",
            "stopping", "month", "tsr_cap"])

    # ------------------------------------------------------------------ #
    # observable network state
    # ------------------------------------------------------------------ #
    def _train_positions(self, truth, now: float):
        pos = {}
        for tid, run in truth.items():
            t = self.sim.trains[tid]
            last = None
            for s in t.path:
                d = run.dep.get(s)
                if d is not None and d <= now:
                    last = s
                else:
                    break
            if last is None:
                if run.dep.get(t.origin, 1e18) <= now:
                    continue
                pos[tid] = dict(started=False, last=t.origin,
                                km=self.cor.stations[t.origin].km,
                                delay=max(0.0, now - t.dep_minute),
                                momentum=0.0, unsched=0)
                continue
            if last == t.dest:
                continue
            k = t.path.index(last)
            delay = run.dep[last] - t.sched_dep.get(last, run.dep[last])
            back = t.path[max(0, k - 3)]
            delay3 = run.dep.get(back, run.dep[last]) - t.sched_dep.get(back, 0.0)
            pos[tid] = dict(
                started=True, last=last, km=self.cor.stations[last].km,
                delay=delay, momentum=delay - delay3,
                unsched=sum(run.unsched_stops.get(s, 0)
                            for s in t.path[max(0, k - 3):k + 1]))
        return pos

    def _tsr_prefix(self, t, env):
        """Cumulative count of remaining sections carrying a speed restriction."""
        out, c = {}, 0
        for a, b in zip(t.path[:-1], t.path[1:]):
            lo, hi = (a, b) if t.up else (b, a)
            if any(x in env.tsr for x in self.cor.section_blocks(lo, hi)):
                c += 1
            out[b] = c
        return out

    # ------------------------------------------------------------------ #
    def snapshot_rows(self, day: int, now: float, fr_table: dict):
        env, truth = self.days[day]
        fc = env.forecast()
        month, dow = calendar(day)
        rep = self.sim.forward_replay(fc, now, truth, fr_table=fr_table)
        pos = self._train_positions(truth, now)
        rows = []
        for tid, p in pos.items():
            t = self.sim.trains[tid]
            run, pred = truth[tid], rep.get(tid)
            if pred is None:
                continue
            i, path = p["last"], t.path
            k = path.index(i)
            others = [(q["km"], q["delay"]) for o, q in pos.items()
                      if o != tid and self.sim.trains[o].up == t.up]
            sgn = 1.0 if t.up else -1.0
            ahead = [d for km, d in others if 0 < sgn * (km - p["km"]) < 120]
            near = sum(1 for km, _ in others if abs(km - p["km"]) < 60)
            ahead_delay = float(np.mean(ahead)) if ahead else 0.0
            tsrp = self._tsr_prefix(t, env)
            pad_cum = 0.0
            for h, j in enumerate(path[k + 1:], start=1):
                pad_cum += t.pad.get(j, 0.0)
                if j not in run.arr or j not in pred.arr:
                    continue
                lead = t.sched_arr[j] - now
                if lead < -300 or lead > 46 * 60:
                    continue
                free = self.cor.free_run_span(i, j, t.max_speed)
                rows.append((
                    day, month, dow, tid, KLASS_ID[t.klass], int(t.up),
                    self.cor.code(i), self.cor.code(j), i, j,
                    now, h, lead, pred.arr[j] - now,
                    p["delay"], p["momentum"], int(p["started"]), p["unsched"],
                    pad_cum, abs(self.cor.stations[j].km - p["km"]),
                    free, pred.arr[j] - now - free,
                    CLASSES[t.klass]["priority"], t.max_speed,
                    near, ahead_delay, tsrp.get(j, 0) - tsrp.get(i, 0),
                    fc.factor(self.cor.stations[j].zone, t.sched_arr[j]),
                    ZONE_ID[self.cor.stations[j].zone], int((now / 60) % 24),
                    t.sched_arr[j], pred.arr[j], run.arr[j],
                ))
        return rows


COLUMNS = [
    "day", "month", "dow", "train", "klass", "up",
    "from_code", "to_code", "from_idx", "to_idx",
    "now", "hops", "sched_lead", "rules_lead",
    "cur_delay", "delay_momentum", "started", "unsched_stops",
    "pad_remaining", "dist_remaining_km",
    "free_run_min", "conflict_pred_min",
    "priority", "max_speed",
    "trains_near", "ahead_delay", "tsr_sections",
    "weather_fc", "zone", "hour",
    "sched_arr", "rules_arr", "act_arr",
]

FEATURES = [
    "hops", "sched_lead", "rules_lead", "cur_delay", "delay_momentum",
    "started", "unsched_stops", "pad_remaining", "dist_remaining_km",
    "free_run_min", "conflict_pred_min", "priority", "max_speed",
    "trains_near", "ahead_delay", "tsr_sections", "weather_fc",
    "zone", "hour", "klass", "up", "month", "dow",
]
CATEGORICAL = ["zone", "klass", "month", "dow"]

PRETTY = {
    "rules_lead": "Simulated conflict-resolved ETA",
    "cur_delay": "Current delay",
    "conflict_pred_min": "Conflict minutes predicted ahead",
    "sched_lead": "Scheduled lead time",
    "pad_remaining": "Recovery padding remaining",
    "delay_momentum": "Delay momentum (last 3 sections)",
    "ahead_delay": "Delay of trains ahead",
    "dist_remaining_km": "Distance remaining",
    "hops": "Stations remaining",
    "free_run_min": "Free-run time remaining",
    "trains_near": "Trains within 60 km",
    "tsr_sections": "Sections under speed restriction",
    "weather_fc": "Weather factor (forecast)",
    "zone": "Zone", "hour": "Time of day", "klass": "Train class",
    "priority": "Priority class", "max_speed": "Max permissible speed",
    "up": "Direction", "month": "Month", "dow": "Day of week",
    "started": "Has departed origin", "unsched_stops": "Unscheduled stops (phone-sensed)",
}


def build_dataset(world: World, fr_table: dict, rng, snapshots: int = SNAPSHOTS_PER_DAY) -> pd.DataFrame:
    rows = []
    for d in range(world.n_days):
        for now in np.sort(rng.uniform(0, 1440, snapshots)):
            rows.extend(world.snapshot_rows(d, float(now), fr_table))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["residual"] = df["act_arr"] - df["rules_arr"]
    df["baseline_arr"] = df["sched_arr"] + df["cur_delay"]
    absorb = np.minimum(np.maximum(df["cur_delay"], 0.0), 0.5 * df["pad_remaining"])
    df["baseline_pad_arr"] = df["baseline_arr"] - absorb
    return df

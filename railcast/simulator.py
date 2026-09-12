"""
Event-driven movement simulator over the block-section graph.

Two jobs, one engine:

  run_day        - generates ground truth. Stochastic: run-time noise, weather,
                   temporary speed restrictions, incidents, and conflicts
                   resolved by priority rules.
  forward_replay - the forecasting primitive. Deterministic, started from the
                   network state at a snapshot time, using forecast weather and
                   known restrictions. This is stage B of the RAILCAST loop:
                   every train advanced over the block graph until it conflicts.

The replay never sees future noise, so it is an honest estimator - which is
exactly what leaves a residual for the learned layer to correct.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .corridor import CORRIDOR, Corridor
from .noise import DayNoise, draw_day
from .trains import Train

HEADWAY = 3.0          # minutes of absolute block working, same direction
MAX_HOLD = 25.0        # a loop hold is never open-ended
PRECEDENCE_LOOKAHEAD = 20.0


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
@dataclass
class DayEnvironment:
    """Weather and restrictions for one operating day."""
    day: int
    month: int
    fog: dict           # zone -> night speed multiplier
    rain: dict          # zone -> all-day speed multiplier
    tsr: dict           # block idx -> capped speed km/h

    def factor(self, zone: str, minute: float) -> float:
        hour = (minute / 60.0) % 24
        f = self.rain.get(zone, 1.0)
        if hour >= 22 or hour <= 7:
            f *= self.fog.get(zone, 1.0)
        return f

    def forecast(self) -> "DayEnvironment":
        """Weather as known at forecast time: damped towards normal."""
        fog = {z: 1.0 - (1.0 - v) * 0.72 for z, v in self.fog.items()}
        rain = {z: 1.0 - (1.0 - v) * 0.80 for z, v in self.rain.items()}
        return DayEnvironment(self.day, self.month, fog, rain, dict(self.tsr))


def make_environment(day: int, rng, cor: Corridor = CORRIDOR) -> DayEnvironment:
    month = (day // 30) % 12 + 1
    fog, rain = {}, {}
    if month in (12, 1, 2):                       # north India winter fog
        sev = float(rng.beta(1.6, 3.0))
        for z, w in (("NR", 1.0), ("NCR", 0.95), ("WCR", 0.55), ("WR", 0.1)):
            fog[z] = 1.0 - 0.26 * sev * w
    if month in (6, 7, 8, 9):                     # monsoon, strongest on WR
        sev = float(rng.beta(1.5, 3.5))
        for z, w in (("WR", 1.0), ("WCR", 0.7), ("NCR", 0.4), ("NR", 0.3)):
            rain[z] = 1.0 - 0.11 * sev * w
    tsr = {}
    for _ in range(int(rng.poisson(3.0))):
        b = int(rng.integers(0, len(cor.blocks)))
        span = int(rng.integers(1, 5))
        cap = float(rng.choice([30.0, 45.0, 60.0, 75.0]))
        for k in range(b, min(b + span, len(cor.blocks))):
            tsr[k] = min(tsr.get(k, 1e9), cap)
    return DayEnvironment(day, month, fog, rain, tsr)


# --------------------------------------------------------------------------- #
# simulator
# --------------------------------------------------------------------------- #
@dataclass
class TrainRun:
    tid: str
    arr: dict = field(default_factory=dict)
    dep: dict = field(default_factory=dict)
    conflict: dict = field(default_factory=dict)     # minutes lost before station
    held_by: dict = field(default_factory=dict)
    incident: dict = field(default_factory=dict)
    unsched_stops: dict = field(default_factory=dict)


class Simulator:
    def __init__(self, cor: Corridor, trains):
        self.cor = cor
        self.trains = {t.number: t for t in trains}
        self.seqs = {}
        self.section_end = {}
        for t in trains:
            blocks, ends = [], {}
            path = t.path
            for a, b in zip(path[:-1], path[1:]):
                lo, hi = (a, b) if t.up else (b, a)
                ids = cor.section_blocks(lo, hi)
                ids = ids if t.up else list(reversed(ids))
                blocks.extend(ids)
                ends[len(blocks) - 1] = b          # last block of this section
            self.seqs[t.number] = blocks
            self.section_end[t.number] = ends
        self.priority_override: dict = {}
        self.precedence_swap: set = set()   # pairs whose precedence is reversed

    def prio(self, tid: str) -> int:
        return self.priority_override.get(tid, self.trains[tid].priority)

    # -- core loop ---------------------------------------------------------- #
    def _run(self, env: DayEnvironment, starts: dict,
             noise: DayNoise | None = None, fr_table: dict | None = None):
        """starts: tid -> (block-sequence position, ready time).

        noise     - pre-drawn stochastic terms. Ground truth only.
        fr_table  - learned free-run multiplier, (class, zone, hour band) ->
                    factor. Stage A of the forecast; replay only.
        """
        cor = self.cor
        stochastic = noise is not None
        fr_table = fr_table or {}
        occ = {}
        last_user = {}
        runs = {tid: TrainRun(tid) for tid in starts}
        state = {}                                  # tid -> (km, time)

        heap = []
        for k, (tid, (pos, t0)) in enumerate(starts.items()):
            heapq.heappush(heap, (t0, k, tid, pos))
            seq = self.seqs[tid]
            km = cor.blocks[seq[pos]].km_start if pos < len(seq) else 0.0
            state[tid] = (km, t0)
            # the station the train is standing at when the run begins
            stn0 = self._prev_station(tid, pos) if pos else self.trains[tid].origin
            runs[tid].arr[stn0] = t0
            runs[tid].dep[stn0] = t0
        seq_counter = len(starts)

        while heap:
            t, _, tid, pos = heapq.heappop(heap)
            train = self.trains[tid]
            seq = self.seqs[tid]
            if pos >= len(seq):
                continue
            b = seq[pos]
            blk = cor.blocks[b]
            key = (train.up, b)

            # --- block occupancy: absolute block working ---
            free_at = occ.get(key, -1e9) + HEADWAY
            entry = max(t, free_at)
            wait = entry - t
            if wait > 0.2:
                stn = self._prev_station(tid, pos)
                runs[tid].conflict[stn] = runs[tid].conflict.get(stn, 0.0) + wait
                other = last_user.get(key)
                if other and other != tid:
                    runs[tid].held_by[stn] = other
                if wait > 1.5:
                    runs[tid].unsched_stops[stn] = runs[tid].unsched_stops.get(stn, 0) + 1

            # --- traverse ---
            wf = env.factor(blk.zone, entry)
            v = min(blk.speed, train.max_speed) * wf
            v = min(v, env.tsr.get(b, 1e9))
            travel = blk.length / max(v, 15.0) * 60.0
            if stochastic:
                travel *= noise.friction[tid] * float(noise.block[tid][pos])
            else:
                travel *= fr_table.get(
                    (train.klass, blk.zone, int((entry / 60.0) % 24) // 6), 1.0)
            exit_t = entry + travel
            occ[key] = exit_t
            last_user[key] = tid
            state[tid] = (blk.km_end, exit_t)

            stn = self.section_end[tid].get(pos)
            if stn is None:
                heapq.heappush(heap, (exit_t, seq_counter, tid, pos + 1))
                seq_counter += 1
                continue

            # --- arrival at a station ---
            arr = exit_t + (1.4 if stn in train.halts else 0.6)
            if stochastic and (tid, stn) in noise.incident:
                inc = noise.incident[(tid, stn)]
                arr += inc
                runs[tid].incident[stn] = inc
            runs[tid].arr[stn] = arr

            if pos + 1 >= len(seq):
                runs[tid].dep[stn] = arr
                continue

            dwell = 0.0
            if stn in train.halts:
                sd = train.sched_dep.get(stn, arr) - train.sched_arr.get(stn, arr)
                dwell = max(1.0, sd)
                if stochastic:
                    dwell += float(noise.dwell[tid][train.path.index(stn)])
            dep = arr + dwell
            # operating rule: a train is not run ahead of its booked path. It
            # waits at the last station rather than arriving early. This is how
            # schedule padding is actually absorbed.
            dep = max(dep, train.sched_dep.get(stn, dep))

            # --- precedence: hold the lower-priority train in the loop ---
            hold, blocker = self._precedence(tid, stn, dep, state)
            if hold > 0:
                dep += hold
                runs[tid].conflict[stn] = runs[tid].conflict.get(stn, 0.0) + hold
                runs[tid].held_by[stn] = blocker
            runs[tid].dep[stn] = dep
            state[tid] = (cor.stations[stn].km, dep)
            heapq.heappush(heap, (dep, seq_counter, tid, pos + 1))
            seq_counter += 1
        return runs

    def _prev_station(self, tid: str, pos: int) -> int:
        ends = self.section_end[tid]
        for p in range(pos - 1, -1, -1):
            if p in ends:
                return ends[p]
        return self.trains[tid].origin

    def _precedence(self, tid: str, stn: int, dep: float, state: dict):
        """Is a higher-priority train closing up behind? If so, loop this one."""
        me = self.trains[tid]
        if not self.cor.stations[stn].has_loop:
            return 0.0, ""
        my_km = self.cor.stations[stn].km
        best, blocker = 0.0, ""
        for otid, (okm, ot) in state.items():
            if otid == tid:
                continue
            other = self.trains[otid]
            if other.up != me.up:
                continue
            gives_way = self.prio(otid) < self.prio(tid)
            if frozenset((tid, otid)) in self.precedence_swap:
                gives_way = not gives_way          # the counterfactual decision
            if not gives_way:
                continue
            gap = (my_km - okm) if me.up else (okm - my_km)
            if not (0 < gap < 75):
                continue
            eta = ot + gap / max(other.max_speed * 0.85, 40.0) * 60.0
            slack = eta - dep
            if -6.0 < slack < PRECEDENCE_LOOKAHEAD:
                hold = min(MAX_HOLD, max(0.0, slack) + HEADWAY + 2.0)
                if hold > best:
                    best, blocker = hold, otid
        return best, blocker

    # -- public entry points ------------------------------------------------ #
    def run_day(self, env: DayEnvironment, rng=None, noise: DayNoise | None = None):
        noise = noise if noise is not None else draw_day(self, rng)
        starts = {tid: (0, t.dep_minute + noise.origin_delay[tid])
                  for tid, t in self.trains.items()}
        return self._run(env, starts, noise)

    def forward_replay(self, env_fc: DayEnvironment, now: float, truth: dict,
                       fr_table: dict | None = None):
        """Forecast every running train forward from the state observed at now."""
        starts = {}
        for tid, t in self.trains.items():
            seq = self.seqs[tid]
            run = truth[tid]
            path = t.path
            pos = ready = None
            last = None
            for s in path:
                d = run.dep.get(s)
                if d is not None and d <= now:
                    last = s
                else:
                    break
            if last is None:
                if run.dep.get(t.origin, 1e9) > now:
                    pos, ready = 0, max(now, t.dep_minute)
            elif last == t.dest:
                continue
            else:
                k = path.index(last)
                nblocks = 0
                for a, b in zip(path[:k], path[1:k + 1]):
                    lo, hi = (a, b) if t.up else (b, a)
                    nblocks += len(self.cor.section_blocks(lo, hi))
                pos = nblocks
                ready = max(now, run.dep[last])
            if pos is not None and pos < len(seq):
                starts[tid] = (pos, ready)
        return self._run(env_fc, starts, None, fr_table=fr_table)

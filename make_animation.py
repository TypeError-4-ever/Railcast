"""
RAILCAST - animated demo of one simulated day on the real corridor.

    python make_animation.py                 # 1080p MP4, ~25 s
    python make_animation.py --seconds 15 --gif

Nothing here is drawn by hand. Train positions are interpolated from the
simulator's own arrival and departure times, and every conflict caption names
the train that actually caused it, taken from the run log.

Note on what this shows: trains never collide. Absolute block working makes that
impossible - a train cannot enter a block section another train occupies. What
it shows instead is what actually costs Indian Railways time: a faster train
closing on a slower one, a train looped at a station so a premier service can
pass, and the delay cascading down the line behind it.
"""
from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch

from railcast.feeds import DatameetFeed
from railcast.feeds.corridor import build_corridor, build_roster
from railcast.feeds.openmeteo import build_environments
from railcast.simconfig import SimConfig
from railcast.simulator import Simulator

BG = "#0B0E14"
PANEL = "#141924"
INK = "#F2F5FA"
MUTE = "#8B94A3"
GRID = "#232A38"
ACCENT = "#F2A03D"
BAD = "#E2574C"
GOOD = "#4CAF7D"
CLASS_COLOR = {"RAJDHANI": "#F2A03D", "DURONTO": "#E2574C", "SHATABDI": "#C77DFF",
               "SUPERFAST": "#4D8FE8", "EXPRESS": "#3DA5A0", "PASSENGER": "#8A94A6"}
ANCHORS = ("NDLS", "MTJ", "KOTA", "RTM", "BRC", "ST", "BCT")


# --------------------------------------------------------------------------- #
class Playback:
    """Where every train is, at any minute of the simulated day."""

    def __init__(self, sim, truth, cor):
        self.cor = cor
        self.tracks: dict[str, tuple] = {}
        self.events: list[dict] = []
        for tid, run in truth.items():
            t = sim.trains[tid]
            ts, kms = [], []
            for s in t.path:
                km = cor.stations[s].km
                if s in run.arr:
                    ts.append(run.arr[s])
                    kms.append(km)
                if s in run.dep:
                    ts.append(run.dep[s])
                    kms.append(km)
            if len(ts) < 4:
                continue
            order = np.argsort(ts)
            self.tracks[tid] = (np.array(ts)[order], np.array(kms)[order],
                                t.klass, t.up, t.name)
            for stn, mins in run.conflict.items():
                if mins < 12 or stn not in run.arr:
                    continue
                self.events.append({
                    "t": run.arr[stn], "train": tid, "km": cor.stations[stn].km,
                    "station": cor.code(stn), "minutes": mins,
                    "blocker": run.held_by.get(stn, ""),
                    "cause": run.cause.get(stn, "block"), "up": t.up})
        self.events.sort(key=lambda e: e["t"])

    def at(self, t: float):
        out = []
        for tid, (ts, kms, klass, up, name) in self.tracks.items():
            if t < ts[0] or t > ts[-1]:
                continue
            out.append((tid, float(np.interp(t, ts, kms)), klass, up, name))
        return out

    def trail(self, tid: str, t: float, span: float = 240.0):
        ts, kms, *_ = self.tracks[tid]
        m = (ts >= t - span) & (ts <= t)
        return ts[m], kms[m]


def _hm(minute: float) -> str:
    m = int(minute) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


# --------------------------------------------------------------------------- #
def pick_day(sim, envs, cfg, days: int, rng_seed: int = 4):
    """The day with the most conflict, because that is the day worth watching."""
    best = None
    for d in range(days):
        rng = np.random.default_rng(rng_seed + d)
        truth = sim.run_day(envs[d], rng)
        score = sum(sum(r.conflict.values()) for r in truth.values())
        if best is None or score > best[0]:
            best = (score, d, truth)
    return best[1], best[2], best[0]


# --------------------------------------------------------------------------- #
def build(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    feed = DatameetFeed()
    cor, meta = build_corridor(feed, args.reference)
    trains = build_roster(feed, cor, limit=args.trains)
    envs = build_environments(cor, feed, args.start_date, 8,
                              np.random.default_rng(1))
    cfg = SimConfig.load()
    sim = Simulator(cor, trains, cfg)
    day, truth, conflict_total = pick_day(sim, envs, cfg, 8)
    pb = Playback(sim, truth, cor)
    print(f"  day {day}: {len(pb.tracks)} trains, "
          f"{len(pb.events)} conflict events, "
          f"{conflict_total:.0f} conflict-minutes")

    t0 = min(ts[0] for ts, *_ in pb.tracks.values())
    t1 = min(max(ts[-1] for ts, *_ in pb.tracks.values()),
             t0 + args.hours * 60.0)
    frames = int(args.seconds * args.fps)
    clock = np.linspace(t0, t1, frames)

    anchors = [(i, cor.code(i)) for i in range(cor.n_stations)
               if cor.code(i) in ANCHORS]
    total_km = cor.stations[-1].km

    # ---------------- figure ---------------- #
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    fig.patch.set_facecolor(BG)
    ax_strip = fig.add_axes((0.04, 0.46, 0.92, 0.34))
    ax_marey = fig.add_axes((0.04, 0.07, 0.55, 0.31))
    ax_info = fig.add_axes((0.63, 0.07, 0.33, 0.31))
    for ax in (ax_strip, ax_marey, ax_info):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.text(0.04, 0.945, "RAILCAST", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.207, 0.951, "one simulated day, New Delhi - Mumbai Central",
             fontsize=13, color=MUTE)
    fig.text(0.04, 0.905, f"{meta['spine_stations']} stations   "
             f"{meta['route_km']:.0f} km   {len(pb.tracks)} trains   "
             "real working timetable, observed weather",
             fontsize=11, color=MUTE)
    fig.text(0.04, 0.018, "Trains never collide - absolute block working makes "
             "that impossible. What costs time is conflict: a faster train "
             "closing on a slower one, and the hold that follows.",
             fontsize=9.5, color=MUTE)

    # corridor strip
    ax_strip.set_xlim(-30, total_km + 30)
    ax_strip.set_ylim(-1.5, 1.6)
    ax_strip.set_yticks([])
    ax_strip.set_xticks([])
    for lane, label in ((0.55, "UP   Delhi to Mumbai"), (-0.55, "DOWN   Mumbai to Delhi")):
        ax_strip.plot([0, total_km], [lane, lane], color=GRID, lw=7,
                      solid_capstyle="round", zorder=1)
        ax_strip.text(-25, lane + 0.26, label, fontsize=9, color=MUTE)
    for i, code in anchors:
        km = cor.stations[i].km
        ax_strip.plot([km, km], [-0.85, 0.85], color=GRID, lw=1.2, zorder=0)
        ax_strip.text(km, -1.2, code, ha="center", fontsize=10.5,
                      color=MUTE, fontweight="bold")

    # marey
    ax_marey.set_xlim(t0, t1)
    ax_marey.set_ylim(0, total_km)
    ax_marey.set_ylabel("km from New Delhi", fontsize=9, color=MUTE)
    ax_marey.tick_params(colors=MUTE, labelsize=8)
    ax_marey.grid(color=GRID, lw=0.5)
    ticks = np.arange(t0, t1, 240)
    ax_marey.set_xticks(ticks)
    ax_marey.set_xticklabels([f"+{(x - t0) / 60:.0f}h" for x in ticks])
    ax_marey.set_title("time-distance: every train, and where they meet",
                       fontsize=10.5, color=MUTE, loc="left")
    ax_info.axis("off")

    clock_txt = fig.text(0.80, 0.938, "", fontsize=30, fontweight="bold",
                         color=ACCENT, ha="right", family="monospace")
    count_txt = fig.text(0.965, 0.938, "", fontsize=12, color=MUTE, ha="right")

    writer = (PillowWriter(fps=args.fps) if args.gif
              else FFMpegWriter(fps=args.fps, bitrate=args.bitrate,
                                metadata={"title": "RAILCAST prototype"}))
    started = _time.time()
    with writer.saving(fig, str(out), args.dpi):
        dyn: list = []
        for k, t in enumerate(clock):
            for a in dyn:
                a.remove()
            dyn = []

            # --- marey traces grow with the clock ---
            for tid, (ts, kms, klass, up, _n) in pb.tracks.items():
                m = ts <= t
                if m.sum() > 1:
                    ln, = ax_marey.plot(ts[m], kms[m], lw=1.0,
                                        color=CLASS_COLOR.get(klass, MUTE),
                                        alpha=0.75, zorder=2)
                    dyn.append(ln)
            vl = ax_marey.axvline(t, color=INK, lw=1.0, alpha=0.6)
            dyn.append(vl)

            # --- trains on the strip ---
            live = pb.at(t)
            for tid, km, klass, up, _name in live:
                lane = 0.55 if up else -0.55
                col = CLASS_COLOR.get(klass, MUTE)
                tr_t, tr_k = pb.trail(tid, t, 90)
                if len(tr_k) > 1:
                    ln, = ax_strip.plot(tr_k, [lane] * len(tr_k), color=col,
                                        lw=5, alpha=0.25, solid_capstyle="round",
                                        zorder=2)
                    dyn.append(ln)
                mk, = ax_strip.plot([km], [lane], marker="o", ms=11, color=col,
                                    mec=BG, mew=1.6, zorder=4)
                dyn.append(mk)

            # --- conflict callouts ---
            active = [e for e in pb.events if 0 <= t - e["t"] < 90]
            for e in active[-1:]:
                age = (t - e["t"]) / 90.0
                lane = 0.55 if e["up"] else -0.55
                fade = max(0.0, 1.0 - age)
                sc = ax_strip.scatter([e["km"]], [lane], s=180 + 2600 * age,
                                      facecolors="none", edgecolors=BAD,
                                      linewidths=2.6, alpha=fade, zorder=5)
                dyn.append(sc)
                # a looped train is "held for"; a train queued behind one that
                # still occupies the block ahead is "waiting behind"
                verb = ("looped at" if e["cause"] == "precedence"
                        else "waiting behind a train at")
                why = f"{e['train']}  {verb} {e['station']}  -  "
                why += f"{e['minutes']:.0f} min lost"
                if e["blocker"]:
                    why += f"  ({e['blocker']} goes first)"
                xx = min(max(e["km"], 230), total_km - 230)
                txt = ax_strip.text(
                    xx, lane + (0.82 if e["up"] else -0.95), why,
                    ha="center", va="center", fontsize=11, color=INK,
                    zorder=7, alpha=max(0.0, 1 - age * 0.75),
                    bbox=dict(boxstyle="round,pad=0.55", fc=PANEL, ec=BAD,
                              lw=1.3, alpha=max(0.0, 1 - age * 0.75)))
                dyn.append(txt)

            # --- info panel: the running network state ---
            seen = [e for e in pb.events if e["t"] <= t]
            lost = sum(e["minutes"] for e in seen)
            late = [k_ for k_ in live]
            lines = [
                ("TRAINS RUNNING", f"{len(live)}"),
                ("CONFLICTS SO FAR", f"{len(seen)}"),
                ("MINUTES LOST TO CONFLICT", f"{lost:,.0f}"),
            ]
            y = 0.97
            for label, val in lines:
                dyn.append(ax_info.text(0.0, y, label, fontsize=10, color=MUTE,
                                        transform=ax_info.transAxes))
                dyn.append(ax_info.text(0.0, y - 0.155, val, fontsize=25,
                                        fontweight="bold", color=INK,
                                        transform=ax_info.transAxes))
                y -= 0.31
            if seen:
                last = seen[-1]
                dyn.append(ax_info.text(
                    0.0, 0.02,
                    f"last: {last['train']} at {last['station']}\n"
                    f"cause: {last['cause']}",
                    fontsize=10, color=BAD, transform=ax_info.transAxes,
                    va="bottom"))

            clock_txt.set_text(_hm(t))
            count_txt.set_text(f"day {day}")
            writer.grab_frame(facecolor=BG)
            if k % 60 == 0:
                print(f"    frame {k}/{frames}  {_hm(t)}  "
                      f"({_time.time() - started:.0f}s)", flush=True)
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{_time.time() - started:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/video/railcast_corridor.mp4")
    ap.add_argument("--seconds", type=float, default=24.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--bitrate", type=int, default=6000)
    ap.add_argument("--trains", type=int, default=30)
    ap.add_argument("--reference", default="12951")
    ap.add_argument("--start-date", default="2024-01-08")
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()
    if args.gif and args.out.endswith(".mp4"):
        args.out = args.out[:-4] + ".gif"
    build(args)


if __name__ == "__main__":
    main()

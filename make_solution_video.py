"""
RAILCAST - the same run, forecast.

    python make_solution_video.py            # 1080p MP4
    python make_solution_video.py --gif

Companion to make_animation.py. That one shows the problem: trains conflicting
and delay cascading. This one shows what the model does about it.

Every number on screen is real output from the evaluation run, read out of
outputs/data/tracked_journey.csv - one journey re-forecast every thirty minutes
from departure to arrival, on days the model never trained on. The forecast
steps rather than glides because that is what it does: a discrete re-forecast
each time new state arrives.
"""
from __future__ import annotations

import argparse
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, PillowWriter

from railcast.feeds import DatameetFeed
from railcast.feeds.corridor import build_corridor

BG = "#0B0E14"
PANEL = "#141924"
INK = "#F2F5FA"
MUTE = "#8B94A3"
GRID = "#232A38"
ACCENT = "#F2A03D"
BASE = "#8A94A6"
BAD = "#E2574C"
GOOD = "#4CAF7D"


def _hm(minute: float) -> str:
    m = int(minute) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def card(fig, x, y, w, h):
    ax = fig.add_axes((x, y, w, h))
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    return ax


def build(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    tid = str(df["train"].iloc[0])
    cor, _ = build_corridor(DatameetFeed(), "12951")
    km_of = {s.code: s.km for s in cor.stations}
    df = df[df["to_code"].isin(km_of)].copy()
    df["km_post"] = df["to_code"].map(km_of)
    # this train may run either direction along the corridor; plot distance
    # already run so the journey always reads left to right
    dest_km = km_of[df.loc[df["hops"].idxmax(), "to_code"]]
    origin_km = df["km_post"].iloc[(df["km_post"] - dest_km).abs().idxmax()]         if False else float(max(df["km_post"], key=lambda k: abs(k - dest_km)))
    df["km"] = (df["km_post"] - origin_km).abs()
    df["late_fc"] = df["railcast_arr"] - df["sched_arr"]
    df["late_lo"] = df["lo_arr"] - df["sched_arr"]
    df["late_hi"] = df["hi_arr"] - df["sched_arr"]
    df["late_cf"] = df["baseline"] - df["sched_arr"]
    df["late_act"] = df["act_arr"] - df["sched_arr"]

    snaps = sorted(df["now"].unique())
    dest_code = df.loc[df["hops"].idxmax(), "to_code"]
    dest = df[df["to_code"] == dest_code].sort_values("now")
    dest_act = float(dest["act_arr"].iloc[0])
    dest_sched = float(dest["sched_arr"].iloc[0])
    lo_km, hi_km = df["km"].min(), df["km"].max()
    y_lo = min(df["late_lo"].min(), df["late_act"].min()) - 12
    y_hi = max(df["late_hi"].max(), df["late_act"].max()) + 12

    # ---------------- figure ---------------- #
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0.05, 0.30, 0.60, 0.50))
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color=GRID, lw=0.5)
    ax.set_xlim(lo_km - 20, hi_km + 20)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.set_xlabel("km run since departure", color=MUTE, fontsize=10)
    ax.set_ylabel("minutes late against the timetable", color=MUTE, fontsize=10)
    ax.axhline(0, color=MUTE, lw=0.9, ls=":")

    ax_w = fig.add_axes((0.05, 0.09, 0.60, 0.145))
    ax_w.set_facecolor(BG)
    for sp in ax_w.spines.values():
        sp.set_visible(False)
    ax_w.grid(color=GRID, lw=0.5)
    ax_w.set_xlim(snaps[0], snaps[-1])
    ax_w.set_ylim(0, (dest["late_hi"] - dest["late_lo"]).max() * 1.15)
    ax_w.tick_params(colors=MUTE, labelsize=8)
    ax_w.set_ylabel("window (min)", color=MUTE, fontsize=9)
    ticks = np.linspace(snaps[0], snaps[-1], 7)
    ax_w.set_xticks(ticks)
    ax_w.set_xticklabels([_hm(x) for x in ticks])
    ax_w.set_title("width of the 80% window for the destination",
                   fontsize=10, color=MUTE, loc="left")

    c_eta = card(fig, 0.69, 0.55, 0.27, 0.25)
    c_err = card(fig, 0.69, 0.30, 0.27, 0.22)
    c_why = card(fig, 0.69, 0.09, 0.27, 0.18)

    fig.text(0.05, 0.945, "RAILCAST", fontsize=30, fontweight="bold", color=INK)
    fig.text(0.215, 0.951, f"the same run, forecast   -   train {tid}",
             fontsize=13, color=MUTE)
    fig.text(0.05, 0.905,
             f"re-forecast every 30 minutes, destination {dest_code}   -   "
             "held-out day the model never trained on", fontsize=11, color=MUTE)
    fig.legend(handles=[
        plt.Line2D([], [], color=ACCENT, lw=2.8, label="RAILCAST forecast"),
        plt.Line2D([], [], color=ACCENT, lw=8, alpha=0.3,
                   label="80% arrival window"),
        plt.Line2D([], [], color=BASE, lw=1.8, ls="--",
                   label="carry-forward (today)"),
        plt.Line2D([], [], color=INK, lw=0, marker="o", ms=6,
                   label="what actually happened")],
        frameon=False, ncol=4, fontsize=10.5, labelcolor=MUTE,
        loc="upper left", bbox_to_anchor=(0.045, 0.875))
    clock = fig.text(0.96, 0.938, "", fontsize=28, fontweight="bold",
                     color=ACCENT, ha="right", family="monospace")
    fig.text(0.05, 0.018,
             "Carry-forward repeats the delay it can already see. RAILCAST "
             "forecasts where that delay is going, and states how sure it is.",
             fontsize=9.5, color=MUTE)

    writer = (PillowWriter(fps=args.fps) if args.gif
              else FFMpegWriter(fps=args.fps, bitrate=args.bitrate))
    hold = max(1, int(round(args.seconds * args.fps / len(snaps))))
    started = _time.time()

    with writer.saving(fig, str(out), args.dpi):
        dyn = []
        for si, now in enumerate(snaps):
            g = df[df["now"] == now].sort_values("km")
            if g.empty:
                continue
            seen = df[(df["act_arr"] <= now)].drop_duplicates("to_code")
            drow = g[g["to_code"] == dest_code]
            wlog = dest[dest["now"] <= now]

            for _ in range(hold):
                for a in dyn:
                    a.remove()
                dyn = []

                band = ax.fill_between(g["km"], g["late_lo"], g["late_hi"],
                                       color=ACCENT, alpha=0.22, zorder=2)
                dyn.append(band)
                dyn.append(ax.plot(g["km"], g["late_fc"], color=ACCENT, lw=2.8,
                                   zorder=4)[0])
                dyn.append(ax.plot(g["km"], g["late_cf"], color=BASE, lw=1.8,
                                   ls="--", zorder=3)[0])
                if len(seen):
                    dyn.append(ax.plot(seen["km"], seen["late_act"], lw=0,
                                       marker="o", ms=6, color=INK, zorder=5)[0])
                pos = float(g.loc[g["hops"].idxmin(), "km"])
                dyn.append(ax.axvline(pos, color=GOOD, lw=1.6, alpha=0.9,
                                      zorder=6))
                dyn.append(ax.text(pos, y_hi - 4, "  train is here", color=GOOD,
                                   fontsize=10.5, va="top", fontweight="bold"))

                if len(wlog):
                    dyn.append(ax_w.plot(wlog["now"],
                                         wlog["late_hi"] - wlog["late_lo"],
                                         color=ACCENT, lw=2.6)[0])

                # --- cards ---
                if len(drow):
                    r = drow.iloc[0]
                    w = float(r["late_hi"] - r["late_lo"])
                    e_rc = abs(float(r["railcast_arr"]) - dest_act)
                    e_cf = abs(float(r["baseline"]) - dest_act)
                    dyn += _eta_card(c_eta, dest_code, dest_sched,
                                     float(r["railcast_arr"]), float(r["lo_arr"]),
                                     float(r["hi_arr"]), w)
                    m_cf = float((wlog["baseline"] - dest_act).abs().mean())
                    m_rc = float((wlog["railcast_arr"] - dest_act).abs().mean())
                    dyn += _err_card(c_err, e_cf, e_rc, m_cf, m_rc)
                    dyn += _why_card(c_why, float(r["cur_delay"])
                                     if "cur_delay" in r else float(g["late_cf"].iloc[0]),
                                     len(seen), len(g))
                clock.set_text(_hm(now))
                writer.grab_frame(facecolor=BG)
            if si % 10 == 0:
                print(f"    snapshot {si}/{len(snaps)}  {_hm(now)}  "
                      f"({_time.time() - started:.0f}s)", flush=True)
    plt.close(fig)
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, "
          f"{_time.time() - started:.0f}s)")


def _eta_card(ax, code, sched, eta, lo, hi, w):
    o = []
    o.append(ax.text(0.06, 0.86, f"ARRIVAL AT {code}", fontsize=10, color=MUTE,
                     transform=ax.transAxes))
    o.append(ax.text(0.06, 0.60, _hm(eta), fontsize=40, fontweight="bold",
                     color=ACCENT, transform=ax.transAxes))
    o.append(ax.text(0.06, 0.44, f"booked {_hm(sched)}   -   "
                                 f"{eta - sched:+.0f} min", fontsize=11,
                     color=MUTE, transform=ax.transAxes))
    o.append(ax.text(0.06, 0.22, f"80%   {_hm(lo)} - {_hm(hi)}", fontsize=15,
                     color=INK, transform=ax.transAxes))
    o.append(ax.text(0.06, 0.07, f"window {w:.0f} min wide", fontsize=11,
                     color=MUTE, transform=ax.transAxes))
    return o


def _err_card(ax, e_cf, e_rc, m_cf, m_rc):
    """Colour by which method is actually closer right now.

    Near the destination carry-forward is hard to beat - repeating a delay you
    can already see is a good forecast when there is no run left for it to
    change. Fixing the colours would have shown green on the worse number.
    """
    o = []
    o.append(ax.text(0.05, 0.84, "ERROR AT THIS MOMENT", fontsize=9.5,
                     color=MUTE, transform=ax.transAxes))
    win_cf = e_cf <= e_rc
    o.append(ax.text(0.05, 0.55, f"{e_cf:.0f}", fontsize=30, fontweight="bold",
                     color=GOOD if win_cf else BAD, transform=ax.transAxes))
    o.append(ax.text(0.22, 0.585, "min  carry-forward", fontsize=10.5,
                     color=MUTE, transform=ax.transAxes))
    o.append(ax.text(0.05, 0.30, f"{e_rc:.0f}", fontsize=30, fontweight="bold",
                     color=BAD if win_cf else GOOD, transform=ax.transAxes))
    o.append(ax.text(0.22, 0.335, "min  RAILCAST", fontsize=10.5, color=MUTE,
                     transform=ax.transAxes))
    o.append(ax.text(0.05, 0.07, f"average so far:   carry-forward {m_cf:.0f} min"
                                 f"    RAILCAST {m_rc:.0f} min",
                     fontsize=9, color=INK, transform=ax.transAxes))
    return o


def _why_card(ax, delay, seen, remaining):
    o = []
    o.append(ax.text(0.06, 0.78, "THE LOOP", fontsize=10, color=MUTE,
                     transform=ax.transAxes))
    o.append(ax.text(0.06, 0.50, f"{seen} actual arrivals in",
                     fontsize=13, color=INK, transform=ax.transAxes))
    o.append(ax.text(0.06, 0.28, f"{remaining} stations still to forecast",
                     fontsize=13, color=INK, transform=ax.transAxes))
    o.append(ax.text(0.06, 0.07, "every arrival is a free label",
                     fontsize=10.5, color=ACCENT, transform=ax.transAxes))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/data/tracked_journey.csv")
    ap.add_argument("--out", default="outputs/video/railcast_forecast.mp4")
    ap.add_argument("--seconds", type=float, default=22.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--bitrate", type=int, default=6000)
    ap.add_argument("--gif", action="store_true")
    a = ap.parse_args()
    if a.gif and a.out.endswith(".mp4"):
        a.out = a.out[:-4] + ".gif"
    build(a)


if __name__ == "__main__":
    main()

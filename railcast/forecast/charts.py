"""
The three evaluation charts that matter for the trip planner.

    01  accuracy by horizon      is the forecast better than repeating today's delay?
    02  connection risk          when we say 90%, do 90% of connections get made?
    03  one journey forecast     what a forecast looks like for one train

Each chart carries a footer stating where the data came from. Until real running
data is collected, the delays it is scored on are simulated, and the footer says so.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .models import HORIZON_LABELS

OUT = Path("outputs/charts")
FOOT = "Simulated corridor - prototype result, not an operational measurement"

THEME = dict(bg="#FFFFFF", panel="#F4F5F7", ink="#12161F", mute="#5B6472",
             grid="#D9DDE4")
ACCENT = "#F2A03D"      # RAILCAST
BASE = "#8A94A6"        # carry-forward baseline
SIM = "#3DA5A0"         # simulator only

_charts: list = []


def chart(fn):
    _charts.append(fn)
    return fn


def _style():
    t = THEME
    plt.rcParams.update({
        "figure.facecolor": t["bg"], "axes.facecolor": t["bg"],
        "savefig.facecolor": t["bg"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["grid"],
        "xtick.color": t["mute"], "ytick.color": t["mute"],
        "grid.color": t["grid"], "font.size": 11,
        "font.family": "DejaVu Sans", "axes.titlesize": 14,
        "axes.titleweight": "bold", "figure.dpi": 120,
    })
    return t


def _finish(fig, ax, t, title, subtitle=None):
    axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]
    for a in np.ravel(axes):
        a.set_axisbelow(True)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a.spines[sp].set_color(t["grid"])
    fig.suptitle(title, x=0.012, y=0.975, ha="left", fontsize=15,
                 fontweight="bold", color=t["ink"])
    wrapped = ""
    if subtitle:
        wrapped = textwrap.fill(subtitle, width=int(fig.get_figwidth() * 10.2))
        fig.text(0.012, 0.915, wrapped, ha="left", va="top", fontsize=10.5,
                 color=t["mute"], linespacing=1.35)
    fig.text(0.012, 0.018, FOOT, ha="left", fontsize=8, color=t["mute"])
    nl = wrapped.count(chr(10)) + 1 if subtitle else 0
    fig.tight_layout(rect=(0, 0.035, 1, 0.945 - 0.040 * nl))


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def set_footer(summary) -> None:
    """State the provenance of the run on every chart."""
    global FOOT
    src = (summary or {}).get("data_source", {})
    if src.get("name") == "real":
        FOOT = ("Real Indian Railways working timetable and stations (datameet, CC0) "
                "with observed ERA5 weather - delay generation is still modelled, "
                "not measured")
    else:
        FOOT = ("Hand-built New Delhi - Mumbai Central corridor - prototype result, "
                "not an operational measurement")


def _hm(x):
    """Minutes from midnight -> HH:MM clock label."""
    x = float(x) % 1440
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


# --------------------------------------------------------------------------- #
# 01 - accuracy against today's practice, by how far ahead
# --------------------------------------------------------------------------- #
@chart
def c01_accuracy_by_horizon(ctx):
    t = _style()
    tab = ctx["tables"]["by_horizon"].set_index("horizon").reindex(HORIZON_LABELS).dropna()
    x = np.arange(len(tab))
    w = 0.26
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(x - w, tab["baseline_mae"], w, label="Carry-forward (today)", color=BASE)
    ax.bar(x, tab["simulator_mae"], w, label="Conflict simulator only", color=SIM)
    ax.bar(x + w, tab["railcast_mae"], w, label="RAILCAST (full stack)", color=ACCENT)
    for i, r in enumerate(tab.itertuples()):
        ax.text(i + w, r.railcast_mae + 0.9, f"-{r.improvement_pct:.0f}%",
                ha="center", fontsize=10, fontweight="bold", color=ACCENT)
        ax.text(i - w, r.baseline_mae + 0.9, f"{r.baseline_mae:.0f}",
                ha="center", fontsize=9, color=t["mute"])
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, fontsize=10)
    ax.set_ylabel("Mean absolute ETA error (minutes)")
    ax.grid(axis="y", lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=10)
    ax.set_ylim(0, tab["baseline_mae"].max() * 1.28)
    _finish(fig, ax, t, "Error by forecast horizon, not one pooled number",
            "Carry-forward is adequate for the next station. It collapses beyond "
            "a few hours - which is the horizon passengers actually plan against.")
    _save(fig, "01_accuracy_by_horizon")


# --------------------------------------------------------------------------- #
# 02 - the connection probability, and whether it can be trusted
# --------------------------------------------------------------------------- #
@chart
def c02_connection_risk(ctx):
    t = _style()
    cr = ctx["connection"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax.plot(cr["buffer"], cr["p_make"], color=ACCENT, lw=3, marker="o", ms=5,
            label="RAILCAST stated probability")
    ax.fill_between(cr["buffer"], 0, cr["p_make"], color=ACCENT, alpha=0.15)
    ax.plot(cr["buffer"], cr["observed"], color=t["ink"], lw=0, marker="D", ms=5.5,
            label="actually made it")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    ax.set_xlabel("Booked connection buffer (minutes)")
    ax.set_ylabel("Probability of making the connection")
    ax.set_ylim(0, 1)
    ax.grid(lw=0.6)
    ax.set_title("Connection-make probability", fontsize=11, color=t["mute"], loc="left")

    ax2.plot([0, 1], [0, 1], color=t["mute"], lw=1.2, ls="--")
    ax2.plot(cr["pred_bin"], cr["obs_bin"], color=SIM, lw=2.6, marker="o", ms=6)
    ax2.set_xlabel("Probability RAILCAST stated")
    ax2.set_ylabel("Fraction that actually made it")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(lw=0.6)
    ax2.set_title("Is that probability honest?", fontsize=11, color=t["mute"], loc="left")
    _finish(fig, [ax, ax2], t, "The number behind Safe / Likely / Risky",
            "Left: the make-probability a passenger is shown, against booked buffer. "
            "Right: stated probability against realised outcome on held-out days - "
            f"reliability error {cr['ece']:.3f}.")
    _save(fig, "02_connection_risk")


# --------------------------------------------------------------------------- #
# 03 - one forecast, every remaining station
# --------------------------------------------------------------------------- #
def _hero_rows(ctx):
    """One re-forecast of the tracked journey: the first made while the train
    is running late with most of its run still ahead."""
    tr = ctx["track"]
    nows = sorted(tr["now"].unique())
    now = nows[max(1, int(len(nows) * 0.35))]
    for n in nows:
        snap = tr[tr["now"] == n]
        if snap["cur_delay"].iloc[0] >= 20 and len(snap) >= 8:
            now = n
            break
    r = tr[(tr["now"] == now) & (tr["hops"] >= 1)].sort_values("hops")
    return r, int(r["day"].iloc[0]), float(now)


@chart
def c03_journey_forecast(ctx):
    t = _style()
    r, day, now = _hero_rows(ctx)
    tid = r["train"].iloc[0]
    name = ctx["world"].sim.trains[tid].name
    km = r["dist_remaining_km"].to_numpy()       # along the direction of travel
    base = r["sched_arr"].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.fill_between(km, r["lo_arr"] - base, r["hi_arr"] - base, color=ACCENT,
                    alpha=0.22, label="80% arrival window")
    ax.plot(km, r["railcast_arr"] - base, color=ACCENT, lw=2.6, label="RAILCAST forecast")
    ax.plot(km, r["baseline"] - base, color=BASE, lw=1.8, ls="--",
            label="Carry-forward (today)")
    ax.plot(km, r["act_arr"] - base, color=t["ink"], lw=0, marker="o", ms=5.5,
            label="What actually happened")
    ax.axhline(0, color=t["mute"], lw=0.9, ls=":")
    for _, row in r.iterrows():
        if row["hops"] % 3 == 1:
            ax.annotate(row["to_code"], (row["dist_remaining_km"],
                                         row["act_arr"] - row["sched_arr"]),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8, color=t["mute"])
    ax.set_xlabel(f"Kilometres ahead of the train (forecast made at {r['from_code'].iloc[0]})")
    ax.set_ylabel("Minutes late against the timetable")
    ax.grid(lw=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=10)
    _finish(fig, ax, t, f"One forecast, made at {_hm(now)} - train {tid}, {name}",
            "Every remaining station gets an interval. Carry-forward repeats the "
            "delay it can see; RAILCAST forecasts where that delay is going.")
    _save(fig, "03_journey_forecast")


# --------------------------------------------------------------------------- #
def render_all(ctx) -> int:
    set_footer(ctx.get("summary"))
    n = 0
    for fn in _charts:
        try:
            fn(ctx)
            n += 1
        except Exception as exc:                         # keep the rest rendering
            print(f"  ! {fn.__name__} failed: {exc}")
    return n

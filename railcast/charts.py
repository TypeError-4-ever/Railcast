"""
Every figure the deck needs, rendered twice: light for print, dark for slides.

Each chart carries a one-line caption saying exactly what it is. The corridor is
simulated, so the captions say so - a judge who spots that for themselves after
you claimed otherwise is a slide you lose.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from .corridor import CORRIDOR
from .models import HORIZON_LABELS

OUT = Path("outputs/charts")
FOOT = "Simulated New Delhi - Mumbai Central corridor - prototype result, not an operational measurement"

THEMES = {
    "light": dict(bg="#FFFFFF", panel="#F4F5F7", ink="#12161F", mute="#5B6472",
                  grid="#D9DDE4"),
    "dark":  dict(bg="#0B0E14", panel="#141924", ink="#F2F5FA", mute="#8B94A3",
                  grid="#232A38"),
}
ACCENT = "#F2A03D"      # RAILCAST
BASE = "#8A94A6"        # carry-forward baseline
SIM = "#3DA5A0"         # simulator only
GOOD = "#4CAF7D"
BAD = "#E2574C"
CLASS_COLOR = {"RAJDHANI": "#F2A03D", "DURONTO": "#E2574C", "SHATABDI": "#C77DFF",
               "SUPERFAST": "#4D8FE8", "EXPRESS": "#3DA5A0", "PASSENGER": "#8A94A6"}

_charts: list = []


def chart(fn):
    _charts.append(fn)
    return fn


def _style(theme: str):
    t = THEMES[theme]
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


def _finish(fig, ax, t, title, subtitle=None, foot=FOOT):
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
    if foot:
        fig.text(0.012, 0.018, foot, ha="left", fontsize=8, color=t["mute"])
    nl = wrapped.count(chr(10)) + 1 if subtitle else 0
    fig.tight_layout(rect=(0, 0.035, 1, 0.945 - 0.040 * nl))


def _save(fig, name, theme):
    d = OUT / theme
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png", dpi=200)
    plt.close(fig)


def _hm(x):
    """Minutes from midnight -> HH:MM clock label."""
    x = float(x) % 1440
    return f"{int(x // 60):02d}:{int(x % 60):02d}"


# --------------------------------------------------------------------------- #
# 01 - the headline accuracy chart
# --------------------------------------------------------------------------- #
@chart
def c01_mae_by_horizon(ctx, theme):
    t = _style(theme)
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
    _save(fig, "01_mae_by_horizon", theme)


# --------------------------------------------------------------------------- #
# 02 - error growth with lead time
# --------------------------------------------------------------------------- #
@chart
def c02_error_vs_lead(ctx, theme):
    t = _style(theme)
    d = ctx["scored"].copy()
    d["lead_h"] = d["sched_lead"] / 60.0
    d = d[(d["lead_h"] >= 0) & (d["lead_h"] <= 42)]
    bins = np.arange(0, 43, 2)
    d["b"] = pd.cut(d["lead_h"], bins, labels=bins[:-1] + 1)
    g = d.groupby("b", observed=True)
    mid = g.apply(lambda x: pd.Series({
        "base": np.mean(np.abs(x["err_base"])),
        "rc": np.mean(np.abs(x["err_rc"])),
        "rc90": np.percentile(np.abs(x["err_rc"]), 90),
        "base90": np.percentile(np.abs(x["err_base"]), 90),
    }), include_groups=False)
    xs = mid.index.astype(float)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.fill_between(xs, 0, mid["base90"], color=BASE, alpha=0.16)
    ax.fill_between(xs, 0, mid["rc90"], color=ACCENT, alpha=0.18)
    ax.plot(xs, mid["base"], color=BASE, lw=2.6, label="Carry-forward, MAE")
    ax.plot(xs, mid["base90"], color=BASE, lw=1.2, ls="--", label="Carry-forward, P90")
    ax.plot(xs, mid["rc"], color=ACCENT, lw=2.8, label="RAILCAST, MAE")
    ax.plot(xs, mid["rc90"], color=ACCENT, lw=1.2, ls="--", label="RAILCAST, P90")
    ax.set_xlabel("Forecast lead time (hours ahead)")
    ax.set_ylabel("Absolute ETA error (minutes)")
    ax.grid(lw=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=10)
    ax.set_xlim(0, 42)
    _finish(fig, ax, t, "Error against how far ahead the forecast is made",
            "The gap opens from about three hours out and never closes again. "
            "Shaded bands are the 90th percentile of absolute error.")
    _save(fig, "02_error_vs_lead", theme)


# --------------------------------------------------------------------------- #
# 03 - the coverage audit
# --------------------------------------------------------------------------- #
@chart
def c03_coverage_heatmap(ctx, theme):
    t = _style(theme)
    tab = ctx["tables"]["by_zone_horizon"]
    piv = tab.pivot(index="zone_name", columns="horizon", values="coverage_pct")
    piv = piv.reindex(index=["NR", "NCR", "WCR", "WR"],
                      columns=[h for h in HORIZON_LABELS if h in piv.columns])
    cmap = LinearSegmentedColormap.from_list(
        "cov", [BAD, "#E9B949", GOOD, "#E9B949", BAD])
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    im = ax.imshow(piv.values, cmap=cmap, vmin=70, vmax=90, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color="#0B0E14", fontsize=11, fontweight="bold")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, fontsize=9.5)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=11)
    ax.set_xticks(np.arange(-.5, piv.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, piv.shape[0], 1), minor=True)
    ax.grid(which="minor", color=t["bg"], lw=3)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Achieved coverage of the nominal 80% window")
    cb.outline.set_visible(False)
    _finish(fig, ax, t, "Coverage audited per zone and per horizon",
            "Mondrian conformal calibration. Green is on target. A single pooled "
            "coverage number would hide every cell that is not.")
    _save(fig, "03_coverage_heatmap", theme)


# --------------------------------------------------------------------------- #
# 04 - coverage holds across the year
# --------------------------------------------------------------------------- #
@chart
def c04_coverage_by_month(ctx, theme):
    t = _style(theme)
    tab = ctx["tables"]["by_month"]
    order = [m for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                         "Sep", "Oct", "Nov", "Dec"] if m in set(tab["month_name"])]
    tab = tab.set_index("month_name").reindex(order)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), width_ratios=[1.25, 1])
    cols = [GOOD if abs(v - 80) <= 3 else "#E9B949" if abs(v - 80) <= 6 else BAD
            for v in tab["coverage_pct"]]
    ax.bar(tab.index, tab["coverage_pct"], color=cols, width=0.62)
    ax.axhline(80, color=t["ink"], lw=1.4, ls="--")
    ax.text(len(tab) - 0.4, 80.7, "nominal 80%", ha="right", fontsize=9, color=t["ink"])
    ax.set_ylim(60, 95)
    ax.set_ylabel("Achieved coverage (%)")
    ax.grid(axis="y", lw=0.6)

    ax2.bar(tab.index, tab["width_min"], color=ACCENT, width=0.62)
    ax2.set_ylabel("Mean window width (minutes)")
    ax2.grid(axis="y", lw=0.6)
    ax2.set_title("Window width", fontsize=11, color=t["mute"], loc="left")
    ax.set_title("Coverage by month of the test period", fontsize=11,
                 color=t["mute"], loc="left")
    _finish(fig, [ax, ax2], t, "The window is honest in every month, not on average",
            "Coverage stays on target while the interval widens in the harder "
            "months - the model pays for difficulty with width, not with a broken promise.")
    _save(fig, "04_coverage_by_month", theme)


# --------------------------------------------------------------------------- #
# 05 - the window widens with distance
# --------------------------------------------------------------------------- #
@chart
def c05_window_vs_lead(ctx, theme):
    t = _style(theme)
    d = ctx["scored"].copy()
    d["lead_h"] = d["sched_lead"] / 60.0
    d = d[(d["lead_h"] >= 0) & (d["lead_h"] <= 42)]
    bins = np.arange(0, 43, 2)
    d["b"] = pd.cut(d["lead_h"], bins, labels=bins[:-1] + 1)
    g = d.groupby("b", observed=True).apply(lambda x: pd.Series({
        "w": np.mean(x["hi_arr"] - x["lo_arr"]),
        "w25": np.percentile(x["hi_arr"] - x["lo_arr"], 25),
        "w75": np.percentile(x["hi_arr"] - x["lo_arr"], 75),
        "cov": 100 * x["covered"].mean()}), include_groups=False)
    xs = g.index.astype(float)
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.fill_between(xs, g["w25"], g["w75"], color=ACCENT, alpha=0.22,
                    label="interquartile range")
    ax.plot(xs, g["w"], color=ACCENT, lw=3, label="mean 80% window width")
    ax.set_xlabel("Forecast lead time (hours ahead)")
    ax.set_ylabel("Width of the 80% arrival window (minutes)")
    ax.grid(lw=0.6)
    ax2 = ax.twinx()
    ax2.plot(xs, g["cov"], color=SIM, lw=1.6, ls="--", label="achieved coverage")
    ax2.axhline(80, color=t["mute"], lw=0.9, ls=":")
    ax2.set_ylim(50, 100)
    ax2.set_ylabel("Achieved coverage (%)", color=SIM)
    ax2.tick_params(axis="y", colors=SIM)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=10)
    _finish(fig, ax, t, "The window widens with distance, and stays calibrated",
            "Uncertainty is priced into the width. Coverage does not drift with "
            "horizon, because it is calibrated per horizon bucket.")
    _save(fig, "05_window_vs_lead", theme)


# --------------------------------------------------------------------------- #
# 06 - hero: one journey, one forecast
# --------------------------------------------------------------------------- #
def _hero_rows(ctx):
    """One re-forecast, taken part-way through the tracked journey."""
    tr = ctx["track"]
    nows = sorted(tr["now"].unique())
    now = nows[max(1, int(len(nows) * 0.35))]
    r = tr[(tr["now"] == now) & (tr["hops"] >= 1)].sort_values("hops")
    return r, int(r["day"].iloc[0]), float(now)


@chart
def c06_journey(ctx, theme):
    t = _style(theme)
    r, day, now = _hero_rows(ctx)
    tid = r["train"].iloc[0]
    name = ctx["world"].sim.trains[tid].name
    km = np.array([CORRIDOR.stations[j].km for j in r["to_idx"]])
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
            ax.annotate(row["to_code"], (CORRIDOR.stations[row["to_idx"]].km,
                                         row["act_arr"] - row["sched_arr"]),
                        textcoords="offset points", xytext=(0, 9), ha="center",
                        fontsize=8, color=t["mute"])
    ax.set_xlabel("Kilometres from New Delhi")
    ax.set_ylabel("Minutes late against the timetable")
    ax.grid(lw=0.6)
    ax.legend(frameon=False, ncol=4, loc="upper left", fontsize=10)
    _finish(fig, ax, t, f"One forecast, made at {_hm(now)} - train {tid}, {name}",
            "Every remaining station gets an interval. Carry-forward repeats the "
            "delay it can see; RAILCAST forecasts where that delay is going.")
    _save(fig, "06_journey_forecast", theme)


# --------------------------------------------------------------------------- #
# 07 - the window closing as the train runs
# --------------------------------------------------------------------------- #
@chart
def c07_funnel(ctx, theme):
    t = _style(theme)
    tr = ctx["track"]
    tid = tr["train"].iloc[0]
    dest = ctx["world"].sim.trains[tid].dest
    g = tr[tr["to_idx"] == dest].sort_values("now")
    name = ctx["world"].sim.trains[tid].name
    actual = g["act_arr"].iloc[0]
    x = g["now"].to_numpy() / 60.0
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.fill_between(x, g["lo_arr"] - actual, g["hi_arr"] - actual, color=ACCENT,
                    alpha=0.24, label="80% window for the destination")
    ax.plot(x, g["railcast_arr"] - actual, color=ACCENT, lw=2.6,
            label="RAILCAST point forecast")
    ax.plot(x, g["baseline"] - actual, color=BASE, lw=1.8, ls="--",
            label="Carry-forward")
    ax.axhline(0, color=t["ink"], lw=1.6)
    ax.text(x[-1], 2.0, "actual arrival", ha="right", fontsize=9.5, color=t["ink"])
    w = (g["hi_arr"] - g["lo_arr"]).to_numpy()
    step = max(1, len(x) // 8)
    for i in range(0, len(x), step):
        ax.annotate(f"{w[i]:.0f} min", (x[i], (g["hi_arr"] - actual).iloc[i]),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8.5, color=ACCENT)
    ax.set_xlabel("Clock time at which the forecast was made (hours)")
    ax.set_ylabel("Forecast error against the actual arrival (minutes)")
    ax.grid(lw=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=10)
    _finish(fig, ax, t,
            f"The window narrows as the train runs - train {tid}, {name}",
            f"The destination arrival re-forecast every {int(ctx['track_step'])} "
            "minutes for the whole run. Each actual arrival along the way is a free "
            f"label: the window closes from {w[0]:.0f} to {w[-1]:.0f} minutes.")
    _save(fig, "07_window_closing", theme)


# --------------------------------------------------------------------------- #
# 08 - the cascade the simulator can see
# --------------------------------------------------------------------------- #
@chart
def c08_marey(ctx, theme):
    t = _style(theme)
    day, tid, stn, mins, blocker = ctx["cascade"]
    env, truth = ctx["world"].days[day]
    sim = ctx["world"].sim
    fig, ax = plt.subplots(figsize=(12, 6.4))
    for otid, run in truth.items():
        tr = sim.trains[otid]
        pts = [(run.arr[s], CORRIDOR.stations[s].km) for s in tr.path if s in run.arr]
        if len(pts) < 3:
            continue
        xs, ys = zip(*pts)
        hot = otid in (tid, blocker)
        ax.plot(np.array(xs) / 60.0, ys, lw=2.6 if hot else 0.8,
                color=CLASS_COLOR[tr.klass] if hot else t["grid"],
                alpha=1.0 if hot else 0.85, zorder=3 if hot else 1)
        if hot:
            ax.annotate(f"{otid}", (xs[len(xs) // 2] / 60.0, ys[len(ys) // 2]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9.5,
                        fontweight="bold", color=CLASS_COLOR[tr.klass])
    hk = CORRIDOR.stations[stn].km
    ht = truth[tid].arr[stn] / 60.0
    ax.plot([ht], [hk], marker="o", ms=13, mfc="none", mec=BAD, mew=2.4, zorder=5)
    ax.annotate(f"{tid} held {mins:.0f} min at {CORRIDOR.code(stn)}\nfor {blocker}",
                (ht, hk), textcoords="offset points", xytext=(14, -34), fontsize=10,
                color=BAD, fontweight="bold")
    ax.set_yticks([CORRIDOR.stations[i].km for i in range(0, 35, 4)])
    ax.set_yticklabels([CORRIDOR.code(i) for i in range(0, 35, 4)], fontsize=9)
    ax.set_xlabel("Hours from midnight")
    ax.set_ylabel("Station along the corridor")
    ax.grid(lw=0.5, alpha=0.6)
    ax.legend(handles=[Patch(color=CLASS_COLOR[k], label=k.title())
                       for k in ("RAJDHANI", "SUPERFAST", "EXPRESS", "PASSENGER")],
              frameon=False, ncol=4, fontsize=9.5, loc="upper left")
    _finish(fig, ax, t, "The dependency no deployed ETA model represents",
            "Time-distance diagram for one simulated day. One train looped for a "
            "faster one behind it - the hold is a consequence of the block graph, "
            "so the forecast can name its cause.")
    _save(fig, "08_cascade_marey", theme)


# --------------------------------------------------------------------------- #
# 09 - what the precedence decision costs the network
# --------------------------------------------------------------------------- #
@chart
def c09_precedence_cost(ctx, theme):
    t = _style(theme)
    pc = ctx["precedence"]
    d, se = pc["delta_by"], pc["delta_se"]
    keys = ["held", "blocker", "others", "total"]
    labels = [f"{pc['b']}\n(now goes first)", f"{pc['a']}\n(now gives way)",
              "Every other train\non the corridor", "Net effect\non the network"]
    vals = [d[k] for k in keys]
    errs = [se[k] for k in keys]
    cols = [GOOD if v < 0 else BAD for v in vals]
    cols[-1] = ACCENT
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    bars = ax.bar(labels, vals, color=cols, width=0.5, yerr=errs, capsize=6,
                  ecolor=t["mute"])
    ax.axhline(0, color=t["ink"], lw=1.3)
    for bar, v, e in zip(bars, vals, errs):
        off = (e + 3) * (1 if v >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2, v + off, f"{v:+.0f} min",
                ha="center", va="bottom" if v >= 0 else "top",
                fontweight="bold", fontsize=12, color=t["ink"])
    pad = max(abs(min(vals)), abs(max(vals))) * 0.45
    ax.set_ylim(min(vals) - pad, max(vals) + pad)
    ax.set_ylabel("Change in delay if the other train goes first (minutes)")
    ax.grid(axis="y", lw=0.6)
    _finish(fig, ax, t, "Which train goes first, and what the other choice costs",
            f"One real conflict, re-run {pc['runs']} times each way on identical "
            f"days. Reversing the call saves {pc['b']} {abs(d['held']):.0f} minutes "
            f"and costs the corridor {d['total']:+.0f} on net - the number a section "
            "controller never gets today.")
    _save(fig, "09_precedence_cost", theme)


# --------------------------------------------------------------------------- #
# 10 - connection risk, and whether the probability is honest
# --------------------------------------------------------------------------- #
@chart
def c10_connection_risk(ctx, theme):
    t = _style(theme)
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
    _finish(fig, [ax, ax2], t, "Turning the window into a decision",
            "Left: the make-probability a passenger is shown, against booked buffer. "
            "Right: stated probability against realised outcome on held-out days - "
            f"reliability error {cr['ece']:.3f}.")
    _save(fig, "10_connection_risk", theme)


# --------------------------------------------------------------------------- #
# 11 - why calibration is a stage of its own
# --------------------------------------------------------------------------- #
@chart
def c11_calibration_effect(ctx, theme):
    t = _style(theme)
    ce = ctx["calibration"]
    x = np.arange(len(ce["groups"]))
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    ax.bar(x - 0.2, ce["raw"], 0.4, color=BASE, label="Quantile model, uncalibrated")
    ax.bar(x + 0.2, ce["cal"], 0.4, color=ACCENT, label="After Mondrian conformal")
    ax.axhline(80, color=t["ink"], lw=1.4, ls="--")
    ax.text(len(x) - 0.4, 81, "nominal 80%", ha="right", fontsize=9, color=t["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(ce["groups"], fontsize=9)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Achieved coverage (%)")
    ax.grid(axis="y", lw=0.6)
    ax.legend(frameon=False, ncol=2, fontsize=10, loc="lower left")
    _finish(fig, ax, t, "Why calibration is a stage and not a footnote",
            "A quantile model alone under-covers where the data is hardest. "
            "Conformal calibration pulls every group back onto the nominal level.")
    _save(fig, "11_calibration_effect", theme)


# --------------------------------------------------------------------------- #
# 12 - what the model is actually using
# --------------------------------------------------------------------------- #
@chart
def c12_importance(ctx, theme):
    from .dataset import PRETTY
    t = _style(theme)
    imp = ctx["importance"].head(14)[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.0))
    ax.barh([PRETTY.get(i, i) for i in imp.index], imp.values, color=ACCENT, height=0.66)
    for y, v in enumerate(imp.values):
        ax.text(v + 0.4, y, f"{v:.1f}%", va="center", fontsize=9, color=t["mute"])
    ax.set_xlabel("Share of total split gain (%)")
    ax.grid(axis="x", lw=0.6)
    ax.set_xlim(0, imp.values.max() * 1.15)
    _finish(fig, ax, t, "Every forecast can name what moved it",
            "Gain attribution on the residual model. The simulated conflict term "
            "and the live delay state carry the forecast - the same quantities the "
            "per-section attribution shows a controller.")
    _save(fig, "12_feature_importance", theme)


# --------------------------------------------------------------------------- #
# 13 - error distributions
# --------------------------------------------------------------------------- #
@chart
def c13_error_distribution(ctx, theme):
    t = _style(theme)
    d = ctx["scored"]
    bins = np.linspace(-90, 90, 73)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))
    ax.hist(np.clip(d["err_base"], -90, 90), bins=bins, color=BASE, alpha=0.85,
            label="Carry-forward")
    ax.hist(np.clip(d["err_rc"], -90, 90), bins=bins, color=ACCENT, alpha=0.85,
            label="RAILCAST")
    ax.axvline(0, color=t["ink"], lw=1.0)
    ax.set_xlabel("Forecast minus actual arrival (minutes)")
    ax.set_ylabel("Forecasts")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", lw=0.6)
    ax.set_title("Where the error sits", fontsize=11, color=t["mute"], loc="left")

    delays = ctx["final_delays"]
    ax2.hist(np.clip(delays, -30, 240), bins=54, color=SIM, alpha=0.9)
    ax2.axvline(15, color=t["ink"], lw=1.2, ls="--")
    ax2.text(17, ax2.get_ylim()[1] * 0.92,
             f"{100 * np.mean(np.asarray(delays) < 15):.0f}% arrive within 15 min",
             fontsize=9.5, color=t["ink"])
    ax2.set_xlabel("Delay at destination (minutes)")
    ax2.set_ylabel("Train journeys")
    ax2.grid(axis="y", lw=0.6)
    ax2.set_title("The corridor the model is learning on", fontsize=11,
                  color=t["mute"], loc="left")
    _finish(fig, [ax, ax2], t, "Error distribution, and the operating reality behind it",
            "Left: carry-forward is biased late as well as noisy. Right: the "
            "delay distribution the simulator produces. Its parameters were set "
            "by hand, not fitted to Indian Railways punctuality returns - "
            "calibrating against those returns is step 2 of the roadmap.")
    _save(fig, "13_error_distribution", theme)


# --------------------------------------------------------------------------- #
# 14 - serving cost
# --------------------------------------------------------------------------- #
@chart
def c14_latency(ctx, theme):
    t = _style(theme)
    lat = ctx["latency"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0), width_ratios=[1.2, 1])
    ax.hist(lat["per_forecast_ms"], bins=40, color=ACCENT, alpha=0.9)
    for q, c in ((50, t["ink"]), (95, BAD)):
        v = np.percentile(lat["per_forecast_ms"], q)
        ax.axvline(v, color=c, lw=1.4, ls="--")
        ax.text(v, ax.get_ylim()[1] * (0.94 if q == 50 else 0.84), f" p{q} {v:.2f} ms",
                fontsize=9.5, color=c)
    ax.set_xlabel("Milliseconds per station forecast")
    ax.set_ylabel("Re-forecast cycles")
    ax.grid(axis="y", lw=0.6)
    ax.set_title("Cost of one station prediction", fontsize=11, color=t["mute"], loc="left")

    nat = lat["cycle_s"] * 13000 / lat["trains"]
    names = ["This corridor\n(%d trains)" % lat["trains"], "One zone\n(~800 trains)",
             "National,\none process", "National,\n16 zonal shards"]
    vals = [lat["cycle_s"], lat["cycle_s"] * 800 / lat["trains"], nat, nat / 16]
    cols = [GOOD if v < 30 else BAD for v in vals]
    ax2.bar(names, vals, color=cols, width=0.58)
    ax2.axhline(30, color=t["ink"], lw=1.3, ls="--")
    ax2.text(3.45, 32, "30 s budget", ha="right", fontsize=9, color=t["ink"])
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.06, f"{v:.1f} s", ha="center", fontweight="bold", fontsize=10.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Seconds for one full re-forecast (log scale)")
    ax2.grid(axis="y", lw=0.6)
    ax2.set_title("Scaled on one commodity node", fontsize=11, color=t["mute"], loc="left")
    _finish(fig, [ax, ax2], t, "What a full-network re-forecast costs",
            "Measured on this machine, single process, no GPU. National figures are "
            "linear extrapolation by train count: one process misses the 30-second "
            "budget, sharding the block graph by zone clears it with room to spare.")
    _save(fig, "14_latency_scaling", theme)


# --------------------------------------------------------------------------- #
def render_all(ctx) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    for fn in _charts:
        for theme in ("light", "dark"):
            try:
                fn(ctx, theme)
                n += 1
            except Exception as exc:                     # keep the rest rendering
                print(f"  ! {fn.__name__} [{theme}] failed: {exc}")
    return n


# --------------------------------------------------------------------------- #
# 15 - the passenger-facing card, filled from a real prototype forecast
# --------------------------------------------------------------------------- #
@chart
def c15_console_card(ctx, theme):
    t = _style(theme)
    c = ctx["card"]
    fig = plt.figure(figsize=(11.5, 6.2))
    ax = fig.add_axes((0.04, 0.08, 0.92, 0.76))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=THEMES[theme]["panel"], edgecolor=t["grid"],
                               lw=1.2, zorder=0))
    L, R = 0.045, 0.53
    ax.text(L, 0.90, f"{c['train']}  {c['name']}", fontsize=17, fontweight="bold",
            color=t["ink"], transform=ax.transAxes)
    ax.text(L, 0.82, f"at {c['at']}   ·   {_hm(c['now'])}   ·   "
                     f"delay {c['cur_delay']:+.0f} min"
                     f"{', easing' if c['momentum'] < -1 else ', building' if c['momentum'] > 1 else ''}",
            fontsize=11.5, color=t["mute"], transform=ax.transAxes)
    ax.plot([L, 0.955], [0.755, 0.755], color=t["grid"], lw=1.2,
            transform=ax.transAxes, clip_on=False)

    def block(x, y, label, code, eta, lo, hi, sched=None):
        ax.text(x, y, label, fontsize=10, color=t["mute"], transform=ax.transAxes)
        ax.text(x, y - 0.10, code, fontsize=15, fontweight="bold", color=t["ink"],
                transform=ax.transAxes)
        ax.text(x + 0.17, y - 0.10, _hm(eta), fontsize=24, fontweight="bold",
                color=ACCENT, transform=ax.transAxes)
        ax.text(x + 0.17, y - 0.185, f"80%   {_hm(lo)} – {_hm(hi)}   "
                                     f"({hi - lo:.0f} min wide)",
                fontsize=11, color=t["ink"], transform=ax.transAxes)
        if sched is not None:
            ax.text(x, y - 0.185, f"booked {_hm(sched)}", fontsize=10,
                    color=t["mute"], transform=ax.transAxes)

    block(L, 0.66, "NEXT STATION", c["next_code"], c["next_eta"],
          c["next_lo"], c["next_hi"])
    block(R, 0.66, "DESTINATION", c["dest_code"], c["dest_eta"],
          c["dest_lo"], c["dest_hi"], c["dest_sched"])

    ax.plot([L, 0.955], [0.40, 0.40], color=t["grid"], lw=1.2,
            transform=ax.transAxes, clip_on=False)
    why = (f"Held for {c['blocker']} at {c['blocked_at']}"
           if c["blocker"] else "Running clear; no conflict ahead")
    ax.text(L, 0.30, "WHY", fontsize=10, color=t["mute"], transform=ax.transAxes)
    ax.text(L, 0.21, why, fontsize=13, color=t["ink"], transform=ax.transAxes)
    ax.text(R, 0.30, "CONNECTION, 45 MIN BUFFER", fontsize=10, color=t["mute"],
            transform=ax.transAxes)
    p = c["p_connection"]
    ax.text(R, 0.19, f"{100 * p:.0f}%", fontsize=26, fontweight="bold",
            color=GOOD if p > 0.75 else "#E9B949" if p > 0.5 else BAD,
            transform=ax.transAxes)
    ax.text(R + 0.11, 0.21, "likely to make it", fontsize=12, color=t["ink"],
            transform=ax.transAxes)
    ax.add_patch(plt.Rectangle((R, 0.09), 0.40, 0.035, transform=ax.transAxes,
                               facecolor=t["grid"], zorder=1))
    ax.add_patch(plt.Rectangle((R, 0.09), 0.40 * p, 0.035, transform=ax.transAxes,
                               facecolor=GOOD if p > 0.75 else "#E9B949" if p > 0.5 else BAD,
                               zorder=2))
    _finish(fig, ax, t, "What the passenger actually sees",
            "Every number here is produced by the prototype, for one held-out forecast.")
    _save(fig, "15_console_card", theme)

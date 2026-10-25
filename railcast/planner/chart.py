"""
One chart per response - the two things a passenger is deciding on.

    left    chance of reaching B for each option, coloured Safe / Likely / Risky
    right   the weakest connection among the picks: when the first train is
            likely to arrive against when the next one leaves. The shaded part
            is the chance of missing it.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .. import config
from ..dist import quantile_sample
from ..forecast.charts import THEME, _finish
from . import delays as dl
from . import risk
from .network import Network
from .respond import OUT, Response
from .search import Leg

COLORS = {"Safe": "#4CAF7D", "Likely": "#F2A03D", "Risky": "#E2574C", "Unchecked": "#8A94A6"}


def render(net: Network, resp: Response, top: int = 12) -> str:
    t = THEME
    opts = [resp.options[i] for ids in resp.groups.values() for i in ids][:top]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.6),
                                  gridspec_kw={"width_ratios": [1.15, 1]})
    if not opts:
        plt.close(fig)
        raise RuntimeError("nothing to draw - no itinerary found")

    # left: every option's chance of reaching B
    y = np.arange(len(opts))[::-1]
    ax.barh(y, [o.p_reach for o in opts], color=[COLORS[o.group] for o in opts], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{o.dep.strftime('%a %H:%M')}  " + " + ".join(l.train for l in o.legs)
                        + (f" via {', '.join(c.at for c in o.connections)}" if o.connections else "")
                        for o in opts], fontsize=8.5)
    for v in (config.LIKELY, config.SAFE):
        ax.axvline(v, color=t["mute"], lw=0.9, ls="--")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Chance of reaching " + resp.query["to"] +
                  ("" if any(o.seats_checked for o in opts) else " (connections only)"))
    ax.grid(axis="x", lw=0.6)

    # right: the weakest connection among the picks - the one worth worrying about
    with_conn = [resp.options[i] for i in dict.fromkeys(resp.picks.values())
                 if resp.options[i].connections]
    if with_conn:
        pick = min(with_conn, key=lambda o: min(c.p_make for c in o.connections))
        _connection_panel(ax2, net, resp, pick, t)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "The top picks are direct trains -\nno connection to miss.",
                 ha="center", va="center", color=t["mute"], fontsize=11)

    q = resp.query
    _finish(fig, [ax, ax2], t, f"{q['from_name']} to {q['to_name']}, {q['date']}",
            f"Safe is {config.SAFE:.0%} or more, Likely {config.LIKELY:.0%}-{config.SAFE:.0%}. "
            f"Delay model: {resp.delay_model}. Seats: {resp.seats}.",
            foot="Timetable: datameet snapshot, running days not in the data - "
                 "confirm each train runs on your date")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{q['from']}-{q['to']}-{q['date']}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _connection_panel(ax, net, resp, o, t) -> None:
    k = int(np.argmin([c.p_make for c in o.connections]))
    conn = o.connections[k]
    arriving, leaving = o.legs[k], o.legs[k + 1]
    # rebuild the two legs' delay spreads with the same model the response used
    from datetime import date
    day0 = date.fromisoformat(resp.query["date"])
    legs = _legs_for(net, o, day0)
    sc = risk.score(net, [legs], day0, dl.load(_which(resp)))[0]
    rng = np.random.default_rng(3)
    d1 = quantile_sample(*([v] for v in sc.arr_q[k]), config.MONTE_CARLO_DRAWS, rng)[0]
    d2 = np.maximum(quantile_sample(*([v] for v in sc.dep_q[k + 1]),
                                    config.MONTE_CARLO_DRAWS, rng)[0], 0.0)
    from .network import ALIASES
    transfer = net.transfer_min(ALIASES.get(conn.at, conn.at),
                                ALIASES.get(conn.board_at, conn.board_at))
    slack = conn.layover_min + d2 - d1 - transfer       # minutes to spare
    bins = np.linspace(np.percentile(slack, 0.5), np.percentile(slack, 99.5), 60)
    hist, edges = np.histogram(slack, bins=bins, density=True)
    mids = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(mids, hist, where=mids >= 0, color=COLORS["Safe"], alpha=0.55, step="mid",
                    label=f"make it ({conn.p_make:.0%})")
    ax.fill_between(mids, hist, where=mids < 0, color=COLORS["Risky"], alpha=0.65, step="mid",
                    label=f"miss it ({1 - conn.p_make:.0%})")
    ax.axvline(0, color=t["ink"], lw=1.2)
    ax.set_xlabel(f"Minutes to spare at {conn.at_name} after the transfer")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title(f"{arriving.train} arrives, {leaving.train} leaves - "
                 f"{conn.layover_min:.0f} min on the timetable",
                 fontsize=11, color=t["mute"], loc="left")


def _legs_for(net, o, day0):
    """The search legs behind a response option."""
    from datetime import datetime, time
    from .search import _assemble
    base = datetime.combine(day0, time())
    from .network import ALIASES
    legs = []
    for lv in o.legs:
        seq = net.calls[lv.train]
        frm, to = ALIASES.get(lv.frm, lv.frm), ALIASES.get(lv.to, lv.to)
        i = next(n for n, c in enumerate(seq) if c.code == frm and c.halt)
        j = next(n for n, c in enumerate(seq) if c.code == to and n > i and c.halt)
        dep_min = (lv.dep - base).total_seconds() / 60
        legs.append(Leg(lv.train, int(round((dep_min - seq[i].dep) / 1440)), i, j))
    return _assemble(net, legs)


def _which(resp) -> str:
    label = resp.delay_model
    if label.startswith("prior"):
        return "prior"
    return "sim" if "simulated" in label else "live"

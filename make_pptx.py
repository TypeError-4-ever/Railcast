"""
Build the projectable RAILCAST deck.

    python make_pptx.py            -> outputs/RAILCAST_SIH26028.pptx

Native PowerPoint shapes throughout, not pictures of slides: every box, diamond
and arrow is a real shape and every string is real text, so the team can edit
the deck on any machine. Fonts are Segoe UI and Consolas because they ship with
Windows and will not substitute on a hall projector.

Type is sized for projection rather than for reading on a laptop, so the slides
carry less text than the canvas version of the same deck.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path("outputs/RAILCAST_SIH26028.pptx")
ASSETS = Path("outputs/pptx")

# ---- palette (light theme) ------------------------------------------------ #
INK = RGBColor(0x12, 0x16, 0x1F)
MUTE = RGBColor(0x5B, 0x64, 0x72)
FAINT = RGBColor(0x8A, 0x94, 0xA3)
LINE = RGBColor(0xDD, 0xE1, 0xE8)
SURF = RGBColor(0xF6, 0xF7, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
AMBER = RGBColor(0xB4, 0x53, 0x09)
AMBER_BG = RGBColor(0xFD, 0xF6, 0xEC)
TEAL = RGBColor(0x0F, 0x76, 0x6E)
TEAL_BG = RGBColor(0xEF, 0xF5, 0xF4)
RED = RGBColor(0xB9, 0x1C, 0x1C)
RED_BG = RGBColor(0xFB, 0xF7, 0xF6)

SANS = "Segoe UI"
MONO = "Consolas"

W, H = 13.333, 7.5           # inches, 16:9
M = 0.55                     # side margin
CW = W - 2 * M               # content width


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def deck() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    return prs


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = WHITE
    return s


def _noline(shape):
    shape.line.fill.background()


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.0, shape=MSO_SHAPE.RECTANGLE):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.shadow.inherit = False
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        _noline(sp)
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(lw)
    sp.text_frame.word_wrap = True
    return sp


def text(slide, x, y, w, h, runs, size=12, color=INK, bold=False, font=SANS,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=1.0, space_after=0):
    """runs: a string, or a list of (text, {overrides}) tuples, or list of lines."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    lines = runs if isinstance(runs, list) else [runs]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        parts = ln if isinstance(ln, list) else [(ln, {})]
        for t, ov in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = ov.get("font", font)
            f.size = Pt(ov.get("size", size))
            f.bold = ov.get("bold", bold)
            f.color.rgb = ov.get("color", color)
    return tb


def label(slide, x, y, w, s, color=AMBER, size=9.5):
    return text(slide, x, y, w, 0.2, s.upper(), size=size, color=color,
                bold=True, font=MONO)


def hline(slide, x, y, w, color=LINE, lw=0.9):
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y),
                                    Inches(x + w), Inches(y))
    ln.line.color.rgb = color
    ln.line.width = Pt(lw)
    return ln


def _arrowhead(conn, kind="triangle", dashed=False, color=FAINT, lw=1.1):
    conn.line.color.rgb = color
    conn.line.width = Pt(lw)
    ln = conn.line._get_or_add_ln()
    if dashed:
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    ln.append(ln.makeelement(qn("a:tailEnd"),
                             {"type": kind, "w": "med", "len": "med"}))


def arrow(slide, x1, y1, x2, y2, color=FAINT, dashed=False, lw=1.1):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                   Inches(x2), Inches(y2))
    _arrowhead(c, dashed=dashed, color=color, lw=lw)
    return c


def elbow(slide, pts, color=FAINT, dashed=False, lw=1.1, head=True):
    """Draw an orthogonal path through [(x,y), ...]; arrowhead on the last leg."""
    segs = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(pts[:-1], pts[1:])):
        c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
        last = i == len(pts) - 2
        if last and head:
            _arrowhead(c, dashed=dashed, color=color, lw=lw)
        else:
            c.line.color.rgb = color
            c.line.width = Pt(lw)
            if dashed:
                ln = c.line._get_or_add_ln()
                ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
        segs.append(c)
    return segs


def node(slide, kind, x, y, w, h, title, sub=None, accent=AMBER, size=11.5):
    """A flowchart node. kind: process | decision | terminator | data."""
    shape, fill, line = {
        "process":    (MSO_SHAPE.RECTANGLE, WHITE, INK),
        "decision":   (MSO_SHAPE.DIAMOND, AMBER_BG, AMBER),
        "terminator": (MSO_SHAPE.ROUNDED_RECTANGLE, TEAL_BG, TEAL),
        "data":       (MSO_SHAPE.PARALLELOGRAM, TEAL_BG, TEAL),
        "alt":        (MSO_SHAPE.RECTANGLE, SURF, LINE),
    }[kind]
    sp = rect(slide, x, y, w, h, fill=fill, line=line, lw=1.25, shape=shape)
    if kind == "terminator":
        sp.adjustments[0] = 0.5
    tf = sp.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.line_spacing = 0.95
    r = p.add_run()
    r.text = title
    r.font.name = SANS
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = TEAL if kind == "terminator" else INK
    if sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.line_spacing = 0.95
        r2 = p2.add_run()
        r2.text = sub
        r2.font.name = SANS
        r2.font.size = Pt(size - 2.5)
        r2.font.color.rgb = MUTE
    return sp


def card(slide, x, y, w, h, title, body, accent=LINE, top_accent=False,
         fill=WHITE, tsize=12.5, bsize=10.5):
    rect(slide, x, y, w, h, fill=fill, line=LINE, lw=0.9)
    if top_accent:
        rect(slide, x, y, w, 0.035, fill=accent)
    else:
        rect(slide, x, y, 0.035, h, fill=accent)
    text(slide, x + 0.14, y + 0.12, w - 0.28, 0.25, title, size=tsize, bold=True)
    if body:
        text(slide, x + 0.14, y + 0.40, w - 0.28, h - 0.5, body, size=bsize,
             color=MUTE, spacing=1.12)


def chrome(slide, title, sub, num):
    rect(slide, 0, 0, W, 0.055, fill=AMBER)
    text(slide, M, 0.30, 8.2, 0.45, title, size=26, bold=True)
    if sub:
        text(slide, M, 0.80, 9.6, 0.3, sub, size=11.5, color=MUTE)
    text(slide, W - M - 0.7, 0.32, 0.7, 0.25, num, size=10, color=FAINT,
         font=MONO, align=PP_ALIGN.RIGHT)
    hline(slide, M, 1.12, CW)


def footer(slide, s, color=FAINT):
    text(slide, M, H - 0.42, CW, 0.25, s, size=9, color=color)


# --------------------------------------------------------------------------- #
# slides
# --------------------------------------------------------------------------- #
CHART = Path("outputs/charts/light")


def s01_title(prs):
    s = blank(prs)
    rect(s, 0, 0, W, 0.055, fill=AMBER)
    label(s, M, 0.55, 6, "Smart India Hackathon 2026", color=AMBER, size=10.5)
    text(s, M, 0.92, 7.0, 1.0, "RAILCAST", size=58, bold=True)
    text(s, M, 1.92, 7.0, 0.4, "Network-aware arrival forecasting for Indian Railways",
         size=17, color=MUTE)
    hline(s, M, 2.46, 6.9)
    text(s, M, 2.64, 6.9, 1.0,
         [[("Today a passenger gets ", {}), ("one number with nothing attached to it", {"bold": True}),
           (". RAILCAST gives an arrival window whose coverage is audited on held-out "
            "data — per segment and per horizon, not asserted once and pooled.", {})]],
         size=13, spacing=1.25)

    vals = [("60%", "lower ETA error than\ncarry-forward", AMBER),
            ("80%", "arrival window,\ncoverage audited", TEAL),
            ("40h", "forecast horizon,\nstation by station", TEAL)]
    for i, (v, lab, col) in enumerate(vals):
        x = M + i * 2.36
        rect(s, x, 3.62, 2.18, 1.12, fill=WHITE, line=LINE, lw=0.9)
        rect(s, x, 3.62, 2.18, 0.035, fill=col)
        text(s, x + 0.15, 3.76, 1.9, 0.4, v, size=27, bold=True, color=col)
        text(s, x + 0.15, 4.22, 1.9, 0.45, lab.split("\n"), size=9.5, color=MUTE,
             spacing=1.1)

    text(s, M, 5.02, 6.9, 0.3,
         "Forecasting how delay will move through the network — not repeating "
         "the delay we already have.", size=11, color=FAINT)

    # hero
    s.shapes.add_picture(str(CHART / "07_window_closing.png"), Inches(7.62),
                         Inches(0.92), width=Inches(5.16))
    text(s, 7.62, 3.62, 5.16, 0.7,
         [[("The window narrows as the train runs. ", {"bold": True, "color": INK}),
           ("Every remaining station gets an interval. It widens with distance and "
            "closes as each actual arrival comes in — 62 minutes down to 21.", {})]],
         size=10, color=MUTE, spacing=1.18)

    # metadata strip
    rect(s, 0, 5.62, W, 1.88, fill=SURF)
    hline(s, 0, 5.62, W)
    meta = [("PS ID", "SIH26028"),
            ("PROBLEM STATEMENT", "Dynamic Forecast of Expected\nTime of Arrival (ETA)"),
            ("ORGANISATION", "Ministry of Railways"),
            ("THEME", "Smart Automation"),
            ("CATEGORY", "Software"),
            ("TEAM", "[TEAM NAME · ID]")]
    for i, (k, v) in enumerate(meta):
        x = M + i * 2.05
        text(s, x, 5.92, 1.95, 0.2, k, size=8, color=FAINT, font=MONO, bold=True)
        text(s, x, 6.18, 1.95, 0.6, v.split("\n"), size=11.5, bold=True,
             color=AMBER if k == "TEAM" else INK, spacing=1.1)
    text(s, M, 7.05, CW, 0.25,
         "Prototype built on the real New Delhi – Mumbai Central working "
         "timetable with observed ERA5 weather.", size=9, color=FAINT)
    return s


def s02_solution(prs):
    s = blank(prs)
    chrome(s, "Proposed Solution", "forecast the cause, not the clock", "02")

    label(s, M, 1.30, 4, "Problem at hand", color=RED)
    probs = [("Delay is repeated, not forecast",
              "Today's ETA carries the current delay forward and trusts timetable "
              "padding. It cannot see congestion building ahead."),
             ("The network goes blind between stations",
              "Position is reported only at instrumented stations — gaps of "
              "40–60 minutes on long sections."),
             ("Cascades are invisible",
              "One train held at a junction delays every train behind it, yet no "
              "deployed ETA model represents that dependency."),
             ("One number hides the risk",
              "No confidence attached, so nobody knows when it is safe to act — "
              "and nothing handles a diversion mid-run.")]
    for i, (t, b) in enumerate(probs):
        card(s, M, 1.62 + i * 1.28, 3.55, 1.14, t, b, accent=RED, fill=RED_BG,
             tsize=11.5, bsize=9.5)

    # centre flow
    label(s, 4.46, 1.30, 4, "How one forecast is made", color=AMBER)
    cx, bw = 4.46, 4.0
    node(s, "data", cx, 1.60, bw, 0.52, "LIVE STATE",
         "position, delay, block occupancy, restrictions")
    arrow(s, cx + bw / 2, 2.12, cx + bw / 2, 2.34)
    node(s, "process", cx + 0.25, 2.34, bw - 0.5, 0.50, "A · Free-run time",
         "the physics of the section")
    arrow(s, cx + bw / 2, 2.84, cx + bw / 2, 3.06)
    node(s, "decision", cx + 0.45, 3.06, bw - 0.9, 0.86,
         "Conflict on the block ahead?")
    text(s, cx + bw / 2 + 0.06, 3.94, 0.5, 0.2, "yes", size=9, bold=True, color=AMBER)
    arrow(s, cx + bw / 2, 3.92, cx + bw / 2, 4.14)
    # no-branch bypasses the simulator
    elbow(s, [(cx + 0.45, 3.49), (cx - 0.18, 3.49), (cx - 0.18, 5.44), (cx + 0.25, 5.44)],
          dashed=True)
    text(s, cx - 0.16, 3.26, 0.6, 0.2, "no", size=9, bold=True, color=MUTE)

    node(s, "process", cx + 0.25, 4.14, bw - 0.5, 0.56, "B · Conflict simulation",
         "advance every train on the block graph")
    arrow(s, cx + bw / 2, 4.70, cx + bw / 2, 4.92)
    node(s, "process", cx + 0.25, 4.92, bw - 0.5, 0.50, "C · Learned residual",
         "corrects what the rules cannot")
    arrow(s, cx + bw / 2, 5.42, cx + bw / 2, 5.64)
    node(s, "process", cx + 0.25, 5.64, bw - 0.5, 0.50, "D · Calibrated window",
         "audited 80% coverage")
    arrow(s, cx + bw / 2, 6.14, cx + bw / 2, 6.36)
    node(s, "terminator", cx + 0.15, 6.36, bw - 0.3, 0.52, "DECISION SURFACES",
         "window · connection risk · costed precedence")
    # feedback
    elbow(s, [(cx + bw - 0.15, 6.62), (cx + bw + 0.18, 6.62), (cx + bw + 0.18, 1.86),
              (cx + bw, 1.86)], color=AMBER, dashed=True)
    text(s, cx + bw + 0.24, 3.9, 1.0, 0.6,
         ["every actual", "arrival re-fits", "the rest of", "the run"], size=8.5,
         color=AMBER, bold=True, spacing=1.05)

    label(s, 9.72, 1.30, 3.2, "Innovation", color=TEAL)
    inno = [("Forecasts the cause, not the clock",
             "Trains advanced over the real block-section graph until they conflict, "
             "so the ETA moves for a reason we can name."),
            ("Says how sure it is, and proves it",
             "Conformal calibration gives an 80% window checked per horizon, segment "
             "and time of day."),
            ("Turns the window into a decision",
             "Connection-make probability, a leave-home prompt, and the cost in "
             "network-minutes of letting one train go first."),
            ("Survives a disrupted network",
             "Cancellations and diversions re-plan the remaining sections instead of "
             "breaking the forecast.")]
    for i, (t, b) in enumerate(inno):
        card(s, 9.72, 1.62 + i * 1.28, 3.06, 1.14, t, b, accent=TEAL, fill=TEAL_BG,
             tsize=11.5, bsize=9.5)
    return s


def s03_technical(prs):
    s = blank(prs)
    chrome(s, "Technical Approach", "five stages, each one testable on its own", "03")

    y, hh = 1.48, 1.05
    node(s, "data", 0.55, y, 1.95, hh, "1 · DATA IN",
         "position feed, timetable,\nblock graph, weather")
    arrow(s, 2.56, y + hh / 2, 2.84, y + hh / 2)
    node(s, "process", 2.90, y, 1.95, hh, "2 · STATE & FEATURES",
         "delay momentum, padding left,\ntrains in the same block")
    arrow(s, 4.91, y + hh / 2, 5.19, y + hh / 2)
    core = node(s, "process", 5.25, y - 0.1, 2.55, hh + 0.2, "3 · FORECAST CORE",
                "A free-run · B conflict simulator\nC learned residual · D conformal")
    core.line.color.rgb = AMBER
    core.fill.fore_color.rgb = AMBER_BG
    arrow(s, 7.86, y + hh / 2, 8.14, y + hh / 2)
    node(s, "decision", 8.20, y - 0.22, 2.30, hh + 0.44,
         "4 · Coverage on target,\nper segment and horizon?", size=10.5)
    arrow(s, 10.56, y + hh / 2, 10.84, y + hh / 2)
    text(s, 10.58, y + hh / 2 - 0.24, 0.4, 0.2, "yes", size=9, bold=True, color=TEAL)
    node(s, "terminator", 10.90, y, 1.88, hh, "5 · DELIVERY",
         "app · displays · SMS/IVR\ncontrol console · API")

    # no -> back into the core
    elbow(s, [(9.35, y + hh + 0.22), (9.35, 3.28), (6.52, 3.28), (6.52, y + hh + 0.1)],
          color=AMBER, dashed=True)
    text(s, 9.42, 2.78, 1.6, 0.2, "no", size=9, bold=True, color=AMBER)
    text(s, 6.7, 3.32, 3.0, 0.2, "re-fit the conformal quantiles for the group that missed",
         size=9, color=AMBER)

    label(s, M, 3.72, 5, "What makes the forecast move")
    items = [("Arrivals forced monotone",
              "A run can never arrive somewhere before the station behind it."),
             ("Attribution on every change",
              "Names the blocking train, the section and the feature responsible."),
             ("Online correction at every halt",
              "Each actual arrival is a free label: the rest of the run re-forecasts."),
             ("Drift monitor and retraining",
              "Weekly refit, alarm on coverage or MAE drift by segment.")]
    for i, (t, b) in enumerate(items):
        card(s, M + (i % 2) * 3.00, 4.02 + (i // 2) * 1.06, 2.84, 0.94, t, b,
             accent=AMBER, tsize=11, bsize=9)

    label(s, 6.92, 3.72, 5, "Tech stack", color=TEAL)
    stack = [("Models", "Python · LightGBM\nscikit-learn · MAPIE · SHAP"),
             ("Simulation", "NetworkX block graph\nevent-driven, priority rules"),
             ("Serving", "FastAPI · Redis Streams\nPostgreSQL + PostGIS · K8s"),
             ("Interface", "React · MapLibre GL\nREST / WebSocket · SMS gateway")]
    for i, (t, b) in enumerate(stack):
        x = 6.92 + (i % 2) * 3.00
        yy = 4.02 + (i // 2) * 1.06
        rect(s, x, yy, 2.84, 0.94, fill=WHITE, line=LINE, lw=0.9)
        rect(s, x, yy, 2.84, 0.03, fill=TEAL)
        text(s, x + 0.14, yy + 0.13, 2.6, 0.2, t, size=11, bold=True)
        text(s, x + 0.14, yy + 0.40, 2.6, 0.5, b.split("\n"), size=9, color=MUTE,
             font=MONO, spacing=1.15)

    footer(s, "Every component is established open source with published benchmarks. "
              "No new trackside hardware — the system consumes position feeds Indian "
              "Railways already produces, through a swappable adapter.")
    return s


def s04_stack(prs):
    s = blank(prs)
    chrome(s, "Tech Stack — system design",
           "six layers, one direction of flow, and the loop that keeps it honest", "04")

    rows = [
        ("SOURCES", FAINT, SURF, LINE,
         [("NTES / FOIS", "live position feed"), ("Working timetable", "schedules, halts, padding"),
          ("OSM / Overpass", "block sections, loops"), ("Open-Meteo ERA5", "observed + forecast weather")]),
        ("INGEST", TEAL, TEAL_BG, TEAL,
         [("StaticFeed", "stations, roster, WTT"), ("WeatherFeed", "fog index, rain"),
          ("LiveFeed", "position + delay, polled"), ("Phone motion", "opt-in, optional")]),
        ("STORAGE", FAINT, WHITE, LINE,
         [("PostgreSQL + PostGIS", "block graph, run history"), ("Redis Streams", "live state, feature cache"),
          ("LiveStore", "observed arrivals, append-only"), ("Object store", "models, calibration sets")]),
        ("COMPUTE", AMBER, AMBER_BG, AMBER,
         [("Offline — nightly", "free-run + residual fit"), ("Conformal calibration", "per group quantiles"),
          ("Online — every 30 s", "block-graph replay"), ("Batched inference", "0.46 ms per station")]),
        ("SERVING", FAINT, WHITE, LINE,
         [("FastAPI", "REST + WebSocket"), ("Docker · Kubernetes", "one node per zone, no GPU"),
          ("Sharded by zone", "national run inside 30 s"), ("p95 < 200 ms", "API latency budget")]),
        ("CLIENTS", TEAL, TEAL_BG, TEAL,
         [("Passenger app", "React · MapLibre GL"), ("Station displays", "window + platform"),
          ("SMS / IVR 139", "regional languages"), ("Control console", "costed precedence")]),
    ]
    y = 1.30
    for name, lc, fill, ln, cells in rows:
        text(s, M, y + 0.24, 0.95, 0.2, name, size=8.5, bold=True, font=MONO, color=lc)
        for j, (t, b) in enumerate(cells):
            x = 1.58 + j * 2.32
            rect(s, x, y, 2.18, 0.68, fill=fill, line=ln, lw=0.9)
            text(s, x + 0.11, y + 0.09, 1.98, 0.2, t, size=10, bold=True)
            text(s, x + 0.11, y + 0.32, 1.98, 0.3, b, size=8, color=MUTE, font=MONO)
        y += 0.68
        if name != "CLIENTS":
            arrow(s, 6.9, y + 0.04, 6.9, y + 0.22)
            y += 0.26

    # calibration loop rail
    rx = 10.98
    rect(s, rx, 1.30, 1.80, 5.02, fill=RGBColor(0xFD, 0xFA, 0xF5), line=AMBER, lw=1.1)
    text(s, rx + 0.13, 1.44, 1.55, 0.2, "CALIBRATION\nLOOP".split("\n"), size=8.5,
         bold=True, font=MONO, color=AMBER, spacing=1.15)
    text(s, rx + 0.13, 1.92, 1.55, 0.7,
         "No public archive of past arrivals exists. History is accumulated by "
         "polling, then used to fit what cannot be measured.", size=8, color=MUTE,
         spacing=1.15)
    steps = [("collect", "poll the live feed"), ("validate", "observed vs simulated"),
             ("calibrate", "fit to the real distribution"), ("SimConfig", "carries its own residuals")]
    yy = 2.86
    for i, (t, b) in enumerate(steps):
        rect(s, rx + 0.13, yy, 1.55, 0.62, fill=WHITE,
             line=AMBER if i == 3 else RGBColor(0xE8, 0xD9, 0xC4), lw=0.9)
        text(s, rx + 0.24, yy + 0.09, 1.35, 0.2, t, size=9.5, bold=True)
        text(s, rx + 0.24, yy + 0.31, 1.35, 0.25, b, size=7.5, color=MUTE, font=MONO)
        yy += 0.62
        if i < 3:
            arrow(s, rx + 0.9, yy + 0.03, rx + 0.9, yy + 0.18, color=AMBER)
            yy += 0.22

    footer(s, "Sources, corridor, timetable and weather are measured from published "
              "data. How delay arises is modelled — the calibration loop fits it to "
              "live running data as that is collected.")
    return s


def s05_flow(prs):
    s = blank(prs)
    chrome(s, "Runtime flow — one re-forecast",
           "what happens every 30 seconds, for every running train", "05")

    lx = 8.24
    for i, (kind, lab) in enumerate([("terminator", "start / end"), ("process", "process"),
                                     ("decision", "decision"), ("data", "data")]):
        node(s, kind, lx + i * 0.88, 0.34, 0.32, 0.19, "")
        text(s, lx + i * 0.88 + 0.37, 0.37, 0.52, 0.18, lab, size=7.5,
             color=MUTE)

    cx = [0.55, 3.07, 5.59, 8.11, 10.63]
    cw = 2.15
    r1, r2, r3 = 1.50, 3.22, 4.94
    bh, dh = 0.86, 1.06

    def mid(i):
        return cx[i] + cw / 2

    # ---- row 1, left to right ----
    node(s, "terminator", cx[0], r1, cw, bh, "POSITION REPORT", "or the 30 s clock fires")
    arrow(s, cx[0] + cw, r1 + bh / 2, cx[1], r1 + bh / 2)
    node(s, "process", cx[1], r1, cw, bh, "Update live state",
         "delay, momentum, block occupancy")
    arrow(s, cx[1] + cw, r1 + bh / 2, cx[2], r1 + bh / 2)
    node(s, "decision", cx[2], r1 - 0.10, cw, dh, "Train running?")
    arrow(s, cx[2] + cw, r1 + bh / 2, cx[3], r1 + bh / 2)
    text(s, cx[2] + cw + 0.02, r1 + bh / 2 - 0.22, 0.4, 0.2, "yes", size=8.5,
         bold=True, color=TEAL)
    node(s, "process", cx[3], r1, cw, bh, "A · Free-run time", "per remaining section")
    arrow(s, cx[3] + cw, r1 + bh / 2, cx[4], r1 + bh / 2)
    node(s, "process", cx[4], r1, cw, bh, "B · Advance the block graph",
         "every train, in event order", size=10.5)

    elbow(s, [(mid(2), r1 + dh - 0.10), (mid(2), 2.66), (6.13, 2.66)], head=True)
    node(s, "alt", cx[1] + 0.62, 2.46, 2.40, 0.40, "Show booked schedule", None, size=9.5)
    text(s, mid(2) + 0.06, 2.30, 0.4, 0.2, "no", size=8.5, bold=True, color=MUTE)

    elbow(s, [(mid(4), r1 + bh), (mid(4), r2 - 0.20)])

    # ---- row 2, right to left ----
    node(s, "decision", cx[4], r2 - 0.10, cw, dh, "Block section ahead occupied?", size=10)
    arrow(s, cx[4], r2 + bh / 2, cx[3] + cw, r2 + bh / 2)
    text(s, cx[4] - 0.44, r2 + bh / 2 - 0.25, 0.5, 0.2, "yes", size=8.5, bold=True,
         color=AMBER)
    node(s, "decision", cx[3], r2 - 0.10, cw, dh, "Higher-priority train behind?", size=10)
    node(s, "process", cx[2], r2 - 0.16, cw, 0.48, "Loop the slower train", None, size=10)
    node(s, "process", cx[2], r2 + 0.50, cw, 0.48, "Queue behind the block", None, size=10)
    elbow(s, [(cx[3], r2 + 0.08), (cx[2] + cw, r2 + 0.08)], head=True)
    elbow(s, [(cx[3], r2 + 0.74), (cx[2] + cw, r2 + 0.74)], head=True)
    text(s, 7.77, r2 - 0.14, 0.34, 0.18, "yes", size=8, bold=True, color=AMBER)
    text(s, 7.80, r2 + 0.80, 0.30, 0.18, "no", size=8, bold=True, color=MUTE)
    elbow(s, [(mid(2), r2 + 0.98), (mid(2), 4.44), (cx[1] + cw, 4.44)], head=False)
    arrow(s, cx[2], r2 + bh / 2, cx[1] + cw, r2 + bh / 2)
    node(s, "process", cx[1], r2, cw, bh, "C · Learned residual",
         "one shot per horizon, never chained", size=10.5)
    arrow(s, cx[1], r2 + bh / 2, cx[0] + cw, r2 + bh / 2)
    node(s, "process", cx[0], r2, cw, bh, "D · Conformal interval",
         "horizon × segment × time of day", size=10.5)

    elbow(s, [(mid(4), r2 + dh - 0.10), (mid(4), 4.52), (mid(1), 4.52), (mid(1), r2 + bh)],
          dashed=True)
    text(s, mid(4) + 0.08, 4.24, 1.4, 0.2, "no — runs clear", size=8, bold=True, color=MUTE)

    elbow(s, [(mid(0), r2 + bh), (mid(0), r3 - 0.20)])

    # ---- row 3 ----
    node(s, "decision", cx[0], r3 - 0.10, cw, dh, "Coverage on target for this group?",
         size=10)
    arrow(s, cx[0] + cw, r3 + bh / 2, cx[1], r3 + bh / 2)
    text(s, cx[0] + cw + 0.02, r3 + bh / 2 - 0.25, 0.4, 0.2, "yes", size=8.5,
         bold=True, color=TEAL)
    node(s, "terminator", cx[1], r3, cw * 2 + 0.37, bh, "PUBLISH THE ARRIVAL WINDOW",
         "monotone arrivals · attribution · connection risk")

    elbow(s, [(mid(0), r3 + dh - 0.10), (mid(0), 6.30), (cx[3] + 0.28, 6.30)],
          head=True, color=AMBER, dashed=True)
    text(s, mid(0) + 0.06, 6.04, 0.4, 0.2, "no", size=8.5, bold=True, color=AMBER)
    node(s, "process", cx[3] + 0.30, 6.08, 2.60, 0.44, "Re-fit the quantiles", None, size=10)
    elbow(s, [(cx[3] + 2.92, 6.30), (12.95, 6.30), (12.95, r2 + 0.20),
              (cx[0] + cw, r2 + 0.20)], color=AMBER, dashed=True)

    footer(s, "Two mechanisms cost time and the flow keeps them apart: a train LOOPED "
              "at a station is a decision by control, a train QUEUED behind an occupied "
              "block is physics. Only the first is a choice the costed-precedence "
              "surface can price.")
    return s


def s06_feasibility(prs):
    s = blank(prs)
    chrome(s, "Feasibility and Viability",
           "mature components, a stated baseline, every risk given a retirement plan",
           "06")

    label(s, M, 1.30, 4, "Technical feasibility", color=TEAL)
    feas = [("Every component is mature",
             "LightGBM and conformal prediction are established open source."),
            ("No new trackside hardware",
             "Consumes position feeds IR already produces, via a swappable adapter."),
            ("Cheap at national scale",
             "One commodity node per zone; no GPU at serving time."),
            ("The simulator is its own fallback",
             "With the learned layer off, rules still beat carry-forward past 3 hours."),
            ("Phone sensing is optional",
             "An accuracy bonus, never a dependency.")]
    for i, (t, b) in enumerate(feas):
        card(s, M, 1.62 + i * 0.92, 3.90, 0.80, t, b, accent=TEAL, tsize=10.5, bsize=8.8)

    label(s, 4.72, 1.30, 4, "How we prove it")
    proof = [("1  Baseline stated",
              "Current delay carried forward, on exactly the same rows."),
             ("2  Temporal holdout",
              "Train on earlier months, test on later. A random split leaks."),
             ("3  Error by horizon",
              "Next station, 1-3 h, 3-8 h, 8-24 h, beyond 24 h."),
             ("4  Coverage audited",
              "Against nominal 80%, by segment, horizon and time of day."),
             ("5  Shadow mode",
              "Run live beside the existing ETA before a passenger sees it.")]
    for i, (t, b) in enumerate(proof):
        card(s, 4.72, 1.62 + i * 0.92, 3.90, 0.80, t, b, accent=AMBER, tsize=10.5,
             bsize=8.8)

    label(s, 8.88, 1.30, 4, "Risks and how we retire them", color=RED)
    risks = [("No live IR feed in development",
              "FeedAdapter plus a movement simulator; swap at deployment."),
             ("Error compounds over a 40-hour run",
              "Direct multi-horizon residual - never chained."),
             ("Coverage fails in the monsoon",
              "Mondrian calibration by segment and season, alarmed not assumed."),
             ("Controllers distrust a black box",
              "Per-section attribution; advisory only, never a movement order."),
             ("Phone motion is not train motion",
              "Gyro gating and consensus; degrades to the timetable model.")]
    for i, (t, b) in enumerate(risks):
        card(s, 8.88, 1.62 + i * 0.92, 3.90, 0.80, t, b, accent=RED, fill=RED_BG,
             tsize=10.5, bsize=8.8)

    label(s, M, 6.32, 5, "Implementation roadmap")
    steps = ["Build the block-section graph", "Calibrate the simulator",
             "Train free-run and residual", "Add conformal calibration",
             "Ship the decision layer", "Shadow-mode pilot, then zone by zone"]
    bw = 1.90
    for i, t in enumerate(steps):
        x = M + i * (bw + 0.13)
        last = i == len(steps) - 1
        rect(s, x, 6.60, bw, 0.54, fill=TEAL_BG if last else SURF,
             line=TEAL if last else LINE, lw=0.9)
        text(s, x + 0.10, 6.66, 0.4, 0.16, "%02d" % (i + 1), size=8, bold=True,
             font=MONO, color=AMBER)
        text(s, x + 0.10, 6.84, bw - 0.2, 0.26, t, size=8.5, spacing=1.05)
        if not last:
            arrow(s, x + bw, 6.87, x + bw + 0.11, 6.87)
    return s


def s07_impact(prs):
    s = blank(prs)
    chrome(s, "Impact and Benefits",
           "measured on held-out days against a stated baseline", "07")

    mets = [("60%", "lower mean ETA error", "carry-forward 26.9 min\nRAILCAST 10.9 min", AMBER),
            ("62%", "lower P90 error", "the bad days:\n77 min against 29", AMBER),
            ("78.4%", "achieved coverage", "of the nominal 80% window,\naudited per segment", TEAL),
            ("0.006", "reliability error", "on connection-make\nprobability", TEAL)]
    for i, (v, lab, sub, col) in enumerate(mets):
        x = M + i * 3.10
        rect(s, x, 1.30, 2.94, 1.14, fill=WHITE, line=LINE, lw=0.9)
        rect(s, x, 1.30, 2.94, 0.035, fill=col)
        text(s, x + 0.14, 1.44, 2.6, 0.36, v, size=25, bold=True, color=col)
        text(s, x + 0.14, 1.84, 2.66, 0.2, lab, size=10, bold=True)
        text(s, x + 0.14, 2.06, 2.66, 0.34, sub.split("\n"), size=8.5, color=MUTE,
             spacing=1.1)

    label(s, M, 2.66, 6, "Error by horizon, not one pooled number")
    s.shapes.add_picture(str(CHART / "01_mae_by_horizon.png"), Inches(M),
                         Inches(2.92), width=Inches(6.10))
    text(s, M, 6.10, 6.10, 0.5,
         "Carry-forward is adequate for the next station and collapses beyond a few "
         "hours - which is the horizon passengers actually plan against.",
         size=9.5, color=MUTE, spacing=1.15)

    label(s, 7.02, 2.66, 6, "What changes against today", color=TEAL)
    rows = [("ETA method", "Delay repeated forward", "Per-section forecast, conflicts simulated"),
            ("Horizon", "The next few stops", "The whole run, 40+ hours"),
            ("Uncertainty", "None stated", "Audited 80% arrival window"),
            ("Explains cause", "No", "Names the blocking train and section"),
            ("Between stations", "Blind for 40-60 min", "Phone-sensed stop and crawl events"),
            ("Disruption", "Estimate simply breaks", "Remaining route re-planned mid-run"),
            ("Decision support", "None", "Connection risk and costed precedence")]
    y = 2.92
    rect(s, 7.02, y, 5.76, 0.30, fill=SURF, line=LINE, lw=0.9)
    for j, hcap in enumerate(["PARAMETER", "TODAY", "RAILCAST"]):
        text(s, 7.14 + j * 1.92, y + 0.09, 1.85, 0.18, hcap, size=7.5, bold=True,
             font=MONO, color=AMBER if j == 2 else FAINT)
    y += 0.30
    for i, (a, b, c) in enumerate(rows):
        rect(s, 7.02, y, 5.76, 0.44, fill=WHITE, line=LINE, lw=0.6)
        text(s, 7.14, y + 0.13, 1.80, 0.3, a, size=9, bold=True)
        text(s, 9.06, y + 0.13, 1.80, 0.3, b, size=9, color=MUTE)
        text(s, 10.98, y + 0.13, 1.74, 0.3, c, size=9)
        y += 0.44

    footer(s, "Scored on 73,268 held-out forecasts on the real New Delhi - Mumbai "
              "Central working timetable with observed ERA5 weather. How delay arises "
              "is still modelled; the calibration loop fits it to live running data.")
    return s


def s08_simulation(prs):
    s = blank(prs)
    chrome(s, "The prototype, running",
           "both clips are the simulator itself - nothing is hand-animated", "08")

    vids = [("outputs/video/railcast_corridor.mp4", "poster_corridor.png",
             "The problem — conflict and cascade",
             "40 trains on the real 1384 km corridor over one day. Captions name the "
             "train that actually caused each hold, read out of the run log."),
            ("outputs/video/railcast_forecast.mp4", "poster_forecast.png",
             "The answer — one journey re-forecast",
             "The 80% window narrowing from 62 to 21 minutes as actual arrivals come "
             "in. Carry-forward 57 min average error against RAILCAST 15.")]
    for i, (mp4, poster, title, body) in enumerate(vids):
        x = M + i * 6.28
        p = Path(mp4)
        if p.exists():
            s.shapes.add_movie(str(p), Inches(x), Inches(1.42), Inches(5.94),
                               Inches(3.34),
                               poster_frame_image=str(ASSETS / poster),
                               mime_type="video/mp4")
        text(s, x, 4.92, 5.94, 0.3, title, size=14, bold=True)
        text(s, x, 5.24, 5.94, 0.7, body, size=10.5, color=MUTE, spacing=1.2)

    rect(s, M, 6.22, CW, 0.72, fill=AMBER_BG, line=AMBER, lw=1.0)
    text(s, M + 0.18, 6.34, CW - 0.4, 0.5,
         [[("Trains never collide. ", {"bold": True, "color": INK}),
           ("Absolute block working makes that impossible - a train cannot enter a "
            "section another train occupies. What costs Indian Railways time is "
            "conflict: a faster train closing on a slower one, the hold that follows, "
            "and the delay cascading behind it. That is what the left-hand clip shows.",
            {})]], size=10, color=MUTE, spacing=1.2)

    footer(s, "Click a clip to play in presentation mode. If the deck is opened "
              "somewhere the embedded video will not run, the same files are in "
              "outputs/video/ as MP4 and GIF.")
    return s


def s09_references(prs):
    s = blank(prs)
    chrome(s, "Research and References",
           "what we build on, and where we add to it", "09")

    label(s, M, 1.30, 4, "Scale of deployment")
    scale = [("40", "Prototype corridor", "New Delhi - Mumbai Central, 1384 km, 202 stations", AMBER),
             ("800", "Zonal pilot", "One zone, one control office, one model shard", AMBER),
             ("13,000", "National", "Coaching trains a day, 7,300 stations, 68,000 route km", TEAL)]
    for i, (v, t, b, col) in enumerate(scale):
        y = 1.62 + i * 0.86
        rect(s, M, y, 5.90, 0.74, fill=WHITE, line=LINE, lw=0.9)
        rect(s, M, y, 0.035, 0.74, fill=col)
        text(s, M + 0.18, y + 0.16, 1.20, 0.3, v, size=17, bold=True, color=col)
        text(s, M + 1.46, y + 0.12, 4.2, 0.2, t, size=11, bold=True)
        text(s, M + 1.46, y + 0.36, 4.2, 0.3, b, size=9, color=MUTE)

    label(s, M, 4.30, 4, "Data sources", color=TEAL)
    srcs = [("National Train Enquiry System", "live running position, via FeedAdapter"),
            ("Open Government Data Platform", "schedules and station master"),
            ("OpenStreetMap / Overpass", "block sections, junctions, loops"),
            ("Open-Meteo / ERA5", "historical and forecast weather, no key"),
            ("IR Year Book, WTT, TSR notices", "punctuality returns and restrictions")]
    y = 4.60
    for a, b in srcs:
        rect(s, M, y, 5.90, 0.40, fill=WHITE, line=LINE, lw=0.6)
        text(s, M + 0.14, y + 0.11, 2.75, 0.25, a, size=9.5, bold=True)
        text(s, M + 3.00, y + 0.11, 2.80, 0.25, b, size=9, color=MUTE)
        y += 0.40

    label(s, 6.92, 1.30, 5, "Research we build on")
    refs = ["1  R. Oneto et al., Train delay prediction systems: a big data analytics "
            "perspective, Big Data Research, vol. 11, 2018.",
            "2  P. Kecman and R. M. P. Goverde, Predictive modelling of running and "
            "dwell times in railway traffic, Public Transport, vol. 7, 2015.",
            "3  F. Corman and P. Kecman, Stochastic prediction of train delays in "
            "real time using Bayesian networks, Transportation Research Part C, 2018.",
            "4  T. Buker and B. Seybold, Stochastic modelling of delay propagation in "
            "large networks, J. Rail Transport Planning and Management, 2012.",
            "5  Y. Romano, E. Patterson and E. Candes, Conformalized quantile "
            "regression, NeurIPS 32, 2019.",
            "6  V. Vovk, A. Gammerman and G. Shafer, Algorithmic Learning in a Random "
            "World, Springer, 2005 - Mondrian and conditional validity.",
            "7  G. Ke et al., LightGBM: a highly efficient gradient boosting decision "
            "tree, NeurIPS 30, 2017.",
            "8  The Digital Personal Data Protection Act, 2023 (India)."]
    y = 1.62
    for r in refs:
        text(s, 6.92, y, 5.86, 0.3, r, size=9, color=MUTE, spacing=1.15)
        y += 0.40

    rect(s, 6.92, 4.94, 5.86, 1.04, fill=AMBER_BG, line=AMBER, lw=1.0)
    text(s, 7.08, 5.06, 5.54, 0.2, "WHERE WE ADD TO THIS", size=8.5, bold=True,
         font=MONO, color=AMBER)
    text(s, 7.08, 5.30, 5.54, 0.6,
         "None of the above combines a conflict-resolution simulator over the real "
         "block-section graph, a direct multi-horizon learned residual, and "
         "conditionally calibrated intervals on Indian Railways operating data.",
         size=9.5, spacing=1.2)

    text(s, 6.92, 6.16, 5.86, 0.5,
         [[("Project links  ", {"bold": True}),
           ("technical documentation · demo video · simulator notes · model results "
            "against baseline · GitHub repository", {"color": MUTE})]], size=9,
         spacing=1.2)
    footer(s, "Problem Statement SIH26028  ·  Ministry of Railways  ·  sih.gov.in/sih2026PS")
    return s


def main():
    prs = deck()
    for fn in (s01_title, s02_solution, s03_technical, s04_stack, s05_flow,
               s06_feasibility, s07_impact, s08_simulation, s09_references):
        fn(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    mb = OUT.stat().st_size / 1048576
    print("wrote %s  (%d slides, %.1f MB)" % (OUT, len(prs.slides._sldIdLst), mb))


if __name__ == "__main__":
    main()

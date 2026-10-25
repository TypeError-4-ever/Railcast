"""Scoring: RAILCAST against the carry-forward baseline, and the coverage audit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import HORIZON_LABELS, horizon_bucket, tod_bucket

from .corridor import zone_name as _zone_name
TOD_NAME = {0: "00-06", 1: "06-12", 2: "12-18", 3: "18-24"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def pick_baseline(train: pd.DataFrame) -> str:
    """Carry-forward, with or without padding absorption - whichever is better."""
    cands = {c: np.mean(np.abs(train[c] - train["act_arr"]))
             for c in ("baseline_arr", "baseline_pad_arr")}
    return min(cands, key=cands.get)


def _agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n": len(g),
        "baseline_mae": float(np.mean(np.abs(g["err_base"]))),
        "railcast_mae": float(np.mean(np.abs(g["err_rc"]))),
        "simulator_mae": float(np.mean(np.abs(g["err_rules"]))),
        "baseline_p90": float(np.percentile(np.abs(g["err_base"]), 90)),
        "railcast_p90": float(np.percentile(np.abs(g["err_rc"]), 90)),
        "coverage_pct": float(100 * g["covered"].mean()),
        "width_min": float(np.mean(g["hi_arr"] - g["lo_arr"])),
    })


def score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["err_base"] = out["baseline"] - out["act_arr"]
    out["err_rc"] = out["railcast_arr"] - out["act_arr"]
    out["err_rules"] = out["rules_only"] - out["act_arr"]
    out["covered"] = (out["act_arr"] >= out["lo_arr"]) & (out["act_arr"] <= out["hi_arr"])
    out["hbucket"] = horizon_bucket(out["sched_lead"].to_numpy())
    out["horizon"] = [HORIZON_LABELS[i] for i in out["hbucket"]]
    out["zone_name"] = [_zone_name(z) for z in out["zone"]]
    out["tod"] = [TOD_NAME[i] for i in tod_bucket(out["hour"].to_numpy())]
    out["month_name"] = [MONTHS[m - 1] for m in out["month"]]
    return out


def by(scored: pd.DataFrame, keys) -> pd.DataFrame:
    t = scored.groupby(keys, observed=True).apply(_agg, include_groups=False)
    t = t.reset_index()
    t["improvement_pct"] = 100 * (1 - t["railcast_mae"] / t["baseline_mae"])
    return t


def overall(scored: pd.DataFrame) -> dict:
    s = _agg(scored)
    return {
        "n_forecasts": int(s["n"]),
        "baseline_mae_min": round(float(s["baseline_mae"]), 2),
        "railcast_mae_min": round(float(s["railcast_mae"]), 2),
        "mae_improvement_pct": round(100 * (1 - s["railcast_mae"] / s["baseline_mae"]), 1),
        "baseline_p90_min": round(float(s["baseline_p90"]), 2),
        "railcast_p90_min": round(float(s["railcast_p90"]), 2),
        "coverage_pct": round(float(s["coverage_pct"]), 2),
        "nominal_coverage_pct": 80.0,
        "mean_window_min": round(float(s["width_min"]), 1),
    }

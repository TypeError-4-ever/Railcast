"""
The forecasting core.

  Stage A  free-run time model      LightGBM on observed unimpeded section runs.
  Stage B  conflict simulator       railcast.simulator.forward_replay.
  Stage C  learned residual         LightGBM quantile loss, direct multi-horizon.
  Stage D  calibrated window        Mondrian conformalised quantile regression.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:                                        # pragma: no cover
    lgb = None

from .dataset import CATEGORICAL, FEATURES, ZONE_ID
from .trains import CLASSES

ALPHA = 0.20                       # 80% nominal arrival window
QLO, QHI = ALPHA / 2, 1 - ALPHA / 2


# --------------------------------------------------------------------------- #
# stage A
# --------------------------------------------------------------------------- #
FR_FEATURES = ["theo_min", "priority", "max_speed", "zone", "hour", "length_km",
               "sect_speed", "weather_fc", "stopping", "month", "tsr_cap"]


def fit_free_run(sections: pd.DataFrame):
    """Learn unimpeded section run time. Trained only on conflict-free runs."""
    clean = sections[sections["conflict_min"] < 0.5]
    X, y = clean[FR_FEATURES], clean["obs_min"]
    model = lgb.LGBMRegressor(
        objective="l2", n_estimators=400, learning_rate=0.06, num_leaves=63,
        min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.9, verbose=-1)
    model.fit(X, y, categorical_feature=["zone", "month"])
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    naive = float(np.mean(np.abs(clean["theo_min"] - y)))
    return model, dict(n=int(len(clean)), mae_min=mae, theoretical_mae_min=naive)


def free_run_table(model, sections: pd.DataFrame) -> dict:
    """Collapse stage A into a (class, zone, 6-hour band) run-time multiplier.

    The replay traverses ~170 block sections per train; a table lookup keeps the
    full-network re-forecast inside its 30-second budget.
    """
    med = sections[FR_FEATURES].median()
    grid, keys = [], []
    for klass, spec in CLASSES.items():
        for zone, zid in ZONE_ID.items():
            for band in range(4):
                r = med.copy()
                r["priority"] = spec["priority"]
                r["max_speed"] = spec["max_speed"]
                r["zone"] = zid
                r["hour"] = band * 6 + 3
                grid.append(r)
                keys.append((klass, zone, band))
    G = pd.DataFrame(grid)[FR_FEATURES]
    mult = model.predict(G) / G["theo_min"].to_numpy()
    return {k: float(np.clip(m, 0.85, 1.6)) for k, m in zip(keys, mult)}


# --------------------------------------------------------------------------- #
# stage C
# --------------------------------------------------------------------------- #
def _lgb(objective, alpha=None, n=700):
    kw = dict(objective=objective, n_estimators=n, learning_rate=0.05,
              num_leaves=127, min_child_samples=60, subsample=0.85,
              subsample_freq=1, colsample_bytree=0.85, verbose=-1)
    if alpha is not None:
        kw["alpha"] = alpha
    return lgb.LGBMRegressor(**kw)


class ResidualModel:
    """Direct multi-horizon residual on the simulated ETA, with quantiles."""

    def __init__(self):
        self.mid = _lgb("l1")
        self.lo = _lgb("quantile", QLO)
        self.hi = _lgb("quantile", QHI)

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        y = df["residual"]
        for m in (self.mid, self.lo, self.hi):
            m.fit(X, y, categorical_feature=CATEGORICAL)
        return self

    def predict(self, df: pd.DataFrame):
        X = df[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        mid = self.mid.predict(X)
        lo = self.lo.predict(X)
        hi = self.hi.predict(X)
        # monotone quantiles: an interval may never cross
        lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)
        return mid, lo, hi

    def importance(self) -> pd.Series:
        s = pd.Series(self.mid.booster_.feature_importance("gain"), index=FEATURES)
        return (s / s.sum() * 100).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# stage D: Mondrian conformalised quantile regression
# --------------------------------------------------------------------------- #
HORIZON_EDGES = [0, 60, 180, 480, 1440, 10 ** 9]
HORIZON_LABELS = ["Next station (<1 h)", "1-3 h", "3-8 h", "8-24 h", "24 h +"]


def horizon_bucket(lead_min) -> np.ndarray:
    return np.clip(np.digitize(lead_min, HORIZON_EDGES[1:-1]), 0, 4)


def tod_bucket(hour) -> np.ndarray:
    return (np.asarray(hour) // 6).astype(int)


def mondrian_groups(df: pd.DataFrame) -> pd.Series:
    """Conditioning taxonomy: horizon bucket x zone x time of day."""
    return pd.Series(
        [f"{h}|{z}|{t}" for h, z, t in zip(
            horizon_bucket(df["sched_lead"].to_numpy()),
            df["zone"].to_numpy(),
            tod_bucket(df["hour"].to_numpy()))],
        index=df.index)


class MondrianConformal:
    """CQR: widen the learned quantiles until each group reaches 1-alpha."""

    def __init__(self, alpha: float = ALPHA, min_group: int = 120):
        self.alpha = alpha
        self.min_group = min_group
        self.q: dict[str, float] = {}
        self.global_q = 0.0

    def fit(self, df: pd.DataFrame, lo, hi):
        score = np.maximum(lo - df["residual"].to_numpy(),
                           df["residual"].to_numpy() - hi)
        g = mondrian_groups(df)
        n = len(score)
        self.global_q = float(np.quantile(
            score, min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n), method="higher"))
        for key, idx in g.groupby(g).groups.items():
            s = score[df.index.get_indexer(idx)]
            if len(s) < self.min_group:
                continue
            k = min(1.0, np.ceil((len(s) + 1) * (1 - self.alpha)) / len(s))
            self.q[key] = float(np.quantile(s, k, method="higher"))
        return self

    def apply(self, df: pd.DataFrame, lo, hi):
        g = mondrian_groups(df).map(lambda k: self.q.get(k, self.global_q))
        w = g.to_numpy()
        return lo - w, hi + w


# --------------------------------------------------------------------------- #
def enforce_monotone(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Arrivals along one run can never go backwards in time."""
    df = df.sort_values(["day", "train", "now", "hops"]).copy()
    key = df["day"].astype(str) + "|" + df["train"] + "|" + df["now"].round(3).astype(str)
    for c in cols:
        df[c] = df.groupby(key, sort=False)[c].cummax()
    return df

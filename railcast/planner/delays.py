"""
How late a train will be at a station, as a distribution - for trains that have
not started yet.

The corridor forecaster needs live positions and one corridor's block graph.
Planning a trip across India a day ahead needs something cruder that covers
every train: delay at a station given how far into its run the station is, the
class of train, the zone, the time of day and the season.

    PriorDelays     a stated rule, not fitted to anything. Used until a model
                    has been trained.
    LearnedDelays   LightGBM quantile regression on running records - simulated
                    today, real once the collector has gathered them
                    (`python -m railcast train-delays --source live`).

Progress into the run is measured in scheduled hours, not km: the simulator,
the live feed and the timetable each measure km differently, but a booked time
means the same thing everywhere.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..forecast.trains import CLASSES

MODEL_PATH = Path("models/delays.pkl")

FEATURES = ["klass", "hours_into_run", "zone", "hour", "month", "dow", "halt", "event"]
CATEGORICAL = ["klass", "zone", "month", "dow"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

KLASS_ID = {k: i for i, k in enumerate(CLASSES)}
ZONES = ["NR", "NCR", "NER", "NWR", "NFR", "ECR", "ER", "ECoR", "SER", "SECR",
         "CR", "WR", "WCR", "SR", "SCR", "SWR", "KR"]
ZONE_ID = {z: i for i, z in enumerate(ZONES)}
FOG_ZONES = {"NR", "NCR", "NER", "NWR", "ECR", "NFR", "ER"}


def encode(klass: str, zone: str) -> tuple[int, int]:
    return KLASS_ID.get(klass, KLASS_ID["EXPRESS"]), ZONE_ID.get(zone, -1)


# --------------------------------------------------------------------------- #
class PriorDelays:
    """Delay grows with time into the run, by class; fog doubles it in winter
    on the northern zones. Numbers chosen to look like Indian Railways, not
    fitted to its data."""

    label = "prior rule, not fitted to any data"
    PER_HOUR = {"RAJDHANI": 1.5, "DURONTO": 1.5, "SHATABDI": 1.2,
                "SUPERFAST": 2.5, "EXPRESS": 3.5, "PASSENGER": 4.0}

    def predict(self, df: pd.DataFrame):
        inv = {v: k for k, v in KLASS_ID.items()}
        rate = np.array([self.PER_HOUR[inv.get(int(k), "EXPRESS")] for k in df["klass"]])
        zinv = {v: k for k, v in ZONE_ID.items()}
        fog = np.array([1.0 + (zinv.get(int(z)) in FOG_ZONES) * (m in (12, 1)) +
                        0.4 * (zinv.get(int(z)) in FOG_ZONES) * (m == 2)
                        for z, m in zip(df["zone"], df["month"])])
        h = np.maximum(df["hours_into_run"].to_numpy(dtype=float), 0.0)
        q50 = rate * h * fog + 2.0 * df["event"].to_numpy()
        q10 = np.maximum(-5.0, 0.15 * q50 - 5.0)
        q90 = 2.8 * q50 + 15.0
        return q10, q50, q90


class LearnedDelays:
    def __init__(self, models: dict, meta: dict):
        self.models = models
        self.meta = meta

    @property
    def label(self) -> str:
        src = {"sim": "simulated delays on one corridor",
               "live": "real running data"}.get(self.meta.get("source"), "data")
        return f"learned from {src} ({self.meta.get('rows', 0):,} records)"

    def predict(self, df: pd.DataFrame):
        X = df[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        q10, q50, q90 = (self.models[k].predict(X) for k in ("q10", "q50", "q90"))
        q50 = np.maximum(q50, -10.0)
        return np.minimum(q10, q50), q50, np.maximum(q90, q50)


def load(path: Path = MODEL_PATH):
    """The trained model if there is one, else the stated prior."""
    if path.exists():
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return LearnedDelays(blob["models"], blob["meta"])
    return PriorDelays()


# --------------------------------------------------------------------------- #
def fit(rows: pd.DataFrame, source: str, holdout: float = 0.2) -> tuple[LearnedDelays, dict]:
    """Quantile models on the earliest days, scored on the latest."""
    import lightgbm as lgb

    days = np.sort(rows["run_day"].unique())
    cut = days[int(len(days) * (1 - holdout))] if len(days) > 4 else days[-1] + 1
    tr, te = rows[rows["run_day"] < cut], rows[rows["run_day"] >= cut]

    def frame(d):
        X = d[FEATURES].copy()
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        return X

    models = {}
    for name, alpha in QUANTILES.items():
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=400,
                              learning_rate=0.05, num_leaves=63, min_child_samples=40,
                              subsample=0.9, subsample_freq=1, verbose=-1)
        m.fit(frame(tr), tr["delay"], categorical_feature=CATEGORICAL)
        models[name] = m
    model = LearnedDelays(models, {"source": source, "rows": int(len(rows))})

    metrics = {"rows_train": int(len(tr)), "rows_test": int(len(te))}
    if len(te):
        q10, q50, q90 = model.predict(te)
        y = te["delay"].to_numpy()
        prior = PriorDelays().predict(te)
        metrics.update({
            "mae_q50_min": round(float(np.mean(np.abs(q50 - y))), 2),
            "prior_mae_q50_min": round(float(np.mean(np.abs(prior[1] - y))), 2),
            "coverage_80_pct": round(float(100 * np.mean((y >= q10) & (y <= q90))), 1),
            "median_delay_min": round(float(np.median(y)), 1),
        })
    model.meta.update(metrics)
    return model, metrics


def save(model: LearnedDelays, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"models": model.models, "meta": model.meta}, fh)

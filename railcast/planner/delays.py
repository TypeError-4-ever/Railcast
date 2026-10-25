"""
How late a train will be at a station, as a distribution - for trains that have
not started yet.

The corridor forecaster needs live positions and one corridor's block graph.
Planning a trip across India a day ahead needs something cruder that covers
every train: delay at a station given how far into its run the station is, the
class of train, the zone, the time of day and the season.

    PriorDelays     a stated rule, scaled to the 241 arrivals observed so far.
                    What the planner uses until a model has been trained on
                    real running data.
    LearnedDelays   LightGBM quantile regression on running records.
                    `train-delays --source live` trains on what the collector
                    gathered and becomes the planner's default.
                    `train-delays --source sim` trains on simulated runs of one
                    corridor. It exercises the training path, but the simulator's
                    hand-set rules decide how each class of train behaves, so the
                    planner only uses it when asked (`plan --delays sim`).

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

MODEL_PATHS = {"live": Path("models/delays.pkl"), "sim": Path("models/delays_sim.pkl")}

FEATURES = ["klass", "hours_into_run", "zone", "night", "month", "dow", "event"]
CATEGORICAL = ["klass", "zone"]
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

KLASS_ID = {k: i for i, k in enumerate(CLASSES)}
ZONES = ["NR", "NCR", "NER", "NWR", "NFR", "ECR", "ER", "ECoR", "SER", "SECR",
         "CR", "WR", "WCR", "SR", "SCR", "SWR", "KR"]
ZONE_ID = {z: i for i, z in enumerate(ZONES)}
FOG_ZONES = {"NR", "NCR", "NER", "NWR", "ECR", "NFR", "ER"}


def encode(klass: str, zone: str) -> tuple[int, int]:
    return KLASS_ID.get(klass, KLASS_ID["EXPRESS"]), ZONE_ID.get(zone, -1)


def night(minutes: float) -> int:
    """22:00-07:00, when winter fog slows trains. Coarser than the hour on
    purpose: with few trains, the exact hour identifies the train."""
    h = int(minutes // 60) % 24
    return int(h >= 22 or h < 7)


# --------------------------------------------------------------------------- #
class PriorDelays:
    """Delay grows with time into the run, by class; fog doubles it in winter
    on the northern zones.

    Scaled so its overall spread matches the first 241 arrivals the collector
    observed on the Delhi - Mumbai corridor (median about 15 min, P90 about
    57 min, 47% within 15 min - see README). Not fitted per train, zone or
    season: a stated rule with a real anchor, until there is enough running
    data to train LearnedDelays on.
    """

    label = "prior rule scaled to 241 observed arrivals, not fitted per train"
    PER_HOUR = {"RAJDHANI": 0.8, "DURONTO": 0.8, "SHATABDI": 0.7,
                "SUPERFAST": 1.3, "EXPRESS": 1.6, "PASSENGER": 2.0}

    def predict(self, df: pd.DataFrame):
        inv = {v: k for k, v in KLASS_ID.items()}
        rate = np.array([self.PER_HOUR[inv.get(int(k), "EXPRESS")] for k in df["klass"]])
        zinv = {v: k for k, v in ZONE_ID.items()}
        fog = np.array([1.0 + (zinv.get(int(z)) in FOG_ZONES) * (m in (12, 1)) +
                        0.4 * (zinv.get(int(z)) in FOG_ZONES) * (m == 2)
                        for z, m in zip(df["zone"], df["month"])])
        h = np.maximum(df["hours_into_run"].to_numpy(dtype=float), 0.0)
        q50 = rate * h * fog + 2.0 * df["event"].to_numpy()
        q10 = np.maximum(-5.0, 0.2 * q50 - 3.0)
        q90 = 2.5 * q50 + 15.0
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
        # trees extrapolate badly past the longest run they saw: hold it there
        X["hours_into_run"] = X["hours_into_run"].clip(
            upper=self.meta.get("max_hours", np.inf))
        for c in CATEGORICAL:
            X[c] = X[c].astype("category")
        q10, q50, q90 = (self.models[k].predict(X) for k in ("q10", "q50", "q90"))
        q50 = np.maximum(q50, -10.0)                 # trains are rarely early
        q10 = np.clip(q10, -15.0, q50)
        return q10, q50, np.maximum(q90, q50)


def load(which: str = "auto"):
    """auto: the model trained on real data if there is one, else the prior.
    sim / live: that trained model. prior: the stated rule."""
    if which == "prior":
        return PriorDelays()
    path = MODEL_PATHS["live" if which == "auto" else which]
    if path.exists():
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        return LearnedDelays(blob["models"], blob["meta"])
    if which == "auto":
        return PriorDelays()
    raise FileNotFoundError(f"{path} missing - run `python -m railcast train-delays "
                            f"--source {which}` first")


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
        m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=300,
                              learning_rate=0.05, num_leaves=15, min_child_samples=400,
                              subsample=0.9, subsample_freq=1, verbose=-1)
        m.fit(frame(tr), tr["delay"], categorical_feature=CATEGORICAL)
        models[name] = m
    model = LearnedDelays(models, {"source": source, "rows": int(len(rows)),
                                   "max_hours": float(tr["hours_into_run"].max())})

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


def save(model: LearnedDelays) -> Path:
    path = MODEL_PATHS[model.meta["source"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"models": model.models, "meta": model.meta}, fh)
    return path

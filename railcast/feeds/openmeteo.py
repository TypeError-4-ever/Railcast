"""
Real weather, from Open-Meteo. No API key, no registration.

    archive-api.open-meteo.com   ERA5 reanalysis, 1940 to five days ago
    api.open-meteo.com           16-day forecast

The archive does not carry visibility, so fog is derived the way a forecaster
derives it: a small dew-point depression with high humidity through the night.
That index is what the simulator and the feature service consume, so observed
and forecast weather are computed the same way and stay comparable.
"""
from __future__ import annotations

import numpy as np

from .base import WeatherFeed, get_json

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOURLY = ("temperature_2m,relative_humidity_2m,dew_point_2m,"
          "precipitation,wind_speed_10m")
NIGHT = set(range(22, 24)) | set(range(0, 9))


def _fog_index(temp, dew, rh, wind) -> float:
    """0 (clear) to 1 (dense fog likely).

    The fog that stops trains in India is north-Indian winter radiation fog, and
    it needs four things at once: saturated air (small dew-point depression),
    high humidity, a cold surface, and little wind to mix it away. Requiring all
    four is what keeps a humid Mumbai night from scoring as a fog night.
    """
    if temp is None or dew is None or rh is None:
        return 0.0
    t = float(temp)
    close = np.clip((2.5 - (t - float(dew))) / 2.5, 0.0, 1.0)
    humid = np.clip((float(rh) - 80.0) / 15.0, 0.0, 1.0)
    cold = np.clip((18.0 - t) / 8.0, 0.0, 1.0)
    calm = np.clip((12.0 - float(wind or 0.0)) / 8.0, 0.0, 1.0)
    return float(close * humid * cold * calm)


class OpenMeteoFeed(WeatherFeed):
    name = "open-meteo"
    licence = "CC BY 4.0 (Open-Meteo), ERA5 by Copernicus"

    def __init__(self, ttl_hours: float = 24 * 30):
        self.ttl = ttl_hours

    def daily(self, lat: float, lon: float, start: str, end: str) -> dict:
        """Date -> {'rain_mm', 'fog_risk', 'tmin_c'}."""
        url = (f"{ARCHIVE}?latitude={lat:.3f}&longitude={lon:.3f}"
               f"&start_date={start}&end_date={end}"
               f"&hourly={HOURLY}&daily=precipitation_sum,temperature_2m_min"
               f"&timezone=Asia%2FKolkata")
        d = get_json(url, ttl_hours=self.ttl)
        out = {}
        for i, day in enumerate(d["daily"]["time"]):
            out[day] = {
                "rain_mm": float(d["daily"]["precipitation_sum"][i] or 0.0),
                "tmin_c": float(d["daily"]["temperature_2m_min"][i] or 0.0),
                "fog_risk": 0.0,
            }
        h = d["hourly"]
        night: dict[str, list] = {}
        for i, stamp in enumerate(h["time"]):
            day, hh = stamp.split("T")
            if int(hh[:2]) not in NIGHT:
                continue
            night.setdefault(day, []).append(_fog_index(
                h["temperature_2m"][i], h["dew_point_2m"][i],
                h["relative_humidity_2m"][i], h["wind_speed_10m"][i]))
        for day, vals in night.items():
            if day in out and vals:
                out[day]["fog_risk"] = float(np.mean(vals))
        return out

    def forecast(self, lat: float, lon: float, days: int = 7) -> dict:
        url = (f"{FORECAST}?latitude={lat:.3f}&longitude={lon:.3f}"
               f"&hourly={HOURLY}&daily=precipitation_sum,temperature_2m_min"
               f"&forecast_days={days}&timezone=Asia%2FKolkata")
        d = get_json(url, ttl_hours=1.0)
        out = {}
        for i, day in enumerate(d["daily"]["time"]):
            out[day] = {"rain_mm": float(d["daily"]["precipitation_sum"][i] or 0.0),
                        "tmin_c": float(d["daily"]["temperature_2m_min"][i] or 0.0),
                        "fog_risk": 0.0}
        h = d["hourly"]
        night: dict[str, list] = {}
        for i, stamp in enumerate(h["time"]):
            day, hh = stamp.split("T")
            if int(hh[:2]) in NIGHT:
                night.setdefault(day, []).append(_fog_index(
                    h["temperature_2m"][i], h["dew_point_2m"][i],
                    h["relative_humidity_2m"][i], h["wind_speed_10m"][i]))
        for day, vals in night.items():
            if day in out and vals:
                out[day]["fog_risk"] = float(np.mean(vals))
        return out


def speed_factors(obs: dict) -> tuple[float, float]:
    """Observed weather -> (fog multiplier, rain multiplier) on section speed.

    Dense fog forces caution orders and, on non-TPWS sections, a hard speed cap.
    Heavy rain brings watchman patrolling and precautionary restrictions. The
    coefficients below are the prototype's assumption and are the first thing to
    re-fit once real running data has been collected.
    """
    fog = 1.0 - 0.30 * float(obs.get("fog_risk", 0.0))
    rain = float(obs.get("rain_mm", 0.0))
    wet = 1.0 - min(0.12, 0.012 * rain)
    return fog, wet


# --------------------------------------------------------------------------- #
def segment_centroids(cor, feed) -> dict:
    """One representative point per corridor segment, for weather lookup."""
    pts: dict[str, list] = {}
    st = feed.stations()
    for s in cor.stations:
        r = st.get(s.code)
        if r is not None:
            pts.setdefault(s.zone, []).append((r.lat, r.lon))
    return {seg: (float(np.mean([p[0] for p in v])),
                  float(np.mean([p[1] for p in v]))) for seg, v in pts.items()}


def build_environments(cor, static_feed, start_date: str, n_days: int, rng,
                       weather: "OpenMeteoFeed | None" = None) -> dict:
    """Real observed weather for every simulated day, one call per segment.

    Returns {day index -> DayEnvironment}. Speed restrictions stay synthetic:
    TSR and engineering-block circulars are not published as data, so they are
    the one environmental input this prototype still invents.
    """
    from datetime import date, timedelta

    from ..simulator import DayEnvironment

    weather = weather or OpenMeteoFeed()
    d0 = date.fromisoformat(start_date)
    d1 = d0 + timedelta(days=n_days - 1)
    cents = segment_centroids(cor, static_feed)
    obs = {seg: weather.daily(lat, lon, d0.isoformat(), d1.isoformat())
           for seg, (lat, lon) in cents.items()}

    envs = {}
    for d in range(n_days):
        day_iso = (d0 + timedelta(days=d)).isoformat()
        fog, rain = {}, {}
        for seg, series in obs.items():
            rec = series.get(day_iso)
            if not rec:
                continue
            f, w = speed_factors(rec)
            fog[seg], rain[seg] = f, w
        tsr = {}
        for _ in range(int(rng.poisson(3.0))):
            b = int(rng.integers(0, len(cor.blocks)))
            span = int(rng.integers(1, 5))
            cap = float(rng.choice([30.0, 45.0, 60.0, 75.0]))
            for k in range(b, min(b + span, len(cor.blocks))):
                tsr[k] = min(tsr.get(k, 1e9), cap)
        env = DayEnvironment(d, int(day_iso[5:7]), fog, rain, tsr)
        env.date = day_iso
        envs[d] = env
    return envs

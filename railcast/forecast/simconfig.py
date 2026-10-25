"""
Every parameter of the simulator that is *not* measured from data.

The corridor, the roster, the booked times, the sectional speeds and the weather
all come from real sources. What does not, and cannot yet, is how delay is
generated: friction, incident rate, block clearance and how hard a controller
works to clear a path for a premier train. No public historical delay dataset
for Indian Railways exists - NTES serves current state only - so these cannot be
fitted from published data.

Keeping them in one object means two things. They can be listed honestly as
assumptions, and they can be fitted the moment there is something to fit them
against: either running data collected through the live feed, or an explicit
punctuality target (see railcast.calibrate).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_PATH = Path("data/simconfig.json")


@dataclass
class SimConfig:
    # --- running time -------------------------------------------------- #
    friction_scale: float = 0.45
    """Centre of the per-run friction draw, as a fraction of the recovery
    padding the train actually carries. 1.0 would consume all of it."""

    friction_sd: float = 0.025
    """Spread of that draw across days. Sets how variable journeys are."""

    block_noise_sd: float = 0.08
    """Per-block lognormal run-time noise."""

    # --- disruption ------------------------------------------------------ #
    incidents_per_1000km: float = 0.55
    incident_mean_min: float = 12.0
    origin_delay_mean_min: float = 3.6
    dwell_overrun_mean_min: float = 1.2

    # --- capacity and control -------------------------------------------- #
    headway_min: float = 1.5
    """Clearance on top of block occupancy. Block transit already enforces the
    separation, so this is signal and overlap time, not a full headway."""

    precedence_lookahead_min: float = 25.0
    """How far ahead a controller looks when deciding to loop a slower train."""

    precedence_range_km: float = 100.0
    max_hold_min: float = 30.0

    priority_protection: float = 1.0
    """How hard the path of a higher-priority train is cleared. 1.0 is the
    plain rule; above 1.0 loops slower trains earlier and for longer, which is
    what a real control office does for a premier service."""

    # --- provenance ------------------------------------------------------- #
    fitted: bool = False
    fitted_against: str = ""
    fit_residual: dict = field(default_factory=dict)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "SimConfig":
        if path.exists():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def provenance(self) -> dict:
        return {
            "fitted": self.fitted,
            "fitted_against": self.fitted_against or "nothing - defaults in use",
            "residual": self.fit_residual,
            "note": ("These govern how delay is generated. They are the part of "
                     "the simulator that is not measured from published data."),
        }


DEFAULT = SimConfig()

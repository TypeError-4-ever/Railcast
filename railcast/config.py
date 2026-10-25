"""
Every rule and threshold the trip planner uses, in one place.

Booking rules change (the chart moved from 4 h to 8 h before departure in
2025), so nothing below is hard-coded where it is used. Seat-status
probabilities are judgement calls, not fitted values - they are labelled so in
every response until availability history exists to fit them.
"""
from __future__ import annotations

# --- grouping options by the chance of reaching B --------------------------- #
SAFE = 0.90
LIKELY = 0.70

# --- connections ------------------------------------------------------------ #
MIN_LAYOVER_SAME_STATION = 20.0     # minutes, platform change included
MAX_LAYOVER = 8 * 60.0
CROSS_CITY_BASE_MIN = 30.0          # change to another station in the same city:
CROSS_CITY_MIN_PER_KM = 4.0         #   base + per-km travel, clamped below
CROSS_CITY_CLAMP = (45.0, 120.0)

# --- search ------------------------------------------------------------------ #
DETOUR_GEO = 1.6                    # prune change stations far off the A-B line
DETOUR_KM = 1.4                     # final cap on route km vs the direct route
DETOUR_TIME = 1.5                   # final cap on duration vs the fastest option
DETOUR_TIME_FLOOR = 6 * 60.0        #   ... but always allow at least this much extra
DEFAULT_WINDOW_H = 24.0
UNRESERVED_TYPES = {"Pass", "MEMU", "DEMU", "Toy", "Metro", ""}
MAX_CANDIDATES = 200                # itineraries scored for risk
SEAT_CHECKS = 10                    # itineraries whose seats are checked (API cost)

# --- seat status -> probability of travelling on that leg -------------------- #
# Judgement, not fitted. CNF loses a little to selling out before the user
# books; RAC means you may board (shared berth); a waitlisted e-ticket still on
# the waitlist at chart time is cancelled automatically.
P_SEAT = {"CNF": 0.95, "RAC": 1.00, "WL": 0.10, "REGRET": 0.0}
P_SEAT_WL_SHORT = 0.35              # WL 1-5 with more than 24 h to the chart

# --- booking windows ------------------------------------------------------- #
AC_CLASSES = {"1A", "2A", "3A", "3E", "CC", "EC"}
TATKAL_OPEN_AC = (10, 0)            # day before the train's start date
TATKAL_OPEN_NON_AC = (11, 0)
CHART_BEFORE_DEPARTURE_H = 8.0      # at the train's origin; current booking after

# --- delay model --------------------------------------------------------------- #
MONTE_CARLO_DRAWS = 4000

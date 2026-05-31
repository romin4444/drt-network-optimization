"""
DRT planning configuration — single source of truth.
===================================================
Every service standard, cost assumption, and shared geo/time helper lives here
so the scorecard, optimizer, validator and report can't drift apart. Previously
these constants (and the haversine / time-parsing functions) were copy-pasted
into three separate files; a change in one silently disagreed with the others.

Cost and fleet figures are documented assumptions, not magic numbers. Override
them by editing this file or by setting the matching environment variable
(DRT_BUS_CAPITAL_COST, DRT_OPERATING_COST_PER_HR, etc.) — see _envf below.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _envf(name: str, default: float) -> float:
    """Read a float override from the environment, falling back to default."""
    v = os.environ.get(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Service standards (verified against DRT Service Guidelines / TCQSM)
# ---------------------------------------------------------------------------
STANDARDS = {
    "pulse_min_freq": 15,          # PULSE minimum headway (min)
    "base_min_freq": 30,           # Base minimum headway
    "frequent_target": 15,         # Frequent Network target headway
    "frequent_promote_headway_min": 20,  # a B route already <=20min is a 15-min candidate
    "speed_min": 22,               # km/h; below this flags route deviation
    "cov_frequent": 0.21,          # TCQSM LOS A regularity
    "cov_base": 0.30,              # acceptable regularity for base
    "stops_per_km_arterial": 2.5,  # ~400 m spacing
    "stops_per_km_pulse": 2.0,     # ~500 m spacing on rapid corridors
    "coverage_walk_m": 400,        # walk-distance buffer for "covered"
    "on_demand_convert": 8,        # boardings/hr to graduate OnDemand -> fixed
    "stop_consolidation_saving_sec": 20.0,  # dwell+decel time saved per cut stop
    "target_stops_per_km": 2.5,
}

PULSE_ROUTES = {"900", "901", "915", "916"}  # verified roster (+902 being added)

# A route is "lifeline" (protect from on-demand conversion) when this fraction
# of its stops have no other route within the walk buffer. Single source of
# truth so equity.py and the optimizer can't disagree (they used to: 0.5 vs 0.65).
LIFELINE_THRESHOLD = 0.65
PARTIAL_UNIQUE_THRESHOLD = 0.40

# On-time-performance window (seconds relative to schedule)
OTP_EARLY_SEC = -60     # earlier than this = "early" (bad)
OTP_LATE_SEC = 300      # later than this = "late" (bad)


# ---------------------------------------------------------------------------
# Fleet & cost model — turns "add N buses" into dollars and revenue-hours.
# All figures are planning-level estimates; document & tune for a real budget.
# ---------------------------------------------------------------------------
FLEET = {
    # Layover/recovery added to round-trip cycle so PVR is realistic (drivers
    # need recovery time at terminals; 0 recovery understates the fleet).
    "recovery_factor": 0.12,         # +12% of running time, industry typical
    "spare_ratio": 0.20,             # spare buses held against the peak fleet
    "service_span_hr": 18.0,         # ~05:00-23:00 weekday span (all-day buses)
    "peak_span_hr": 6.0,             # ~3h AM + 3h PM (buses added only for peak)
    "annual_service_days": 254,      # weekday-equivalent operating days
    # When a fixed route converts to on-demand, the zone still needs vehicles to
    # run microtransit. We assume it needs this fraction of the route's peak
    # fleet — so freed buses are NOT counted as fully available elsewhere.
    "on_demand_vehicle_ratio": 0.5,
}

COST = {
    # 40-ft conventional bus, capital amortised over its service life.
    "bus_capital": _envf("DRT_BUS_CAPITAL_COST", 750_000.0),   # CAD per bus
    "bus_life_years": 12,
    # Fully-loaded operating cost per revenue vehicle-hour (driver, fuel, maint).
    "operating_per_rev_hr": _envf("DRT_OPERATING_COST_PER_HR", 135.0),  # CAD/hr
    # On-demand microtransit unit economics (per the freed-vehicle trade-off).
    "on_demand_per_hr": _envf("DRT_ON_DEMAND_COST_PER_HR", 95.0),       # CAD/hr
}


def annualized_bus_capital() -> float:
    """Per-bus capital cost spread over its service life (straight-line)."""
    return COST["bus_capital"] / COST["bus_life_years"]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
GTFS_DIR = ROOT / "drt" / "gtfs"
MAP_DATA = ROOT / "drt" / "map_data"
BOARDING_CSV = ROOT / "TRNST_Bus_Boarding_Points.csv"


# ---------------------------------------------------------------------------
# Shared geo / time helpers (were duplicated in every script)
# ---------------------------------------------------------------------------
EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Works on scalars or numpy arrays."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def shape_length_km(pts) -> float:
    """Summed great-circle length of an ordered [(lat,lon),...] polyline."""
    return float(sum(
        haversine_km(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        for i in range(len(pts) - 1)
    ))


def t_to_sec(t):
    """Parse GTFS HH:MM:SS (may exceed 24:00:00) to seconds; NaN if blank."""
    if t is None or (isinstance(t, float) and math.isnan(t)) or str(t).strip() == "":
        return np.nan
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def current_weekday_services(calendar: pd.DataFrame, target_date=None) -> list[str]:
    """Weekday service IDs that are actually *active on a given date*.

    GTFS ships several sequential schedule versions; summing trips across all of
    them double-counts service that never runs concurrently. The correct filter
    is "services whose [start_date, end_date] window brackets the target date" —
    NOT merely "the most recent start_date", which can select a future schedule
    period that the agency has pre-loaded but isn't running yet.

    `target_date` accepts a date/datetime/'YYYY-MM-DD'/'YYYYMMDD' (default: today).
    Robust to whether calendar was read with str or int dtypes.

    Note: this selects the regular weekly pattern. Holiday add/remove exceptions
    live in calendar_dates.txt and are applied per-day in the schedule indexer
    (_service_days), not here.
    """
    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(int)
    cal["end_date"] = cal["end_date"].astype(int)
    dow = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    # astype(str).eq("1") normalises 1 (int) and "1" (str) identically.
    weekday = cal[dow].astype(str).eq("1").any(axis=1)

    if target_date is None:
        target = int(pd.Timestamp.today().strftime("%Y%m%d"))
    else:
        target = int(pd.Timestamp(str(target_date)).strftime("%Y%m%d"))

    active = (cal["start_date"] <= target) & (cal["end_date"] >= target)
    wd = cal[weekday & active]
    if wd.empty:
        # Target falls outside every window (e.g. analysing an archived feed):
        # fall back to the most recent weekday period so we still return something.
        wk = cal[weekday]
        if wk.empty:
            return []
        latest = wk["start_date"].max()
        wd = wk[wk["start_date"] == latest]
    return wd["service_id"].astype(str).tolist()


def route_family(route_id) -> str:
    """Map a route_id to its Durham municipality / service family."""
    r = str(route_id)
    if r.startswith("N"):
        return "night"
    d = r[0] if r and r[0].isdigit() else "?"
    return {
        "1": "Pickering", "2": "Ajax", "3": "Whitby", "4": "Oshawa",
        "5": "Clarington", "6": "Rural", "9": "PULSE/Regional",
    }.get(d, "other")

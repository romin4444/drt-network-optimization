"""
================================================================================
DRT (Durham Region Transit) Bus Efficiency Pipeline — SINGLE-FILE VERSION
================================================================================
End-to-end: schedule indexing, GTFS-RT logging, feature engineering, and
LightGBM on-time-performance modelling, in one runnable file.

------------------------------ COLAB DATA PATHS --------------------------------
This script auto-detects Colab. On Colab the data root is:

    /content/drt/                          <-- DATA_DIR
        GTFS_Durham_TXT.zip                <-- you upload this
        gtfs/                              <-- script unzips here
        schedule_index/                    <-- script writes Parquet per day
        rt_log/                            <-- logger writes Parquet per day
        features/                          <-- feature_engineering writes here
        baseline_report.csv                <-- per-route diagnostics

HOW TO PUT YOUR DATA IN COLAB (pick one):

  Option A — Upload manually:
    1. Click the folder icon (left sidebar) → upload icon
    2. Upload GTFS_Durham_TXT.zip into /content/
    3. Run a cell:   !mkdir -p /content/drt && mv /content/GTFS_Durham_TXT.zip /content/drt/

  Option B — Download fresh from DRT:
    !mkdir -p /content/drt
    !wget -O /content/drt/GTFS_Durham_TXT.zip \
        https://maps.durham.ca/OpenDataGTFS/GTFS_Durham_TXT.zip

  Option C — Persist across sessions via Google Drive:
    from google.colab import drive
    drive.mount('/content/drive')
    # Then change DATA_DIR below to /content/drive/MyDrive/drt
--------------------------------------------------------------------------------

USAGE (Colab cell):
    !python drt_pipeline.py            # run the full schedule-only pipeline
    # OR call functions directly in a notebook:
    #   colab_setup(); extract_gtfs(); build_schedule_index(); baseline_analysis()
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import math
import zipfile
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ================================================================================
# CONFIG — change DATA_DIR if you mounted Google Drive
# ================================================================================
IS_COLAB = "google.colab" in sys.modules
DATA_DIR = Path("/content/drt" if IS_COLAB else "./drt")

GTFS_ZIP   = DATA_DIR / "GTFS_Durham_TXT.zip"
GTFS_DIR   = DATA_DIR / "gtfs"
INDEX_DIR  = DATA_DIR / "schedule_index"
RT_LOG_DIR = DATA_DIR / "rt_log"
FEAT_DIR   = DATA_DIR / "features"
REPORT_CSV = DATA_DIR / "baseline_report.csv"

# GTFS-RT feed URLs — get the actual .pb URLs from:
#   https://opendata.durham.ca → "GTFS-RT Vehicle Positions"
VEHICLE_POSITIONS_URL = "https://maps.durham.ca/OpenDataGTFS/VehiclePositions.pb"
TRIP_UPDATES_URL      = "https://maps.durham.ca/OpenDataGTFS/TripUpdates.pb"

# Static schedule URL (used by extract_gtfs if zip not present)
STATIC_GTFS_URL = "https://maps.durham.ca/OpenDataGTFS/GTFS_Durham_TXT.zip"

TZ = ZoneInfo("America/Toronto")


# ================================================================================
# COLAB SETUP — installs deps, creates dirs
# ================================================================================
def colab_setup():
    """Install dependencies and ensure all working directories exist."""
    if IS_COLAB:
        # Quiet install; rerun is fast because pip caches.
        os.system("pip install -q gtfs-realtime-bindings pyarrow lightgbm shap requests")
    for d in (DATA_DIR, GTFS_DIR, INDEX_DIR, RT_LOG_DIR, FEAT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Environment: {'Colab' if IS_COLAB else 'local'}")
    print(f"Data root:    {DATA_DIR}")
    print(f"GTFS zip:     {GTFS_ZIP} (exists: {GTFS_ZIP.exists()})")


# ================================================================================
# HELPERS
# ================================================================================
def t_to_sec(t: str):
    """Parse HH:MM:SS (which may exceed 24:00:00 in GTFS) to seconds since midnight.

    Returns NaN for blank/missing values instead of raising, so feeds with
    empty arrival/departure times (common for non-timepoint stops) don't crash
    the pipeline.
    """
    if t is None or (isinstance(t, float) and math.isnan(t)) or str(t).strip() == "":
        return np.nan
    h, m, s = map(int, str(t).split(":"))
    return h * 3600 + m * 60 + s


def current_weekday_services(calendar: pd.DataFrame) -> list[str]:
    """Service IDs for the *current* weekday schedule period.

    GTFS feeds ship several sequential schedule versions (e.g. an expiring
    period and the upcoming one). Summing trips across all of them double-counts
    service that never runs concurrently. We keep only services whose start_date
    is the most recent and that operate on at least one weekday (this also picks
    up the overnight service, which covers weekday nights).
    """
    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(int)
    weekday_mask = cal[["monday", "tuesday", "wednesday", "thursday", "friday"]].eq(1).any(axis=1)
    wk = cal[weekday_mask]
    if wk.empty:
        return []
    latest = wk["start_date"].max()
    return wk[wk["start_date"] == latest]["service_id"].astype(str).tolist()


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def route_family(route_id) -> str:
    r = str(route_id)
    if r.startswith("N"):
        return "night"
    d = r[0] if r and r[0].isdigit() else "?"
    return {
        "1": "Pickering", "2": "Ajax", "3": "Whitby", "4": "Oshawa",
        "5": "Clarington", "6": "Rural", "9": "PULSE/Regional",
    }.get(d, "other")


# ================================================================================
# STEP 1 — Extract / download the static GTFS
# ================================================================================
def extract_gtfs():
    """Unzip the static GTFS to GTFS_DIR. Downloads it if not present."""
    if not GTFS_ZIP.exists():
        print(f"GTFS zip not found at {GTFS_ZIP}; attempting download...")
        try:
            import requests
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            r = requests.get(STATIC_GTFS_URL, timeout=60)
            r.raise_for_status()
            GTFS_ZIP.write_bytes(r.content)
            print(f"Downloaded {len(r.content):,} bytes")
        except Exception as e:
            raise FileNotFoundError(
                f"Could not get GTFS zip. Upload it to {GTFS_ZIP} manually.\n"
                f"Download failed with: {e}"
            )
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GTFS_ZIP) as zf:
        zf.extractall(GTFS_DIR)
    files = sorted(GTFS_DIR.glob("*.txt"))
    print(f"Extracted {len(files)} files to {GTFS_DIR}")
    return files


# ================================================================================
# STEP 2 — Build the per-day schedule index (absolute UTC timestamps)
# ================================================================================
def _service_days(calendar: pd.DataFrame, calendar_dates: pd.DataFrame) -> dict:
    """For each service_id, return the list of dates it actually operates."""
    out: dict = {}
    dow = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for _, row in calendar.iterrows():
        sid = row["service_id"]
        start = datetime.strptime(str(row["start_date"]), "%Y%m%d").date()
        end   = datetime.strptime(str(row["end_date"]),   "%Y%m%d").date()
        days = []
        d = start
        while d <= end:
            if row[dow[d.weekday()]] == 1:
                days.append(d)
            d += timedelta(days=1)
        out[sid] = days
    for _, row in calendar_dates.iterrows():
        sid = row["service_id"]
        d = datetime.strptime(str(row["date"]), "%Y%m%d").date()
        if row["exception_type"] == 1 and sid in out and d not in out[sid]:
            out[sid].append(d)
        elif row["exception_type"] == 2 and sid in out and d in out[sid]:
            out[sid].remove(d)
    return out


def build_schedule_index():
    """Expand stop_times.txt × calendar into one Parquet file per service-date."""
    trips          = pd.read_csv(GTFS_DIR / "trips.txt")
    stop_times     = pd.read_csv(GTFS_DIR / "stop_times.txt")
    calendar       = pd.read_csv(GTFS_DIR / "calendar.txt")
    calendar_dates = pd.read_csv(GTFS_DIR / "calendar_dates.txt")

    stop_times["arr_sec"] = stop_times["arrival_time"].apply(t_to_sec)
    stop_times["dep_sec"] = stop_times["departure_time"].apply(t_to_sec)
    st = stop_times.merge(
        trips[["trip_id", "service_id", "route_id", "direction_id", "shape_id"]],
        on="trip_id", how="left",
    )
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    days_map = _service_days(calendar, calendar_dates)

    # Invert the service->days map into days->services so that every service
    # active on a given date is written into that date's file. Writing per
    # service would overwrite the file each time and keep only the last service
    # (e.g. losing all weekday trips behind the overnight service).
    date_services: dict = {}
    for sid, day_list in days_map.items():
        for d in day_list:
            date_services.setdefault(d, []).append(sid)

    keep = ["service_date", "route_id", "trip_id", "stop_id", "stop_sequence",
            "direction_id", "scheduled_arr_unix", "scheduled_dep_unix"]
    total = 0
    for d, sids in date_services.items():
        subset = st[st["service_id"].isin(sids)]
        if subset.empty:
            continue
        midnight = datetime(d.year, d.month, d.day, tzinfo=TZ)
        mid_unix = int(midnight.timestamp())
        day_df = subset.copy()
        day_df["service_date"]       = d.isoformat()
        day_df["scheduled_arr_unix"] = mid_unix + day_df["arr_sec"]
        day_df["scheduled_dep_unix"] = mid_unix + day_df["dep_sec"]
        day_df[keep].to_parquet(
            INDEX_DIR / f"date={d.isoformat()}.parquet",
            index=False, compression="snappy",
        )
        total += len(day_df)
    n_files = len(list(INDEX_DIR.glob("*.parquet")))
    print(f"Schedule index: {n_files} files, {total:,} scheduled stop-arrivals total")
    return total


# ================================================================================
# STEP 3 — Baseline schedule-only analysis
# ================================================================================
def baseline_analysis(show: bool = True) -> pd.DataFrame:
    """Per-route diagnostics. Uses only the static GTFS; no RT data required."""
    routes     = pd.read_csv(GTFS_DIR / "routes.txt")
    trips      = pd.read_csv(GTFS_DIR / "trips.txt")
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt")
    calendar   = pd.read_csv(GTFS_DIR / "calendar.txt")
    shapes     = pd.read_csv(GTFS_DIR / "shapes.txt")

    # Trip duration in seconds
    stop_times["arr_sec"] = stop_times["arrival_time"].apply(t_to_sec)

    # Compute shape distance via haversine summation
    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)
    shapes["lat2"] = shapes.groupby("shape_id")["shape_pt_lat"].shift(-1)
    shapes["lon2"] = shapes.groupby("shape_id")["shape_pt_lon"].shift(-1)
    shapes["seg_km"] = haversine_km(
        shapes["shape_pt_lat"], shapes["shape_pt_lon"], shapes["lat2"], shapes["lon2"],
    )
    shape_dist = shapes.groupby("shape_id")["seg_km"].sum().rename("distance_km").reset_index()

    trip_spans = (
        stop_times.groupby("trip_id")
        .agg(start_sec=("arr_sec", "min"), end_sec=("arr_sec", "max"),
             n_stops=("stop_id", "count"))
        .reset_index()
    )
    trip_spans["duration_min"] = (trip_spans["end_sec"] - trip_spans["start_sec"]) / 60
    trip_spans = trip_spans.merge(
        trips[["trip_id", "route_id", "service_id", "direction_id", "shape_id"]], on="trip_id",
    ).merge(shape_dist, on="shape_id", how="left")
    trip_spans["speed_kmh"] = trip_spans["distance_km"] / (trip_spans["duration_min"] / 60)

    weekday_sids = current_weekday_services(calendar)
    wk = trip_spans[trip_spans["service_id"].astype(str).isin(weekday_sids)].copy()

    agg = (
        wk.groupby("route_id")
        .agg(weekday_trips=("trip_id", "count"),
             avg_duration_min=("duration_min", "mean"),
             avg_distance_km=("distance_km", "mean"),
             avg_n_stops=("n_stops", "mean"),
             avg_speed_kmh=("speed_kmh", "mean"))
        .round(2).reset_index()
    )
    agg["stops_per_km"]          = (agg["avg_n_stops"] / agg["avg_distance_km"]).round(2)
    agg["weekday_service_hours"] = (agg["weekday_trips"] * agg["avg_duration_min"] / 60).round(1)

    # Headway regularity in the 06:00-21:00 window
    hw_rows = []
    for r in agg["route_id"]:
        sub = wk[wk["route_id"] == r].sort_values(["direction_id", "start_sec"])
        for dirn, g in sub.groupby("direction_id"):
            starts = g["start_sec"].values
            s = starts[(starts >= 6 * 3600) & (starts <= 21 * 3600)]
            if len(s) < 3:
                continue
            gaps = np.diff(np.sort(s)) / 60
            hw_rows.append({
                "route_id": r, "direction_id": dirn,
                "median_headway_min": float(np.median(gaps)),
                "headway_cov": float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else None,
            })
    hw = pd.DataFrame(hw_rows)
    if not hw.empty:
        hw_summary = hw.groupby("route_id").agg(
            median_headway_min=("median_headway_min", "mean"),
            headway_cov=("headway_cov", "mean"),
        ).round(2)
        agg = agg.merge(hw_summary, on="route_id", how="left")

    # Peak share
    def peak_share(r):
        sub = wk[wk["route_id"] == r]
        if sub.empty:
            return None
        h = sub["start_sec"] / 3600
        return round((((h >= 7) & (h < 10)) | ((h >= 16) & (h < 19))).mean(), 3)
    agg["peak_trip_share"] = agg["route_id"].map(peak_share)

    agg["region"] = agg["route_id"].astype(str).map(route_family)
    agg = agg.sort_values("weekday_service_hours", ascending=False).reset_index(drop=True)
    agg.to_csv(REPORT_CSV, index=False)

    if show:
        print("=" * 78)
        print("BASELINE SCHEDULE ANALYSIS")
        print("=" * 78)
        print(f"  Routes:               {agg['route_id'].nunique()}")
        print(f"  Weekday trips:        {len(wk):,}")
        print(f"  Service-hours/weekday:{agg['weekday_service_hours'].sum():.0f}")
        print(f"  Median speed:         {agg['avg_speed_kmh'].median():.1f} km/h")
        print("\n--- Top 10 routes by weekday service hours ---")
        cols = ["route_id", "region", "weekday_trips", "weekday_service_hours",
                "avg_speed_kmh", "median_headway_min", "headway_cov", "peak_trip_share"]
        print(agg.head(10)[cols].to_string(index=False))
        print(f"\nFull report: {REPORT_CSV}")
    return agg


# ================================================================================
# STEP 4 — GTFS-RT logger (vehicle positions + trip updates)
#         NOTE: Colab sessions are ephemeral (~12 h max, free tier idles sooner).
#               For multi-week collection, run this on a Raspberry Pi or a $5 VPS,
#               not Colab. Use duration_min for a bounded test run in Colab.
# ================================================================================
def _fetch_rt(url, timeout=15):
    import requests
    from google.transit import gtfs_realtime_pb2
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        m = gtfs_realtime_pb2.FeedMessage()
        m.ParseFromString(r.content)
        return m
    except Exception as e:
        print(f"  fetch failed for {url}: {e}")
        return None


def _vp_to_rows(msg):
    poll_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for entity in msg.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        rows.append({
            "poll_ts": poll_ts, "feed_ts": msg.header.timestamp,
            "trip_id": v.trip.trip_id if v.HasField("trip") else None,
            "route_id": v.trip.route_id if v.HasField("trip") else None,
            "direction_id": v.trip.direction_id if v.HasField("trip") else None,
            "start_date": v.trip.start_date if v.HasField("trip") else None,
            "vehicle_id": v.vehicle.id if v.HasField("vehicle") else None,
            "latitude": v.position.latitude if v.HasField("position") else None,
            "longitude": v.position.longitude if v.HasField("position") else None,
            "bearing": v.position.bearing if v.HasField("position") else None,
            "speed": v.position.speed if v.HasField("position") else None,
            "stop_id": v.stop_id if v.HasField("stop_id") else None,
            "current_status": v.current_status if v.HasField("current_status") else None,
            "current_stop_seq": v.current_stop_sequence if v.HasField("current_stop_sequence") else None,
            "vehicle_ts": v.timestamp if v.HasField("timestamp") else None,
            "occupancy": v.occupancy_status if v.HasField("occupancy_status") else None,
        })
    return rows


def _tu_to_rows(msg):
    poll_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for entity in msg.entity:
        if not entity.HasField("trip_update"):
            continue
        tu = entity.trip_update
        for stu in tu.stop_time_update:
            rows.append({
                "poll_ts": poll_ts, "feed_ts": msg.header.timestamp,
                "trip_id": tu.trip.trip_id, "route_id": tu.trip.route_id,
                "start_date": tu.trip.start_date,
                "vehicle_id": tu.vehicle.id if tu.HasField("vehicle") else None,
                "stop_id": stu.stop_id, "stop_sequence": stu.stop_sequence,
                "arrival_time":  stu.arrival.time  if stu.HasField("arrival")   else None,
                "arrival_delay": stu.arrival.delay if stu.HasField("arrival")   else None,
                "departure_time":  stu.departure.time  if stu.HasField("departure") else None,
                "departure_delay": stu.departure.delay if stu.HasField("departure") else None,
                "schedule_relationship": stu.schedule_relationship,
            })
    return rows


def _append_parquet(rows, kind):
    if not rows:
        return
    df = pd.DataFrame(rows)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = RT_LOG_DIR / f"date={today}" / f"{kind}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False, compression="snappy")


def run_logger(interval: int = 20, duration_min: int | None = None,
               skip_trip_updates: bool = False):
    """Poll the GTFS-RT feed and write Parquet partitions.

    Args:
        interval: poll interval in seconds (DRT updates every ~15-30 s)
        duration_min: stop after this many minutes (None = run forever)
        skip_trip_updates: set True if you only want vehicle positions
    """
    RT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"Logging to {RT_LOG_DIR}, interval={interval}s, duration={duration_min}min")
    while True:
        try:
            vp = _fetch_rt(VEHICLE_POSITIONS_URL)
            if vp:
                rows = _vp_to_rows(vp)
                _append_parquet(rows, "vehicle_positions")
                print(f"[{datetime.now():%H:%M:%S}] VP: {len(rows)} vehicles")
            if not skip_trip_updates:
                tu = _fetch_rt(TRIP_UPDATES_URL)
                if tu:
                    rows = _tu_to_rows(tu)
                    _append_parquet(rows, "trip_updates")
                    print(f"[{datetime.now():%H:%M:%S}] TU: {len(rows)} stop-updates")
            if duration_min and (time.time() - start) > duration_min * 60:
                print("duration elapsed, exiting")
                return
            time.sleep(interval)
        except KeyboardInterrupt:
            print("interrupted, exiting")
            return


# ================================================================================
# STEP 5 — Feature engineering: join RT actuals with the schedule index
# ================================================================================
STOPPED_AT = 1  # VehiclePosition.current_status enum


def _detect_arrivals_from_vp(vp: pd.DataFrame) -> pd.DataFrame:
    if vp.empty:
        return pd.DataFrame()
    vp = vp.copy()
    vp["t"] = vp["vehicle_ts"].fillna(vp["feed_ts"])
    vp = vp.dropna(subset=["trip_id", "stop_id", "t"])
    vp = vp.sort_values(["trip_id", "vehicle_id", "t"])
    at_stop = vp[vp["current_status"] == STOPPED_AT]
    arrivals = (
        at_stop.groupby(["trip_id", "stop_id", "current_stop_seq"], as_index=False)
        .agg(actual_arr_unix=("t", "min"),
             route_id=("route_id", "first"),
             vehicle_id=("vehicle_id", "first"))
        .rename(columns={"current_stop_seq": "stop_sequence"})
    )
    arrivals["stop_sequence"] = arrivals["stop_sequence"].astype("Int64")
    return arrivals


def _actuals_from_tu(tu: pd.DataFrame) -> pd.DataFrame:
    if tu.empty:
        return pd.DataFrame()
    tu = tu.copy().sort_values(["trip_id", "stop_id", "poll_ts"])
    return (
        tu.dropna(subset=["arrival_time"])
        .groupby(["trip_id", "stop_id", "stop_sequence"], as_index=False)
        .agg(actual_arr_unix=("arrival_time", "last"),
             arrival_delay_reported=("arrival_delay", "last"),
             route_id=("route_id", "first"))
    )


def build_features(date_str: str) -> int:
    """Join the RT log for one service-date with the schedule index, write features."""
    sched_path = INDEX_DIR / f"date={date_str}.parquet"
    if not sched_path.exists():
        print(f"no schedule index for {date_str}")
        return 0
    sched = pd.read_parquet(sched_path)

    vp_path = RT_LOG_DIR / f"date={date_str}" / "vehicle_positions.parquet"
    tu_path = RT_LOG_DIR / f"date={date_str}" / "trip_updates.parquet"
    actuals_vp = _detect_arrivals_from_vp(pd.read_parquet(vp_path)) if vp_path.exists() else pd.DataFrame()
    actuals_tu = _actuals_from_tu(pd.read_parquet(tu_path)) if tu_path.exists() else pd.DataFrame()
    if not actuals_tu.empty:
        actuals = actuals_tu
        if not actuals_vp.empty:
            extras = actuals_vp.merge(
                actuals_tu[["trip_id", "stop_id"]], on=["trip_id", "stop_id"],
                how="left", indicator=True,
            ).query("_merge == 'left_only'").drop(columns="_merge")
            actuals = pd.concat([actuals, extras], ignore_index=True)
    else:
        actuals = actuals_vp
    if actuals.empty:
        print(f"no actuals on {date_str}")
        return 0

    joined = sched.merge(
        actuals[["trip_id", "stop_id", "stop_sequence", "actual_arr_unix", "vehicle_id"]],
        on=["trip_id", "stop_id", "stop_sequence"], how="inner",
    )
    joined["delay_sec"] = joined["actual_arr_unix"] - joined["scheduled_arr_unix"]
    joined["on_time"]     = ((joined["delay_sec"] >= -60) & (joined["delay_sec"] <= 300)).astype(int)
    joined["late_label"]  = (joined["delay_sec"] >  300).astype(int)
    joined["early_label"] = (joined["delay_sec"] < -60 ).astype(int)

    ts = pd.to_datetime(joined["scheduled_arr_unix"], unit="s", utc=True).dt.tz_convert("America/Toronto")
    joined["hour"]       = ts.dt.hour
    joined["minute"]     = ts.dt.minute
    joined["weekday"]    = ts.dt.weekday
    joined["is_weekend"] = (joined["weekday"] >= 5).astype(int)
    joined["is_peak_am"] = ts.dt.hour.between(7, 9).astype(int)
    joined["is_peak_pm"] = ts.dt.hour.between(16, 18).astype(int)

    trip_size = joined.groupby("trip_id")["stop_sequence"].transform("max")
    joined["frac_of_trip"] = joined["stop_sequence"] / trip_size
    joined = joined.sort_values(["trip_id", "stop_sequence"])
    joined["upstream_delay_sec"] = joined.groupby("trip_id")["delay_sec"].shift(1).fillna(0)
    joined["route_family"] = joined["route_id"].astype(str).map(route_family)

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"date={date_str}.parquet"
    joined.to_parquet(out, index=False, compression="snappy")
    print(f"{date_str}: {len(joined):,} rows, OTP={joined['on_time'].mean():.1%}")
    return len(joined)


# ================================================================================
# STEP 6 — LightGBM on-time-performance model
# ================================================================================
def train_otp_model(date_strs: list[str] | None = None, test_size: float = 0.2):
    """Train an OTP classifier and evaluate it honestly against baselines.

    Methodology notes (these are what make the numbers trustworthy):
      * Three-way split (train/val/test). The validation set drives early
        stopping; the test set is touched ONCE for final metrics. Using the
        test set for early stopping (as a naive setup does) leaks it into model
        selection and inflates the score.
      * Splits are GROUPED BY trip_id so consecutive stops of one trip never
        straddle a boundary — otherwise `upstream_delay_sec` leaks the label.
      * When >= 3 service-days are available the split is TEMPORAL (train on the
        earliest days, test on the latest) which is the only honest protocol for
        a forecasting task. With fewer days we fall back to a grouped split and
        say so loudly.
      * We report against two baselines (majority-class and route×hour mean) so
        AUC/accuracy are interpretable. On a ~80% on-time base rate, "always
        predict on-time" already scores 0.80 accuracy; the model must beat that.

    Returns (model, feature_cols, metrics_dict). metrics_dict is JSON-friendly.
    """
    import json

    import lightgbm as lgb
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score, f1_score,
        confusion_matrix, classification_report,
    )

    if date_strs is None:
        files = sorted(FEAT_DIR.glob("date=*.parquet"))
    else:
        files = [FEAT_DIR / f"date={d}.parquet" for d in date_strs]
        files = [f for f in files if f.exists()]
    if not files:
        print("no feature parquet files found — log some RT data first, then run build_features()")
        return None, None, None

    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d["_date"] = f.stem.replace("date=", "")
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    n_days = df["_date"].nunique()
    print(f"Training on {len(df):,} rows from {n_days} day(s)")
    if n_days < 5:
        print("  WARNING: < 5 days of data. Metrics below are indicative only and "
              "WILL overfit — collect several weeks before trusting them.")

    cat_cols = ["route_family", "route_id", "direction_id"]
    num_cols = ["hour", "minute", "weekday", "is_weekend", "is_peak_am", "is_peak_pm",
                "stop_sequence", "frac_of_trip", "upstream_delay_sec"]
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
    y = df["on_time"].astype(int)

    # ---- split: temporal when we have enough days, else grouped-by-trip ----
    def grouped_split(idx, frac):
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=42)
        a, b = next(gss.split(X.iloc[idx], y.iloc[idx], groups=df["trip_id"].iloc[idx]))
        return idx[a], idx[b]

    all_idx = np.arange(len(df))
    if n_days >= 3:
        protocol = "temporal (train=early days, test=latest day)"
        days = sorted(df["_date"].unique())
        test_days = {days[-1]}
        val_days = {days[-2]}
        te_idx = all_idx[df["_date"].isin(test_days).to_numpy()]
        va_idx = all_idx[df["_date"].isin(val_days).to_numpy()]
        tr_idx = all_idx[(~df["_date"].isin(test_days | val_days)).to_numpy()]
    else:
        protocol = "grouped-by-trip (insufficient days for a temporal split)"
        tr_full, te_idx = grouped_split(all_idx, test_size)
        tr_idx, va_idx = grouped_split(tr_full, 0.2)

    print(f"  Split protocol: {protocol}")
    print(f"  train={len(tr_idx):,}  val={len(va_idx):,}  test={len(te_idx):,}")

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    # ---- baselines (so the model's numbers mean something) ----
    base_rate = float(y_tr.mean())                       # P(on_time) in train
    majority = int(base_rate >= 0.5)
    base_acc = accuracy_score(y_te, np.full(len(y_te), majority))
    # route x hour historical on-time rate, learned on train, scored on test
    rh = df.iloc[tr_idx].groupby(["route_id", "hour"])["on_time"].mean()
    global_rate = base_rate
    rh_pred = df.iloc[te_idx].apply(
        lambda r: rh.get((r["route_id"], r["hour"]), global_rate), axis=1).to_numpy()
    rh_auc = roc_auc_score(y_te, rh_pred) if y_te.nunique() > 1 else float("nan")

    # ---- model ----
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],   # early stop on VAL, not test
              categorical_feature=cat_cols,
              callbacks=[lgb.early_stopping(30, verbose=False)])

    proba = model.predict_proba(X_te)[:, 1]
    labels = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba) if y_te.nunique() > 1 else float("nan")
    pr_auc = average_precision_score(y_te, proba) if y_te.nunique() > 1 else float("nan")
    acc = accuracy_score(y_te, labels)
    f1 = f1_score(y_te, labels, zero_division=0)
    cm = confusion_matrix(y_te, labels).tolist()

    print("\n--- TEST-SET RESULTS (touched once) ---")
    print(f"  Model      : AUC={auc:.3f}  PR-AUC={pr_auc:.3f}  acc={acc:.3f}  f1={f1:.3f}")
    print(f"  Baseline 1 : majority-class acc={base_acc:.3f}  (on-time base rate={base_rate:.3f})")
    print(f"  Baseline 2 : route-x-hour mean AUC={rh_auc:.3f}")
    lift = acc - base_acc
    print(f"  Lift over majority baseline: {lift:+.3f} accuracy")
    print(f"  Confusion matrix [[TN,FP],[FN,TP]]: {cm}")
    print("\n" + classification_report(y_te, labels, digits=3, zero_division=0))

    model_path = DATA_DIR / "otp_model.txt"
    model.booster_.save_model(str(model_path))
    metrics = {
        "n_rows": int(len(df)), "n_days": int(n_days), "protocol": protocol,
        "test_auc": float(auc), "test_pr_auc": float(pr_auc),
        "test_accuracy": float(acc), "test_f1": float(f1),
        "baseline_majority_accuracy": float(base_acc),
        "baseline_routehour_auc": float(rh_auc),
        "on_time_base_rate": float(base_rate),
        "confusion_matrix": cm,
    }
    (DATA_DIR / "otp_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nModel saved to {model_path}")
    print(f"Metrics saved to {DATA_DIR / 'otp_metrics.json'}")
    return model, feature_cols, metrics


def explain_model(model, feature_cols, sample_size: int = 5000):
    """Generate SHAP values to rank features by impact on delay."""
    import shap
    files = sorted(FEAT_DIR.glob("date=*.parquet"))
    if not files:
        print("no features to explain")
        return
    df = pd.concat([pd.read_parquet(f) for f in files[-3:]], ignore_index=True)
    cat_cols = ["route_family", "route_id", "direction_id"]
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    X = X.sample(min(sample_size, len(X)), random_state=42)
    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X)
    mean_abs   = pd.Series(np.abs(shap_vals).mean(axis=0), index=X.columns)
    print("\nTop 15 features by mean |SHAP value|:")
    print(mean_abs.sort_values(ascending=False).head(15).to_string())
    return mean_abs


# ================================================================================
# MAIN — orchestrate the schedule-only portion (Steps 1-3) end to end
# ================================================================================
def main():
    print("=" * 78)
    print("DRT Bus Efficiency Pipeline")
    print("=" * 78)
    colab_setup()
    if not list(GTFS_DIR.glob("*.txt")):
        extract_gtfs()
    build_schedule_index()
    baseline_analysis(show=True)
    print("\n" + "=" * 78)
    print("NEXT STEPS")
    print("=" * 78)
    print(" 1. Paste the real .pb URLs (from opendata.durham.ca) into")
    print("    VEHICLE_POSITIONS_URL and TRIP_UPDATES_URL at the top of this file.")
    print(" 2. Start the logger on a long-running machine (Pi/VPS, not Colab):")
    print("       python drt_pipeline.py --logger --interval 20")
    print(" 3. After 7+ days, build features and train the model:")
    print("       python drt_pipeline.py --features 2026-05-21")
    print("       python drt_pipeline.py --train")


# ================================================================================
# CLI entry — works as a plain script too
# ================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRT Bus Efficiency Pipeline")
    parser.add_argument("--logger", action="store_true",
                        help="run the GTFS-RT logger instead of the schedule analysis")
    parser.add_argument("--interval", type=int, default=20,
                        help="logger poll interval (seconds)")
    parser.add_argument("--duration-min", type=int, default=None,
                        help="logger run length in minutes (None = forever)")
    parser.add_argument("--features", type=str, default=None,
                        help="build features for YYYY-MM-DD")
    parser.add_argument("--train", action="store_true",
                        help="train the LightGBM OTP model")
    args = parser.parse_args()

    colab_setup()
    if args.logger:
        run_logger(interval=args.interval, duration_min=args.duration_min)
    elif args.features:
        build_features(args.features)
    elif args.train:
        train_otp_model()
    else:
        main()

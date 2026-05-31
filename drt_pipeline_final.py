from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# =============================================================================
# DRT Bus Efficiency Pipeline — final single-file version
#   - Static GTFS download/extraction
#   - Schedule index generation
#   - Baseline route diagnostics + chart
#   - GTFS-RT logger
#   - Feature engineering
#   - OTP model training scaffold
#   - Optional enrichment with uploaded TRNST stop/route GIS CSVs
#
# Colab data root:
#   /content/drt/
#     GTFS_Durham_TXT.zip
#     gtfs/
#     schedule_index/
#     rt_log/
#     features/
#     baseline_report.csv
# =============================================================================

IS_COLAB = "google.colab" in sys.modules
DATA_DIR = Path("/content/drt" if IS_COLAB else "./drt")
GTFS_ZIP = DATA_DIR / "GTFS_Durham_TXT.zip"
GTFS_DIR = DATA_DIR / "gtfs"
INDEX_DIR = DATA_DIR / "schedule_index"
RT_LOG_DIR = DATA_DIR / "rt_log"
FEAT_DIR = DATA_DIR / "features"
REPORT_CSV = DATA_DIR / "baseline_report.csv"
CHART_PNG = DATA_DIR / "oshawa_chart.png"
MODEL_DIR = DATA_DIR / "models"
TZ = ZoneInfo("America/Toronto")

STATIC_GTFS_URL = "https://maps.durham.ca/OpenDataGTFS/GTFS_Durham_TXT.zip"
# Verified live in the uploaded project context.
VEHICLE_POSITIONS_URL = "https://drtonline.durhamregiontransit.com/gtfsrealtime/VehiclePositions"
TRIP_UPDATES_URL = "https://drtonline.durhamregiontransit.com/gtfsrealtime/TripUpdates"

PULSE_ROUTES = {"900", "901", "915", "916"}
STOPPED_AT = 1  # GTFS-RT VehiclePosition.current_status == STOPPED_AT


# =============================================================================
# Setup / helpers
# =============================================================================

def colab_setup() -> None:
    """Install dependencies in Colab and create working directories."""
    if IS_COLAB:
        os.system("pip install -q pandas numpy pyarrow matplotlib requests gtfs-realtime-bindings lightgbm scikit-learn joblib")
    for d in (DATA_DIR, GTFS_DIR, INDEX_DIR, RT_LOG_DIR, FEAT_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Environment: {'Colab' if IS_COLAB else 'local'}")
    print(f"Data root:    {DATA_DIR}")
    print(f"GTFS zip:     {GTFS_ZIP} (exists: {GTFS_ZIP.exists()})")


def t_to_sec(t: str | float | int | None) -> float:
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return np.nan
    s = str(t).strip()
    if not s:
        return np.nan
    h, m, sec = map(int, s.split(":"))
    return h * 3600 + m * 60 + sec


def unix_to_local_dt(unix_s: float | int | None) -> pd.Timestamp:
    if pd.isna(unix_s):
        return pd.NaT
    return pd.to_datetime(int(unix_s), unit="s", utc=True).tz_convert(TZ)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def route_family(route_id) -> str:
    r = str(route_id)
    if r.startswith("N"):
        return "night"
    if not r:
        return "other"
    return {
        "1": "Pickering",
        "2": "Ajax",
        "3": "Whitby",
        "4": "Oshawa",
        "5": "Clarington",
        "6": "Rural",
        "9": "PULSE/Regional",
    }.get(r[0], "other")


def _read_csv_if_exists(path: Path, **kwargs) -> pd.DataFrame | None:
    return pd.read_csv(path, **kwargs) if path.exists() else None


def load_stop_metadata() -> pd.DataFrame:
    """Optional stop attributes from the uploaded TRNST_Bus_Boarding_Points.csv."""
    path = Path("TRNST_Bus_Boarding_Points.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[(df["LAT"].notna()) & (df["LON"].notna())].copy()
    if "NUMBER" not in df.columns:
        return pd.DataFrame()
    df["stop_id"] = df["NUMBER"].astype(str)
    cols = [c for c in ["stop_id", "ACCESSIBLE", "SHELTER", "STATUS", "NAME", "CITY", "DIR", "LAT", "LON"] if c in df.columns]
    out = df[cols].drop_duplicates("stop_id").copy()
    out["ACCESSIBLE"] = pd.to_numeric(out.get("ACCESSIBLE"), errors="coerce").fillna(0).astype(int)
    out["SHELTER"] = pd.to_numeric(out.get("SHELTER"), errors="coerce").fillna(0).astype(int)
    return out


def load_route_gis() -> pd.DataFrame:
    """Optional route length metadata from the uploaded TRNST_Routes.csv."""
    path = Path("TRNST_Routes.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ROUTE_ID" not in df.columns:
        return pd.DataFrame()
    out = df[[c for c in ["ROUTE_ID", "ROUTE_NAME", "GIS_STATUS", "SHAPESTLength"] if c in df.columns]].copy()
    out["ROUTE_ID"] = out["ROUTE_ID"].astype(str)
    out["SHAPESTLength_km"] = pd.to_numeric(out.get("SHAPESTLength"), errors="coerce") / 1000.0
    out = out.sort_values(["ROUTE_ID", "SHAPESTLength_km"], ascending=[True, False]).drop_duplicates("ROUTE_ID")
    return out


# =============================================================================
# Static GTFS
# =============================================================================

def extract_gtfs(force_download: bool = False) -> list[Path]:
    if force_download or not GTFS_ZIP.exists():
        print("Downloading GTFS…")
        import requests
        r = requests.get(STATIC_GTFS_URL, timeout=120)
        r.raise_for_status()
        GTFS_DIR.mkdir(parents=True, exist_ok=True)
        GTFS_ZIP.parent.mkdir(parents=True, exist_ok=True)
        GTFS_ZIP.write_bytes(r.content)
    GTFS_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(GTFS_ZIP) as zf:
        zf.extractall(GTFS_DIR)
    files = sorted(GTFS_DIR.glob("*.txt"))
    print(f"Extracted {len(files)} GTFS text files to {GTFS_DIR}")
    return files


def _service_days(calendar: pd.DataFrame, calendar_dates: pd.DataFrame | None) -> dict[str, list[datetime.date]]:
    out: dict[str, list[datetime.date]] = {}
    dow = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for _, row in calendar.iterrows():
        sid = str(row["service_id"])
        start = datetime.strptime(str(row["start_date"]), "%Y%m%d").date()
        end = datetime.strptime(str(row["end_date"]), "%Y%m%d").date()
        days = []
        d = start
        while d <= end:
            val = str(row[dow[d.weekday()]])
            if val == "1":
                days.append(d)
            d += timedelta(days=1)
        out[sid] = days

    if calendar_dates is not None and not calendar_dates.empty:
        for _, row in calendar_dates.iterrows():
            sid = str(row["service_id"])
            d = datetime.strptime(str(row["date"]), "%Y%m%d").date()
            exc = str(row["exception_type"])
            out.setdefault(sid, [])
            if exc == "1" and d not in out[sid]:
                out[sid].append(d)
            elif exc == "2" and d in out[sid]:
                out[sid].remove(d)
    return out


def build_schedule_index() -> int:
    trips = pd.read_csv(GTFS_DIR / "trips.txt", dtype=str)
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", dtype=str)
    calendar = pd.read_csv(GTFS_DIR / "calendar.txt", dtype=str)
    cal_dates = _read_csv_if_exists(GTFS_DIR / "calendar_dates.txt", dtype=str)

    stop_times = stop_times.copy()
    stop_times["arr_sec"] = stop_times["arrival_time"].map(t_to_sec)
    stop_times["dep_sec"] = stop_times["departure_time"].map(t_to_sec)

    st = stop_times.merge(trips[["trip_id", "service_id", "route_id", "direction_id", "shape_id"]], on="trip_id", how="left")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    days_map = _service_days(calendar, cal_dates)
    total = 0
    # Accumulate all rows per date across service_ids to avoid overwrites
    date_frames: dict[str, list[pd.DataFrame]] = {}
    for sid, day_list in days_map.items():
        subset = st[st["service_id"] == sid]
        if subset.empty:
            continue
        for d in day_list:
            midnight = datetime(d.year, d.month, d.day, tzinfo=TZ)
            mid_unix = int(midnight.timestamp())
            day_df = subset.copy()
            day_df["service_date"] = d.isoformat()
            day_df["scheduled_arr_unix"] = mid_unix + day_df["arr_sec"].astype(float)
            day_df["scheduled_dep_unix"] = mid_unix + day_df["dep_sec"].astype(float)
            keep = ["service_date", "route_id", "trip_id", "stop_id", "stop_sequence", "direction_id", "shape_id", "scheduled_arr_unix", "scheduled_dep_unix"]
            date_key = d.isoformat()
            date_frames.setdefault(date_key, []).append(day_df[keep])

    for date_key, frames in date_frames.items():
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(INDEX_DIR / f"date={date_key}.parquet", index=False, compression="snappy")
        total += len(combined)
    print(f"Schedule index: {len(date_frames)} files, {total:,} scheduled stop-arrivals")
    return total


# =============================================================================
# Baseline analysis
# =============================================================================

def baseline_analysis(show: bool = True) -> pd.DataFrame:
    trips = pd.read_csv(GTFS_DIR / "trips.txt", dtype=str)
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", dtype=str)
    calendar = pd.read_csv(GTFS_DIR / "calendar.txt", dtype=str)
    shapes = pd.read_csv(GTFS_DIR / "shapes.txt")

    stop_times = stop_times.copy()
    stop_times["arr_sec"] = stop_times["arrival_time"].map(t_to_sec)

    shapes = shapes.sort_values(["shape_id", "shape_pt_sequence"]).reset_index(drop=True)
    shapes["lat2"] = shapes.groupby("shape_id")["shape_pt_lat"].shift(-1)
    shapes["lon2"] = shapes.groupby("shape_id")["shape_pt_lon"].shift(-1)
    shapes["seg_km"] = haversine_km(shapes["shape_pt_lat"], shapes["shape_pt_lon"], shapes["lat2"], shapes["lon2"])
    shape_dist = shapes.groupby("shape_id", as_index=False)["seg_km"].sum().rename(columns={"seg_km": "distance_km"})

    trip_spans = (
        stop_times.groupby("trip_id", as_index=False)
        .agg(start_sec=("arr_sec", "min"), end_sec=("arr_sec", "max"), n_stops=("stop_id", "count"))
    )
    trip_spans["duration_min"] = (trip_spans["end_sec"] - trip_spans["start_sec"]) / 60.0
    trip_spans = trip_spans.merge(trips[["trip_id", "route_id", "service_id", "direction_id", "shape_id"]], on="trip_id", how="left")
    trip_spans = trip_spans.merge(shape_dist, on="shape_id", how="left")
    trip_spans["speed_kmh"] = trip_spans["distance_km"] / (trip_spans["duration_min"] / 60.0)

    # Weekday services (Mon-Fri active)
    weekday_services = calendar.loc[
        calendar[["monday", "tuesday", "wednesday", "thursday", "friday"]].eq("1").any(axis=1), "service_id"
    ].astype(str).tolist()
    wk = trip_spans[trip_spans["service_id"].isin(weekday_services)].copy()

    agg = (
        wk.groupby("route_id", as_index=False)
        .agg(
            weekday_trips=("trip_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_distance_km=("distance_km", "mean"),
            avg_n_stops=("n_stops", "mean"),
            avg_speed_kmh=("speed_kmh", "mean"),
        )
        .round(2)
    )
    agg["stops_per_km"] = (agg["avg_n_stops"] / agg["avg_distance_km"]).round(2)
    agg["weekday_service_hours"] = (agg["weekday_trips"] * agg["avg_duration_min"] / 60.0).round(1)
    agg["region"] = agg["route_id"].astype(str).map(route_family)

    # Headway regularity, route-level (correcting for overlapping services)
    hw_rows = []
    for rid in agg["route_id"]:
        sub = wk[wk["route_id"] == rid]
        for (sid, dirn), g in sub.groupby(["service_id", "direction_id"]):
            starts = g["start_sec"].dropna().to_numpy(dtype=float)
            s = starts[(starts >= 6 * 3600) & (starts <= 21 * 3600)]
            if len(s) < 3:
                continue
            gaps = np.diff(np.sort(s)) / 60.0
            if len(gaps) == 0:
                continue
            hw_rows.append({
                "route_id": rid,
                "service_id": sid,
                "direction_id": dirn,
                "median_headway_min": float(np.median(gaps)),
                "headway_cov": float(np.std(gaps) / np.mean(gaps)) if np.mean(gaps) > 0 else np.nan,
                "peak_trip_share": float((((s / 3600) >= 7) & ((s / 3600) < 10) | ((s / 3600) >= 16) & ((s / 3600) < 19)).mean()),
            })
    hw = pd.DataFrame(hw_rows)
    if not hw.empty:
        hw_summary = hw.groupby("route_id", as_index=False).agg(
            median_headway_min=("median_headway_min", "mean"),
            headway_cov=("headway_cov", "mean"),
            peak_trip_share=("peak_trip_share", "mean"),
        ).round(2)
        agg = agg.merge(hw_summary, on="route_id", how="left")
    else:
        agg["median_headway_min"] = np.nan
        agg["headway_cov"] = np.nan
        agg["peak_trip_share"] = np.nan

    # Simple bucket heuristic (for the map/report)
    def classify(row):
        rid = str(row["route_id"])
        pulse = rid in PULSE_ROUTES
        hw = row.get("median_headway_min", np.nan)
        cov = row.get("headway_cov", np.nan)
        peak = row.get("peak_trip_share", np.nan)
        n = row.get("weekday_trips", 0)
        if pulse:
            return "A"
        if n <= 30 and (pd.isna(hw) or hw >= 30):
            return "D"
        if pd.notna(hw) and hw <= 20 and (pd.isna(cov) or cov <= 0.35):
            return "B"
        if pd.notna(peak) and peak >= 0.6 and pd.notna(cov) and cov >= 0.45:
            return "C"
        return "B"

    agg["bucket"] = agg.apply(classify, axis=1)

    # Optional GIS route lengths
    route_gis = load_route_gis()
    if not route_gis.empty:
        agg = agg.merge(route_gis[["ROUTE_ID", "ROUTE_NAME", "GIS_STATUS", "SHAPESTLength_km"]], left_on="route_id", right_on="ROUTE_ID", how="left")
        agg["gis_vs_gtfs_ratio"] = (agg["SHAPESTLength_km"] / agg["avg_distance_km"]).round(2)

    agg = agg.sort_values("weekday_service_hours", ascending=False).reset_index(drop=True)
    agg.to_csv(REPORT_CSV, index=False)

    if show:
        print("=" * 78)
        print("BASELINE SCHEDULE ANALYSIS")
        print("=" * 78)
        print(f"Routes:               {agg['route_id'].nunique()}")
        print(f"Weekday trips:        {int(agg['weekday_trips'].sum()):,}")
        print(f"Service-hours/weekday: {agg['weekday_service_hours'].sum():.0f}")
        print(f"Median speed:          {agg['avg_speed_kmh'].median():.1f} km/h")
        print("\nTop 10 routes by weekday service hours:")
        cols = [c for c in ["route_id", "region", "weekday_trips", "weekday_service_hours", "avg_speed_kmh", "median_headway_min", "headway_cov", "peak_trip_share", "bucket"] if c in agg.columns]
        print(agg.head(10)[cols].to_string(index=False))
        print(f"\nFull report: {REPORT_CSV}")
    return agg


def chart(df: pd.DataFrame) -> Path:
    import matplotlib
    matplotlib.use("Agg" if not IS_COLAB else "module://matplotlib_inline.backend_inline")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    osh = df[df["route_id"].astype(str).str.startswith("4")]
    pul = df[df["route_id"].astype(str).isin(PULSE_ROUTES)]

    for sub, label in [(osh, "Oshawa locals"), (pul, "PULSE")]:
        s = sub.dropna(subset=["headway_cov", "avg_speed_kmh"]).copy()
        if s.empty:
            continue
        ax.scatter(
            s["avg_speed_kmh"], s["headway_cov"],
            s=s["weekday_service_hours"] * 2.2,
            alpha=0.78, edgecolor="k", linewidth=.5, label=label, zorder=3,
        )
        for _, r in s.iterrows():
            ax.annotate(str(r["route_id"]), (r["avg_speed_kmh"], r["headway_cov"]), fontsize=8, ha="center", va="center")

    med = df["avg_speed_kmh"].median()
    max_cov = df["headway_cov"].max(skipna=True)
    ax.axvline(med, ls="--", c="gray", alpha=.6, label=f"median {med:.0f} km/h")
    ax.axhline(0.5, ls=":", c="#ff5c66", alpha=.7, label="irregular (CoV 0.5)")
    ax.set_xlabel("Commercial speed (km/h) → faster")
    ax.set_ylabel("Headway CoV → lower is more regular")
    ax.set_title("DRT routes: speed vs scheduling regularity")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-.05, min(1.05, float(max_cov) * 1.1 if pd.notna(max_cov) else 1.0))
    fig.tight_layout()
    fig.savefig(CHART_PNG, dpi=140, facecolor="white")
    print(f"Chart saved: {CHART_PNG}")
    if IS_COLAB:
        import matplotlib.pyplot as plt2
        plt2.show()
    return CHART_PNG


# =============================================================================
# GTFS-Realtime logger
# =============================================================================

def _fetch_rt(url: str, timeout: int = 15):
    import requests
    try:
        from google.transit import gtfs_realtime_pb2
    except Exception as e:
        raise RuntimeError("gtfs-realtime-bindings is required for realtime logging") from e

    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        msg = gtfs_realtime_pb2.FeedMessage()
        msg.ParseFromString(r.content)
        return msg
    except Exception as e:
        print(f"  fetch failed for {url}: {e}")
        return None


def _vp_to_rows(msg) -> list[dict]:
    poll_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for ent in msg.entity:
        if not ent.HasField("vehicle"):
            continue
        v = ent.vehicle
        rows.append({
            "poll_ts": poll_ts,
            "feed_ts": getattr(msg.header, "timestamp", None),
            "kind": "vp",
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


def _tu_to_rows(msg) -> list[dict]:
    poll_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for ent in msg.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        for stu in tu.stop_time_update:
            rows.append({
                "poll_ts": poll_ts,
                "feed_ts": getattr(msg.header, "timestamp", None),
                "kind": "tu",
                "trip_id": tu.trip.trip_id if tu.HasField("trip") else None,
                "route_id": tu.trip.route_id if tu.HasField("trip") else None,
                "direction_id": tu.trip.direction_id if tu.HasField("trip") else None,
                "start_date": tu.trip.start_date if tu.HasField("trip") else None,
                "vehicle_id": tu.vehicle.id if tu.HasField("vehicle") else None,
                "stop_id": stu.stop_id,
                "stop_sequence": stu.stop_sequence,
                "arrival_time": stu.arrival.time if stu.HasField("arrival") else None,
                "arrival_delay": stu.arrival.delay if stu.HasField("arrival") else None,
                "departure_time": stu.departure.time if stu.HasField("departure") else None,
                "departure_delay": stu.departure.delay if stu.HasField("departure") else None,
                "schedule_relationship": stu.schedule_relationship,
            })
    return rows


def _append_parquet(rows: list[dict], kind: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = RT_LOG_DIR / f"date={today}" / f"{kind}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df.to_parquet(path, index=False, compression="snappy")


def run_logger(interval: int = 20, duration_min: int | None = None, skip_trip_updates: bool = False) -> None:
    RT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"Logging realtime feed to {RT_LOG_DIR} (interval={interval}s, duration={duration_min}min)")
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
            if duration_min is not None and (time.time() - start) > duration_min * 60:
                print("duration elapsed, exiting")
                return
            time.sleep(interval)
        except KeyboardInterrupt:
            print("interrupted, exiting")
            return


# =============================================================================
# Feature engineering
# =============================================================================

def _detect_arrivals_from_vp(vp: pd.DataFrame) -> pd.DataFrame:
    if vp.empty:
        return pd.DataFrame()
    vp = vp.copy()
    vp["t"] = vp["vehicle_ts"].fillna(vp["feed_ts"])
    vp = vp.dropna(subset=["trip_id", "stop_id", "t"])
    vp = vp.sort_values(["trip_id", "vehicle_id", "t"])
    stopped = vp[vp["current_status"] == STOPPED_AT]
    if stopped.empty:
        return pd.DataFrame()
    arrivals = (
        stopped.groupby(["trip_id", "stop_id", "current_stop_seq"], as_index=False)
        .agg(actual_arr_unix=("t", "min"), route_id=("route_id", "first"), vehicle_id=("vehicle_id", "first"))
        .rename(columns={"current_stop_seq": "stop_sequence"})
    )
    arrivals["stop_sequence"] = pd.to_numeric(arrivals["stop_sequence"], errors="coerce").astype("Int64")
    return arrivals


def _actuals_from_tu(tu: pd.DataFrame) -> pd.DataFrame:
    if tu.empty:
        return pd.DataFrame()
    tu = tu.copy().sort_values(["trip_id", "stop_id", "poll_ts"])
    out = tu.dropna(subset=["arrival_time"]).groupby(["trip_id", "stop_id", "stop_sequence"], as_index=False).agg(
        actual_arr_unix=("arrival_time", "last"),
        arrival_delay_reported=("arrival_delay", "last"),
        route_id=("route_id", "first"),
    )
    return out


def build_features(date_str: str) -> int:
    sched_path = INDEX_DIR / f"date={date_str}.parquet"
    if not sched_path.exists():
        print(f"No schedule index for {date_str}")
        return 0

    sched = pd.read_parquet(sched_path)

    vp_path = RT_LOG_DIR / f"date={date_str}" / "vehicle_positions.parquet"
    tu_path = RT_LOG_DIR / f"date={date_str}" / "trip_updates.parquet"
    actuals_vp = _detect_arrivals_from_vp(pd.read_parquet(vp_path)) if vp_path.exists() else pd.DataFrame()
    actuals_tu = _actuals_from_tu(pd.read_parquet(tu_path)) if tu_path.exists() else pd.DataFrame()

    if actuals_tu.empty and actuals_vp.empty:
        print(f"No realtime log for {date_str}")
        return 0

    if not actuals_tu.empty:
        actuals = actuals_tu.copy()
        if not actuals_vp.empty:
            extras = actuals_vp.merge(actuals_tu[["trip_id", "stop_id"]], on=["trip_id", "stop_id"], how="left", indicator=True)
            extras = extras.query("_merge == 'left_only'").drop(columns="_merge")
            actuals = pd.concat([actuals, extras], ignore_index=True, sort=False)
    else:
        actuals = actuals_vp.copy()

    # Align stop_sequence datatypes to avoid pandas merge value error
    sched["stop_sequence"] = pd.to_numeric(sched["stop_sequence"], errors="coerce").astype("Int64")
    if "stop_sequence" in actuals.columns:
        actuals["stop_sequence"] = pd.to_numeric(actuals["stop_sequence"], errors="coerce").astype("Int64")

    features = sched.merge(
        actuals[[c for c in ["trip_id", "stop_id", "stop_sequence", "route_id", "actual_arr_unix", "arrival_delay_reported", "vehicle_id"] if c in actuals.columns]],
        on=["trip_id", "stop_id", "stop_sequence"],
        how="left",
        suffixes=("", "_act"),
    )

    features["delay_sec"] = features["actual_arr_unix"] - features["scheduled_arr_unix"]
    features["otp_label"] = features["delay_sec"].between(-60, 300).astype("Int64")

    # Sequence features
    features = features.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    features["upstream_delay_sec"] = features.groupby("trip_id")["delay_sec"].shift(1)
    features["lagged_sched_gap_sec"] = features.groupby("trip_id")["scheduled_arr_unix"].diff()

    # Calendar features
    dt = pd.to_datetime(features["scheduled_arr_unix"], unit="s", utc=True).dt.tz_convert(TZ)
    features["service_hour"] = dt.dt.hour
    features["dayofweek"] = dt.dt.dayofweek
    features["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    features["is_peak"] = (((dt.dt.hour >= 7) & (dt.dt.hour < 10)) | ((dt.dt.hour >= 16) & (dt.dt.hour < 19))).astype(int)

    # Optional stop metadata enrichment
    stops_meta = load_stop_metadata()
    if not stops_meta.empty:
        features = features.merge(stops_meta[[c for c in ["stop_id", "ACCESSIBLE", "SHELTER", "STATUS"] if c in stops_meta.columns]], on="stop_id", how="left")
        features["ACCESSIBLE"] = pd.to_numeric(features.get("ACCESSIBLE"), errors="coerce").fillna(0).astype(int)
        features["SHELTER"] = pd.to_numeric(features.get("SHELTER"), errors="coerce").fillna(0).astype(int)
        features["STATUS"] = features.get("STATUS").fillna("Unknown")
    else:
        features["ACCESSIBLE"] = 0
        features["SHELTER"] = 0
        features["STATUS"] = "Unknown"

    # Optional route GIS lengths
    route_gis = load_route_gis()
    if not route_gis.empty:
        features = features.merge(route_gis[["ROUTE_ID", "SHAPESTLength_km"]], left_on="route_id", right_on="ROUTE_ID", how="left")

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"date={date_str}.parquet"
    features.to_parquet(out, index=False, compression="snappy")
    print(f"Wrote features: {out} ({len(features):,} rows)")
    return len(features)


# =============================================================================
# Training
# =============================================================================

def train_otp_model() -> dict:
    feat_files = sorted(FEAT_DIR.glob("date=*.parquet"))
    if not feat_files:
        raise FileNotFoundError(f"No feature files found in {FEAT_DIR}. Run build_features(date_str) first.")

    df = pd.concat([pd.read_parquet(f) for f in feat_files], ignore_index=True)
    df = df.dropna(subset=["otp_label", "delay_sec"]).copy()
    if df.empty:
        raise ValueError("Feature files exist, but no labelled rows were found.")

    # Avoid leakage columns and non-feature identifiers.
    drop_cols = {
        "actual_arr_unix", "arrival_delay_reported", "delay_sec", "otp_label", "poll_ts", "feed_ts",
        "kind", "vehicle_id", "scheduled_dep_unix", "scheduled_arr_unix", "service_date",
        "ROUTE_ID", "ROUTE_NAME", "GIS_STATUS", "SHAPESTLength_km",
        "trip_id", "shape_id", "route_id_act",
    }
    cat_cols = [c for c in ["route_id", "stop_id", "direction_id", "STATUS"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Simple encoding for categorical cols.
    for c in cat_cols:
        X[c] = X[c].astype("category").cat.codes

    y = df["otp_label"].astype(int)

    # Time-based split by service_date when present and has multiple dates,
    # else random but reproducible.
    if "service_date" in df.columns:
        dates = sorted(df["service_date"].dropna().astype(str).unique())
    else:
        dates = []

    if len(dates) >= 2:
        cut = max(1, int(len(dates) * 0.8))
        train_dates = set(dates[:cut])
        train_idx = df["service_date"].astype(str).isin(train_dates)
        test_idx = ~train_idx
    else:
        rng = np.random.default_rng(42)
        mask = rng.random(len(df)) < 0.8
        train_idx = mask
        test_idx = ~mask

    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    # Impute numeric features.
    for c in X_train.columns:
        if X_train[c].dtype.kind in "if":
            med = X_train[c].median()
            X_train[c] = X_train[c].fillna(med)
            X_test[c] = X_test[c].fillna(med)
        else:
            X_train[c] = X_train[c].fillna(0)
            X_test[c] = X_test[c].fillna(0)

    model = None
    model_name = ""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(
            n_estimators=400,
            learning_rate=0.04,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            class_weight="balanced",
        )
        model_name = "lightgbm"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(random_state=42)
        model_name = "hist_gradient_boosting"

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    pred = (proba >= 0.5).astype(int) if hasattr(model, "predict_proba") else model.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
        joblib.dump({"model": model, "features": list(X.columns), "metrics": metrics}, MODEL_DIR / "otp_model.joblib")
    except Exception:
        pass

    (MODEL_DIR / "otp_model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


# =============================================================================
# Convenience
# =============================================================================

def download_results() -> None:
    if not IS_COLAB:
        print(f"Outputs are in {DATA_DIR}")
        return
    from google.colab import files
    for p in [REPORT_CSV, CHART_PNG, MODEL_DIR / "otp_model_metrics.json"]:
        if p.exists():
            try:
                files.download(str(p))
            except Exception as e:
                print(f"skip {p}: {e}")


def full_run() -> pd.DataFrame:
    extract_gtfs()
    build_schedule_index()
    df = baseline_analysis(show=True)
    chart(df)
    download_results()
    return df


def print_next_steps() -> None:
    print("\nNext steps:")
    print("  1) Run the schedule pipeline: python drt_pipeline_final.py")
    print("  2) Start realtime logging:    python drt_pipeline_final.py --logger --interval 20 --duration-min 10")
    print("  3) Build features for a day:  python drt_pipeline_final.py --features YYYY-MM-DD")
    print("  4) Train OTP model:           python drt_pipeline_final.py --train")
    print("\nColab data path:")
    print("  /content/drt/GTFS_Durham_TXT.zip")
    print("  /content/drt/rt_log/")
    print("  /content/drt/features/")


# =============================================================================
# Main / CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="DRT Bus Efficiency Pipeline")
    parser.add_argument("--logger", action="store_true", help="run the GTFS-RT logger")
    parser.add_argument("--interval", type=int, default=20, help="logger poll interval in seconds")
    parser.add_argument("--duration-min", type=int, default=None, help="logger duration in minutes")
    parser.add_argument("--features", type=str, default=None, help="build features for YYYY-MM-DD")
    parser.add_argument("--train", action="store_true", help="train the OTP model")
    parser.add_argument("--force-download", action="store_true", help="force download the GTFS zip")
    parser.add_argument("--all", action="store_true", help="run the full static pipeline")
    args = parser.parse_args()

    colab_setup()

    if args.logger:
        run_logger(interval=args.interval, duration_min=args.duration_min)
        return
    if args.features:
        build_features(args.features)
        return
    if args.train:
        train_otp_model()
        return
    if args.all or not any([args.logger, args.features, args.train]):
        extract_gtfs(force_download=args.force_download)
        build_schedule_index()
        df = baseline_analysis(show=True)
        chart(df)
        print_next_steps()
        download_results()


if __name__ == "__main__":
    main()

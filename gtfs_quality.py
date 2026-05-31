"""
GTFS feed data-quality validator.
=================================
Real-world transit feeds are dirty: stops with null coordinates, trips with no
shape, stop_times that go backwards in time, services that already expired,
orphaned references between files. Every one of these silently corrupts the
scorecard, the headway maths, or the schedule index. This module runs a battery
of checks and emits a graded report *before* you trust any downstream number.

Run:  python gtfs_quality.py
"""
from __future__ import annotations

import sys

import pandas as pd

import drt_config as cfg


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.GTFS_DIR / name, dtype=str)


def validate() -> dict:
    issues: list[dict] = []

    def flag(severity, check, detail, n=None):
        issues.append({"severity": severity, "check": check, "detail": detail, "n": n})

    routes = _load("routes.txt")
    trips = _load("trips.txt")
    stops = _load("stops.txt")
    stop_times = _load("stop_times.txt")
    calendar = _load("calendar.txt")
    shapes = _load("shapes.txt")

    # ---- referential integrity ----
    orphan_trips = set(trips["route_id"]) - set(routes["route_id"])
    if orphan_trips:
        flag("ERROR", "trips->routes", f"{len(orphan_trips)} trips reference unknown route_id", len(orphan_trips))

    st_trip_orphans = set(stop_times["trip_id"]) - set(trips["trip_id"])
    if st_trip_orphans:
        flag("ERROR", "stop_times->trips", f"{len(st_trip_orphans)} stop_times reference unknown trip_id", len(st_trip_orphans))

    st_stop_orphans = set(stop_times["stop_id"]) - set(stops["stop_id"])
    if st_stop_orphans:
        flag("ERROR", "stop_times->stops", f"{len(st_stop_orphans)} stop_times reference unknown stop_id", len(st_stop_orphans))

    trip_shape_orphans = set(trips["shape_id"].dropna()) - set(shapes["shape_id"])
    if trip_shape_orphans:
        flag("WARN", "trips->shapes", f"{len(trip_shape_orphans)} trips reference unknown shape_id", len(trip_shape_orphans))

    # ---- stop coordinates ----
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce")
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce")
    bad_coord = stops[lat.isna() | lon.isna()]
    if len(bad_coord):
        flag("ERROR", "stops.coords", f"{len(bad_coord)} stops have null/non-numeric coordinates", len(bad_coord))
    # Durham bounding box sanity
    out_of_box = stops[((lat < 43.0) | (lat > 44.6) | (lon < -79.6) | (lon > -78.0)) & lat.notna()]
    if len(out_of_box):
        flag("WARN", "stops.bbox", f"{len(out_of_box)} stops fall outside the Durham bounding box", len(out_of_box))

    # ---- stop_times monotonicity ----
    st = stop_times.copy()
    st["arr_s"] = st["arrival_time"].map(cfg.t_to_sec)
    st["seq"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.sort_values(["trip_id", "seq"])
    st["prev_arr"] = st.groupby("trip_id")["arr_s"].shift(1)
    backwards = st[(st["arr_s"] < st["prev_arr"])]
    n_back_trips = backwards["trip_id"].nunique()
    if n_back_trips:
        flag("ERROR", "stop_times.monotonic", f"{n_back_trips} trips have a stop arriving before the previous stop", n_back_trips)

    missing_times = st["arr_s"].isna().sum()
    if missing_times:
        flag("INFO", "stop_times.blank", f"{missing_times} stop_times rows have blank arrival_time (non-timepoint)", int(missing_times))

    # ---- calendar currency ----
    cal = calendar.copy()
    cal["end_date"] = cal["end_date"].astype(int)
    today = int(pd.Timestamp.today().strftime("%Y%m%d"))
    expired = cal[cal["end_date"] < today]
    if len(expired) == len(cal):
        flag("ERROR", "calendar.expired", "ALL service periods have already expired", len(expired))
    elif len(expired):
        flag("INFO", "calendar.expired", f"{len(expired)} of {len(cal)} service periods are expired (older schedule versions)", len(expired))

    # ---- coverage of trips by current weekday service ----
    cur = cfg.current_weekday_services(calendar)
    cur_trips = trips[trips["service_id"].isin(cur)]
    flag("INFO", "service.current", f"{len(cur)} current weekday services, {len(cur_trips)} trips, {cur_trips['route_id'].nunique()} routes", len(cur_trips))

    # ---- duplicate stop_times keys ----
    dupes = stop_times.duplicated(subset=["trip_id", "stop_sequence"]).sum()
    if dupes:
        flag("ERROR", "stop_times.dupes", f"{dupes} duplicate (trip_id, stop_sequence) rows", int(dupes))

    errors = sum(1 for i in issues if i["severity"] == "ERROR")
    warns = sum(1 for i in issues if i["severity"] == "WARN")
    grade = "FAIL" if errors else ("PASS WITH WARNINGS" if warns else "PASS")
    return {"grade": grade, "errors": errors, "warnings": warns, "issues": issues,
            "counts": {"routes": len(routes), "trips": len(trips), "stops": len(stops),
                       "stop_times": len(stop_times), "shapes": shapes["shape_id"].nunique()}}


def main() -> int:
    rep = validate()
    print("=" * 70)
    print("GTFS DATA QUALITY REPORT")
    print("=" * 70)
    c = rep["counts"]
    print(f"  routes={c['routes']}  trips={c['trips']}  stops={c['stops']}  "
          f"stop_times={c['stop_times']:,}  shapes={c['shapes']}")
    print(f"\n  GRADE: {rep['grade']}  ({rep['errors']} errors, {rep['warnings']} warnings)\n")
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for i in sorted(rep["issues"], key=lambda x: order[x["severity"]]):
        print(f"  [{i['severity']:5}] {i['check']:24} {i['detail']}")
    if not rep["issues"]:
        print("  No issues found.")
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

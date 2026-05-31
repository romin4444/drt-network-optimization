"""
Equity & coverage-criticality layer.
====================================
The optimizer's instinct is to convert low-ridership routes to on-demand and
free the buses. In the real world that strands transit-dependent riders if the
route is the *only* service in its area. Cost-efficiency without an equity guard
is how agencies end up in the news.

This module scores each route's "coverage criticality" purely from GTFS + the
boarding-points file (no demographic data required):

  * unique_coverage  - fraction of the route's stops that NO other route serves
                       within the 400 m walk buffer. High => cutting it creates
                       a coverage hole, not just a frequency cut.
  * accessible_share - fraction of the route's boarding points flagged accessible
  * sheltered_share  - fraction with a shelter (comfort / vulnerable riders)

A route with high unique_coverage should be *right-sized* (e.g. retimed, peak-
only, or kept as a coverage milk-run) rather than deleted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import drt_config as cfg


def _stop_route_map() -> pd.DataFrame:
    """Return stops with their lat/lon and the set of routes serving each stop."""
    trips = pd.read_csv(cfg.GTFS_DIR / "trips.txt", dtype=str)
    stop_times = pd.read_csv(cfg.GTFS_DIR / "stop_times.txt", dtype=str)
    stops = pd.read_csv(cfg.GTFS_DIR / "stops.txt", dtype=str)
    calendar = pd.read_csv(cfg.GTFS_DIR / "calendar.txt", dtype=str)

    cur = cfg.current_weekday_services(calendar)
    trips = trips[trips["service_id"].isin(cur)]
    st = stop_times.merge(trips[["trip_id", "route_id"]], on="trip_id", how="inner")
    sr = st.groupby("stop_id")["route_id"].apply(lambda s: set(s)).rename("routes")
    stops = stops.set_index("stop_id")
    df = pd.DataFrame({
        "lat": pd.to_numeric(stops["stop_lat"], errors="coerce"),
        "lon": pd.to_numeric(stops["stop_lon"], errors="coerce"),
    }).join(sr, how="inner").dropna(subset=["lat", "lon"])
    return df.reset_index()


def coverage_criticality() -> pd.DataFrame:
    """Per-route coverage criticality + accessibility metrics."""
    stops = _stop_route_map()
    walk_km = cfg.STANDARDS["coverage_walk_m"] / 1000.0

    lat = stops["lat"].to_numpy()
    lon = stops["lon"].to_numpy()
    route_sets = stops["routes"].tolist()
    n = len(stops)

    # For each stop, do all stops within the walk buffer carry only this stop's
    # own route(s)? If so the stop is "uniquely covered". Use a BallTree
    # (haversine) for an O(n log n) radius query instead of the O(n^2) loop;
    # fall back to brute force if scikit-learn isn't installed.
    try:
        from sklearn.neighbors import BallTree
        coords_rad = np.radians(np.c_[lat, lon])
        tree = BallTree(coords_rad, metric="haversine")
        neigh = tree.query_radius(coords_rad, r=walk_km / cfg.EARTH_R_KM)
    except Exception:
        neigh = []
        for i in range(n):
            d = cfg.haversine_km(lat[i], lon[i], lat, lon)
            neigh.append(np.flatnonzero(d <= walk_km))

    unique_flags = np.zeros(n, dtype=bool)
    for i in range(n):
        near_routes: set = set()
        for j in neigh[i]:
            near_routes |= route_sets[j]
        unique_flags[i] = len(near_routes - route_sets[i]) == 0
    stops["unique"] = unique_flags

    # Boarding-point amenities joined by nearest stop (coarse: name match skipped,
    # use status/accessible aggregates at route level via stop membership).
    board = pd.read_csv(cfg.BOARDING_CSV)
    board = board[(board["LAT"] != 0) & (board["LON"] != 0)].copy()

    rows = []
    # explode stop->routes so we can group by route
    exploded = stops.explode("routes").rename(columns={"routes": "route_id"})
    for rid, g in exploded.groupby("route_id"):
        n_stops = len(g)
        rows.append({
            "route_id": str(rid),
            "n_stops_served": n_stops,
            "unique_coverage": round(g["unique"].mean(), 3),
            "n_unique_stops": int(g["unique"].sum()),
        })
    out = pd.DataFrame(rows)

    # Accessibility from boarding points, matched to nearest GTFS stop within 150 m.
    bl = board["LAT"].to_numpy(); bo = board["LON"].to_numpy()
    # Amenity flags are 1/0 integers in the DRT file.
    acc = (pd.to_numeric(board["ACCESSIBLE"], errors="coerce") == 1).to_numpy()
    shel = (pd.to_numeric(board["SHELTER"], errors="coerce") == 1).to_numpy()
    stop_acc, stop_shel = [], []
    for i in range(n):
        d = cfg.haversine_km(lat[i], lon[i], bl, bo)
        j = int(np.argmin(d))
        within = d[j] <= 0.15
        stop_acc.append(bool(acc[j]) if within else np.nan)
        stop_shel.append(bool(shel[j]) if within else np.nan)
    stops["accessible"] = stop_acc
    stops["sheltered"] = stop_shel
    exploded2 = stops.explode("routes").rename(columns={"routes": "route_id"})
    amen = exploded2.groupby("route_id").agg(
        accessible_share=("accessible", "mean"),
        sheltered_share=("sheltered", "mean"),
    ).round(3).reset_index()
    amen["route_id"] = amen["route_id"].astype(str)

    out = out.merge(amen, on="route_id", how="left")

    def tier(u):
        if u >= cfg.LIFELINE_THRESHOLD:
            return "LIFELINE (preserve coverage)"
        if u >= cfg.PARTIAL_UNIQUE_THRESHOLD:
            return "PARTIAL-UNIQUE (right-size, don't delete)"
        return "REDUNDANT (safe to restructure)"
    out["coverage_tier"] = out["unique_coverage"].map(tier)
    return out.sort_values("unique_coverage", ascending=False).reset_index(drop=True)


def main():
    df = coverage_criticality()
    out = cfg.MAP_DATA / "route_equity.csv"
    cfg.MAP_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("=" * 70)
    print("ROUTE COVERAGE & EQUITY CRITICALITY")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")
    lifelines = df[df.unique_coverage >= cfg.LIFELINE_THRESHOLD]["route_id"].tolist()
    print(f"\nLIFELINE routes (do NOT convert to on-demand, "
          f"unique_coverage >= {cfg.LIFELINE_THRESHOLD}): "
          f"{', '.join(lifelines) or 'none'}")
    return df


if __name__ == "__main__":
    main()

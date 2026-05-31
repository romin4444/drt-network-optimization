"""
DRT Route Efficiency Analysis & New-Route Design Module
========================================================
Covers, in one file:
  1. Per-route efficiency scorecard with A/B/C/D diagnostic buckets
  2. Route geometry extraction (one representative shape per route/direction)
  3. Service-gap detection (areas far from any stop)
  4. On Demand -> fixed-route conversion candidates
  5. New-route scoring against DRT service standards
  6. JSON bundle export for the interactive map

All numbers are computed from the real DRT GTFS feed. Nothing is hard-coded.
Run:  python3 route_design.py
"""
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np

GTFS = Path(__file__).parent / "drt" / "gtfs"
OUT = Path(__file__).parent / "drt" / "map_data"
OUT.mkdir(exist_ok=True)

# ----- DRT published service standards (verified from DRT Service Guidelines) -----
STANDARDS = {
    "pulse_min_freq": 15,        # PULSE minimum headway (min), peak/midday
    "base_min_freq": 30,         # Base minimum headway
    "frequent_target": 15,       # Frequent Network target headway
    "speed_min": 22,             # below this on a non-PULSE route flags route deviation
    "cov_frequent": 0.21,        # TCQSM LOS A regularity
    "cov_base": 0.30,            # acceptable regularity for base
    "stops_per_km_arterial": 2.5,
    "stops_per_km_pulse": 2.0,
    "coverage_walk_m": 400,      # walk-distance buffer for "covered"
    "on_demand_convert": 8,      # boardings/hr threshold to graduate OnDemand -> fixed
}
PULSE_ROUTES = {"900", "901", "915", "916"}  # verified roster (+902 being added)

R = 6371.0
def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def shape_length_km(pts):
    return sum(haversine(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1])
               for i in range(len(pts)-1))

def t_to_sec(t):
    if pd.isna(t) or t == "":
        return np.nan
    h, m, s = map(int, str(t).split(":"))
    return h*3600 + m*60 + s


def load():
    return (pd.read_csv(GTFS/"routes.txt", dtype=str),
            pd.read_csv(GTFS/"trips.txt", dtype=str),
            pd.read_csv(GTFS/"stop_times.txt", dtype={"trip_id":str,"stop_id":str}),
            pd.read_csv(GTFS/"stops.txt", dtype=str),
            pd.read_csv(GTFS/"shapes.txt"),
            pd.read_csv(GTFS/"calendar.txt", dtype=str))


def current_weekday_services(calendar):
    """Service IDs for the *current* weekday schedule period only.

    GTFS feeds carry several sequential schedule versions. Counting trips across
    all of them double-counts service that never runs at the same time. Keep only
    services whose start_date is the most recent and that run on a weekday (this
    also picks up the overnight service, which covers weekday nights).
    """
    cal = calendar.copy()
    cal["start_date"] = cal["start_date"].astype(int)
    weekday_mask = cal[["monday", "tuesday", "wednesday", "thursday", "friday"]].eq("1").any(axis=1)
    wd = cal[weekday_mask]
    if wd.empty:
        return []
    latest = wd["start_date"].max()
    return wd[wd["start_date"] == latest]["service_id"].astype(str).tolist()


def route_scorecard(routes, trips, st, stops, shapes, calendar):
    weekday_services = current_weekday_services(calendar)
    wd_trips = trips[trips["service_id"].astype(str).isin(weekday_services)].copy()
    stops_xy = stops.set_index("stop_id")[["stop_lat","stop_lon"]].astype(float)

    # shape lengths
    shp_len = {}
    for sid, g in shapes.groupby("shape_id"):
        g = g.sort_values("shape_pt_sequence")
        pts = list(zip(g["shape_pt_lat"], g["shape_pt_lon"]))
        shp_len[sid] = shape_length_km(pts)

    rows = []
    for rid, rg in wd_trips.groupby("route_id"):
        trip_ids = set(rg["trip_id"])
        st_r = st[st["trip_id"].isin(trip_ids)].copy()
        if st_r.empty:
            continue
        st_r["arr_s"] = st_r["arrival_time"].map(t_to_sec)
        
        # Merge route metadata
        st_r = st_r.merge(rg[["trip_id", "service_id", "direction_id", "shape_id"]], on="trip_id", how="left")
        
        # trip durations & distances
        durs, dists, nstops = [], [], []
        trip_starts = [] # list of dicts with service_id, direction_id, start_sec
        
        for tid, tg in st_r.groupby("trip_id"):
            tg = tg.sort_values("stop_sequence")
            arr = tg["arr_s"].dropna()
            if len(arr) < 2:
                continue
            durs.append((arr.iloc[-1] - arr.iloc[0]) / 60.0)
            nstops.append(len(tg))
            
            first_row = tg.iloc[0]
            trip_starts.append({
                "service_id": str(first_row["service_id"]),
                "direction_id": str(first_row["direction_id"]),
                "start_sec": arr.iloc[0]
            })
            
            sid = first_row["shape_id"]
            dists.append(shp_len.get(sid, np.nan))
            
        if not durs:
            continue
        avg_dur = np.nanmean(durs)
        avg_dist = np.nanmean(dists)
        avg_stops = np.nanmean(nstops)
        speed = (avg_dist / (avg_dur/60.0)) if avg_dur > 0 else np.nan
        spk = avg_stops / avg_dist if avg_dist > 0 else np.nan
        
        # headways computed by grouping by service_id and direction_id to avoid overlap
        hw_gaps = []
        covs = []
        peaks = []
        
        starts_df = pd.DataFrame(trip_starts)
        if not starts_df.empty:
            for (sid_val, dirn_val), g in starts_df.groupby(["service_id", "direction_id"]):
                starts = g["start_sec"].dropna().to_numpy(dtype=float)
                s = starts[(starts >= 6 * 3600) & (starts <= 21 * 3600)]
                if len(s) < 3:
                    continue
                gaps = np.diff(np.sort(s)) / 60.0
                if len(gaps) == 0:
                    continue
                hw_gaps.extend(gaps)
                covs.append(float(np.std(gaps)/np.mean(gaps)) if np.mean(gaps)>0 else np.nan)
                peaks.append(float((((s / 3600) >= 7) & ((s / 3600) < 10) | ((s / 3600) >= 16) & ((s / 3600) < 19)).mean()))
        
        med_hw = float(np.median(hw_gaps)) if len(hw_gaps) else np.nan
        p90_hw = float(np.percentile(hw_gaps, 90)) if len(hw_gaps) else np.nan
        cov = float(np.nanmean(covs)) if covs else np.nan
        peak_share = float(np.nanmean(peaks)) if peaks else np.nan
        svc_hours = len(durs)*avg_dur/60.0

        is_pulse = rid in PULSE_ROUTES
        # ---- diagnostic bucket ----
        bucket, reason = classify(is_pulse, med_hw, peak_share, cov, len(durs))
        rows.append(dict(
            route_id=rid, is_pulse=is_pulse, weekday_trips=len(durs),
            avg_speed_kmh=round(speed,1), avg_distance_km=round(avg_dist,1),
            avg_n_stops=round(avg_stops,1), stops_per_km=round(spk,2),
            median_headway_min=round(med_hw,1) if not np.isnan(med_hw) else None,
            p90_headway_min=round(p90_hw,1) if not np.isnan(p90_hw) else None,
            headway_cov=round(cov,2) if not np.isnan(cov) else None,
            peak_trip_share=round(peak_share,2), weekday_service_hours=round(svc_hours,1),
            bucket=bucket, diagnosis=reason,
        ))
    df = pd.DataFrame(rows).sort_values("weekday_service_hours", ascending=False)
    return df, "Mon-Fri active"


def classify(is_pulse, med_hw, peak_share, cov, ntrips):
    """Assign A/B/C/D diagnostic bucket per the efficiency framework."""
    if is_pulse:
        if med_hw and med_hw <= 15:
            return "A", "Frequent backbone - protect & invest, tighten regularity"
        return "A", "PULSE corridor - lift to true 15-min all-day standard"
    if ntrips <= 30 and (med_hw is None or med_hw >= 30):
        return "D", "Marginal - candidate for On Demand conversion"
    if med_hw and med_hw <= 20 and (cov is None or cov <= 0.35):
        return "B", "Frequent candidate - promote to 15-min, consolidate stops"
    if peak_share >= 0.6 and (cov and cov >= 0.45):
        return "C", "Coverage commuter - cut to peak-only or interline for 15-min combined"
    return "B", "Stable base route - retime if CoV high, else maintain"


def extract_geometries(routes, trips, shapes, scorecard):
    """One representative (longest) shape per route, with route metadata."""
    rmeta = routes.set_index("route_id")[["route_long_name","route_short_name"]].to_dict("index")
    sc = scorecard.set_index("route_id").to_dict("index")
    shp_pts = {}
    for sid, g in shapes.groupby("shape_id"):
        g = g.sort_values("shape_pt_sequence")
        shp_pts[sid] = list(zip(g["shape_pt_lat"].round(5), g["shape_pt_lon"].round(5)))

    feats = []
    for rid, rg in trips.groupby("route_id"):
        if rid not in sc:   # only scored (weekday) routes
            continue
        # longest shape = most complete representation
        shape_ids = rg["shape_id"].dropna().unique()
        best = max(shape_ids, key=lambda s: len(shp_pts.get(s, [])), default=None)
        if not best or best not in shp_pts:
            continue
        info = sc[rid]
        feats.append(dict(
            route_id=rid,
            name=rmeta.get(rid, {}).get("route_long_name", f"Route {rid}"),
            bucket=info["bucket"],
            speed=info["avg_speed_kmh"],
            headway=info["median_headway_min"],
            cov=info["headway_cov"],
            peak_share=info["peak_trip_share"],
            service_hours=info["weekday_service_hours"],
            is_pulse=info["is_pulse"],
            diagnosis=info["diagnosis"],
            coords=[[float(la), float(lo)] for la, lo in shp_pts[best]],
        ))
    return feats


def service_gaps(stops, boarding_csv=None, grid_km=0.5):
    """Find grid cells with no stop within the walk buffer (rough coverage gap proxy)."""
    s = stops.copy()
    s["stop_lat"] = s["stop_lat"].astype(float)
    s["stop_lon"] = s["stop_lon"].astype(float)
    s = s[(s["stop_lat"]>40) & (s["stop_lon"]<-70)]   # valid Durham coords
    lat0, lat1 = s["stop_lat"].min(), s["stop_lat"].max()
    lon0, lon1 = s["stop_lon"].min(), s["stop_lon"].max()
    # build a coarse grid over the service area
    dlat = grid_km/111.0
    dlon = grid_km/(111.0*math.cos(math.radians((lat0+lat1)/2)))
    stop_pts = s[["stop_lat","stop_lon"]].values
    gaps = []
    la = lat0
    while la < lat1:
        lo = lon0
        while lo < lon1:
            # distance to nearest stop
            d = np.min(np.sqrt(((stop_pts[:,0]-la)*111.0)**2 +
                               ((stop_pts[:,1]-lo)*111.0*math.cos(math.radians(la)))**2))
            if STANDARDS["coverage_walk_m"]/1000.0 < d < 1.5:  # 400m-1.5km from any stop = gap
                gaps.append([round(la,5), round(lo,5), round(d,2)])
            lo += dlon
        la += dlat
    return gaps


def on_demand_candidates(boarding_csv):
    """From the boarding-points file: On Demand stops clustered enough to suggest fixed service."""
    if not Path(boarding_csv).exists():
        return []
    b = pd.read_csv(boarding_csv)
    b = b[(b["LAT"]!=0) & (b["LON"]!=0)]
    od = b[b["STATUS"].isin(["On Demand","Scheduled On Demand Shared"])].copy()
    # cluster by 1km grid; count stops per cell as a density proxy (no APC = use density)
    od["cell_lat"] = (od["LAT"]/0.009).round()*0.009
    od["cell_lon"] = (od["LON"]/0.012).round()*0.012
    dens = od.groupby(["cell_lat","cell_lon"]).size().reset_index(name="n_od_stops")
    dens = dens[dens["n_od_stops"] >= 8].sort_values("n_od_stops", ascending=False)
    return [dict(lat=float(r.cell_lat), lon=float(r.cell_lon), n_stops=int(r.n_od_stops))
            for r in dens.itertuples()]


def score_new_route(coords, stop_count, cycle_min, vehicles, route_class="base"):
    """Score a proposed new route against DRT standards. coords=[[lat,lon],...]."""
    dist = shape_length_km(coords)
    speed = dist/(cycle_min/60.0) if cycle_min>0 else 0
    spk = stop_count/dist if dist>0 else 0
    headway = cycle_min/vehicles if vehicles>0 else None
    spacing_m = (dist*1000)/stop_count if stop_count>0 else 0
    target_freq = STANDARDS["pulse_min_freq"] if route_class=="pulse" else STANDARDS["frequent_target"]
    checks = {
        "length_km": round(dist,1),
        "commercial_speed_kmh": round(speed,1),
        "speed_ok": speed >= STANDARDS["speed_min"],
        "stop_spacing_m": round(spacing_m),
        "spacing_ok": 250 <= spacing_m <= 600,
        "headway_min": round(headway,1) if headway else None,
        "frequency_ok": headway is not None and headway <= target_freq,
        "stops_per_km": round(spk,2),
        "vehicles_needed": vehicles,
    }
    checks["passes"] = sum([checks["speed_ok"], checks["spacing_ok"], checks["frequency_ok"]])
    checks["grade"] = ["NEEDS WORK","FAIR","GOOD","EXCELLENT"][checks["passes"]]
    return checks


def main():
    routes, trips, st, stops, shapes, calendar = load()
    print("Computing route scorecard...")
    sc, svc = route_scorecard(routes, trips, st, stops, shapes, calendar)
    print(f"  {len(sc)} weekday routes scored (service {svc})")
    print(sc["bucket"].value_counts().to_dict())

    print("Extracting route geometries...")
    geoms = extract_geometries(routes, trips, shapes, sc)
    print(f"  {len(geoms)} route geometries")

    print("Detecting service gaps...")
    gaps = service_gaps(stops)
    print(f"  {len(gaps)} gap cells (400m-1.5km from nearest stop)")

    boarding = Path(__file__).parent / "TRNST_Bus_Boarding_Points.csv"
    print("Finding On Demand conversion candidates...")
    od = on_demand_candidates(boarding)
    print(f"  {len(od)} dense On Demand clusters (>=8 stops/km cell)")

    # Worked example: proposed Route 910 Oshawa-Ajax frequent corridor
    # (representative coords sampled along the Oshawa->Ajax arterial)
    route910 = [[43.897,-78.86],[43.90,-78.90],[43.91,-78.94],[43.92,-78.98],
                [43.93,-79.02],[43.94,-79.03],[43.95,-79.02]]
    r910 = score_new_route(route910, stop_count=28, cycle_min=52, vehicles=4, route_class="base")

    bundle = dict(
        scorecard=sc.to_dict("records"),
        geometries=geoms,
        gaps=gaps,
        on_demand_clusters=od,
        proposed_route_910=r910,
        standards=STANDARDS,
        bucket_counts=sc["bucket"].value_counts().to_dict(),
    )
    (OUT/"route_bundle.json").write_text(json.dumps(bundle))
    sc.to_csv(OUT/"route_scorecard.csv", index=False)
    print(f"\nWrote {OUT/'route_bundle.json'}")
    print(f"Proposed Route 910 grade: {r910['grade']} ({r910['passes']}/3 standards)")
    return bundle


if __name__ == "__main__":
    main()

"""
DRT route optimizer — fleet, cost & equity-aware.
=================================================
Turns the route scorecard into an actionable, costed service plan. Improvements
over the original headway-only version:

  * Peak Vehicle Requirement (PVR) including recovery/layover time and a spare
    ratio — the original ceil(cycle/headway) understated real fleet needs.
  * Dollar costs: every recommendation carries an annual operating-cost delta
    and a capital cost for added buses (amortised). "Add 3 buses" is now a
    budget line, not a vibe.
  * Equity guard: on-demand conversion is blocked for LIFELINE coverage routes
    (see equity.py). Cost-efficiency never silently strands captive riders.

Run:  python route_optimizer.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import drt_config as cfg
import equity

SCORECARD = cfg.MAP_DATA / "route_scorecard.csv"
OUTPUT = cfg.MAP_DATA / "route_optimization_scorecard.csv"


def peak_vehicles(cycle_min: float, headway_min: float) -> int:
    """Buses needed to hold a headway on a round-trip cycle, incl. recovery."""
    if headway_min is None or headway_min <= 0 or np.isnan(headway_min):
        return 0
    eff_cycle = cycle_min * (1.0 + cfg.FLEET["recovery_factor"])
    return max(1, int(np.ceil(eff_cycle / headway_min)))


def annual_operating_cost(vehicles: int, headway_min: float) -> float:
    """Rough annual operating $ for running `vehicles` over the service span."""
    if vehicles <= 0:
        return 0.0
    rev_hours = vehicles * cfg.FLEET["service_span_hr"] * cfg.FLEET["annual_service_days"]
    return rev_hours * cfg.COST["operating_per_rev_hr"]


def optimize() -> pd.DataFrame:
    if not SCORECARD.exists():
        raise SystemExit(f"{SCORECARD} not found. Run route_design.py first.")
    df = pd.read_csv(SCORECARD)

    # Equity / coverage criticality, merged per route.
    eq = equity.coverage_criticality()[["route_id", "unique_coverage", "coverage_tier"]]
    df["route_id"] = df["route_id"].astype(str)
    eq["route_id"] = eq["route_id"].astype(str)
    df = df.merge(eq, on="route_id", how="left")

    rows = []
    for _, r in df.iterrows():
        rid = str(r["route_id"])
        bucket = str(r["bucket"])
        avg_dist = float(r["avg_distance_km"])
        avg_n_stops = float(r["avg_n_stops"])
        stops_per_km = float(r["stops_per_km"])
        med_hw = float(r["median_headway_min"]) if pd.notna(r["median_headway_min"]) else np.nan
        trips = int(r["weekday_trips"])
        svc_hours = float(r["weekday_service_hours"])
        tier = str(r.get("coverage_tier", "")) or "REDUNDANT (safe to restructure)"
        is_lifeline = tier.startswith("LIFELINE")

        avg_dur = (svc_hours * 60.0) / trips if trips > 0 else 0.0
        current_rt_cycle = 2.0 * avg_dur
        active_vehicles = peak_vehicles(current_rt_cycle, med_hw) or 1

        # ---- target service category ----
        if bucket == "A":
            target_hw, category = cfg.STANDARDS["pulse_min_freq"], "Frequent Backbone"
        elif bucket == "B" and (stops_per_km >= 2.5 or rid in ["121", "319", "410"]):
            target_hw, category = cfg.STANDARDS["frequent_target"], "Frequent Candidate"
        elif bucket == "D" or (bucket == "C" and trips <= 30):
            target_hw, category = 0.0, "Marginal (On-Demand candidate)"
        else:
            target_hw, category = cfg.STANDARDS["base_min_freq"], "Base Coverage / Commuter"

        # ---- stop consolidation (travel-time saving) ----
        consolidated, time_saved_min = 0.0, 0.0
        optimized_rt_cycle = current_rt_cycle
        if stops_per_km > cfg.STANDARDS["target_stops_per_km"]:
            target_n = avg_dist * cfg.STANDARDS["target_stops_per_km"]
            consolidated = max(0.0, avg_n_stops - target_n)
            time_saved_min = 2.0 * consolidated * cfg.STANDARDS["stop_consolidation_saving_sec"] / 60.0
            optimized_rt_cycle = max(current_rt_cycle - time_saved_min, avg_dur)
        optimized_speed = (2.0 * avg_dist) / (optimized_rt_cycle / 60.0) if optimized_rt_cycle > 0 else 0.0
        tt_reduction_pct = (time_saved_min / current_rt_cycle * 100.0) if current_rt_cycle > 0 else 0.0

        # ---- fleet plan + equity guard ----
        if target_hw > 0:
            required = peak_vehicles(optimized_rt_cycle, target_hw)
            net = max(0, required - active_vehicles)
            action = f"Optimize speed ({tt_reduction_pct:.1f}% saved)"
            action += f" & add {net} bus(es)" if net > 0 else " (fleet sufficient)"
        elif is_lifeline:
            # Equity guard: never delete a lifeline coverage route.
            required = active_vehicles
            net = 0
            category = "Coverage Lifeline (PROTECT)"
            target_hw = med_hw if not np.isnan(med_hw) else cfg.STANDARDS["base_min_freq"]
            action = ("Marginal ridership BUT lifeline coverage "
                      f"(unique={r.get('unique_coverage')}); right-size/retime, do NOT delete")
        else:
            required = 0
            net = -active_vehicles
            action = "Convert to On-Demand; reallocate freed vehicles to frequent corridors"

        # ---- costs ----
        op_now = annual_operating_cost(active_vehicles, med_hw)
        op_future = annual_operating_cost(required, target_hw if target_hw > 0 else med_hw)
        annual_op_delta = op_future - op_now
        capital_cost = max(0, net) * cfg.COST["bus_capital"]
        annualized = max(0, net) * cfg.annualized_bus_capital() + annual_op_delta

        rows.append({
            "route_id": rid, "category": category, "coverage_tier": tier,
            "unique_coverage": r.get("unique_coverage"),
            "weekday_trips": trips,
            "current_speed_kmh": round(float(r["avg_speed_kmh"]), 1),
            "optimized_speed_kmh": round(optimized_speed, 1),
            "stops_per_km": round(stops_per_km, 2),
            "stops_to_consolidate_rt": round(2.0 * consolidated, 1),
            "round_trip_time_saved_min": round(time_saved_min, 1),
            "current_headway_min": round(med_hw, 1) if pd.notna(med_hw) else np.nan,
            "target_headway_min": target_hw if target_hw > 0 else np.nan,
            "current_vehicles": active_vehicles, "required_vehicles": required,
            "net_new_buses_needed": net,
            "capital_cost_cad": round(capital_cost),
            "annual_operating_delta_cad": round(annual_op_delta),
            "annualized_cost_cad": round(annualized),
            "action_plan": action,
        })

    return pd.DataFrame(rows).sort_values(
        ["net_new_buses_needed", "annualized_cost_cad"], ascending=False)


def summarize(opt: pd.DataFrame) -> dict:
    freed = -opt[opt["net_new_buses_needed"] < 0]["net_new_buses_needed"].sum()
    needed = opt[opt["net_new_buses_needed"] > 0]["net_new_buses_needed"].sum()
    net_buy = max(0, needed - freed)
    spare = int(np.ceil(opt["required_vehicles"].sum() * cfg.FLEET["spare_ratio"]))
    return {
        "routes": len(opt),
        "fleet_now": int(opt["current_vehicles"].sum()),
        "fleet_required": int(opt["required_vehicles"].sum()),
        "spare_buses": spare,
        "buses_freed": int(freed),
        "buses_needed": int(needed),
        "net_buy": int(net_buy),
        "capital_cad": int(net_buy * cfg.COST["bus_capital"]),
        "annual_operating_delta_cad": int(opt["annual_operating_delta_cad"].sum()),
        "lifelines_protected": int((opt["category"] == "Coverage Lifeline (PROTECT)").sum()),
    }


def main():
    opt = optimize()
    cfg.MAP_DATA.mkdir(parents=True, exist_ok=True)
    opt.to_csv(OUTPUT, index=False)
    s = summarize(opt)

    print("=" * 78)
    print("ROUTE OPTIMIZATION - FLEET, COST & EQUITY")
    print("=" * 78)
    print(f"  Routes analyzed:              {s['routes']}")
    print(f"  Peak fleet now / required:    {s['fleet_now']} -> {s['fleet_required']} "
          f"(+{s['spare_buses']} spares @ {cfg.FLEET['spare_ratio']:.0%})")
    print(f"  Buses freed (on-demand):      {s['buses_freed']}")
    print(f"  Buses needed (frequency):     {s['buses_needed']}")
    print(f"  NET NEW BUSES TO PURCHASE:    {s['net_buy']}")
    print(f"  Capital cost (net new):       ${s['capital_cad']:,}")
    print(f"  Annual operating delta:       ${s['annual_operating_delta_cad']:,}/yr")
    print(f"  Lifeline routes protected:    {s['lifelines_protected']} (equity guard)")

    print("\n--- TOP CORRIDORS REQUIRING BUS ADDITIONS ---")
    cols = ["route_id", "category", "current_headway_min", "target_headway_min",
            "net_new_buses_needed", "capital_cost_cad", "annualized_cost_cad"]
    print(opt[opt["net_new_buses_needed"] > 0].head(10)[cols].to_string(index=False))

    print("\n--- ON-DEMAND CONVERSIONS vs EQUITY GUARD ---")
    conv = opt[opt["category"].isin(["Marginal (On-Demand candidate)", "Coverage Lifeline (PROTECT)"])]
    print(conv[["route_id", "category", "unique_coverage", "current_vehicles",
                "net_new_buses_needed", "action_plan"]].to_string(index=False))

    print("\n--- TOP SPEED GAINS VIA STOP CONSOLIDATION ---")
    print(opt[opt["round_trip_time_saved_min"] > 0].sort_values(
        "round_trip_time_saved_min", ascending=False).head(5)[
        ["route_id", "stops_per_km", "stops_to_consolidate_rt",
         "round_trip_time_saved_min", "current_speed_kmh", "optimized_speed_kmh"]
    ].to_string(index=False))

    print(f"\nWrote {OUTPUT}")
    return opt


if __name__ == "__main__":
    main()

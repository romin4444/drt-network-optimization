"""
Consolidated planning report generator.
=======================================
Ties the whole pipeline into a single decision-ready Markdown brief: data-quality
grade, network baseline, diagnostic buckets, the costed fleet plan, and the
equity guard. This is the artifact you hand to a planning committee instead of
four loose CSVs.

Run:  python generate_report.py   ->   drt/map_data/DRT_PLAN.md
"""
from __future__ import annotations

from datetime import date

import pandas as pd

import drt_config as cfg
import gtfs_quality
import route_optimizer


def _fmt_money(x) -> str:
    return f"${x:,.0f}"


def build() -> str:
    q = gtfs_quality.validate()
    opt = route_optimizer.optimize()
    s = route_optimizer.summarize(opt)
    sc = pd.read_csv(cfg.MAP_DATA / "route_scorecard.csv")

    buckets = sc["bucket"].value_counts().to_dict()
    lines: list[str] = []
    w = lines.append

    w(f"# DRT Network Optimization Plan")
    w(f"_Generated {date.today().isoformat()} from the live Durham GTFS feed._\n")

    w("## 1. Data quality")
    w(f"- Feed grade: **{q['grade']}** ({q['errors']} errors, {q['warnings']} warnings)")
    c = q["counts"]
    w(f"- {c['routes']} routes, {c['trips']:,} trips, {c['stops']:,} stops, "
      f"{c['stop_times']:,} stop-times, {c['shapes']} shapes\n")

    w("## 2. Network baseline (current weekday service)")
    w(f"- Routes scored: **{len(sc)}**")
    w(f"- Weekday trips: **{int(sc['weekday_trips'].sum()):,}**")
    w(f"- Weekday service-hours: **{sc['weekday_service_hours'].sum():,.0f}**")
    w(f"- Median commercial speed: **{sc['avg_speed_kmh'].median():.1f} km/h**")
    w(f"- Diagnostic buckets: " + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())) + "\n")

    w("## 3. Fleet & budget plan")
    w(f"- Peak fleet: **{s['fleet_now']} → {s['fleet_required']}** buses "
      f"(+{s['spare_buses']} spares @ {cfg.FLEET['spare_ratio']:.0%})")
    w(f"- Buses freed by on-demand conversion: **{s['buses_freed']}**")
    w(f"- Buses needed for frequency upgrades: **{s['buses_needed']}**")
    w(f"- **Net new buses to purchase: {s['net_buy']}**")
    w(f"- Capital cost (net new fleet): **{_fmt_money(s['capital_cad'])}**")
    w(f"- Annual operating cost change: **{_fmt_money(s['annual_operating_delta_cad'])}/yr**\n")

    w("### Top corridors requiring investment")
    top = opt[opt["net_new_buses_needed"] > 0].head(8)
    w("| Route | Category | Headway now→target | New buses | Capital | Annualized |")
    w("|---|---|---|---|---|---|")
    for _, r in top.iterrows():
        hw = f"{r['current_headway_min']:.0f}→{r['target_headway_min']:.0f}" if pd.notna(r["current_headway_min"]) else f"—→{r['target_headway_min']:.0f}"
        w(f"| {r['route_id']} | {r['category']} | {hw} min | "
          f"{int(r['net_new_buses_needed'])} | {_fmt_money(r['capital_cost_cad'])} | {_fmt_money(r['annualized_cost_cad'])} |")
    w("")

    w("## 4. Equity guard")
    w(f"The optimizer flagged marginal routes for on-demand conversion, but **{s['lifelines_protected']} "
      f"routes were protected** as lifeline coverage (the only service within a 400 m walk for most "
      f"of their stops). Deleting these would create coverage holes, not just frequency cuts.\n")
    protect = opt[opt["category"] == "Coverage Lifeline (PROTECT)"]
    if len(protect):
        w("| Route | Unique coverage | Decision |")
        w("|---|---|---|")
        for _, r in protect.iterrows():
            w(f"| {r['route_id']} | {r['unique_coverage']:.0%} | Right-size / retime — do **not** delete |")
        w("")
    conv = opt[opt["category"] == "Marginal (On-Demand candidate)"]
    if len(conv):
        w(f"Cleared for on-demand conversion: **{', '.join(conv['route_id'].tolist())}** "
          f"(freeing {int(-conv['net_new_buses_needed'].sum())} buses).\n")

    w("## 5. Speed gains via stop consolidation (no new buses)")
    cons = opt[opt["round_trip_time_saved_min"] > 0].sort_values(
        "round_trip_time_saved_min", ascending=False).head(6)
    w("| Route | Stops/km | Stops to cut (RT) | Time saved | Speed |")
    w("|---|---|---|---|---|")
    for _, r in cons.iterrows():
        w(f"| {r['route_id']} | {r['stops_per_km']} | {r['stops_to_consolidate_rt']} | "
          f"{r['round_trip_time_saved_min']} min | {r['current_speed_kmh']}→{r['optimized_speed_kmh']} km/h |")
    w("")

    w("---")
    w("_Cost assumptions (editable in `drt_config.py`): "
      f"bus capital {_fmt_money(cfg.COST['bus_capital'])} over {cfg.COST['bus_life_years']} yr, "
      f"operating {_fmt_money(cfg.COST['operating_per_rev_hr'])}/rev-hr, "
      f"recovery {cfg.FLEET['recovery_factor']:.0%}, spare ratio {cfg.FLEET['spare_ratio']:.0%}._")

    return "\n".join(lines)


def main():
    md = build()
    out = cfg.MAP_DATA / "DRT_PLAN.md"
    cfg.MAP_DATA.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    # Console may be cp1252 on Windows; print without crashing on arrows/dashes.
    import sys
    sys.stdout.buffer.write(md.encode("utf-8", errors="replace"))
    print(f"\n\nWrote {out}")


if __name__ == "__main__":
    main()

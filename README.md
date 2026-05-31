# DRT Network Optimization Toolkit

End-to-end analysis of Durham Region Transit: ingest the real GTFS feed, score
every route, model on-time performance from GTFS-RT, and produce a **costed,
equity-aware service plan** — not just a pile of metrics.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          # full planning pipeline -> drt/map_data/DRT_PLAN.md
python test_drt.py         # regression tests (stdlib, no pytest needed)
```

## Pipeline

```
GTFS static feed (drt/gtfs/)
   │
   ├─ gtfs_quality.py      data-quality gate (referential integrity, coords,
   │                       monotonic times, expired calendars) → PASS/FAIL
   ├─ drt_pipeline.py      schedule index (per-day parquet) + baseline report
   │     --logger          poll GTFS-RT → vehicle_positions / trip_updates
   │     --features DATE    join RT actuals to schedule → delay/OTP features
   │     --train           LightGBM OTP model (native categoricals, grouped
   │                       split by trip to avoid leakage) → drt/otp_model.txt
   ├─ route_design.py      A/B/C/D diagnostic scorecard, geometries, gaps
   ├─ equity.py            coverage criticality (lifeline vs redundant routes)
   ├─ route_optimizer.py   peak vehicle requirement + $ cost + equity guard
   └─ generate_report.py   consolidated Markdown brief (DRT_PLAN.md)
```

`drt_config.py` is the single source of truth for service standards, cost/fleet
assumptions, and shared geo/time helpers — override costs via environment
variables (`DRT_BUS_CAPITAL_COST`, `DRT_OPERATING_COST_PER_HR`, …).

## What makes this "planning-grade"

| Concern | How it's handled |
|---|---|
| **Schedule versions** | Only the *current* weekday period is counted — sequential GTFS versions are no longer double-counted (~2× trip inflation fixed). |
| **Fleet sizing** | Peak Vehicle Requirement includes recovery/layover + a spare ratio, not bare `cycle/headway`. |
| **Budgeting** | Every recommendation carries capital + annual operating cost in CAD. |
| **Equity** | On-demand conversion is blocked for *lifeline* routes (sole coverage within a 400 m walk). |
| **Data trust** | A validator gates the pipeline; dirty feeds fail loudly before they corrupt numbers. |
| **Model integrity** | OTP model splits by `trip_id` so upstream-delay features can't leak the label. |
| **Regression safety** | `test_drt.py` locks in the fixed bugs and core invariants. |

## Key outputs (`drt/map_data/`)

- `DRT_PLAN.md` — the decision-ready brief
- `route_scorecard.csv` — per-route diagnostics + bucket
- `route_equity.csv` — coverage criticality / accessibility
- `route_optimization_scorecard.csv` — costed fleet plan
- `route_bundle.json` — geometry bundle for the interactive map

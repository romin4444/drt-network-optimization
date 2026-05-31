# 🚌 Durham Region Transit — Network Optimization Toolkit

[![CI](https://github.com/romin4444/drt-network-optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/romin4444/drt-network-optimization/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/romin4444/drt-network-optimization/blob/main/drt_pipeline.ipynb)

> **In one sentence:** this project studies every bus route in Durham Region
> (Ontario) using the transit agency's own public data, then recommends how to
> make the network faster, more frequent, and fairer — and tells you **what it
> would cost**.

It answers questions a transit planner (or a curious taxpayer) actually asks:
*Which routes are great and which are struggling? Where should we add buses, and
how much would that cost? Which slow routes could be sped up for free? And which
low-ridership routes are someone's only ride and must NOT be cut?*

---

## 🎯 The headline finding

Using the live Durham GTFS feed (**38 weekday routes, ~2,455 daily trips**), the
toolkit produces a costed, equity-checked plan:

| Recommendation | Result |
|---|---|
| 🟢 Speed up slow routes by removing over-close stops | up to **+7 min/round-trip**, no new buses |
| 🔵 Upgrade key corridors to 15-min frequency | **+12 buses**, ~**$9 M** capital |
| 🔴 Convert truly marginal routes to on-demand | frees **11 buses** to redeploy |
| 🛡️ **Protect lifeline routes from cuts** | **5 routes** kept (sole service for their riders) |

➡️ **The full plain-English plan is in [`drt/map_data/DRT_PLAN.md`](drt/map_data/DRT_PLAN.md).**

---

## 🗺️ See it

**[▶ Open the interactive map (`map.html`)](map.html)** — every route, coloured by
health, click for details. (Download the file and open it in any browser.)

| The network, by route health | Where to add buses | Free speed-ups |
|---|---|---|
| ![network](assets/network_map.png) | ![fleet](assets/fleet.png) | ![speed](assets/speed.png) |

![buckets](assets/buckets.png)

---

## 🧒 Explain it like I'm five

Buses run on timetables. This project downloads Durham's real timetable (and
live GPS data), measures how each route behaves, and sorts every route into four
groups:

- **🟢 A — Frequent backbone:** the busy, reliable routes. *Protect & invest.*
- **🔵 B — Stable / promote:** solid routes; some could run more often.
- **🟠 C — Coverage commuter:** mostly rush-hour, irregular. *Restructure.*
- **🔴 D — Marginal:** very few trips. *Maybe replace with on-demand vans —*
  *unless they're the only service in the area (then we keep them).*

Then it works out the buses and dollars needed to improve things, and double-
checks that no money-saving idea accidentally strands people who depend on the
bus.

---

## 📖 Glossary (no jargon left behind)

| Term | Plain meaning |
|---|---|
| **GTFS** | The standard public file format transit agencies publish their schedules in. |
| **GTFS-RT** | The *real-time* version — live bus GPS and delays. |
| **Headway** | Minutes between buses on a route. "15-min headway" = a bus every 15 min. |
| **OTP / on-time** | On-Time Performance — did the bus arrive close to schedule (−1 to +5 min)? |
| **CoV** | How *irregular* the gaps between buses are. High = bunched/unreliable. |
| **PVR** | Peak Vehicle Requirement — how many buses a route needs at once, incl. rest time. |
| **Lifeline route** | A route that is the *only* service within a 400 m walk for most of its stops. |
| **Bucket A/B/C/D** | The four health grades above. |

---

## 🏃 Run it yourself

**Non-coders:** click the **Open in Colab** badge above — it runs in your browser,
no install.

**Locally:**
```bash
pip install -r requirements.txt
python run_all.py        # full pipeline  -> drt/map_data/DRT_PLAN.md
python make_visuals.py   # charts + map.html
python test_drt.py       # regression tests (no pytest needed)
```

---

## 🛠️ How it works (for engineers)

```
GTFS static feed (drt/gtfs/)
   ├─ gtfs_quality.py      data-quality gate (integrity, coords, monotonic times)
   ├─ drt_pipeline.py      schedule index + baseline report
   │     --logger          poll GTFS-RT -> vehicle_positions / trip_updates
   │     --features DATE    join RT actuals to schedule -> delay/OTP features
   │     --train           LightGBM OTP model (see MODEL_CARD.md)
   ├─ route_design.py      A/B/C/D scorecard, geometries, service gaps
   ├─ equity.py            coverage criticality (lifeline vs redundant routes)
   ├─ route_optimizer.py   peak vehicle requirement + $ cost + equity guard
   ├─ generate_report.py   consolidated Markdown brief (DRT_PLAN.md)
   └─ make_visuals.py      PNG charts + interactive map.html
```

`drt_config.py` is the single source of truth for service standards, cost/fleet
assumptions, and shared helpers (override costs via env vars like
`DRT_BUS_CAPITAL_COST`).

### What makes it more than a notebook
| Concern | How it's handled |
|---|---|
| **Schedule versions** | Only the *current* weekday period is counted (no double-counting). |
| **Fleet sizing** | Peak Vehicle Requirement incl. recovery time + spare ratio. |
| **Budgeting** | Every recommendation carries capital + annual operating cost (CAD). |
| **Equity** | On-demand conversion is blocked for lifeline routes. |
| **Data trust** | A validator gates the pipeline; dirty feeds fail loudly. |
| **Model integrity** | Train/val/test split, grouped by trip, with baselines — see [`MODEL_CARD.md`](MODEL_CARD.md). |
| **Regression safety** | `test_drt.py` + GitHub Actions CI. |

## 🤖 The model

A LightGBM classifier predicts whether an arrival will be on time. Read
[`MODEL_CARD.md`](MODEL_CARD.md) first — it is **honest about the current
limitation that only one day of real-time data has been collected**, so the
model is a validated *scaffold*, not yet a trustworthy predictor. Several weeks
of logging are needed before its metrics mean anything.

## 📂 Key outputs (`drt/map_data/`)
- `DRT_PLAN.md` — the decision-ready brief
- `route_scorecard.csv` — per-route diagnostics + bucket
- `route_equity.csv` — coverage criticality / accessibility
- `route_optimization_scorecard.csv` — costed fleet plan
- `route_bundle.json` — data behind the interactive map

## 📄 License
MIT (code) — see [LICENSE](LICENSE). Transit data © Durham Region Open Data.

# DRT Network Optimization Plan
_Generated 2026-05-31 from the live Durham GTFS feed._

## 1. Data quality
- Feed grade: **PASS** (0 errors, 0 warnings)
- 39 routes, 7,755 trips, 1,969 stops, 308,096 stop-times, 308 shapes

## 2. Network baseline (current weekday service)
- Routes scored: **38**
- Weekday trips: **2,455**
- Weekday service-hours: **1,658**
- Median commercial speed: **29.1 km/h**
- Diagnostic buckets: A=4, B=24, C=1, D=9

## 3. Fleet & budget plan
- Peak fleet: **147 → 147** buses (+28 spares @ 20%)
- Buses freed by on-demand conversion: **4**
- Buses needed for frequency upgrades: **23**
- **Net new buses to purchase: 19**
- Capital cost (net new fleet): **$14,250,000**
- Annual operating cost change: **$10,888,447/yr**

### Top corridors requiring investment
| Route | Category | Headway now→target | New buses | Capital | Annualized |
|---|---|---|---|---|---|
| N1 | Base Coverage / Commuter | —→30 min | 5 | $3,750,000 | $312,500 |
| 121 | Frequent Candidate | 30→15 min | 3 | $2,250,000 | $1,661,970 |
| 916 | Frequent Backbone | 20→15 min | 2 | $1,500,000 | $1,346,867 |
| 405 | Frequent Candidate | 30→15 min | 2 | $1,500,000 | $1,325,150 |
| 224 | Frequent Candidate | 30→15 min | 2 | $1,500,000 | $1,318,292 |
| 915 | Frequent Backbone | 20→15 min | 2 | $1,500,000 | $1,277,144 |
| N2 | Base Coverage / Commuter | —→30 min | 2 | $1,500,000 | $125,000 |
| 410 | Frequent Candidate | 30→15 min | 1 | $750,000 | $1,255,792 |

## 4. Equity guard
The optimizer flagged marginal routes for on-demand conversion, but **5 routes were protected** as lifeline coverage (the only service within a 400 m walk for most of their stops). Deleting these would create coverage holes, not just frequency cuts.

| Route | Unique coverage | Decision |
|---|---|---|
| 211 | 69% | Right-size / retime — do **not** delete |
| 505 | 79% | Right-size / retime — do **not** delete |
| 605 | 85% | Right-size / retime — do **not** delete |
| 507 | 97% | Right-size / retime — do **not** delete |
| 101 | 87% | Right-size / retime — do **not** delete |

Cleared for on-demand conversion: **315, 306, 618, 227** (freeing 4 buses).

## 5. Speed gains via stop consolidation (no new buses)
| Route | Stops/km | Stops to cut (RT) | Time saved | Speed |
|---|---|---|---|---|
| 410 | 3.31 | 21.0 | 7.0 min | 27.3→31.1 km/h |
| 423 | 3.01 | 14.0 | 4.7 min | 28.9→31.5 km/h |
| 216 | 2.9 | 9.4 | 3.1 min | 26.5→28.1 km/h |
| 224 | 2.74 | 6.7 | 2.2 min | 27.6→28.6 km/h |
| 405 | 2.58 | 2.5 | 0.8 min | 28.9→29.2 km/h |
| 605 | 2.61 | 2.5 | 0.8 min | 25.9→26.2 km/h |

---
_Cost assumptions (editable in `drt_config.py`): bus capital $750,000 over 12 yr, operating $135/rev-hr, recovery 12%, spare ratio 20%._
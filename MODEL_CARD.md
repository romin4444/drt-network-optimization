# Model Card — DRT On-Time-Performance (OTP) Classifier

A short, honest description of the machine-learning model in this project, what
it is for, and — importantly — what it is **not** ready for.

## Overview

| | |
|---|---|
| **Model** | LightGBM gradient-boosted trees (binary classifier) |
| **Task** | Predict whether a scheduled bus arrival will be **on time** (actual arrival within −60 s to +300 s of schedule) |
| **Output** | Probability of on-time, plus a 0/1 label at a 0.5 threshold |
| **Code** | `train_otp_model()` in `drt_pipeline.py` |
| **Artifacts** | `drt/otp_model.txt` (LightGBM booster), `drt/otp_metrics.json` (metrics) |

## Intended use

- **Intended:** exploratory analysis of *which conditions* (hour, peak, route,
  position along the trip, upstream delay) are associated with lateness, to
  support service-planning decisions. Pair with SHAP (`explain_model()`) for
  feature attribution.
- **Not intended:** passenger-facing arrival predictions, operational dispatch,
  or any decision affecting individuals. It is a planning aid, not a real-time
  prediction service.

## Data

- **Source:** Durham Region Transit static GTFS + GTFS-Realtime (vehicle
  positions and trip updates), joined per service-date in `build_features()`.
- **Label:** derived by comparing actual arrival (from RT) to the scheduled
  arrival in the schedule index.
- **Features:** `hour, minute, weekday, is_weekend, is_peak_am, is_peak_pm,
  stop_sequence, frac_of_trip, upstream_delay_sec, direction_id, route_family,
  route_id` (categoricals handled natively by LightGBM).

## Evaluation methodology

- **Three-way split** (train / validation / test). Early stopping uses the
  **validation** set; the **test** set is scored once. (Using test for early
  stopping would leak it into model selection.)
- **Grouped by `trip_id`** so consecutive stops of one trip never straddle a
  split boundary — otherwise `upstream_delay_sec` leaks the label.
- **Temporal split** (train on earliest days, test on the latest) is used
  automatically once ≥ 3 service-days are available; otherwise a grouped random
  split is used and the run prints a warning.
- **Baselines reported alongside the model** so the numbers are interpretable:
  1. majority-class ("always on-time"),
  2. route × hour historical on-time rate.

## ⚠️ Known limitations (read before quoting any metric)

1. **Only one day of real-time data has been collected so far**
   (`date=2026-05-29`). With a single day, reported metrics (AUC ≈ 0.99) are
   **overfit and not meaningful** — the run prints a warning to this effect.
   **Collect several weeks of RT data before trusting the model.**
2. **`upstream_delay_sec` is the dominant signal and is only available at
   *real-time* inference**, not at planning time. If you intend planning-time
   prediction, drop this feature and re-evaluate — the honest accuracy will be
   substantially lower.
3. **No hyperparameter tuning** — parameters are sensible defaults, not
   optimized.
4. **Coverage bias:** the label only exists for stops where RT actuals matched
   the schedule index; stops with missing RT data are absent, which can bias the
   sample toward better-instrumented trips/routes.

## Fairness / equity note

This model predicts a service-quality outcome, not anything about individuals.
Equity considerations in this project live in the **planning** layer
(`equity.py` / the route optimizer's lifeline guard), not in this classifier.

## Reproducing

```bash
python drt_pipeline.py --features 2026-05-29   # build features for a logged day
python drt_pipeline.py --train                 # train + evaluate against baselines
```

Random seed is fixed (`random_state=42`). Metrics are written to
`drt/otp_metrics.json`.

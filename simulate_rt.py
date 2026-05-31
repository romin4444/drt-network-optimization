"""
Synthetic GTFS-RT simulator — for validating the ML PIPELINE, not reality.
=========================================================================
Only one day of real Durham RT data has been collected, so the OTP model can't
be evaluated honestly yet (see MODEL_CARD.md). This module generates many days
of *plausible* delays from the real schedule index, so we can:

  1. exercise the temporal train/val/test split (needs >= 14 days),
  2. show the model genuinely beats its baselines on signal that isn't just
     leakage, and
  3. demonstrate honest *planning-time* performance by dropping the real-time-only
     feature `upstream_delay_sec` (train with --no-upstream).

IMPORTANT: these labels are SIMULATED. Numbers from a simulated run say the
pipeline works; they say nothing about real DRT on-time performance. Real
multi-week logging is still required before trusting the model on live data.

Generative delay model (documented, so the "signal" is honest, not magic):
    delay_sec(stop) =  route_base[route]                # fixed per route
                     + peak_term (AM/PM rush)           # structural, learnable
                     + hour_curve(hour)                 # midday dip, evening rise
                     + position_term * frac_of_trip     # delay grows along a trip
                     + trip_walk                        # per-trip cumulative noise
                     + Gaussian noise
`trip_walk` is a cumulative random walk along each trip, which is what makes the
previous stop's delay (upstream_delay_sec) genuinely predictive — exactly the
real-world autocorrelation the feature is meant to capture.

Run:
    python simulate_rt.py --days 20            # write 20 days of synthetic features
    python drt_pipeline.py --train             # temporal split kicks in (>=14 days)
    python drt_pipeline.py --train --no-upstream   # honest planning-time view
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import drt_config as cfg
from drt_pipeline import INDEX_DIR, FEAT_DIR, route_family
from drt_config import OTP_EARLY_SEC, OTP_LATE_SEC

TZ = "America/Toronto"


def _weekday_index_dates() -> list[str]:
    """Service-dates that have a schedule index AND fall on a weekday."""
    out = []
    for p in sorted(INDEX_DIR.glob("date=*.parquet")):
        d = p.stem.replace("date=", "")
        try:
            if pd.Timestamp(d).weekday() < 5:
                out.append(d)
        except ValueError:
            continue
    return out


def _is_real_features(path) -> bool:
    """True if an existing feature file holds REAL (non-simulated) data."""
    if not path.exists():
        return False
    import pyarrow.parquet as pq
    return "is_simulated" not in pq.ParquetFile(path).schema.names


def simulate_day(date_str: str, seed: int, overwrite_real: bool = False) -> int:
    sched_path = INDEX_DIR / f"date={date_str}.parquet"
    if not sched_path.exists():
        return 0
    out = FEAT_DIR / f"date={date_str}.parquet"
    if _is_real_features(out) and not overwrite_real:
        print(f"  skip {date_str}: real RT features exist (use --overwrite-real to replace)")
        return 0
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(sched_path).copy()
    df = df.dropna(subset=["scheduled_arr_unix"])
    if df.empty:
        return 0

    # time-of-day features (local)
    ts = pd.to_datetime(df["scheduled_arr_unix"], unit="s", utc=True).dt.tz_convert(TZ)
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_peak_am"] = ts.dt.hour.between(7, 9).astype(int)
    df["is_peak_pm"] = ts.dt.hour.between(16, 18).astype(int)
    df = df.sort_values(["trip_id", "stop_sequence"])
    trip_max = df.groupby("trip_id")["stop_sequence"].transform("max").replace(0, 1)
    df["frac_of_trip"] = df["stop_sequence"] / trip_max

    # ---- generative delay model ----
    routes = df["route_id"].astype(str).unique()
    route_base = {r: rng.normal(20, 35) for r in routes}            # sec, fixed/route
    day_effect = rng.normal(0, 25)                                  # whole-day shift
    hour_curve = {h: (40 if h >= 18 else (-15 if 10 <= h <= 14 else 0)) for h in range(28)}

    base = df["route_id"].astype(str).map(route_base).to_numpy()
    peak = 80.0 * df["is_peak_am"].to_numpy() + 70.0 * df["is_peak_pm"].to_numpy()
    hourc = df["hour"].map(hour_curve).fillna(0).to_numpy()
    position = 35.0 * df["frac_of_trip"].to_numpy()

    # per-trip cumulative random walk -> makes upstream_delay genuinely predictive
    step = rng.normal(0, 22, size=len(df))
    df["_step"] = step
    trip_walk = df.groupby("trip_id")["_step"].cumsum().to_numpy()

    noise = rng.normal(0, 45, size=len(df))
    delay = base + peak + hourc + position + trip_walk + day_effect + noise
    df["delay_sec"] = delay
    df["on_time"] = ((delay >= OTP_EARLY_SEC) & (delay <= OTP_LATE_SEC)).astype(int)

    df["upstream_delay_sec"] = df.groupby("trip_id")["delay_sec"].shift(1).fillna(0)
    df["route_family"] = df["route_id"].astype(str).map(route_family)
    df["_date"] = date_str
    df["is_simulated"] = True

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    out = FEAT_DIR / f"date={date_str}.parquet"
    df.drop(columns=["_step"]).to_parquet(out, index=False, compression="snappy")
    return len(df)


def simulate_range(n_days: int, start_seed: int = 1000, overwrite_real: bool = False) -> int:
    dates = _weekday_index_dates()[:n_days]
    if len(dates) < n_days:
        print(f"WARNING: only {len(dates)} weekday index dates available "
              f"(asked for {n_days}). Build the schedule index first.")
    total = 0
    otp_rates = []
    for i, d in enumerate(dates):
        n = simulate_day(d, seed=start_seed + i, overwrite_real=overwrite_real)
        if n:
            rate = pd.read_parquet(FEAT_DIR / f"date={d}.parquet")["on_time"].mean()
            otp_rates.append(rate)
            total += n
    mean_otp = float(np.mean(otp_rates)) if otp_rates else 0.0
    print(f"Simulated {len(dates)} day(s), {total:,} stop-arrivals "
          f"(mean on-time {mean_otp:.1%}).")
    print("NOTE: labels are SIMULATED - for pipeline validation only, not real OTP.")
    return total


def main():
    ap = argparse.ArgumentParser(description="Synthetic GTFS-RT feature generator")
    ap.add_argument("--days", type=int, default=20, help="number of weekday days to simulate")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--overwrite-real", action="store_true",
                    help="overwrite existing REAL feature files (default: preserve them)")
    args = ap.parse_args()
    simulate_range(args.days, start_seed=args.seed, overwrite_real=args.overwrite_real)


if __name__ == "__main__":
    main()

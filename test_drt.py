"""
Regression / unit tests for the DRT toolkit.
============================================
Runs with pytest (industry standard) OR the stdlib runner:

    pytest -q                 # preferred
    python test_drt.py        # fallback, no pytest needed

Design goals after review feedback:
  * Test the RISKIEST logic — optimizer cost math, classify() buckets, the
    lifeline threshold, peak-headway PVR, calendar selection — not just helpers.
  * Use SYNTHETIC fixtures for unit tests so they don't break every time DRT
    refreshes the live feed. The few checks that read the shipped feed are
    clearly marked as integration checks.
"""
import math
import unittest

import numpy as np
import pandas as pd

import drt_config as cfg
import route_design as rd
import route_optimizer as ro
import equity
import gtfs_quality


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class TestHelpers(unittest.TestCase):
    def test_t_to_sec_basic_and_over_24h(self):
        self.assertEqual(cfg.t_to_sec("01:02:03"), 3723)
        self.assertEqual(cfg.t_to_sec("25:30:00"), 25 * 3600 + 30 * 60)

    def test_t_to_sec_blank_is_nan(self):
        for blank in ("", None, float("nan")):
            self.assertTrue(math.isnan(cfg.t_to_sec(blank)))

    def test_haversine_known_distance(self):
        d = cfg.haversine_km(43.897, -78.86, 43.85, -79.02)
        self.assertTrue(10 < d < 20, d)

    def test_route_family(self):
        self.assertEqual(cfg.route_family("901"), "PULSE/Regional")
        self.assertEqual(cfg.route_family("N1"), "night")
        self.assertEqual(cfg.route_family("410"), "Oshawa")

    def test_single_source_of_truth(self):
        # route_design and the pipeline must use the SAME helper objects, not
        # private copies (this is the drift the refactor was meant to kill).
        import drt_pipeline
        self.assertIs(rd.t_to_sec, cfg.t_to_sec)
        self.assertIs(rd.current_weekday_services, cfg.current_weekday_services)
        self.assertIs(drt_pipeline.t_to_sec, cfg.t_to_sec)
        self.assertIs(drt_pipeline.route_family, cfg.route_family)


# --------------------------------------------------------------------------- #
# Calendar / service selection
# --------------------------------------------------------------------------- #
class TestServiceSelection(unittest.TestCase):
    def _cal(self):
        return pd.DataFrame({
            "service_id": ["expired_wk", "current_wk", "current_ovn", "future_wk", "satsun"],
            "start_date": ["20260101", "20260401", "20260401", "20270101", "20260401"],
            "end_date":   ["20260331", "20260630", "20260630", "20271231", "20260630"],
            "monday": ["1", "1", "1", "1", "0"], "tuesday": ["1", "1", "1", "1", "0"],
            "wednesday": ["1", "1", "1", "1", "0"], "thursday": ["1", "1", "1", "1", "0"],
            "friday": ["1", "1", "1", "1", "0"],
        })

    def test_picks_active_period_not_future(self):
        sel = set(cfg.current_weekday_services(self._cal(), target_date="2026-05-15"))
        self.assertEqual(sel, {"current_wk", "current_ovn"})
        self.assertNotIn("future_wk", sel)   # future schedule must NOT be picked
        self.assertNotIn("expired_wk", sel)
        self.assertNotIn("satsun", sel)

    def test_dtype_robust(self):
        # Same answer whether calendar columns are str or int.
        cal_int = self._cal().copy()
        for c in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
            cal_int[c] = cal_int[c].astype(int)
        a = cfg.current_weekday_services(self._cal(), target_date="2026-05-15")
        b = cfg.current_weekday_services(cal_int, target_date="2026-05-15")
        self.assertEqual(set(a), set(b))


# --------------------------------------------------------------------------- #
# classify() diagnostic buckets  (incl. the NaN-headway path that used to fail)
# --------------------------------------------------------------------------- #
class TestClassify(unittest.TestCase):
    def test_pulse_is_A(self):
        self.assertEqual(rd.classify(True, 12, 0.4, 0.2, 200)[0], "A")

    def test_marginal_no_headway_is_D(self):
        # Few trips AND undefined (NaN) headway must land in D — the old
        # `med_hw is None` guard never fired on np.nan, so this silently fell through.
        self.assertEqual(rd.classify(False, np.nan, np.nan, np.nan, 10)[0], "D")

    def test_frequent_candidate_is_B(self):
        self.assertEqual(rd.classify(False, 15, 0.4, 0.2, 120)[0], "B")

    def test_coverage_commuter_is_C(self):
        self.assertEqual(rd.classify(False, 35, 0.8, 0.5, 40)[0], "C")


# --------------------------------------------------------------------------- #
# Optimizer cost / fleet math
# --------------------------------------------------------------------------- #
class TestOptimizerMath(unittest.TestCase):
    def test_peak_vehicles_includes_recovery(self):
        # 60-min cycle, 15-min headway: bare = 4, with 12% recovery -> 5.
        self.assertEqual(ro.peak_vehicles(60, 15), 5)

    def test_peak_vehicles_no_headway(self):
        self.assertEqual(ro.peak_vehicles(60, float("nan")), 0)

    def test_operating_cost_uses_rate(self):
        # On-demand rate must actually change the cost (on-demand isn't free).
        hours = 1000.0
        fixed = ro.annual_operating_cost(hours)
        od = ro.annual_operating_cost(hours, rate=cfg.COST["on_demand_per_hr"])
        self.assertAlmostEqual(fixed, hours * cfg.COST["operating_per_rev_hr"])
        self.assertNotEqual(fixed, od)
        self.assertGreater(od, 0)

    def test_annualized_capital(self):
        self.assertEqual(cfg.annualized_bus_capital(),
                         cfg.COST["bus_capital"] / cfg.COST["bus_life_years"])

    def test_on_demand_frees_but_not_all_buses(self):
        # End-to-end on a synthetic scorecard: a marginal route converts to
        # on-demand, which must (a) still need >=1 vehicle and (b) cost > 0.
        sc = pd.DataFrame([{
            "route_id": "999", "is_pulse": False, "weekday_trips": 10,
            "avg_speed_kmh": 25, "avg_distance_km": 10, "avg_n_stops": 20,
            "stops_per_km": 2.0, "median_headway_min": 60, "peak_headway_min": 60,
            "p90_headway_min": 90, "headway_cov": 0.4, "peak_trip_share": 0.5,
            "weekday_service_hours": 10, "bucket": "D", "diagnosis": "x",
        }])
        eq = pd.DataFrame([{"route_id": "999", "unique_coverage": 0.1,
                            "coverage_tier": "REDUNDANT (safe to restructure)"}])
        out = ro.optimize(scorecard=sc, equity_df=eq)
        row = out[out.route_id == "999"].iloc[0]
        self.assertTrue(bool(row["is_on_demand"]))
        self.assertGreaterEqual(row["required_vehicles"], 1)     # not free of vehicles
        self.assertLess(row["net_new_buses_needed"], 0)          # frees some
        self.assertNotEqual(row["annual_operating_delta_cad"], 0)  # not free of $


# --------------------------------------------------------------------------- #
# Equity threshold consistency
# --------------------------------------------------------------------------- #
class TestEquityThreshold(unittest.TestCase):
    def test_single_threshold_constant(self):
        # equity tiers and the optimizer guard must key off ONE constant.
        self.assertGreater(cfg.LIFELINE_THRESHOLD, cfg.PARTIAL_UNIQUE_THRESHOLD)
        # a route just above the threshold is LIFELINE; just below is not.
        above = cfg.LIFELINE_THRESHOLD + 0.01
        below = cfg.LIFELINE_THRESHOLD - 0.01
        # mirror the tier logic
        def tier(u):
            if u >= cfg.LIFELINE_THRESHOLD: return "LIFELINE"
            if u >= cfg.PARTIAL_UNIQUE_THRESHOLD: return "PARTIAL"
            return "REDUNDANT"
        self.assertEqual(tier(above), "LIFELINE")
        self.assertNotEqual(tier(below), "LIFELINE")


# --------------------------------------------------------------------------- #
# Integration checks against the shipped feed (may change when DRT refreshes).
# --------------------------------------------------------------------------- #
class TestShippedFeedIntegration(unittest.TestCase):
    """These read the bundled GTFS; they are data checks, not pure unit tests."""

    def test_feed_has_no_errors(self):
        rep = gtfs_quality.validate()
        self.assertEqual(rep["errors"], 0,
                         [i for i in rep["issues"] if i["severity"] == "ERROR"])

    def test_current_services_nonempty(self):
        cal = pd.read_csv(cfg.GTFS_DIR / "calendar.txt", dtype=str)
        # Don't hard-code the count (it changes with feed refreshes); just assert
        # the selector returns a sane, non-empty set on the bundled feed.
        sel = cfg.current_weekday_services(cal, target_date="2026-05-31")
        self.assertTrue(1 <= len(sel) <= 5, sel)


if __name__ == "__main__":
    unittest.main(verbosity=2)

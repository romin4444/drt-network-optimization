"""
Regression tests for the DRT pipeline.
======================================
Pure-stdlib (unittest) so they run anywhere with no pip install:

    python test_drt.py

They lock in the three bugs fixed during the upgrade so they can't silently
come back, plus the core helper and cost-model invariants.
"""
import math
import unittest

import numpy as np
import pandas as pd

import drt_config as cfg
import gtfs_quality


class TestHelpers(unittest.TestCase):
    def test_t_to_sec_basic(self):
        self.assertEqual(cfg.t_to_sec("01:02:03"), 3723)

    def test_t_to_sec_over_24h(self):
        # GTFS allows times past midnight (e.g. 25:30:00 for a 1:30am trip).
        self.assertEqual(cfg.t_to_sec("25:30:00"), 25 * 3600 + 30 * 60)

    def test_t_to_sec_blank_is_nan(self):
        # Regression: blank times must not raise (non-timepoint stops).
        self.assertTrue(math.isnan(cfg.t_to_sec("")))
        self.assertTrue(math.isnan(cfg.t_to_sec(None)))
        self.assertTrue(math.isnan(cfg.t_to_sec(float("nan"))))

    def test_haversine_known_distance(self):
        # Oshawa to Ajax is roughly 20 km; assert within a sane band.
        d = cfg.haversine_km(43.897, -78.86, 43.85, -79.02)
        self.assertTrue(10 < d < 20, d)

    def test_route_family(self):
        self.assertEqual(cfg.route_family("901"), "PULSE/Regional")
        self.assertEqual(cfg.route_family("N1"), "night")
        self.assertEqual(cfg.route_family("410"), "Oshawa")


class TestServiceSelection(unittest.TestCase):
    def test_current_period_only(self):
        # Regression: must pick only the latest schedule period, not sum
        # sequential versions (which double-counts trips).
        cal = pd.DataFrame({
            "service_id": ["old_wk", "new_wk", "new_overnight", "new_satsun"],
            "start_date": ["20260101", "20260413", "20260413", "20260413"],
            "monday": ["1", "1", "1", "0"],
            "tuesday": ["1", "1", "1", "0"],
            "wednesday": ["1", "1", "1", "0"],
            "thursday": ["1", "1", "1", "0"],
            "friday": ["1", "1", "1", "0"],
        })
        sel = set(cfg.current_weekday_services(cal))
        self.assertEqual(sel, {"new_wk", "new_overnight"})
        self.assertNotIn("old_wk", sel)       # expired period excluded
        self.assertNotIn("new_satsun", sel)   # weekend-only excluded


class TestCostModel(unittest.TestCase):
    def test_annualized_capital_positive(self):
        self.assertGreater(cfg.annualized_bus_capital(), 0)
        self.assertEqual(cfg.annualized_bus_capital(),
                         cfg.COST["bus_capital"] / cfg.COST["bus_life_years"])

    def test_peak_vehicles_includes_recovery(self):
        import route_optimizer as ro
        # 60-min cycle, 15-min headway: bare math = 4, with 12% recovery -> 5.
        self.assertEqual(ro.peak_vehicles(60, 15), 5)

    def test_peak_vehicles_no_headway(self):
        import route_optimizer as ro
        self.assertEqual(ro.peak_vehicles(60, float("nan")), 0)


class TestGTFSQuality(unittest.TestCase):
    def test_feed_passes(self):
        rep = gtfs_quality.validate()
        # The shipped feed must be free of ERROR-level issues.
        self.assertEqual(rep["errors"], 0, [i for i in rep["issues"] if i["severity"] == "ERROR"])

    def test_current_service_count(self):
        cal = pd.read_csv(cfg.GTFS_DIR / "calendar.txt", dtype=str)
        # Exactly the two current weekday-active services (weekday + overnight).
        self.assertEqual(len(cfg.current_weekday_services(cal)), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

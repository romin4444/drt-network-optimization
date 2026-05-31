"""
One-command pipeline orchestrator.
==================================
Runs the full planning pipeline in dependency order and stops on the first hard
failure (e.g. a feed that fails data-quality validation). This is the entry
point a scheduler or CI job should call.

    python run_all.py            # validate -> design -> optimize -> equity -> report
    python run_all.py --skip-validate
"""
from __future__ import annotations

import argparse
import sys


def step(name: str, fn):
    print("\n" + "#" * 78)
    print(f"# {name}")
    print("#" * 78)
    fn()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the GTFS data-quality gate")
    args = ap.parse_args()

    if not args.skip_validate:
        import gtfs_quality
        rc = gtfs_quality.main()
        if rc != 0:
            print("\nABORT: feed failed data-quality validation. Fix errors or "
                  "rerun with --skip-validate to override.")
            return 1

    import route_design
    step("ROUTE DESIGN (scorecard, geometries, gaps)", route_design.main)

    import equity
    step("EQUITY / COVERAGE CRITICALITY", equity.main)

    import route_optimizer
    step("ROUTE OPTIMIZER (fleet, cost, equity)", route_optimizer.main)

    import generate_report
    step("CONSOLIDATED PLANNING REPORT", generate_report.main)

    print("\nDone. Key outputs in drt/map_data/:")
    print("  route_scorecard.csv  route_equity.csv  "
          "route_optimization_scorecard.csv  DRT_PLAN.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

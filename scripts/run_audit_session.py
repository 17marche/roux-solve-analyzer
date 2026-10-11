#!/usr/bin/env python3
"""Interactive Roux Ergonomic Benchmark & Human Audit Session Runner.

Feeds curated scrambles from data/roux_solves.json to the domain expert,
displays solver recommendations with kinematic grip telemetry, and
collects physical evaluation verdicts for human-in-the-loop calibration.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional

from roux_engine.core.parser import MoveParser
from roux_engine.ergonomics.models import HandProfile
from roux_engine.inspector.phase_inspector import inspect_phase, format_inspect_report
from roux_engine.inspector.audit import (
    perform_audit,
    AuditRecord,
    AuditAbortedError,
    DEFAULT_AUDIT_MD_PATH,
    DEFAULT_AUDIT_JSONL_PATH,
)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Loads and validates speedcube reconstructions from data/roux_solves.json."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Solves dataset not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    valid_solves = [
        s for s in data
        if s.get("is_valid", False) and s.get("scramble", "").strip()
    ]
    return valid_solves


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_audit_session",
        description="Run an interactive on-cube ergonomic audit session across human solves.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["fb", "sb", "lse"],
        default="fb",
        help="Roux phase to audit: 'fb', 'sb', or 'lse' (default: fb)",
    )
    parser.add_argument(
        "-n", "--count",
        type=int,
        default=10,
        help="Number of scrambles to audit in this session (default: 10)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/roux_solves.json"),
        help="Path to roux_solves.json dataset",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["2H", "OH"],
        default="2H",
        help="Solving style profile (default: 2H)",
    )
    parser.add_argument(
        "--m-slice-hand",
        type=str,
        choices=["right", "left"],
        default="right",
        help="M-slice flicking hand for 2H mode (default: right)",
    )
    parser.add_argument(
        "--solve-id",
        type=str,
        default=None,
        help="Specific solve ID to test (optional)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling scrambles (default: 42)",
    )

    args = parser.parse_args(argv)

    try:
        solves = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    if args.solve_id:
        selected_solves = [s for s in solves if str(s.get("id")) == str(args.solve_id)]
        if not selected_solves:
            print(f"Error: Solve ID {args.solve_id} not found in dataset.", file=sys.stderr)
            return 1
    else:
        rng = random.Random(args.seed)
        sample_size = min(args.count, len(solves))
        selected_solves = rng.sample(solves, sample_size)

    hand_profile = HandProfile(solving_mode=args.profile, m_slice_hand=args.m_slice_hand)

    print("=" * 80)
    print("        ROUX SPEEDCUBE ERGONOMIC AUDIT BENCHMARK SESSION")
    print("=" * 80)
    print(f"  Phase under test:  {args.phase.upper()}")
    print(f"  Hand Profile:      {args.profile} ({args.m_slice_hand}-handed M-slice)")
    print(f"  Cases to evaluate: {len(selected_solves)}")
    print(f"  Audit Output MD:   {DEFAULT_AUDIT_MD_PATH}")
    print(f"  Audit Output JSON: {DEFAULT_AUDIT_JSONL_PATH}")
    print("=" * 80)
    print("\nInstructions:")
    print("  1. Scramble your real cube with the displayed scramble sequence.")
    print("  2. Review the solver recommendations, grips, and regrip predictions.")
    print("  3. Physically execute the lines on your cube to test ergonomic feel.")
    print("  4. Enter [1] if you Agree, or [2-4] to log an ergonomic flaw with notes.")
    print("  Press Ctrl+C at any time to finish the session early and view summary.\n")

    input("Press Enter to begin the audit session...")

    session_records: List[AuditRecord] = []

    for idx, solve in enumerate(selected_solves, 1):
        scramble = solve.get("scramble", "").strip()
        solver_name = solve.get("solver") or "Unknown"
        solve_id = solve.get("id") or idx

        print("\n" + "#" * 80)
        print(f"  CASE {idx}/{len(selected_solves)}: Solve #{solve_id} ({solver_name})")
        print("#" * 80)
        print(f"\n  Scramble to execute on your cube:\n")
        print(f"    {scramble}\n")
        print("-" * 80)

        try:
            result = inspect_phase(
                scramble=scramble,
                phase=args.phase,
                profile=hand_profile,
            )
            report_str = format_inspect_report(result)
            print(report_str)

            record = perform_audit(
                result=result,
                md_path=DEFAULT_AUDIT_MD_PATH,
                jsonl_path=DEFAULT_AUDIT_JSONL_PATH,
            )
            session_records.append(record)

        except (KeyboardInterrupt, AuditAbortedError):
            print("\n\nSession interrupted by user. Finalizing audit summary...")
            break
        except Exception as e:
            print(f"\nError inspecting scramble {scramble}: {e}", file=sys.stderr)
            continue

    # Session Summary Report
    total_evaluated = len(session_records)
    if total_evaluated == 0:
        print("\nNo audit cases recorded in this session.")
        return 0

    agree_count = sum(1 for r in session_records if r.verdict == "agree")
    disagree_count = total_evaluated - agree_count
    agreement_rate = (agree_count / total_evaluated) * 100.0

    print("\n" + "=" * 80)
    print("                    AUDIT SESSION SUMMARY REPORT")
    print("=" * 80)
    print(f"  Total Cases Evaluated:   {total_evaluated}")
    print(f"  Solver-Human Agreement:  {agree_count}/{total_evaluated} ({agreement_rate:.1f}%)")
    print(f"  Discrepancies Flagged:   {disagree_count}/{total_evaluated}")

    if disagree_count > 0:
        print("-" * 80)
        print("  Flagged Flaw Categories:")
        cat_counts: Dict[str, int] = {}
        for r in session_records:
            if r.verdict == "disagree" and r.failure_category:
                cat_counts[r.failure_category] = cat_counts.get(r.failure_category, 0) + 1
        for cat, cnt in cat_counts.items():
            print(f"    • {cat.replace('_', ' ').title():<28} {cnt} case{'s' if cnt != 1 else ''}")

    print("=" * 80)
    print(f"  Findings successfully saved to:")
    print(f"    • {DEFAULT_AUDIT_MD_PATH}")
    print(f"    • {DEFAULT_AUDIT_JSONL_PATH}")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())

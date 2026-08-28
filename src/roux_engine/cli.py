"""CLI entry point for Roux solve segmentation and analysis."""

from __future__ import annotations
import argparse
import json
import sys
from typing import Optional

from .segmenter.segmenter import RouxSegmenter
from .segmenter.models import SegmentedSolve


def format_solve_report(solve: SegmentedSolve) -> str:
    """Formats a SegmentedSolve into a clean speedcubing terminal report."""
    lines = []
    lines.append("=" * 80)
    lines.append("                      ROUX SOLVE SEGMENTATION REPORT")
    lines.append("=" * 80)
    lines.append(f"  Scramble: {solve.scramble}")
    if solve.total_time_ms is not None:
        duration_sec = solve.total_time_ms / 1000.0
        tps_str = f" @ {solve.tps:.2f} TPS" if solve.tps else ""
        lines.append(f"  Time:     {duration_sec:.2f}s{tps_str}")
    lines.append(f"  Total:    {solve.total_moves_stm} STM moves ({solve.solution_moves_count} parsed events)")
    lines.append(f"  Status:   {'VALID ROUX SOLVE' if solve.is_valid else 'INVALID / INCOMPLETE'}")

    if not solve.is_valid:
        lines.append("-" * 80)
        lines.append("  Warnings / Errors:")
        for w in solve.warnings:
            lines.append(f"    • {w}")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append("-" * 80)
    lines.append("Phase Breakdown:")
    lines.append("-" * 80)

    # 1. First Block
    if solve.fb:
        ori_str = ""
        if solve.fb.orientation:
            ori_rot = f" ({solve.fb.orientation.rotations})" if solve.fb.orientation.rotations else ""
            ori_str = f" [{solve.fb.orientation.bottom_color.name} bottom, {solve.fb.orientation.left_color.name} left{ori_rot}]"
        fb_s = "s" if solve.fb.move_count_stm != 1 else ""
        lines.append(f"  • First Block (FB):   {solve.fb.move_count_stm:>2} move{fb_s}  [{solve.fb.moves_str}]")
        lines.append(f"    - Orientation:     {ori_str.strip() if ori_str else 'Canonical'}")
        concurrent = []
        if solve.fb.concurrent_sb.dr_solved:
            concurrent.append("DR edge")
        if solve.fb.concurrent_sb.sb_square_solved:
            concurrent.append("SB square")
        elif solve.fb.concurrent_sb.sb_pair_solved:
            concurrent.append("SB pair")
        lines.append(f"    - Concurrent SB:   {', '.join(concurrent) if concurrent else 'None'}")
        lines.append("")

    # 2. Second Block
    if solve.sb:
        sb_s = "s" if solve.sb.move_count_stm != 1 else ""
        lines.append(f"  • Second Block (SB):  {solve.sb.move_count_stm:>2} move{sb_s}  [{solve.sb.moves_str}]")
        dr_idx_str = f"Move {solve.sb.dr_placement_idx - solve.sb.start_move_idx + 1}" if solve.sb.dr_placement_idx is not None else "N/A"
        lines.append(f"    - DR Placement:    {dr_idx_str}")
        lines.append(f"    - Pair 1:          {solve.sb.pair1_type.capitalize()} pair")
        lines.append(f"    - Rotations:       {solve.sb.rotation_count} rotation(s)")
        if solve.sb.non_ergonomic_moves:
            lines.append(f"    - Non-Ergonomic:   {', '.join(solve.sb.non_ergonomic_moves)}")
        lines.append("")

    # 3. CMLL
    if solve.cmll:
        cmll_s = "s" if solve.cmll.move_count_stm != 1 else ""
        lines.append(f"  • CMLL:               {solve.cmll.move_count_stm:>2} move{cmll_s}  [{solve.cmll.moves_str}]")
        lines.append(f"    - Group / Case:    {solve.cmll.group} ({solve.cmll.case_id})")
        lines.append(f"    - Pre-AUF:         '{solve.cmll.pre_auf}'" if solve.cmll.pre_auf else "    - Pre-AUF:         None")
        lines.append(f"    - Alg Type:        {'Standard 2H/OH alg' if solve.cmll.is_standard_alg else 'Alternative / Custom'}")
        lines.append("")

    # 4. LSE
    if solve.lse:
        lse_s = "s" if solve.lse.move_count_stm != 1 else ""
        lines.append(f"  • LSE (Step 4):       {solve.lse.move_count_stm:>2} move{lse_s}  [{solve.lse.moves_str}]")
        if solve.lse.step_4a:
            s4a = "s" if solve.lse.step_4a.move_count_stm != 1 else ""
            lines.append(f"    - Step 4a (EO):    {solve.lse.step_4a.move_count_stm} move{s4a} ({solve.lse.step_4a.variant}, {solve.lse.step_4a.center_state}) [{solve.lse.step_4a.moves_str}]")
        if solve.lse.step_4b:
            s4b = "s" if solve.lse.step_4b.move_count_stm != 1 else ""
            lines.append(f"    - Step 4b (UL/UR): {solve.lse.step_4b.move_count_stm} move{s4b}{' (Skipped)' if solve.lse.step_4b.skipped else ''} [{solve.lse.step_4b.moves_str}]")
        if solve.lse.step_4c:
            s4c = "s" if solve.lse.step_4c.move_count_stm != 1 else ""
            lines.append(f"    - Step 4c (M/C):   {solve.lse.step_4c.move_count_stm} move{s4c} ({solve.lse.step_4c.case}) [{solve.lse.step_4c.moves_str}]")

    lines.append("=" * 80)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Roux Speedcube Solve Segmenter and Diagnostics CLI"
    )
    parser.add_argument("-s", "--scramble", type=str, help="Scramble move sequence")
    parser.add_argument("-sol", "--solution", type=str, help="Solution move sequence")
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON format")
    parser.add_argument("--color-neutral", action="store_true", help="Enable 24-orientation full color neutrality")

    args = parser.parse_args(argv)

    scramble = args.scramble
    solution = args.solution

    # Interactive prompt if flags not provided
    if not scramble:
        try:
            scramble = input("Enter Scramble: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not solution:
        try:
            solution = input("Enter Solution: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1

    if not scramble or not solution:
        print("Error: Both scramble and solution must be provided.", file=sys.stderr)
        return 1

    segmenter = RouxSegmenter(full_color_neutral=args.color_neutral)
    result = segmenter.segment(scramble, solution)

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(format_solve_report(result))

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())

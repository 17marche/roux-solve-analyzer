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
    if solve.total_time_ms is not None and solve.total_time_ms > 0:
        duration_sec = solve.total_time_ms / 1000.0
        tps_str = f" [{solve.tps:.2f} TPS]" if solve.tps is not None else ""
        lines.append(f"  Total:    {solve.total_moves_stm} STM moves [{duration_sec:.2f} sec]{tps_str}")
    else:
        lines.append(f"  Total:    {solve.total_moves_stm} STM moves")
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

    # 0. Inspection
    inspection_str = "None"
    if solve.fb and solve.fb.orientation and solve.fb.orientation.rotations:
        inspection_str = solve.fb.orientation.rotations
    lines.append(f"  • {'Inspection:':<22} {inspection_str}")
    lines.append("")

    # 1. First Block
    if solve.fb:
        fb_moves = solve.fb.moves_str
        if solve.fb.orientation and solve.fb.orientation.rotations:
            rot_prefix = solve.fb.orientation.rotations
            if fb_moves.startswith(rot_prefix):
                fb_moves = fb_moves[len(rot_prefix):].strip()

        if solve.fb.move_count_stm == 0:
            fb_header = "0 moves"
        else:
            fb_s = "s" if solve.fb.move_count_stm != 1 else ""
            fb_header = f"{fb_moves} // {solve.fb.move_count_stm} move{fb_s}" if fb_moves else f"0 moves"

        lines.append(f"  • {'First Block (FB):':<22} {fb_header}")

        ori_str = "ORANGE - WHITE"
        if solve.fb.orientation:
            ori_str = f"{solve.fb.orientation.left_color.name} - {solve.fb.orientation.bottom_color.name}"
        lines.append(f"    - {'Orientation:':<20} {ori_str}")
        lines.append("")

    # 2. Second Block
    if solve.sb:
        sb_parts = [p for p in [solve.sb.dr_moves_str, solve.sb.pair1_moves_str, solve.sb.pair2_moves_str] if p]
        sb_moves_chain = " / ".join(sb_parts)
        if solve.sb.move_count_stm == 0:
            sb_header = "0 moves"
        else:
            sb_s = "s" if solve.sb.move_count_stm != 1 else ""
            sb_header = f"{sb_moves_chain} // {solve.sb.move_count_stm} move{sb_s}" if sb_moves_chain else f"{solve.sb.moves_str} // {solve.sb.move_count_stm} move{sb_s}"

        lines.append(f"  • {'Second Block (SB):':<22} {sb_header}")

        dr_s = "s" if solve.sb.dr_moves_stm != 1 else ""
        lines.append(f"    - {'DR Placement:':<20} {solve.sb.dr_moves_stm} move{dr_s}")

        pair1_label = "Front pair" if solve.sb.pair1_type == "front" else ("Back pair" if solve.sb.pair1_type == "back" else ("Simultaneous pairs" if solve.sb.pair1_type == "both_simultaneous" else solve.sb.pair1_type.capitalize()))
        p1_s = "s" if solve.sb.pair1_moves_stm != 1 else ""
        lines.append(f"    - {'First Pair:':<20} {pair1_label} - {solve.sb.pair1_moves_stm} move{p1_s}")

        pair2_label = "Back pair" if solve.sb.pair2_type == "back" else ("Front pair" if solve.sb.pair2_type == "front" else ("Simultaneous pairs" if solve.sb.pair2_type == "both_simultaneous" else solve.sb.pair2_type.capitalize()))
        p2_s = "s" if solve.sb.pair2_moves_stm != 1 else ""
        lines.append(f"    - {'Second Pair:':<20} {pair2_label} - {solve.sb.pair2_moves_stm} move{p2_s}")

        if solve.sb.rotation_count > 0:
            rot_s = "s" if solve.sb.rotation_count != 1 else ""
            lines.append(f"    - {'Rotations:':<20} {solve.sb.rotation_count} rotation{rot_s}")
        if solve.sb.non_ergonomic_moves:
            lines.append(f"    - {'Non-Ergonomic:':<20} {', '.join(solve.sb.non_ergonomic_moves)}")
        lines.append("")

    # 3. CMLL
    if solve.cmll:
        if solve.cmll.move_count_stm == 0 or solve.cmll.case_id == "solved":
            cmll_header = "0 moves"
            group_str = "Skip"
            pre_auf_str = "Skip"
            alg_type_str = "Skip"
        else:
            cmll_s = "s" if solve.cmll.move_count_stm != 1 else ""
            cmll_header = f"{solve.cmll.moves_str} // {solve.cmll.move_count_stm} move{cmll_s}"
            group_str = f"{solve.cmll.group} ({solve.cmll.case_id})"
            pre_auf_str = f"'{solve.cmll.pre_auf}'" if solve.cmll.pre_auf else "None"
            alg_type_str = "Standard 2H/OH alg" if solve.cmll.is_standard_alg else "Alternative / Custom"

        lines.append(f"  • {'CMLL:':<22} {cmll_header}")
        lines.append(f"    - {'Group / Case:':<20} {group_str}")
        lines.append(f"    - {'Pre-AUF:':<20} {pre_auf_str}")
        lines.append(f"    - {'Alg Type:':<20} {alg_type_str}")
        lines.append("")

    # 4. LSE
    if solve.lse:
        lse_parts = []
        if solve.lse.step_4a and solve.lse.step_4a.moves_str:
            lse_parts.append(solve.lse.step_4a.moves_str)
        if solve.lse.step_4b and solve.lse.step_4b.moves_str:
            lse_parts.append(solve.lse.step_4b.moves_str)
        if solve.lse.step_4c and solve.lse.step_4c.moves_str:
            lse_parts.append(solve.lse.step_4c.moves_str)

        lse_chain = " / ".join(lse_parts)
        if solve.lse.move_count_stm == 0:
            lse_header = "0 moves"
        else:
            lse_s = "s" if solve.lse.move_count_stm != 1 else ""
            lse_header = f"{lse_chain} // {solve.lse.move_count_stm} move{lse_s}" if lse_chain else f"{solve.lse.moves_str} // {solve.lse.move_count_stm} move{lse_s}"

        lines.append(f"  • {'LSE (Step 4):':<22} {lse_header}")

        if solve.lse.step_4a:
            if solve.lse.step_4a.move_count_stm == 0:
                s4a_str = "0 moves // Skip"
            else:
                s4a_s = "s" if solve.lse.step_4a.move_count_stm != 1 else ""
                center_desc = "oriented centers" if solve.lse.step_4a.center_state == "axis_aligned" else "misaligned centers"
                s4a_str = f"{solve.lse.step_4a.move_count_stm} move{s4a_s} // {solve.lse.step_4a.variant} - {center_desc}"
            lines.append(f"    - {'Step 4a:':<20} {s4a_str}")

        if solve.lse.step_4b:
            if solve.lse.step_4b.skipped or solve.lse.step_4b.move_count_stm == 0:
                s4b_str = "0 moves // Skip"
            else:
                s4b_s = "s" if solve.lse.step_4b.move_count_stm != 1 else ""
                s4b_str = f"{solve.lse.step_4b.move_count_stm} move{s4b_s} // UL/UR solved"
            lines.append(f"    - {'Step 4b:':<20} {s4b_str}")

        if solve.lse.step_4c:
            if solve.lse.step_4c.move_count_stm == 0 or solve.lse.step_4c.case == "solved":
                s4c_str = "0 moves // Skip"
            else:
                s4c_s = "s" if solve.lse.step_4c.move_count_stm != 1 else ""
                s4c_str = f"{solve.lse.step_4c.move_count_stm} move{s4c_s} // {solve.lse.step_4c.case}"
            lines.append(f"    - {'Step 4c:':<20} {s4c_str}")

    lines.append("=" * 80)
    return "\n".join(lines)


def verify_pdb_completeness(pdb_path: Path, quiet: bool = False) -> None:
    """Verifies that the First Block PDB file is complete, valid, and fully reachable."""
    from .solver.fb_pdb import FBPDB, FILE_SIZE_BYTES, TOTAL_STATES
    import numpy as np

    if not quiet:
        print("-" * 80)
        print("  Verifying database completeness and integrity...")

    pdb = FBPDB(pdb_path=pdb_path)
    if pdb.size != TOTAL_STATES:
        raise ValueError(f"State count mismatch: {pdb.size} != {TOTAL_STATES}")
    if pdb.file_size_bytes != FILE_SIZE_BYTES:
        raise ValueError(f"File size mismatch: {pdb.file_size_bytes} != {FILE_SIZE_BYTES}")
    if pdb.get_distance(0) != 0:
        raise ValueError(f"Canonical solved distance mismatch: {pdb.get_distance(0)} != 0")

    # Verify no nibble exceeds maximum God's number depth of 9 (or unvisited 0xF)
    raw_bytes = np.asarray(pdb._data)
    low_nibbles = raw_bytes & 0x0F
    high_nibbles = raw_bytes >> 4
    if np.any(low_nibbles > 9) or np.any(high_nibbles > 9):
        raise ValueError("Database contains corrupted or unvisited distance values (> 9)")

    if not quiet:
        print(f"  Database verified: all {TOTAL_STATES:,} states reachable within <= 9 moves.")
        print("  Canonical solved state distance is 0.")
        print("=" * 80)


def handle_generate_fb_pdb(argv: list[str]) -> int:
    """Handles the generate-fb-pdb subcommand."""
    import time
    from pathlib import Path
    from .solver.pdb_generator import generate_fb_pdb, get_default_pdb_path
    from .solver.fb_pdb import FILE_SIZE_BYTES, TOTAL_STATES

    parser = argparse.ArgumentParser(
        prog="roux generate-fb-pdb",
        description="Generate the canonical 5.32M state First Block Pattern Database (PDB)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom destination path for fb_pdb.bin (default: src/roux_engine/data/fb_pdb.bin)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing database file completeness and integrity without regenerating"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Alias to ensure completeness verification runs after generation (default: True)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress depth-by-depth progress output"
    )
    args = parser.parse_args(argv)

    out_path = Path(args.output) if args.output else get_default_pdb_path()

    if args.verify_only:
        if not args.quiet:
            print("=" * 80)
            print("         FIRST BLOCK PATTERN DATABASE (PDB) VERIFICATION")
            print("=" * 80)
            print(f"  Target file:  {out_path}")
        verify_pdb_completeness(out_path, quiet=args.quiet)
        return 0

    if not args.quiet:
        print("=" * 80)
        print("           FIRST BLOCK PATTERN DATABASE (PDB) GENERATOR")
        print("=" * 80)
        print(f"  Target file:  {out_path}")
        print(f"  Total states: {TOTAL_STATES:,}")
        print(f"  Moveset:      <U, D, R, F, B, r, M> (21 moves, ADR-0001)")
        print(f"  Storage:      4-bit nibbles ({FILE_SIZE_BYTES:,} bytes)")
        print("-" * 80)
        print(f"  {'Depth':<7} | {'New States':<14} | {'Total States':<14} | {'Layer Time':<10}")
        print("-" * 80)

    t0 = time.time()

    def progress_callback(depth: int, num_new: int, total: int, elapsed: float) -> None:
        if not args.quiet:
            print(f"  Depth {depth:<2} | {num_new:>14,} | {total:>14,} | {elapsed:>8.2f}s", flush=True)

    generated_path = generate_fb_pdb(output_path=out_path, progress_callback=progress_callback)
    total_time = time.time() - t0

    if not args.quiet:
        print("-" * 80)
        print(f"  Generation finished in {total_time:.2f}s.")
        print(f"  Saved to {generated_path} ({generated_path.stat().st_size:,} bytes).")

    # Verify table completeness by default
    verify_pdb_completeness(generated_path, quiet=args.quiet)

    return 0



def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "generate-fb-pdb":
        return handle_generate_fb_pdb(argv[1:])

    parser = argparse.ArgumentParser(
        prog="roux",
        description="Roux Speedcube Solve Segmenter and Diagnostics CLI",
        epilog="Subcommands:\n  generate-fb-pdb    Generate First Block Pattern Database",
    )
    parser.add_argument("-s", "--scramble", type=str, help="Scramble move sequence")
    parser.add_argument("-sol", "--solution", type=str, help="Solution move sequence")
    parser.add_argument("-t", "--time", type=float, default=None, help="Solve time in seconds (optional)")
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON format")
    parser.add_argument("--color-neutral", action="store_true", help="Enable 24-orientation full color neutrality")

    args = parser.parse_args(argv)


    scramble = args.scramble
    solution = args.solution
    time_sec = args.time

    # Interactive prompt if flags not provided
    is_interactive = False
    if not scramble:
        is_interactive = True
        try:
            scramble = input("Enter Scramble: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
    if not solution:
        is_interactive = True
        try:
            solution = input("Enter Solution: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1

    # Optional time prompt if interactive (when time not given via CLI flag)
    if is_interactive and time_sec is None and sys.stdin.isatty() and not args.json:
        try:
            time_input = input("Enter Time in seconds (optional, press Enter to skip): ").strip()
            if time_input:
                try:
                    time_sec = float(time_input)
                except ValueError:
                    print("Warning: Invalid time format, skipping time.", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            pass

    if not scramble or not solution:
        print("Error: Both scramble and solution must be provided.", file=sys.stderr)
        return 1

    segmenter = RouxSegmenter(full_color_neutral=args.color_neutral)
    result = segmenter.segment(scramble, solution, time_sec=time_sec)

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(format_solve_report(result))

    return 0 if result.is_valid else 1


if __name__ == "__main__":
    sys.exit(main())

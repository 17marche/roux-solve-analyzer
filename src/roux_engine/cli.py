"""CLI entry point for Roux solve segmentation and analysis."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Union

from .segmenter.segmenter import RouxSegmenter
from .segmenter.models import SegmentedSolve
from .solver.scramble_solver import FullSolveResult
from .ergonomics.models import FlowScore, HandProfile


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
            fb_header = f"{fb_moves} // {solve.fb.move_count_stm} move{fb_s}" if fb_moves else "0 moves"

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
        print("  Moveset:      <U, D, R, F, B, r, M> (21 moves, ADR-0001)")
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


def verify_sb_pdb_completeness(sb_path: Path, rbs_path: Path, rfs_path: Path, quiet: bool = False) -> None:
    """Verifies that all three Second Block PDB files are complete, valid, and fully reachable."""
    from .solver.sb_pdb import (
        SBPDB,
        RightBackSquarePDB,
        RightFrontSquarePDB,
        SB_FILE_SIZE_BYTES,
        RBS_FILE_SIZE_BYTES,
        RFS_FILE_SIZE_BYTES,
    )
    from .solver.sb_indexer import (
        TOTAL_SB_STATES,
        TOTAL_RBS_STATES,
        TOTAL_RFS_STATES,
    )
    import numpy as np

    if not quiet:
        print("-" * 80)
        print("  Verifying Second Block pattern databases completeness and integrity...")

    sb_pdb = SBPDB(pdb_path=sb_path)
    sb_pdb.verify_integrity(max_depth=14)

    rbs_pdb = RightBackSquarePDB(pdb_path=rbs_path)
    rbs_pdb.verify_integrity(max_depth=10)

    rfs_pdb = RightFrontSquarePDB(pdb_path=rfs_path)
    rfs_pdb.verify_integrity(max_depth=10)

    total_bytes = SB_FILE_SIZE_BYTES + RBS_FILE_SIZE_BYTES + RFS_FILE_SIZE_BYTES
    if not quiet:
        print("  All databases verified:")
        print(f"    • Full SB:      {TOTAL_SB_STATES:,} states within <= 14 moves ({SB_FILE_SIZE_BYTES:,} bytes)")
        print(f"    • Right Back:   {TOTAL_RBS_STATES:,} states within <= 10 moves ({RBS_FILE_SIZE_BYTES:,} bytes)")
        print(f"    • Right Front:  {TOTAL_RFS_STATES:,} states within <= 10 moves ({RFS_FILE_SIZE_BYTES:,} bytes)")
        print(f"    • Total on-disk footprint: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB < 600 KB)")
        print("    • Solved state distance is 0 across all tables.")
        print("=" * 80)


def handle_generate_sb_pdb(argv: list[str]) -> int:
    """Handles the generate-sb-pdb subcommand."""
    import time
    from pathlib import Path
    from .solver.sb_pdb_generator import (
        generate_all_sb_pdbs,
        get_default_sb_pdb_path,
        get_default_rbs_pdb_path,
        get_default_rfs_pdb_path,
    )
    from .solver.sb_pdb import (
        SB_FILE_SIZE_BYTES,
        RBS_FILE_SIZE_BYTES,
        RFS_FILE_SIZE_BYTES,
    )
    from .solver.sb_indexer import (
        TOTAL_SB_STATES,
        TOTAL_RBS_STATES,
        TOTAL_RFS_STATES,
    )

    parser = argparse.ArgumentParser(
        prog="roux generate-sb-pdb",
        description="Generate the canonical Second Block and Right Square Pattern Databases (PDBs)."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Custom destination directory for binary files (default: src/roux_engine/data)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing database files completeness and integrity without regenerating"
    )
    parser.add_argument(
        "--verify",
        dest="verify",
        action="store_true",
        default=True,
        help="Ensure completeness verification runs after generation (default: True)"
    )
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="Skip completeness verification after generation"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress depth-by-depth progress output"
    )
    args = parser.parse_args(argv)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        sb_path = out_dir / "sb_pdb.bin"
        rbs_path = out_dir / "rbs_pdb.bin"
        rfs_path = out_dir / "rfs_pdb.bin"
    else:
        sb_path = get_default_sb_pdb_path()
        rbs_path = get_default_rbs_pdb_path()
        rfs_path = get_default_rfs_pdb_path()

    if args.verify_only:
        if not args.quiet:
            print("=" * 80)
            print("     SECOND BLOCK PATTERN DATABASES (PDB) VERIFICATION")
            print("=" * 80)
            print(f"  Target files: {sb_path}, {rbs_path}, {rfs_path}")
        verify_sb_pdb_completeness(sb_path, rbs_path, rfs_path, quiet=args.quiet)
        return 0

    if not args.quiet:
        print("=" * 80)
        print("       SECOND BLOCK PATTERN DATABASES (PDB) GENERATOR")
        print("=" * 80)
        print("  Moveset:      <R, U, r, M> (12 moves, preserving First Block)")
        print(f"  Full SB:      {TOTAL_SB_STATES:,} states ({SB_FILE_SIZE_BYTES:,} bytes)")
        print(f"  Right Back:   {TOTAL_RBS_STATES:,} states ({RBS_FILE_SIZE_BYTES:,} bytes)")
        print(f"  Right Front:  {TOTAL_RFS_STATES:,} states ({RFS_FILE_SIZE_BYTES:,} bytes)")
        print(f"  Total space:  {SB_FILE_SIZE_BYTES + RBS_FILE_SIZE_BYTES + RFS_FILE_SIZE_BYTES:,} bytes (~549.2 KB)")
        print("-" * 80)

    t0 = time.time()

    def progress_callback(name: str, depth: int, num_new: int, total: int, elapsed: float) -> None:
        if not args.quiet:
            print(f"  [{name.upper():<3}] Depth {depth:<2} | {num_new:>10,} | {total:>10,} | {elapsed:>6.2f}s", flush=True)

    out_sb, out_rbs, out_rfs = generate_all_sb_pdbs(
        output_dir=args.output_dir,
        progress_callback=progress_callback,
    )
    total_time = time.time() - t0

    if not args.quiet:
        print("-" * 80)
        print(f"  Generation finished in {total_time:.2f}s.")
        print(f"  Saved tables to {out_sb.parent}.")

    if args.verify:
        verify_sb_pdb_completeness(out_sb, out_rbs, out_rfs, quiet=args.quiet)
    return 0


def format_solver_report(result: FullSolveResult) -> str:
    """Formats a FullSolveResult into a clean terminal report."""
    lines = []
    lines.append("=" * 80)
    lines.append("                      ROUX SCRAMBLE SOLVER REPORT")
    lines.append("=" * 80)
    lines.append(f"  Scramble: {result.scramble}")
    dur_str = f" [{result.duration_ms:.2f} ms]"
    lines.append(f"  Total:    {result.total_stm} STM moves // {result.total_e_stm:.2f} E-STM{dur_str}")
    lines.append(f"  Flow:     {result.kinematic_efficiency:.1f}% Kinematic Flow Efficiency")
    if result.profile == "2H" and result.m_slice_hand:
        profile_desc = f"2H ({result.m_slice_hand}-handed M-slice)"
    else:
        profile_desc = result.profile
    lines.append(f"  Profile:  {profile_desc} [ranked by {result.rank_by}]")
    status_str = "VALID ROUX SOLVE" if result.is_valid else "INVALID / INCOMPLETE"
    lines.append(f"  Status:   {status_str} (style={result.style})")
    lines.append("-" * 80)
    lines.append("Phase Breakdown:")
    lines.append("-" * 80)

    # 0. Inspection
    insp = result.fb.inspection_rotation if result.fb.inspection_rotation else "None"
    lines.append(f"  • {'Inspection:':<22} {insp}")
    lines.append("")

    # 1. FB
    fb_moves_str = " ".join(result.fb.moves) if result.fb.moves else "0 moves"
    fb_s = "s" if result.fb.move_count != 1 else ""
    fb_e_val = f" ({result.fb.e_stm:.2f} E-STM)" if result.fb.e_stm is not None else ""
    lines.append(f"  • {'First Block (FB):':<22} {fb_moves_str} // {result.fb.move_count} move{fb_s}{fb_e_val}")
    lines.append(f"    - {'Orientation:':<20} {result.fb.orientation}")
    if result.fb.e_stm is not None:
        lines.append(f"    - {'E-STM:':<20} {result.fb.e_stm:.2f}")
    lines.append("")

    # 2. SB
    sb_moves_str = " ".join(result.sb.moves) if result.sb.moves else "0 moves"
    sb_s = "s" if result.sb.move_count != 1 else ""
    sb_e_val = f" ({result.sb.e_stm:.2f} E-STM)" if result.sb.e_stm is not None else ""
    style_label = "Free Blockbuilding" if result.sb.style == "free" else (
        "Classical Standard" if result.sb.style == "classical" else "Square + Pair"
    )
    lines.append(f"  • {'Second Block (SB):':<22} {sb_moves_str} // {result.sb.move_count} move{sb_s}{sb_e_val}")
    lines.append(f"    - {'Paradigm:':<20} {style_label}")
    if result.sb.e_stm is not None:
        lines.append(f"    - {'E-STM:':<20} {result.sb.e_stm:.2f}")
    lines.append("")

    # 3. CMLL
    cmll_moves_str = " ".join(result.cmll_moves) if result.cmll_moves else "0 moves // Skip"
    cmll_s = "s" if result.cmll_stm != 1 else ""
    cmll_e_val = f" ({result.cmll_e_stm:.2f} E-STM)"
    lines.append(f"  • {'CMLL:':<22} {cmll_moves_str} // {result.cmll_stm} move{cmll_s}{cmll_e_val}")
    pre_auf_desc = f"'{result.cmll_pre_auf}'" if result.cmll_pre_auf else "None"
    lines.append(f"    - {'Pre-AUF:':<20} {pre_auf_desc}")
    post_auf_desc = f"'{result.cmll_post_auf}'" if result.cmll_post_auf else "None"
    lines.append(f"    - {'Post-AUF:':<20} {post_auf_desc}")
    lines.append(f"    - {'E-STM:':<20} {result.cmll_e_stm:.2f}")
    lines.append("")

    # 4. LSE
    lse_moves_str = " ".join(result.lse.moves) if result.lse.moves else "0 moves // Skip"
    lse_s = "s" if result.lse.move_count != 1 else ""
    lse_e_val = f" ({result.lse.e_stm:.2f} E-STM)" if result.lse.e_stm is not None else ""
    lines.append(f"  • {'Last Six Edges (LSE):':<22} {lse_moves_str} // {result.lse.move_count} move{lse_s}{lse_e_val}")
    lines.append(f"    - {'Target:':<20} {result.lse.target}")
    if result.lse.e_stm is not None:
        lines.append(f"    - {'E-STM:':<20} {result.lse.e_stm:.2f}")
    lines.append("-" * 80)

    # Full Solution
    lines.append("Full Solution:")
    lines.append(f"  {result.full_moves_str}")
    lines.append("-" * 80)

    # 3D Interactive Visualization
    lines.append("3D Interactive Visualization (alg.cubing.net):")
    lines.append(f"  {result.alg_cubing_url}")
    lines.append("=" * 80)
    if result.style == "free":
        lines.append("  💡 Tip: Run with `--style classical` for human-mimetic pair building!")
        lines.append("=" * 80)
    return "\n".join(lines)


def handle_solve(argv: list[str]) -> int:
    """Handles the roux solve subcommand."""
    from .solver.scramble_solver import solve_scramble

    parser = argparse.ArgumentParser(
        prog="roux solve",
        description="Solve a Rubik's Cube scramble from scratch using the 4-phase Roux method.",
    )
    parser.add_argument(
        "scramble_pos",
        nargs="?",
        type=str,
        default=None,
        help="Scramble move sequence string (positional argument)",
    )
    parser.add_argument("-s", "--scramble", type=str, default=None, help="Scramble move sequence string")
    parser.add_argument(
        "--style",
        type=str,
        choices=["free", "classical", "square_pair"],
        default="free",
        help="SB paradigm: 'free' (optimal), 'classical' (human-style), or 'square_pair'",
    )
    parser.add_argument(
        "--rank-by",
        type=str,
        choices=["e_stm", "stm"],
        default="e_stm",
        help="Candidate ranking metric: 'e_stm' (ergonomic, default) or 'stm' (raw movecount)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["2H", "OH"],
        default="2H",
        help="Solving style profile: '2H' (Two-Handed, default) or 'OH' (One-Handed)",
    )
    parser.add_argument(
        "--m-slice-hand",
        type=str,
        choices=["right", "left"],
        default="right",
        help="Hand used for M-slice turns in 2H mode: 'right' (default) or 'left'",
    )
    parser.add_argument("-j", "--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args(argv)
    scramble = args.scramble if args.scramble is not None else args.scramble_pos
    if not scramble:
        try:
            scramble = input("Enter Scramble: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1

    if not scramble:
        print("Error: Scramble must be provided.", file=sys.stderr)
        return 1

    hand_profile = HandProfile(solving_mode=args.profile, m_slice_hand=args.m_slice_hand)
    result = solve_scramble(
        scramble=scramble,
        style=args.style,
        rank_by=args.rank_by,
        hand_profile=hand_profile,
    )
    if args.json:
        print(result.to_json(indent=2))
    else:
        print(format_solver_report(result))
    return 0 if result.is_valid else 1


def handle_analyze(argv: list[str]) -> int:
    """Handles the roux analyze subcommand."""
    parser = argparse.ArgumentParser(
        prog="roux analyze",
        description="Roux Speedcube Solve Segmenter and Diagnostics CLI",
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


def format_flow_report(
    score: FlowScore,
    profile: Union[str, HandProfile] = "2H",
    tempo: Optional[float] = None,
    source_name: Optional[str] = None,
) -> str:
    """Formats a FlowScore into a clean speedcubing terminal report."""
    lines = []
    lines.append("=" * 80)
    lines.append("                      ROUX BIOMECHANICAL FLOW REPORT")
    lines.append("=" * 80)
    if source_name:
        lines.append(f"  Source:     {source_name}")
    prof_mode = profile.solving_mode if isinstance(profile, HandProfile) else str(profile)
    lines.append(f"  Profile:    {prof_mode} (solving mode)")
    if isinstance(profile, HandProfile) and profile.m_slice_hand == "left":
        lines.append("  M-Slice:    left hand")
    if tempo is not None:
        tps_str = f" [{1.0 / tempo:.2f} TPS]" if tempo > 0 else ""
        lines.append(f"  Tempo:      {tempo:.2f} sec/move{tps_str}")
    lines.append(f"  Raw STM:    {score.raw_stm} moves")
    lines.append(f"  Effective STM (E-STM):       {score.e_stm:.3f}")
    lines.append(f"  Kinematic Flow Efficiency:   {score.kinematic_efficiency:.1f}%")
    lines.append(f"  Regrip Count:                {score.regrip_count}")

    if score.macro_triggers:
        lines.append(f"  Macro Triggers:              {', '.join(score.macro_triggers)}")
    else:
        lines.append("  Macro Triggers:              None")

    if score.turning_ratio is not None or score.rhythm_cv is not None or score.stream_flow_index is not None:
        lines.append("-" * 80)
        lines.append("Stream Rhythm & Continuity:")
        lines.append("-" * 80)
        if score.turning_ratio is not None:
            lines.append(f"  • Turning Ratio:             {score.turning_ratio * 100:.1f}% active turning")
        if score.rhythm_cv is not None:
            lines.append(f"  • Rhythm Consistency (CV):   {score.rhythm_cv:.3f}")
        if score.stream_flow_index is not None:
            lines.append(f"  • Stream Flow Index:         {score.stream_flow_index:.1f} / 100")
        if score.pause_breakdown:
            lines.append("  • Pause Breakdown:")
            for ptype, count in score.pause_breakdown.items():
                lines.append(f"    - {ptype:<24}: {count}")

    if score.per_move_analysis:
        lines.append("-" * 80)
        lines.append("Per-Move Biomechanical Breakdown:")
        lines.append("-" * 80)
        lines.append(f"  {'#':<4} {'Move':<6} {'Grip (Before -> After)':<25} {'Regrip':<8} {'Effort':<8} {'Pause Type'}")
        lines.append("  " + "-" * 76)
        for idx, m in enumerate(score.per_move_analysis, 1):
            grip_before_str = m.grip_before.value if hasattr(m.grip_before, "value") else str(m.grip_before)
            grip_after_str = m.grip_after.value if hasattr(m.grip_after, "value") else str(m.grip_after)
            grip_str = f"{grip_before_str} -> {grip_after_str}"
            regrip_str = "YES" if m.regrip else "-"
            pause_str = str(m.pause_type) if m.pause_type else "-"
            lines.append(f"  {idx:<4} {m.move:<6} {grip_str:<25} {regrip_str:<8} {m.transition_effort:<8.2f} {pause_str}")

    lines.append("=" * 80)
    return "\n".join(lines)


def handle_flow(argv: list[str]) -> int:
    """Handles the roux flow subcommand."""
    from .ergonomics.flow_scorer import FlowScorer

    parser = argparse.ArgumentParser(
        prog="roux flow",
        description="Biomechanical flow, Effective STM (E-STM), and smart-cube stream rhythm evaluation.",
    )
    parser.add_argument(
        "moves_or_file",
        nargs="?",
        type=str,
        default=None,
        help="Move sequence string, or path to a text file / JSON smart-cube stream",
    )
    parser.add_argument(
        "-m", "--moves",
        type=str,
        default=None,
        help="Move sequence string (alternative to positional argument)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["2H", "OH"],
        default="2H",
        help="Solving style profile: '2H' (Two-Handed) or 'OH' (One-Handed) (default: 2H)",
    )
    parser.add_argument(
        "--m-slice-hand",
        type=str,
        choices=["right", "left"],
        default="right",
        help="Hand used for M-slice turns in 2H mode: 'right' (default) or 'left'",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=None,
        help="Personal seconds per move tempo (e.g. 0.25 for 4 TPS)",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output raw JSON format",
    )

    args = parser.parse_args(argv)
    if args.moves is not None:
        raw_input = args.moves
    elif args.moves_or_file is not None:
        raw_input = args.moves_or_file
    else:
        raw_input = None
        if not sys.stdin.isatty():
            try:
                raw_input = sys.stdin.read().strip()
            except (OSError, ValueError):
                pass
        if not raw_input:
            try:
                raw_input = input("Enter moves or JSON stream: ").strip()
            except (EOFError, KeyboardInterrupt):
                return 1

    input_text = raw_input.strip() if raw_input else ""
    source_name = None

    if input_text:
        p = Path(input_text)
        try:
            is_file = p.is_file()
        except (OSError, ValueError):
            is_file = False

        if is_file:
            source_name = str(p)
            try:
                input_text = p.read_text().strip()
            except Exception as e:
                print(f"Error reading file {p}: {e}", file=sys.stderr)
                return 1

    if not input_text:
        print("Error: Move sequence or file must be provided.", file=sys.stderr)
        return 1

    hand_profile = HandProfile(solving_mode=args.profile, m_slice_hand=args.m_slice_hand)
    scorer = FlowScorer(profile=hand_profile)
    try:
        score = scorer.score(
            input_text,
            tempo=args.tempo,
            profile=hand_profile,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.json:
        result_dict = score.to_dict()
        result_dict["profile"] = args.profile
        result_dict["m_slice_hand"] = args.m_slice_hand
        if args.tempo is not None:
            result_dict["tempo"] = args.tempo
        print(json.dumps(result_dict, indent=2))
    else:
        report = format_flow_report(
            score,
            profile=hand_profile,
            tempo=args.tempo,
            source_name=source_name,
        )
        print(report)

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        cmd = argv[0]
        if cmd == "solve":
            return handle_solve(argv[1:])
        if cmd in ("analyze", "segment"):
            return handle_analyze(argv[1:])
        if cmd == "flow":
            return handle_flow(argv[1:])
        if cmd == "generate-fb-pdb":
            return handle_generate_fb_pdb(argv[1:])
        if cmd == "generate-sb-pdb":
            return handle_generate_sb_pdb(argv[1:])
        if cmd in ("-h", "--help"):
            print("Usage: roux <subcommand> [options]\n")
            print("Subcommands:")
            print("  solve              Find an optimal Roux solution for a scramble")
            print("  analyze            Segment and analyze a human Roux solve")
            print("  flow               Evaluate biomechanical flow, E-STM, and stream rhythm")
            print("  generate-fb-pdb    Generate First Block Pattern Database")
            print("  generate-sb-pdb    Generate Second Block Pattern Databases")
            print("\nOptions:")
            print("  -s, --scramble     Scramble move sequence")
            print("  -sol, --solution   Solution move sequence (triggers analysis)")
            print("  -h, --help         Show this help message")
            return 0

    # Smart default: If -sol passed, route to analyze; if only -s passed, route to solve.
    if "-sol" in argv or "--solution" in argv:
        return handle_analyze(argv)
    if "-s" in argv or "--scramble" in argv:
        return handle_solve(argv)

    # Interactive prompt default: route to analyze
    return handle_analyze(argv)


if __name__ == "__main__":
    sys.exit(main())


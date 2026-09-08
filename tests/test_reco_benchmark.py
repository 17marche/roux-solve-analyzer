"""Roux Reconstructions Benchmark Suite: Validating RouxSegmenter and SB Solver against reco.nz solves."""

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import pytest
from roux_engine.data.loader import RecoDatasetLoader, SolveRecord
from roux_engine.segmenter.segmenter import RouxSegmenter
from roux_engine.core.parser import MoveParser
from roux_engine.core.cube import CubeState
from roux_engine.solver.sb_solver import solve_sb, is_center_aligned_sb_solved
from roux_engine.solver.sb_pdb_generator import SB_MOVESET
from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS, FBDetector, BlockDefinition


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "roux_solves.json")


@pytest.fixture(scope="module")
def dataset_records() -> List[SolveRecord]:
    """Loads and caches the full reco.nz solve dataset."""
    if not os.path.exists(DATASET_PATH):
        pytest.skip(f"Dataset file '{DATASET_PATH}' not found. Run scripts/fetch_reco_solves.py first.")
    loader = RecoDatasetLoader()
    return loader.load_from_json(DATASET_PATH)


@pytest.fixture(scope="module")
def valid_solves(dataset_records: List[SolveRecord]) -> List[SolveRecord]:
    """Filters only verified, valid Roux solve records."""
    loader = RecoDatasetLoader()
    valid = loader.filter_valid(dataset_records)
    assert len(valid) >= 1000, f"Expected at least 1,000 valid solves, found {len(valid)}"
    return valid


def test_reco_benchmark_dataset_integrity(dataset_records: List[SolveRecord]):
    """Verify total solve count, validity ratio, and human splits coverage."""
    total = len(dataset_records)
    assert total == 1181, f"Expected 1,181 total solves, found {total}"

    valid_count = sum(1 for r in dataset_records if r.is_valid)
    assert valid_count >= 1100, f"Expected >= 1,100 valid Roux solves, found {valid_count}"

    splits_count = sum(1 for r in dataset_records if r.human_splits is not None)
    assert splits_count >= 1100, f"Expected human splits extracted for majority of dataset, found {splits_count}"


def test_reco_benchmark_segmentation_100_percent_success(valid_solves: List[SolveRecord]):
    """Verify 100% segmentation success on all valid reconstructed solves."""
    segmenter = RouxSegmenter()
    fcn_segmenter = RouxSegmenter(full_color_neutral=True)

    failed_solves = []
    fb_moves_total = 0
    sb_moves_total = 0
    cmll_moves_total = 0
    lse_moves_total = 0

    eolr_count = 0
    eolr_b_count = 0
    standard_eo_count = 0

    for record in valid_solves:
        events = MoveParser.parse_string(record.solution)
        seg = segmenter.segment_events(record.scramble, events)

        # Fallback to FCN if needed
        if not seg.is_valid:
            seg = fcn_segmenter.segment_events(record.scramble, events)

        if not seg.is_valid:
            failed_solves.append((record.id, record.solver, seg.warnings))
            continue

        # Accumulate metrics
        fb_moves_total += seg.fb.move_count_stm
        sb_moves_total += seg.sb.move_count_stm
        cmll_moves_total += seg.cmll.move_count_stm
        lse_moves_total += seg.lse.move_count_stm

        if seg.lse.step_4a:
            v = seg.lse.step_4a.variant
            if v == "eolr":
                eolr_count += 1
            elif v == "eolr_b":
                eolr_b_count += 1
            else:
                standard_eo_count += 1

    assert len(failed_solves) == 0, f"Segmentation failed on {len(failed_solves)} valid solves: {failed_solves[:5]}"

    n = len(valid_solves)
    print(f"\n--- Roux Benchmark Statistics across {n} solves ---")
    print(f"  Average FB:   {fb_moves_total / n:.2f} STM")
    print(f"  Average SB:   {sb_moves_total / n:.2f} STM")
    print(f"  Average CMLL: {cmll_moves_total / n:.2f} STM")
    print(f"  Average LSE:  {lse_moves_total / n:.2f} STM")
    print(f"  Average Total:{ (fb_moves_total + sb_moves_total + cmll_moves_total + lse_moves_total) / n:.2f} STM")
    print(f"  LSE 4a Breakdown: EOLR={eolr_count} ({(eolr_count/n)*100:.1f}%), EOLR-b={eolr_b_count} ({(eolr_b_count/n)*100:.1f}%), Standard EO={standard_eo_count} ({(standard_eo_count/n)*100:.1f}%)")


def test_reco_benchmark_human_splits_correlation(valid_solves: List[SolveRecord]):
    """Verify that automated segmentation phase STM movecounts closely match human reconstructor annotations."""
    segmenter = RouxSegmenter()
    fcn_segmenter = RouxSegmenter(full_color_neutral=True)

    tested_with_splits = 0
    exact_fb_matches = 0
    exact_cmll_matches = 0

    for record in valid_solves:
        if not record.human_splits or "stm" not in record.human_splits:
            continue

        human_stm = record.human_splits["stm"]
        if "fb" not in human_stm or "cmll" not in human_stm:
            continue

        events = MoveParser.parse_string(record.solution)
        seg = segmenter.segment_events(record.scramble, events)
        if not seg.is_valid:
            seg = fcn_segmenter.segment_events(record.scramble, events)

        if not seg.is_valid:
            continue

        tested_with_splits += 1
        human_fb = human_stm["fb"]
        human_cmll = human_stm["cmll"]

        # Check FB movecount comparison (exact match or within 1 move due to inspection/AUF interpretation)
        if isinstance(human_fb, (int, float)) and seg.fb.move_count_stm == int(human_fb):
            exact_fb_matches += 1

        # Check CMLL movecount comparison (exact match or within AUF)
        if isinstance(human_cmll, (int, float)) and seg.cmll.move_count_stm == int(human_cmll):
            exact_cmll_matches += 1

    assert tested_with_splits >= 1000
    fb_match_rate = (exact_fb_matches / tested_with_splits) * 100
    cmll_match_rate = (exact_cmll_matches / tested_with_splits) * 100

    print(f"\n--- Human Split Alignment ({tested_with_splits} solves) ---")
    print(f"  Exact FB Match Rate:   {fb_match_rate:.1f}%")
    print(f"  Exact CMLL Match Rate: {cmll_match_rate:.1f}%")

    # Both should have extremely high alignment (> 90%)
    assert fb_match_rate > 90.0, f"FB match rate {fb_match_rate:.1f}% lower than expected (>90%)"
    assert cmll_match_rate > 90.0, f"CMLL match rate {cmll_match_rate:.1f}% lower than expected (>90%)"


# =============================================================================
# Milestone 3.5: Real-World Solve Dataset SB Solver Benchmark & Verification
# =============================================================================

@dataclass(frozen=True)
class SBBenchmarkCase:
    """Benchmark test case encapsulating a reconstructed solve state after First Block."""
    record: SolveRecord
    fb_cube: CubeState
    human_sb_stm: int
    block: BlockDefinition


@pytest.fixture(scope="module")
def sb_benchmark_solves(valid_solves: List[SolveRecord]) -> List[SBBenchmarkCase]:
    """Segments valid solves and extracts states at the start of Second Block for dual-neutral solves."""
    segmenter = RouxSegmenter(full_color_neutral=False)
    benchmark_solves: List[SBBenchmarkCase] = []

    for record in valid_solves:
        events = MoveParser.parse_string(record.solution)
        seg = segmenter.segment_events(record.scramble, events)
        if not (seg.is_valid and seg.fb and seg.sb):
            continue

        # Prepare cube state at completion of First Block
        cube = CubeState().apply_moves(record.scramble)
        for event in events[:seg.fb.end_move_idx + 1]:
            cube.apply_move(event.move)

        block = FBDetector.match_fb_block(cube)
        if block is None:
            continue

        # Extract human reconstructor's SB movecount in Slice Turn Metric (STM)
        human_sb_stm = seg.sb.move_count_stm
        if record.human_splits and "stm" in record.human_splits and "sb" in record.human_splits["stm"]:
            human_val = record.human_splits["stm"]["sb"]
            if isinstance(human_val, (int, float)):
                human_sb_stm = int(human_val)

        benchmark_solves.append(
            SBBenchmarkCase(
                record=record,
                fb_cube=cube,
                human_sb_stm=human_sb_stm,
                block=block,
            )
        )

    assert len(benchmark_solves) >= 900, (
        f"Expected >= 900 dual-neutral SB benchmark solves, got {len(benchmark_solves)}"
    )
    return benchmark_solves


def _verify_candidate_solution_invariants(
    case: SBBenchmarkCase,
    sol: SBSolution,
    valid_moveset: set[str],
) -> None:
    """Verifies that a candidate solution adheres to moveset generator, preserves FB, and aligns centers."""
    for move in sol.moves:
        assert move in valid_moveset, (
            f"Solve {case.record.id}: move '{move}' not in allowed generator <R, U, r, M>"
        )

    sim = case.fb_cube.copy()
    if sol.moves:
        sim.apply_moves(" ".join(sol.moves))

    matched_fb = FBDetector.match_fb_block(sim)
    assert matched_fb == case.block, (
        f"Solve {case.record.id}: candidate solution corrupted First Block: {sol.moves}"
    )
    assert is_center_aligned_sb_solved(sim, case.block), (
        f"Solve {case.record.id}: candidate solution failed to solve Center-Aligned SB: {sol.moves}"
    )


def test_reco_benchmark_sb_stm_superiority_and_soundness(
    sb_benchmark_solves: List[SBBenchmarkCase]
):
    """Verify solver achieves STM <= human in >= 95% of solves and preserves FB with aligned centers across 100% of solutions."""
    valid_moveset = set(SB_MOVESET)
    shorter_or_equal_count = 0
    total_human_stm = 0
    total_solver_stm = 0
    tested = len(sb_benchmark_solves)

    for case in sb_benchmark_solves:
        sols = solve_sb(case.fb_cube, top_k=1, style="all")
        assert len(sols) >= 1, f"No SB solution found for solve {case.record.id}"
        best_sol = sols[0]

        _verify_candidate_solution_invariants(case, best_sol, valid_moveset)

        total_human_stm += case.human_sb_stm
        total_solver_stm += best_sol.move_count

        if best_sol.move_count <= case.human_sb_stm:
            shorter_or_equal_count += 1

    superiority_rate = (shorter_or_equal_count / tested) * 100.0
    avg_human = total_human_stm / tested
    avg_solver = total_solver_stm / tested

    print(f"\n--- Real-World Second Block STM Benchmark ({tested} solves) ---")
    print(f"  Human Reconstructor Average SB: {avg_human:.2f} STM")
    print(f"  PDB Solver Average SB:          {avg_solver:.2f} STM")
    print(f"  Average STM Savings:            {avg_human - avg_solver:.2f} STM ({(1 - avg_solver/avg_human)*100:.1f}%)")
    print(f"  Shorter or Equal to Human:      {shorter_or_equal_count}/{tested} ({superiority_rate:.2f}%)")

    # In at least 95% of benchmarked solves, solver STM <= human reconstructor's STM
    assert superiority_rate >= 95.0, (
        f"Solver superiority rate {superiority_rate:.2f}% lower than target 95.0%"
    )


def test_reco_benchmark_sb_single_search_latency_under_10ms(
    sb_benchmark_solves: List[SBBenchmarkCase]
):
    """Verify single Second Block solve search latency averages < 10ms on CPU for both free and all styles."""
    sample = sb_benchmark_solves[:100]
    latencies_free_ms: List[float] = []
    latencies_all_ms: List[float] = []

    # Warmup
    solve_sb(sample[0].fb_cube, top_k=1, style="free")
    solve_sb(sample[0].fb_cube, top_k=1, style="all")

    for case in sample:
        # Measure single Free search
        t0 = time.perf_counter()
        sols_free = solve_sb(case.fb_cube, top_k=1, style="free")
        latencies_free_ms.append((time.perf_counter() - t0) * 1000.0)
        assert len(sols_free) >= 1

        # Measure master style="all" candidate query
        t1 = time.perf_counter()
        sols_all = solve_sb(case.fb_cube, top_k=1, style="all")
        latencies_all_ms.append((time.perf_counter() - t1) * 1000.0)
        assert len(sols_all) >= 1

    avg_free = sum(latencies_free_ms) / len(latencies_free_ms)
    avg_all = sum(latencies_all_ms) / len(latencies_all_ms)

    print(f"\n--- Second Block Search Latency Benchmark ({len(sample)} solves) ---")
    print(f"  Free Style Average Latency: {avg_free:.2f} ms")
    print(f"  All Style Average Latency:  {avg_all:.2f} ms")

    assert avg_free < 10.0, f"Average Free search latency {avg_free:.2f}ms exceeds 10ms limit"
    assert avg_all < 10.0, f"Average All search latency {avg_all:.2f}ms exceeds 10ms limit"


def test_reco_benchmark_sb_multi_style_candidates(
    sb_benchmark_solves: List[SBBenchmarkCase]
):
    """Verify that multi-style candidate generation succeeds and preserves invariants across real solve states."""
    sample = sb_benchmark_solves[:25]
    valid_moveset = set(SB_MOVESET)

    for case in sample:
        for test_style in ("free", "square_pair", "classical"):
            sols = solve_sb(case.fb_cube, top_k=2, style=test_style)
            assert len(sols) >= 1, f"Solve {case.record.id}: no solutions for style={test_style}"

            for sol in sols:
                assert sol.style == test_style
                assert sol.resulting_cmll_case != ""
                _verify_candidate_solution_invariants(case, sol, valid_moveset)




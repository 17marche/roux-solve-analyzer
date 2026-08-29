"""Milestone 2 Benchmark Test Suite: Validating RouxSegmenter against 1,181 reco.nz solves."""

import os
import pytest
from typing import List

from roux_engine.data.loader import RecoDatasetLoader, SolveRecord
from roux_engine.segmenter.segmenter import RouxSegmenter
from roux_engine.core.parser import MoveParser


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

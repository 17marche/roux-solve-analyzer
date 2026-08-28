"""Unit tests for First Block (FB) Detector."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.core.constants import Corner, Edge, Center
from roux_engine.segmenter.fb_detector import (
    FBDetector,
    ALL_BLOCK_DEFINITIONS,
    DUAL_NEUTRAL_ORIENTATIONS,
    FULL_COLOR_NEUTRAL_ORIENTATIONS
)


def test_fb_detector_canonical():
    """Test FB detection on a canonical First Block solve."""
    clean_cube = CubeState()
    assert FBDetector.is_canonical_fb_solved(clean_cube)

    # Twisted corner at DLF should NOT be considered solved
    twisted_cube = CubeState()
    twisted_cube.co[Corner.DLF] = 1
    assert not FBDetector.is_canonical_fb_solved(twisted_cube)

    # Flipped edge at DL should NOT be considered solved
    flipped_cube = CubeState()
    flipped_cube.eo[Edge.DL] = 1
    assert not FBDetector.is_canonical_fb_solved(flipped_cube)


def test_fb_detection_across_dual_neutral_orientations():
    """Verify that FB is detected accurately across all 8 dual-neutral orientations."""
    for ori in DUAL_NEUTRAL_ORIENTATIONS:
        # Setup cube in orientation ori with R/U scrambled (FB preserved)
        initial = CubeState()
        if ori:
            initial.apply_moves(ori)
        initial.apply_moves("R U R' U R U2 R'")
        
        events = MoveParser.parse_string("R U R'")
        res = FBDetector.detect_fb(initial, events, full_color_neutral=False)
        assert res is not None, f"Failed for orientation {ori}"
        fb_phase, detected_ori, final_state, matched_block = res
        assert fb_phase.orientation is not None
        assert fb_phase.orientation.bottom_color == matched_block.bottom_color


def test_fb_detection_full_color_neutral():
    """Verify that FB is detected accurately across all 24 color-neutral orientations when enabled."""
    for ori in FULL_COLOR_NEUTRAL_ORIENTATIONS:
        initial = CubeState()
        if ori:
            initial.apply_moves(ori)
        initial.apply_moves("R U R' U R U2 R'")

        events = MoveParser.parse_string("R U R'")
        res = FBDetector.detect_fb(initial, events, full_color_neutral=True)
        assert res is not None, f"Failed for CN orientation {ori}"
        fb_phase, detected_ori, final_state, matched_block = res
        assert fb_phase.orientation is not None
        assert fb_phase.orientation.bottom_color == matched_block.bottom_color


def test_fb_concurrent_sb_tracking():
    """Verify that concurrent SB pieces (DR, square, pair) are detected with exact IDs and orientations."""
    block = ALL_BLOCK_DEFINITIONS[""]

    # State 1: DR edge disturbed
    cube1 = CubeState()
    cube1.apply_moves("R2 U2 R2")
    concurrent1 = FBDetector.check_concurrent_sb(cube1, block)
    assert not concurrent1.dr_solved

    # State 2: Solved cube has both FB and SB solved
    solved_cube = CubeState()
    concurrent2 = FBDetector.check_concurrent_sb(solved_cube, block)
    assert concurrent2.dr_solved is True
    assert concurrent2.sb_square_solved is True
    assert concurrent2.sb_pair_solved is True


def test_fb_solved_at_inspection_off_by_one():
    """Verify that FB solved with 0 execution moves sets end_move_idx to -1 without crashing or mispointing."""
    initial = CubeState()  # FB already solved
    events = MoveParser.parse_string("R U R'")

    res = FBDetector.detect_fb(initial, events, full_color_neutral=False)
    assert res is not None
    fb_phase = res[0]
    assert fb_phase.end_move_idx == -1
    assert fb_phase.move_count_stm == 0

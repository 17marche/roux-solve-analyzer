"""Unit tests for Second Block (SB) Detector."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.core.constants import Corner, Edge
from roux_engine.segmenter.sb_detector import SBDetector
from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS


def test_sb_detector_canonical():
    """Test full SB detection on canonical solve and exact orientation rejection."""
    clean_cube = CubeState()
    assert SBDetector.is_canonical_sb_solved(clean_cube)
    assert SBDetector.is_dr_solved(clean_cube)
    assert SBDetector.is_back_pair_solved(clean_cube)
    assert SBDetector.is_front_pair_solved(clean_cube)
    assert SBDetector.is_right_1x2x3_block_solved(clean_cube)

    # Twisted corner at DFR should NOT be considered solved
    twisted_cube = CubeState()
    twisted_cube.co[Corner.DFR] = 1
    assert not SBDetector.is_front_pair_solved(twisted_cube)
    assert not SBDetector.is_right_1x2x3_block_solved(twisted_cube)

    # Flipped edge at FR should NOT be considered solved
    flipped_cube = CubeState()
    flipped_cube.eo[Edge.FR] = 1
    assert not SBDetector.is_front_pair_solved(flipped_cube)
    assert not SBDetector.is_right_1x2x3_block_solved(flipped_cube)


def test_sb_pair_ordering_back_first():
    """Test detection of Back Pair first in SB starting from fully scrambled SB pieces."""
    # SB solve sequence:
    # 1. Moves 0..6: Solves DR edge + Back pair (DR=True, Back=True, Front=False)
    # 2. Moves 7..10: Solves Front pair (DR=True, Back=True, Front=True -> SB complete!)
    sb_moves = "R2 U' R2 U R2 U R' U' R U2 R'"
    inv = MoveParser.invert_moves(sb_moves)

    cube = CubeState().apply_moves(inv)

    # Verify that initial state has ALL SB pieces completely scrambled/unsolved
    assert not SBDetector.is_dr_solved(cube)
    assert not SBDetector.is_back_pair_solved(cube)
    assert not SBDetector.is_front_pair_solved(cube)
    assert not SBDetector.is_canonical_sb_solved(cube)

    events = MoveParser.parse_string(sb_moves)
    res = SBDetector.detect_sb(
        fb_state=cube,
        events=events,
        fb_end_idx=-1,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res

    # Verify Back pair is detected first at move index 6, and Front pair at move index 10
    assert phase.pair1_type == "back"
    assert phase.pair1_idx == 6
    assert phase.pair2_idx == 10
    assert phase.sb_square_idx == 6
    assert phase.dr_placement_idx == 6
    assert SBDetector.is_canonical_sb_solved(final_state)


def test_sb_rotations_and_ergonomics_tracking():
    """Verify that rotations (y, x, z) and non-ergonomic face turns (F, B, D) are tracked by detect_sb."""
    # Setup SB solve containing rotations (y, y') and non-ergonomic face turns (F, F')
    sb_moves = "y r U R' y' F R U' R' F' R U R'"
    inv = MoveParser.invert_moves(sb_moves)

    cube = CubeState()
    cube.apply_moves(inv)

    events = MoveParser.parse_string(sb_moves)
    res = SBDetector.detect_sb(
        fb_state=cube,
        events=events,
        fb_end_idx=-1,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, _ = res
    # Directly verify automated tracking on phase object
    assert phase.rotation_count == 2
    assert phase.non_ergonomic_moves == ["F", "F'"]


def test_sb_concurrent_partial_progress():
    """Verify true concurrent SB tracking when DR and Back pair are formed during FB."""
    # Set up cube state at fb_end_idx = 4 where DR and Back pair are solved, but Front pair is scrambled
    front_pair_solve = "U R U' R'"
    inv_front = MoveParser.invert_moves(front_pair_solve)

    fb_state = CubeState().apply_moves(inv_front)
    assert SBDetector.is_dr_solved(fb_state)
    assert SBDetector.is_back_pair_solved(fb_state)
    assert not SBDetector.is_front_pair_solved(fb_state)
    assert not SBDetector.is_canonical_sb_solved(fb_state)

    # Full events sequence covering previous FB moves (indices 0..4) and subsequent SB moves (indices 5..8)
    fb_moves = "D' F' L2 D B"
    full_events = MoveParser.parse_string(f"{fb_moves} {front_pair_solve}")

    res = SBDetector.detect_sb(
        fb_state=fb_state,
        events=full_events,
        fb_end_idx=4,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res

    # Verify that initial concurrent SB progress is recorded at fb_end_idx = 4
    assert phase.dr_placement_idx == 4
    assert phase.pair1_idx == 4
    assert phase.pair1_type == "back"
    assert phase.sb_square_idx == 4
    # Verify that remaining Front pair completes at move index 8
    assert phase.pair2_idx == 8
    assert phase.move_count_stm == 4
    assert phase.moves_str == "U R U' R'"
    assert SBDetector.is_canonical_sb_solved(final_state)


def test_sb_already_fully_solved_at_fb_end():
    """Verify 0-move SB skip when full Right Block is already completed at FB end."""
    clean_cube = CubeState()
    fb_moves = "D' F' L2 D B"
    events = MoveParser.parse_string(f"{fb_moves} U R U' R'")

    res = SBDetector.detect_sb(
        fb_state=clean_cube,
        events=events,
        fb_end_idx=4,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res

    assert phase.move_count_stm == 0
    assert phase.dr_placement_idx == 4
    assert phase.pair1_idx == 4
    assert phase.pair2_idx == 4
    assert phase.pair1_type == "both_simultaneous"
    assert phase.moves_str == ""
    assert SBDetector.is_canonical_sb_solved(final_state)


def test_sb_detector_delegates_to_roux_orientation():
    """Verify SBDetector methods delegate to RouxOrientation across orientations and accept string identifiers."""
    from roux_engine.core.orientation import get_orientation, get_all_orientations

    # 1. Test across all 24 orientations with RouxOrientation instances
    for ori in get_all_orientations():
        cube = CubeState()
        if ori.rotations:
            cube.apply_moves(ori.rotations)

        assert SBDetector.is_dr_solved(cube, ori)
        assert SBDetector.is_back_pair_solved(cube, ori)
        assert SBDetector.is_front_pair_solved(cube, ori)
        assert SBDetector.is_right_1x2x3_block_solved(cube, ori)

        # Also accepts rotation string identifier directly
        assert SBDetector.is_dr_solved(cube, ori.rotations)
        assert SBDetector.is_back_pair_solved(cube, ori.rotations)
        assert SBDetector.is_front_pair_solved(cube, ori.rotations)
        assert SBDetector.is_right_1x2x3_block_solved(cube, ori.rotations)


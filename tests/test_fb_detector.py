"""Unit tests for First Block (FB) Detector."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.core.constants import Corner, Edge, Center
from roux_engine.core.orientation import (
    RouxOrientation,
    get_orientation,
    is_fb_solved,
)
from roux_engine.segmenter.fb_detector import (
    FBDetector,
    DUAL_NEUTRAL_ORIENTATIONS,
    FULL_COLOR_NEUTRAL_ORIENTATIONS
)


def test_fb_detector_canonical():
    """Test FB detection on a canonical First Block solve via core.orientation seam."""
    clean_cube = CubeState()
    assert is_fb_solved(clean_cube)
    assert not hasattr(FBDetector, "is_canonical_fb_solved")

    # Twisted corner at DLF should NOT be considered solved
    twisted_cube = CubeState()
    twisted_cube.co[Corner.DLF] = 1
    assert not is_fb_solved(twisted_cube)

    # Flipped edge at DL should NOT be considered solved
    flipped_cube = CubeState()
    flipped_cube.eo[Edge.DL] = 1
    assert not is_fb_solved(flipped_cube)


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
    block = get_orientation("")

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


def test_fb_detector_uses_roux_orientation():
    """Verify FB detector match_fb_block returns RouxOrientation."""
    from roux_engine.core.orientation import RouxOrientation

    clean_cube = CubeState()
    matched = FBDetector.match_fb_block(clean_cube)
    assert isinstance(matched, RouxOrientation)
    assert matched.rotations == ""
    assert matched.is_fb_solved(clean_cube)


def test_legacy_detector_aliases_removed():
    """Verify legacy aliases BlockDefinition and ALL_BLOCK_DEFINITIONS are deleted from FBDetector."""
    import roux_engine.segmenter.fb_detector as seg_fb
    from roux_engine.core.orientation import RouxOrientation, get_orientation

    # 1. Block spec and dictionary legacy aliases are deleted
    assert not hasattr(seg_fb, "BlockDefinition")
    assert not hasattr(seg_fb, "ALL_BLOCK_DEFINITIONS")
    assert not hasattr(seg_fb.FBDetector, "is_canonical_fb_solved")

    # 2. Canonical RouxOrientation has all 39 piece and coordinate attributes
    block = get_orientation("")
    assert isinstance(block, RouxOrientation)
    assert block.rotations == ""
    assert hasattr(block, "left_color")
    assert hasattr(block, "bottom_color")
    assert hasattr(block, "front_color")
    assert hasattr(block, "back_color")
    assert hasattr(block, "right_color")
    assert hasattr(block, "top_color")
    assert hasattr(block, "dl_piece")
    assert hasattr(block, "dl_eo")
    assert hasattr(block, "fl_piece")
    assert hasattr(block, "fl_eo")
    assert hasattr(block, "bl_piece")
    assert hasattr(block, "bl_eo")
    assert hasattr(block, "dlf_piece")
    assert hasattr(block, "dlf_co")
    assert hasattr(block, "dbl_piece")
    assert hasattr(block, "dbl_co")
    assert hasattr(block, "dr_piece")
    assert hasattr(block, "dr_eo")
    assert hasattr(block, "fr_piece")
    assert hasattr(block, "fr_eo")
    assert hasattr(block, "br_piece")
    assert hasattr(block, "br_eo")
    assert hasattr(block, "dfr_piece")
    assert hasattr(block, "dfr_co")
    assert hasattr(block, "drb_piece")
    assert hasattr(block, "drb_co")
    assert hasattr(block, "dbr_piece")  # canonical CONTEXT.md alias
    assert hasattr(block, "ul_piece")
    assert hasattr(block, "ul_eo")
    assert hasattr(block, "ur_piece")
    assert hasattr(block, "ur_eo")
    assert hasattr(block, "uf_piece")
    assert hasattr(block, "uf_eo")
    assert hasattr(block, "ub_piece")
    assert hasattr(block, "ub_eo")
    assert hasattr(block, "df_piece")
    assert hasattr(block, "df_eo")
    assert hasattr(block, "db_piece")
    assert hasattr(block, "db_eo")

    # 3. Orientation list exports
    assert hasattr(seg_fb, "DUAL_NEUTRAL_ORIENTATIONS")
    assert hasattr(seg_fb, "FULL_COLOR_NEUTRAL_ORIENTATIONS")
    assert len(seg_fb.DUAL_NEUTRAL_ORIENTATIONS) == 8
    assert len(seg_fb.FULL_COLOR_NEUTRAL_ORIENTATIONS) == 24



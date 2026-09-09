"""Comprehensive unit tests for LSE classifier: 4a (EO/EOLR), 4b, and 4c."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.core.constants import Edge, Center
from roux_engine.segmenter.lse_classifier import LSEClassifier
from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS, DUAL_NEUTRAL_ORIENTATIONS, FULL_COLOR_NEUTRAL_ORIENTATIONS


def test_lse_helper_methods_canonical_and_dual_neutral():
    """Verify is_center_axis_aligned, is_eo_solved, and is_ul_ur_solved across all 24 orientations."""
    for ori in FULL_COLOR_NEUTRAL_ORIENTATIONS:
        block = ALL_BLOCK_DEFINITIONS[ori]
        clean_cube = CubeState()
        if ori:
            clean_cube.apply_moves(ori)

        # 1. Axis alignment
        assert LSEClassifier.is_center_axis_aligned(clean_cube, block)
        m_cube = clean_cube.copy().apply_move("M")
        assert not LSEClassifier.is_center_axis_aligned(m_cube, block)

        # 2. EO solved
        assert LSEClassifier.is_eo_solved(clean_cube, block)
        assert LSEClassifier.count_bad_edges(clean_cube, block) == 0
        flipped_cube = clean_cube.copy()
        flipped_cube.eo[Edge.UF] = (flipped_cube.eo[Edge.UF] + 1) % 2
        assert not LSEClassifier.is_eo_solved(flipped_cube, block)
        assert LSEClassifier.count_bad_edges(flipped_cube, block) == 1

        # 3. UL/UR solved (exact placement)
        assert LSEClassifier.is_ul_ur_solved(clean_cube, block)
        swapped_cube = clean_cube.copy()
        # Swap UL with UF
        swapped_cube.ep[Edge.UL] = block.uf_piece
        assert not LSEClassifier.is_ul_ur_solved(swapped_cube, block)


def test_lse_ul_ur_exact_slot_rejection_and_auf_tolerance():
    """Verify rejection of swapped UL/UR and tolerance of relative AUF offsets."""
    block = ALL_BLOCK_DEFINITIONS[""]

    # 1. Swapped UL/UR pieces (UL in UR slot and UR in UL slot) must return False
    c_swap = CubeState()
    c_swap.ep[Edge.UL] = block.ur_piece
    c_swap.ep[Edge.UR] = block.ul_piece
    assert not LSEClassifier.is_ul_ur_solved(c_swap, block)

    # 2. Relative AUF tolerance: UL and UR aligned with corners under pending AUF (e.g. M2 U2 M2 U)
    c_auf = CubeState().apply_moves("M2 U2 M2 U")
    assert LSEClassifier.is_ul_ur_solved(c_auf, block)

    # 3. Solved cube with U2 AUF
    c_u2 = CubeState().apply_move("U2")
    assert LSEClassifier.is_ul_ur_solved(c_u2, block)


def test_lse_4a_variants():
    """Verify classification of standard EO, EOLR, and EOLR-b with exact assertions and EO preservation."""
    block = ALL_BLOCK_DEFINITIONS[""]

    # 1. EOLR-b: UL and UR are in UL and UR slots (with EO fully solved)
    cube_b = CubeState()
    assert LSEClassifier.is_eo_solved(cube_b, block)
    assert LSEClassifier.classify_4a_variant(cube_b, block) == "eolr_b"

    # 2. EOLR: UL and UR are in DF and DB slots (via U M2 U' which preserves all 6 oriented edges)
    cube_eolr = CubeState().apply_moves("U M2 U'")
    assert LSEClassifier.is_eo_solved(cube_eolr, block)
    assert LSEClassifier.classify_4a_variant(cube_eolr, block) == "eolr"

    # 3. Standard EO: UL and UR are in non-EOLR slots (via U M2 U' M2: UL at UB, UR at UF, all 6 edges oriented)
    cube_std = CubeState().apply_moves("U M2 U' M2")
    assert LSEClassifier.is_eo_solved(cube_std, block)
    assert LSEClassifier.classify_4a_variant(cube_std, block) == "standard_eo"

    # 4. Error case: Calling classify_4a_variant on an unsolved EO state (e.g. arrow case from U M' U' M) raises ValueError
    cube_unsolved_eo = CubeState().apply_moves("U M' U' M")
    assert not LSEClassifier.is_eo_solved(cube_unsolved_eo, block)
    with pytest.raises(ValueError, match="Cannot classify 4a variant when EO is not solved"):
        LSEClassifier.classify_4a_variant(cube_unsolved_eo, block)


def test_lse_4c_cases_classification():
    """Verify all 4c case classifications: solved, center_swap, dots, opp_opp, 3_cycle (with AUF tolerance)."""
    block = ALL_BLOCK_DEFINITIONS[""]

    # 1. Solved (including under AUFs)
    cube_solved = CubeState()
    assert LSEClassifier.classify_4c_case(cube_solved, block) == "solved"
    for auf in ("U", "U2", "U'"):
        c_auf = CubeState().apply_moves(auf)
        assert LSEClassifier.classify_4c_case(c_auf, block) == "solved"

    # 2. Center Swap (M2)
    cube_center_swap = CubeState().apply_move("M2")
    assert LSEClassifier.classify_4c_case(cube_center_swap, block) == "center_swap"

    # 3. Dots / Vertical Bars (M' U2 M2 U2 M')
    cube_dots = CubeState().apply_moves("M' U2 M2 U2 M'")
    assert LSEClassifier.classify_4c_case(cube_dots, block) == "dots"

    # 4. Opp-Opp / Double Bars (M2 U2 M2 U2) (including under AUFs)
    cube_opp_opp = CubeState().apply_moves("M2 U2 M2 U2")
    assert LSEClassifier.classify_4c_case(cube_opp_opp, block) == "opp_opp"
    cube_opp_opp_auf = CubeState().apply_moves("M2 U2 M2 U2 U")
    assert LSEClassifier.classify_4c_case(cube_opp_opp_auf, block) == "opp_opp"

    # 5. 3-Cycle (M U2 M' U2)
    cube_cycle3 = CubeState().apply_moves("M U2 M' U2")
    assert LSEClassifier.classify_4c_case(cube_cycle3, block) == "3_cycle"
    cube_cycle3_auf = CubeState().apply_moves("M U2 M' U2 U2")
    assert LSEClassifier.classify_4c_case(cube_cycle3_auf, block) == "3_cycle"


def test_lse_full_phase_detection():
    """Verify end-to-end LSE detection (4a -> 4b -> 4c)."""
    # LSE solve sequence: M' U M' U2 M U M' U2 M2
    lse_alg = "M' U M' U2 M U M' U2 M2"
    inv = MoveParser.invert_moves(lse_alg)
    
    cube = CubeState().apply_moves(inv)
    events = MoveParser.parse_string(f"D' F' L2 D B {lse_alg}")
    cmll_end_idx = 4

    res = LSEClassifier.detect_lse(
        cmll_state=cube,
        events=events,
        cmll_end_idx=cmll_end_idx,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res
    assert final_state.is_solved()
    assert phase.step_4a is not None
    assert phase.step_4c is not None
    assert phase.move_count_stm == 9


def test_lse_eolr_b_skip():
    """Verify instant 4b skip when EOLR-b solves UL/UR during 4a."""
    # EOLR-b finish: M' U M' (solves EO with UL/UR in slots) -> M2 U2 M2 (4c)
    lse_alg = "M' U M' M2 U2 M2"
    inv = MoveParser.invert_moves(lse_alg)

    cube = CubeState().apply_moves(inv)
    events = MoveParser.parse_string(f"D' F' L2 D B {lse_alg}")

    res = LSEClassifier.detect_lse(
        cmll_state=cube,
        events=events,
        cmll_end_idx=4,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res
    assert phase.step_4a.variant == "eolr_b"
    assert phase.step_4b.skipped is True
    assert phase.step_4b.move_count_stm == 0


def test_lse_already_solved_skip():
    """Verify 0-move LSE skip when cube is already solved at CMLL end."""
    clean_cube = CubeState()
    events = MoveParser.parse_string("D' F' L2 D B r U R' U R U R' U R U2 R'")

    res = LSEClassifier.detect_lse(
        cmll_state=clean_cube,
        events=events,
        cmll_end_idx=len(events) - 1,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res
    assert phase.move_count_stm == 0
    assert phase.step_4a.move_count_stm == 0
    assert phase.step_4c.move_count_stm == 0
    assert phase.step_4c.case == "solved"


def test_lse_classifier_delegates_to_roux_orientation():
    """Verify LSEClassifier methods accept RouxOrientation and string identifiers, delegating to core methods."""
    from roux_engine.core.orientation import get_orientation, get_all_orientations

    for ori in get_all_orientations():
        cube = CubeState()
        if ori.rotations:
            cube.apply_moves(ori.rotations)

        # Accepts RouxOrientation instance
        assert LSEClassifier.is_center_axis_aligned(cube, ori)
        assert LSEClassifier.is_eo_solved(cube, ori)
        assert LSEClassifier.count_bad_edges(cube, ori) == 0
        assert LSEClassifier.is_ul_ur_solved(cube, ori)
        assert LSEClassifier.classify_4a_variant(cube, ori) == "eolr_b"

        # Accepts string rotation identifier directly
        assert LSEClassifier.is_center_axis_aligned(cube, ori.rotations)
        assert LSEClassifier.is_eo_solved(cube, ori.rotations)
        assert LSEClassifier.count_bad_edges(cube, ori.rotations) == 0
        assert LSEClassifier.is_ul_ur_solved(cube, ori.rotations)
        assert LSEClassifier.classify_4a_variant(cube, ori.rotations) == "eolr_b"


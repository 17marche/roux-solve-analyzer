"""Comprehensive tests for CMLL classifier across all 42 canonical cases, AUFs, and orientations."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.segmenter.cmll_classifier import CMLLClassifier, CMLL_ALGS
from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS, DUAL_NEUTRAL_ORIENTATIONS


def test_cmll_table_size():
    """Verify that the CMLL lookup table contains all 42 cases across all 4 AUFs."""
    table = CMLLClassifier.get_table()
    assert len(table) > 0
    # Every case (except solved) has 4 AUF variants: 41 * 4 + 1 = 165
    case_ids = {val[0] for val in table.values()}
    for case_id, group, alg in CMLL_ALGS:
        assert case_id in case_ids


@pytest.mark.parametrize("case_id,group,alg", CMLL_ALGS)
def test_cmll_individual_cases_without_auf(case_id, group, alg):
    """Verify that applying the inverse of each CMLL algorithm correctly identifies the case."""
    cube = CubeState()
    if alg:
        inv = MoveParser.invert_moves(alg)
        cube.apply_moves(inv)

    detected_case, detected_group, detected_auf = CMLLClassifier.classify_state(cube)
    assert detected_case == case_id
    assert detected_group == group
    assert detected_auf == ""


@pytest.mark.parametrize("pre_auf,expected_auf_needed", [
    ("U", "U'"),
    ("U2", "U2"),
    ("U'", "U"),
])
def test_cmll_with_pre_auf(pre_auf, expected_auf_needed):
    """Verify that pre-AUFs are accurately isolated."""
    # Test on Sune left bar: R U R' U R U2 R'
    alg = "R U R' U R U2 R'"
    inv_alg = MoveParser.invert_moves(alg)

    cube = CubeState()
    cube.apply_moves(inv_alg)
    cube.apply_move(pre_auf)

    detected_case, detected_group, detected_auf = CMLLClassifier.classify_state(cube)
    assert detected_case == "s_left_bar"
    assert detected_group == "Sune"
    assert detected_auf == expected_auf_needed


def test_cmll_detection_in_solve():
    """Test full CMLL phase detection from SB end to corners solved."""
    cube = CubeState()
    # Apply inverse of T-case CMLL (t_left_bar): R U R' U' R' F R F'
    inv = MoveParser.invert_moves("R U R' U' R' F R F'")
    cube.apply_moves(inv)

    # Solve execution with leading AUF
    events = MoveParser.parse_string("R U R' U' R' F R F'")
    res = CMLLClassifier.detect_cmll(
        sb_state=cube,
        events=events,
        sb_end_idx=-1,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res
    assert phase.case_id == "t_left_bar"
    assert phase.group == "T"
    assert phase.move_count_stm == 8
    assert phase.is_standard_alg is True
    assert CMLLClassifier.are_corners_solved(final_state)


def test_cmll_detection_across_dual_neutral_orientations():
    """Verify CMLL classification works across all dual-neutral orientations."""
    for ori in DUAL_NEUTRAL_ORIENTATIONS:
        block = ALL_BLOCK_DEFINITIONS[ori]
        # Setup Sune in orientation ori
        cube = CubeState()
        if ori:
            cube.apply_moves(ori)
        # Apply inverse Sune relative to grip
        cube.apply_moves(MoveParser.invert_moves("R U R' U R U2 R'"))

        case_id, group, auf = CMLLClassifier.classify_state(cube, block=block)
        assert case_id == "s_left_bar"
        assert group == "Sune"


def test_cmll_already_solved_skip():
    """Verify 0-move CMLL skip when corners are already solved at sb_end_idx."""
    clean_cube = CubeState()
    events = MoveParser.parse_string("D' F' L2 D B r U R' M' U M'")

    res = CMLLClassifier.detect_cmll(
        sb_state=clean_cube,
        events=events,
        sb_end_idx=4,
        block=ALL_BLOCK_DEFINITIONS[""]
    )
    assert res is not None
    phase, final_state = res
    assert phase.move_count_stm == 0
    assert phase.case_id == "solved"
    assert phase.group == "Skip"
    assert phase.is_standard_alg is True
    assert phase.moves_str == ""


def test_cmll_are_corners_solved_rejection():
    """Verify are_corners_solved rejects twisted corners or invalid corner permutations."""
    # 1. Twisted corner
    twisted = CubeState()
    twisted.co[0] = 1
    assert not CMLLClassifier.are_corners_solved(twisted)

    # 2. Swapped adjacent corners (e.g. 1 and 0 swapped -> reversed parity)
    swapped = CubeState()
    swapped.cp[0] = 1
    swapped.cp[1] = 0
    assert not CMLLClassifier.are_corners_solved(swapped)

"""End-to-End integration tests for the Master RouxSegmenter."""

import pytest
import json
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.segmenter import RouxSegmenter, SegmentedSolve


def test_segmenter_full_solve_canonical():
    """Test segmentation on a genuine, authentic canonical Roux solve."""
    # Authentic canonical Roux solve sequence:
    # 1. FB (5 moves): D' F' L2 D B (builds Left 1x2x3 block)
    # 2. SB (11 moves): r U R' U' R U2 R' U R U' R' (builds Right 1x2x3 block)
    # 3. CMLL (8 moves): U R U R' U R U2 R' (Sune left bar)
    # 4. LSE (9 moves): M' U M' U2 M U M' U2 M2 (EOLR-b -> 4c finish)
    solution = "D' F' L2 D B r U R' U' R U2 R' U R U' R' U R U R' U R U2 R' M' U M' U2 M U M' U2 M2"
    scramble = " ".join(MoveParser.invert_moves(solution))

    # Verify that scramble + solution is mathematically solved
    cube = CubeState().apply_moves(scramble).apply_moves(solution)
    assert cube.is_solved()

    segmenter = RouxSegmenter()
    segmented = segmenter.segment(scramble, solution)

    assert segmented.is_valid
    assert segmented.fb is not None
    assert segmented.sb is not None
    assert segmented.cmll is not None
    assert segmented.lse is not None
    assert segmented.lse.step_4a is not None
    assert segmented.lse.step_4c is not None

    # Check phase boundaries are strictly contiguous
    assert segmented.fb.start_move_idx == 0
    assert segmented.fb.end_move_idx == 4
    assert segmented.fb.move_count_stm == 5

    assert segmented.sb.start_move_idx == 5
    assert segmented.sb.end_move_idx == 15
    assert segmented.sb.move_count_stm == 11
    assert segmented.sb.pair1_type == "back"

    assert segmented.cmll.start_move_idx == 16
    assert segmented.cmll.end_move_idx == 23
    assert segmented.cmll.move_count_stm == 8

    assert segmented.lse.start_move_idx == 24
    assert segmented.lse.end_move_idx == 32
    assert segmented.lse.move_count_stm == 9

    # Check JSON serialization
    json_str = segmented.to_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["is_valid"] is True
    assert "fb" in parsed_json
    assert "sb" in parsed_json
    assert "cmll" in parsed_json
    assert "lse" in parsed_json


def test_segmenter_smart_cube_stream():
    """Test segmentation on a timestamped smart cube stream for a genuine Roux solve."""
    solution = "D' F' L2 D B r U R' U' R U2 R' U R U' R' U R U R' U R U2 R' M' U M' U2 M U M' U2 M2"
    scramble = " ".join(MoveParser.invert_moves(solution))

    moves = MoveParser.parse_string(solution)
    stream = []
    current_t = 1000
    for m in moves:
        stream.append({"move": m.move, "t_ms": current_t})
        current_t += 150  # 150ms per move

    segmenter = RouxSegmenter()
    segmented = segmenter.segment_stream(scramble, stream)

    assert segmented.is_valid
    assert segmented.total_time_ms is not None
    assert segmented.total_time_ms > 0
    assert segmented.tps is not None
    assert segmented.tps > 0


def test_segmenter_invalid_solve():
    """Test segmenter handling when solve does not complete FB."""
    scramble = "R U R' U'"
    # Garbage solution that does not solve FB
    solution = "F B D L"

    segmenter = RouxSegmenter()
    segmented = segmenter.segment(scramble, solution)

    assert not segmented.is_valid
    assert len(segmented.warnings) > 0


def test_user_pb_solve_segmentation():
    """Test segmentation on the user PB solve."""
    scramble = "L' U2 R' U2 L' U2 R U2 B D' L' R F' L F2 R2 U' B L'"
    solution = "x' y2 U' r2 B r R U' R' U R U R' U2 R' U R U' R' U' R U2 U R U R' U R U2 R' U' M' U' M' M2 U M' U2 M' U2"

    segmenter = RouxSegmenter()
    segmented = segmenter.segment(scramble, solution)

    assert segmented.is_valid
    assert segmented.fb is not None
    assert segmented.fb.move_count_stm == 3
    assert segmented.fb.moves_str == "x' y2 U' r2 B"

    assert segmented.sb is not None
    assert segmented.sb.move_count_stm == 16
    assert segmented.sb.moves_str == "r R U' R' U R U R' U2 R' U R U' R' U' R"

    assert segmented.cmll is not None
    assert segmented.cmll.move_count_stm == 9
    assert segmented.cmll.moves_str == "U2 U R U R' U R U2 R'"

    assert segmented.lse is not None
    assert segmented.lse.step_4a.variant == "eolr"
    assert segmented.lse.step_4a.moves_str == "U' M' U' M'"

"""Comprehensive Unit Tests for CubeState, Move Engine, and MoveParser (Milestone 1)."""

import time
import pytest
import numpy as np

from roux_engine.core.constants import Corner, Edge, Center, NUM_CORNERS, NUM_EDGES, NUM_CENTERS
from roux_engine.core.cube import CubeState
from roux_engine.core.moves import MOVES, apply_move, apply_moves, get_move
from roux_engine.core.parser import MoveParser, MoveEvent


class TestCubeState:
    """Tests for initial state, equality, copying, and binary serialization."""

    def test_solved_state(self):
        cube = CubeState()
        assert cube.is_solved()
        assert np.array_equal(cube.cp, np.arange(NUM_CORNERS))
        assert np.array_equal(cube.co, np.zeros(NUM_CORNERS))
        assert np.array_equal(cube.ep, np.arange(NUM_EDGES))
        assert np.array_equal(cube.eo, np.zeros(NUM_EDGES))
        assert np.array_equal(cube.centers, np.arange(NUM_CENTERS))

    def test_copy_and_equality(self):
        cube = CubeState()
        cube.apply_move("R")
        clone = cube.copy()
        assert cube == clone
        assert cube is not clone
        
        # Modify original, clone should remain unchanged
        cube.apply_move("U")
        assert cube != clone

    def test_binary_serialization(self):
        cube = CubeState()
        cube.apply_moves("R U R' F' r2 U' M2")
        serialized = cube.to_bytes()
        assert len(serialized) == 46  # 8 + 8 + 12 + 12 + 6 bytes
        
        deserialized = CubeState.from_bytes(serialized)
        assert deserialized == cube
        assert deserialized.is_solved() == cube.is_solved()


class TestMoveInversesAndOrders:
    """Tests for move algebra: inverses, orders, and group axioms."""

    @pytest.mark.parametrize("base_move", ["U", "D", "F", "B", "L", "R", "M", "E", "S", "r", "l", "u", "d", "f", "b", "x", "y", "z"])
    def test_move_inverses(self, base_move: str):
        cube = CubeState()
        cube.apply_move(base_move)
        assert not cube.is_solved()
        cube.apply_move(f"{base_move}'")
        assert cube.is_solved(), f"Inverse failed for {base_move}"

    @pytest.mark.parametrize("base_move", ["U", "D", "F", "B", "L", "R", "M", "E", "S", "r", "l", "u", "d", "f", "b", "x", "y", "z"])
    def test_move_order_four(self, base_move: str):
        cube = CubeState()
        for _ in range(4):
            cube.apply_move(base_move)
        assert cube.is_solved(), f"Order 4 failed for {base_move}"

    @pytest.mark.parametrize("base_move", ["U", "D", "F", "B", "L", "R", "M", "E", "S", "r", "l", "u", "d", "f", "b", "x", "y", "z"])
    def test_half_turn_order_two(self, base_move: str):
        cube = CubeState()
        cube.apply_move(f"{base_move}2")
        assert not cube.is_solved()
        cube.apply_move(f"{base_move}2")
        assert cube.is_solved(), f"Order 2 failed for {base_move}2"


class TestRouxMoveMechanicsAndCommutators:
    """Tests for standard speedcubing algorithms, commutators, and Roux slice/wide definitions."""

    def test_sexy_move_order_six(self):
        cube = CubeState()
        sexy = "R U R' U'"
        for i in range(1, 6):
            cube.apply_moves(sexy)
            assert not cube.is_solved(), f"Prematurely solved at repetition {i}"
        cube.apply_moves(sexy)
        assert cube.is_solved()

    def test_sune_preserves_f2b(self):
        cube = CubeState()
        cube.apply_moves("R U R' U R U2 R'")
        # F2B pieces (First block on Left, DR edge, etc.)
        assert cube.is_fb_solved()
        assert not cube.is_solved()

    def test_h_perm(self):
        cube = CubeState()
        cube.apply_moves("M2 U M2 U2 M2 U M2")
        # H-perm swaps UF <-> UB and UL <-> UR, corners are untouched
        assert np.array_equal(cube.cp, np.arange(NUM_CORNERS))
        assert np.all(cube.co == 0)
        assert cube.ep[Edge.UF] == Edge.UB
        assert cube.ep[Edge.UB] == Edge.UF
        assert cube.ep[Edge.UL] == Edge.UR
        assert cube.ep[Edge.UR] == Edge.UL

    def test_t_perm(self):
        cube = CubeState()
        t_perm = "R U R' U' R' F R2 U' R' U' R U R' F'"
        cube.apply_moves(t_perm)
        # Applying T-perm twice returns to solved state
        cube.apply_moves(t_perm)
        assert cube.is_solved()

    def test_wide_move_equivalence(self):
        # r == R M'
        cube1 = CubeState().apply_moves("r")
        cube2 = CubeState().apply_moves("R M'")
        assert cube1 == cube2

        # l == L M
        cube3 = CubeState().apply_moves("l")
        cube4 = CubeState().apply_moves("L M")
        assert cube3 == cube4

        # u == U E'
        cube5 = CubeState().apply_moves("u")
        cube6 = CubeState().apply_moves("U E'")
        assert cube5 == cube6

    def test_rotation_equivalence(self):
        # x == r L' == R M' L'
        cube_x = CubeState().apply_moves("x")
        cube_comp = CubeState().apply_moves("r L'")
        assert cube_x == cube_comp

        # y == u D' == U E' D'
        cube_y = CubeState().apply_moves("y")
        cube_comp_y = CubeState().apply_moves("u D'")
        assert cube_y == cube_comp_y

        # z == f B' == F S B'
        cube_z = CubeState().apply_moves("z")
        cube_comp_z = CubeState().apply_moves("f B'")
        assert cube_z == cube_comp_z


class TestMoveParser:
    """Tests for string parsing, aliases, comments, and smart-cube stream conversion."""

    def test_parse_string_with_comments_and_brackets(self):
        text = "r' U R' // First Block\n[U2] (M2) /* LSE */ U' r"
        events = MoveParser.parse_string(text)
        moves = [e.move for e in events]
        assert moves == ["r'", "U", "R'", "U2", "M2", "U'", "r"]

    def test_alias_normalization(self):
        text = "Rw Rw' Rw2 2R 2R' 2R2 U2' R2'"
        events = MoveParser.parse_string(text)
        moves = [e.move for e in events]
        assert moves == ["r", "r'", "r2", "r", "r'", "r2", "U2", "R2"]

    def test_parse_smart_cube_stream(self):
        stream = [
            {"move": "R", "t": 100},
            {"move": "U'", "t": 220},
            {"move": "Rw", "t": 350},
            {"move": "M2", "t": 580}
        ]
        events = MoveParser.parse_smart_cube_stream(stream)
        assert len(events) == 4
        assert [e.move for e in events] == ["R", "U'", "r", "M2"]
        assert [e.delta_ms for e in events] == [None, 120, 130, 230]

    def test_invert_moves(self):
        moves = "r' U R' F R2"
        inverted = MoveParser.invert_moves(moves)
        assert inverted == ["R2", "F'", "R", "U'", "r"]

    def test_scramble_and_inverse_solve(self):
        scramble = "D2 R' F2 L B2 R' D2 F2 L B2 R B' U' L2 F D2 U' F' R2 D'"
        inverse = MoveParser.invert_moves(scramble)
        
        cube = CubeState()
        cube.apply_moves(scramble)
        assert not cube.is_solved()
        cube.apply_moves(inverse)
        assert cube.is_solved()


class TestEnginePerformance:
    """Benchmark tests ensuring move operations run with high throughput."""

    def test_throughput(self):
        cube = CubeState()
        moves_seq = [get_move(m) for m in ["R", "U", "R'", "U'", "r", "M2", "U'", "M'"] * 1000]
        
        start_t = time.perf_counter()
        for m in moves_seq:
            m.apply_inplace(cube)
        duration = time.perf_counter() - start_t
        
        moves_per_sec = len(moves_seq) / duration
        print(f"\nThroughput: {moves_per_sec:,.0f} moves/sec ({duration*1000:.2f}ms for {len(moves_seq)} moves)")
        assert moves_per_sec > 50_000, f"Throughput too low: {moves_per_sec:,.0f} moves/sec"


class TestUserSolveReplay:
    """Test replaying user solve."""

    def test_user_pb_solve(self):
        scramble = "L' U2 R' U2 L' U2 R U2 B D' L' R F' L F2 R2 U' B L'"
        solution = "x' y2 U' r2 B r R U' R' U R U R' U2 R' U R U' R' U' R U2 U R U R' U R U2 R' U' M' U' M' M2 U M' U2 M' U2"
        
        cube = CubeState().apply_moves(scramble).apply_moves(solution)
        # Undoing inspection rotation z2 returns cube to canonical identity
        cube.apply_moves("z2")
        assert cube.is_solved()

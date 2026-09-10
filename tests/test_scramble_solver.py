"""Tests for end-to-end RouxScrambleSolver and solve_scramble."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.parser import MoveParser
from roux_engine.solver.scramble_solver import (
    RouxScrambleSolver,
    FullSolveResult,
    solve_scramble,
)


@pytest.fixture
def sample_scramble() -> str:
    return "D2 F2 R2 B2 U L2 U2 F2 D R2 B2 R B U2 L B2 D2 F R2 B2"


def test_solve_scramble_free_style_solves_cube(sample_scramble: str):
    """Verifies that solve_scramble with free SB style produces a 100% solved cube."""
    result: FullSolveResult = solve_scramble(sample_scramble, style="free")

    assert result.is_valid is True
    assert result.scramble == sample_scramble
    assert result.style == "free"
    assert result.total_stm > 0
    assert result.duration_ms >= 0.0

    # Test applying moves to a scrambled cube reaches solved identity
    cube = CubeState().apply_moves(sample_scramble)
    cube.apply_moves(result.full_moves_str)
    assert cube.is_solved() is True

    # Move count breakdown consistency
    expected_stm = (
        result.fb.move_count
        + result.sb.move_count
        + result.cmll_stm
        + result.lse.move_count
    )
    assert result.total_stm == expected_stm


def test_solve_scramble_classical_style_solves_cube(sample_scramble: str):
    """Verifies that solve_scramble with classical SB style produces a 100% solved cube."""
    result: FullSolveResult = solve_scramble(sample_scramble, style="classical")

    assert result.is_valid is True
    assert result.style == "classical"
    assert result.sb.style == "classical"

    cube = CubeState().apply_moves(sample_scramble)
    cube.apply_moves(result.full_moves_str)
    assert cube.is_solved() is True


def test_solve_scramble_alg_cubing_url(sample_scramble: str):
    """Verifies that alg_cubing_url formats an accessible 3D visualization link."""
    result: FullSolveResult = solve_scramble(sample_scramble, style="free")
    url = result.alg_cubing_url

    assert url.startswith("https://alg.cubing.net/?")
    assert "setup=" in url
    assert "alg=" in url


def test_solve_scramble_json_serialization(sample_scramble: str):
    """Verifies full JSON and dictionary serialization of solve results."""
    result: FullSolveResult = solve_scramble(sample_scramble, style="free")
    d = result.to_dict()

    assert d["scramble"] == sample_scramble
    assert d["is_valid"] is True
    assert d["total_stm"] == result.total_stm
    assert "fb" in d
    assert "sb" in d
    assert "cmll" in d
    assert "lse" in d
    assert "full_solution" in d

def test_solve_scramble_square_pair_style_solves_cube(sample_scramble: str):
    """Verifies that solve_scramble with square_pair SB style produces a 100% solved cube."""
    result: FullSolveResult = solve_scramble(sample_scramble, style="square_pair")

    assert result.is_valid is True
    assert result.style == "square_pair"
    assert result.sb.style == "square_pair"

    cube = CubeState().apply_moves(sample_scramble)
    cube.apply_moves(result.full_moves_str)
    assert cube.is_solved() is True


@pytest.mark.parametrize(
    "scramble",
    [
        "R' U' F R2 B2 D2 L' D2 B2 F2 R' B2 R' D U2 F' L D F2 D2 R' F2 R' U' F",
        "D' R2 B2 D F2 D' B2 L2 D2 L2 R2 B' L R' D' B D2 L D2 B2 R'",
        "B2 R2 D2 F2 D' F2 L2 D' R2 U' L' U' B F' L' U R F' R' U2",
    ],
)
def test_solve_scramble_diverse_scrambles(scramble: str):
    """Verifies that diverse WCA scrambles are 100% solved."""
    result = solve_scramble(scramble, style="free")
    assert result.is_valid is True
    cube = CubeState().apply_moves(scramble)
    cube.apply_moves(result.full_moves_str)
    assert cube.is_solved(allow_rotations=True) is True

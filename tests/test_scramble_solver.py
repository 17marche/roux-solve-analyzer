"""Tests for end-to-end RouxScrambleSolver and solve_scramble."""

import json
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
    assert cube.is_solved(allow_rotations=True) is True

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
    assert cube.is_solved(allow_rotations=True) is True


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
    assert cube.is_solved(allow_rotations=True) is True


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


def test_solve_scramble_ergonomic_metrics_default(sample_scramble: str):
    """Verifies that default solve_scramble returns full ergonomic metrics and consistent E-STM sums."""
    result: FullSolveResult = solve_scramble(sample_scramble)

    assert result.is_valid is True
    assert result.rank_by == "e_stm"
    assert result.profile == "2H"
    assert result.m_slice_hand == "right"

    # Per-phase E-STM metrics
    assert result.fb.e_stm is not None and result.fb.e_stm > 0
    assert result.sb.e_stm is not None and result.sb.e_stm > 0
    assert result.cmll_e_stm > 0
    assert result.lse.e_stm is not None and result.lse.e_stm > 0

    # Total E-STM and Kinematic Flow Efficiency
    expected_e_stm = result.fb.e_stm + result.sb.e_stm + result.cmll_e_stm + result.lse.e_stm
    assert result.total_e_stm == pytest.approx(expected_e_stm, abs=1e-3)

    expected_eff = min(100.0, (result.total_stm / result.total_e_stm * 100.0))
    assert result.kinematic_efficiency == pytest.approx(expected_eff, abs=0.05)
    assert result.kinematic_efficiency <= 100.0


def test_solve_scramble_rank_by_stm(sample_scramble: str):
    """Verifies that solve_scramble accepts rank_by='stm' and populates ergonomic metrics."""
    result: FullSolveResult = solve_scramble(sample_scramble, rank_by="stm")

    assert result.is_valid is True
    assert result.rank_by == "stm"
    assert result.fb.e_stm is not None
    assert result.sb.e_stm is not None
    assert result.lse.e_stm is not None
    assert result.cmll_e_stm > 0
    assert result.total_e_stm > 0
    assert result.kinematic_efficiency <= 100.0


def test_solve_scramble_profile_and_hand_preference(sample_scramble: str):
    """Verifies that solve_scramble accepts profile and m_slice_hand configurations."""
    result_left = solve_scramble(sample_scramble, profile="2H", m_slice_hand="left")
    assert result_left.is_valid is True
    assert result_left.profile == "2H"
    assert result_left.m_slice_hand == "left"

    result_oh = solve_scramble(sample_scramble, profile="OH")
    assert result_oh.is_valid is True
    assert result_oh.profile == "OH"


def test_solve_scramble_json_serialization_ergonomics(sample_scramble: str):
    """Verifies that to_dict and to_json serialize all ergonomic metrics cleanly."""
    result = solve_scramble(sample_scramble)
    d = result.to_dict()

    assert "total_e_stm" in d
    assert d["total_e_stm"] == pytest.approx(result.total_e_stm, abs=1e-3)
    assert "kinematic_efficiency" in d
    assert d["kinematic_efficiency"] == pytest.approx(result.kinematic_efficiency, abs=1e-2)
    assert d["rank_by"] == "e_stm"
    assert d["profile"] == "2H"
    assert d["m_slice_hand"] == "right"

    assert "e_stm" in d["fb"]
    assert d["fb"]["e_stm"] == result.fb.e_stm
    assert "e_stm" in d["sb"]
    assert d["sb"]["e_stm"] == result.sb.e_stm
    assert "e_stm" in d["cmll"]
    assert d["cmll"]["e_stm"] == result.cmll_e_stm
    assert "e_stm" in d["lse"]
    assert d["lse"]["e_stm"] == result.lse.e_stm

    # Check to_json() valid JSON parsing
    parsed = json.loads(result.to_json())
    assert parsed["total_e_stm"] == d["total_e_stm"]
    assert parsed["kinematic_efficiency"] == d["kinematic_efficiency"]


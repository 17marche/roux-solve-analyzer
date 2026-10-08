"""Tests for roux solve CLI subcommand."""

import json
import pytest
from roux_engine.cli import main, handle_solve, format_solver_report


@pytest.fixture
def sample_scramble() -> str:
    return "D2 F2 R2 B2 U L2 U2 F2 D R2 B2 R B U2 L B2 D2 F R2 B2"


class TestCliSolve:
    """Seam 2: Unified Solve CLI (roux solve)."""

    def test_solve_cli_defaults_text_report(self, sample_scramble: str, capsys):
        code = main(["solve", "-s", sample_scramble])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX SCRAMBLE SOLVER REPORT" in out
        assert "Total:" in out
        assert "STM moves" in out
        assert "E-STM" in out
        assert "Kinematic Flow Efficiency" in out
        assert "Profile:" in out
        assert "2H" in out
        assert "right" in out

        # Phase breakdown must report both STM and E-STM for all 4 phases
        assert "First Block (FB):" in out
        assert "Second Block (SB):" in out
        assert "CMLL:" in out
        assert "Last Six Edges (LSE):" in out

        # Each phase line or sub-line must include E-STM
        fb_section = out.split("First Block (FB):")[1].split("Second Block (SB):")[0]
        assert "E-STM" in fb_section

        sb_section = out.split("Second Block (SB):")[1].split("CMLL:")[0]
        assert "E-STM" in sb_section

        cmll_section = out.split("CMLL:")[1].split("Last Six Edges (LSE):")[0]
        assert "E-STM" in cmll_section

        lse_section = out.split("Last Six Edges (LSE):")[1].split("Full Solution:")[0]
        assert "E-STM" in lse_section

        assert "Full Solution:" in out
        assert "3D Interactive Visualization" in out

    def test_solve_cli_json_report(self, sample_scramble: str, capsys):
        code = main(["solve", "-s", sample_scramble, "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["is_valid"] is True
        assert data["total_stm"] > 0
        assert "total_e_stm" in data and data["total_e_stm"] > 0
        assert "kinematic_efficiency" in data
        assert 0.0 < data["kinematic_efficiency"] <= 100.0
        assert data["rank_by"] == "e_stm"
        assert data["profile"] == "2H"
        assert data["m_slice_hand"] == "right"

        assert "e_stm" in data["fb"] and data["fb"]["e_stm"] > 0
        assert "e_stm" in data["sb"] and data["sb"]["e_stm"] > 0
        assert "e_stm" in data["cmll"] and data["cmll"]["e_stm"] > 0
        assert "e_stm" in data["lse"] and data["lse"]["e_stm"] > 0

    def test_solve_cli_rank_by_stm(self, sample_scramble: str, capsys):
        code_json = main(["solve", "-s", sample_scramble, "--rank-by", "stm", "--json"])
        assert code_json == 0
        data = json.loads(capsys.readouterr().out)
        assert data["rank_by"] == "stm"
        assert "total_e_stm" in data

        code_text = main(["solve", "-s", sample_scramble, "--rank-by", "stm"])
        assert code_text == 0
        out = capsys.readouterr().out
        assert "ROUX SCRAMBLE SOLVER REPORT" in out
        assert "ranked by stm" in out
        assert "E-STM" in out

    def test_solve_cli_profile_oh(self, sample_scramble: str, capsys):
        code_json = main(["solve", "-s", sample_scramble, "--profile", "OH", "--json"])
        assert code_json == 0
        data = json.loads(capsys.readouterr().out)
        assert data["profile"] == "OH"

        code_text = main(["solve", "-s", sample_scramble, "--profile", "OH"])
        assert code_text == 0
        out = capsys.readouterr().out
        assert "ROUX SCRAMBLE SOLVER REPORT" in out
        assert "Profile:  OH [ranked by e_stm]" in out

    def test_solve_cli_m_slice_hand_left(self, sample_scramble: str, capsys):
        code_json = main(["solve", "-s", sample_scramble, "--m-slice-hand", "left", "--json"])
        assert code_json == 0
        data = json.loads(capsys.readouterr().out)
        assert data["m_slice_hand"] == "left"

        code_text = main(["solve", "-s", sample_scramble, "--m-slice-hand", "left"])
        assert code_text == 0
        out = capsys.readouterr().out
        assert "ROUX SCRAMBLE SOLVER REPORT" in out
        assert "left-handed M-slice" in out

    def test_solve_cli_positional_scramble(self, sample_scramble: str, capsys):
        code = main(["solve", sample_scramble])
        assert code == 0
        captured = capsys.readouterr()
        assert "ROUX SCRAMBLE SOLVER REPORT" in captured.out

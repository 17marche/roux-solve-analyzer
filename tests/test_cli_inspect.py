"""Tests for roux inspect CLI subcommand and PhaseInspector."""

import json
import pytest
from roux_engine.cli import main


@pytest.fixture
def sample_scramble() -> str:
    return "D2 F2 R2 B2 U L2 U2 F2 D R2 B2 R B U2 L B2 D2 F R2 B2"


class TestPhaseInspectorFB:
    """Slice 1: First Block Inspection and Kinematic Grip Telemetry."""

    def test_fb_inspection_text_report(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "fb", "--top-k", "3"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX ERGONOMIC PHASE INSPECTOR" in out
        assert "Phase:      First Block (FB)" in out
        assert "STM:" in out
        assert "E-STM:" in out
        assert "Flow Efficiency:" in out
        assert "Grip:" in out
        assert "Start in HOME" in out

    def test_kinematic_grip_telemetry_zero_regrips(self, capsys):
        # Sequence with 0 regrips: r U R' U2 R' U'
        # We test grip formatting on a known clean sequence via inspect
        code = main(["inspect", "-s", "r U R' U2 R' U'", "--phase", "fb", "--top-k", "1"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "0 regrips" in out
        assert "Regrip between move" not in out

    def test_kinematic_grip_telemetry_forced_regrip(self, capsys):
        # R R forces a regrip between move 1 (R) and move 2 (R): from R_AWAY to HOME
        # Test directly via grip reporting helper or CLI
        from roux_engine.inspector.phase_inspector import format_grip_telemetry
        lines = format_grip_telemetry(["R", "R"])
        assert any("1 regrip" in l for l in lines)
        assert any("Regrip between move 1 (R) and move 2 (R): from R_AWAY to HOME" in l for l in lines)


class TestPhaseInspectorSB:
    """Slice 2: Second Block Auto-Cascading, Multi-Paradigm Comparison & Style Drilldown."""

    def test_sb_multi_paradigm_comparison_table(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "sb"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX ERGONOMIC PHASE INSPECTOR" in out
        assert "Phase:      Second Block (SB)" in out
        assert "Preceding Setup Moves:" in out
        assert "First Block (FB):" in out
        assert "Second Block Multi-Paradigm Comparison:" in out
        assert "Free Blockbuilding" in out
        assert "Classical (Back-first)" in out
        assert "Classical (Front-first)" in out
        assert "Square + Pair" in out
        assert "Macro-Trigger" in out
        assert "*" in out  # best candidate indicator

    def test_sb_style_drilldown(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "sb", "--style", "classical", "--top-k", "3"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX ERGONOMIC PHASE INSPECTOR" in out
        assert "Candidate Solutions (Top" in out
        assert "Multi-Paradigm Comparison" not in out
        assert "STM:" in out
        assert "E-STM:" in out
        assert "Flow Efficiency:" in out
        assert "Grip:" in out

    def test_sb_style_drilldown_order(self, sample_scramble: str, capsys):
        code = main([
            "inspect", "-s", sample_scramble, "--phase", "sb",
            "--style", "classical", "--order", "back_first", "--top-k", "2",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Candidate Solutions (Top" in captured.out


class TestPhaseInspectorLSE:
    """Slice 3: Last Six Edges Auto-Cascading, Multi-Path Comparison & Micro-Step Drilldown."""

    def test_lse_multi_path_comparison_table(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "lse"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX ERGONOMIC PHASE INSPECTOR" in out
        assert "Phase:      Last Six Edges (LSE)" in out
        assert "Preceding Setup Moves:" in out
        assert "First Block (FB):" in out
        assert "Second Block (SB):" in out
        assert "Last Six Edges Multi-Path Comparison:" in out
        assert "Standard EO" in out
        assert "EOLR Aligned" in out
        assert "EOLR Misoriented" in out
        assert "EOLR-b" in out
        assert "*" in out  # best path indicator
        assert "4a:" in out
        assert "4b:" in out
        assert "4c:" in out

    def test_lse_target_drilldown_eolr(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "lse", "--target", "eolr", "--top-k", "3"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out

        assert "ROUX ERGONOMIC PHASE INSPECTOR" in out
        assert "Candidate Solutions (Top" in out
        assert "Multi-Path Comparison" not in out
        assert "STM:" in out
        assert "E-STM:" in out
        assert "Flow Efficiency:" in out
        assert "Grip:" in out

    def test_lse_target_drilldown_4b(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "lse", "--target", "4b", "--top-k", "2"])
        assert code == 0
        captured = capsys.readouterr()
        out = captured.out
        assert "Candidate Solutions (Top" in out


class TestPhaseInspectorOverridesAndErrors:
    """Slice 4: Preceding State Overrides and Error Handling."""

    def test_manual_fb_override_valid(self, capsys):
        # A simple 2-move scramble with a known 2-move FB solution: U' R' solved by R U
        scramble = "U' R'"
        fb_sol = "R U"
        code = main(["inspect", "-s", scramble, "--phase", "sb", "--fb", fb_sol])
        assert code == 0
        captured = capsys.readouterr()
        assert "ROUX ERGONOMIC PHASE INSPECTOR" in captured.out
        assert fb_sol in captured.out

    def test_manual_fb_override_invalid_blocks(self, sample_scramble: str, capsys):
        # Providing an arbitrary move that does not solve FB must error
        code = main(["inspect", "-s", sample_scramble, "--phase", "sb", "--fb", "U"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "First Block is not solved" in captured.err

    def test_manual_moves_override_invalid_for_lse(self, sample_scramble: str, capsys):
        # Providing moves that don't solve FB+SB must error when inspecting LSE
        code = main(["inspect", "-s", sample_scramble, "--phase", "lse", "--moves", "U R U' R'"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "First Block and Second Block are not solved" in captured.err

    def test_manual_sb_override_invalid_for_lse(self, sample_scramble: str, capsys):
        # An invalid SB override that fails to solve SB
        code = main([
            "inspect", "-s", sample_scramble, "--phase", "lse",
            "--fb", "r U R' U2 R' U'", "--sb", "U",
        ])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "not solved" in captured.err

    def test_invalid_move_token_produces_error(self, capsys):
        code = main(["inspect", "-s", "INVALID_MOVE", "--phase", "fb"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err


class TestPhaseInspectorFlagsAndJson:
    """Slice 5: Profile, Handedness, Top-K, Positional Scramble, and JSON Output."""

    def test_inspect_json_fb(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "fb", "--top-k", "3", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert data["phase"] == "fb"
        assert data["scramble"] == sample_scramble
        assert data["profile"] == "2H"
        assert data["m_slice_hand"] == "right"
        assert len(data["candidates"]) == 3
        cand = data["candidates"][0]
        assert cand["stm"] > 0
        assert cand["e_stm"] > 0
        assert cand["flow_efficiency"] > 0
        assert "grip" in cand
        assert cand["grip"]["start"] == "HOME"
        assert "regrip_count" in cand["grip"]
        assert "regrips" in cand["grip"]

    def test_inspect_json_sb_comparison(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "sb", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)

        assert data["phase"] == "sb"
        assert data["is_comparison"] is True
        assert "comparison_table" in data
        table = data["comparison_table"]
        assert "Free Blockbuilding" in table
        assert "Classical (Back-first)" in table
        assert "Classical (Front-first)" in table
        assert "Square + Pair" in table
        assert "Macro-Trigger" in table

    def test_inspect_json_lse_comparison(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "lse", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)

        assert data["phase"] == "lse"
        assert data["is_comparison"] is True
        assert "comparison_table" in data
        table = data["comparison_table"]
        assert "Standard EO" in table
        assert "EOLR Aligned" in table
        assert "EOLR Misoriented" in table
        assert "EOLR-b" in table

    def test_inspect_profile_oh(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "fb", "--profile", "OH", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["profile"] == "OH"

    def test_inspect_m_slice_hand_left(self, sample_scramble: str, capsys):
        code = main(["inspect", "-s", sample_scramble, "--phase", "fb", "--m-slice-hand", "left", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["m_slice_hand"] == "left"

    def test_inspect_positional_scramble(self, sample_scramble: str, capsys):
        code = main(["inspect", sample_scramble, "--phase", "fb", "--top-k", "1"])
        assert code == 0
        captured = capsys.readouterr()
        assert "ROUX ERGONOMIC PHASE INSPECTOR" in captured.out





"""Tests for roux flow CLI subcommand."""

import json
from pathlib import Path
import pytest
from roux_engine.cli import main, handle_flow, format_flow_report
from roux_engine.ergonomics.flow_scorer import FlowScorer
from roux_engine.ergonomics.models import HandProfile


class TestCliFlow:
    """Seam 2: Unified Flow CLI (roux flow)."""

    def test_flow_static_moves_text_report(self, capsys):
        code = main(["flow", "R U R' U'"])
        assert code == 0
        captured = capsys.readouterr()
        assert "ROUX BIOMECHANICAL FLOW REPORT" in captured.out
        assert "Raw STM:" in captured.out
        assert "Effective STM (E-STM):" in captured.out
        assert "Kinematic Flow Efficiency:" in captured.out
        assert "Regrip Count:" in captured.out

    def test_flow_static_moves_json_report(self, capsys):
        code = main(["flow", "R U R' U'", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 4
        assert "e_stm" in data
        assert "kinematic_efficiency" in data
        assert data["regrip_count"] == 0

    def test_flow_macro_trigger_reporting(self, capsys):
        code = main(["flow", "R' F R F'"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Macro Triggers:" in captured.out
        assert "sledgehammer" in captured.out

    def test_flow_with_profile_oh(self, capsys):
        code_2h = main(["flow", "R U R' U'", "--json", "--profile", "2H"])
        out_2h = json.loads(capsys.readouterr().out)

        code_oh = main(["flow", "R U R' U'", "--json", "--profile", "OH"])
        out_oh = json.loads(capsys.readouterr().out)

        assert code_2h == 0
        assert code_oh == 0
        assert out_2h["profile"] == "2H"
        assert out_oh["profile"] == "OH"

    def test_flow_with_tempo(self, capsys):
        code = main(["flow", "R U R' U'", "--tempo", "0.25", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data.get("tempo") == 0.25

    def test_flow_with_stream_json_file(self, tmp_path, capsys):
        stream_data = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 250},
            {"move": "R'", "timestamp_ms": 400},
            {"move": "U'", "timestamp_ms": 550},
        ]
        stream_file = tmp_path / "stream.json"
        stream_file.write_text(json.dumps(stream_data))

        code = main(["flow", str(stream_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert "ROUX BIOMECHANICAL FLOW REPORT" in captured.out
        assert "Turning Ratio:" in captured.out
        assert "Rhythm Consistency (CV):" in captured.out
        assert "Stream Flow Index:" in captured.out

    def test_flow_with_moves_text_file(self, tmp_path, capsys):
        moves_file = tmp_path / "solve.txt"
        moves_file.write_text("r U R' U2 R' U'")

        code = main(["flow", str(moves_file), "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 6
        assert data["regrip_count"] == 0

    def test_flow_empty_input_errors(self, capsys):
        code = main(["flow", ""])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

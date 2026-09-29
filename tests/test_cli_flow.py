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
        assert "Macro Triggers:              None" in captured.out

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

    def test_flow_stream_with_tempo(self, tmp_path, capsys):
        stream_data = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 500},
        ]
        stream_file = tmp_path / "stream_tempo.json"
        stream_file.write_text(json.dumps(stream_data))

        code = main(["flow", str(stream_file), "--tempo", "0.2", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data.get("tempo") == 0.2

    def test_flow_empty_input_errors(self, capsys):
        code = main(["flow", ""])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_flow_direct_json_string_argument(self, capsys):
        stream_json = json.dumps([
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 250},
            {"move": "R'", "timestamp_ms": 400},
            {"move": "U'", "timestamp_ms": 550},
        ])
        code = main(["flow", stream_json, "--json", "--profile", "2H", "--tempo", "0.2"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 4
        assert data["profile"] == "2H"
        assert data["tempo"] == 0.2
        assert "stream_flow_index" in data
        assert data["stream_flow_index"] is not None

    def test_flow_direct_json_token_list_string(self, capsys):
        tokens_json = json.dumps(["R", "U", "R'", "U'"])
        code = main(["flow", tokens_json, "-j"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 4
        assert data["regrip_count"] == 0

    def test_flow_piped_stdin_raw_moves(self, monkeypatch, capsys):
        import io
        fake_stdin = io.StringIO("R U R' U'")
        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

        code = main(["flow", "-j", "--tempo", "0.25", "--profile", "OH"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 4
        assert data["profile"] == "OH"
        assert data["tempo"] == 0.25

    def test_flow_piped_stdin_json_stream(self, monkeypatch, capsys):
        import io
        stream_json = json.dumps([
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 250},
        ])
        fake_stdin = io.StringIO(stream_json)
        monkeypatch.setattr("sys.stdin", fake_stdin)
        monkeypatch.setattr(fake_stdin, "isatty", lambda: False)

        code = main(["flow", "-j", "--tempo", "0.15"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 2
        assert data["tempo"] == 0.15
        assert data["turning_ratio"] is not None

    def test_flow_flag_moves_option(self, capsys):
        code = main(["flow", "-m", "R U R' U'", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["raw_stm"] == 4

    def test_flow_cli_source_does_not_contain_stream_pause_detector(self):
        import inspect
        import roux_engine.cli as cli_mod
        source = inspect.getsource(cli_mod)
        assert "StreamPauseDetector" not in source, "cli.py must not instantiate or import StreamPauseDetector"
        assert "simulated_stream" not in source, "cli.py must not manually fabricate simulated timestamp streams"

    def test_flow_delegates_directly_to_score(self, monkeypatch, capsys):
        from unittest.mock import MagicMock
        score_called = False

        original_score = FlowScorer.score

        def mock_score(self, moves, tempo=None, profile=None, **kwargs):
            nonlocal score_called
            score_called = True
            return original_score(self, moves, tempo=tempo, profile=profile, **kwargs)

        def forbidden_call(*args, **kwargs):
            raise AssertionError("CLI must delegate directly to FlowScorer.score(), not score_moves/score_stream")

        monkeypatch.setattr(FlowScorer, "score", mock_score)
        monkeypatch.setattr(FlowScorer, "score_moves", forbidden_call)
        monkeypatch.setattr(FlowScorer, "score_stream", forbidden_call)

        code = main(["flow", "R U R' U'", "--tempo", "0.25", "--profile", "OH"])
        assert code == 0
        assert score_called is True

    def test_flow_invalid_moves_error(self, capsys):
        code = main(["flow", "R U INVALID_MOVE"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_flow_negative_tempo_error(self, capsys):
        code = main(["flow", "R U R' U'", "--tempo", "-0.25"])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_flow_empty_file_error(self, tmp_path, capsys):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n  ")
        code = main(["flow", str(empty_file)])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err





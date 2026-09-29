"""Tests for FlowScorer, Effective STM (E-STM), and Kinematic Flow Efficiency."""

import pytest
from roux_engine.ergonomics.models import GripState, FlowScore
from roux_engine.ergonomics.flow_scorer import FlowScorer


class TestFlowScorerStaticSequences:
    """Seam 3.1: FlowScorer on static move sequences."""

    def test_empty_sequence(self):
        scorer = FlowScorer()
        score = scorer.score_moves("")
        assert score.raw_stm == 0
        assert score.e_stm == 0.0
        assert score.kinematic_efficiency == 0.0
        assert score.regrip_count == 0
        assert score.per_move_analysis == []

    def test_pure_home_grip_sequence_metrics(self):
        scorer = FlowScorer()
        # R U R' U' (4 STM) from home grip: 0 regrips, fluid triggers
        score = scorer.score_moves("R U R' U'")
        assert score.raw_stm == 4
        assert score.regrip_count == 0
        # E-STM should be lower than 4.0 due to sub-1.0 fluid bigram efforts
        assert score.e_stm < 4.0
        # Kinematic efficiency is capped at 100.0%
        assert score.kinematic_efficiency == 100.0
        assert len(score.per_move_analysis) == 4
        assert not any(m.regrip for m in score.per_move_analysis)

    def test_awkward_r_r_forces_higher_e_stm_and_lower_efficiency(self):
        scorer = FlowScorer()
        # R -> R forces a regrip because R is blocked from R_AWAY
        score_rr = scorer.score_moves("R R")
        assert score_rr.raw_stm == 2
        assert score_rr.regrip_count == 1
        assert score_rr.e_stm > 3.0
        assert score_rr.kinematic_efficiency < 70.0

        # Compare per-move effort with fluid R U R' U'
        score_fluid = scorer.score_moves("R U R' U'")
        effort_per_move_rr = score_rr.e_stm / score_rr.raw_stm
        effort_per_move_fluid = score_fluid.e_stm / score_fluid.raw_stm
        assert effort_per_move_rr > effort_per_move_fluid * 2.0
        assert score_fluid.kinematic_efficiency > score_rr.kinematic_efficiency

    def test_awkward_r2_b_forces_regrip_and_lower_efficiency(self):
        scorer = FlowScorer()
        score_r2b = scorer.score_moves("R2 B")
        assert score_r2b.raw_stm == 2
        assert score_r2b.regrip_count == 1
        assert score_r2b.e_stm > 3.5
        assert score_r2b.kinematic_efficiency < 60.0

        score_fluid = scorer.score_moves("R U R' U'")
        assert score_fluid.kinematic_efficiency > score_r2b.kinematic_efficiency

    def test_whole_cube_rotations_have_zero_stm_impact(self):
        scorer = FlowScorer()
        # Rotations count as 0 STM
        score_with_rot = scorer.score_moves("x R U R' U'")
        assert score_with_rot.raw_stm == 4
        assert score_with_rot.regrip_count == 0

    def test_score_alias_matches_score_moves(self):
        scorer = FlowScorer()
        s1 = scorer.score_moves("r U R' U' r' R U")
        s2 = scorer.score("r U R' U' r' R U")
        assert s1.raw_stm == s2.raw_stm
        assert s1.e_stm == s2.e_stm
        assert s1.regrip_count == s2.regrip_count
        assert s1.kinematic_efficiency == s2.kinematic_efficiency

    def test_move_event_sequence_with_timestamps(self):
        from roux_engine.core.parser import MoveEvent
        scorer = FlowScorer()
        events = [
            MoveEvent(move="R", raw_token="R", timestamp_ms=100, delta_ms=100),
            MoveEvent(move="U", raw_token="U", timestamp_ms=250, delta_ms=150),
        ]
        score = scorer.score_moves(events)
        assert score.raw_stm == 2
        assert len(score.per_move_analysis) == 2
        assert score.per_move_analysis[0].timestamp_ms == 100
        assert score.per_move_analysis[0].delta_ms == 100
        assert score.per_move_analysis[1].timestamp_ms == 250
        assert score.per_move_analysis[1].delta_ms == 150


class TestFlowScorerStreamScoring:
    """Seam 3.4: FlowScorer stream flow scoring, pace invariance, and pause breakdown."""

    def test_score_stream_with_dict_stream(self):
        scorer = FlowScorer()
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
            {"move": "R'", "timestamp_ms": 340},
            {"move": "U'", "timestamp_ms": 460},
        ]
        score = scorer.score_stream(stream)
        assert score.raw_stm == 4
        assert score.turning_ratio is not None
        assert score.rhythm_cv is not None
        assert score.stream_flow_index is not None
        assert score.turning_ratio == 1.0
        assert score.rhythm_cv == 0.0
        assert score.stream_flow_index == 100.0

    def test_score_stream_pace_invariance(self):
        scorer = FlowScorer()
        stream_7s = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 120},
            {"move": "R'", "timestamp_ms": 240},
            {"move": "U'", "timestamp_ms": 360},
            {"move": "R", "timestamp_ms": 1200},  # Pause
            {"move": "U", "timestamp_ms": 1320},
        ]
        stream_14s = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 240},
            {"move": "R'", "timestamp_ms": 480},
            {"move": "U'", "timestamp_ms": 720},
            {"move": "R", "timestamp_ms": 2400},  # Pause
            {"move": "U", "timestamp_ms": 2640},
        ]

        score_7s = scorer.score_stream(stream_7s)
        score_14s = scorer.score_stream(stream_14s)

        assert score_7s.raw_stm == score_14s.raw_stm == 6
        assert round(score_7s.turning_ratio, 4) == round(score_14s.turning_ratio, 4)
        assert round(score_7s.rhythm_cv, 4) == round(score_14s.rhythm_cv, 4)
        assert round(score_7s.stream_flow_index, 2) == round(score_14s.stream_flow_index, 2)

    def test_score_detects_stream_automatically(self):
        scorer = FlowScorer()
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
        ]
        score = scorer.score(stream)
        assert score.turning_ratio is not None
        assert score.stream_flow_index is not None

    def test_score_stream_pause_breakdown_and_per_move_analysis(self):
        scorer = FlowScorer()
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "R", "timestamp_ms": 600},   # 500ms: R->R forced regrip pause -> PHYSICAL_REGRIP
            {"move": "U", "timestamp_ms": 720},   # 120ms
            {"move": "R'", "timestamp_ms": 840},  # 120ms
            {"move": "U'", "timestamp_ms": 960},  # 120ms
        ]
        score = scorer.score_stream(stream)
        assert len(score.per_move_analysis) == 5
        assert score.per_move_analysis[1].regrip is True
        assert score.per_move_analysis[1].pause_type == "PHYSICAL_REGRIP"
        assert score.pause_breakdown.get("PHYSICAL_REGRIP", 0) == 1


class TestFlowScorerUnifiedInterface:
    """Seam: Unified FlowScorer.score() entry point."""

    def test_score_accepts_various_input_formats(self):
        scorer = FlowScorer()

        # 1. Space-delimited string
        s_str = scorer.score("R U R' U'")
        assert s_str.raw_stm == 4
        assert [m.move for m in s_str.per_move_analysis] == ["R", "U", "R'", "U'"]

        # 2. Token list of strings
        s_list = scorer.score(["R", "U", "R'", "U'"])
        assert s_list.raw_stm == 4
        assert s_list.e_stm == s_str.e_stm

        # 3. Raw JSON string of token list
        s_json_tokens = scorer.score('["R", "U", "R\'", "U\'"]')
        assert s_json_tokens.raw_stm == 4
        assert s_json_tokens.e_stm == s_str.e_stm

        # 4. Raw JSON string of smart-cube stream dicts
        s_json_stream = scorer.score(
            '[{"move": "R", "timestamp_ms": 100}, {"move": "U", "timestamp_ms": 220}]'
        )
        assert s_json_stream.raw_stm == 2
        assert s_json_stream.turning_ratio is not None
        assert s_json_stream.per_move_analysis[0].timestamp_ms == 100

        # 5. Smart-cube stream list of dicts
        s_stream_list = scorer.score([
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
        ])
        assert s_stream_list.raw_stm == 2
        assert s_stream_list.turning_ratio is not None
        assert s_stream_list.per_move_analysis[1].delta_ms == 120

    def test_score_invalid_cube_tokens_and_args_raise_value_error(self):
        scorer = FlowScorer()

        # 1. Invalid token in string
        with pytest.raises(ValueError, match="Invalid cube move token: 'INVALID'"):
            scorer.score("R U INVALID U'")

        # 2. Invalid token in sequence of strings
        with pytest.raises(ValueError, match="Invalid cube move token: 'BAD_TOKEN'"):
            scorer.score(["R", "BAD_TOKEN", "U"])

        # 3. Invalid token in JSON token array
        with pytest.raises(ValueError, match="Invalid cube move token: 'NOT_A_MOVE'"):
            scorer.score('["R", "NOT_A_MOVE"]')

        # 4. Invalid token in stream dict sequence
        with pytest.raises(ValueError, match="Invalid cube move token: 'WRONG'"):
            scorer.score([{"move": "R"}, {"move": "WRONG"}])

        # 5. Invalid token in JSON stream string
        with pytest.raises(ValueError, match="Invalid cube move token: 'CORRUPT'"):
            scorer.score('[{"move": "CORRUPT", "timestamp_ms": 100}]')

        # 6. Negative or zero tempo raises ValueError
        with pytest.raises(ValueError, match="Tempo must be a positive float"):
            scorer.score("R U", tempo=-0.5)

        with pytest.raises(ValueError, match="Tempo must be a positive float"):
            scorer.score("R U", tempo=0.0)

        # 7. Unsupported profile raises ValueError
        with pytest.raises(ValueError, match="Unsupported profile: '3H'"):
            scorer.score("R U", profile="3H")

    def test_score_empty_inputs_return_zeroed_metrics_and_collections(self):
        scorer = FlowScorer()

        empty_cases = [
            "",
            "   ",
            [],
            "[]",
            None,
        ]

        for inp in empty_cases:
            score = scorer.score(inp)
            assert score.raw_stm == 0
            assert score.e_stm == 0.0
            assert score.kinematic_efficiency == 0.0
            assert score.regrip_count == 0
            assert score.macro_triggers == []
            assert score.per_move_analysis == []
            assert score.pause_breakdown == {}

        # Empty input with tempo calculates zeroed stream metrics
        score_tempo = scorer.score("", tempo=0.25)
        assert score_tempo.raw_stm == 0
        assert score_tempo.e_stm == 0.0
        assert score_tempo.kinematic_efficiency == 0.0
        assert score_tempo.regrip_count == 0
        assert score_tempo.turning_ratio == 0.0
        assert score_tempo.rhythm_cv == 0.0
        assert score_tempo.stream_flow_index == 0.0
        assert score_tempo.macro_triggers == []
        assert score_tempo.per_move_analysis == []
        assert score_tempo.pause_breakdown == {}

    def test_score_static_moves_with_tempo_synthesizes_uniform_timestamps(self):
        scorer = FlowScorer()

        # 4 moves at tempo 0.25s (4 TPS) -> 250ms intervals
        score = scorer.score("R U R' U'", tempo=0.25)
        assert score.raw_stm == 4
        assert score.turning_ratio == 1.0
        assert score.rhythm_cv == 0.0
        assert score.stream_flow_index == 100.0

        assert len(score.per_move_analysis) == 4
        expected_timestamps = [0, 250, 500, 750]
        for idx, (analysis, expected_t) in enumerate(zip(score.per_move_analysis, expected_timestamps)):
            assert analysis.timestamp_ms == expected_t
            if idx == 0:
                assert analysis.delta_ms in (0, None)
            else:
                assert analysis.delta_ms == 250

        # Token list with tempo 0.20s (5 TPS) -> 200ms intervals
        score_tokens = scorer.score(["R", "U", "R'", "U'"], tempo=0.20)
        assert score_tokens.raw_stm == 4
        assert score_tokens.turning_ratio == 1.0
        assert score_tokens.rhythm_cv == 0.0
        assert score_tokens.stream_flow_index == 100.0
        assert [m.timestamp_ms for m in score_tokens.per_move_analysis] == [0, 200, 400, 600]
        assert [m.delta_ms for m in score_tokens.per_move_analysis[1:]] == [200, 200, 200]

    def test_score_stream_with_tempo_calibrates_pause_threshold_without_altering_timestamps(self):
        scorer = FlowScorer()

        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 250},   # 150ms
            {"move": "R'", "timestamp_ms": 600},  # 350ms
            {"move": "U'", "timestamp_ms": 750},  # 150ms
        ]

        # With tempo = 0.40 (400ms threshold): 350ms is below threshold -> 0 pauses detected
        score_relaxed = scorer.score(stream, tempo=0.40)
        assert sum(score_relaxed.pause_breakdown.values()) == 0
        assert not any(m.pause_type is not None for m in score_relaxed.per_move_analysis)

        # Timestamps and deltas preserved exactly as recorded
        assert [m.timestamp_ms for m in score_relaxed.per_move_analysis] == [100, 250, 600, 750]
        assert score_relaxed.per_move_analysis[1].delta_ms == 150
        assert score_relaxed.per_move_analysis[2].delta_ms == 350
        assert score_relaxed.per_move_analysis[3].delta_ms == 150

        # With tempo = 0.30 (300ms threshold): 350ms exceeds threshold -> pause detected at move 2 (R')
        score_strict = scorer.score(stream, tempo=0.30)
        assert sum(score_strict.pause_breakdown.values()) >= 1
        assert score_strict.per_move_analysis[2].pause_type is not None

        # Timestamps still unaltered
        assert [m.timestamp_ms for m in score_strict.per_move_analysis] == [100, 250, 600, 750]

    def test_score_dynamic_profile_switching_and_matrix_caching(self):
        from roux_engine.ergonomics.models import HandProfile
        scorer = FlowScorer()  # Default 2H profile

        assert "2H" in scorer._matrix_cache
        assert "OH" not in scorer._matrix_cache

        moves = "M U M' U2"
        # 1. Default (2H) evaluation
        score_2h = scorer.score(moves)
        score_2h_explicit = scorer.score(moves, profile="2H")
        assert score_2h.e_stm == score_2h_explicit.e_stm

        # 2. Dynamic OH evaluation
        score_oh = scorer.score(moves, profile="OH")
        # OH requires significantly more effort for M slice moves than 2H
        assert score_oh.e_stm > score_2h.e_stm

        # 3. Transition matrix caching
        assert "OH" in scorer._matrix_cache
        cached_matrix = scorer._matrix_cache["OH"]
        # Subsequent score with profile="OH" reuses cached matrix instance
        score_oh_cached = scorer.score(moves, profile="OH")
        assert scorer._matrix_cache["OH"] is cached_matrix
        assert score_oh_cached.e_stm == score_oh.e_stm

        # 4. Supports HandProfile object
        score_oh_obj = scorer.score(moves, profile=HandProfile(solving_mode="OH"))
        assert score_oh_obj.e_stm == score_oh.e_stm

    def test_backward_compatible_delegates_forward_to_score(self):
        scorer = FlowScorer()

        moves_str = "R U R' U'"
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 250},
        ]

        # score_moves delegate
        s_moves = scorer.score_moves(moves_str, initial_grip=GripState.HOME, tempo=0.25, profile="2H")
        s_direct = scorer.score(moves_str, initial_grip=GripState.HOME, tempo=0.25, profile="2H")
        assert s_moves.raw_stm == s_direct.raw_stm
        assert s_moves.e_stm == s_direct.e_stm
        assert s_moves.turning_ratio == s_direct.turning_ratio
        assert s_moves.stream_flow_index == s_direct.stream_flow_index

        # score_stream delegate
        s_stream = scorer.score_stream(stream, initial_grip=GripState.HOME, tempo=0.30, profile="OH")
        s_stream_direct = scorer.score(stream, initial_grip=GripState.HOME, tempo=0.30, profile="OH")
        assert s_stream.raw_stm == s_stream_direct.raw_stm
        assert s_stream.e_stm == s_stream_direct.e_stm
        assert s_stream.turning_ratio == s_stream_direct.turning_ratio
        assert s_stream.stream_flow_index == s_stream_direct.stream_flow_index







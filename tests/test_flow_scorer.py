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





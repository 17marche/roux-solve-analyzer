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
        # Kinematic efficiency should be > 100% because actual effort is lower than baseline 1.0/move
        assert score.kinematic_efficiency > 100.0
        assert len(score.per_move_analysis) == 4
        assert not any(m.regrip for m in score.per_move_analysis)

    def test_awkward_r_f_forces_higher_e_stm_and_lower_efficiency(self):
        scorer = FlowScorer()
        score_rf = scorer.score_moves("R F")
        assert score_rf.raw_stm == 2
        assert score_rf.regrip_count == 1
        # E-STM = effort(R) + effort(F) + 2.0 * 1
        # RF transition effort is high (~3.0) + regrip (2.0) + baseline R (1.0) => ~6.0
        assert score_rf.e_stm > 4.5
        assert score_rf.kinematic_efficiency < 50.0

        # Compare per-move effort with fluid R U R' U'
        score_fluid = scorer.score_moves("R U R' U'")
        effort_per_move_rf = score_rf.e_stm / score_rf.raw_stm
        effort_per_move_fluid = score_fluid.e_stm / score_fluid.raw_stm
        assert effort_per_move_rf > effort_per_move_fluid * 3.0
        assert score_fluid.kinematic_efficiency > score_rf.kinematic_efficiency

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


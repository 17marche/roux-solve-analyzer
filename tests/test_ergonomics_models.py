"""Unit tests for ergonomics domain models and dataclasses."""

import pytest
from roux_engine.ergonomics.models import (
    GripState,
    HandProfile,
    MoveAnalysis,
    FlowScore,
)


class TestGripStateEnum:
    """Seam 1.1: GripState enum invariants."""

    def test_grip_state_members(self):
        assert GripState.HOME.value == "HOME"
        assert GripState.R_AWAY.value == "R_AWAY"
        assert GripState.R_PRIME_AWAY.value == "R_PRIME_AWAY"
        assert GripState.R2_AWAY.value == "R2_AWAY"
        assert len(GripState) == 4

    def test_grip_state_str_compatibility(self):
        assert GripState.HOME == "HOME"
        assert GripState.R_AWAY == "R_AWAY"


class TestHandProfile:
    """Seam 1.2: HandProfile dataclass defaults and configuration."""

    def test_hand_profile_defaults(self):
        profile = HandProfile()
        assert profile.solving_mode == "2H"
        assert profile.style == "2H"
        assert profile.m_slice_hand == "right"
        assert profile.dominant_hand == "right"

    def test_custom_hand_profile(self):
        profile = HandProfile(solving_mode="OH", m_slice_hand="left", dominant_hand="left")
        assert profile.solving_mode == "OH"
        assert profile.style == "OH"
        assert profile.m_slice_hand == "left"
        assert profile.dominant_hand == "left"



class TestMoveAnalysis:
    """Seam 1.3: MoveAnalysis dataclass."""

    def test_move_analysis_creation(self):
        analysis = MoveAnalysis(
            move="R",
            grip_before=GripState.HOME,
            grip_after=GripState.R_AWAY,
            regrip=False,
            transition_effort=1.0,
        )
        assert analysis.move == "R"
        assert analysis.grip_before == GripState.HOME
        assert analysis.grip_after == GripState.R_AWAY
        assert analysis.regrip is False
        assert analysis.transition_effort == 1.0
        assert analysis.timestamp_ms is None
        assert analysis.delta_ms is None


class TestFlowScore:
    """Seam 1.4: FlowScore dataclass invariants."""

    def test_flow_score_minimal(self):
        score = FlowScore(
            raw_stm=6,
            e_stm=5.8,
            kinematic_efficiency=103.45,
            regrip_count=0,
        )
        assert score.raw_stm == 6
        assert score.e_stm == 5.8
        assert score.kinematic_efficiency == 103.45
        assert score.regrip_count == 0
        assert score.macro_triggers == []
        assert score.turning_ratio is None
        assert score.rhythm_cv is None
        assert score.stream_flow_index is None
        assert score.per_move_analysis == []

    def test_flow_score_with_stream_data(self):
        analysis = [
            MoveAnalysis(
                move="R",
                grip_before=GripState.HOME,
                grip_after=GripState.R_AWAY,
                regrip=False,
                transition_effort=1.0,
                timestamp_ms=100,
                delta_ms=100,
            )
        ]
        score = FlowScore(
            raw_stm=1,
            e_stm=1.0,
            kinematic_efficiency=100.0,
            regrip_count=0,
            macro_triggers=[],
            turning_ratio=0.85,
            rhythm_cv=0.12,
            stream_flow_index=75.89,
            per_move_analysis=analysis,
        )
        assert score.turning_ratio == 0.85
        assert score.rhythm_cv == 0.12
        assert score.stream_flow_index == 75.89
        assert len(score.per_move_analysis) == 1

"""Biomechanical Flow Scorer evaluating Effective STM (E-STM) and Kinematic Flow Efficiency."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..core.parser import MoveParser, MoveEvent
from .models import GripState, MoveAnalysis, FlowScore, HandProfile
from .grip_tracker import GripTracker, GripTrackingResult
from .transition_matrix import TransitionMatrix


class FlowScorer:
    """Evaluates the biomechanical flow and Effective STM of move sequences.
    
    Formulae:
      E_STM = Sum(Transition_Effort) + (2.0 * Regrip_Count)
      Kinematic_Flow_Efficiency = (Raw_STM / E_STM) * 100%
    """

    BASELINE_LATENCY_SEC: float = 0.10

    def __init__(
        self,
        grip_tracker: Optional[GripTracker] = None,
        transition_matrix: Optional[TransitionMatrix] = None,
        profile: Optional[HandProfile] = None,
        custom_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.grip_tracker = grip_tracker or GripTracker()
        self.profile = profile or (transition_matrix.profile if transition_matrix else HandProfile())
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        elif custom_matrix is not None:
            self.transition_matrix = TransitionMatrix(
                transitions=custom_matrix,
                profile=self.profile,
            )
        else:
            self.transition_matrix = TransitionMatrix.load(profile=self.profile)

    @property
    def _transitions(self) -> Dict[str, float]:
        """Backwards-compatible access to underlying transitions mapping."""
        return self.transition_matrix.transitions

    def get_transition_effort(self, prev_move: Optional[str], move: str) -> float:
        """Returns the dimensionless relative effort multiplier for a move or bigram transition."""
        return self.transition_matrix.get_effort(prev_move, move)

    def score_moves(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent]]],
        initial_grip: GripState = GripState.HOME,
    ) -> FlowScore:
        """Computes Effective STM (E-STM), kinematic efficiency, and regrip breakdown."""
        if isinstance(moves, str):
            events = MoveParser.parse_string(moves)
        else:
            events = [
                m if isinstance(m, MoveEvent) else MoveEvent(move=MoveParser.normalize_token(str(m)), raw_token=str(m))
                for m in moves
            ]
        # Filter out empty tokens
        events = [e for e in events if e.move]

        # Raw STM: turns count as 1, whole-cube rotations (x, y, z) count as 0
        turning_events = [e for e in events if not e.move.startswith(("x", "y", "z"))]
        raw_stm = len(turning_events)

        if raw_stm == 0:
            return FlowScore(
                raw_stm=0,
                e_stm=0.0,
                kinematic_efficiency=0.0,
                regrip_count=0,
                per_move_analysis=[],
            )

        # Track grips across all events
        tracking: GripTrackingResult = self.grip_tracker.track(events, initial_grip=initial_grip)

        per_move_analysis: List[MoveAnalysis] = []
        total_transition_effort = 0.0
        prev_move: Optional[str] = None

        # Correlate tracking steps with moves
        for step, ev in zip(tracking.steps, events):
            if ev.move.startswith(("x", "y", "z")):
                continue

            effort = self.get_transition_effort(prev_move, ev.move)
            total_transition_effort += effort
            prev_move = ev.move

            per_move_analysis.append(
                MoveAnalysis(
                    move=ev.move,
                    grip_before=step.grip_before,
                    grip_after=step.grip_after,
                    regrip=step.regrip,
                    transition_effort=round(effort, 4),
                    timestamp_ms=ev.timestamp_ms,
                    delta_ms=ev.delta_ms,
                )
            )

        regrip_count = tracking.regrip_count
        e_stm = total_transition_effort + (2.0 * regrip_count)
        kinematic_efficiency = (raw_stm / e_stm * 100.0) if e_stm > 0 else 0.0

        return FlowScore(
            raw_stm=raw_stm,
            e_stm=round(e_stm, 4),
            kinematic_efficiency=round(kinematic_efficiency, 2),
            regrip_count=regrip_count,
            per_move_analysis=per_move_analysis,
        )

    def score(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent]]],
        initial_grip: GripState = GripState.HOME,
    ) -> FlowScore:
        """Alias for score_moves to support unified interface."""
        return self.score_moves(moves, initial_grip=initial_grip)

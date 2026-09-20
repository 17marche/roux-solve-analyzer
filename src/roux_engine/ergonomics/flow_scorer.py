"""Biomechanical Flow Scorer evaluating Effective STM (E-STM) and Kinematic Flow Efficiency."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Any

from ..core.parser import MoveParser, MoveEvent
from .models import GripState, MoveAnalysis, FlowScore, HandProfile
from .grip_tracker import GripTracker, GripTrackingResult
from .transition_matrix import TransitionMatrix
from .pause_detector import StreamPauseDetector, StreamMetrics, PauseEvent
from .macro_triggers import match_macro_triggers


class FlowScorer:
    """Evaluates the biomechanical flow, Effective STM, and smart-cube stream rhythm of move sequences.
    
    Formulae:
      E_STM = Sum(Transition_Effort) + (2.0 * Regrip_Count)
      Kinematic_Flow_Efficiency = (Raw_STM / E_STM) * 100%
      Stream_Flow_Index = 100 * Turning_Ratio * (1 / (1 + Rhythm_CV))
    """

    BASELINE_LATENCY_SEC: float = 0.10

    def __init__(
        self,
        grip_tracker: Optional[GripTracker] = None,
        transition_matrix: Optional[TransitionMatrix] = None,
        profile: Optional[HandProfile] = None,
        custom_matrix: Optional[Dict[str, float]] = None,
        pause_detector: Optional[StreamPauseDetector] = None,
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
        self.pause_detector = pause_detector or StreamPauseDetector(
            grip_tracker=self.grip_tracker,
            transition_matrix=self.transition_matrix,
        )

    @property
    def _transitions(self) -> Dict[str, float]:
        """Backwards-compatible access to underlying transitions mapping."""
        return self.transition_matrix.transitions

    def get_transition_effort(self, prev_move: Optional[str], move: str) -> float:
        """Returns the dimensionless relative effort multiplier for a move or bigram transition."""
        return self.transition_matrix.get_effort(prev_move, move)

    def _evaluate_events(
        self,
        events: List[MoveEvent],
        initial_grip: GripState = GripState.HOME,
        metrics: Optional[StreamMetrics] = None,
    ) -> FlowScore:
        """Shared evaluation engine for static move sequences and timestamped streams."""
        turning_events = [e for e in events if not e.move.startswith(("x", "y", "z"))]
        raw_stm = len(turning_events)

        if raw_stm == 0:
            return FlowScore(
                raw_stm=0,
                e_stm=0.0,
                kinematic_efficiency=0.0,
                regrip_count=0,
                turning_ratio=0.0 if metrics is not None else None,
                rhythm_cv=0.0 if metrics is not None else None,
                stream_flow_index=0.0 if metrics is not None else None,
                per_move_analysis=[],
                pause_breakdown={} if metrics is not None else {},
            )

        # Track grips across all events
        tracking: GripTrackingResult = self.grip_tracker.track(events, initial_grip=initial_grip)
        pause_map: Dict[int, PauseEvent] = {p.move_index: p for p in metrics.pauses} if metrics else {}

        macro_matches = match_macro_triggers(events)
        macro_indices = set()
        for mm in macro_matches:
            for idx in range(mm.start_index, mm.end_index):
                macro_indices.add(idx)

        per_move_analysis: List[MoveAnalysis] = []
        total_transition_effort = 0.0
        prev_move: Optional[str] = None

        for i, (step, ev) in enumerate(zip(tracking.steps, events)):
            if ev.move.startswith(("x", "y", "z")):
                continue

            effort = self.get_transition_effort(prev_move, ev.move)
            total_transition_effort += effort
            prev_move = ev.move

            pause_event = pause_map.get(i)
            pause_type_enum = pause_event.pause_type if pause_event else None
            is_regrip = False if i in macro_indices else step.regrip

            per_move_analysis.append(
                MoveAnalysis(
                    move=ev.move,
                    grip_before=step.grip_before,
                    grip_after=step.grip_after,
                    regrip=is_regrip,
                    transition_effort=round(effort, 4),
                    timestamp_ms=ev.timestamp_ms,
                    delta_ms=ev.delta_ms,
                    pause_type=pause_type_enum,
                )
            )

        regrip_count = sum(1 for m in per_move_analysis if m.regrip)
        e_stm = total_transition_effort + (2.0 * regrip_count)
        kinematic_efficiency = (raw_stm / e_stm * 100.0) if e_stm > 0 else 0.0

        return FlowScore(
            raw_stm=raw_stm,
            e_stm=round(e_stm, 4),
            kinematic_efficiency=round(kinematic_efficiency, 2),
            regrip_count=regrip_count,
            macro_triggers=[m.trigger.name for m in macro_matches],
            turning_ratio=metrics.turning_ratio if metrics else None,
            rhythm_cv=metrics.rhythm_cv if metrics else None,
            stream_flow_index=metrics.stream_flow_index if metrics else None,
            per_move_analysis=per_move_analysis,
            pause_breakdown=metrics.pause_breakdown if metrics else {},
        )

    def score_moves(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent]]],
        initial_grip: GripState = GripState.HOME,
    ) -> FlowScore:
        """Computes Effective STM (E-STM), kinematic efficiency, and regrip breakdown for static moves."""
        if isinstance(moves, str):
            events = MoveParser.parse_string(moves)
        else:
            events = [
                m if isinstance(m, MoveEvent) else MoveEvent(move=MoveParser.normalize_token(str(m)), raw_token=str(m))
                for m in moves
            ]
        events = [e for e in events if e.move]
        return self._evaluate_events(events, initial_grip=initial_grip, metrics=None)

    def score_stream(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
        pause_detector: Optional[StreamPauseDetector] = None,
    ) -> FlowScore:
        """Computes pace-invariant turning ratio, rhythm CV, stream flow index, and pause classifications."""
        detector = pause_detector or self.pause_detector
        events = StreamPauseDetector.parse_stream(stream)
        events = [e for e in events if e.move]
        metrics = detector.calculate_metrics(events, initial_grip=initial_grip)
        return self._evaluate_events(events, initial_grip=initial_grip, metrics=metrics)

    def _is_stream(self, moves: Any) -> bool:
        """Determines if the provided input represents a timestamped smart-cube stream."""
        if isinstance(moves, Sequence) and len(moves) > 0:
            first = moves[0]
            if isinstance(first, dict) and any(k in first for k in ("timestamp_ms", "t_ms", "t")):
                return True
            if isinstance(first, MoveEvent) and first.timestamp_ms is not None:
                return True

        if isinstance(moves, str):
            text = moves.strip()
            if text.startswith("["):
                try:
                    data = json.loads(text)
                    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                        return any(k in data[0] for k in ("timestamp_ms", "t_ms", "t"))
                except Exception:
                    return False

        return False

    def score(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
    ) -> FlowScore:
        """Unified score interface: automatically dispatches to score_stream or score_moves."""
        if self._is_stream(moves):
            return self.score_stream(moves, initial_grip=initial_grip)
        return self.score_moves(moves, initial_grip=initial_grip)  # type: ignore[arg-type]



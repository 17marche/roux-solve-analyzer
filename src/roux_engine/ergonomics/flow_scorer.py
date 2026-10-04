"""Biomechanical Flow Scorer evaluating Effective STM (E-STM) and Kinematic Flow Efficiency."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union, Any, Tuple

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
        self._matrix_cache: Dict[str, TransitionMatrix] = {
            self._matrix_cache_key(self.profile): self.transition_matrix
        }

    @staticmethod
    def _matrix_cache_key(profile: HandProfile) -> str:
        """Derives a cache key incorporating both solving mode and M-slice handedness."""
        mode = profile.solving_mode.upper()
        if profile.m_slice_hand.lower() == "right":
            return mode
        return f"{mode}:{profile.m_slice_hand.lower()}"

    @property
    def _transitions(self) -> Dict[str, float]:
        """Backwards-compatible access to underlying transitions mapping."""
        return self.transition_matrix.transitions

    def _get_transition_matrix(self, profile: Optional[Union[str, HandProfile]] = None) -> TransitionMatrix:
        """Retrieves or lazily loads and caches the transition matrix for the requested profile."""
        if profile is None:
            return self.transition_matrix
        if isinstance(profile, str):
            mode = profile.strip().upper()
            if mode not in ("2H", "OH"):
                raise ValueError(f"Unsupported profile: '{profile}'. Expected '2H' or 'OH'.")
            target_profile = HandProfile(solving_mode=mode)
        elif isinstance(profile, HandProfile):
            target_profile = profile
        else:
            raise ValueError(f"Invalid profile type: {type(profile)}")

        if target_profile == self.profile and self.transition_matrix is not None:
            return self.transition_matrix

        key = self._matrix_cache_key(target_profile)
        if key not in self._matrix_cache:
            self._matrix_cache[key] = TransitionMatrix.load(profile=target_profile)
        return self._matrix_cache[key]

    def get_transition_effort(
        self,
        prev_move: Optional[str],
        move: str,
        matrix: Optional[TransitionMatrix] = None,
    ) -> float:
        """Returns the dimensionless relative effort multiplier for a move or bigram transition."""
        mat = matrix or self.transition_matrix
        return mat.get_effort(prev_move, move)

    def _evaluate_events(
        self,
        events: List[MoveEvent],
        initial_grip: GripState = GripState.HOME,
        metrics: Optional[StreamMetrics] = None,
        matrix: Optional[TransitionMatrix] = None,
    ) -> FlowScore:
        """Shared evaluation engine for static move sequences and timestamped streams."""
        active_matrix = matrix or self.transition_matrix
        turning_events = [e for e in events if not e.move.startswith(("x", "y", "z"))]
        raw_stm = len(turning_events)

        if raw_stm == 0:
            return FlowScore(
                raw_stm=0,
                e_stm=0.0,
                kinematic_efficiency=0.0,
                regrip_count=0,
                macro_triggers=[],
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

            effort = self.get_transition_effort(prev_move, ev.move, matrix=active_matrix)
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
        kinematic_efficiency = min(100.0, (raw_stm / e_stm * 100.0)) if e_stm > 0 else 0.0

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

    def _parse_input(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
    ) -> Tuple[List[MoveEvent], bool]:
        """Parses moves, sequences, JSON strings, or stream lists into normalized MoveEvents with stream tag.

        Raises:
            ValueError: If any cube token is invalid or stream format is malformed.
        """
        if moves is None:
            return [], False

        if isinstance(moves, str):
            text = moves.strip()
            if not text:
                return [], False

            if text.startswith(("[", "{")):
                try:
                    data = json.loads(text)
                except Exception:
                    data = None

                if data is not None:
                    if isinstance(data, list):
                        if len(data) == 0:
                            return [], True
                        if isinstance(data[0], dict):
                            events = MoveParser.parse_smart_cube_stream(data, strict=True)
                            return events, True
                        else:
                            events = []
                            for item in data:
                                tok = str(item)
                                norm = MoveParser.normalize_token(tok)
                                if not MoveParser.is_valid_move_token(norm):
                                    raise ValueError(f"Invalid cube move token: '{tok}'")
                                events.append(MoveEvent(move=norm, raw_token=tok))
                            return events, False
                    elif isinstance(data, dict):
                        events = MoveParser.parse_smart_cube_stream([data], strict=True)
                        return events, True

            # Standard move sequence string
            events = MoveParser.parse_string(text, strict=True)
            return events, False

        if isinstance(moves, Sequence):
            if len(moves) == 0:
                return [], False

            first = moves[0]
            if isinstance(first, dict):
                events = MoveParser.parse_smart_cube_stream(list(moves), strict=True)  # type: ignore[arg-type]
                return events, True
            elif isinstance(first, MoveEvent):
                events = []
                has_timestamps = False
                for m in moves:
                    if not isinstance(m, MoveEvent):
                        raise ValueError(f"Mixed types in move sequence: expected MoveEvent, got {type(m)}")
                    norm = MoveParser.normalize_token(m.move)
                    if not MoveParser.is_valid_move_token(norm):
                        raise ValueError(f"Invalid cube move token: '{m.move}'")
                    if m.timestamp_ms is not None:
                        has_timestamps = True
                    events.append(
                        MoveEvent(
                            move=norm,
                            raw_token=m.raw_token or m.move,
                            timestamp_ms=m.timestamp_ms,
                            delta_ms=m.delta_ms,
                        )
                    )
                return events, has_timestamps
            else:
                events = []
                for item in moves:
                    tok = str(item)
                    norm = MoveParser.normalize_token(tok)
                    if not MoveParser.is_valid_move_token(norm):
                        raise ValueError(f"Invalid cube move token: '{tok}'")
                    events.append(MoveEvent(move=norm, raw_token=tok))
                return events, False

        raise TypeError(f"Unsupported moves input type: {type(moves)}")

    def _is_stream(self, moves: Any) -> bool:
        """Determines if the provided input represents a timestamped smart-cube stream."""
        try:
            _, is_stream = self._parse_input(moves)
            return is_stream
        except Exception:
            return False

    def score(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        tempo: Optional[float] = None,
        profile: Optional[Union[str, HandProfile]] = None,
        initial_grip: GripState = GripState.HOME,
        pause_detector: Optional[StreamPauseDetector] = None,
    ) -> FlowScore:
        """Evaluates biomechanical flow, Effective STM, and smart-cube rhythm.

        Unified deep public entry point accepting:
          - Space-delimited move strings (e.g. "R U R' U'")
          - Move token sequences (e.g. ["R", "U", "R'", "U'"])
          - Raw JSON strings (token lists or stream dicts)
          - Smart-cube stream lists (e.g. [{"move": "R", "timestamp_ms": 100}, ...])

        Args:
            moves: Move sequence or stream payload.
            tempo: Personal seconds-per-move reference tempo (e.g. 0.25 for 4 TPS).
                   When passed with static moves, synthesizes uniform timestamps.
                   When passed with pre-timestamped streams, calibrates pause cutoff.
            profile: Optional solving style profile ("2H" or "OH" or HandProfile).
            initial_grip: Starting right-hand grip state.
            pause_detector: Optional explicit StreamPauseDetector override.

        Returns:
            FlowScore containing raw STM, E-STM, efficiency, regrip count, and rhythm metrics.

        Raises:
            ValueError: If invalid cube tokens, negative/zero tempo, or unsupported profile is given.
        """
        if tempo is not None and tempo <= 0:
            raise ValueError(f"Tempo must be a positive float, got {tempo}")

        matrix = self._get_transition_matrix(profile)
        events, is_stream = self._parse_input(moves)

        turning_events = [e for e in events if not e.move.startswith(("x", "y", "z"))]
        if len(turning_events) == 0:
            has_stream_metrics = is_stream or (tempo is not None)
            return FlowScore(
                raw_stm=0,
                e_stm=0.0,
                kinematic_efficiency=0.0,
                regrip_count=0,
                macro_triggers=[],
                turning_ratio=0.0 if has_stream_metrics else None,
                rhythm_cv=0.0 if has_stream_metrics else None,
                stream_flow_index=0.0 if has_stream_metrics else None,
                per_move_analysis=[],
                pause_breakdown={},
            )

        # Handle reference tempo and stream metrics
        if is_stream:
            # Pre-timestamped stream: tempo sets pause threshold cutoff without altering timestamps
            if tempo is not None:
                detector = StreamPauseDetector(
                    grip_tracker=self.grip_tracker,
                    transition_matrix=matrix,
                    pause_threshold_ms=tempo * 1000.0,
                    pace_adaptive=False,
                )
            elif pause_detector is not None:
                detector = pause_detector
            else:
                detector = StreamPauseDetector(
                    grip_tracker=self.grip_tracker,
                    transition_matrix=matrix,
                )
            metrics = detector.calculate_metrics(events, initial_grip=initial_grip)
            return self._evaluate_events(events, initial_grip=initial_grip, metrics=metrics, matrix=matrix)

        if tempo is not None:
            # Static moves with reference tempo: synthesize uniform timestamps
            step_ms = int(tempo * 1000.0)
            for i, ev in enumerate(events):
                ev.timestamp_ms = int(i * step_ms)
                ev.delta_ms = step_ms if i > 0 else 0

            detector = StreamPauseDetector(
                grip_tracker=self.grip_tracker,
                transition_matrix=matrix,
                pause_threshold_ms=float(step_ms),
                pace_adaptive=False,
            )
            metrics = detector.calculate_metrics(events, initial_grip=initial_grip)
            return self._evaluate_events(events, initial_grip=initial_grip, metrics=metrics, matrix=matrix)

        # Static moves without tempo
        return self._evaluate_events(events, initial_grip=initial_grip, metrics=None, matrix=matrix)

    def score_moves(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent]]],
        initial_grip: GripState = GripState.HOME,
        tempo: Optional[float] = None,
        profile: Optional[Union[str, HandProfile]] = None,
    ) -> FlowScore:
        """Backward-compatible delegate forwarding to score()."""
        return self.score(moves, tempo=tempo, profile=profile, initial_grip=initial_grip)

    def score_stream(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
        pause_detector: Optional[StreamPauseDetector] = None,
        tempo: Optional[float] = None,
        profile: Optional[Union[str, HandProfile]] = None,
    ) -> FlowScore:
        """Backward-compatible delegate forwarding to score()."""
        return self.score(
            stream,
            tempo=tempo,
            profile=profile,
            initial_grip=initial_grip,
            pause_detector=pause_detector,
        )



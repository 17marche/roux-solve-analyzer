"""Smart-cube stream pause detector, pace-invariant flow scoring, and hesitation classifier."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union, Any, Tuple
import numpy as np

from ..core.parser import MoveParser, MoveEvent
from .models import GripState, PauseType, HandProfile
from .grip_tracker import GripTracker, GripTrackingResult
from .transition_matrix import TransitionMatrix


@dataclass(frozen=True)
class PauseEvent:
    """Represents a classified execution pause or hesitation in a smart-cube stream."""
    move_index: int                       # 0-based index of the move in the sequence
    move: str                             # Normalized move token (e.g. "R", "U'")
    prev_move: Optional[str]              # Preceding move token
    delta_ms: int                         # Inter-move elapsed interval (ms)
    pause_type: PauseType                 # PHYSICAL_REGRIP, COGNITIVE_HESITATION, or EXECUTION_LOCKUP
    expected_ms: float                    # Expected baseline transition time (ms)
    ratio: float                          # delta_ms / expected_ms
    timestamp_ms: Optional[int] = None    # Move completion timestamp (ms)


@dataclass(frozen=True)
class StreamMetrics:
    """Computed scale-invariant stream metrics and pause classification breakdown."""
    turning_ratio: float                  # Active turning time / total time (0.0 to 1.0)
    rhythm_cv: float                      # Rhythm consistency coefficient of variation (sigma / mu)
    stream_flow_index: float              # 100 * turning_ratio * (1 / (1 + rhythm_cv))
    pauses: List[PauseEvent]              # Detected pause events
    total_time_ms: int                    # Total elapsed duration (ms)
    turning_time_ms: float                # Active turning time (ms)
    pause_time_ms: float                  # Dead pause time (ms)
    pause_breakdown: Dict[str, int]       # Count per PauseType


class StreamPauseDetector:
    """Analyzes Bluetooth smart-cube move streams to classify pauses and score flow continuity.

    Cross-references kinematic regrip predictions with inter-move transition intervals:
      - PHYSICAL_REGRIP: pause occurs where the kinematic model predicts a forced regrip.
      - COGNITIVE_HESITATION: prolonged pause without regrip (lookahead or recognition lag).
      - EXECUTION_LOCKUP: brief execution snag or stumble during continuous turning without regrip.

    Provides pace-invariant Turning Ratio (TR) and Rhythm Consistency (CV) calculations.
    """

    DEFAULT_PAUSE_THRESHOLD_MS: float = 250.0
    DEFAULT_COGNITIVE_THRESHOLD_MS: float = 500.0
    DEFAULT_THRESHOLD_FACTOR: float = 2.0
    DEFAULT_COGNITIVE_FACTOR: float = 4.0

    def __init__(
        self,
        grip_tracker: Optional[GripTracker] = None,
        transition_matrix: Optional[TransitionMatrix] = None,
        pause_threshold_ms: Optional[float] = None,
        cognitive_threshold_ms: Optional[float] = None,
        pace_adaptive: Optional[bool] = None,
        threshold_factor: float = DEFAULT_THRESHOLD_FACTOR,
        cognitive_factor: float = DEFAULT_COGNITIVE_FACTOR,
    ) -> None:
        self.grip_tracker = grip_tracker or GripTracker()
        self.transition_matrix = transition_matrix or TransitionMatrix.load()
        
        # When pause_threshold_ms is explicitly passed and pace_adaptive is not, disable adaptive mode
        if pace_adaptive is None:
            self.pace_adaptive = (pause_threshold_ms is None)
        else:
            self.pace_adaptive = pace_adaptive

        self.pause_threshold_ms = (
            pause_threshold_ms if pause_threshold_ms is not None else self.DEFAULT_PAUSE_THRESHOLD_MS
        )
        self.cognitive_threshold_ms = (
            cognitive_threshold_ms if cognitive_threshold_ms is not None else self.DEFAULT_COGNITIVE_THRESHOLD_MS
        )
        self.threshold_factor = threshold_factor
        self.cognitive_factor = cognitive_factor

    @classmethod
    def parse_stream(
        cls,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
    ) -> List[MoveEvent]:
        """Parses various smart-cube stream formats into a normalized list of MoveEvents."""
        events: List[MoveEvent] = []

        if isinstance(stream, str):
            text = stream.strip()
            if text.startswith("["):
                try:
                    raw_data = json.loads(text)
                    if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], dict):
                        return MoveParser.parse_smart_cube_stream(raw_data)
                    else:
                        events = MoveParser.parse_string(text)
                except Exception:
                    events = MoveParser.parse_string(text)
            else:
                events = MoveParser.parse_string(text)
        elif len(stream) > 0 and isinstance(stream[0], dict):
            return MoveParser.parse_smart_cube_stream(list(stream))  # type: ignore[arg-type]
        else:
            for item in stream:
                if isinstance(item, MoveEvent):
                    events.append(
                        MoveEvent(
                            move=item.move,
                            raw_token=item.raw_token,
                            timestamp_ms=item.timestamp_ms,
                            delta_ms=item.delta_ms,
                        )
                    )
                else:
                    norm = MoveParser.normalize_token(str(item))
                    events.append(MoveEvent(move=norm, raw_token=str(item)))

        # Ensure delta_ms is calculated for consecutive events with timestamps
        prev_t: Optional[int] = None
        for ev in events:
            if ev.timestamp_ms is not None:
                if ev.delta_ms is None and prev_t is not None:
                    ev.delta_ms = max(0, ev.timestamp_ms - prev_t)
                prev_t = ev.timestamp_ms

        return events

    def calculate_metrics(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
    ) -> StreamMetrics:
        """Computes pace-invariant turning ratio, rhythm CV, and classified pause events."""
        events = self.parse_stream(stream)
        # Filter out empty tokens
        events = [e for e in events if e.move]

        intervals: List[Tuple[int, int]] = []
        for i, ev in enumerate(events):
            if i > 0 and ev.delta_ms is not None:
                intervals.append((i, ev.delta_ms))

        if not intervals:
            is_empty = (len(events) == 0)
            return StreamMetrics(
                turning_ratio=0.0 if is_empty else 1.0,
                rhythm_cv=0.0,
                stream_flow_index=0.0 if is_empty else 100.0,
                pauses=[],
                total_time_ms=0,
                turning_time_ms=0.0,
                pause_time_ms=0.0,
                pause_breakdown={
                    PauseType.COGNITIVE_HESITATION.value: 0,
                    PauseType.PHYSICAL_REGRIP.value: 0,
                    PauseType.EXECUTION_LOCKUP.value: 0,
                },
            )

        deltas = [d for _, d in intervals]
        total_time_ms = sum(deltas)
        mean_delta = float(np.mean(deltas))
        std_delta = float(np.std(deltas))
        rhythm_cv = round(std_delta / mean_delta, 4) if mean_delta > 0 else 0.0
        if len(deltas) <= 3:
            median_delta = float(min(deltas))
        else:
            median_delta = float(np.median(deltas))


        # Compute transition efforts for all intervals
        efforts: List[float] = []
        for i, _ in intervals:
            prev_move = events[i - 1].move
            curr_move = events[i].move
            if curr_move.startswith(("x", "y", "z")):
                efforts.append(1.0)
            else:
                efforts.append(self.transition_matrix.get_effort(prev_move, curr_move))

        median_effort = float(np.median(efforts)) if efforts else 1.0

        if median_effort <= 0:
            median_effort = 1.0

        if self.pace_adaptive:
            base_latency = median_delta / median_effort
        else:
            base_latency = self.pause_threshold_ms / self.threshold_factor

        # Kinematic grip tracking to identify physical regrips
        tracking: GripTrackingResult = self.grip_tracker.track(events, initial_grip=initial_grip)

        pauses: List[PauseEvent] = []
        turning_time_ms = 0.0
        pause_time_ms = 0.0

        for (i, delta_ms), effort in zip(intervals, efforts):
            prev_move = events[i - 1].move
            curr_move = events[i].move
            expected_ms = max(1.0, base_latency * effort)

            if self.pace_adaptive:
                thresh = self.threshold_factor * expected_ms
                cog_thresh = self.cognitive_factor * expected_ms
            else:
                thresh = self.pause_threshold_ms
                cog_thresh = self.cognitive_threshold_ms

            if delta_ms > thresh:
                ratio = round(delta_ms / expected_ms, 2)
                is_regrip = tracking.steps[i].regrip if i < len(tracking.steps) else False

                if delta_ms >= cog_thresh:
                    pause_type = PauseType.COGNITIVE_HESITATION
                elif is_regrip:
                    pause_type = PauseType.PHYSICAL_REGRIP
                else:
                    pause_type = PauseType.EXECUTION_LOCKUP

                pauses.append(
                    PauseEvent(
                        move_index=i,
                        move=curr_move,
                        prev_move=prev_move,
                        delta_ms=delta_ms,
                        pause_type=pause_type,
                        expected_ms=round(expected_ms, 2),
                        ratio=ratio,
                        timestamp_ms=events[i].timestamp_ms,
                    )
                )

                turn_time = min(float(delta_ms), expected_ms)
                pause_time = float(delta_ms) - turn_time
            else:
                turn_time = float(delta_ms)
                pause_time = 0.0

            turning_time_ms += turn_time
            pause_time_ms += pause_time

        turning_ratio = (

            round(turning_time_ms / total_time_ms, 4) if total_time_ms > 0 else 1.0
        )
        turning_ratio = min(1.0, max(0.0, turning_ratio))

        stream_flow_index = round(
            100.0 * turning_ratio * (1.0 / (1.0 + rhythm_cv)), 2
        )

        pause_breakdown = {
            PauseType.COGNITIVE_HESITATION.value: sum(
                1 for p in pauses if p.pause_type == PauseType.COGNITIVE_HESITATION
            ),
            PauseType.PHYSICAL_REGRIP.value: sum(
                1 for p in pauses if p.pause_type == PauseType.PHYSICAL_REGRIP
            ),
            PauseType.EXECUTION_LOCKUP.value: sum(
                1 for p in pauses if p.pause_type == PauseType.EXECUTION_LOCKUP
            ),
        }

        return StreamMetrics(
            turning_ratio=turning_ratio,
            rhythm_cv=rhythm_cv,
            stream_flow_index=stream_flow_index,
            pauses=pauses,
            total_time_ms=total_time_ms,
            turning_time_ms=round(turning_time_ms, 2),
            pause_time_ms=round(pause_time_ms, 2),
            pause_breakdown=pause_breakdown,
        )

    def detect_pauses(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
    ) -> List[PauseEvent]:
        """Classifies and returns all pause events detected in the move stream."""
        metrics = self.calculate_metrics(stream, initial_grip=initial_grip)
        return metrics.pauses

    def get_pause_breakdown(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
    ) -> Dict[str, int]:
        """Returns the count of pauses grouped by classified PauseType."""
        metrics = self.calculate_metrics(stream, initial_grip=initial_grip)
        return metrics.pause_breakdown

    def analyze(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
        initial_grip: GripState = GripState.HOME,
    ) -> StreamMetrics:
        """Alias for calculate_metrics to provide unified analysis interface."""
        return self.calculate_metrics(stream, initial_grip=initial_grip)

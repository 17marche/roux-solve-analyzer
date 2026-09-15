"""Biomechanical Flow Scorer evaluating Effective STM (E-STM) and Kinematic Flow Efficiency."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ..core.parser import MoveParser, MoveEvent
from .models import GripState, MoveAnalysis, FlowScore
from .grip_tracker import GripTracker, GripTrackingResult


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
        custom_matrix: Optional[Dict[str, float]] = None,
    ) -> None:
        self.grip_tracker = grip_tracker or GripTracker()
        self._transitions: Dict[str, float] = {}
        if custom_matrix is not None:
            self._transitions = dict(custom_matrix)
        else:
            self._load_default_transitions()

    def _load_default_transitions(self) -> None:
        """Loads calibrated transition data from references or packaged data if present."""
        search_paths = [
            # Issue 02 target location
            Path(__file__).resolve().parents[1] / "data" / "transitions" / "matrix_2h.json",
            # Reference dataset from onionhoney/roux-trainers
            Path(__file__).resolve().parents[3] / "references" / "roux_trainers" / "two_gram_v1.json",
        ]
        for path in search_paths:
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    for k, v in raw.items():
                        # Normalize into dimensionless effort multiplier
                        val = float(v)
                        # If values are already normalized near 1.0 vs raw latencies in seconds (~0.05 - 0.35s)
                        multiplier = val if val > 0.5 and val < 5.0 and "matrix_2h" in str(path) else (val / self.BASELINE_LATENCY_SEC)
                        self._transitions[k] = multiplier
                    return
                except Exception:
                    continue

        # Fallback calibrated defaults if no external file is found
        self._transitions = {
            "RU": 0.50, "UR'": 0.50, "R'U'": 0.49, "U'R": 1.08,
            "RF": 3.03, "R2B": 1.30, "R'F": 0.50,
            "rU": 0.50, "Ur'": 0.50, "r'U'": 0.50, "U'r": 0.88,
            "MU": 0.60, "UM'": 0.50, "M'U'": 0.50, "U'M": 0.80,
            "R": 0.80, "R'": 0.60, "R2": 1.00,
            "U": 0.70, "U'": 0.80, "U2": 1.00,
            "F": 1.00, "F'": 0.80, "F2": 1.50,
            "B": 1.30, "B'": 1.50, "B2": 1.80,
        }

    def get_transition_effort(self, prev_move: Optional[str], move: str) -> float:
        """Returns the dimensionless relative effort multiplier for a move or bigram transition."""
        norm_curr = MoveParser.normalize_token(move)
        if norm_curr.startswith(("x", "y", "z")):
            return 0.0

        if prev_move is None:
            # Single move baseline lookup
            if norm_curr in self._transitions:
                return self._transitions[norm_curr]
            return 1.0

        norm_prev = MoveParser.normalize_token(prev_move)
        if norm_prev.startswith(("x", "y", "z")):
            return self.get_transition_effort(None, norm_curr)

        pair_key = f"{norm_prev}{norm_curr}"
        if pair_key in self._transitions:
            return self._transitions[pair_key]

        # Heuristic fallback if bigram not explicitly in matrix
        return 1.0

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
                    grip_before=step.grip_during,
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

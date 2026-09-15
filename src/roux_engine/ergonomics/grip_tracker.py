"""Kinematic Grip Tracker modeling 4-state right-hand ergonomics and regrip minimization."""

from __future__ import annotations
from typing import Dict, Optional, Tuple, Set, Sequence, Union, List
from dataclasses import dataclass

from ..core.parser import MoveParser, MoveEvent
from .models import GripState


@dataclass(frozen=True)
class GripStep:
    """Represents a single step in a tracked grip path."""
    move: str
    grip_before: GripState     # Grip state before any regrip
    grip_during: GripState     # Grip state in which the move is executed
    grip_after: GripState      # Grip state resulting from the move
    regrip: bool               # True if grip_before != grip_during


@dataclass(frozen=True)
class GripTrackingResult:
    """Full tracking result of a sequence of moves through the kinematic grip model."""
    regrip_count: int
    steps: List[GripStep]
    initial_grip: GripState
    final_grip: GripState


class GripTracker:
    """4-state right-hand kinematic model tracking wrist orientation and forced regrips.
    
    States:
      - HOME: Neutral home grip (thumb on Front/FR, fingers on Back/BR).
      - R_AWAY: +90° clockwise rotation (thumb on U, fingers on D).
      - R_PRIME_AWAY: -90° counter-clockwise rotation (thumb on D, fingers on U).
      - R2_AWAY: 180° rotation (thumb on Back, fingers on Front).
    """

    # Move transition mapping for turns that physically rotate the right hand
    # {move: {from_grip: to_grip_or_None}}
    _R_TRANSITIONS: Dict[str, Dict[GripState, Optional[GripState]]] = {
        "R": {
            GripState.HOME: GripState.R_AWAY,
            GripState.R_AWAY: GripState.R2_AWAY,
            GripState.R_PRIME_AWAY: GripState.HOME,
            GripState.R2_AWAY: None,  # DEAD_END: anatomical limit
        },
        "R'": {
            GripState.HOME: GripState.R_PRIME_AWAY,
            GripState.R_AWAY: GripState.HOME,
            GripState.R_PRIME_AWAY: None,  # DEAD_END: anatomical limit
            GripState.R2_AWAY: GripState.R_AWAY,
        },
        "R2": {
            GripState.HOME: GripState.R2_AWAY,
            GripState.R_AWAY: GripState.R_PRIME_AWAY,
            GripState.R_PRIME_AWAY: GripState.R_AWAY,
            GripState.R2_AWAY: GripState.HOME,
        },
        "r": {
            GripState.HOME: GripState.R_AWAY,
            GripState.R_AWAY: GripState.R2_AWAY,
            GripState.R_PRIME_AWAY: GripState.HOME,
            GripState.R2_AWAY: None,
        },
        "r'": {
            GripState.HOME: GripState.R_PRIME_AWAY,
            GripState.R_AWAY: GripState.HOME,
            GripState.R_PRIME_AWAY: None,
            GripState.R2_AWAY: GripState.R_AWAY,
        },
        "r2": {
            GripState.HOME: GripState.R2_AWAY,
            GripState.R_AWAY: GripState.R_PRIME_AWAY,
            GripState.R_PRIME_AWAY: GripState.R_AWAY,
            GripState.R2_AWAY: GripState.HOME,
        },
    }

    # Set of moves considered anatomically impossible or blocked from each grip state without regrip
    _BLOCKED_MOVES: Dict[GripState, Set[str]] = {
        GripState.HOME: {
            "F2", "B",
        },
        GripState.R_AWAY: {
            "F", "r2",
        },
        GripState.R_PRIME_AWAY: {
            "U", "R'", "r'",
        },
        GripState.R2_AWAY: {
            "R", "r", "r'", "r2",
            "U", "F", "F'", "F2",
            "B", "B'", "B2",
            "D", "D'", "D2",
            "M",
        },
    }

    @classmethod
    def is_valid_turn(cls, move: str, grip: GripState) -> bool:
        """Determines whether a move can be physically executed from the given grip state without regrip."""
        norm = MoveParser.normalize_token(move)
        # Check explicit blocked moves
        if norm in cls._BLOCKED_MOVES.get(grip, set()):
            return False

        # If it is an R-family turn, check if it leads to a dead end
        if norm in cls._R_TRANSITIONS:
            return cls._R_TRANSITIONS[norm].get(grip) is not None

        return True

    @classmethod
    def get_next_grip(cls, move: str, grip: GripState) -> Optional[GripState]:
        """Returns the resulting GripState after executing move from grip, or None if impossible."""
        if not cls.is_valid_turn(move, grip):
            return None

        norm = MoveParser.normalize_token(move)
        # Whole-cube rotations reset or preserve home grip
        if norm.startswith(("x", "y", "z")):
            return GripState.HOME

        # R / r family turns physically rotate the right hand
        if norm in cls._R_TRANSITIONS:
            return cls._R_TRANSITIONS[norm].get(grip)

        # Non-R turns do not rotate the right wrist
        return grip

    _GRIP_STRAIN: Dict[GripState, float] = {
        GripState.HOME: 0.0,
        GripState.R_AWAY: 0.1,
        GripState.R_PRIME_AWAY: 0.15,
        GripState.R2_AWAY: 0.3,
    }

    _REGRIP_BASE_COST: float = 1000.0

    def track(
        self,
        moves: Union[str, Sequence[Union[str, MoveEvent]]],
        initial_grip: GripState = GripState.HOME,
    ) -> GripTrackingResult:
        """Finds the optimal sequence of grip transitions minimizing mechanical regrips."""
        if isinstance(moves, str):
            events = MoveParser.parse_string(moves)
            tokens = [e.move for e in events]
        else:
            tokens = [
                m.move if isinstance(m, MoveEvent) else MoveParser.normalize_token(str(m))
                for m in moves
            ]

        # Filter out empty tokens
        tokens = [t for t in tokens if t]

        if not tokens:
            return GripTrackingResult(
                regrip_count=0,
                steps=[],
                initial_grip=initial_grip,
                final_grip=initial_grip,
            )

        all_grips = [
            GripState.HOME,
            GripState.R_AWAY,
            GripState.R_PRIME_AWAY,
            GripState.R2_AWAY,
        ]

        # current_dp[grip] = (cost, list_of_steps)
        current_dp: Dict[GripState, Tuple[float, List[GripStep]]] = {
            g: (float("inf"), []) for g in all_grips
        }
        current_dp[initial_grip] = (0.0, [])
        for g in all_grips:
            if g != initial_grip:
                current_dp[g] = (self._REGRIP_BASE_COST + self._GRIP_STRAIN[g], [])

        for move in tokens:
            next_dp: Dict[GripState, Tuple[float, List[GripStep]]] = {
                g: (float("inf"), []) for g in all_grips
            }

            for prev_grip in all_grips:
                prev_cost, prev_steps = current_dp[prev_grip]
                if prev_cost == float("inf"):
                    continue

                for shifted_grip in all_grips:
                    if not self.is_valid_turn(move, shifted_grip):
                        continue
                    next_grip = self.get_next_grip(move, shifted_grip)
                    if next_grip is None:
                        continue

                    is_regrip = (shifted_grip != prev_grip)
                    step_cost = (
                        (self._REGRIP_BASE_COST if is_regrip else 0.0)
                        + self._GRIP_STRAIN[shifted_grip]
                    )
                    total_cost = prev_cost + step_cost

                    if total_cost < next_dp[next_grip][0]:
                        step = GripStep(
                            move=move,
                            grip_before=prev_grip,
                            grip_during=shifted_grip,
                            grip_after=next_grip,
                            regrip=is_regrip,
                        )
                        next_dp[next_grip] = (total_cost, prev_steps + [step])

            current_dp = next_dp

        # Find best terminal grip state, tie-breaking in favor of HOME
        best_grip: Optional[GripState] = None
        best_cost = float("inf")
        best_steps: List[GripStep] = []

        for g in all_grips:
            cost, steps = current_dp[g]
            if cost < best_cost or (cost == best_cost and g == GripState.HOME):
                best_cost = cost
                best_grip = g
                best_steps = steps

        if best_grip is None:
            best_grip = initial_grip
            best_steps = []

        regrip_count = sum(1 for s in best_steps if s.regrip)
        return GripTrackingResult(
            regrip_count=regrip_count,
            steps=best_steps,
            initial_grip=initial_grip,
            final_grip=best_grip,
        )


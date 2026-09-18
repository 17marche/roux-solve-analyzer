"""Domain models and data structures for ergonomics and biomechanical flow evaluation."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class GripState(str, Enum):
    """Mechanical orientation state of the solver's right hand relative to the cube."""
    HOME = "HOME"
    R_AWAY = "R_AWAY"
    R_PRIME_AWAY = "R_PRIME_AWAY"
    R2_AWAY = "R2_AWAY"


class PauseType(str, Enum):
    """Classification of speedcube stream pauses and execution micro-pauses."""
    COGNITIVE_HESITATION = "COGNITIVE_HESITATION"
    PHYSICAL_REGRIP = "PHYSICAL_REGRIP"
    EXECUTION_LOCKUP = "EXECUTION_LOCKUP"
    # Canonical domain aliases from CONTEXT.md
    COGNITIVE_MICRO_PAUSE = "COGNITIVE_HESITATION"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class HandProfile:
    """Solver-specific physical biomechanics configuration."""
    solving_mode: str = "2H"            # "2H" or "OH" (canonical domain term from CONTEXT.md)
    m_slice_hand: str = "right"         # "right" or "left"
    dominant_hand: str = "right"        # "right" or "left"
    style: str = ""                     # Backwards-compatible alias for solving_mode

    def __post_init__(self) -> None:
        if self.style and self.solving_mode == "2H" and self.style != "2H":
            object.__setattr__(self, "solving_mode", self.style)
        elif not self.style:
            object.__setattr__(self, "style", self.solving_mode)



@dataclass(frozen=True)
class MoveAnalysis:
    """Per-move ergonomic and kinematic evaluation record."""
    move: str
    grip_before: GripState
    grip_after: GripState
    regrip: bool = False
    transition_effort: float = 1.0
    timestamp_ms: Optional[int] = None
    delta_ms: Optional[int] = None
    pause_type: Optional[PauseType] = None


@dataclass(frozen=True)
class FlowScore:
    """Structured evaluation record for a move sequence's biomechanical flow."""
    raw_stm: int
    e_stm: float
    kinematic_efficiency: float
    regrip_count: int
    macro_triggers: List[str] = field(default_factory=list)
    turning_ratio: Optional[float] = None
    rhythm_cv: Optional[float] = None
    stream_flow_index: Optional[float] = None
    per_move_analysis: List[MoveAnalysis] = field(default_factory=list)
    pause_breakdown: Dict[str, int] = field(default_factory=dict)



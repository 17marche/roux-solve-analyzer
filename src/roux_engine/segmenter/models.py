"""Data models and schemas for the Roux Phase Segmenter."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json

from ..core.constants import Color


@dataclass
class Orientation:
    """Solve orientation description."""
    bottom_color: Color
    left_color: Color
    rotations: str  # e.g. "", "y", "x2", "x2y", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottom_color": self.bottom_color.name,
            "left_color": self.left_color.name,
            "rotations": self.rotations
        }


@dataclass
class ConcurrentSBProgress:
    """Tracks concurrent Second Block pieces formed during First Block."""
    dr_solved: bool = False
    sb_square_solved: bool = False
    sb_pair_solved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FBPhase:
    """First Block (FB) segmentation result."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    orientation: Optional[Orientation] = None
    concurrent_sb: ConcurrentSBProgress = field(default_factory=ConcurrentSBProgress)
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.orientation:
            d["orientation"] = self.orientation.to_dict()
        return d


@dataclass
class SBPhase:
    """Second Block (SB) segmentation result."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    dr_placement_idx: Optional[int] = None
    sb_square_idx: Optional[int] = None
    pair1_idx: Optional[int] = None
    pair1_type: str = "unknown"  # "back", "front", "line_attach", "non_standard"
    pair2_idx: Optional[int] = None
    rotation_count: int = 0
    non_ergonomic_moves: List[str] = field(default_factory=list)
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CMLLPhase:
    """Corners of Last Layer (CMLL) segmentation result."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    case_id: str = "solved"  # e.g. "pi_right_bar", "s_left_bar", "solved"
    group: str = "Skip"      # "Sune", "Antisune", "H", "Pi", "T", "U", "L", "O", "Skip"
    pre_auf: str = ""        # "", "U", "U'", "U2"
    is_standard_alg: bool = True
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step4a:
    """Step 4a: Edge Orientation (EO / EOLR)."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    variant: str = "standard_eo"  # "standard_eo", "eolr", "eolr_b"
    initial_bad_edges: int = 0
    center_state: str = "axis_aligned"  # "axis_aligned", "misaligned"
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step4b:
    """Step 4b: UL/UR Edge Insertion into top slots."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    skipped: bool = False
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Step4c:
    """Step 4c: M-slice and Center Permutation."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    case: str = "solved"  # "solved", "dots", "bars", "opp_opp", "3_cycle", "cycle_4", "center_swap"
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LSEPhase:
    """Last Six Edges (LSE / Step 4) segmentation result."""
    start_move_idx: int
    end_move_idx: int
    move_count_stm: int
    time_ms: Optional[int] = None
    step_4a: Optional[Step4a] = None
    step_4b: Optional[Step4b] = None
    step_4c: Optional[Step4c] = None
    moves_str: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_move_idx": self.start_move_idx,
            "end_move_idx": self.end_move_idx,
            "move_count_stm": self.move_count_stm,
            "time_ms": self.time_ms,
            "step_4a": self.step_4a.to_dict() if self.step_4a else None,
            "step_4b": self.step_4b.to_dict() if self.step_4b else None,
            "step_4c": self.step_4c.to_dict() if self.step_4c else None,
            "moves_str": self.moves_str
        }


@dataclass
class SegmentedSolve:
    """Master container for a fully segmented Roux solve."""
    scramble: str
    solution_moves_count: int
    total_moves_stm: int
    total_time_ms: Optional[int] = None
    tps: Optional[float] = None
    is_valid: bool = True
    fb: Optional[FBPhase] = None
    sb: Optional[SBPhase] = None
    cmll: Optional[CMLLPhase] = None
    lse: Optional[LSEPhase] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scramble": self.scramble,
            "solution_moves_count": self.solution_moves_count,
            "total_moves_stm": self.total_moves_stm,
            "total_time_ms": self.total_time_ms,
            "tps": self.tps,
            "is_valid": self.is_valid,
            "fb": self.fb.to_dict() if self.fb else None,
            "sb": self.sb.to_dict() if self.sb else None,
            "cmll": self.cmll.to_dict() if self.cmll else None,
            "lse": self.lse.to_dict() if self.lse else None,
            "warnings": self.warnings
        }

    def to_json(self, indent: int = 2) -> str:
        def _default_serializer(obj: Any) -> Any:
            if hasattr(obj, "item"):
                return obj.item()
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(self.to_dict(), indent=indent, default=_default_serializer)

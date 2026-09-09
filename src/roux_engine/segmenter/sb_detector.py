"""Second Block (SB) Detector for Roux method."""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Union, Any
import numpy as np

from ..core.constants import Corner, Edge, Center
from ..core.cube import CubeState
from ..core.parser import MoveEvent, MoveParser
from ..core.orientation import RouxOrientation, get_orientation
from .models import SBPhase
from .fb_detector import BlockDefinition, ALL_BLOCK_DEFINITIONS


ROUX_SB_ERGONOMIC_PREFIXES = ("R", "r", "M", "U", "u")


class SBDetector:
    """Detects Second Block completion, tracks sub-phases (DR, Pair 1, Pair 2), and flags rotations."""

    @staticmethod
    def is_canonical_sb_solved(state: CubeState) -> bool:
        """Checks if the canonical Right Second Block (DR, FR, BR, DFR, DRB) is solved."""
        return get_orientation("").is_sb_solved(state)

    @staticmethod
    def is_dr_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if DR edge is solved in its designated slot with exact piece ID and orientation."""
        return get_orientation(block).is_dr_solved(state)

    @staticmethod
    def is_back_pair_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if Right back pair (BR edge + DRB corner) is solved with exact piece IDs and orientations."""
        return get_orientation(block).is_back_pair_solved(state)

    @staticmethod
    def is_front_pair_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if Right front pair (FR edge + DFR corner) is solved with exact piece IDs and orientations."""
        return get_orientation(block).is_front_pair_solved(state)

    @staticmethod
    def is_right_1x2x3_block_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if all 5 Right Block pieces are simultaneously solved with exact piece IDs and orientations."""
        return get_orientation(block).is_sb_solved(state)

    @classmethod
    def detect_sb(
        cls,
        fb_state: CubeState,
        events: Sequence[MoveEvent],
        fb_end_idx: int,
        block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]
    ) -> Optional[Tuple[SBPhase, CubeState]]:
        """Tracks the solve starting directly from fb_state at fb_end_idx until SB is completed.
        
        Returns:
            Tuple of (SBPhase, cube_state_at_sb) or None.
        """
        ori = get_orientation(block)
        dr_placement_idx: Optional[int] = None
        sb_square_idx: Optional[int] = None
        pair1_idx: Optional[int] = None
        pair1_type: str = "unknown"
        pair2_idx: Optional[int] = None
        rotation_count = 0
        non_ergonomic_moves: List[str] = []

        # Start simulation directly from fb_state (no redundant re-simulation from move 0)
        sim = fb_state.copy()

        # Concurrent progress inspection at fb_end_idx before executing SB turns
        if ori.is_dr_solved(sim):
            dr_placement_idx = fb_end_idx

        back_solved_init = ori.is_back_pair_solved(sim)
        front_solved_init = ori.is_front_pair_solved(sim)

        if dr_placement_idx is not None:
            if back_solved_init and not front_solved_init:
                pair1_idx = fb_end_idx
                pair1_type = "back"
                sb_square_idx = fb_end_idx
            elif front_solved_init and not back_solved_init:
                pair1_idx = fb_end_idx
                pair1_type = "front"
                sb_square_idx = fb_end_idx
            elif back_solved_init and front_solved_init:
                pair1_idx = fb_end_idx
                pair1_type = "both_simultaneous"
                sb_square_idx = fb_end_idx

        # Check if SB was already completely solved at FB end
        if ori.is_sb_solved(sim):
            pair2_idx = fb_end_idx
            if pair1_idx is None:
                pair1_idx = fb_end_idx
                pair1_type = "both_simultaneous"
                sb_square_idx = fb_end_idx
            time_ms = events[fb_end_idx].timestamp_ms if fb_end_idx >= 0 and fb_end_idx < len(events) and events[fb_end_idx].timestamp_ms is not None else None
            phase = SBPhase(
                start_move_idx=fb_end_idx + 1,
                end_move_idx=fb_end_idx,
                move_count_stm=0,
                time_ms=time_ms,
                dr_placement_idx=dr_placement_idx,
                sb_square_idx=sb_square_idx,
                pair1_idx=pair1_idx,
                pair1_type=pair1_type,
                pair2_idx=pair2_idx,
                rotation_count=0,
                non_ergonomic_moves=[],
                moves_str=""
            )
            return phase, sim.copy()

        # Move-by-move tracking through SB execution
        for i in range(fb_end_idx + 1, len(events)):
            ev = events[i]
            move_token = ev.move
            sim.apply_move(move_token)

            if move_token.startswith(('x', 'y', 'z')):
                rotation_count += 1
            else:
                if not any(move_token.startswith(p) for p in ROUX_SB_ERGONOMIC_PREFIXES):
                    non_ergonomic_moves.append(move_token)

            if dr_placement_idx is None and ori.is_dr_solved(sim):
                dr_placement_idx = i

            back_solved = ori.is_back_pair_solved(sim)
            front_solved = ori.is_front_pair_solved(sim)
            dr_solved = ori.is_dr_solved(sim)

            # Pair 1 & SB Square detection (coupled with DR)
            if pair1_idx is None and dr_solved:
                if back_solved and not front_solved:
                    pair1_idx = i
                    pair1_type = "back"
                    sb_square_idx = i
                elif front_solved and not back_solved:
                    pair1_idx = i
                    pair1_type = "front"
                    sb_square_idx = i
                elif back_solved and front_solved:
                    pair1_idx = i
                    pair1_type = "both_simultaneous"
                    sb_square_idx = i

            # Full Right Block SB Completion
            if ori.is_sb_solved(sim):
                pair2_idx = i
                if pair1_idx is None:
                    pair1_idx = i
                    pair1_type = "both_simultaneous"
                    sb_square_idx = i
                if dr_placement_idx is None:
                    dr_placement_idx = i

                pair2_type = "unknown"
                if pair1_type == "front":
                    pair2_type = "back"
                elif pair1_type == "back":
                    pair2_type = "front"
                elif pair1_type == "both_simultaneous":
                    pair2_type = "both_simultaneous"

                # Sub-slice move strings and counts
                dr_events = events[fb_end_idx + 1:dr_placement_idx + 1] if dr_placement_idx is not None and dr_placement_idx > fb_end_idx else []
                dr_moves_str = " ".join(e.move for e in dr_events)
                dr_moves_stm = len([e for e in dr_events if not e.move.startswith(('x', 'y', 'z'))])

                p1_start = dr_placement_idx if dr_placement_idx is not None and dr_placement_idx > fb_end_idx else fb_end_idx
                p1_events = events[p1_start + 1:pair1_idx + 1] if pair1_idx is not None and pair1_idx > p1_start else []
                pair1_moves_str = " ".join(e.move for e in p1_events)
                pair1_moves_stm = len([e for e in p1_events if not e.move.startswith(('x', 'y', 'z'))])

                p2_start = pair1_idx if pair1_idx is not None and pair1_idx > fb_end_idx else p1_start
                p2_events = events[p2_start + 1:i + 1] if i > p2_start else []
                pair2_moves_str = " ".join(e.move for e in p2_events)
                pair2_moves_stm = len([e for e in p2_events if not e.move.startswith(('x', 'y', 'z'))])

                sb_events = events[fb_end_idx + 1:i + 1]
                moves_stm = len([e for e in sb_events if not e.move.startswith(('x', 'y', 'z'))])
                time_ms = sb_events[-1].timestamp_ms if sb_events and sb_events[-1].timestamp_ms is not None else None

                phase = SBPhase(
                    start_move_idx=fb_end_idx + 1,
                    end_move_idx=i,
                    move_count_stm=moves_stm,
                    time_ms=time_ms,
                    dr_placement_idx=dr_placement_idx,
                    dr_moves_str=dr_moves_str,
                    dr_moves_stm=dr_moves_stm,
                    sb_square_idx=sb_square_idx,
                    pair1_idx=pair1_idx,
                    pair1_type=pair1_type,
                    pair1_moves_str=pair1_moves_str,
                    pair1_moves_stm=pair1_moves_stm,
                    pair2_idx=pair2_idx,
                    pair2_type=pair2_type,
                    pair2_moves_str=pair2_moves_str,
                    pair2_moves_stm=pair2_moves_stm,
                    rotation_count=rotation_count,
                    non_ergonomic_moves=non_ergonomic_moves,
                    moves_str=" ".join(e.move for e in sb_events)
                )
                return phase, sim.copy()

        return None

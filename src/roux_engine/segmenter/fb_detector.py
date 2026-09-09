"""First Block (FB) Detector for Roux method."""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Dict, NamedTuple
import numpy as np

from ..core.constants import Corner, Edge, Center, Color
from ..core.cube import CubeState
from ..core.moves import Move, apply_move, apply_moves, get_move
from ..core.parser import MoveEvent, MoveParser
from ..core.orientation import (
    RouxOrientation,
    CanonicalSymmetry,
    get_all_orientations,
    get_dual_neutral_orientations,
    get_orientation,
)
from .models import FBPhase, Orientation, ConcurrentSBProgress


# Export BlockDefinition and ALL_BLOCK_DEFINITIONS as aliases to RouxOrientation
# for backward compatibility with unmigrated solver imports.
BlockDefinition = RouxOrientation
ALL_BLOCK_DEFINITIONS: Dict[str, RouxOrientation] = {
    o.rotations: o for o in get_all_orientations()
}

# 8 Dual-Neutral Orientations (x2y group)
DUAL_NEUTRAL_ORIENTATIONS: List[str] = [o.rotations for o in get_dual_neutral_orientations()]

# 24 Full Color-Neutral Orientations
FULL_COLOR_NEUTRAL_ORIENTATIONS: List[str] = [o.rotations for o in get_all_orientations()]


class FBDetector:
    """Detects First Block completion across orientations and checks concurrent SB progress."""

    @staticmethod
    def is_canonical_fb_solved(state: CubeState) -> bool:
        """Checks if the canonical Left First Block (DL, FL, BL, DFL, DBL, L center) is solved."""
        return get_orientation("").is_fb_solved(state)

    @staticmethod
    def match_fb_block(
        state: CubeState,
        full_color_neutral: bool = False
    ) -> Optional[RouxOrientation]:
        """Matches state against valid block definitions, checking exact piece IDs and orientations."""
        candidates = get_all_orientations() if full_color_neutral else get_dual_neutral_orientations()
        for ori in candidates:
            if ori.is_fb_solved(state):
                return ori
        return None

    @staticmethod
    def check_concurrent_sb(state: CubeState, block: RouxOrientation) -> ConcurrentSBProgress:
        """Inspects state for concurrent Right block (SB) pieces in designated slots with exact IDs and orientations."""
        dr_solved = block.is_dr_solved(state)
        back_pair_solved = block.is_back_pair_solved(state)
        front_pair_solved = block.is_front_pair_solved(state)

        sb_square_solved = dr_solved and (back_pair_solved or front_pair_solved)
        sb_pair_solved = back_pair_solved or front_pair_solved

        return ConcurrentSBProgress(
            dr_solved=bool(dr_solved),
            sb_square_solved=bool(sb_square_solved),
            sb_pair_solved=bool(sb_pair_solved)
        )

    @classmethod
    def detect_fb(
        cls,
        initial_state: CubeState,
        events: Sequence[MoveEvent],
        full_color_neutral: bool = False
    ) -> Optional[Tuple[FBPhase, str, CubeState, BlockDefinition]]:
        """Finds the earliest move index where FB is solved.
        
        Returns:
            Tuple of (FBPhase, orientation_rotations, cube_state_at_fb, matched_block) or None.
        """
        # Separate initial inspection rotations
        inspection_rotations: List[str] = []
        start_exec_idx = 0
        while start_exec_idx < len(events) and events[start_exec_idx].move.startswith(('x', 'y', 'z')):
            inspection_rotations.append(events[start_exec_idx].move)
            start_exec_idx += 1

        sim = initial_state.copy()
        for rot in inspection_rotations:
            sim.apply_move(rot)

        # Check if FB was already solved at inspection (before any execution moves)
        matched_block = cls.match_fb_block(sim, full_color_neutral=full_color_neutral)
        if matched_block is not None:
            concurrent_sb = cls.check_concurrent_sb(sim, matched_block)
            ori_info = Orientation(
                bottom_color=matched_block.bottom_color,
                left_color=matched_block.left_color,
                rotations=" ".join(inspection_rotations)
            )
            # When FB is solved at start, end_move_idx is start_exec_idx - 1 (-1 if 0 moves executed)
            end_idx = start_exec_idx - 1
            phase = FBPhase(
                start_move_idx=0,
                end_move_idx=end_idx,
                move_count_stm=0,
                time_ms=events[end_idx].timestamp_ms if end_idx >= 0 and events[end_idx].timestamp_ms is not None else None,
                orientation=ori_info,
                concurrent_sb=concurrent_sb,
                moves_str=" ".join(inspection_rotations)
            )
            return phase, " ".join(inspection_rotations), sim.copy(), matched_block

        # Track simulation move by move
        for i in range(start_exec_idx, len(events)):
            ev = events[i]
            sim.apply_move(ev.move)

            matched_block = cls.match_fb_block(sim, full_color_neutral=full_color_neutral)
            if matched_block is not None:
                end_idx = i
                fb_events = events[:end_idx + 1]
                moves_stm = len([e for e in fb_events if not e.move.startswith(('x', 'y', 'z'))])
                time_ms = fb_events[-1].timestamp_ms if fb_events and fb_events[-1].timestamp_ms is not None else None
                
                concurrent_sb = cls.check_concurrent_sb(sim, matched_block)
                ori_info = Orientation(
                    bottom_color=matched_block.bottom_color,
                    left_color=matched_block.left_color,
                    rotations=" ".join(inspection_rotations)
                )

                phase = FBPhase(
                    start_move_idx=0,
                    end_move_idx=end_idx,
                    move_count_stm=moves_stm,
                    time_ms=time_ms,
                    orientation=ori_info,
                    concurrent_sb=concurrent_sb,
                    moves_str=" ".join(e.move for e in fb_events)
                )
                return phase, " ".join(inspection_rotations), sim.copy(), matched_block

        return None

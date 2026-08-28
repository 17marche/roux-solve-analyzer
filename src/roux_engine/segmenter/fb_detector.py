"""First Block (FB) Detector for Roux method."""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Dict, NamedTuple
import numpy as np

from ..core.constants import Corner, Edge, Center, Color
from ..core.cube import CubeState
from ..core.moves import Move, apply_move, apply_moves, get_move
from ..core.parser import MoveEvent, MoveParser
from .models import FBPhase, Orientation, ConcurrentSBProgress


class BlockDefinition(NamedTuple):
    """Specification of a 1x2x3 Left Block and its corresponding Right Block / LSE pieces."""
    rotations: str
    left_color: Color
    bottom_color: Color
    front_color: Color
    back_color: Color
    right_color: Color
    top_color: Color
    # Left Block piece requirements (slot position: piece ID, expected orientation)
    dl_piece: int
    dl_eo: int
    fl_piece: int
    fl_eo: int
    bl_piece: int
    bl_eo: int
    dlf_piece: int
    dlf_co: int
    dbl_piece: int
    dbl_co: int
    # Right Block piece requirements (for concurrent SB tracking)
    dr_piece: int
    dr_eo: int
    fr_piece: int
    fr_eo: int
    br_piece: int
    br_eo: int
    dfr_piece: int
    dfr_co: int
    drb_piece: int
    drb_co: int
    # LSE piece requirements
    ul_piece: int
    ul_eo: int
    ur_piece: int
    ur_eo: int
    uf_piece: int
    uf_eo: int
    ub_piece: int
    ub_eo: int
    df_piece: int
    df_eo: int
    db_piece: int
    db_eo: int


def _generate_block_definitions() -> Dict[str, BlockDefinition]:
    """Precomputes exact piece IDs and orientations for all 24 orientations."""
    orientations = [
        # D on bottom (Dual Neutral 1-4)
        "", "y", "y2", "y'",
        # U on bottom (Dual Neutral 5-8)
        "x2", "x2 y", "x2 y2", "x2 y'",
        # F on bottom
        "x", "x y", "x y2", "x y'",
        # B on bottom
        "x'", "x' y", "x' y2", "x' y'",
        # L on bottom
        "z", "z y", "z y2", "z y'",
        # R on bottom
        "z'", "z' y", "z' y2", "z' y'",
    ]

    defs: Dict[str, BlockDefinition] = {}
    for ori in orientations:
        c = CubeState()
        if ori:
            c.apply_moves(ori)

        b = BlockDefinition(
            rotations=ori,
            left_color=Color(int(c.centers[Center.L])),
            bottom_color=Color(int(c.centers[Center.D])),
            front_color=Color(int(c.centers[Center.F])),
            back_color=Color(int(c.centers[Center.B])),
            right_color=Color(int(c.centers[Center.R])),
            top_color=Color(int(c.centers[Center.U])),
            dl_piece=int(c.ep[Edge.DL]),
            dl_eo=int(c.eo[Edge.DL]),
            fl_piece=int(c.ep[Edge.FL]),
            fl_eo=int(c.eo[Edge.FL]),
            bl_piece=int(c.ep[Edge.BL]),
            bl_eo=int(c.eo[Edge.BL]),
            dlf_piece=int(c.cp[Corner.DLF]),
            dlf_co=int(c.co[Corner.DLF]),
            dbl_piece=int(c.cp[Corner.DBL]),
            dbl_co=int(c.co[Corner.DBL]),
            dr_piece=int(c.ep[Edge.DR]),
            dr_eo=int(c.eo[Edge.DR]),
            fr_piece=int(c.ep[Edge.FR]),
            fr_eo=int(c.eo[Edge.FR]),
            br_piece=int(c.ep[Edge.BR]),
            br_eo=int(c.eo[Edge.BR]),
            dfr_piece=int(c.cp[Corner.DFR]),
            dfr_co=int(c.co[Corner.DFR]),
            drb_piece=int(c.cp[Corner.DRB]),
            drb_co=int(c.co[Corner.DRB]),
            ul_piece=int(c.ep[Edge.UL]),
            ul_eo=int(c.eo[Edge.UL]),
            ur_piece=int(c.ep[Edge.UR]),
            ur_eo=int(c.eo[Edge.UR]),
            uf_piece=int(c.ep[Edge.UF]),
            uf_eo=int(c.eo[Edge.UF]),
            ub_piece=int(c.ep[Edge.UB]),
            ub_eo=int(c.eo[Edge.UB]),
            df_piece=int(c.ep[Edge.DF]),
            df_eo=int(c.eo[Edge.DF]),
            db_piece=int(c.ep[Edge.DB]),
            db_eo=int(c.eo[Edge.DB]),
        )
        defs[ori] = b

    return defs


ALL_BLOCK_DEFINITIONS = _generate_block_definitions()

# 8 Dual-Neutral Orientations (x2y group)
DUAL_NEUTRAL_ORIENTATIONS = [
    "", "y", "y2", "y'",
    "x2", "x2 y", "x2 y2", "x2 y'"
]

# 24 Full Color-Neutral Orientations
FULL_COLOR_NEUTRAL_ORIENTATIONS = list(ALL_BLOCK_DEFINITIONS.keys())


class FBDetector:
    """Detects First Block completion across orientations and checks concurrent SB progress."""

    @staticmethod
    def is_canonical_fb_solved(state: CubeState) -> bool:
        """Checks if the canonical Left First Block (DL, FL, BL, DFL, DBL, L center) is solved.
        
        Enforces exact canonical colors and piece IDs:
        - Center L == Orange
        - DLF == Corner.DLF (co == 0)
        - DBL == Corner.DBL (co == 0)
        - DL == Edge.DL (eo == 0)
        - FL == Edge.FL (eo == 0)
        - BL == Edge.BL (eo == 0)
        """
        if state.centers[Center.L] != Center.L:
            return False
        if state.cp[Corner.DLF] != Corner.DLF or state.co[Corner.DLF] != 0:
            return False
        if state.cp[Corner.DBL] != Corner.DBL or state.co[Corner.DBL] != 0:
            return False
        if state.ep[Edge.DL] != Edge.DL or state.eo[Edge.DL] != 0:
            return False
        if state.ep[Edge.FL] != Edge.FL or state.eo[Edge.FL] != 0:
            return False
        if state.ep[Edge.BL] != Edge.BL or state.eo[Edge.BL] != 0:
            return False
        return True

    @staticmethod
    def match_fb_block(
        state: CubeState,
        full_color_neutral: bool = False
    ) -> Optional[BlockDefinition]:
        """Matches state against valid block definitions, checking exact piece IDs and orientations."""
        candidate_keys = FULL_COLOR_NEUTRAL_ORIENTATIONS if full_color_neutral else DUAL_NEUTRAL_ORIENTATIONS
        
        c_l = int(state.centers[Center.L])

        for key in candidate_keys:
            b = ALL_BLOCK_DEFINITIONS[key]
            # Center L must match block's left color
            if c_l != b.left_color.value:
                continue

            # Verify all 5 Left Block pieces and exact orientations
            if state.ep[Edge.DL] != b.dl_piece or state.eo[Edge.DL] != b.dl_eo:
                continue
            if state.ep[Edge.FL] != b.fl_piece or state.eo[Edge.FL] != b.fl_eo:
                continue
            if state.ep[Edge.BL] != b.bl_piece or state.eo[Edge.BL] != b.bl_eo:
                continue
            if state.cp[Corner.DLF] != b.dlf_piece or state.co[Corner.DLF] != b.dlf_co:
                continue
            if state.cp[Corner.DBL] != b.dbl_piece or state.co[Corner.DBL] != b.dbl_co:
                continue

            return b

        return None

    @staticmethod
    def check_concurrent_sb(state: CubeState, block: BlockDefinition) -> ConcurrentSBProgress:
        """Inspects state for concurrent Right block (SB) pieces in designated slots with exact IDs and orientations."""
        # 1. DR edge solved in DR slot
        dr_solved = (state.ep[Edge.DR] == block.dr_piece and state.eo[Edge.DR] == block.dr_eo)

        # 2. Back pair solved in BR and DRB slots
        back_pair_solved = (
            state.ep[Edge.BR] == block.br_piece and state.eo[Edge.BR] == block.br_eo and
            state.cp[Corner.DRB] == block.drb_piece and state.co[Corner.DRB] == block.drb_co
        )

        # 3. Front pair solved in FR and DFR slots
        front_pair_solved = (
            state.ep[Edge.FR] == block.fr_piece and state.eo[Edge.FR] == block.fr_eo and
            state.cp[Corner.DFR] == block.dfr_piece and state.co[Corner.DFR] == block.dfr_co
        )

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

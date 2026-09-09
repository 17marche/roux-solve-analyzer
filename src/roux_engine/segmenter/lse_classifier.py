"""Last Six Edges (LSE / Step 4) Classifier for 4a (EO/EOLR), 4b, and 4c."""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Union, Any
import numpy as np

from ..core.constants import Edge, Center
from ..core.cube import CubeState
from ..core.parser import MoveEvent, MoveParser
from ..core.orientation import RouxOrientation, get_orientation
from .models import LSEPhase, Step4a, Step4b, Step4c
from .fb_detector import BlockDefinition, ALL_BLOCK_DEFINITIONS


LSE_EDGES = (Edge.UF, Edge.UB, Edge.UL, Edge.UR, Edge.DF, Edge.DB)
M_SLICE_EDGES = (Edge.UF, Edge.UB, Edge.DF, Edge.DB)
AUF_MOVES = ("", "U", "U2", "U'")


class LSEClassifier:
    """Classifies LSE micro-steps: 4a (EO/EOLR/EOLR-b), 4b (UL/UR), and 4c (M-slice permutation)."""

    @staticmethod
    def is_center_axis_aligned(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if U and D centers occupy the U/D axis defined by block."""
        return get_orientation(block).is_center_axis_aligned(state)

    @staticmethod
    def count_bad_edges(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> int:
        """Counts how many of the 6 LSE edges have bad orientation."""
        return get_orientation(block).count_bad_edges(state)

    @staticmethod
    def is_eo_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if all 6 LSE edges are oriented along the U/D axis."""
        return get_orientation(block).is_eo_solved(state)

    @staticmethod
    def is_ul_ur_solved(state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if UL and UR edges are placed in their correct slots relative to U-layer corners (up to AUF)."""
        return get_orientation(block).is_ul_ur_solved(state)

    @classmethod
    def classify_4a_variant(cls, state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> str:
        """Classifies EO completion into 'eolr_b', 'eolr', or 'standard_eo'.
        
        Requires all 6 LSE edges to be oriented along the U/D axis.
        """
        ori = get_orientation(block)
        if not ori.is_eo_solved(state):
            raise ValueError("Cannot classify 4a variant when EO is not solved.")

        # 1. EOLR-b: UL and UR are already placed into UL and UR slots relative to U-corners
        if ori.is_ul_ur_solved(state):
            return "eolr_b"

        # 2. EOLR: UL and UR are positioned into bottom DF and DB slots
        df_edge = int(state.ep[Edge.DF])
        db_edge = int(state.ep[Edge.DB])
        if {df_edge, db_edge} == {ori.ul_piece, ori.ur_piece}:
            return "eolr"

        return "standard_eo"

    @classmethod
    def is_fully_solved(cls, state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if the cube is fully solved relative to the block orientation."""
        ori = get_orientation(block)
        target = CubeState()
        if ori.rotations:
            target.apply_moves(ori.rotations)
        return state == target

    @classmethod
    def classify_4c_case(cls, state: CubeState, block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]) -> str:
        """Classifies the initial 4c permutation case:
        - solved: all 4 M-slice edges and centers solved
        - center_swap: M2 center swap
        - dots: vertical bars / opposite centers (UF<->DF, UB<->DB)
        - bars: horizontal bars (UF<->UB or DF<->DB)
        - opp_opp: horizontal bars on both top and bottom (UF<->UB and DF<->DB)
        - 3_cycle: 3 M-slice edges cycled (e.g. M U2 M' U2)
        - cycle_4: 4 M-slice edges cycled
        """
        ori = get_orientation(block)
        target = CubeState()
        if ori.rotations:
            target.apply_moves(ori.rotations)
        expected_cp = tuple(int(x) for x in target.cp[0:4])

        # Align U-layer corners with the solve orientation to eliminate AUF distortion
        aligned_state = state.copy()
        for u in AUF_MOVES:
            test_c = state.copy()
            if u:
                test_c.apply_move(u)
            if tuple(int(x) for x in test_c.cp[0:4]) == expected_cp:
                aligned_state = test_c
                break

        if cls.is_fully_solved(aligned_state, ori):
            return "solved"

        test_m2 = aligned_state.copy()
        test_m2.apply_move("M2")
        if cls.is_fully_solved(test_m2, ori):
            return "center_swap"

        ep_m = (
            int(aligned_state.ep[Edge.UF]),
            int(aligned_state.ep[Edge.UB]),
            int(aligned_state.ep[Edge.DF]),
            int(aligned_state.ep[Edge.DB])
        )

        # Dots (vertical bars): all 4 edges swapped across U and D
        if ep_m in (
            (ori.df_piece, ori.db_piece, ori.uf_piece, ori.ub_piece),
            (ori.db_piece, ori.df_piece, ori.ub_piece, ori.uf_piece)
        ):
            return "dots"

        # Opp-Opp: opposite on top (UF<->UB) and bottom (DF<->DB)
        if ep_m == (ori.ub_piece, ori.uf_piece, ori.db_piece, ori.df_piece):
            return "opp_opp"

        displaced = sum(
            1 for actual, expected in zip(
                ep_m,
                (ori.uf_piece, ori.ub_piece, ori.df_piece, ori.db_piece)
            ) if actual != expected
        )

        if displaced == 2:
            return "bars"
        elif displaced == 3:
            return "3_cycle"
        elif displaced == 4:
            return "cycle_4"

        return "bars"

    @classmethod
    def detect_lse(
        cls,
        cmll_state: CubeState,
        events: Sequence[MoveEvent],
        cmll_end_idx: int,
        block: Union[RouxOrientation, str, Any] = ALL_BLOCK_DEFINITIONS[""]
    ) -> Optional[Tuple[LSEPhase, CubeState]]:
        """Segments LSE into 4a (EO/EOLR), 4b (UL/UR), and 4c (M-slice permutation) starting from cmll_state.
        
        Returns:
            Tuple of (LSEPhase, final_cube_state) or None.
        """
        ori = get_orientation(block)
        sim = cmll_state.copy()
        initial_bad_edges = ori.count_bad_edges(sim)

        step_4a: Optional[Step4a] = None
        step_4b: Optional[Step4b] = None
        step_4c: Optional[Step4c] = None

        idx_4a_end: Optional[int] = None
        idx_4b_end: Optional[int] = None
        sim_at_4b: Optional[CubeState] = None

        # Check initial skips entering LSE at cmll_end_idx
        if ori.is_eo_solved(sim):
            idx_4a_end = cmll_end_idx
            variant = cls.classify_4a_variant(sim, ori)
            center_state = "axis_aligned" if ori.is_center_axis_aligned(sim) else "misaligned"
            time_4a = events[cmll_end_idx].timestamp_ms if cmll_end_idx >= 0 and cmll_end_idx < len(events) and events[cmll_end_idx].timestamp_ms is not None else None
            step_4a = Step4a(
                start_move_idx=cmll_end_idx + 1,
                end_move_idx=cmll_end_idx,
                move_count_stm=0,
                time_ms=time_4a,
                variant=variant,
                initial_bad_edges=0,
                center_state=center_state,
                moves_str=""
            )

        if step_4a is not None and step_4a.variant == "eolr_b":
            idx_4b_end = cmll_end_idx
            sim_at_4b = sim.copy()
            step_4b = Step4b(
                start_move_idx=cmll_end_idx + 1,
                end_move_idx=cmll_end_idx,
                move_count_stm=0,
                time_ms=step_4a.time_ms,
                skipped=True,
                moves_str=""
            )

        if step_4b is not None and sim_at_4b is not None and cls.is_fully_solved(sim, ori):
            step_4c = Step4c(
                start_move_idx=cmll_end_idx + 1,
                end_move_idx=cmll_end_idx,
                move_count_stm=0,
                time_ms=step_4b.time_ms,
                case="solved",
                moves_str=""
            )
            phase = LSEPhase(
                start_move_idx=cmll_end_idx + 1,
                end_move_idx=cmll_end_idx,
                move_count_stm=0,
                time_ms=step_4b.time_ms,
                step_4a=step_4a,
                step_4b=step_4b,
                step_4c=step_4c,
                moves_str=""
            )
            return phase, sim.copy()

        # Iterate moves through LSE
        for i in range(cmll_end_idx + 1, len(events)):
            ev = events[i]
            sim.apply_move(ev.move)
            current_time = ev.timestamp_ms

            # 1. Detect 4a Completion
            if step_4a is None and ori.is_eo_solved(sim):
                idx_4a_end = i
                events_4a = events[cmll_end_idx + 1:i + 1]
                moves_4a = len([e for e in events_4a if not e.move.startswith(('x', 'y', 'z'))])
                time_4a = current_time if current_time is not None else (events_4a[-1].timestamp_ms if events_4a and events_4a[-1].timestamp_ms is not None else None)
                variant = cls.classify_4a_variant(sim, ori)
                center_state = "axis_aligned" if ori.is_center_axis_aligned(sim) else "misaligned"

                step_4a = Step4a(
                    start_move_idx=cmll_end_idx + 1,
                    end_move_idx=i,
                    move_count_stm=moves_4a,
                    time_ms=time_4a,
                    variant=variant,
                    initial_bad_edges=initial_bad_edges,
                    center_state=center_state,
                    moves_str=" ".join(e.move for e in events_4a)
                )

            # 2. Detect 4b Completion
            if step_4a is not None and step_4b is None and idx_4a_end is not None:
                if step_4a.variant == "eolr_b":
                    idx_4b_end = idx_4a_end
                    sim_at_4b = sim.copy()
                    step_4b = Step4b(
                        start_move_idx=idx_4a_end + 1,
                        end_move_idx=idx_4a_end,
                        move_count_stm=0,
                        time_ms=step_4a.time_ms,
                        skipped=True,
                        moves_str=""
                    )
                elif ori.is_ul_ur_solved(sim):
                    idx_4b_end = i
                    sim_at_4b = sim.copy()
                    events_4b = events[idx_4a_end + 1:i + 1]
                    moves_4b = len([e for e in events_4b if not e.move.startswith(('x', 'y', 'z'))])
                    time_4b = current_time if current_time is not None else (events_4b[-1].timestamp_ms if events_4b and events_4b[-1].timestamp_ms is not None else None)

                    step_4b = Step4b(
                        start_move_idx=idx_4a_end + 1,
                        end_move_idx=i,
                        move_count_stm=moves_4b,
                        time_ms=time_4b,
                        skipped=False,
                        moves_str=" ".join(e.move for e in events_4b)
                    )

            # 3. Detect 4c Completion (Full Solve)
            if step_4b is not None and step_4c is None and idx_4b_end is not None and sim_at_4b is not None:
                if cls.is_fully_solved(sim, ori):
                    events_4c = events[idx_4b_end + 1:i + 1]
                    moves_4c = len([e for e in events_4c if not e.move.startswith(('x', 'y', 'z'))])
                    time_4c = current_time if current_time is not None else (events_4c[-1].timestamp_ms if events_4c and events_4c[-1].timestamp_ms is not None else None)

                    case_4c = cls.classify_4c_case(sim_at_4b, ori)

                    step_4c = Step4c(
                        start_move_idx=idx_4b_end + 1,
                        end_move_idx=i,
                        move_count_stm=moves_4c,
                        time_ms=time_4c,
                        case=case_4c,
                        moves_str=" ".join(e.move for e in events_4c)
                    )

                    lse_events = events[cmll_end_idx + 1:i + 1]
                    lse_moves_stm = len([e for e in lse_events if not e.move.startswith(('x', 'y', 'z'))])
                    lse_time = lse_events[-1].timestamp_ms if lse_events and lse_events[-1].timestamp_ms is not None else None

                    phase = LSEPhase(
                        start_move_idx=cmll_end_idx + 1,
                        end_move_idx=i,
                        move_count_stm=lse_moves_stm,
                        time_ms=lse_time,
                        step_4a=step_4a,
                        step_4b=step_4b,
                        step_4c=step_4c,
                        moves_str=" ".join(e.move for e in lse_events)
                    )
                    return phase, sim.copy()

        return None


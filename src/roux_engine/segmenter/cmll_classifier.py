"""CMLL Classifier for 42 Cases, AUF isolation, and execution tracking."""

from __future__ import annotations
from typing import List, Optional, Tuple, Sequence, Dict, Set
import numpy as np

from ..core.constants import Corner
from ..core.cube import CubeState
from ..core.parser import MoveEvent, MoveParser
from .models import CMLLPhase
from .fb_detector import BlockDefinition, ALL_BLOCK_DEFINITIONS, FULL_COLOR_NEUTRAL_ORIENTATIONS


CMLL_ALGS: List[Tuple[str, str, str]] = [
    ("solved", "Skip", ""),
    ("o_adjacent_swap", "O", "R U R' F' R U R' U' R' F R2 U' R'"),
    ("o_diagonal_swap", "O", "F R U' R' U' R U R' F' R U R' U' R' F R F'"),
    ("h_columns", "H", "U' R U R' U R U' R' U R U2 R'"),
    ("h_rows", "H", "F R U R' U' R U R' U' R U R' U' F'"),
    ("h_column", "H", "U' R U2' R2' F R F' U2 R' F R F'"),
    ("h_row", "H", "r U' r2' D' r U' r' D r2 U r'"),
    ("pi_right_bar", "Pi", "F R U R' U' R U R' U' F'"),
    ("pi_back_slash", "Pi", "U F R' F' R U2 R U' R' U R U2' R'"),
    ("pi_x_checkerboard", "Pi", "U' R' F R U F U' R U R' U' F'"),
    ("pi_forward_slash", "Pi", "R U2 R' U' R U R' U2' R' F R F'"),
    ("pi_columns", "Pi", "U' r U' r2' D' r U r' D r2 U r'"),
    ("pi_left_bar", "Pi", "U' R' U' R' F R F' R U' R' U2 R"),
    ("u_forward_slash", "U", "U2 R2 D R' U2 R D' R' U2 R'"),
    ("u_back_slash", "U", "R2' D' R U2 R' D R U2 R"),
    ("u_front_row", "U", "R' U' R U' R' U2 R2 U R' U R U2 R'"),
    ("u_rows", "U", "U' F R2 D R' U R D' R2' U' F'"),
    ("u_x_checkerboard", "U", "U2 r U' r' U r' D' r U' r' D r"),
    ("u_back_row", "U", "U' F R U R' U' F'"),
    ("t_left_bar", "T", "U' R U R' U' R' F R F'"),
    ("t_right_bar", "T", "U L' U' L U L F' L' F"),
    ("t_rows", "T", "R U2 R' U' R U' R2' U2' R U R' U R"),
    ("t_front_row", "T", "r' U r U2' R2' F R F' R"),
    ("t_back_row", "T", "r' D' r U r' D r U' r U r'"),
    ("t_columns", "T", "U2 r2' D' r U r' D r2 U' r' U' r"),
    ("s_left_bar", "Sune", "R U R' U R U2 R'"),
    ("s_x_checkerboard", "Sune", "L' U2 L U2' L F' L' F"),
    ("s_forward_slash", "Sune", "F R' F' R U2 R U2' R'"),
    ("s_columns", "Sune", "R U R' U' R' F R F' R U R' U R U2' R'"),
    ("s_right_bar", "Sune", "U2' R U R' U R' F R F' R U2' R'"),
    ("s_back_slash", "Sune", "R U' L' U R' U' L"),
    ("as_right_bar", "Antisune", "U' R U2' R' U' R U' R'"),
    ("as_columns", "Antisune", "R2 D R' U R D' R' U R' U' R U' R'"),
    ("as_back_slash", "Antisune", "F' r U r' U2' r' F2 r"),
    ("as_x_checkerboard", "Antisune", "R U2' R' U2' R' F R F'"),
    ("as_forward_slash", "Antisune", "L' U R U' L U R'"),
    ("as_left_bar", "Antisune", "U2' R U2' R' F R' F' R U' R U' R'"),
    ("l_mirror", "L", "F R U' R' U' R U R' F'"),
    ("l_inverse", "L", "F R' F' R U R U' R'"),
    ("l_pure", "L", "U2 R U R' U R U' R' U R U' R' U R U2' R'"),
    ("l_front_commutator", "L", "R U2 R D R' U2 R D' R2'"),
    ("l_diag", "L", "U2 R' U' R U R' F' R U R' U' R' F R2"),
    ("l_back_commutator", "L", "U' R' U2 R' D' R U2 R' D R2"),
]

CASE_TO_ALG: Dict[str, Tuple[str, str]] = {
    case_id: (group, alg) for case_id, group, alg in CMLL_ALGS
}


def _precompute_solved_corner_perms() -> Tuple[Dict[str, Set[Tuple[int, ...]]], Dict[str, Tuple[int, ...]]]:
    """Precomputes valid solved corner permutations and expected orientations for all 24 orientations."""
    valid_perms: Dict[str, Set[Tuple[int, ...]]] = {}
    expected_co: Dict[str, Tuple[int, ...]] = {}

    for ori_name in FULL_COLOR_NEUTRAL_ORIENTATIONS:
        perms = set()
        for auf in ("", "U", "U2", "U'"):
            c = CubeState()
            if ori_name:
                c.apply_moves(ori_name)
            if auf:
                c.apply_move(auf)
            perms.add(tuple(int(x) for x in c.cp[0:4]))
        valid_perms[ori_name] = perms

        c0 = CubeState()
        if ori_name:
            c0.apply_moves(ori_name)
        expected_co[ori_name] = tuple(int(x) for x in c0.co[0:4])

    return valid_perms, expected_co


VALID_SOLVED_CORNER_PERMS, EXPECTED_SOLVED_CO = _precompute_solved_corner_perms()


class CMLLClassifier:
    """Classifies 42 CMLL cases, isolates AUF, and tracks execution."""

    _TABLE: Optional[Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], Tuple[str, str, str]]] = None

    @classmethod
    def _build_table(cls) -> Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], Tuple[str, str, str]]:
        table: Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], Tuple[str, str, str]] = {}
        auf_moves = [("", ""), ("U", "U'"), ("U2", "U2"), ("U'", "U")]

        for ori_name in FULL_COLOR_NEUTRAL_ORIENTATIONS:
            for case_id, group, alg in CMLL_ALGS:
                if not alg:
                    # Solved case
                    c_base = CubeState()
                    if ori_name:
                        c_base.apply_moves(ori_name)
                    for auf_applied, _ in auf_moves:
                        c = c_base.copy()
                        if auf_applied:
                            c.apply_move(auf_applied)
                        key = (ori_name, tuple(int(x) for x in c.co[0:4]), tuple(int(x) for x in c.cp[0:4]))
                        table[key] = ("solved", "Skip", "")
                    continue

                inv_alg = MoveParser.invert_moves(alg)

                for post_auf_applied, _ in auf_moves:
                    c_base = CubeState()
                    if ori_name:
                        c_base.apply_moves(ori_name)
                    if post_auf_applied:
                        c_base.apply_move(post_auf_applied)
                    c_base.apply_moves(inv_alg)

                    for pre_auf_applied, pre_auf_needed in auf_moves:
                        c = c_base.copy()
                        if pre_auf_applied:
                            c.apply_move(pre_auf_applied)

                        co_key = tuple(int(x) for x in c.co[0:4])
                        cp_key = tuple(int(x) for x in c.cp[0:4])
                        key = (ori_name, co_key, cp_key)

                        if key not in table:
                            table[key] = (case_id, group, pre_auf_needed)

        return table

    @classmethod
    def get_table(cls) -> Dict[Tuple[str, Tuple[int, ...], Tuple[int, ...]], Tuple[str, str, str]]:
        if cls._TABLE is None:
            cls._TABLE = cls._build_table()
        return cls._TABLE

    @staticmethod
    def are_corners_solved(state: CubeState, block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]) -> bool:
        """Checks if all 4 U corners are oriented and permuted correctly relative to each other (up to AUF)."""
        ori = block.rotations
        valid_perms = VALID_SOLVED_CORNER_PERMS.get(ori, VALID_SOLVED_CORNER_PERMS[""])
        expected_co = EXPECTED_SOLVED_CO.get(ori, (0, 0, 0, 0))

        co_tuple = tuple(int(x) for x in state.co[0:4])
        cp_tuple = tuple(int(x) for x in state.cp[0:4])

        return co_tuple == expected_co and (cp_tuple in valid_perms)

    @classmethod
    def classify_state(
        cls,
        state: CubeState,
        block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]
    ) -> Tuple[str, str, str]:
        """Classifies the CMLL case from the current state and orientation basis.
        
        Returns:
            Tuple of (case_id, group, pre_auf)
        """
        table = cls.get_table()
        ori = block.rotations

        co_key = tuple(int(x) for x in state.co[0:4])
        cp_key = tuple(int(x) for x in state.cp[0:4])
        
        key = (ori, co_key, cp_key)
        if key in table:
            return table[key]

        if cls.are_corners_solved(state, block):
            return ("solved", "Skip", "")

        return ("unknown", "Unknown", "")

    @staticmethod
    def check_is_standard_alg(executed_moves: Sequence[MoveEvent], case_id: str) -> bool:
        """Verifies if the executed moves (ignoring rotations) match the standard algorithm for the case."""
        if case_id == "solved":
            return len(executed_moves) == 0

        if case_id not in CASE_TO_ALG:
            return False

        _, std_alg = CASE_TO_ALG[case_id]
        if not std_alg:
            return False

        std_moves = [e.move for e in MoveParser.parse_string(std_alg) if not e.move.startswith(('x', 'y', 'z'))]
        exec_moves = [e.move for e in executed_moves if not e.move.startswith(('x', 'y', 'z'))]

        # 1. Direct match with standard algorithm tokens
        if exec_moves == std_moves:
            return True

        # 2. Match after stripping leading AUF from standard algorithm
        if std_moves and std_moves[0].startswith('U'):
            std_no_auf = std_moves[1:]
            if exec_moves == std_no_auf:
                return True

        # 3. Match after stripping leading AUF from executed moves
        if exec_moves and exec_moves[0].startswith('U'):
            exec_no_auf = exec_moves[1:]
            if exec_no_auf == std_moves:
                return True

        return False

    @classmethod
    def detect_cmll(
        cls,
        sb_state: CubeState,
        events: Sequence[MoveEvent],
        sb_end_idx: int,
        block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]
    ) -> Optional[Tuple[CMLLPhase, CubeState]]:
        """Segments CMLL execution starting directly from sb_state at sb_end_idx until corners are solved.
        
        Returns:
            Tuple of (CMLLPhase, cube_state_at_cmll_end) or None.
        """
        sim = sb_state.copy()

        # Check if CMLL was already solved entering the phase
        if cls.are_corners_solved(sim, block):
            time_ms = events[sb_end_idx].timestamp_ms if sb_end_idx >= 0 and sb_end_idx < len(events) and events[sb_end_idx].timestamp_ms is not None else None
            phase = CMLLPhase(
                start_move_idx=sb_end_idx + 1,
                end_move_idx=sb_end_idx,
                move_count_stm=0,
                time_ms=time_ms,
                case_id="solved",
                group="Skip",
                pre_auf="",
                is_standard_alg=True,
                moves_str=""
            )
            return phase, sim.copy()

        # Identify CMLL case before execution
        case_id, group, expected_pre_auf = cls.classify_state(sim, block)

        for i in range(sb_end_idx + 1, len(events)):
            ev = events[i]
            sim.apply_move(ev.move)

            if cls.are_corners_solved(sim, block):
                cmll_events = events[sb_end_idx + 1:i + 1]
                moves_stm = len([e for e in cmll_events if not e.move.startswith(('x', 'y', 'z'))])
                time_ms = cmll_events[-1].timestamp_ms if cmll_events and cmll_events[-1].timestamp_ms is not None else None

                # Isolate executed pre_auf (all leading U turns executed before the main alg)
                leading_u_count = 0
                while leading_u_count < len(cmll_events) and cmll_events[leading_u_count].move.startswith('U'):
                    leading_u_count += 1

                pre_auf_str = " ".join(e.move for e in cmll_events[:leading_u_count])
                alg_events = cmll_events[leading_u_count:]

                is_std = cls.check_is_standard_alg(alg_events if leading_u_count > 0 else cmll_events, case_id)

                phase = CMLLPhase(
                    start_move_idx=sb_end_idx + 1,
                    end_move_idx=i,
                    move_count_stm=moves_stm,
                    time_ms=time_ms,
                    case_id=case_id,
                    group=group,
                    pre_auf=pre_auf_str,
                    is_standard_alg=is_std,
                    moves_str=" ".join(e.move for e in cmll_events)
                )
                return phase, sim.copy()

        return None

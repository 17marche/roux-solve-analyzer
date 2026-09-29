"""Second Block (SB) Center-Aligned Free Blockbuilding IDA* Solver with Symmetry Re-mapping.

Uses the 1.08M state Second Block Pattern Database (PDB) to find top-K shortest
candidate paths directly to full Second Block in the rotationless <R, U, r, M> generator.
Enforces Center-Aligned SB goal condition and dual-neutral symmetry re-mapping.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional, Union, Any, Literal

from .sb_pdb_generator import SB_MOVESET


# -----------------------------------------------------------------------------
# SB Move Families & Branch Pruning Rules
# -----------------------------------------------------------------------------

# SB_MOVESET:
#  0: R,  1: R2,  2: R'   (Family 0: R)
#  3: U,  4: U2,  5: U'   (Family 1: U)
#  6: r,  7: r2,  8: r'   (Family 2: r)
#  9: M, 10: M2, 11: M'   (Family 3: M)

_SB_MOVE_FAMILIES: Tuple[int, ...] = (
    0, 0, 0,  # R
    1, 1, 1,  # U
    2, 2, 2,  # r
    3, 3, 3,  # M
)

_SB_MOVE_INDEX: dict[str, int] = {m: i for i, m in enumerate(SB_MOVESET)}
_PREV_M_AFTER_HEDGE: int = -2
_ALLOWED_AFTER_HEDGE: Tuple[int, ...] = tuple(i for i, m in enumerate(SB_MOVESET) if m != "R'")


def _build_sb_allowed_moves() -> Tuple[Tuple[int, ...], ...]:
    """Precomputes allowed subsequent move indices for SB branch pruning:
    - Same-family consecutive turn pruning: forbid adjacent turns within R, U, r, or M.
    - Parallel/slice group canonical ordering in {R, r, M}:
      allow R followed by M; prune all other adjacent combinations within {R, r, M}.
    """
    allowed: List[Tuple[int, ...]] = []
    for m1 in range(len(SB_MOVESET)):
        f1 = _SB_MOVE_FAMILIES[m1]
        allowed_m2: List[int] = []
        for m2 in range(len(SB_MOVESET)):
            f2 = _SB_MOVE_FAMILIES[m2]
            # 1. Same-family turn is always pruned (e.g. R R2, U U', r r2, M M')
            if f1 == f2:
                continue
            # 2. Parallel/slice group canonical ordering:
            # Families 0 (R), 2 (r), 3 (M) are parallel/overlapping.
            # We only allow R (0) followed by M (3). All other adjacent pairs in {R, r, M} are redundant.
            if f1 in (0, 2, 3) and f2 in (0, 2, 3):
                if not (f1 == 0 and f2 == 3):
                    continue
            allowed_m2.append(m2)
        allowed.append(tuple(allowed_m2))
    return tuple(allowed)


# -----------------------------------------------------------------------------
# Center Offset Tracking & Transitions
# -----------------------------------------------------------------------------

# M-slice center shift delta mod 4 for each move in SB_MOVESET:
# R(0), R2(0), R'(0), U(0), U2(0), U'(0) -> 0
# r(+3), r2(+2), r'(+1)                  -> r = R * M'
# M(+1), M2(+2), M'(+3)                  -> M cycles U->F->D->B
_MOVE_CENTER_DELTAS: Tuple[int, ...] = (
    0, 0, 0,  # R
    0, 0, 0,  # U
    3, 2, 1,  # r
    1, 2, 3,  # M
)


def _build_center_transitions() -> Tuple[Tuple[int, ...], ...]:
    """Precomputes center offset transitions (shape 4 x 12).
    Offset 0: aligned, 1: M, 2: M2, 3: M'.
    """
    return tuple(
        tuple((offset + _MOVE_CENTER_DELTAS[m]) % 4 for m in range(len(SB_MOVESET)))
        for offset in range(4)
    )


def _build_dr_transitions() -> Tuple[Tuple[int, ...], ...]:
    """Precomputes DR transition table for all 24 states (12 slots x 2 orientations) across 12 SB moves.
    State: slot * 2 + eo. Solved canonical DR is slot 7 (Edge.DR) with eo 0: 7 * 2 + 0 = 14.
    """
    moves_inv_cp, moves_co_ori, moves_inv_ep, moves_eo_ori = _precompute_move_inverses(SB_MOVESET)
    table = []
    for slot in range(12):
        for eo in range(2):
            row = []
            for m in range(len(SB_MOVESET)):
                new_slot = moves_inv_ep[m][slot]
                new_eo = (eo + moves_eo_ori[m][new_slot]) % 2
                row.append(new_slot * 2 + new_eo)
            table.append(tuple(row))
    return tuple(table)


# -----------------------------------------------------------------------------
# Public Dataclass & Goal Evaluation
# -----------------------------------------------------------------------------

from ..core.constants import Color, Center
from ..core.cube import CubeState
from ..core.orientation import (
    RouxOrientation,
    SBPlacement,
    get_orientation,
    get_dual_neutral_orientations,
    get_all_orientations,
)



@dataclass(frozen=True)
class SBSolution:
    """Immutable solution representation for Second Block (SB)."""
    moves: tuple[str, ...]
    move_count: int
    style: str = "free"
    order: str = "direct"
    dr_move_idx: Optional[int] = None
    pair1_move_idx: Optional[int] = None
    square_move_idx: Optional[int] = None
    resulting_cmll_case: str = "Skip"
    orientation: str = ""
    e_stm: Optional[float] = None


# -----------------------------------------------------------------------------
# IDA* Search Engine
# -----------------------------------------------------------------------------

import time
from typing import Dict, Set
from ..core.parser import MoveParser
from ..segmenter.cmll_classifier import CMLLClassifier
from .sb_indexer import (
    SBIndexer,
    NUM_SB_CORNER_CONFIGS,
    NUM_SB_EDGE_CONFIGS,
    RightBackSquareIndexer,
    NUM_RBS_CORNER_CONFIGS,
    RightFrontSquareIndexer,
    NUM_RFS_CORNER_CONFIGS,
)
from .sb_pdb import SBPDB, RightBackSquarePDB, RightFrontSquarePDB
from .sb_pdb_generator import (
    build_sb_transition_tables,
    build_rbs_transition_tables,
    build_rfs_transition_tables,
    build_macro_sb_transition_tables,
    _precompute_move_inverses,
)
from ..ergonomics.macro_triggers import (
    MacroTrigger,
    SLEDGEHAMMER,
    HEDGE,
)
from .symmetry import (
    CanonicalSymmetry,
    get_symmetry,
    get_all_symmetries,
    is_fb_solved_for_symmetry,
    translate_moves_to_original,
)


class SBSolver:
    """High-performance Top-K Candidate Search IDA* Second Block solver."""

    _instance: Optional[SBSolver] = None

    def __init__(
        self,
        pdb: Optional[SBPDB] = None,
        rbs_pdb: Optional[RightBackSquarePDB] = None,
        rfs_pdb: Optional[RightFrontSquarePDB] = None,
    ) -> None:
        self.pdb: SBPDB = pdb if pdb is not None else SBPDB()
        self.rbs_pdb: RightBackSquarePDB = rbs_pdb if rbs_pdb is not None else RightBackSquarePDB()
        self.rfs_pdb: RightFrontSquarePDB = rfs_pdb if rfs_pdb is not None else RightFrontSquarePDB()
        self._get_distance = self.pdb.get_distance_by_index
        self._get_rbs_distance = self.rbs_pdb.get_distance_by_index
        self._get_rfs_distance = self.rfs_pdb.get_distance_by_index

        c_arr, e_arr = build_sb_transition_tables(SB_MOVESET)
        self.corner_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in c_arr)
        self.edge_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in e_arr)
        self.allowed_moves: Tuple[Tuple[int, ...], ...] = _build_sb_allowed_moves()
        self.center_trans: Tuple[Tuple[int, ...], ...] = _build_center_transitions()

        rbs_c_arr, rbs_e_arr = build_rbs_transition_tables(SB_MOVESET)
        self.rbs_corner_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in rbs_c_arr)
        self.rbs_edge_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in rbs_e_arr)

        rfs_c_arr, rfs_e_arr = build_rfs_transition_tables(SB_MOVESET)
        self.rfs_corner_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in rfs_c_arr)
        self.rfs_edge_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in rfs_e_arr)
        self.dr_trans: Tuple[Tuple[int, ...], ...] = _build_dr_transitions()

        self.macro_triggers: Tuple[MacroTrigger, ...] = (SLEDGEHAMMER, HEDGE)
        self.macro_corner_trans, self.macro_edge_trans = build_macro_sb_transition_tables(
            [t.moves for t in self.macro_triggers]
        )
        self._flow_scorer: Optional[Any] = None

    @property
    def flow_scorer(self) -> Any:
        """Lazy-loaded FlowScorer for ergonomic E-STM ranking."""
        if self._flow_scorer is None:
            from ..ergonomics.flow_scorer import FlowScorer
            self._flow_scorer = FlowScorer()
        return self._flow_scorer

    @classmethod
    def get_instance(cls) -> SBSolver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def solve_sb(
        cls,
        scramble_or_cube: Union[str, CubeState],
        k: int = 5,
        top_k: Optional[int] = None,
        style: str = "all",
        order: str = "best",
        orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
        timeout_ms: Optional[float] = None,
        allow_macro_triggers: bool = False,
        rank_by: Literal["stm", "e_stm"] = "stm",
    ) -> List[SBSolution]:
        """Convenience class method forwarding to get_instance().solve."""
        return cls.get_instance().solve(
            scramble_or_cube=scramble_or_cube,
            k=k,
            top_k=top_k,
            style=style,
            order=order,
            orientation=orientation,
            timeout_ms=timeout_ms,
            allow_macro_triggers=allow_macro_triggers,
            rank_by=rank_by,
        )

    def _heuristic(self, state_idx: int, center_off: int) -> int:
        """Returns exact PDB heuristic distance with Center-Aligned SB penalty."""
        h = self._get_distance(state_idx)
        if center_off % 2 != 0 and h == 0:
            return 1
        return h

    def _ida_search(
        self,
        c_idx: int,
        e_idx: int,
        center_off: int,
        g: int,
        max_depth: int,
        prev_m: int,
        path: List[str],
        solutions: List[Tuple[str, ...]],
        needed: int,
        deadline: Optional[float],
        allow_macro_triggers: bool = False,
        macros_used: int = 0,
    ) -> None:
        """Recursive depth-bounded search with exact PDB pruning and center alignment."""
        if deadline is not None and time.perf_counter() > deadline:
            return

        state_idx = e_idx * NUM_SB_CORNER_CONFIGS + c_idx
        h_pure = self._heuristic(state_idx, center_off)
        h_eff = h_pure if (not allow_macro_triggers or macros_used >= 1) else max(0, h_pure - 4)

        if g + h_eff > max_depth:
            return

        if h_pure == 0:
            if g == max_depth:
                solutions.append(tuple(path))
            return

        next_moves: Sequence[int]
        if prev_m == -1:
            next_moves = range(len(SB_MOVESET))
        elif prev_m == _PREV_M_AFTER_HEDGE:
            next_moves = _ALLOWED_AFTER_HEDGE
        else:
            next_moves = self.allowed_moves[prev_m]

        for m in next_moves:
            nc = self.corner_trans[c_idx][m]
            ne = self.edge_trans[e_idx][m]
            ncent = self.center_trans[center_off][m]
            path.append(SB_MOVESET[m])
            self._ida_search(
                nc, ne, ncent, g + 1, max_depth, m, path, solutions, needed, deadline,
                allow_macro_triggers=allow_macro_triggers,
                macros_used=macros_used,
            )
            path.pop()
            if len(solutions) >= needed:
                return

        if allow_macro_triggers and macros_used < 1 and g + 4 <= max_depth:
            for macro_idx, trig in enumerate(self.macro_triggers):
                # Branch pruning: Sledgehammer starts with R'.
                # Prune if preceded by R/r turns or Hedge (which ends in R).
                if trig.name == "sledgehammer" and (
                    prev_m == _PREV_M_AFTER_HEDGE
                    or (prev_m != -1 and _SB_MOVE_FAMILIES[prev_m] in (0, 2))
                ):
                    continue
                nc = self.macro_corner_trans[c_idx][macro_idx]
                ne = self.macro_edge_trans[e_idx][macro_idx]
                path.extend(trig.moves)
                # Next prev_m: Sledgehammer ends in F' -> not in SB_MOVESET (-1)
                # Hedge ends in R -> _PREV_M_AFTER_HEDGE (forbids immediate R' cancellation)
                next_pm = _PREV_M_AFTER_HEDGE if trig.name == "hedge" else -1
                self._ida_search(
                    nc, ne, center_off, g + 4, max_depth, next_pm, path, solutions, needed, deadline,
                    allow_macro_triggers=allow_macro_triggers,
                    macros_used=macros_used + 1,
                )
                del path[-4:]
                if len(solutions) >= needed:
                    return

    def _build_solution(
        self,
        base_cube: CubeState,
        moves: Tuple[str, ...],
        ori: RouxOrientation,
        orientation: str,
        uninspected: bool,
        sym: CanonicalSymmetry,
        style: str = "free",
        order: str = "direct",
    ) -> SBSolution:
        """Builds a structured SBSolution and previews the resulting CMLL case."""
        # Detect milestones on the solved cube
        c_eval = base_cube.copy()
        if moves:
            c_eval.apply_moves(" ".join(moves))

        case_id, group, pre_auf = CMLLClassifier.classify_state(c_eval, ori)
        cmll_case = "Skip" if case_id == "solved" else case_id

        # If input was in uninspected frame, transform moves back to original user frame
        final_moves = moves
        if uninspected and sym.value:
            translated = translate_moves_to_original(moves, sym)
            final_moves = tuple(translated)

        # Milestone sub-phase tracking
        dr_move_idx: Optional[int] = None
        pair1_move_idx: Optional[int] = None
        square_move_idx: Optional[int] = None

        sim = base_cube.copy()
        if ori.is_dr_solved(sim):
            dr_move_idx = 0
        back_init = ori.is_back_pair_solved(sim)
        front_init = ori.is_front_pair_solved(sim)
        if back_init or front_init:
            pair1_move_idx = 0
        if dr_move_idx == 0 and pair1_move_idx == 0:
            square_move_idx = 0

        for idx, m_token in enumerate(moves):
            sim.apply_move(m_token)
            dr_ok = ori.is_dr_solved(sim)
            back_ok = ori.is_back_pair_solved(sim)
            front_ok = ori.is_front_pair_solved(sim)
            pair1_ok = back_ok or front_ok
            square_ok = dr_ok and pair1_ok

            if dr_move_idx is None and dr_ok:
                dr_move_idx = idx
            if pair1_move_idx is None and pair1_ok:
                pair1_move_idx = idx
            if square_move_idx is None and square_ok:
                square_move_idx = idx

        return SBSolution(
            moves=final_moves,
            move_count=len(final_moves),
            style=style,
            order=order,
            dr_move_idx=dr_move_idx,
            pair1_move_idx=pair1_move_idx,
            square_move_idx=square_move_idx,
            resulting_cmll_case=cmll_case,
            orientation=orientation,
        )

    def _advance_sb_state(
        self,
        c_idx: int,
        e_idx: int,
        center_off: int,
        moves: Sequence[str],
    ) -> Tuple[int, int, int]:
        """Simulates moves from given SB state indices returning new (c_idx, e_idx, center_off)."""
        c, e, cent = c_idx, e_idx, center_off
        for m_name in moves:
            m = _SB_MOVE_INDEX[m_name]
            c = self.corner_trans[c][m]
            e = self.edge_trans[e][m]
            cent = self.center_trans[cent][m]
        return c, e, cent

    def _search_square(
        self,
        square_type: str,  # "back" or "front"
        c_idx: int,
        e_idx: int,
        k: int = 3,
        prev_m: int = -1,
        deadline: Optional[float] = None,
    ) -> List[Tuple[str, ...]]:
        """Searches for shortest path(s) to solved 1x2x2 square (back or front) using Square PDB."""
        if square_type == "back":
            num_c = NUM_RBS_CORNER_CONFIGS
            pdb_get = self._get_rbs_distance
            c_trans = self.rbs_corner_trans
            e_trans = self.rbs_edge_trans
        else:
            num_c = NUM_RFS_CORNER_CONFIGS
            pdb_get = self._get_rfs_distance
            c_trans = self.rfs_corner_trans
            e_trans = self.rfs_edge_trans

        state_idx = e_idx * num_c + c_idx
        h0 = pdb_get(state_idx)
        if h0 == 0:
            return [()]

        sols: List[Tuple[str, ...]] = []
        depth = h0

        def _dfs(c: int, e: int, g: int, max_d: int, pm: int, path: List[str]) -> None:
            if deadline is not None and time.perf_counter() > deadline:
                return
            h = pdb_get(e * num_c + c)
            if g + h > max_d:
                return
            if h == 0:
                if g == max_d:
                    sols.append(tuple(path))
                return
            next_moves = range(len(SB_MOVESET)) if pm == -1 else self.allowed_moves[pm]
            for m in next_moves:
                nc = c_trans[c][m]
                ne = e_trans[e][m]
                path.append(SB_MOVESET[m])
                _dfs(nc, ne, g + 1, max_d, m, path)
                path.pop()
                if len(sols) >= k:
                    return

        while len(sols) < k and depth <= 12:
            if deadline is not None and time.perf_counter() > deadline:
                break
            _dfs(c_idx, e_idx, 0, depth, prev_m, [])
            depth += 1

        return sols[:k]

    def _solve_square_pair_order(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        placement: SBPlacement,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int,
        order: str,
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Solves SB using Square + Pair paradigm for a specific pair order."""
        if order == "back_first":
            sq_c = RightBackSquareIndexer.encode_corner(placement.dbr_slot, placement.dbr_co)
            sq_e = RightBackSquareIndexer.encode_edges(placement.dr_slot, placement.dr_eo, placement.br_slot, placement.br_eo)
            stage1_paths = self._search_square("back", sq_c, sq_e, k=min(k, 3), deadline=deadline)
        else:
            sq_c = RightFrontSquareIndexer.encode_corner(placement.dfr_slot, placement.dfr_co)
            sq_e = RightFrontSquareIndexer.encode_edges(placement.dr_slot, placement.dr_eo, placement.fr_slot, placement.fr_eo)
            stage1_paths = self._search_square("front", sq_c, sq_e, k=min(k, 3), deadline=deadline)

        if not stage1_paths:
            return []

        solutions: List[SBSolution] = []
        seen_moves: Set[Tuple[str, ...]] = set()

        for p1 in stage1_paths:
            if deadline is not None and time.perf_counter() > deadline:
                break
            c2, e2, cent2 = self._advance_sb_state(c_idx, e_idx, center_off, p1)
            prev_m = _SB_MOVE_INDEX[p1[-1]] if p1 else -1
            state_idx = e2 * NUM_SB_CORNER_CONFIGS + c2
            h2 = self._heuristic(state_idx, cent2)

            if h2 == 0:
                combined = p1
                if combined not in seen_moves:
                    seen_moves.add(combined)
                    solutions.append(
                        self._build_solution(
                            base_cube=base_cube,
                            moves=combined,
                            ori=ori,
                            orientation=ori_str,
                            uninspected=uninspected,
                            sym=sym,
                            style="square_pair",
                            order=order,
                        )
                    )
            else:
                depth = max(0, h2 - 4) if allow_macro_triggers else h2
                stage2_sols: List[Tuple[str, ...]] = []
                while len(stage2_sols) < k and depth <= 15:
                    if deadline is not None and time.perf_counter() > deadline:
                        break
                    needed = k - len(stage2_sols)
                    depth_sols: List[Tuple[str, ...]] = []
                    self._ida_search(
                        c2, e2, cent2, 0, depth, prev_m, [], depth_sols, needed, deadline,
                        allow_macro_triggers=allow_macro_triggers,
                    )
                    for p2 in depth_sols:
                        if p2 not in stage2_sols:
                            stage2_sols.append(p2)
                    depth += 1

                for p2 in stage2_sols:
                    combined = p1 + p2
                    if combined not in seen_moves:
                        seen_moves.add(combined)
                        solutions.append(
                            self._build_solution(
                                base_cube=base_cube,
                                moves=combined,
                                ori=ori,
                                orientation=ori_str,
                                uninspected=uninspected,
                                sym=sym,
                                style="square_pair",
                                order=order,
                            )
                        )

        solutions.sort(key=lambda s: s.move_count)
        return solutions[:k]

    def _solve_square_pair(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        placement: SBPlacement,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int = 5,
        order: str = "best",
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Solves SB using Square + Pair paradigm (best, back_first, or front_first)."""
        if order in ("back_first", "front_first"):
            return self._solve_square_pair_order(
                base_cube, ori, sym, uninspected, ori_str,
                placement, c_idx, e_idx, center_off, k, order, deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
        sols_back = self._solve_square_pair_order(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, "back_first", deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        sols_front = self._solve_square_pair_order(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, "front_first", deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        combined: List[SBSolution] = []
        seen: Set[Tuple[str, ...]] = set()
        for s in sols_back + sols_front:
            if s.moves not in seen:
                seen.add(s.moves)
                combined.append(s)
        combined.sort(key=lambda s: s.move_count)
        return combined[:k]

    def _search_dr(
        self,
        dr_slot: int,
        dr_eo: int,
        k: int = 3,
        deadline: Optional[float] = None,
    ) -> List[Tuple[str, ...]]:
        """Finds shortest move sequence(s) to solve DR edge (slot 7, eo 0 -> state 14)."""
        start_state = dr_slot * 2 + dr_eo
        if start_state == 14:
            return [()]

        queue: deque[Tuple[int, int, Tuple[str, ...]]] = deque([(start_state, -1, ())])
        sols: List[Tuple[str, ...]] = []
        found_depth: Optional[int] = None

        while queue:
            if deadline is not None and time.perf_counter() > deadline:
                break
            st, prev_m, path = queue.popleft()
            if found_depth is not None and len(path) > found_depth:
                break

            next_moves = range(len(SB_MOVESET)) if prev_m == -1 else self.allowed_moves[prev_m]
            for m in next_moves:
                nst = self.dr_trans[st][m]
                npath = path + (SB_MOVESET[m],)
                if nst == 14:
                    if found_depth is None:
                        found_depth = len(npath)
                    sols.append(npath)
                    if len(sols) >= k:
                        return sols
                else:
                    if found_depth is None and len(npath) < 6:
                        queue.append((nst, m, npath))

        return sols if sols else [()]

    def _solve_classical_order(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        placement: SBPlacement,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int,
        order: str,
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Solves SB using Classical Standard paradigm (DR -> Pair 1 -> Pair 2)."""
        dr_paths = self._search_dr(placement.dr_slot, placement.dr_eo, k=min(k, 3), deadline=deadline)
        if not dr_paths:
            dr_paths = [()]

        solutions: List[SBSolution] = []
        seen_moves: Set[Tuple[str, ...]] = set()

        for p1 in dr_paths:
            if deadline is not None and time.perf_counter() > deadline:
                break
            c1, e1, cent1 = self._advance_sb_state(c_idx, e_idx, center_off, p1)
            p1_prev_m = _SB_MOVE_INDEX[p1[-1]] if p1 else -1

            if order == "back_first":
                dfr, dfr_co, dbr, dbr_co = SBIndexer.decode_corners(c1)
                dr, dr_eo, fr, fr_eo, br, br_eo = SBIndexer.decode_edges(e1)
                sq_c = RightBackSquareIndexer.encode_corner(dbr, dbr_co)
                sq_e = RightBackSquareIndexer.encode_edges(dr, dr_eo, br, br_eo)
                sq_paths = self._search_square("back", sq_c, sq_e, k=min(k, 3), prev_m=p1_prev_m, deadline=deadline)
            else:
                dfr, dfr_co, dbr, dbr_co = SBIndexer.decode_corners(c1)
                dr, dr_eo, fr, fr_eo, br, br_eo = SBIndexer.decode_edges(e1)
                sq_c = RightFrontSquareIndexer.encode_corner(dfr, dfr_co)
                sq_e = RightFrontSquareIndexer.encode_edges(dr, dr_eo, fr, fr_eo)
                sq_paths = self._search_square("front", sq_c, sq_e, k=min(k, 3), prev_m=p1_prev_m, deadline=deadline)

            if not sq_paths:
                sq_paths = [()]

            for p2 in sq_paths:
                if deadline is not None and time.perf_counter() > deadline:
                    break
                c2, e2, cent2 = self._advance_sb_state(c1, e1, cent1, p2)
                prev_m = _SB_MOVE_INDEX[p2[-1]] if p2 else (_SB_MOVE_INDEX[p1[-1]] if p1 else -1)
                state_idx = e2 * NUM_SB_CORNER_CONFIGS + c2
                h3 = self._heuristic(state_idx, cent2)

                if h3 == 0:
                    combined = p1 + p2
                    if combined not in seen_moves:
                        seen_moves.add(combined)
                        solutions.append(
                            self._build_solution(
                                base_cube=base_cube,
                                moves=combined,
                                ori=ori,
                                orientation=ori_str,
                                uninspected=uninspected,
                                sym=sym,
                                style="classical",
                                order=order,
                            )
                        )
                else:
                    depth = max(0, h3 - 4) if allow_macro_triggers else h3
                    stage3_sols: List[Tuple[str, ...]] = []
                    while len(stage3_sols) < k and depth <= 15:
                        if deadline is not None and time.perf_counter() > deadline:
                            break
                        needed = k - len(stage3_sols)
                        depth_sols: List[Tuple[str, ...]] = []
                        self._ida_search(
                            c2, e2, cent2, 0, depth, prev_m, [], depth_sols, needed, deadline,
                            allow_macro_triggers=allow_macro_triggers,
                        )
                        for p3 in depth_sols:
                            if p3 not in stage3_sols:
                                stage3_sols.append(p3)
                        depth += 1

                    for p3 in stage3_sols:
                        combined = p1 + p2 + p3
                        if combined not in seen_moves:
                            seen_moves.add(combined)
                            solutions.append(
                                self._build_solution(
                                    base_cube=base_cube,
                                    moves=combined,
                                    ori=ori,
                                    orientation=ori_str,
                                    uninspected=uninspected,
                                    sym=sym,
                                    style="classical",
                                    order=order,
                                )
                            )

        solutions.sort(key=lambda s: s.move_count)
        return solutions[:k]

    def _solve_classical(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        placement: SBPlacement,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int = 5,
        order: str = "best",
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Solves SB using Classical Standard paradigm (best, back_first, or front_first)."""
        if order in ("back_first", "front_first"):
            return self._solve_classical_order(
                base_cube, ori, sym, uninspected, ori_str,
                placement, c_idx, e_idx, center_off, k, order, deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
        sols_back = self._solve_classical_order(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, "back_first", deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        sols_front = self._solve_classical_order(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, "front_first", deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        combined: List[SBSolution] = []
        seen: Set[Tuple[str, ...]] = set()
        for s in sols_back + sols_front:
            if s.moves not in seen:
                seen.add(s.moves)
                combined.append(s)
        combined.sort(key=lambda s: s.move_count)
        return combined[:k]

    def _solve_free(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int = 5,
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Direct Center-Aligned IDA* heuristic search to full Second Block."""
        state_idx = e_idx * NUM_SB_CORNER_CONFIGS + c_idx
        h0 = self._heuristic(state_idx, center_off)

        # Handle already solved state (0 moves)
        if h0 == 0:
            sol = self._build_solution(
                base_cube=base_cube,
                moves=(),
                ori=ori,
                orientation=ori_str,
                uninspected=uninspected,
                sym=sym,
                style="free",
                order="direct",
            )
            return [sol]

        # Top-K Candidate Search
        depth = max(0, h0 - 4) if allow_macro_triggers else h0
        sols: List[Tuple[str, ...]] = []
        seen_paths: Set[Tuple[str, ...]] = set()

        while len(sols) < k and depth <= 15:
            if deadline is not None and time.perf_counter() > deadline:
                break
            needed = k - len(sols)
            depth_sols: List[Tuple[str, ...]] = []
            self._ida_search(
                c_idx, e_idx, center_off, 0, depth, -1, [], depth_sols, needed, deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
            for path_tuple in depth_sols:
                if path_tuple not in seen_paths:
                    seen_paths.add(path_tuple)
                    sols.append(path_tuple)
            depth += 1

        results: List[SBSolution] = []
        for path_tuple in sols[:k]:
            results.append(
                self._build_solution(
                    base_cube=base_cube,
                    moves=path_tuple,
                    ori=ori,
                    orientation=ori_str,
                    uninspected=uninspected,
                    sym=sym,
                    style="free",
                    order="direct",
                )
            )

        results.sort(key=lambda s: s.move_count)
        return results

    def _solve_all(
        self,
        base_cube: CubeState,
        ori: RouxOrientation,
        sym: CanonicalSymmetry,
        uninspected: bool,
        ori_str: str,
        placement: SBPlacement,
        c_idx: int,
        e_idx: int,
        center_off: int,
        k: int = 5,
        order: str = "best",
        deadline: Optional[float] = None,
        allow_macro_triggers: bool = False,
    ) -> List[SBSolution]:
        """Aggregates candidate solutions across all Second Block search paradigms."""
        sols_free = self._solve_free(
            base_cube, ori, sym, uninspected, ori_str,
            c_idx, e_idx, center_off, k, deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        sols_sq = self._solve_square_pair(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, order, deadline,
            allow_macro_triggers=allow_macro_triggers,
        )
        sols_cl = self._solve_classical(
            base_cube, ori, sym, uninspected, ori_str,
            placement, c_idx, e_idx, center_off, k, order, deadline,
            allow_macro_triggers=allow_macro_triggers,
        )

        style_priority = {"classical": 0, "square_pair": 1, "free": 2}
        candidates_by_moves: Dict[Tuple[str, ...], SBSolution] = {}
        for sol in sorted(sols_cl + sols_sq + sols_free, key=lambda s: (s.move_count, style_priority.get(s.style, 99))):
            if sol.moves not in candidates_by_moves:
                candidates_by_moves[sol.moves] = sol

        aggregated = list(candidates_by_moves.values())
        aggregated.sort(key=lambda s: s.move_count)
        return aggregated[:k]

    def solve(
        self,
        scramble_or_cube: Union[str, CubeState],
        k: int = 5,
        style: str = "free",
        order: str = "best",
        orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
        timeout_ms: Optional[float] = None,
        allow_macro_triggers: bool = False,
        rank_by: Literal["stm", "e_stm"] = "stm",
        top_k: Optional[int] = None,
    ) -> List[SBSolution]:
        """Discovers top-K candidate Second Block solutions using IDA* search.

        Args:
            scramble_or_cube: Scramble move sequence string or initialized CubeState.
            k: Number of candidate paths to return (default 5).
            style: Search style ("all", "free", "square_pair", "classical").
            order: Sub-step ordering ("best", "back_first", "front_first").
            orientation: Optional orientation filter (restricts to single color scheme).
            timeout_ms: Optional search timeout in milliseconds.
            allow_macro_triggers: Whether to evaluate 4-STM compound moves (Sledgehammer, Hedge).
            rank_by: Ranking metric for candidate solutions ("stm" or "e_stm").
            top_k: Optional alias for k.

        Returns:
            List of top-K SBSolution candidate move sequences.
        """
        if isinstance(scramble_or_cube, str):
            cube = CubeState().apply_moves(scramble_or_cube)
        elif isinstance(scramble_or_cube, CubeState):
            cube = scramble_or_cube.copy()
        else:
            raise TypeError(f"Expected str or CubeState, got {type(scramble_or_cube).__name__}")

        effective_k = top_k if top_k is not None else k
        if effective_k <= 0:
            raise ValueError(f"k must be at least 1, got {effective_k}")

        pool_target = max(effective_k * 4, 20) if rank_by == "e_stm" else effective_k
        deadline = (time.perf_counter() + timeout_ms / 1000.0) if timeout_ms is not None else None

        # Resolve orientation and symmetry frame
        sym: CanonicalSymmetry
        ori: RouxOrientation
        uninspected: bool = False

        if orientation is not None:
            ori = get_orientation(orientation)
            if ori.symmetry is None:
                raise ValueError(f"Orientation {ori.rotations!r} is not dual-neutral")
            sym = ori.symmetry
            # Verify FB is solved on cube
            if is_fb_solved_for_symmetry(cube, sym, inspected=True):
                uninspected = False
            elif is_fb_solved_for_symmetry(cube, sym, inspected=False):
                uninspected = True
                if sym.inspection_rotation:
                    cube.apply_moves(sym.inspection_rotation)
            else:
                raise ValueError(
                    f"First Block is not solved for specified orientation {sym.name} "
                    f"(bottom={sym.bottom_color.name}, left={sym.left_color.name})"
                )
        else:
            # Auto-detect orientation:
            # 1. Check if FB is already on Left face (inspected solve frame)
            matched: Optional[RouxOrientation] = None
            for candidate in get_dual_neutral_orientations():
                if candidate.is_fb_solved(cube):
                    matched = candidate
                    break

            if matched is not None:
                ori = matched
                assert matched.symmetry is not None
                sym = matched.symmetry  # guaranteed non-None for dual-neutral
                uninspected = False
            else:
                # 2. Check if FB was solved in uninspected frame
                found_sym: Optional[CanonicalSymmetry] = None
                for s in get_all_symmetries():
                    if is_fb_solved_for_symmetry(cube, s, inspected=False):
                        found_sym = s
                        break
                if found_sym is not None:
                    sym = found_sym
                    ori = get_orientation(sym)
                    uninspected = True
                    if sym.inspection_rotation:
                        cube.apply_moves(sym.inspection_rotation)
                else:
                    raise ValueError(
                        "First Block completion not found on cube in any dual-neutral orientation. "
                        "First Block must be solved before solving Second Block."
                    )

        ori_str = sym.value

        # Extract piece coordinates and center offset
        placement = ori.extract_sb_placement(cube)
        c_idx = SBIndexer.encode_corners(
            placement.dfr_slot, placement.dfr_co,
            placement.dbr_slot, placement.dbr_co
        )
        e_idx = SBIndexer.encode_edges(
            placement.dr_slot, placement.dr_eo,
            placement.fr_slot, placement.fr_eo,
            placement.br_slot, placement.br_eo
        )
        center_off = ori.get_m_slice_center_offset(cube)

        VALID_STYLES = {"all", "free", "square_pair", "classical"}
        if style not in VALID_STYLES:
            raise ValueError(f"Unknown style '{style}'. Allowed styles: {sorted(VALID_STYLES)}")

        VALID_ORDERS = {"best", "back_first", "front_first"}
        if order not in VALID_ORDERS:
            raise ValueError(f"Unknown order '{order}'. Allowed orders: {sorted(VALID_ORDERS)}")

        if style == "all":
            raw_sols = self._solve_all(
                base_cube=cube,
                ori=ori,
                sym=sym,
                uninspected=uninspected,
                ori_str=ori_str,
                placement=placement,
                c_idx=c_idx,
                e_idx=e_idx,
                center_off=center_off,
                k=pool_target,
                order=order,
                deadline=deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
        elif style == "square_pair":
            raw_sols = self._solve_square_pair(
                base_cube=cube,
                ori=ori,
                sym=sym,
                uninspected=uninspected,
                ori_str=ori_str,
                placement=placement,
                c_idx=c_idx,
                e_idx=e_idx,
                center_off=center_off,
                k=pool_target,
                order=order,
                deadline=deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
        elif style == "classical":
            raw_sols = self._solve_classical(
                base_cube=cube,
                ori=ori,
                sym=sym,
                uninspected=uninspected,
                ori_str=ori_str,
                placement=placement,
                c_idx=c_idx,
                e_idx=e_idx,
                center_off=center_off,
                k=pool_target,
                order=order,
                deadline=deadline,
                allow_macro_triggers=allow_macro_triggers,
            )
        else:
            raw_sols = self._solve_free(
                base_cube=cube,
                ori=ori,
                sym=sym,
                uninspected=uninspected,
                ori_str=ori_str,
                c_idx=c_idx,
                e_idx=e_idx,
                center_off=center_off,
                k=pool_target,
                deadline=deadline,
                allow_macro_triggers=allow_macro_triggers,
            )

        if rank_by == "e_stm":
            scored: List[SBSolution] = []
            for sol in raw_sols:
                score = self.flow_scorer.score_moves(sol.moves)
                scored.append(
                    SBSolution(
                        moves=sol.moves,
                        move_count=sol.move_count,
                        style=sol.style,
                        order=sol.order,
                        dr_move_idx=sol.dr_move_idx,
                        pair1_move_idx=sol.pair1_move_idx,
                        square_move_idx=sol.square_move_idx,
                        resulting_cmll_case=sol.resulting_cmll_case,
                        orientation=sol.orientation,
                        e_stm=score.e_stm,
                    )
                )
            scored.sort(
                key=lambda s: (s.e_stm if s.e_stm is not None else float("inf"), s.move_count)
            )
            return scored[:effective_k]

        return raw_sols[:effective_k]


def solve_sb(
    scramble_or_cube: Union[str, CubeState],
    k: int = 5,
    top_k: Optional[int] = None,
    style: str = "all",
    order: str = "best",
    orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
    timeout_ms: Optional[float] = None,
    allow_macro_triggers: bool = False,
    rank_by: Literal["stm", "e_stm"] = "stm",
) -> List[SBSolution]:
    """Solves Second Block using Center-Aligned IDA* heuristic search.

    Args:
        scramble_or_cube: Scramble move sequence string or CubeState with FB solved.
        k: Maximum candidate solutions to return (default 5).
        top_k: Alias for k.
        style: Search style ("all", "free", "classical", "square_pair").
        order: Pair ordering ("best", "back_first", "front_first").
        orientation: Optional First Block orientation filter.
        timeout_ms: Optional search timeout in milliseconds.
        allow_macro_triggers: Whether to evaluate 4-STM compound moves (Sledgehammer, Hedge).
        rank_by: Ranking metric for candidate solutions ("stm" or "e_stm").

    Returns:
        List of top-K SBSolution candidate move sequences.
    """
    effective_k = top_k if top_k is not None else k
    solver = SBSolver.get_instance()
    return solver.solve(
        scramble_or_cube=scramble_or_cube,
        k=effective_k,
        style=style,
        order=order,
        orientation=orientation,
        timeout_ms=timeout_ms,
        allow_macro_triggers=allow_macro_triggers,
        rank_by=rank_by,
    )


__all__ = [
    "SBSolution",
    "SBSolver",
    "solve_sb",
]



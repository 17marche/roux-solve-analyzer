"""Second Block (SB) Center-Aligned Free Blockbuilding IDA* Solver with Symmetry Re-mapping.

Uses the 1.08M state Second Block Pattern Database (PDB) to find top-K shortest
candidate paths directly to full Second Block in the rotationless <R, U, r, M> generator.
Enforces Center-Aligned SB goal condition and dual-neutral symmetry re-mapping.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Sequence, Optional

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


# -----------------------------------------------------------------------------
# Public Dataclass & Goal Evaluation
# -----------------------------------------------------------------------------

from ..core.constants import Color, Center
from ..core.cube import CubeState
from ..segmenter.fb_detector import BlockDefinition, ALL_BLOCK_DEFINITIONS, FBDetector
from ..segmenter.sb_detector import SBDetector


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


def get_m_slice_center_offset(
    cube: CubeState,
    block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]
) -> int:
    """Determines the M-slice rotation offset in {0, 1, 2, 3} for the cube relative to block definition.
    0: Aligned with First Block.
    1: Rotated by M.
    2: Rotated by M2.
    3: Rotated by M'.
    """
    u_col = Color(cube.centers[Center.U])
    if u_col == block.top_color:
        return 0
    elif u_col == block.back_color:
        return 1
    elif u_col == block.bottom_color:
        return 2
    elif u_col == block.front_color:
        return 3
    else:
        raise ValueError(
            f"Center U color {u_col.name} is not in M-slice for block {block.rotations!r}"
        )


def is_center_aligned_sb_solved(
    cube: CubeState,
    block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]
) -> bool:
    """Checks if Center-Aligned Second Block is strictly solved:
    1. All 5 Right-block pieces (DR, FR, BR, DFR, DRB) are solved.
    2. M-slice centers (U, D, F, B) match their solved colors (offset 0).
    """
    if not SBDetector.is_right_1x2x3_block_solved(cube, block):
        return False
    return get_m_slice_center_offset(cube, block) == 0


# -----------------------------------------------------------------------------
# IDA* Search Engine
# -----------------------------------------------------------------------------

import time
from typing import Dict, Set, Union
from ..core.parser import MoveParser
from ..segmenter.cmll_classifier import CMLLClassifier
from .sb_indexer import SBIndexer, SBPlacement, NUM_SB_CORNER_CONFIGS
from .sb_pdb import SBPDB
from .sb_pdb_generator import build_sb_transition_tables
from .symmetry import (
    CanonicalSymmetry,
    get_symmetry,
    get_all_symmetries,
    is_fb_solved_for_symmetry,
    translate_moves_to_original,
)


def extract_sb_placement(
    cube: CubeState,
    block: BlockDefinition = ALL_BLOCK_DEFINITIONS[""]
) -> SBPlacement:
    """Extracts SBPlacement relative to a given dual-neutral block definition."""
    cp = cube.cp.tolist()
    co = cube.co.tolist()
    ep = cube.ep.tolist()
    eo = cube.eo.tolist()

    dfr_slot = cp.index(block.dfr_piece)
    dbr_slot = cp.index(block.drb_piece)
    dr_slot = ep.index(block.dr_piece)
    fr_slot = ep.index(block.fr_piece)
    br_slot = ep.index(block.br_piece)

    return SBPlacement(
        dr_slot=dr_slot,
        dr_eo=(eo[dr_slot] - block.dr_eo) % 2,
        fr_slot=fr_slot,
        fr_eo=(eo[fr_slot] - block.fr_eo) % 2,
        br_slot=br_slot,
        br_eo=(eo[br_slot] - block.br_eo) % 2,
        dfr_slot=dfr_slot,
        dfr_co=(co[dfr_slot] - block.dfr_co) % 3,
        dbr_slot=dbr_slot,
        dbr_co=(co[dbr_slot] - block.drb_co) % 3,
    )


class SBSolver:
    """High-performance Top-K Candidate Search IDA* Second Block solver."""

    _instance: Optional[SBSolver] = None

    def __init__(self, pdb: Optional[SBPDB] = None) -> None:
        self.pdb: SBPDB = pdb if pdb is not None else SBPDB()
        self._get_distance = self.pdb.get_distance_by_index
        c_arr, e_arr = build_sb_transition_tables(SB_MOVESET)
        self.corner_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in c_arr)
        self.edge_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in e_arr)
        self.allowed_moves: Tuple[Tuple[int, ...], ...] = _build_sb_allowed_moves()
        self.center_trans: Tuple[Tuple[int, ...], ...] = _build_center_transitions()

    @classmethod
    def get_instance(cls) -> SBSolver:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

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
    ) -> None:
        """Recursive depth-bounded search with exact PDB pruning and center alignment."""
        if deadline is not None and time.perf_counter() > deadline:
            return

        state_idx = e_idx * NUM_SB_CORNER_CONFIGS + c_idx
        h = self._get_distance(state_idx)
        if center_off != 0 and h == 0:
            h = 1

        if g + h > max_depth:
            return

        if h == 0 and center_off == 0:
            if g == max_depth:
                solutions.append(tuple(path))
            return

        next_moves = range(len(SB_MOVESET)) if prev_m == -1 else self.allowed_moves[prev_m]
        for m in next_moves:
            nc = self.corner_trans[c_idx][m]
            ne = self.edge_trans[e_idx][m]
            ncent = self.center_trans[center_off][m]
            path.append(SB_MOVESET[m])
            self._ida_search(nc, ne, ncent, g + 1, max_depth, m, path, solutions, needed, deadline)
            path.pop()
            if len(solutions) >= needed:
                return

    def _build_solution(
        self,
        base_cube: CubeState,
        moves: Tuple[str, ...],
        block: BlockDefinition,
        orientation: str,
        uninspected: bool,
        sym: CanonicalSymmetry,
    ) -> SBSolution:
        """Builds a structured SBSolution and previews the resulting CMLL case."""
        # Detect milestones on the solved cube
        c_eval = base_cube.copy()
        if moves:
            c_eval.apply_moves(" ".join(moves))

        case_id, group, pre_auf = CMLLClassifier.classify_state(c_eval, block)
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

        if len(moves) == 0:
            dr_move_idx = 0
            pair1_move_idx = 0
            square_move_idx = 0
        else:
            sim = base_cube.copy()
            for idx, m_token in enumerate(moves):
                sim.apply_move(m_token)
                dr_ok = SBDetector.is_dr_solved(sim, block)
                back_ok = SBDetector.is_back_pair_solved(sim, block)
                front_ok = SBDetector.is_front_pair_solved(sim, block)

                if dr_move_idx is None and dr_ok:
                    dr_move_idx = idx
                if pair1_move_idx is None and dr_ok and (back_ok or front_ok):
                    pair1_move_idx = idx
                    square_move_idx = idx

        return SBSolution(
            moves=final_moves,
            move_count=len(final_moves),
            style="free",
            order="direct",
            dr_move_idx=dr_move_idx,
            pair1_move_idx=pair1_move_idx,
            square_move_idx=square_move_idx,
            resulting_cmll_case=cmll_case,
            orientation=orientation,
        )

    def solve(
        self,
        scramble_or_cube: Union[str, CubeState],
        k: int = 5,
        style: str = "free",
        orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
        timeout_ms: Optional[float] = None,
    ) -> List[SBSolution]:
        """Discovers top-K candidate Second Block solutions using IDA* search.

        Args:
            scramble_or_cube: Scramble move sequence string or initialized CubeState.
            k: Number of candidate paths to return (default 5).
            style: Search style ("free", "all").
            orientation: Optional orientation filter (restricts to single color scheme).
            timeout_ms: Optional search timeout in milliseconds.

        Returns:
            List of top-K SBSolution candidate move sequences.
        """
        if isinstance(scramble_or_cube, str):
            cube = CubeState().apply_moves(scramble_or_cube)
        elif isinstance(scramble_or_cube, CubeState):
            cube = scramble_or_cube.copy()
        else:
            raise TypeError(f"Expected str or CubeState, got {type(scramble_or_cube).__name__}")

        if k <= 0:
            raise ValueError(f"k must be at least 1, got {k}")

        deadline = (time.perf_counter() + timeout_ms / 1000.0) if timeout_ms is not None else None

        # Resolve orientation and symmetry frame
        sym: CanonicalSymmetry
        block: BlockDefinition
        uninspected: bool = False

        if orientation is not None:
            sym = get_symmetry(orientation)
            block = ALL_BLOCK_DEFINITIONS[sym.value]
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
            matched = FBDetector.match_fb_block(cube)
            if matched is not None:
                block = matched
                sym = get_symmetry(block.rotations)
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
                    block = ALL_BLOCK_DEFINITIONS[sym.value]
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
        placement = extract_sb_placement(cube, block)
        c_idx = SBIndexer.encode_corners(
            placement.dfr_slot, placement.dfr_co,
            placement.dbr_slot, placement.dbr_co
        )
        e_idx = SBIndexer.encode_edges(
            placement.dr_slot, placement.dr_eo,
            placement.fr_slot, placement.fr_eo,
            placement.br_slot, placement.br_eo
        )
        center_off = get_m_slice_center_offset(cube, block)

        state_idx = e_idx * NUM_SB_CORNER_CONFIGS + c_idx
        h0 = self._get_distance(state_idx)
        if center_off != 0 and h0 == 0:
            h0 = 1

        # Handle already solved state (0 moves)
        if h0 == 0 and center_off == 0:
            sol = self._build_solution(
                base_cube=cube,
                moves=(),
                block=block,
                orientation=ori_str,
                uninspected=uninspected,
                sym=sym,
            )
            return [sol]

        # Multi-path IDA* search
        depth = h0
        sols: List[Tuple[str, ...]] = []
        seen_paths: Set[Tuple[str, ...]] = set()

        while len(sols) < k and depth <= 15:
            if deadline is not None and time.perf_counter() > deadline:
                break
            needed = k - len(sols)
            depth_sols: List[Tuple[str, ...]] = []
            self._ida_search(
                c_idx, e_idx, center_off, 0, depth, -1, [], depth_sols, needed, deadline
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
                    base_cube=cube,
                    moves=path_tuple,
                    block=block,
                    orientation=ori_str,
                    uninspected=uninspected,
                    sym=sym,
                )
            )

        results.sort(key=lambda s: s.move_count)
        return results


def solve_sb(
    scramble_or_cube: Union[str, CubeState],
    k: int = 5,
    top_k: Optional[int] = None,
    style: str = "free",
    order: str = "best",
    orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
    timeout_ms: Optional[float] = None,
) -> List[SBSolution]:
    """Solves Second Block using Center-Aligned IDA* heuristic search.

    Args:
        scramble_or_cube: Scramble move sequence string or CubeState with FB solved.
        k: Maximum candidate solutions to return (default 5).
        top_k: Alias for k.
        style: Search style ("free", "all", "classical", "square_pair").
        order: Pair ordering ("best", "back_first", "front_first", "direct").
        orientation: Optional First Block orientation filter.
        timeout_ms: Optional search timeout in milliseconds.

    Returns:
        List of top-K SBSolution candidate move sequences.
    """
    effective_k = top_k if top_k is not None else k
    solver = SBSolver.get_instance()
    return solver.solve(
        scramble_or_cube=scramble_or_cube,
        k=effective_k,
        style=style,
        orientation=orientation,
        timeout_ms=timeout_ms,
    )


__all__ = [
    "SBSolution",
    "SBSolver",
    "solve_sb",
    "is_center_aligned_sb_solved",
    "get_m_slice_center_offset",
    "extract_sb_placement",
]



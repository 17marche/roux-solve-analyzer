"""Top-K Candidate Search IDA* First Block (FB) solver and candidate engine.

Uses the admissible 5.32M state First Block Pattern Database (PDB) with branch pruning
(inverse moves, commutative ordering, redundant wide turns) to find top-K candidate
solutions across all 8 dual-neutral orientations and 4 M-slice inspection pitch angles.
Strictly excludes physical L turns per ADR-0001.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union, NamedTuple
import numpy as np

from ..core.constants import Color
from ..core.cube import CubeState
from .fb_indexer import FBIndexer
from .fb_pdb import FBPDB
from .pdb_generator import FB_MOVESET, build_transition_tables
from .symmetry import (
    CanonicalSymmetry,
    extract_canonical_placement,
    get_all_symmetries,
    get_symmetry,
)

# -----------------------------------------------------------------------------
# Move Families & Branch Pruning Rules
# -----------------------------------------------------------------------------

# FB_MOVESET:
#  0: U,   1: U2,  2: U'   (Family 0: U)
#  3: D,   4: D2,  5: D'   (Family 1: D)
#  6: R,   7: R2,  8: R'   (Family 2: R)
#  9: F,  10: F2, 11: F'   (Family 3: F)
# 12: B,  13: B2, 14: B'   (Family 4: B)
# 15: r,  16: r2, 17: r'   (Family 5: r)
# 18: M,  19: M2, 20: M'   (Family 6: M)

_MOVE_FAMILIES = (
    0, 0, 0,  # U
    1, 1, 1,  # D
    2, 2, 2,  # R
    3, 3, 3,  # F
    4, 4, 4,  # B
    5, 5, 5,  # r
    6, 6, 6,  # M
)

PITCH_ANGLES = ("", "x", "x2", "x'")


def _build_allowed_moves() -> Tuple[Tuple[int, ...], ...]:
    """Precomputes allowed subsequent move indices for branch pruning:
    - Prune immediate same-family / inverse moves (e.g. U then U').
    - Commutative face ordering: allow U then D (prune D then U); allow F then B (prune B then F).
    - Redundant wide / slice moves in {R, r, M}: allow R then M; prune all other adjacent combinations in {R, r, M}.
    """
    allowed: List[Tuple[int, ...]] = []
    for m1 in range(len(FB_MOVESET)):
        f1 = _MOVE_FAMILIES[m1]
        allowed_m2: List[int] = []
        for m2 in range(len(FB_MOVESET)):
            f2 = _MOVE_FAMILIES[m2]
            # 1. Same family turn is always pruned (e.g. U U', R R2)
            if f1 == f2:
                continue
            # 2. Commutative face ordering:
            if f1 == 1 and f2 == 0:  # D before U is pruned (enforce U then D)
                continue
            if f1 == 4 and f2 == 3:  # B before F is pruned (enforce F then B)
                continue
            # 3. R, r, M group redundancies:
            # Faces 2 (R), 5 (r), 6 (M) are parallel/overlapping.
            # We only allow R followed by M. All other adjacent pairs in {R, r, M} are redundant.
            if f1 in (2, 5, 6) and f2 in (2, 5, 6):
                if not (f1 == 2 and f2 == 6):
                    continue
            allowed_m2.append(m2)
        allowed.append(tuple(allowed_m2))
    return tuple(allowed)


def normalize_rotations(rotations: str) -> str:
    """Normalizes inspection rotations, canceling adjacent rotations on the same axis."""
    if not rotations.strip():
        return ""
    turn_val = {
        "x": 1, "x2": 2, "x'": 3,
        "y": 1, "y2": 2, "y'": 3,
        "z": 1, "z2": 2, "z'": 3,
    }
    rot_map = {
        "x": {1: "x", 2: "x2", 3: "x'"},
        "y": {1: "y", 2: "y2", 3: "y'"},
        "z": {1: "z", 2: "z2", 3: "z'"},
    }
    stack: List[Tuple[str, int]] = []
    for token in rotations.split():
        if not token:
            continue
        axis = token[0].lower()
        if axis not in rot_map or token not in turn_val:
            continue
        val = turn_val[token]
        if stack and stack[-1][0] == axis:
            prev_axis, prev_val = stack.pop()
            new_val = (prev_val + val) % 4
            if new_val != 0:
                stack.append((axis, new_val))
        else:
            stack.append((axis, val))
    return " ".join(rot_map[axis][val] for axis, val in stack)


# -----------------------------------------------------------------------------
# Public Dataclass
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class FBSolution:
    """Immutable solution representation for First Block (FB)."""
    moves: list[str]
    move_count: int
    orientation: str
    inspection_rotation: str


class _SearchConfig(NamedTuple):
    """Internal search configuration for an evaluated inspection state."""
    h0: int
    sym: CanonicalSymmetry
    orientation: str
    rotation: str
    c_idx: int
    e_idx: int


# -----------------------------------------------------------------------------
# IDA* Top-K Candidate Search Engine
# -----------------------------------------------------------------------------

class FBSolver:
    """High-performance Top-K Candidate Search IDA* First Block solver."""

    _instance: Optional[FBSolver] = None

    def __init__(self, pdb: Optional[FBPDB] = None) -> None:
        self.pdb: FBPDB = pdb if pdb is not None else FBPDB()
        self._get_distance = self.pdb.get_distance_by_index
        c_arr, e_arr = build_transition_tables(FB_MOVESET)
        self.corner_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in c_arr)
        self.edge_trans: Tuple[Tuple[int, ...], ...] = tuple(tuple(int(x) for x in row) for row in e_arr)
        self.allowed_moves: Tuple[Tuple[int, ...], ...] = _build_allowed_moves()

    @classmethod
    def get_instance(cls) -> FBSolver:
        """Returns the singleton instance of FBSolver."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ida_search(
        self,
        c_idx: int,
        e_idx: int,
        g: int,
        max_depth: int,
        prev_m: int,
        path: List[str],
        solutions: List[List[str]],
        needed: int,
        deadline: Optional[float],
    ) -> None:
        """Recursive depth-bounded search with exact PDB pruning."""
        if deadline is not None and time.perf_counter() > deadline:
            return

        state_idx = e_idx * 504 + c_idx
        h = self._get_distance(state_idx)

        if g + h > max_depth:
            return

        if h == 0:
            if g == max_depth:
                solutions.append(list(path))
            return

        next_moves = range(21) if prev_m == -1 else self.allowed_moves[prev_m]
        for m in next_moves:
            nc = self.corner_trans[c_idx][m]
            ne = self.edge_trans[e_idx][m]
            path.append(FB_MOVESET[m])
            self._ida_search(nc, ne, g + 1, max_depth, m, path, solutions, needed, deadline)
            path.pop()
            if len(solutions) >= needed:
                return

    def _collect_candidates_at_depth(
        self,
        configs: List[_SearchConfig],
        depth: int,
        k: int,
        candidates: List[FBSolution],
        seen_paths: Set[Tuple[str, Tuple[str, ...]]],
        deadline: Optional[float],
    ) -> None:
        """Searches configurations at a fixed depth, collecting unique solutions up to k candidates."""
        for cfg in configs:
            if cfg.h0 > depth:
                break
            needed = k - len(candidates)
            sols: List[List[str]] = []
            self._ida_search(cfg.c_idx, cfg.e_idx, 0, depth, -1, [], sols, needed, deadline)
            for path in sols:
                key = (cfg.rotation, tuple(path))
                if key not in seen_paths:
                    seen_paths.add(key)
                    candidates.append(
                        FBSolution(
                            moves=path,
                            move_count=depth,
                            orientation=cfg.orientation,
                            inspection_rotation=cfg.rotation,
                        )
                    )
            if len(candidates) >= k:
                break

    def solve(
        self,
        scramble_or_cube: Union[str, CubeState],
        k: int = 5,
        orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
        timeout_ms: Optional[float] = None,
    ) -> List[FBSolution]:
        """Discovers the top-K candidate First Block solutions using IDA* search.

        Args:
            scramble_or_cube: Scramble move sequence string or initialized CubeState.
            k: Number of candidate paths to return (default 5).
            orientation: Optional orientation filter (restricts to single color scheme).
            timeout_ms: Optional search timeout in milliseconds.

        Returns:
            List of top-K FBSolution objects ranked by move count (ascending).
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

        if orientation is not None:
            symmetries = [get_symmetry(orientation)]
        else:
            symmetries = get_all_symmetries()

        # Evaluate all candidate inspection configurations (symmetries x 4 pitch angles)
        configs: List[_SearchConfig] = []
        for sym in symmetries:
            base = sym.inspection_rotation
            ori_name = f"{sym.bottom_color.name}-{sym.left_color.name}"
            for p in PITCH_ANGLES:
                rot = normalize_rotations(f"{base} {p}")
                c_insp = cube.copy()
                if rot:
                    c_insp.apply_moves(rot)

                p_canon = extract_canonical_placement(c_insp, sym, inspected=True)
                c_idx = FBIndexer.encode_corners(p_canon.dlf_slot, p_canon.dlf_co, p_canon.dbl_slot, p_canon.dbl_co)
                e_idx = FBIndexer.encode_edges(p_canon.dl_slot, p_canon.dl_eo, p_canon.fl_slot, p_canon.fl_eo, p_canon.bl_slot, p_canon.bl_eo)
                state_idx = e_idx * 504 + c_idx
                h0 = self._get_distance(state_idx)
                configs.append(_SearchConfig(h0=h0, sym=sym, orientation=ori_name, rotation=rot, c_idx=c_idx, e_idx=e_idx))

        configs.sort(key=lambda x: x.h0)
        min_h = configs[0].h0

        # Handle already solved state (0 moves)
        if min_h == 0:
            solutions: List[FBSolution] = []
            seen_solved: Set[Tuple[str, str]] = set()
            for cfg in configs:
                if cfg.h0 == 0:
                    key = (cfg.orientation, cfg.rotation)
                    if key not in seen_solved:
                        seen_solved.add(key)
                        solutions.append(
                            FBSolution(
                                moves=[],
                                move_count=0,
                                orientation=cfg.orientation,
                                inspection_rotation=cfg.rotation,
                            )
                        )
                        if len(solutions) >= k:
                            break
            return solutions

        candidates: List[FBSolution] = []
        seen_paths: Set[Tuple[str, Tuple[str, ...]]] = set()

        # 1. Search at optimal depth L
        self._collect_candidates_at_depth(configs, min_h, k, candidates, seen_paths, deadline)

        # 2. Expand search to depth L + 1 if fewer than k candidates found
        if len(candidates) < k:
            self._collect_candidates_at_depth(configs, min_h + 1, k, candidates, seen_paths, deadline)

        candidates.sort(key=lambda s: s.move_count)
        return candidates[:k]


def solve_fb(
    scramble_or_cube: Union[str, CubeState],
    k: int = 5,
    orientation: Optional[Union[str, CanonicalSymmetry, Tuple[Color, Color]]] = None,
    timeout_ms: Optional[float] = 10.0,
) -> List[FBSolution]:
    """Solves First Block using Top-K Candidate Search IDA* heuristic search.

    Args:
        scramble_or_cube: Scramble move sequence string or CubeState.
        k: Maximum candidate solutions to return (default 5).
        orientation: Optional color scheme or symmetry identifier to restrict search.
        timeout_ms: Optional search timeout in milliseconds (default 10ms).

    Returns:
        List of top-K FBSolution candidate move sequences.
    """
    solver = FBSolver.get_instance()
    return solver.solve(
        scramble_or_cube,
        k=k,
        orientation=orientation,
        timeout_ms=timeout_ms,
    )


__all__ = [
    "FBSolution",
    "FBSolver",
    "solve_fb",
]

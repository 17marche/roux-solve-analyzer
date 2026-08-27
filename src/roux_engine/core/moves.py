"""Move definitions, permutations, orientation shifts, and composition operators."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple, Sequence
import numpy as np

from .constants import NUM_CORNERS, NUM_EDGES, NUM_CENTERS


@dataclass(frozen=True)
class Move:
    """Represents a single atomic or composite move operation on a 3D Rubik's cube."""
    name: str
    cp_perm: np.ndarray  # Shape (8,), dtype int8
    co_ori: np.ndarray   # Shape (8,), dtype int8
    ep_perm: np.ndarray  # Shape (12,), dtype int8
    eo_ori: np.ndarray   # Shape (12,), dtype int8
    tp_perm: np.ndarray  # Shape (6,), dtype int8 (Centers: U, D, F, B, L, R)

    def apply(self, cp: np.ndarray, co: np.ndarray, ep: np.ndarray, eo: np.ndarray, tp: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies this move to state arrays, returning new state arrays."""
        new_cp = cp[self.cp_perm]
        new_co = (co[self.cp_perm] + self.co_ori) % 3
        new_ep = ep[self.ep_perm]
        new_eo = (eo[self.ep_perm] + self.eo_ori) % 2
        new_tp = tp[self.tp_perm]
        return new_cp, new_co, new_ep, new_eo, new_tp

    def apply_inplace(self, state: 'CubeState') -> 'CubeState':
        """Applies this move in-place to the given CubeState."""
        state.cp[:] = state.cp[self.cp_perm]
        state.co[:] = (state.co[self.cp_perm] + self.co_ori) % 3
        state.ep[:] = state.ep[self.ep_perm]
        state.eo[:] = (state.eo[self.ep_perm] + self.eo_ori) % 2
        state.centers[:] = state.centers[self.tp_perm]
        return state

    def compose(self, other: Move, name: str = "") -> Move:
        """Composes this move followed by other (i.e. other(self(state)))."""
        composed_name = name if name else f"{self.name} {other.name}"
        cp_perm = self.cp_perm[other.cp_perm]
        co_ori = (self.co_ori[other.cp_perm] + other.co_ori) % 3
        ep_perm = self.ep_perm[other.ep_perm]
        eo_ori = (self.eo_ori[other.ep_perm] + other.eo_ori) % 2
        tp_perm = self.tp_perm[other.tp_perm]
        return Move(
            name=composed_name,
            cp_perm=cp_perm,
            co_ori=co_ori,
            ep_perm=ep_perm,
            eo_ori=eo_ori,
            tp_perm=tp_perm
        )


def _build_base_move(
    name: str,
    cpc: List[Tuple[int, int]],
    coc: List[int],
    epc: List[Tuple[int, int]],
    eoc: List[int],
    tpc: List[Tuple[int, int]]
) -> Move:
    """Builds a base move from cycle permutation specifications."""
    cp = np.arange(NUM_CORNERS, dtype=np.int8)
    co = np.zeros(NUM_CORNERS, dtype=np.int8)
    for (src, dst), ori in zip(cpc, coc):
        cp[dst] = src
        co[dst] = ori

    ep = np.arange(NUM_EDGES, dtype=np.int8)
    eo = np.zeros(NUM_EDGES, dtype=np.int8)
    for (src, dst), ori in zip(epc, eoc):
        ep[dst] = src
        eo[dst] = ori

    tp = np.arange(NUM_CENTERS, dtype=np.int8)
    for src, dst in tpc:
        tp[dst] = src

    return Move(name=name, cp_perm=cp, co_ori=co, ep_perm=ep, eo_ori=eo, tp_perm=tp)


# -----------------------------------------------------------------------------
# Base Primitive Face & Slice Turns
# -----------------------------------------------------------------------------

# U: Corners (0->1->2->3), Edges (0->1->2->3)
_U = _build_base_move("U", [(0, 1), (1, 2), (2, 3), (3, 0)], [0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 0)], [0, 0, 0, 0], [])

# D: Corners (4->7->6->5), Edges (4->7->6->5)
_D = _build_base_move("D", [(4, 7), (7, 6), (6, 5), (5, 4)], [0, 0, 0, 0], [(4, 7), (7, 6), (6, 5), (5, 4)], [0, 0, 0, 0], [])

# F: Corners (0->3->7->4), Edges (0->11->4->8)
_F = _build_base_move("F", [(0, 3), (3, 7), (7, 4), (4, 0)], [1, 2, 1, 2], [(0, 11), (11, 4), (4, 8), (8, 0)], [1, 1, 1, 1], [])

# B: Corners (1->5->6->2), Edges (2->9->6->10)
_B = _build_base_move("B", [(1, 5), (5, 6), (6, 2), (2, 1)], [2, 1, 2, 1], [(2, 9), (9, 6), (6, 10), (10, 2)], [1, 1, 1, 1], [])

# L: Corners (0->4->5->1), Edges (1->8->5->9)
_L = _build_base_move("L", [(0, 4), (4, 5), (5, 1), (1, 0)], [2, 1, 2, 1], [(1, 8), (8, 5), (5, 9), (9, 1)], [0, 0, 0, 0], [])

# R: Corners (3->2->6->7), Edges (3->10->7->11)
_R = _build_base_move("R", [(3, 2), (2, 6), (6, 7), (7, 3)], [1, 2, 1, 2], [(3, 10), (10, 7), (7, 11), (11, 3)], [0, 0, 0, 0], [])

# M: Slices down (in L direction). Edges (0->4->6->2), Centers (0->2->1->3)
_M = _build_base_move("M", [], [], [(0, 4), (4, 6), (6, 2), (2, 0)], [1, 1, 1, 1], [(0, 2), (2, 1), (1, 3), (3, 0)])

# E: Slices right (in D direction). Edges (8->9->10->11), Centers (2->4->3->5)
_E = _build_base_move("E", [], [], [(8, 9), (9, 10), (10, 11), (11, 8)], [1, 1, 1, 1], [(2, 4), (4, 3), (3, 5), (5, 2)])

# S: Slices clockwise (in F direction). Edges (1->3->7->5), Centers (0->5->1->4)
_S = _build_base_move("S", [], [], [(1, 3), (3, 7), (7, 5), (5, 1)], [1, 1, 1, 1], [(0, 5), (5, 1), (1, 4), (4, 0)])


# -----------------------------------------------------------------------------
# Generate Full 54 Moves Dictionary
# -----------------------------------------------------------------------------

MOVES: Dict[str, Move] = {}

def _register_move_family(base: Move, name: str) -> None:
    """Generates and registers name, name2, and name'."""
    m1 = base
    m2 = base.compose(base, f"{name}2")
    m3 = m2.compose(base, f"{name}'")
    MOVES[name] = m1
    MOVES[f"{name}2"] = m2
    MOVES[f"{name}'"] = m3


# 18 Face turns
_register_move_family(_U, "U")
_register_move_family(_D, "D")
_register_move_family(_F, "F")
_register_move_family(_B, "B")
_register_move_family(_L, "L")
_register_move_family(_R, "R")

# 9 Slice turns
_register_move_family(_M, "M")
_register_move_family(_E, "E")
_register_move_family(_S, "S")

# 18 Wide turns
# r = R * M'
_r = MOVES["R"].compose(MOVES["M'"], "r")
_register_move_family(_r, "r")

# l = L * M
_l = MOVES["L"].compose(MOVES["M"], "l")
_register_move_family(_l, "l")

# u = U * E'
_u = MOVES["U"].compose(MOVES["E'"], "u")
_register_move_family(_u, "u")

# d = D * E
_d = MOVES["D"].compose(MOVES["E"], "d")
_register_move_family(_d, "d")

# f = F * S
_f = MOVES["F"].compose(MOVES["S"], "f")
_register_move_family(_f, "f")

# b = B * S'
_b = MOVES["B"].compose(MOVES["S'"], "b")
_register_move_family(_b, "b")

# 9 Cube Rotations
# x = r * L'
_x = MOVES["r"].compose(MOVES["L'"], "x")
_register_move_family(_x, "x")

# y = u * D'
_y = MOVES["u"].compose(MOVES["D'"], "y")
_register_move_family(_y, "y")

# z = f * B'
_z = MOVES["f"].compose(MOVES["B'"], "z")
_register_move_family(_z, "z")


# Add uppercase wide aliases (Rw, Lw, Uw, Dw, Fw, Bw)
for wide_face in ["R", "L", "U", "D", "F", "B"]:
    lower = wide_face.lower()
    MOVES[f"{wide_face}w"] = MOVES[lower]
    MOVES[f"{wide_face}w'"] = MOVES[f"{lower}'"]
    MOVES[f"{wide_face}w2"] = MOVES[f"{lower}2"]


def get_move(name: str) -> Move:
    """Retrieves a Move object by name, raising KeyError if invalid."""
    if name in MOVES:
        return MOVES[name]
    raise KeyError(f"Invalid move name: '{name}'. Supported moves: {list(MOVES.keys())}")


def apply_move(state: 'CubeState', move: Union[str, Move]) -> 'CubeState':
    """Applies a single move in-place to CubeState and returns state."""
    move_obj = move if isinstance(move, Move) else get_move(move)
    return move_obj.apply_inplace(state)


def apply_moves(state: 'CubeState', moves: Union[str, Sequence[Union[str, Move]]]) -> 'CubeState':
    """Applies a sequence of moves in-place to CubeState and returns state."""
    if isinstance(moves, str):
        from .parser import MoveParser
        events = MoveParser.parse_string(moves)
        for ev in events:
            apply_move(state, ev.move)
    else:
        for m in moves:
            apply_move(state, m)
    return state

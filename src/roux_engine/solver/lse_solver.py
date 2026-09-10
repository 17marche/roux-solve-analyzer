"""Dynamic Last Six Edges (LSE) graph and sub-step solver.

Generates the complete 7,680-state <M, U> transition graph dynamically in memory
at startup (approx 20-30ms) for instant O(1) optimal solutions for Step 4a (EO / EOLR / EOLR-b),
Step 4b (UL/UR placement), Step 4c (M-slice & center permutation), and 1-look global LSE.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Set, Union, Any
import numpy as np

from ..core.constants import Edge, Center, Corner, Color
from ..core.cube import CubeState
from ..core.orientation import (
    RouxOrientation,
    CanonicalSymmetry,
    CANONICAL_ORIENTATION,
    get_orientation,
    get_all_orientations,
    get_dual_neutral_orientations,
    translate_moves_to_original,
    _parse_move_sequence,
)
from ..core.moves import MOVES
from ..core.parser import MoveParser
from ._lse_4c_data import LSE_4C_SOLUTIONS


# LSE positions on the cube: UF=0, UL=1, UB=2, UR=3, DF=4, DB=6
POS_LSE = (0, 1, 2, 3, 4, 6)
# Maps POS_LSE physical slot -> local index 0..5
SLOT_TO_LOCAL = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 6: 5}
SLOT_TO_LOCAL_ARR = (0, 1, 2, 3, 4, 0, 5)


def cancel_moves(moves: list[str]) -> list[str]:
    """Cancels adjacent redundant moves in <M, U>."""
    turn_val = {"M": 1, "M2": 2, "M'": 3, "U": 1, "U2": 2, "U'": 3}
    move_map = {
        "M": {1: "M", 2: "M2", 3: "M'"},
        "U": {1: "U", 2: "U2", 3: "U'"},
    }
    stack: list[tuple[str, int]] = []
    for m in moves:
        face = m[0]
        val = turn_val[m]
        if stack and stack[-1][0] == face:
            prev_face, prev_val = stack.pop()
            new_val = (prev_val + val) % 4
            if new_val != 0:
                stack.append((face, new_val))
        else:
            stack.append((face, val))
    return [move_map[face][val] for face, val in stack]

# Piece movement on local indices 0..5:
# M: UF(0) -> DF(4) -> DB(5) -> UB(2) -> UF(0); UL(1) & UR(3) unchanged
M_EDGE = (4, 1, 0, 3, 5, 2)
M_PRIME_EDGE = (2, 1, 5, 3, 0, 4)
M2_EDGE = (5, 1, 4, 3, 2, 0)

# U: UF(0) -> UL(1) -> UB(2) -> UR(3) -> UF(0); DF(4) & DB(5) unchanged
U_EDGE = (1, 2, 3, 0, 4, 5)
U_PRIME_EDGE = (3, 0, 1, 2, 4, 5)
U2_EDGE = (2, 3, 0, 1, 4, 5)

M_EO_FLIP = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 5)

# Precomputed EO permutation lookup tables for 64 possible 6-bit EO bitmasks
EO_M = [0] * 64
EO_M_PRIME = [0] * 64
EO_M2 = [0] * 64
EO_U = [0] * 64
EO_U_PRIME = [0] * 64
EO_U2 = [0] * 64

for _eo in range(64):
    for _i in range(6):
        if (_eo >> _i) & 1:
            EO_M[_eo] |= (1 << M_EDGE[_i])
            EO_M_PRIME[_eo] |= (1 << M_PRIME_EDGE[_i])
            EO_M2[_eo] |= (1 << M2_EDGE[_i])
            EO_U[_eo] |= (1 << U_EDGE[_i])
            EO_U_PRIME[_eo] |= (1 << U_PRIME_EDGE[_i])
            EO_U2[_eo] |= (1 << U2_EDGE[_i])
    EO_M[_eo] ^= M_EO_FLIP
    EO_M_PRIME[_eo] ^= M_EO_FLIP

MOVE_NAMES = ("M", "M2", "M'", "U", "U2", "U'")
INVERSE_MOVES = {"M": "M'", "M'": "M", "M2": "M2", "U": "U'", "U'": "U", "U2": "U2"}


def _step_M(code: int) -> int:
    return (
        (M_EDGE[(code >> 12) & 7] << 12) |
        (M_EDGE[(code >> 9) & 7] << 9) |
        (EO_M[(code >> 3) & 63] << 3) |
        ((1 - ((code >> 2) & 1)) << 2) |
        (code & 3)
    )


def _step_M_prime(code: int) -> int:
    return (
        (M_PRIME_EDGE[(code >> 12) & 7] << 12) |
        (M_PRIME_EDGE[(code >> 9) & 7] << 9) |
        (EO_M_PRIME[(code >> 3) & 63] << 3) |
        ((1 - ((code >> 2) & 1)) << 2) |
        (code & 3)
    )


def _step_M2(code: int) -> int:
    return (
        (M2_EDGE[(code >> 12) & 7] << 12) |
        (M2_EDGE[(code >> 9) & 7] << 9) |
        (EO_M2[(code >> 3) & 63] << 3) |
        (((code >> 2) & 1) << 2) |
        (code & 3)
    )


def _step_U(code: int) -> int:
    return (
        (U_EDGE[(code >> 12) & 7] << 12) |
        (U_EDGE[(code >> 9) & 7] << 9) |
        (EO_U[(code >> 3) & 63] << 3) |
        (((code >> 2) & 1) << 2) |
        (((code & 3) + 3) & 3)
    )


def _step_U_prime(code: int) -> int:
    return (
        (U_PRIME_EDGE[(code >> 12) & 7] << 12) |
        (U_PRIME_EDGE[(code >> 9) & 7] << 9) |
        (EO_U_PRIME[(code >> 3) & 63] << 3) |
        (((code >> 2) & 1) << 2) |
        (((code & 3) + 1) & 3)
    )


def _step_U2(code: int) -> int:
    return (
        (U2_EDGE[(code >> 12) & 7] << 12) |
        (U2_EDGE[(code >> 9) & 7] << 9) |
        (EO_U2[(code >> 3) & 63] << 3) |
        (((code >> 2) & 1) << 2) |
        (((code & 3) + 2) & 3)
    )


_STEP_FNS = (_step_M, _step_M2, _step_M_prime, _step_U, _step_U2, _step_U_prime)


@dataclass(frozen=True)
class LSESolution:
    """Immutable solution representation for Last Six Edges (LSE) steps and full phase."""
    target: str
    moves: list[str]
    move_count: int
    case_name: str
    center_state: str = "axis_aligned"
    orientation: str = ""


@dataclass(frozen=True)
class LSEPath:
    """Represents a complete multi-step LSE solve path from 4a through 4c."""
    name: str
    step_4a: LSESolution
    step_4b: LSESolution
    step_4c: LSESolution
    total_moves: list[str]
    total_move_count: int
    orientation: str = ""


class LSEGraph:
    """In-memory transition graph of the 7,680 reachable states in the Roux LSE <M, U> subgroup."""

    _instance: Optional[LSEGraph] = None

    def __init__(self) -> None:
        self.state_to_id: Dict[int, int] = {}
        self.id_to_state: List[int] = []
        self.transitions: List[Tuple[int, int, int, int, int, int]] = []
        self._build_graph()

    @classmethod
    def get_instance(cls) -> LSEGraph:
        """Returns the singleton instance of LSEGraph, instantiating it if necessary."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _build_graph(self) -> None:
        # Canonical solved state: UL in slot 1 (UL), UR in slot 3 (UR), EO=0, CenterAxis=1, Corner=0
        start_code = (1 << 12) | (3 << 9) | (0 << 3) | (1 << 2) | 0

        self.state_to_id[start_code] = 0
        self.id_to_state.append(start_code)

        from collections import deque
        q = deque([start_code])

        while q:
            u = q.popleft()
            t0 = _step_M(u)
            t1 = _step_M2(u)
            t2 = _step_M_prime(u)
            t3 = _step_U(u)
            t4 = _step_U2(u)
            t5 = _step_U_prime(u)

            for nxt in (t0, t1, t2, t3, t4, t5):
                if nxt not in self.state_to_id:
                    self.state_to_id[nxt] = len(self.id_to_state)
                    self.id_to_state.append(nxt)
                    q.append(nxt)

        self.transitions.append((t0, t1, t2, t3, t4, t5))

        # Lazy-initialized tables for sub-step solvers
        self._tables_initialized = False
        # Aligned center targets (ca == 1)
        self.std_eo_targets: Set[int] = set()
        self.eolr_targets: Set[int] = set()
        self.eolr_b_targets: Set[int] = set()
        self.first_move_std_eo: Dict[int, int] = {}
        self.first_move_eolr: Dict[int, int] = {}
        self.first_move_eolr_b: Dict[int, int] = {}

        # Any center targets (aligned ca == 1 OR misaligned ca == 0)
        self.std_eo_any_targets: Set[int] = set()
        self.eolr_any_targets: Set[int] = set()
        self.eolr_b_any_targets: Set[int] = set()
        self.first_move_std_eo_any: Dict[int, int] = {}
        self.first_move_eolr_any: Dict[int, int] = {}
        self.first_move_eolr_b_any: Dict[int, int] = {}

        # Misaligned-only EOLR targets (ca == 0)
        self.eolr_mis_targets: Set[int] = set()
        self.first_move_eolr_mis: Dict[int, int] = {}

    def _ensure_4a_tables(self) -> None:
        """Precomputes backward BFS shortest paths for standard EO, EOLR, and EOLR-b on first access."""
        if self._tables_initialized:
            return

        for code in self.id_to_state:
            eo = (code >> 3) & 63
            ca = (code >> 2) & 1
            ul = (code >> 12) & 7
            ur = (code >> 9) & 7
            co = code & 3

            # 1. Aligned centers (ca == 1, eo == 0)
            if eo == 0 and ca == 1:
                self.std_eo_targets.add(code)
                self.std_eo_any_targets.add(code)
                if {ul, ur} == {4, 5}:
                    self.eolr_targets.add(code)
                    self.eolr_any_targets.add(code)
                if ul == (1 - co) % 4 and ur == (3 - co) % 4:
                    self.eolr_b_targets.add(code)
                    self.eolr_b_any_targets.add(code)

            # 2. Misaligned centers (ca == 0, UL/UR eo=0, 4 M-edges eo=1)
            elif ca == 0 and eo == (63 ^ (1 << ul) ^ (1 << ur)):
                self.std_eo_any_targets.add(code)
                if {ul, ur} == {4, 5}:
                    self.eolr_any_targets.add(code)
                    self.eolr_mis_targets.add(code)
                if ul == (1 - co) % 4 and ur == (3 - co) % 4:
                    self.eolr_b_any_targets.add(code)

        self._bfs_target_set(self.std_eo_targets, self.first_move_std_eo)
        self._bfs_target_set(self.eolr_targets, self.first_move_eolr)
        self._bfs_target_set(self.eolr_b_targets, self.first_move_eolr_b)

        self._bfs_target_set(self.std_eo_any_targets, self.first_move_std_eo_any)
        self._bfs_target_set(self.eolr_any_targets, self.first_move_eolr_any)
        self._bfs_target_set(self.eolr_b_any_targets, self.first_move_eolr_b_any)
        self._bfs_target_set(self.eolr_mis_targets, self.first_move_eolr_mis)
        self._tables_initialized = True

    def _bfs_target_set(self, target_set: Set[int], first_move_table: Dict[int, int]) -> None:
        """Runs multi-source backward BFS from target_set recording the first move toward targets."""
        from collections import deque
        visited: Set[int] = set(target_set)
        q = deque(target_set)

        inv_move_indices = (2, 1, 0, 5, 4, 3)
        while q:
            v = q.popleft()
            for m_idx in range(6):
                inv_m = inv_move_indices[m_idx]
                u = _STEP_FNS[inv_m](v)
                if u not in visited and u in self.state_to_id:
                    visited.add(u)
                    first_move_table[u] = m_idx
                    q.append(u)

    def _reconstruct_path(
        self, start_code: int, first_move_table: Dict[int, int], target_set: Set[int]
    ) -> tuple[list[str], int]:
        """Reconstructs shortest move sequence from start_code to target_set and returns (moves, end_code)."""
        if start_code in target_set:
            return [], start_code
        moves: list[str] = []
        curr = start_code
        while curr not in target_set:
            if curr not in first_move_table:
                break
            m_idx = first_move_table[curr]
            moves.append(MOVE_NAMES[m_idx])
            curr = _STEP_FNS[m_idx](curr)
        return moves, curr

    def __len__(self) -> int:
        return len(self.id_to_state)

    @property
    def num_states(self) -> int:
        return len(self.id_to_state)

    @staticmethod
    def encode_cube(cube: CubeState) -> int:
        """Encodes an arbitrary CubeState into the 15-bit integer index representing its LSE state."""
        ep_bytes = bytes(cube.ep)
        eo_bytes = bytes(cube.eo)
        c_bytes = bytes(cube.centers)
        cp_bytes = bytes(cube.cp)

        ul_slot = ep_bytes.find(b"\x01")
        ur_slot = ep_bytes.find(b"\x03")
        p_ul = SLOT_TO_LOCAL_ARR[ul_slot]
        p_ur = SLOT_TO_LOCAL_ARR[ur_slot]

        eo_mask = (
            eo_bytes[0]
            | (eo_bytes[1] << 1)
            | (eo_bytes[2] << 2)
            | (eo_bytes[3] << 3)
            | (eo_bytes[4] << 4)
            | (eo_bytes[6] << 5)
        )
        center_axis = 1 if c_bytes[0] in (0, 1) else 0
        corner_auf = cp_bytes[0] & 3

        return (p_ul << 12) | (p_ur << 9) | (eo_mask << 3) | (center_axis << 2) | corner_auf

    def contains_state(self, cube: CubeState) -> bool:
        """Checks if a cube state's LSE configuration is in the 7,680-state graph."""
        code = self.encode_cube(cube)
        return code in self.state_to_id

    def query_move(
        self,
        state_or_code: int | CubeState,
        target: str = "4a",
        allow_misoriented_centers: bool = False,
    ) -> Optional[str]:
        """Returns the single next optimal move in O(1) time (< 1µs) towards the given target."""
        self._ensure_4a_tables()
        code = self.encode_cube(state_or_code) if isinstance(state_or_code, CubeState) else state_or_code
        if target in ("4a", "eo", "standard_eo"):
            table = self.first_move_std_eo
        elif target == "eolr":
            table = self.first_move_eolr_any if allow_misoriented_centers else self.first_move_eolr
        elif target in ("4b", "eolr_b", "ul_ur"):
            table = self.first_move_eolr_b_any if allow_misoriented_centers else self.first_move_eolr_b
        else:
            raise ValueError(f"Unsupported target for single-move query: {target}")
        m_idx = table.get(code)
        return MOVE_NAMES[m_idx] if m_idx is not None else None

    @staticmethod
    def is_in_4c(code: int) -> bool:
        """Returns True if the LSE state is in Step 4c (EO solved and UL/UR placed in top slots)."""
        eo = (code >> 3) & 63
        if eo != 0:
            return False
        co = code & 3
        ul = (code >> 12) & 7
        ur = (code >> 9) & 7
        return ul == (1 - co) % 4 and ur == (3 - co) % 4

    def solve(
        self,
        cube: CubeState,
        target: str = "all",
        allow_misoriented_centers: bool = False,
    ) -> list[LSESolution]:
        """Solves the requested LSE target(s) for the given cube state."""
        valid_targets = ("all", "4a", "eo", "standard_eo", "eolr", "eolr_b", "4b", "ul_ur", "4c", "lse", "1look")
        if target not in valid_targets:
            raise ValueError(f"Unknown target '{target}'. Supported targets are: {valid_targets}")

        solutions: list[LSESolution] = []

        if cube.is_solved():
            if target in ("all", "4a", "eo", "standard_eo", "eolr", "eolr_b"):
                solutions.append(LSESolution(target="4a", moves=[], move_count=0, case_name="eolr_b", center_state="axis_aligned"))
            if target in ("all", "4b", "ul_ur"):
                solutions.append(LSESolution(target="4b", moves=[], move_count=0, case_name="ul_ur_solved", center_state="axis_aligned"))
            if target in ("all", "4c"):
                solutions.append(LSESolution(target="4c", moves=[], move_count=0, case_name="solved", center_state="axis_aligned"))
            if target in ("all", "lse", "1look"):
                solutions.append(LSESolution(target="lse", moves=[], move_count=0, case_name="solved", center_state="axis_aligned"))
            return solutions

        self._ensure_4a_tables()
        code = self.encode_cube(cube)

        if allow_misoriented_centers:
            eolr_targets = self.eolr_any_targets
            eolr_b_targets = self.eolr_b_any_targets
            first_move_eolr = self.first_move_eolr_any
            first_move_eolr_b = self.first_move_eolr_b_any
        else:
            eolr_targets = self.eolr_targets
            eolr_b_targets = self.eolr_b_targets
            first_move_eolr = self.first_move_eolr
            first_move_eolr_b = self.first_move_eolr_b

        # 1. Step 4a targets
        # Standard EO is strictly axis-aligned (learners/standard solvers do not use misoriented centers)
        if target in ("all", "4a", "eo", "standard_eo"):
            if code in self.std_eo_targets:
                case = "eolr_b" if code in self.eolr_b_targets else ("eolr" if code in self.eolr_targets else "standard_eo")
                solutions.append(LSESolution(target="4a", moves=[], move_count=0, case_name=case, center_state="axis_aligned"))
            else:
                moves_std, end_code = self._reconstruct_path(code, self.first_move_std_eo, self.std_eo_targets)
                solutions.append(LSESolution(target="4a", moves=moves_std, move_count=len(moves_std), case_name="standard_eo", center_state="axis_aligned"))

        if target in ("all", "4a", "eolr"):
            if code in eolr_targets or code in eolr_b_targets:
                case = "eolr_b" if code in eolr_b_targets else "eolr"
                if not any(s.case_name == case for s in solutions):
                    c_state = "axis_aligned" if ((code >> 2) & 1) == 1 else "misaligned"
                    solutions.append(LSESolution(target="4a", moves=[], move_count=0, case_name=case, center_state=c_state))
            else:
                moves_eolr, end_code = self._reconstruct_path(code, first_move_eolr, eolr_targets)
                c_state = "axis_aligned" if ((end_code >> 2) & 1) == 1 else "misaligned"
                solutions.append(LSESolution(target="4a", moves=moves_eolr, move_count=len(moves_eolr), case_name="eolr", center_state=c_state))

        if target in ("all", "4a", "eolr_b"):
            if code in eolr_b_targets:
                if not any(s.case_name == "eolr_b" for s in solutions):
                    c_state = "axis_aligned" if ((code >> 2) & 1) == 1 else "misaligned"
                    solutions.append(LSESolution(target="4a", moves=[], move_count=0, case_name="eolr_b", center_state=c_state))
            else:
                moves_eolr_b, end_code = self._reconstruct_path(code, first_move_eolr_b, eolr_b_targets)
                c_state = "axis_aligned" if ((end_code >> 2) & 1) == 1 else "misaligned"
                solutions.append(LSESolution(target="4a", moves=moves_eolr_b, move_count=len(moves_eolr_b), case_name="eolr_b", center_state=c_state))

        # 2. Step 4b target (UL/UR top slot placement)
        if target in ("all", "4b", "ul_ur"):
            if code in eolr_b_targets:
                c_state = "axis_aligned" if ((code >> 2) & 1) == 1 else "misaligned"
                solutions.append(LSESolution(target="4b", moves=[], move_count=0, case_name="ul_ur_solved", center_state=c_state))
            else:
                moves_4b, end_code = self._reconstruct_path(code, first_move_eolr_b, eolr_b_targets)
                c_state = "axis_aligned" if ((end_code >> 2) & 1) == 1 else "misaligned"
                solutions.append(LSESolution(target="4b", moves=moves_4b, move_count=len(moves_4b), case_name="ul_ur", center_state=c_state))

        # 3. Step 4c target (M-slice & center permutation)
        if target in ("all", "4c"):
            if self.is_in_4c(code):
                key = (int(cube.ep[0]), int(cube.ep[2]), int(cube.ep[4]), int(cube.ep[6]), int(cube.centers[0]), int(cube.centers[1]), int(cube.cp[0]))
                case_name, moves_4c = LSE_4C_SOLUTIONS[key]
                c_state = "axis_aligned" if ((code >> 2) & 1) == 1 else "misaligned"
                solutions.append(LSESolution(target="4c", moves=list(moves_4c), move_count=len(moves_4c), case_name=case_name, center_state=c_state))
            elif target == "4c":
                raise ValueError("Cube is not in a valid Step 4c state (UL/UR edges and EO must be solved).")

        # 4. 1-look global LSE target
        if target in ("all", "lse", "1look"):
            if self.is_in_4c(code):
                key = (int(cube.ep[0]), int(cube.ep[2]), int(cube.ep[4]), int(cube.ep[6]), int(cube.centers[0]), int(cube.centers[1]), int(cube.cp[0]))
                case_name, moves_4c = LSE_4C_SOLUTIONS[key]
                solutions.append(LSESolution(target="lse", moves=list(moves_4c), move_count=len(moves_4c), case_name="1look", center_state="axis_aligned"))
            else:
                path_4b, _ = self._reconstruct_path(code, self.first_move_eolr_b, self.eolr_b_targets)
                sim = cube.copy().apply_moves(path_4b)
                k_4c = (int(sim.ep[0]), int(sim.ep[2]), int(sim.ep[4]), int(sim.ep[6]), int(sim.centers[0]), int(sim.centers[1]), int(sim.cp[0]))
                if k_4c in LSE_4C_SOLUTIONS:
                    _, path_4c = LSE_4C_SOLUTIONS[k_4c]
                    full_moves = cancel_moves(path_4b + list(path_4c))
                    solutions.append(LSESolution(target="lse", moves=full_moves, move_count=len(full_moves), case_name="1look", center_state="axis_aligned"))
                else:
                    raise ValueError("Failed to reach a valid Step 4c configuration from 4b.")

        return solutions

    def solve_paths(self, cube: CubeState) -> dict[str, LSEPath]:
        """Solves and compares the 4 complete Roux LSE paths from Step 4a through 4c:
        - standard: Standard Roux (aligned standard EO -> 4b -> 4c)
        - eolr: EOLR Aligned (aligned EOLR -> 4b -> 4c)
        - eolr_misoriented: EOLR Misoriented (misaligned EOLR -> 4b -> 4c)
        - eolr_b: EOLR-b / 1-Look Global LSE (EOLR-b -> 4c)
        """
        self._ensure_4a_tables()
        code = self.encode_cube(cube)

        def _get_4c_sol(c: CubeState) -> tuple[str, list[str]]:
            k = (
                int(c.ep[0]),
                int(c.ep[2]),
                int(c.ep[4]),
                int(c.ep[6]),
                int(c.centers[0]),
                int(c.centers[1]),
                int(c.cp[0]),
            )
            if k in LSE_4C_SOLUTIONS:
                case_name, m_4c = LSE_4C_SOLUTIONS[k]
                return case_name, list(m_4c)
            raise ValueError(f"Cube state {k} not in valid Step 4c configuration.")

        if cube.is_solved():
            sol_4a = LSESolution(target="4a", moves=[], move_count=0, case_name="solved", center_state="axis_aligned")
            sol_4b = LSESolution(target="4b", moves=[], move_count=0, case_name="solved", center_state="axis_aligned")
            sol_4c = LSESolution(target="4c", moves=[], move_count=0, case_name="solved", center_state="axis_aligned")
            return {
                "standard": LSEPath("standard", sol_4a, sol_4b, sol_4c, [], 0),
                "eolr": LSEPath("eolr", sol_4a, sol_4b, sol_4c, [], 0),
                "eolr_misoriented": LSEPath("eolr_misoriented", sol_4a, sol_4b, sol_4c, [], 0),
                "eolr_b": LSEPath("eolr_b", sol_4a, sol_4b, sol_4c, [], 0),
            }

        if self.is_in_4c(code):
            c_name, m_4c = _get_4c_sol(cube)
            sol_4a = LSESolution(target="4a", moves=[], move_count=0, case_name="solved", center_state="axis_aligned")
            sol_4b = LSESolution(target="4b", moves=[], move_count=0, case_name="solved", center_state="axis_aligned")
            sol_4c = LSESolution(target="4c", moves=m_4c, move_count=len(m_4c), case_name=c_name, center_state="axis_aligned")
            return {
                "standard": LSEPath("standard", sol_4a, sol_4b, sol_4c, m_4c, len(m_4c)),
                "eolr": LSEPath("eolr", sol_4a, sol_4b, sol_4c, m_4c, len(m_4c)),
                "eolr_misoriented": LSEPath("eolr_misoriented", sol_4a, sol_4b, sol_4c, m_4c, len(m_4c)),
                "eolr_b": LSEPath("eolr_b", sol_4a, sol_4b, sol_4c, m_4c, len(m_4c)),
            }

        # 1. Standard Path
        m_std, _ = self._reconstruct_path(code, self.first_move_std_eo, self.std_eo_targets)
        c1_std = cube.copy().apply_moves(m_std)
        m_4b_std, _ = self._reconstruct_path(self.encode_cube(c1_std), self.first_move_eolr_b, self.eolr_b_targets)
        c2_std = c1_std.copy().apply_moves(m_4b_std)
        c4c_std, m_4c_std = _get_4c_sol(c2_std)
        tot_std = cancel_moves(m_std + m_4b_std + m_4c_std)
        path_std = LSEPath(
            name="standard",
            step_4a=LSESolution(target="4a", moves=m_std, move_count=len(m_std), case_name="standard_eo", center_state="axis_aligned"),
            step_4b=LSESolution(target="4b", moves=m_4b_std, move_count=len(m_4b_std), case_name="ul_ur" if m_4b_std else "ul_ur_solved", center_state="axis_aligned"),
            step_4c=LSESolution(target="4c", moves=m_4c_std, move_count=len(m_4c_std), case_name=c4c_std, center_state="axis_aligned"),
            total_moves=tot_std,
            total_move_count=len(tot_std),
        )

        # 2. EOLR Aligned Path
        m_eolr, _ = self._reconstruct_path(code, self.first_move_eolr, self.eolr_targets)
        c1_eolr = cube.copy().apply_moves(m_eolr)
        m_4b_eolr, _ = self._reconstruct_path(self.encode_cube(c1_eolr), self.first_move_eolr_b, self.eolr_b_targets)
        c2_eolr = c1_eolr.copy().apply_moves(m_4b_eolr)
        c4c_eolr, m_4c_eolr = _get_4c_sol(c2_eolr)
        tot_eolr = cancel_moves(m_eolr + m_4b_eolr + m_4c_eolr)
        path_eolr = LSEPath(
            name="eolr",
            step_4a=LSESolution(target="4a", moves=m_eolr, move_count=len(m_eolr), case_name="eolr", center_state="axis_aligned"),
            step_4b=LSESolution(target="4b", moves=m_4b_eolr, move_count=len(m_4b_eolr), case_name="ul_ur" if m_4b_eolr else "ul_ur_solved", center_state="axis_aligned"),
            step_4c=LSESolution(target="4c", moves=m_4c_eolr, move_count=len(m_4c_eolr), case_name=c4c_eolr, center_state="axis_aligned"),
            total_moves=tot_eolr,
            total_move_count=len(tot_eolr),
        )

        # 3. EOLR Misoriented Path
        m_mis, _ = self._reconstruct_path(code, self.first_move_eolr_mis, self.eolr_mis_targets)
        c1_mis = cube.copy().apply_moves(m_mis)
        m_4b_mis, _ = self._reconstruct_path(self.encode_cube(c1_mis), self.first_move_eolr_b, self.eolr_b_targets)
        c2_mis = c1_mis.copy().apply_moves(m_4b_mis)
        c4c_mis, m_4c_mis = _get_4c_sol(c2_mis)
        tot_mis = cancel_moves(m_mis + m_4b_mis + m_4c_mis)
        path_mis = LSEPath(
            name="eolr_misoriented",
            step_4a=LSESolution(target="4a", moves=m_mis, move_count=len(m_mis), case_name="eolr", center_state="misaligned"),
            step_4b=LSESolution(target="4b", moves=m_4b_mis, move_count=len(m_4b_mis), case_name="ul_ur" if m_4b_mis else "ul_ur_solved", center_state="axis_aligned"),
            step_4c=LSESolution(target="4c", moves=m_4c_mis, move_count=len(m_4c_mis), case_name=c4c_mis, center_state="axis_aligned"),
            total_moves=tot_mis,
            total_move_count=len(tot_mis),
        )

        # 4. EOLR-b / 1-Look Global LSE Path
        m_b, _ = self._reconstruct_path(code, self.first_move_eolr_b, self.eolr_b_targets)
        c1_b = cube.copy().apply_moves(m_b)
        c4c_b, m_4c_b = _get_4c_sol(c1_b)
        tot_b = cancel_moves(m_b + m_4c_b)
        path_b = LSEPath(
            name="eolr_b",
            step_4a=LSESolution(target="4a", moves=m_b, move_count=len(m_b), case_name="eolr_b", center_state="axis_aligned"),
            step_4b=LSESolution(target="4b", moves=[], move_count=0, case_name="ul_ur_solved", center_state="axis_aligned"),
            step_4c=LSESolution(target="4c", moves=m_4c_b, move_count=len(m_4c_b), case_name=c4c_b, center_state="axis_aligned"),
            total_moves=tot_b,
            total_move_count=len(tot_b),
        )

        return {
            "standard": path_std,
            "eolr": path_eolr,
            "eolr_misoriented": path_mis,
            "eolr_b": path_b,
        }


def solve_lse(
    scramble_or_cube: Optional[Union[str, CubeState]] = None,
    target: str = "all",
    allow_misoriented_centers: bool = False,
    orientation: Optional[Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any]] = None,
    *,
    cube: Optional[Union[str, CubeState]] = None,
) -> list[LSESolution]:
    """Surfaces optimal paths for Step 4a, 4b, 4c, and 1-look global LSE."""
    state = scramble_or_cube if scramble_or_cube is not None else cube
    if state is None:
        raise ValueError("Must provide either scramble_or_cube or cube")
    cube_obj = CubeState().apply_moves(state) if isinstance(state, str) else state

    resolved_ori, is_inspected = resolve_lse_orientation(cube_obj, orientation=orientation)
    c_canon = conjugate_lse_cube(cube_obj, orientation=resolved_ori, is_inspected=is_inspected)

    graph = LSEGraph.get_instance()
    canon_sols = graph.solve(
        c_canon,
        target=target,
        allow_misoriented_centers=allow_misoriented_centers,
    )

    translated_sols: list[LSESolution] = []
    for s in canon_sols:
        m_user = translate_lse_moves(s.moves, orientation=resolved_ori, is_inspected=is_inspected)
        translated_sols.append(
            LSESolution(
                target=s.target,
                moves=m_user,
                move_count=len(m_user),
                case_name=s.case_name,
                center_state=s.center_state,
                orientation=resolved_ori.rotations,
            )
        )
    return translated_sols


def solve_lse_paths(
    scramble_or_cube: Optional[Union[str, CubeState]] = None,
    orientation: Optional[Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any]] = None,
    *,
    cube: Optional[Union[str, CubeState]] = None,
) -> dict[str, LSEPath]:
    """Solves and compares the 4 complete Roux LSE paths from Step 4a through 4c:
    - standard: Standard Roux (aligned standard EO -> 4b -> 4c)
    - eolr: EOLR Aligned (aligned EOLR -> 4b -> 4c)
    - eolr_misoriented: EOLR Misoriented (misaligned EOLR -> 4b -> 4c)
    - eolr_b: EOLR-b / 1-Look Global LSE (EOLR-b -> 4c)
    """
    state = scramble_or_cube if scramble_or_cube is not None else cube
    if state is None:
        raise ValueError("Must provide either scramble_or_cube or cube")
    cube_obj = CubeState().apply_moves(state) if isinstance(state, str) else state

    resolved_ori, is_inspected = resolve_lse_orientation(cube_obj, orientation=orientation)
    c_canon = conjugate_lse_cube(cube_obj, orientation=resolved_ori, is_inspected=is_inspected)

    graph = LSEGraph.get_instance()
    canon_paths = graph.solve_paths(c_canon)

    translated_paths: dict[str, LSEPath] = {}
    for name, p in canon_paths.items():
        m_4a = translate_lse_moves(p.step_4a.moves, orientation=resolved_ori, is_inspected=is_inspected)
        m_4b = translate_lse_moves(p.step_4b.moves, orientation=resolved_ori, is_inspected=is_inspected)
        m_4c = translate_lse_moves(p.step_4c.moves, orientation=resolved_ori, is_inspected=is_inspected)
        m_tot = translate_lse_moves(p.total_moves, orientation=resolved_ori, is_inspected=is_inspected)

        sol_4a = LSESolution(
            target="4a",
            moves=m_4a,
            move_count=len(m_4a),
            case_name=p.step_4a.case_name,
            center_state=p.step_4a.center_state,
            orientation=resolved_ori.rotations,
        )
        sol_4b = LSESolution(
            target="4b",
            moves=m_4b,
            move_count=len(m_4b),
            case_name=p.step_4b.case_name,
            center_state=p.step_4b.center_state,
            orientation=resolved_ori.rotations,
        )
        sol_4c = LSESolution(
            target="4c",
            moves=m_4c,
            move_count=len(m_4c),
            case_name=p.step_4c.case_name,
            center_state=p.step_4c.center_state,
            orientation=resolved_ori.rotations,
        )
        translated_paths[name] = LSEPath(
            name=p.name,
            step_4a=sol_4a,
            step_4b=sol_4b,
            step_4c=sol_4c,
            total_moves=m_tot,
            total_move_count=len(m_tot),
            orientation=resolved_ori.rotations,
        )
    return translated_paths


def resolve_lse_orientation(
    cube: CubeState,
    orientation: Optional[Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any]] = None,
) -> tuple[RouxOrientation, bool]:
    """Resolves the solve orientation and frame type (inspected vs uninspected) for LSE solving.

    Determines the solve orientation from solved First Block and Second Block cubies when
    orientation is not explicitly provided. If orientation is explicitly provided, validates
    and confirms whether the cube is in the inspected hand frame or uninspected world frame.

    Args:
        cube: The cube state to analyze.
        orientation: Optional explicit orientation override (rotation string, color pair tuple,
            CanonicalSymmetry, RouxOrientation, or any object accepted by get_orientation).

    Returns:
        tuple[RouxOrientation, bool]:
            (resolved_orientation, is_inspected)
            where is_inspected is True if the solve is in the inspected hand frame (blocks on Left/Right),
            or False if in an uninspected world frame (omitted inspection rotations).

    Raises:
        ValueError: If an explicit orientation is provided but First Block or Second Block is not solved
            for that orientation, or if the orientation identifier is invalid.
    """
    if orientation is not None:
        ori = get_orientation(orientation)
        # Check inspected frame first (FB on Left, SB on Right)
        if ori.is_fb_solved(cube) and ori.is_sb_solved(cube):
            return ori, True
        # Check uninspected frame (blocks on other faces due to omitted inspection rotation)
        if ori.rotations:
            c_rot = cube.copy().apply_moves(ori.rotations)
            if ori.is_fb_solved(c_rot) and ori.is_sb_solved(c_rot):
                return ori, False
        raise ValueError(
            f"First Block and Second Block are not solved for specified orientation {ori.rotations!r} "
            f"(bottom={ori.bottom_color.name}, left={ori.left_color.name})"
        )

    if cube.is_solved():
        return CANONICAL_ORIENTATION, True

    # Auto-detection:
    # Build candidate list with dual-neutral orientations prioritized
    dn_set = set(get_dual_neutral_orientations())
    candidates = list(get_dual_neutral_orientations()) + [
        o for o in get_all_orientations() if o not in dn_set
    ]

    # 1. Check inspected hand frames (blocks on Left and Right)
    for candidate in candidates:
        if candidate.is_fb_solved(cube) and candidate.is_sb_solved(cube):
            return candidate, True

    # 2. Check uninspected world frames (omitted inspection rotations)
    for candidate in candidates:
        if candidate.rotations:
            c_rot = cube.copy().apply_moves(candidate.rotations)
            if candidate.is_fb_solved(c_rot) and candidate.is_sb_solved(c_rot):
                return candidate, False

    # Undetermined: fallback to canonical
    return CANONICAL_ORIENTATION, True


def conjugate_lse_cube(
    cube: CubeState,
    orientation: Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any] = "",
    is_inspected: bool = True,
) -> CubeState:
    """Conjugates a cube state into the canonical Yellow-bottom/Orange-left (standard Roux) frame.

    Transforms the cube state so that LSE pieces and M-slice centers map to canonical slots
    and expected orientations, making it directly queryable against the canonical LSE graph.

    Args:
        cube: The cube state to conjugate.
        orientation: The solve orientation of the cube (default canonical "").
        is_inspected: True if cube is in inspected hand frame (blocks on Left/Right),
            False if in uninspected world frame (omitted inspection rotation).

    Returns:
        A new canonical CubeState ready for LSEGraph.encode_cube and solving.
    """
    ori = get_orientation(orientation)

    # If in uninspected frame, re-map to inspected frame by applying inspection rotation
    inspected_cube: CubeState
    if not is_inspected:
        inspected_cube = cube.copy()
        if ori.rotations:
            inspected_cube.apply_moves(ori.rotations)
    else:
        inspected_cube = cube

    canon = CubeState()

    # 1. Map M-slice centers (U, D, F, B)
    # Canonical: top -> 0 (White), bottom -> 1 (Yellow), front -> 2 (Green), back -> 3 (Blue)
    color_map = {
        ori.top_color.value: 0,
        ori.bottom_color.value: 1,
        ori.front_color.value: 2,
        ori.back_color.value: 3,
    }
    for c_idx in (Center.U, Center.D, Center.F, Center.B):
        orig_val = int(inspected_cube.centers[c_idx])
        if orig_val in color_map:
            canon.centers[c_idx] = color_map[orig_val]

    # 2. Preserve U-layer corner AUF
    reference_oriented_cube = CubeState()
    if ori.rotations:
        reference_oriented_cube.apply_moves(ori.rotations)
    expected_u_corners = [int(reference_oriented_cube.cp[i]) for i in range(4)]

    current_ufl_corner = int(inspected_cube.cp[0])
    if current_ufl_corner in expected_u_corners:
        idx = expected_u_corners.index(current_ufl_corner)
        if idx == 3:
            canon.apply_move("U")
        elif idx == 2:
            canon.apply_move("U2")
        elif idx == 1:
            canon.apply_move("U'")

    # 3. Map the 6 LSE pieces and orientation deltas
    piece_map = {
        ori.uf_piece: (0, ori.uf_eo),
        ori.ul_piece: (1, ori.ul_eo),
        ori.ub_piece: (2, ori.ub_eo),
        ori.ur_piece: (3, ori.ur_eo),
        ori.df_piece: (4, ori.df_eo),
        ori.db_piece: (6, ori.db_eo),
    }
    for slot in (Edge.UF, Edge.UL, Edge.UB, Edge.UR, Edge.DF, Edge.DB):
        p = int(inspected_cube.ep[slot])
        if p in piece_map:
            canon_p, base_eo = piece_map[p]
            canon.ep[slot] = canon_p
            canon.eo[slot] = (int(inspected_cube.eo[slot]) - base_eo) % 2

    return canon


def translate_lse_moves(
    moves: Union[str, Sequence[str]],
    orientation: Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any] = "",
    is_inspected: bool = True,
) -> list[str]:
    """Translates move sequences generated in the canonical <M, U> frame back to the user's execution frame.

    Args:
        moves: Move sequence string or sequence of move strings.
        orientation: The solve orientation.
        is_inspected: True if the cube was in the inspected hand frame (blocks on Left/Right),
            False if in an uninspected world frame (omitted inspection rotation).

    Returns:
        List of move strings executable in the caller's frame.
    """
    parsed = _parse_move_sequence(moves)
    if is_inspected:
        return parsed

    ori = get_orientation(orientation)
    if not ori.rotations:
        return parsed

    if ori.symmetry is not None:
        return translate_moves_to_original(parsed, ori.symmetry)

    # General whole-cube rotation conjugation for FCN: m_orig = g * m_canon * g^-1
    inverse_rotations = " ".join(MoveParser.invert_moves(ori.rotations))
    translated: list[str] = []
    for m in parsed:
        c_target = CubeState().apply_moves(f"{ori.rotations} {m} {inverse_rotations}")
        matched: Optional[str] = None
        for cand in MOVES.keys():
            if CubeState().apply_move(cand) == c_target:
                matched = cand
                break
        translated.append(matched if matched is not None else m)
    return translated


__all__ = [
    "LSEGraph",
    "LSESolution",
    "LSEPath",
    "solve_lse",
    "solve_lse_paths",
    "resolve_lse_orientation",
    "conjugate_lse_cube",
    "translate_lse_moves",
    "cancel_moves",
]






"""RouxOrientation domain entity and orientation registry.

Foundational core tier representation of the 24 spatial frames (8 Dual-Neutral and 16 Full Color Neutral)
for the Roux speedcubing method. Encapsulates piece geometry, target face colors,
phase completion queries, piece properties, placement extraction, and dual-neutral symmetry.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Sequence, Union, Any
import numpy as np

from .constants import Color, Edge, Corner, Center
from .cube import CubeState
from .parser import MoveParser


class CanonicalSymmetry(Enum):
    """The 8 automorphisms of the x2y subgroup G for Dual-Neutrality."""
    I = ""
    Y = "y"
    Y2 = "y2"
    Y_PRIME = "y'"
    X2 = "x2"
    X2_Y = "x2 y"
    X2_Y2 = "x2 y2"
    X2_Y_PRIME = "x2 y'"

    @property
    def inspection_rotation(self) -> str:
        """The cube rotation string to inspect/orient the cube in this symmetry frame."""
        return self.value

    @property
    def bottom_color(self) -> Color:
        """Bottom face color for this dual-neutral First Block orientation."""
        return _ORIENTATIONS_BY_ROTATION[self.value].bottom_color

    @property
    def left_color(self) -> Color:
        """Left face color for this dual-neutral First Block orientation."""
        return _ORIENTATIONS_BY_ROTATION[self.value].left_color

    @property
    def inverse(self) -> CanonicalSymmetry:
        """The group-theoretic inverse symmetry g^-1 satisfying g * g^-1 = I."""
        return _INVERSE_MAP[self]


# Group inverses in G
_INVERSE_MAP: Dict[CanonicalSymmetry, CanonicalSymmetry] = {
    CanonicalSymmetry.I: CanonicalSymmetry.I,
    CanonicalSymmetry.Y: CanonicalSymmetry.Y_PRIME,
    CanonicalSymmetry.Y2: CanonicalSymmetry.Y2,
    CanonicalSymmetry.Y_PRIME: CanonicalSymmetry.Y,
    CanonicalSymmetry.X2: CanonicalSymmetry.X2,
    CanonicalSymmetry.X2_Y: CanonicalSymmetry.X2_Y,
    CanonicalSymmetry.X2_Y2: CanonicalSymmetry.X2_Y2,
    CanonicalSymmetry.X2_Y_PRIME: CanonicalSymmetry.X2_Y_PRIME,
}

_SYMMETRY_BY_ROTATION: Dict[str, CanonicalSymmetry] = {
    sym.value: sym for sym in CanonicalSymmetry
}


# -----------------------------------------------------------------------------
# Move Translation Tables (Static Automorphisms)
# -----------------------------------------------------------------------------

# Inverse automorphism: canonical -> original: m_orig = g * m_canon * g^-1
_CANON_TO_ORIG_MOVE: Dict[str, Dict[str, str]] = {
    "": {
        "B": "B", "B'": "B'", "B2": "B2",
        "D": "D", "D'": "D'", "D2": "D2",
        "E": "E", "E'": "E'", "E2": "E2",
        "F": "F", "F'": "F'", "F2": "F2",
        "L": "L", "L'": "L'", "L2": "L2",
        "M": "M", "M'": "M'", "M2": "M2",
        "R": "R", "R'": "R'", "R2": "R2",
        "S": "S", "S'": "S'", "S2": "S2",
        "U": "U", "U'": "U'", "U2": "U2",
        "b": "b", "b'": "b'", "b2": "b2",
        "d": "d", "d'": "d'", "d2": "d2",
        "f": "f", "f'": "f'", "f2": "f2",
        "l": "l", "l'": "l'", "l2": "l2",
        "r": "r", "r'": "r'", "r2": "r2",
        "u": "u", "u'": "u'", "u2": "u2",
        "x": "x", "x'": "x'", "x2": "x2",
        "y": "y", "y'": "y'", "y2": "y2",
        "z": "z", "z'": "z'", "z2": "z2",
    },
    "y": {
        "B": "L", "B'": "L'", "B2": "L2",
        "D": "D", "D'": "D'", "D2": "D2",
        "E": "E", "E'": "E'", "E2": "E2",
        "F": "R", "F'": "R'", "F2": "R2",
        "L": "F", "L'": "F'", "L2": "F2",
        "M": "S", "M'": "S'", "M2": "S2",
        "R": "B", "R'": "B'", "R2": "B2",
        "S": "M'", "S'": "M", "S2": "M2",
        "U": "U", "U'": "U'", "U2": "U2",
        "b": "l", "b'": "l'", "b2": "l2",
        "d": "d", "d'": "d'", "d2": "d2",
        "f": "r", "f'": "r'", "f2": "r2",
        "l": "f", "l'": "f'", "l2": "f2",
        "r": "b", "r'": "b'", "r2": "b2",
        "u": "u", "u'": "u'", "u2": "u2",
        "x": "z'", "x'": "z", "x2": "z2",
        "y": "y", "y'": "y'", "y2": "y2",
        "z": "x", "z'": "x'", "z2": "x2",
    },
    "y2": {
        "B": "F", "B'": "F'", "B2": "F2",
        "D": "D", "D'": "D'", "D2": "D2",
        "E": "E", "E'": "E'", "E2": "E2",
        "F": "B", "F'": "B'", "F2": "B2",
        "L": "R", "L'": "R'", "L2": "R2",
        "M": "M'", "M'": "M", "M2": "M2",
        "R": "L", "R'": "L'", "R2": "L2",
        "S": "S'", "S'": "S", "S2": "S2",
        "U": "U", "U'": "U'", "U2": "U2",
        "b": "f", "b'": "f'", "b2": "f2",
        "d": "d", "d'": "d'", "d2": "d2",
        "f": "b", "f'": "b'", "f2": "b2",
        "l": "r", "l'": "r'", "l2": "r2",
        "r": "l", "r'": "l'", "r2": "l2",
        "u": "u", "u'": "u'", "u2": "u2",
        "x": "x'", "x'": "x", "x2": "x2",
        "y": "y", "y'": "y'", "y2": "y2",
        "z": "z'", "z'": "z", "z2": "z2",
    },
    "y'": {
        "B": "R", "B'": "R'", "B2": "R2",
        "D": "D", "D'": "D'", "D2": "D2",
        "E": "E", "E'": "E'", "E2": "E2",
        "F": "L", "F'": "L'", "F2": "L2",
        "L": "B", "L'": "B'", "L2": "B2",
        "M": "S'", "M'": "S", "M2": "S2",
        "R": "F", "R'": "F'", "R2": "F2",
        "S": "M", "S'": "M'", "S2": "M2",
        "U": "U", "U'": "U'", "U2": "U2",
        "b": "r", "b'": "r'", "b2": "r2",
        "d": "d", "d'": "d'", "d2": "d2",
        "f": "l", "f'": "l'", "f2": "l2",
        "l": "b", "l'": "b'", "l2": "b2",
        "r": "f", "r'": "f'", "r2": "f2",
        "u": "u", "u'": "u'", "u2": "u2",
        "x": "z", "x'": "z'", "x2": "z2",
        "y": "y", "y'": "y'", "y2": "y2",
        "z": "x'", "z'": "x", "z2": "x2",
    },
    "x2": {
        "B": "F", "B'": "F'", "B2": "F2",
        "D": "U", "D'": "U'", "D2": "U2",
        "E": "E'", "E'": "E", "E2": "E2",
        "F": "B", "F'": "B'", "F2": "B2",
        "L": "L", "L'": "L'", "L2": "L2",
        "M": "M", "M'": "M'", "M2": "M2",
        "R": "R", "R'": "R'", "R2": "R2",
        "S": "S'", "S'": "S", "S2": "S2",
        "U": "D", "U'": "D'", "U2": "D2",
        "b": "f", "b'": "f'", "b2": "f2",
        "d": "u", "d'": "u'", "d2": "u2",
        "f": "b", "f'": "b'", "f2": "b2",
        "l": "l", "l'": "l'", "l2": "l2",
        "r": "r", "r'": "r'", "r2": "r2",
        "u": "d", "u'": "d'", "u2": "d2",
        "x": "x", "x'": "x'", "x2": "x2",
        "y": "y'", "y'": "y", "y2": "y2",
        "z": "z'", "z'": "z", "z2": "z2",
    },
    "x2 y": {
        "B": "L", "B'": "L'", "B2": "L2",
        "D": "U", "D'": "U'", "D2": "U2",
        "E": "E'", "E'": "E", "E2": "E2",
        "F": "R", "F'": "R'", "F2": "R2",
        "L": "B", "L'": "B'", "L2": "B2",
        "M": "S'", "M'": "S", "M2": "S2",
        "R": "F", "R'": "F'", "R2": "F2",
        "S": "M'", "S'": "M", "S2": "M2",
        "U": "D", "U'": "D'", "U2": "D2",
        "b": "l", "b'": "l'", "b2": "l2",
        "d": "u", "d'": "u'", "d2": "u2",
        "f": "r", "f'": "r'", "f2": "r2",
        "l": "b", "l'": "b'", "l2": "b2",
        "r": "f", "r'": "f'", "r2": "f2",
        "u": "d", "u'": "d'", "u2": "d2",
        "x": "z", "x'": "z'", "x2": "z2",
        "y": "y'", "y'": "y", "y2": "y2",
        "z": "x", "z'": "x'", "z2": "x2",
    },
    "x2 y2": {
        "B": "B", "B'": "B'", "B2": "B2",
        "D": "U", "D'": "U'", "D2": "U2",
        "E": "E'", "E'": "E", "E2": "E2",
        "F": "F", "F'": "F'", "F2": "F2",
        "L": "R", "L'": "R'", "L2": "R2",
        "M": "M'", "M'": "M", "M2": "M2",
        "R": "L", "R'": "L'", "R2": "L2",
        "S": "S", "S'": "S'", "S2": "S2",
        "U": "D", "U'": "D'", "U2": "D2",
        "b": "b", "b'": "b'", "b2": "b2",
        "d": "u", "d'": "u'", "d2": "u2",
        "f": "f", "f'": "f'", "f2": "f2",
        "l": "r", "l'": "r'", "l2": "r2",
        "r": "l", "r'": "l'", "r2": "l2",
        "u": "d", "u'": "d'", "u2": "d2",
        "x": "x'", "x'": "x", "x2": "x2",
        "y": "y'", "y'": "y", "y2": "y2",
        "z": "z", "z'": "z'", "z2": "z2",
    },
    "x2 y'": {
        "B": "R", "B'": "R'", "B2": "R2",
        "D": "U", "D'": "U'", "D2": "U2",
        "E": "E'", "E'": "E", "E2": "E2",
        "F": "L", "F'": "L'", "F2": "L2",
        "L": "F", "L'": "F'", "L2": "F2",
        "M": "S", "M'": "S'", "M2": "S2",
        "R": "B", "R'": "B'", "R2": "B2",
        "S": "M", "S'": "M'", "S2": "M2",
        "U": "D", "U'": "D'", "U2": "D2",
        "b": "r", "b'": "r'", "b2": "r2",
        "d": "u", "d'": "u'", "d2": "u2",
        "f": "l", "f'": "l'", "f2": "l2",
        "l": "f", "l'": "f'", "l2": "f2",
        "r": "b", "r'": "b'", "r2": "b2",
        "u": "d", "u'": "d'", "u2": "d2",
        "x": "z'", "x'": "z", "x2": "z2",
        "y": "y'", "y'": "y", "y2": "y2",
        "z": "x'", "z'": "x", "z2": "x2",
    },
}

_ORIG_TO_CANON_MOVE: Dict[str, Dict[str, str]] = {
    g: {v: k for k, v in table.items()} for g, table in _CANON_TO_ORIG_MOVE.items()
}


def _parse_move_sequence(moves: Union[str, Sequence[str]]) -> List[str]:
    """Helper to parse move input into a list of normalized move names."""
    if isinstance(moves, str):
        events = MoveParser.parse_string(moves)
        return [e.move for e in events]
    parsed: List[str] = []
    for item in moves:
        if isinstance(item, str):
            events = MoveParser.parse_string(item)
            for e in events:
                parsed.append(e.move)
        else:
            raise TypeError(f"Expected str move, got {type(item).__name__}")
    return parsed


def _resolve_symmetry(symmetry: Union[str, CanonicalSymmetry, RouxOrientation, Any]) -> CanonicalSymmetry:
    if isinstance(symmetry, CanonicalSymmetry):
        return symmetry
    if isinstance(symmetry, RouxOrientation):
        if symmetry.symmetry is not None:
            return symmetry.symmetry
        raise ValueError(f"Orientation {symmetry.rotations!r} is not dual-neutral")
    ori = get_orientation(symmetry)
    if ori.symmetry is not None:
        return ori.symmetry
    raise ValueError(f"Orientation {ori.rotations!r} is not dual-neutral")


def translate_moves_to_original(
    moves: Union[str, Sequence[str]],
    symmetry: Union[str, CanonicalSymmetry]
) -> List[str]:
    """Translates a move sequence from canonical/solve frame to original cube frame using the inverse automorphism.
    
    Formula: m_orig = g * m_canon * g^-1
    """
    sym = _resolve_symmetry(symmetry)
    parsed = _parse_move_sequence(moves)
    mapping = _CANON_TO_ORIG_MOVE[sym.value]
    return [mapping.get(m, m) for m in parsed]


def translate_moves_to_canonical(
    moves: Union[str, Sequence[str]],
    symmetry: Union[str, CanonicalSymmetry]
) -> List[str]:
    """Translates a move sequence from original cube frame into canonical frame using forward automorphism.
    
    Formula: m_canon = g^-1 * m_orig * g
    """
    sym = _resolve_symmetry(symmetry)
    parsed = _parse_move_sequence(moves)
    mapping = _ORIG_TO_CANON_MOVE[sym.value]
    return [mapping.get(m, m) for m in parsed]


def translate_moves(
    moves: Union[str, Sequence[str]],
    symmetry: Union[str, CanonicalSymmetry],
    inverse: bool = True
) -> List[str]:
    """Translates move sequences between canonical frame and original cube frame."""
    if inverse:
        return translate_moves_to_original(moves, symmetry)
    return translate_moves_to_canonical(moves, symmetry)


# -----------------------------------------------------------------------------
# Second Block Placement Data Record
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SBPlacement:
    """Structured position and orientation coordinates for Second Block pieces."""
    dr_slot: int
    dr_eo: int
    fr_slot: int
    fr_eo: int
    br_slot: int
    br_eo: int
    dfr_slot: int
    dfr_co: int
    dbr_slot: int
    dbr_co: int


# -----------------------------------------------------------------------------
# RouxOrientation Domain Entity
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RouxOrientation:
    """Immutable domain entity representing a canonical Roux solve orientation frame.
    
    Encapsulates face colors, expected piece identities, and orientations across
    First Block (Left 1x2x3), Second Block (Right 1x2x3), and Last Six Edges (LSE).
    """
    rotations: str
    left_color: Color
    bottom_color: Color
    front_color: Color
    back_color: Color
    right_color: Color
    top_color: Color

    # First Block (FB)
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

    # Second Block (SB)
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

    # Last Six Edges (LSE)
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

    @property
    def dfl_piece(self) -> int:
        """Alias for dlf_piece matching canonical CONTEXT.md corner naming."""
        return self.dlf_piece

    @property
    def dfl_co(self) -> int:
        """Alias for dlf_co matching canonical CONTEXT.md corner naming."""
        return self.dlf_co

    @property
    def dbr_piece(self) -> int:
        """Alias for drb_piece matching canonical CONTEXT.md corner naming."""
        return self.drb_piece

    @property
    def dbr_co(self) -> int:
        """Alias for drb_co matching canonical CONTEXT.md corner naming."""
        return self.drb_co

    @property
    def symmetry(self) -> Optional[CanonicalSymmetry]:
        """Returns the CanonicalSymmetry for dual-neutral orientations, or None for FCN."""
        return _SYMMETRY_BY_ROTATION.get(self.rotations)

    @property
    def is_dual_neutral(self) -> bool:
        """True if this orientation is one of the 8 dual-neutral orientations."""
        return self.rotations in _SYMMETRY_BY_ROTATION

    def is_fb_solved(self, cube: CubeState) -> bool:
        """Checks if First Block (FB: DL, FL, BL, DFL, DBL, L center) is solved in this orientation."""
        return bool(
            cube.centers[Center.L] == self.left_color.value and
            cube.cp[Corner.DLF] == self.dlf_piece and cube.co[Corner.DLF] == self.dlf_co and
            cube.cp[Corner.DBL] == self.dbl_piece and cube.co[Corner.DBL] == self.dbl_co and
            cube.ep[Edge.DL] == self.dl_piece and cube.eo[Edge.DL] == self.dl_eo and
            cube.ep[Edge.FL] == self.fl_piece and cube.eo[Edge.FL] == self.fl_eo and
            cube.ep[Edge.BL] == self.bl_piece and cube.eo[Edge.BL] == self.bl_eo
        )

    def is_dr_solved(self, cube: CubeState) -> bool:
        """Checks if the DR edge is solved in its designated slot with exact piece ID and orientation."""
        return bool(cube.ep[Edge.DR] == self.dr_piece and cube.eo[Edge.DR] == self.dr_eo)

    def is_back_pair_solved(self, cube: CubeState) -> bool:
        """Checks if the SB back pair (BR edge + DBR corner) is solved."""
        return bool(
            cube.ep[Edge.BR] == self.br_piece and cube.eo[Edge.BR] == self.br_eo and
            cube.cp[Corner.DRB] == self.drb_piece and cube.co[Corner.DRB] == self.drb_co
        )

    def is_front_pair_solved(self, cube: CubeState) -> bool:
        """Checks if the SB front pair (FR edge + DFR corner) is solved."""
        return bool(
            cube.ep[Edge.FR] == self.fr_piece and cube.eo[Edge.FR] == self.fr_eo and
            cube.cp[Corner.DFR] == self.dfr_piece and cube.co[Corner.DFR] == self.dfr_co
        )

    def is_sb_solved(self, cube: CubeState) -> bool:
        """Checks if all 5 Second Block (SB) pieces (DR, FR, BR, DFR, DBR) are simultaneously solved."""
        return bool(
            cube.ep[Edge.DR] == self.dr_piece and cube.eo[Edge.DR] == self.dr_eo and
            cube.ep[Edge.BR] == self.br_piece and cube.eo[Edge.BR] == self.br_eo and
            cube.ep[Edge.FR] == self.fr_piece and cube.eo[Edge.FR] == self.fr_eo and
            cube.cp[Corner.DRB] == self.drb_piece and cube.co[Corner.DRB] == self.drb_co and
            cube.cp[Corner.DFR] == self.dfr_piece and cube.co[Corner.DFR] == self.dfr_co
        )

    def get_m_slice_center_offset(self, cube: CubeState) -> int:
        """Determines the M-slice rotation offset in {0, 1, 2, 3} for the cube relative to orientation.
        0: Aligned with First Block.
        1: Rotated by M.
        2: Rotated by M2.
        3: Rotated by M'.
        """
        u_col = Color(int(cube.centers[Center.U]))
        if u_col == self.top_color:
            return 0
        elif u_col == self.back_color:
            return 1
        elif u_col == self.bottom_color:
            return 2
        elif u_col == self.front_color:
            return 3
        else:
            raise ValueError(
                f"Center U color {u_col.name} is not in M-slice for orientation {self.rotations!r}"
            )

    def is_center_aligned_sb_solved(self, cube: CubeState) -> bool:
        """Checks if Center-Aligned Second Block is strictly solved:
        1. All 5 Second Block pieces (DR, FR, BR, DFR, DRB) are solved.
        2. M-slice centers (U, D, F, B) are aligned with the U/D axis (offset 0 or 2).
        """
        if not self.is_sb_solved(cube):
            return False
        u_col = int(cube.centers[Center.U])
        return bool(u_col in (self.top_color.value, self.bottom_color.value))

    def is_center_axis_aligned(self, cube: CubeState) -> bool:
        """Checks if U and D centers occupy the U/D axis defined by this orientation."""
        u_c = int(cube.centers[Center.U])
        d_c = int(cube.centers[Center.D])
        axis_colors = (self.top_color.value, self.bottom_color.value)
        return bool(u_c in axis_colors and d_c in axis_colors)

    def count_bad_edges(self, cube: CubeState) -> int:
        """Counts how many of the 6 LSE edges have bad orientation."""
        expected_eos = (self.uf_eo, self.ub_eo, self.ul_eo, self.ur_eo, self.df_eo, self.db_eo)
        lse_edges = (Edge.UF, Edge.UB, Edge.UL, Edge.UR, Edge.DF, Edge.DB)
        return sum(1 for e, exp_eo in zip(lse_edges, expected_eos) if cube.eo[e] != exp_eo)

    def is_eo_solved(self, cube: CubeState) -> bool:
        """Checks if all 6 LSE edges are oriented along the U/D axis."""
        expected_eos = (self.uf_eo, self.ub_eo, self.ul_eo, self.ur_eo, self.df_eo, self.db_eo)
        lse_edges = (Edge.UF, Edge.UB, Edge.UL, Edge.UR, Edge.DF, Edge.DB)
        return bool(all(cube.eo[e] == exp_eo for e, exp_eo in zip(lse_edges, expected_eos)))

    def is_ul_ur_solved(self, cube: CubeState) -> bool:
        """Checks if UL and UR edges are placed in their correct slots relative to U-layer corners (up to AUF)."""
        expected_cp, expected_co = _U_CORNERS_BY_ROTATION[self.rotations]
        for u in ("", "U", "U2", "U'"):
            test_c = cube.copy()
            if u:
                test_c.apply_move(u)

            if test_c.eo[Edge.UL] != self.ul_eo or test_c.eo[Edge.UR] != self.ur_eo:
                continue
            if test_c.ep[Edge.UL] != self.ul_piece or test_c.ep[Edge.UR] != self.ur_piece:
                continue
            if tuple(int(x) for x in test_c.co[0:4]) != expected_co:
                continue
            if tuple(int(x) for x in test_c.cp[0:4]) != expected_cp:
                continue

            return True

        return False

    def extract_sb_placement(self, cube: CubeState) -> SBPlacement:
        """Extracts Second Block piece placement and orientation deltas relative to this orientation frame."""
        cp = cube.cp.tolist()
        co = cube.co.tolist()
        ep = cube.ep.tolist()
        eo = cube.eo.tolist()

        dfr_slot = cp.index(self.dfr_piece)
        dbr_slot = cp.index(self.drb_piece)
        dr_slot = ep.index(self.dr_piece)
        fr_slot = ep.index(self.fr_piece)
        br_slot = ep.index(self.br_piece)

        return SBPlacement(
            dr_slot=dr_slot,
            dr_eo=(eo[dr_slot] - self.dr_eo) % 2,
            fr_slot=fr_slot,
            fr_eo=(eo[fr_slot] - self.fr_eo) % 2,
            br_slot=br_slot,
            br_eo=(eo[br_slot] - self.br_eo) % 2,
            dfr_slot=dfr_slot,
            dfr_co=(co[dfr_slot] - self.dfr_co) % 3,
            dbr_slot=dbr_slot,
            dbr_co=(co[dbr_slot] - self.drb_co) % 3,
        )


def _precompute_u_corners(rotations_list: Sequence[str]) -> Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Precomputes U-layer corner permutation and orientation for all orientations."""
    res = {}
    for rot in rotations_list:
        target = CubeState()
        if rot:
            target.apply_moves(rot)
        res[rot] = (
            tuple(int(x) for x in target.cp[0:4]),
            tuple(int(x) for x in target.co[0:4]),
        )
    return res


_ROTATIONS_LIST = [
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

_U_CORNERS_BY_ROTATION: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]] = _precompute_u_corners(_ROTATIONS_LIST)


def _generate_all_orientations() -> Dict[str, RouxOrientation]:
    """Precomputes exact piece IDs and orientations for all 24 orientations."""
    defs: Dict[str, RouxOrientation] = {}
    for ori in _ROTATIONS_LIST:
        c = CubeState()
        if ori:
            c.apply_moves(ori)

        orientation = RouxOrientation(
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
        defs[ori] = orientation

    return defs


# Precomputed registries
_ORIENTATIONS_BY_ROTATION: Dict[str, RouxOrientation] = _generate_all_orientations()

_DUAL_NEUTRAL_ROTATIONS = (
    "", "y", "y2", "y'",
    "x2", "x2 y", "x2 y2", "x2 y'"
)

_ORIENTATIONS_BY_COLOR_PAIR: Dict[Tuple[Color, Color], RouxOrientation] = {
    (o.bottom_color, o.left_color): o for o in _ORIENTATIONS_BY_ROTATION.values()
}

_ALL_ORIENTATIONS_LIST: Tuple[RouxOrientation, ...] = tuple(_ORIENTATIONS_BY_ROTATION.values())
_DUAL_NEUTRAL_ORIENTATIONS_LIST: Tuple[RouxOrientation, ...] = tuple(
    _ORIENTATIONS_BY_ROTATION[r] for r in _DUAL_NEUTRAL_ROTATIONS
)


def get_all_orientations() -> Sequence[RouxOrientation]:
    """Returns all 24 canonical orientations in the Full Color Neutral set."""
    return _ALL_ORIENTATIONS_LIST


def get_dual_neutral_orientations() -> Sequence[RouxOrientation]:
    """Returns the 8 dual-neutral orientations (White/Yellow on bottom, First Block on L)."""
    return _DUAL_NEUTRAL_ORIENTATIONS_LIST


def get_orientation(
    identifier: Union[str, RouxOrientation, CanonicalSymmetry, Tuple[Color, Color], Any]
) -> RouxOrientation:
    """Resolves an orientation by instance, rotation string, CanonicalSymmetry, or (bottom_color, left_color) pair.
    
    Args:
        identifier: RouxOrientation instance, CanonicalSymmetry enum, rotation string (e.g. "", "y", "x2 y"),
                    or (bottom_color, left_color) tuple.
    """
    if isinstance(identifier, RouxOrientation):
        return identifier

    if isinstance(identifier, CanonicalSymmetry):
        return _ORIENTATIONS_BY_ROTATION[identifier.value]

    if hasattr(identifier, "value") and isinstance(getattr(identifier, "value"), str):
        val = getattr(identifier, "value")
        if val in _ORIENTATIONS_BY_ROTATION:
            return _ORIENTATIONS_BY_ROTATION[val]

    if hasattr(identifier, "rotations") and isinstance(getattr(identifier, "rotations"), str):
        rot = getattr(identifier, "rotations")
        if rot in _ORIENTATIONS_BY_ROTATION:
            return _ORIENTATIONS_BY_ROTATION[rot]

    if isinstance(identifier, tuple) and len(identifier) == 2:
        c1, c2 = identifier
        if isinstance(c1, int) and not isinstance(c1, Color):
            try:
                c1 = Color(c1)
            except ValueError:
                pass
        if isinstance(c2, int) and not isinstance(c2, Color):
            try:
                c2 = Color(c2)
            except ValueError:
                pass
        if isinstance(c1, Color) and isinstance(c2, Color):
            if (c1, c2) in _ORIENTATIONS_BY_COLOR_PAIR:
                return _ORIENTATIONS_BY_COLOR_PAIR[(c1, c2)]
            raise ValueError(f"No valid orientation with bottom={c1.name}, left={c2.name}")

    if isinstance(identifier, str):
        normalized = identifier.strip()
        if normalized in _ORIENTATIONS_BY_ROTATION:
            return _ORIENTATIONS_BY_ROTATION[normalized]
        # Check if space-collapsed e.g. "x2y" -> "x2 y"
        normalized_lower = normalized.lower()
        alias_map = {
            "identity": "",
            "i": "",
            "x2y": "x2 y",
            "x2y2": "x2 y2",
            "x2y'": "x2 y'",
        }
        if normalized_lower in alias_map:
            return _ORIENTATIONS_BY_ROTATION[alias_map[normalized_lower]]

        if "-" in normalized:
            parts = normalized.split("-")
            if len(parts) == 2:
                try:
                    c_bottom = Color[parts[0].upper()]
                    c_left = Color[parts[1].upper()]
                    if (c_bottom, c_left) in _ORIENTATIONS_BY_COLOR_PAIR:
                        return _ORIENTATIONS_BY_COLOR_PAIR[(c_bottom, c_left)]
                except KeyError:
                    pass

    raise ValueError(f"Unrecognized orientation identifier: {identifier!r}")


__all__ = [
    "CanonicalSymmetry",
    "SBPlacement",
    "RouxOrientation",
    "get_all_orientations",
    "get_dual_neutral_orientations",
    "get_orientation",
    "translate_moves",
    "translate_moves_to_original",
    "translate_moves_to_canonical",
]

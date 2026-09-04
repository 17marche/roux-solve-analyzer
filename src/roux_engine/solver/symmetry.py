"""Canonical Symmetry Re-Mapping for Roux First Block.

Applies the 8-element x2y automorphism group G = {I, y, y2, y', x2, x2y, x2y2, x2y'}
to evaluate all 8 dual-neutral First Blocks (White/Yellow on bottom, First Block on Left)
against a single canonical First Block pattern database.
"""

from __future__ import annotations
from enum import Enum
from typing import Union, Tuple, Sequence, List, Dict, Optional

from ..core.constants import Color, Edge, Corner, Center
from ..core.cube import CubeState
from ..core.parser import MoveParser
from ..segmenter.fb_detector import ALL_BLOCK_DEFINITIONS, BlockDefinition
from .fb_indexer import FBIndexer, FBPlacement


class CanonicalSymmetry(Enum):
    """The 8 automorphisms of the x2y subgroup G."""
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
        return ALL_BLOCK_DEFINITIONS[self.value].bottom_color

    @property
    def left_color(self) -> Color:
        """Left face color for this dual-neutral First Block orientation."""
        return ALL_BLOCK_DEFINITIONS[self.value].left_color

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

# Precomputed lookup maps
_STRING_TO_SYMMETRY: Dict[str, CanonicalSymmetry] = {
    "": CanonicalSymmetry.I,
    "i": CanonicalSymmetry.I,
    "identity": CanonicalSymmetry.I,
    "y": CanonicalSymmetry.Y,
    "y2": CanonicalSymmetry.Y2,
    "y'": CanonicalSymmetry.Y_PRIME,
    "x2": CanonicalSymmetry.X2,
    "x2 y": CanonicalSymmetry.X2_Y,
    "x2y": CanonicalSymmetry.X2_Y,
    "x2 y2": CanonicalSymmetry.X2_Y2,
    "x2y2": CanonicalSymmetry.X2_Y2,
    "x2 y'": CanonicalSymmetry.X2_Y_PRIME,
    "x2y'": CanonicalSymmetry.X2_Y_PRIME,
}

_COLOR_PAIR_TO_SYMMETRY: Dict[Tuple[Color, Color], CanonicalSymmetry] = {
    (sym.bottom_color, sym.left_color): sym for sym in CanonicalSymmetry
}


def get_all_symmetries() -> List[CanonicalSymmetry]:
    """Returns all 8 dual-neutral symmetries in the x2y subgroup G."""
    return list(CanonicalSymmetry)


def get_symmetry(identifier: Union[str, CanonicalSymmetry, Tuple[Color, Color]]) -> CanonicalSymmetry:
    """Resolves a symmetry by rotation string, enum value, or (bottom_color, left_color) pair.
    
    Args:
        identifier: Can be:
            - CanonicalSymmetry instance
            - Rotation string e.g. "x2 y", "x2y", "y'", "identity", ""
            - Color pair tuple e.g. (Color.WHITE, Color.BLUE)
            - Hyphenated color string e.g. "WHITE-BLUE", "yellow-green"
    """
    if isinstance(identifier, CanonicalSymmetry):
        return identifier

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
            if (c1, c2) in _COLOR_PAIR_TO_SYMMETRY:
                return _COLOR_PAIR_TO_SYMMETRY[(c1, c2)]
            raise ValueError(f"No dual-neutral symmetry with colors bottom={c1.name}, left={c2.name}")

    if isinstance(identifier, str):
        normalized = identifier.strip().lower()
        if normalized in _STRING_TO_SYMMETRY:
            return _STRING_TO_SYMMETRY[normalized]

        # Check hyphenated colors e.g. "white-blue" or "yellow-green"
        if "-" in normalized:
            parts = normalized.split("-")
            if len(parts) == 2:
                try:
                    c_bottom = Color[parts[0].upper()]
                    c_left = Color[parts[1].upper()]
                    if (c_bottom, c_left) in _COLOR_PAIR_TO_SYMMETRY:
                        return _COLOR_PAIR_TO_SYMMETRY[(c_bottom, c_left)]
                except KeyError:
                    pass

        # Check if uppercase matches
        if identifier in _STRING_TO_SYMMETRY:
            return _STRING_TO_SYMMETRY[identifier]

    raise ValueError(f"Unrecognized symmetry identifier: {identifier!r}")


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

# Forward automorphism: original -> canonical: m_canon = g^-1 * m_orig * g
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


def translate_moves_to_original(
    moves: Union[str, Sequence[str]],
    symmetry: Union[str, CanonicalSymmetry]
) -> List[str]:
    """Translates a move sequence from canonical/solve frame to original cube frame using the inverse automorphism.
    
    Formula: m_orig = g * m_canon * g^-1
    """
    sym = get_symmetry(symmetry)
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
    sym = get_symmetry(symmetry)
    parsed = _parse_move_sequence(moves)
    mapping = _ORIG_TO_CANON_MOVE[sym.value]
    return [mapping.get(m, m) for m in parsed]


def translate_moves(
    moves: Union[str, Sequence[str]],
    symmetry: Union[str, CanonicalSymmetry],
    inverse: bool = True
) -> List[str]:
    """Translates move sequences between canonical frame and original cube frame.
    
    Args:
        moves: String or sequence of moves.
        symmetry: Target symmetry identifier.
        inverse: If True (default), translates canonical -> original (inverse automorphism).
                 If False, translates original -> canonical (forward automorphism).
    """
    if inverse:
        return translate_moves_to_original(moves, symmetry)
    return translate_moves_to_canonical(moves, symmetry)


# -----------------------------------------------------------------------------
# Cube State Conjugation & Canonical Mapping
# -----------------------------------------------------------------------------

def _ensure_inspected_cube(
    cube: CubeState,
    sym: CanonicalSymmetry,
    inspected: bool
) -> CubeState:
    """Helper to return cube in the symmetry inspection frame."""
    if inspected:
        return cube
    c_inspected = cube.copy()
    if sym.inspection_rotation:
        c_inspected.apply_moves(sym.inspection_rotation)
    return c_inspected


def extract_canonical_placement(
    cube: CubeState,
    symmetry: Union[str, CanonicalSymmetry],
    inspected: bool = False
) -> FBPlacement:
    """Extracts FBPlacement mapped into the canonical First Block coordinate frame.
    
    Args:
        cube: CubeState to extract from.
        symmetry: Target symmetry identifier.
        inspected: Set True if cube is already in the inspected orientation frame.
                   Default False (cube is in original/scramble frame).
    """
    sym = get_symmetry(symmetry)
    b = ALL_BLOCK_DEFINITIONS[sym.value]
    c_inspected = _ensure_inspected_cube(cube, sym, inspected)

    cp = c_inspected.cp.tolist()
    co = c_inspected.co.tolist()
    ep = c_inspected.ep.tolist()
    eo = c_inspected.eo.tolist()

    dl_slot = ep.index(b.dl_piece)
    fl_slot = ep.index(b.fl_piece)
    bl_slot = ep.index(b.bl_piece)
    dlf_slot = cp.index(b.dlf_piece)
    dbl_slot = cp.index(b.dbl_piece)

    dl_eo = (eo[dl_slot] - b.dl_eo) % 2
    fl_eo = (eo[fl_slot] - b.fl_eo) % 2
    bl_eo = (eo[bl_slot] - b.bl_eo) % 2
    dlf_co = (co[dlf_slot] - b.dlf_co) % 3
    dbl_co = (co[dbl_slot] - b.dbl_co) % 3

    return FBPlacement(
        dl_slot=dl_slot, dl_eo=dl_eo,
        fl_slot=fl_slot, fl_eo=fl_eo,
        bl_slot=bl_slot, bl_eo=bl_eo,
        dlf_slot=dlf_slot, dlf_co=dlf_co,
        dbl_slot=dbl_slot, dbl_co=dbl_co,
    )


def conjugate_cube(
    cube: CubeState,
    symmetry: Union[str, CanonicalSymmetry],
    inspected: bool = False
) -> CubeState:
    """Conjugates a cube state into the canonical First Block frame for the given symmetry.
    
    The resulting CubeState places the target First Block pieces into canonical slots/orientations,
    making it directly queryable against the canonical First Block pattern database.
    
    Args:
        cube: CubeState to conjugate.
        symmetry: Target symmetry identifier.
        inspected: Set True if cube is already in inspected orientation frame.
    """
    sym = get_symmetry(symmetry)
    placement = extract_canonical_placement(cube, sym, inspected=inspected)
    index = FBIndexer.encode_placement(placement)
    return FBIndexer.decode(index)


def is_fb_solved_for_symmetry(
    cube: CubeState,
    symmetry: Union[str, CanonicalSymmetry],
    inspected: bool = False
) -> bool:
    """Checks whether the First Block corresponding to symmetry is completely solved on cube.
    
    Args:
        cube: CubeState to inspect.
        symmetry: Target symmetry identifier.
        inspected: Set True if cube has already been rotated by symmetry.inspection_rotation.
    """
    sym = get_symmetry(symmetry)
    b = ALL_BLOCK_DEFINITIONS[sym.value]
    c_inspected = _ensure_inspected_cube(cube, sym, inspected)

    if c_inspected.ep[Edge.DL] != b.dl_piece or c_inspected.eo[Edge.DL] != b.dl_eo:
        return False
    if c_inspected.ep[Edge.FL] != b.fl_piece or c_inspected.eo[Edge.FL] != b.fl_eo:
        return False
    if c_inspected.ep[Edge.BL] != b.bl_piece or c_inspected.eo[Edge.BL] != b.bl_eo:
        return False
    if c_inspected.cp[Corner.DLF] != b.dlf_piece or c_inspected.co[Corner.DLF] != b.dlf_co:
        return False
    if c_inspected.cp[Corner.DBL] != b.dbl_piece or c_inspected.co[Corner.DBL] != b.dbl_co:
        return False
    return True


__all__ = [
    "CanonicalSymmetry",
    "get_all_symmetries",
    "get_symmetry",
    "translate_moves",
    "translate_moves_to_original",
    "translate_moves_to_canonical",
    "extract_canonical_placement",
    "conjugate_cube",
    "is_fb_solved_for_symmetry",
]

"""Canonical Symmetry Re-Mapping for Roux First Block.

Applies the 8-element x2y automorphism group G = {I, y, y2, y', x2, x2y, x2y2, x2y'}
to evaluate all 8 dual-neutral First Blocks (White/Yellow on bottom, First Block on Left)
against a single canonical First Block pattern database.
"""

from __future__ import annotations
from typing import Union, Tuple, Sequence, List, Dict, Optional

from ..core.constants import Color, Edge, Corner, Center
from ..core.cube import CubeState
from ..core.orientation import (
    CanonicalSymmetry,
    get_orientation,
    translate_moves,
    translate_moves_to_original,
    translate_moves_to_canonical,
)
from .fb_indexer import FBIndexer, FBPlacement

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
    ori = get_orientation(sym)
    c_inspected = _ensure_inspected_cube(cube, sym, inspected)

    cp = c_inspected.cp.tolist()
    co = c_inspected.co.tolist()
    ep = c_inspected.ep.tolist()
    eo = c_inspected.eo.tolist()

    dl_slot = ep.index(ori.dl_piece)
    fl_slot = ep.index(ori.fl_piece)
    bl_slot = ep.index(ori.bl_piece)
    dlf_slot = cp.index(ori.dlf_piece)
    dbl_slot = cp.index(ori.dbl_piece)

    dl_eo = (eo[dl_slot] - ori.dl_eo) % 2
    fl_eo = (eo[fl_slot] - ori.fl_eo) % 2
    bl_eo = (eo[bl_slot] - ori.bl_eo) % 2
    dlf_co = (co[dlf_slot] - ori.dlf_co) % 3
    dbl_co = (co[dbl_slot] - ori.dbl_co) % 3

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
    ori = get_orientation(sym)
    c_inspected = _ensure_inspected_cube(cube, sym, inspected)

    if c_inspected.ep[Edge.DL] != ori.dl_piece or c_inspected.eo[Edge.DL] != ori.dl_eo:
        return False
    if c_inspected.ep[Edge.FL] != ori.fl_piece or c_inspected.eo[Edge.FL] != ori.fl_eo:
        return False
    if c_inspected.ep[Edge.BL] != ori.bl_piece or c_inspected.eo[Edge.BL] != ori.bl_eo:
        return False
    if c_inspected.cp[Corner.DLF] != ori.dlf_piece or c_inspected.co[Corner.DLF] != ori.dlf_co:
        return False
    if c_inspected.cp[Corner.DBL] != ori.dbl_piece or c_inspected.co[Corner.DBL] != ori.dbl_co:
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

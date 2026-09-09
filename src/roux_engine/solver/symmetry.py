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
    try:
        ori = get_orientation(identifier)
        if ori.symmetry is not None:
            return ori.symmetry
        raise ValueError(f"Orientation {ori.rotations!r} is not dual-neutral")
    except ValueError:
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
    return ori.is_fb_solved(c_inspected)


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

"""First Block (FB) State Indexer for Roux method.

Maps any Rubik's Cube state's First Block pieces (DL, FL, BL edges and DFL, DBL corners)
to a dense integer in [0, 5,322,239] and back.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from ..core.constants import Corner, Edge, Center
from ..core.cube import CubeState

# -----------------------------------------------------------------------------
# Constants & Slot Mappings
# -----------------------------------------------------------------------------

TOTAL_STATES: int = 5_322_240
NUM_EDGE_CONFIGS: int = 10_560
NUM_CORNER_CONFIGS: int = 504
CANONICAL_SOLVED_INDEX: int = 0

# Canonical solved corner slots: DLF is slot 4, DBL is slot 5.
# Slot bijection maps slot 4 -> 0, slot 5 -> 1 so solved corners map to (0, 1).
CORNER_SLOT_MAP: Tuple[int, ...] = (2, 3, 4, 5, 0, 1, 6, 7)
CORNER_SLOT_UNMAP: Tuple[int, ...] = (4, 5, 0, 1, 2, 3, 6, 7)

# Canonical solved edge slots: DL is slot 5, FL is slot 8, BL is slot 9.
# Slot bijection maps slot 5 -> 0, slot 8 -> 1, slot 9 -> 2 so solved edges map to (0, 1, 2).
EDGE_SLOT_MAP: Tuple[int, ...] = (3, 4, 5, 6, 7, 0, 8, 9, 1, 2, 10, 11)
EDGE_SLOT_UNMAP: Tuple[int, ...] = (5, 8, 9, 0, 1, 2, 3, 4, 6, 7, 10, 11)

# Precomputed binomial coefficients for combinadic ranking
# C(n, k) = n! / (k! * (n - k)!)
_C_T1_2: Tuple[int, ...] = tuple(math.comb(n, 2) for n in range(13))
_C_T2_3: Tuple[int, ...] = tuple(math.comb(n, 3) for n in range(13))

# Precomputed map for combinadic unranking: comb in [0..219] -> (t0, t1, t2) with t0 < t1 < t2
_COMB_TO_SLOTS: List[Tuple[int, int, int]] = [(0, 0, 0)] * 220
for _t2 in range(2, 12):
    for _t1 in range(1, _t2):
        for _t0 in range(_t1):
            _c = _t0 + _C_T1_2[_t1] + _C_T2_3[_t2]
            _COMB_TO_SLOTS[_c] = (_t0, _t1, _t2)

# Permutations of 3 elements for factoradic ranking/unranking
_PERM_TO_PIECES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),  # perm 0 (identity)
    (0, 2, 1),  # perm 1
    (1, 0, 2),  # perm 2
    (1, 2, 0),  # perm 3
    (2, 0, 1),  # perm 4
    (2, 1, 0),  # perm 5
)

# Non-FB pieces for filling decoded states
_REMAINING_CORNERS: Tuple[int, ...] = (0, 1, 2, 3, 6, 7)
_REMAINING_EDGES: Tuple[int, ...] = (0, 1, 2, 3, 4, 6, 7, 10, 11)


@dataclass(frozen=True)
class FBPlacement:
    """Structured position and orientation coordinates for First Block pieces."""
    dl_slot: int
    dl_eo: int
    fl_slot: int
    fl_eo: int
    bl_slot: int
    bl_eo: int
    dlf_slot: int
    dlf_co: int
    dbl_slot: int
    dbl_co: int


class FBIndexer:
    """Bidirectional, zero-waste indexer for Roux First Block states."""

    TOTAL_STATES: int = TOTAL_STATES
    NUM_EDGE_CONFIGS: int = NUM_EDGE_CONFIGS
    NUM_CORNER_CONFIGS: int = NUM_CORNER_CONFIGS
    CANONICAL_SOLVED_INDEX: int = CANONICAL_SOLVED_INDEX

    @classmethod
    def encode_corners(cls, dlf_slot: int, dlf_co: int, dbl_slot: int, dbl_co: int) -> int:
        """Encodes corner positions and orientations into [0, 503]."""
        if not (0 <= dlf_slot < 8 and 0 <= dbl_slot < 8):
            raise ValueError(f"Corner slot out of bounds [0..7]: dlf={dlf_slot}, dbl={dbl_slot}")
        if dlf_slot == dbl_slot:
            raise ValueError(f"Corners cannot occupy the same slot: {dlf_slot}")
        if not (0 <= dlf_co < 3 and 0 <= dbl_co < 3):
            raise ValueError(f"Corner orientation out of bounds [0..2]: dlf_co={dlf_co}, dbl_co={dbl_co}")

        c0 = CORNER_SLOT_MAP[dlf_slot]
        c1 = CORNER_SLOT_MAP[dbl_slot]
        c1_prime = c1 - 1 if c1 > c0 else c1
        pos_rank = c0 * 7 + c1_prime
        ori_rank = dlf_co * 3 + dbl_co
        return pos_rank * 9 + ori_rank

    @classmethod
    def decode_corners(cls, corner_index: int) -> Tuple[int, int, int, int]:
        """Decodes corner index [0, 503] into (dlf_slot, dlf_co, dbl_slot, dbl_co)."""
        if not (0 <= corner_index < NUM_CORNER_CONFIGS):
            raise ValueError(f"Corner index out of bounds [0, 503]: {corner_index}")

        ori_rank = corner_index % 9
        pos_rank = corner_index // 9

        dlf_co = ori_rank // 3
        dbl_co = ori_rank % 3

        c0 = pos_rank // 7
        c1_prime = pos_rank % 7
        c1 = c1_prime + 1 if c1_prime >= c0 else c1_prime

        dlf_slot = CORNER_SLOT_UNMAP[c0]
        dbl_slot = CORNER_SLOT_UNMAP[c1]
        return dlf_slot, dlf_co, dbl_slot, dbl_co

    @classmethod
    def encode_edges(
        cls,
        dl_slot: int, dl_eo: int,
        fl_slot: int, fl_eo: int,
        bl_slot: int, bl_eo: int
    ) -> int:
        """Encodes edge positions and orientations into [0, 10559]."""
        if not (0 <= dl_slot < 12 and 0 <= fl_slot < 12 and 0 <= bl_slot < 12):
            raise ValueError(f"Edge slot out of bounds [0..11]: dl={dl_slot}, fl={fl_slot}, bl={bl_slot}")
        if dl_slot == fl_slot or dl_slot == bl_slot or fl_slot == bl_slot:
            raise ValueError(f"Edges cannot occupy duplicate slots: dl={dl_slot}, fl={fl_slot}, bl={bl_slot}")
        if not (0 <= dl_eo < 2 and 0 <= fl_eo < 2 and 0 <= bl_eo < 2):
            raise ValueError(f"Edge orientation out of bounds [0..1]: dl_eo={dl_eo}, fl_eo={fl_eo}, bl_eo={bl_eo}")

        mapped_dl = EDGE_SLOT_MAP[dl_slot]
        mapped_fl = EDGE_SLOT_MAP[fl_slot]
        mapped_bl = EDGE_SLOT_MAP[bl_slot]

        # Sort mapped slots while tracking which piece (0=DL, 1=FL, 2=BL) is in each slot
        if mapped_dl < mapped_fl:
            if mapped_fl < mapped_bl:
                slot_0, piece_0 = mapped_dl, 0; slot_1, piece_1 = mapped_fl, 1; slot_2, piece_2 = mapped_bl, 2
            elif mapped_dl < mapped_bl:
                slot_0, piece_0 = mapped_dl, 0; slot_1, piece_1 = mapped_bl, 2; slot_2, piece_2 = mapped_fl, 1
            else:
                slot_0, piece_0 = mapped_bl, 2; slot_1, piece_1 = mapped_dl, 0; slot_2, piece_2 = mapped_fl, 1
        else:
            if mapped_dl < mapped_bl:
                slot_0, piece_0 = mapped_fl, 1; slot_1, piece_1 = mapped_dl, 0; slot_2, piece_2 = mapped_bl, 2
            elif mapped_fl < mapped_bl:
                slot_0, piece_0 = mapped_fl, 1; slot_1, piece_1 = mapped_bl, 2; slot_2, piece_2 = mapped_dl, 0
            else:
                slot_0, piece_0 = mapped_bl, 2; slot_1, piece_1 = mapped_fl, 1; slot_2, piece_2 = mapped_dl, 0

        # Combinadic rank for 3 slots out of 12 (range 0..219)
        comb = slot_0 + _C_T1_2[slot_1] + _C_T2_3[slot_2]

        # Factoradic / Lehmer rank for permutation of the 3 pieces (range 0..5)
        lehmer_0 = (1 if piece_1 < piece_0 else 0) + (1 if piece_2 < piece_0 else 0)
        lehmer_1 = (1 if piece_2 < piece_1 else 0)
        perm = lehmer_0 * 2 + lehmer_1

        # Binary orientation rank (range 0..7)
        ori = (dl_eo << 2) | (fl_eo << 1) | bl_eo

        return ((comb * 6 + perm) << 3) | ori

    @classmethod
    def decode_edges(cls, edge_index: int) -> Tuple[int, int, int, int, int, int]:
        """Decodes edge index [0, 10559] into (dl_slot, dl_eo, fl_slot, fl_eo, bl_slot, bl_eo)."""
        if not (0 <= edge_index < NUM_EDGE_CONFIGS):
            raise ValueError(f"Edge index out of bounds [0, 10559]: {edge_index}")

        ori = edge_index & 7
        dl_eo = (ori >> 2) & 1
        fl_eo = (ori >> 1) & 1
        bl_eo = ori & 1

        rem = edge_index >> 3
        perm = rem % 6
        comb = rem // 6

        t0, t1, t2 = _COMB_TO_SLOTS[comb]
        p0, p1, p2 = _PERM_TO_PIECES[perm]

        mapped_slots = [0, 0, 0]
        mapped_slots[p0] = t0
        mapped_slots[p1] = t1
        mapped_slots[p2] = t2

        dl_slot = EDGE_SLOT_UNMAP[mapped_slots[0]]
        fl_slot = EDGE_SLOT_UNMAP[mapped_slots[1]]
        bl_slot = EDGE_SLOT_UNMAP[mapped_slots[2]]

        return dl_slot, dl_eo, fl_slot, fl_eo, bl_slot, bl_eo

    @classmethod
    def extract_placement(cls, cube: CubeState) -> FBPlacement:
        """Extracts the exact slots and orientations of canonical FB pieces from a CubeState."""
        cp = cube.cp.tolist()
        co = cube.co.tolist()
        ep = cube.ep.tolist()
        eo = cube.eo.tolist()

        dlf_slot = cp.index(Corner.DLF)
        dbl_slot = cp.index(Corner.DBL)
        dl_slot = ep.index(Edge.DL)
        fl_slot = ep.index(Edge.FL)
        bl_slot = ep.index(Edge.BL)

        return FBPlacement(
            dl_slot=dl_slot,
            dl_eo=int(eo[dl_slot]),
            fl_slot=fl_slot,
            fl_eo=int(eo[fl_slot]),
            bl_slot=bl_slot,
            bl_eo=int(eo[bl_slot]),
            dlf_slot=dlf_slot,
            dlf_co=int(co[dlf_slot]),
            dbl_slot=dbl_slot,
            dbl_co=int(co[dbl_slot]),
        )

    @classmethod
    def encode_placement(cls, placement: FBPlacement) -> int:
        """Encodes an FBPlacement into a composite index in [0, 5,322,239]."""
        edge_idx = cls.encode_edges(
            placement.dl_slot, placement.dl_eo,
            placement.fl_slot, placement.fl_eo,
            placement.bl_slot, placement.bl_eo
        )
        corner_idx = cls.encode_corners(
            placement.dlf_slot, placement.dlf_co,
            placement.dbl_slot, placement.dbl_co
        )
        return edge_idx * NUM_CORNER_CONFIGS + corner_idx

    @classmethod
    def encode(cls, cube: CubeState) -> int:
        """Maps any CubeState's First Block pieces to a unique dense integer in [0, 5,322,239]."""
        return cls.encode_placement(cls.extract_placement(cube))

    @classmethod
    def decode_to_placement(cls, index: int) -> FBPlacement:
        """Decodes an index in [0, 5,322,239] into an FBPlacement."""
        if not (0 <= index < TOTAL_STATES):
            raise ValueError(f"Index out of bounds [0, 5322239]: {index}")

        corner_idx = index % NUM_CORNER_CONFIGS
        edge_idx = index // NUM_CORNER_CONFIGS

        dlf_slot, dlf_co, dbl_slot, dbl_co = cls.decode_corners(corner_idx)
        dl_slot, dl_eo, fl_slot, fl_eo, bl_slot, bl_eo = cls.decode_edges(edge_idx)

        return FBPlacement(
            dl_slot=dl_slot,
            dl_eo=dl_eo,
            fl_slot=fl_slot,
            fl_eo=fl_eo,
            bl_slot=bl_slot,
            bl_eo=bl_eo,
            dlf_slot=dlf_slot,
            dlf_co=dlf_co,
            dbl_slot=dbl_slot,
            dbl_co=dbl_co,
        )

    @classmethod
    def decode(cls, index: int) -> CubeState:
        """Reconstructs a valid CubeState from an index in [0, 5,322,239]."""
        placement = cls.decode_to_placement(index)

        cp = np.full(8, -1, dtype=np.int8)
        co = np.zeros(8, dtype=np.int8)
        ep = np.full(12, -1, dtype=np.int8)
        eo = np.zeros(12, dtype=np.int8)
        centers = np.arange(6, dtype=np.int8)

        # Place FB corners
        cp[placement.dlf_slot] = Corner.DLF
        co[placement.dlf_slot] = placement.dlf_co
        cp[placement.dbl_slot] = Corner.DBL
        co[placement.dbl_slot] = placement.dbl_co

        # Fill remaining non-FB corners
        c_fill = 0
        for slot in range(8):
            if cp[slot] == -1:
                cp[slot] = _REMAINING_CORNERS[c_fill]
                c_fill += 1

        # Place FB edges
        ep[placement.dl_slot] = Edge.DL
        eo[placement.dl_slot] = placement.dl_eo
        ep[placement.fl_slot] = Edge.FL
        eo[placement.fl_slot] = placement.fl_eo
        ep[placement.bl_slot] = Edge.BL
        eo[placement.bl_slot] = placement.bl_eo

        # Fill remaining non-FB edges
        e_fill = 0
        for slot in range(12):
            if ep[slot] == -1:
                ep[slot] = _REMAINING_EDGES[e_fill]
                e_fill += 1

        return CubeState(cp=cp, co=co, ep=ep, eo=eo, centers=centers)

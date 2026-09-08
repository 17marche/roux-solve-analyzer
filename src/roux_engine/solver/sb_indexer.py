"""Second Block (SB) State Indexer for Roux method.

Maps any Rubik's Cube state's Second Block pieces (DR, FR, BR edges and DFR, DBR corners)
to a dense integer in [0, 1,088,639] and back.
Also provides specialized sub-indexers for Right Back Square (5,184 states)
and Right Front Square (5,184 states).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import math
import numpy as np

from ..core.constants import Corner, Edge, Center
from ..core.cube import CubeState


# -----------------------------------------------------------------------------
# SB Constants & Slot Mappings
# -----------------------------------------------------------------------------

NUM_SB_CORNER_CONFIGS: int = 270  # P(6, 2) * 3^2 = 30 * 9 = 270
NUM_SB_EDGE_CONFIGS: int = 4_032   # P(9, 3) * 2^3 = 504 * 8 = 4,032
TOTAL_SB_STATES: int = 1_088_640   # 4,032 * 270 = 1,088,640

# 6 allowed corner slots outside FB (slots 4 and 5 are FB slots DLF and DBL)
# Solved positions: DFR is slot 7, DRB (DBR) is slot 6.
# Slot bijection maps 7 -> 0, 6 -> 1 so solved corners map to (0, 1).
SB_CORNER_SLOT_MAP: Tuple[int, ...] = (2, 3, 4, 5, -1, -1, 1, 0)
SB_CORNER_SLOT_UNMAP: Tuple[int, ...] = (7, 6, 0, 1, 2, 3)

# 9 allowed edge slots outside FB (slots 5, 8, 9 are FB slots DL, FL, BL)
# Solved positions: DR is slot 7, FR is slot 11, BR is slot 10.
# Slot bijection maps 7 -> 0, 11 -> 1, 10 -> 2 so solved edges map to (0, 1, 2).
SB_EDGE_SLOT_MAP: Tuple[int, ...] = (3, 4, 5, 6, 7, -1, 8, 0, -1, -1, 2, 1)
SB_EDGE_SLOT_UNMAP: Tuple[int, ...] = (7, 11, 10, 0, 1, 2, 3, 4, 6)

# Precomputed binomial coefficients for combinadic ranking of 3 of 9
_C_T1_2: Tuple[int, ...] = tuple(math.comb(n, 2) for n in range(10))
_C_T2_3: Tuple[int, ...] = tuple(math.comb(n, 3) for n in range(10))

# Precomputed map for combinadic unranking: comb in [0..83] -> (t0, t1, t2) with t0 < t1 < t2
_SB_COMB_TO_SLOTS_3_OF_9: List[Tuple[int, int, int]] = [(0, 0, 0)] * 84
for _t2 in range(2, 9):
    for _t1 in range(1, _t2):
        for _t0 in range(_t1):
            _c = _t0 + _C_T1_2[_t1] + _C_T2_3[_t2]
            _SB_COMB_TO_SLOTS_3_OF_9[_c] = (_t0, _t1, _t2)

# Permutations of 3 elements for factoradic ranking/unranking
_PERM_TO_PIECES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),  # perm 0 (identity)
    (0, 2, 1),  # perm 1
    (1, 0, 2),  # perm 2
    (1, 2, 0),  # perm 3
    (2, 0, 1),  # perm 4
    (2, 1, 0),  # perm 5
)

# -----------------------------------------------------------------------------
# Right Back Square (RBS) Constants & Slot Mappings (5,184 states)
# -----------------------------------------------------------------------------

NUM_RBS_CORNER_CONFIGS: int = 18   # 6 slots * 3 orientations = 18
NUM_RBS_EDGE_CONFIGS: int = 288    # P(9, 2) * 4 = 72 * 4 = 288
TOTAL_RBS_STATES: int = 5_184      # 288 * 18 = 5,184

# Solved position for DBR is slot 6 -> 0
RBS_CORNER_SLOT_MAP: Tuple[int, ...] = (2, 3, 4, 5, -1, -1, 0, 1)
RBS_CORNER_SLOT_UNMAP: Tuple[int, ...] = (6, 7, 0, 1, 2, 3)

# Solved positions: DR is slot 7 -> 0, BR is slot 10 -> 1
RBS_EDGE_SLOT_MAP: Tuple[int, ...] = (3, 4, 5, 6, 7, -1, 8, 0, -1, -1, 1, 2)
RBS_EDGE_SLOT_UNMAP: Tuple[int, ...] = (7, 10, 11, 0, 1, 2, 3, 4, 6)


@dataclass(frozen=True)
class RightBackSquarePlacement:
    """Structured position and orientation coordinates for Right Back Square pieces."""
    dr_slot: int
    dr_eo: int
    br_slot: int
    br_eo: int
    dbr_slot: int
    dbr_co: int


# -----------------------------------------------------------------------------
# Right Front Square (RFS) Constants & Slot Mappings (5,184 states)
# -----------------------------------------------------------------------------

NUM_RFS_CORNER_CONFIGS: int = 18   # 6 slots * 3 orientations = 18
NUM_RFS_EDGE_CONFIGS: int = 288    # P(9, 2) * 4 = 72 * 4 = 288
TOTAL_RFS_STATES: int = 5_184      # 288 * 18 = 5,184

# Solved position for DFR is slot 7 -> 0
RFS_CORNER_SLOT_MAP: Tuple[int, ...] = (2, 3, 4, 5, -1, -1, 1, 0)
RFS_CORNER_SLOT_UNMAP: Tuple[int, ...] = (7, 6, 0, 1, 2, 3)

# Solved positions: DR is slot 7 -> 0, FR is slot 11 -> 1
RFS_EDGE_SLOT_MAP: Tuple[int, ...] = (3, 4, 5, 6, 7, -1, 8, 0, -1, -1, 2, 1)
RFS_EDGE_SLOT_UNMAP: Tuple[int, ...] = (7, 11, 10, 0, 1, 2, 3, 4, 6)


@dataclass(frozen=True)
class RightFrontSquarePlacement:
    """Structured position and orientation coordinates for Right Front Square pieces."""
    dr_slot: int
    dr_eo: int
    fr_slot: int
    fr_eo: int
    dfr_slot: int
    dfr_co: int


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
# Right Back Square Indexer (5,184 states)
# -----------------------------------------------------------------------------

class RightBackSquareIndexer:
    """Bidirectional, zero-waste indexer for Right Back Square states (5,184 states)."""

    TOTAL_STATES: int = TOTAL_RBS_STATES
    NUM_EDGE_CONFIGS: int = NUM_RBS_EDGE_CONFIGS
    NUM_CORNER_CONFIGS: int = NUM_RBS_CORNER_CONFIGS
    CANONICAL_SOLVED_INDEX: int = 0

    @classmethod
    def encode_corner(cls, dbr_slot: int, dbr_co: int) -> int:
        """Encodes DBR corner position and orientation into [0, 17]."""
        if not (0 <= dbr_slot < 8):
            raise ValueError(f"DBR corner slot out of bounds [0..7]: {dbr_slot}")
        if not (0 <= dbr_co < 3):
            raise ValueError(f"DBR corner orientation out of bounds [0..2]: {dbr_co}")

        c = RBS_CORNER_SLOT_MAP[dbr_slot]
        if c == -1:
            raise ValueError(f"DBR corner occupies First Block slot {dbr_slot}")

        return c * 3 + dbr_co

    @classmethod
    def decode_corner(cls, corner_index: int) -> Tuple[int, int]:
        """Decodes corner index [0, 17] into (dbr_slot, dbr_co)."""
        if not (0 <= corner_index < NUM_RBS_CORNER_CONFIGS):
            raise ValueError(f"RBS corner index out of bounds [0, 17]: {corner_index}")

        dbr_co = corner_index % 3
        c = corner_index // 3
        dbr_slot = RBS_CORNER_SLOT_UNMAP[c]
        return dbr_slot, dbr_co

    @classmethod
    def encode_edges(cls, dr_slot: int, dr_eo: int, br_slot: int, br_eo: int) -> int:
        """Encodes DR and BR edge positions and orientations into [0, 287]."""
        if not (0 <= dr_slot < 12 and 0 <= br_slot < 12):
            raise ValueError(f"Edge slot out of bounds [0..11]: dr={dr_slot}, br={br_slot}")
        if dr_slot == br_slot:
            raise ValueError(f"Edges cannot occupy duplicate slot: {dr_slot}")
        if not (0 <= dr_eo < 2 and 0 <= br_eo < 2):
            raise ValueError(f"Edge orientation out of bounds [0..1]: dr_eo={dr_eo}, br_eo={br_eo}")

        e0 = RBS_EDGE_SLOT_MAP[dr_slot]
        e1 = RBS_EDGE_SLOT_MAP[br_slot]
        if e0 == -1:
            raise ValueError(f"DR edge occupies First Block slot {dr_slot}")
        if e1 == -1:
            raise ValueError(f"BR edge occupies First Block slot {br_slot}")

        e1_prime = e1 - 1 if e1 > e0 else e1
        pos_rank = e0 * 8 + e1_prime
        ori_rank = (dr_eo << 1) | br_eo
        return pos_rank * 4 + ori_rank

    @classmethod
    def decode_edges(cls, edge_index: int) -> Tuple[int, int, int, int]:
        """Decodes edge index [0, 287] into (dr_slot, dr_eo, br_slot, br_eo)."""
        if not (0 <= edge_index < NUM_RBS_EDGE_CONFIGS):
            raise ValueError(f"RBS edge index out of bounds [0, 287]: {edge_index}")

        ori_rank = edge_index % 4
        dr_eo = (ori_rank >> 1) & 1
        br_eo = ori_rank & 1

        pos_rank = edge_index // 4
        e0 = pos_rank // 8
        e1_prime = pos_rank % 8
        e1 = e1_prime + 1 if e1_prime >= e0 else e1_prime

        dr_slot = RBS_EDGE_SLOT_UNMAP[e0]
        br_slot = RBS_EDGE_SLOT_UNMAP[e1]
        return dr_slot, dr_eo, br_slot, br_eo

    @classmethod
    def extract_placement(cls, cube: CubeState) -> RightBackSquarePlacement:
        """Extracts the exact slots and orientations of RBS pieces from a CubeState."""
        cp = cube.cp.tolist()
        co = cube.co.tolist()
        ep = cube.ep.tolist()
        eo = cube.eo.tolist()

        dbr_slot = cp.index(Corner.DRB)
        dr_slot = ep.index(Edge.DR)
        br_slot = ep.index(Edge.BR)

        return RightBackSquarePlacement(
            dr_slot=dr_slot,
            dr_eo=int(eo[dr_slot]),
            br_slot=br_slot,
            br_eo=int(eo[br_slot]),
            dbr_slot=dbr_slot,
            dbr_co=int(co[dbr_slot]),
        )

    @classmethod
    def encode_placement(cls, placement: RightBackSquarePlacement) -> int:
        """Encodes an RBSPlacement into a composite index in [0, 5,183]."""
        edge_idx = cls.encode_edges(
            placement.dr_slot, placement.dr_eo,
            placement.br_slot, placement.br_eo
        )
        corner_idx = cls.encode_corner(placement.dbr_slot, placement.dbr_co)
        return edge_idx * NUM_RBS_CORNER_CONFIGS + corner_idx

    @classmethod
    def encode(cls, cube: CubeState) -> int:
        """Maps any CubeState's Right Back Square pieces to a dense integer in [0, 5,183]."""
        return cls.encode_placement(cls.extract_placement(cube))

    @classmethod
    def decode_to_placement(cls, index: int) -> RightBackSquarePlacement:
        """Decodes an index in [0, 5,183] into a RightBackSquarePlacement."""
        if not (0 <= index < TOTAL_RBS_STATES):
            raise ValueError(f"Index out of bounds [0, 5183]: {index}")

        corner_idx = index % NUM_RBS_CORNER_CONFIGS
        edge_idx = index // NUM_RBS_CORNER_CONFIGS

        dbr_slot, dbr_co = cls.decode_corner(corner_idx)
        dr_slot, dr_eo, br_slot, br_eo = cls.decode_edges(edge_idx)

        return RightBackSquarePlacement(
            dr_slot=dr_slot,
            dr_eo=dr_eo,
            br_slot=br_slot,
            br_eo=br_eo,
            dbr_slot=dbr_slot,
            dbr_co=dbr_co,
        )

    @classmethod
    def decode(cls, index: int) -> CubeState:
        """Reconstructs a valid CubeState with solved FB and decoded RBS from an index in [0, 5,183]."""
        placement = cls.decode_to_placement(index)

        cp = [int(Corner.DLF) if i == Corner.DLF else int(Corner.DBL) if i == Corner.DBL else -1 for i in range(8)]
        co = [0] * 8
        ep = [-1] * 12
        ep[Edge.DL] = int(Edge.DL)
        ep[Edge.FL] = int(Edge.FL)
        ep[Edge.BL] = int(Edge.BL)
        eo = [0] * 12

        # Place RBS corner
        cp[placement.dbr_slot] = int(Corner.DRB)
        co[placement.dbr_slot] = placement.dbr_co

        # Fill remaining 5 non-FB, non-RBS corners: UFL(0), ULB(1), UBR(2), URF(3), DFR(7)
        rem_corners = (int(Corner.UFL), int(Corner.ULB), int(Corner.UBR), int(Corner.URF), int(Corner.DFR))
        c_fill = 0
        for slot in range(8):
            if cp[slot] == -1:
                cp[slot] = rem_corners[c_fill]
                c_fill += 1

        # Place RBS edges
        ep[placement.dr_slot] = int(Edge.DR)
        eo[placement.dr_slot] = placement.dr_eo
        ep[placement.br_slot] = int(Edge.BR)
        eo[placement.br_slot] = placement.br_eo

        # Fill remaining 7 non-FB, non-RBS edges: UF(0), UL(1), UB(2), UR(3), DF(4), DB(6), FR(11)
        rem_edges = (int(Edge.UF), int(Edge.UL), int(Edge.UB), int(Edge.UR), int(Edge.DF), int(Edge.DB), int(Edge.FR))
        e_fill = 0
        for slot in range(12):
            if ep[slot] == -1:
                ep[slot] = rem_edges[e_fill]
                e_fill += 1

        return CubeState(
            cp=np.array(cp, dtype=np.int8),
            co=np.array(co, dtype=np.int8),
            ep=np.array(ep, dtype=np.int8),
            eo=np.array(eo, dtype=np.int8),
            centers=np.arange(6, dtype=np.int8),
        )


# -----------------------------------------------------------------------------
# Right Front Square Indexer (5,184 states)
# -----------------------------------------------------------------------------

class RightFrontSquareIndexer:
    """Bidirectional, zero-waste indexer for Right Front Square states (5,184 states)."""

    TOTAL_STATES: int = TOTAL_RFS_STATES
    NUM_EDGE_CONFIGS: int = NUM_RFS_EDGE_CONFIGS
    NUM_CORNER_CONFIGS: int = NUM_RFS_CORNER_CONFIGS
    CANONICAL_SOLVED_INDEX: int = 0

    @classmethod
    def encode_corner(cls, dfr_slot: int, dfr_co: int) -> int:
        """Encodes DFR corner position and orientation into [0, 17]."""
        if not (0 <= dfr_slot < 8):
            raise ValueError(f"DFR corner slot out of bounds [0..7]: {dfr_slot}")
        if not (0 <= dfr_co < 3):
            raise ValueError(f"DFR corner orientation out of bounds [0..2]: {dfr_co}")

        c = RFS_CORNER_SLOT_MAP[dfr_slot]
        if c == -1:
            raise ValueError(f"DFR corner occupies First Block slot {dfr_slot}")

        return c * 3 + dfr_co

    @classmethod
    def decode_corner(cls, corner_index: int) -> Tuple[int, int]:
        """Decodes corner index [0, 17] into (dfr_slot, dfr_co)."""
        if not (0 <= corner_index < NUM_RFS_CORNER_CONFIGS):
            raise ValueError(f"RFS corner index out of bounds [0, 17]: {corner_index}")

        dfr_co = corner_index % 3
        c = corner_index // 3
        dfr_slot = RFS_CORNER_SLOT_UNMAP[c]
        return dfr_slot, dfr_co

    @classmethod
    def encode_edges(cls, dr_slot: int, dr_eo: int, fr_slot: int, fr_eo: int) -> int:
        """Encodes DR and FR edge positions and orientations into [0, 287]."""
        if not (0 <= dr_slot < 12 and 0 <= fr_slot < 12):
            raise ValueError(f"Edge slot out of bounds [0..11]: dr={dr_slot}, fr={fr_slot}")
        if dr_slot == fr_slot:
            raise ValueError(f"Edges cannot occupy duplicate slot: {dr_slot}")
        if not (0 <= dr_eo < 2 and 0 <= fr_eo < 2):
            raise ValueError(f"Edge orientation out of bounds [0..1]: dr_eo={dr_eo}, fr_eo={fr_eo}")

        e0 = RFS_EDGE_SLOT_MAP[dr_slot]
        e1 = RFS_EDGE_SLOT_MAP[fr_slot]
        if e0 == -1:
            raise ValueError(f"DR edge occupies First Block slot {dr_slot}")
        if e1 == -1:
            raise ValueError(f"FR edge occupies First Block slot {fr_slot}")

        e1_prime = e1 - 1 if e1 > e0 else e1
        pos_rank = e0 * 8 + e1_prime
        ori_rank = (dr_eo << 1) | fr_eo
        return pos_rank * 4 + ori_rank

    @classmethod
    def decode_edges(cls, edge_index: int) -> Tuple[int, int, int, int]:
        """Decodes edge index [0, 287] into (dr_slot, dr_eo, fr_slot, fr_eo)."""
        if not (0 <= edge_index < NUM_RFS_EDGE_CONFIGS):
            raise ValueError(f"RFS edge index out of bounds [0, 287]: {edge_index}")

        ori_rank = edge_index % 4
        dr_eo = (ori_rank >> 1) & 1
        fr_eo = ori_rank & 1

        pos_rank = edge_index // 4
        e0 = pos_rank // 8
        e1_prime = pos_rank % 8
        e1 = e1_prime + 1 if e1_prime >= e0 else e1_prime

        dr_slot = RFS_EDGE_SLOT_UNMAP[e0]
        fr_slot = RFS_EDGE_SLOT_UNMAP[e1]
        return dr_slot, dr_eo, fr_slot, fr_eo

    @classmethod
    def extract_placement(cls, cube: CubeState) -> RightFrontSquarePlacement:
        """Extracts the exact slots and orientations of RFS pieces from a CubeState."""
        cp = cube.cp.tolist()
        co = cube.co.tolist()
        ep = cube.ep.tolist()
        eo = cube.eo.tolist()

        dfr_slot = cp.index(Corner.DFR)
        dr_slot = ep.index(Edge.DR)
        fr_slot = ep.index(Edge.FR)

        return RightFrontSquarePlacement(
            dr_slot=dr_slot,
            dr_eo=int(eo[dr_slot]),
            fr_slot=fr_slot,
            fr_eo=int(eo[fr_slot]),
            dfr_slot=dfr_slot,
            dfr_co=int(co[dfr_slot]),
        )

    @classmethod
    def encode_placement(cls, placement: RightFrontSquarePlacement) -> int:
        """Encodes an RFSPlacement into a composite index in [0, 5,183]."""
        edge_idx = cls.encode_edges(
            placement.dr_slot, placement.dr_eo,
            placement.fr_slot, placement.fr_eo
        )
        corner_idx = cls.encode_corner(placement.dfr_slot, placement.dfr_co)
        return edge_idx * NUM_RFS_CORNER_CONFIGS + corner_idx

    @classmethod
    def encode(cls, cube: CubeState) -> int:
        """Maps any CubeState's Right Front Square pieces to a dense integer in [0, 5,183]."""
        return cls.encode_placement(cls.extract_placement(cube))

    @classmethod
    def decode_to_placement(cls, index: int) -> RightFrontSquarePlacement:
        """Decodes an index in [0, 5,183] into a RightFrontSquarePlacement."""
        if not (0 <= index < TOTAL_RFS_STATES):
            raise ValueError(f"Index out of bounds [0, 5183]: {index}")

        corner_idx = index % NUM_RFS_CORNER_CONFIGS
        edge_idx = index // NUM_RFS_CORNER_CONFIGS

        dfr_slot, dfr_co = cls.decode_corner(corner_idx)
        dr_slot, dr_eo, fr_slot, fr_eo = cls.decode_edges(edge_idx)

        return RightFrontSquarePlacement(
            dr_slot=dr_slot,
            dr_eo=dr_eo,
            fr_slot=fr_slot,
            fr_eo=fr_eo,
            dfr_slot=dfr_slot,
            dfr_co=dfr_co,
        )

    @classmethod
    def decode(cls, index: int) -> CubeState:
        """Reconstructs a valid CubeState with solved FB and decoded RFS from an index in [0, 5,183]."""
        placement = cls.decode_to_placement(index)

        cp = [int(Corner.DLF) if i == Corner.DLF else int(Corner.DBL) if i == Corner.DBL else -1 for i in range(8)]
        co = [0] * 8
        ep = [-1] * 12
        ep[Edge.DL] = int(Edge.DL)
        ep[Edge.FL] = int(Edge.FL)
        ep[Edge.BL] = int(Edge.BL)
        eo = [0] * 12

        # Place RFS corner
        cp[placement.dfr_slot] = int(Corner.DFR)
        co[placement.dfr_slot] = placement.dfr_co

        # Fill remaining 5 non-FB, non-RFS corners: UFL(0), ULB(1), UBR(2), URF(3), DRB(6)
        rem_corners = (int(Corner.UFL), int(Corner.ULB), int(Corner.UBR), int(Corner.URF), int(Corner.DRB))
        c_fill = 0
        for slot in range(8):
            if cp[slot] == -1:
                cp[slot] = rem_corners[c_fill]
                c_fill += 1

        # Place RFS edges
        ep[placement.dr_slot] = int(Edge.DR)
        eo[placement.dr_slot] = placement.dr_eo
        ep[placement.fr_slot] = int(Edge.FR)
        eo[placement.fr_slot] = placement.fr_eo

        # Fill remaining 7 non-FB, non-RFS edges: UF(0), UL(1), UB(2), UR(3), DF(4), DB(6), BR(10)
        rem_edges = (int(Edge.UF), int(Edge.UL), int(Edge.UB), int(Edge.UR), int(Edge.DF), int(Edge.DB), int(Edge.BR))
        e_fill = 0
        for slot in range(12):
            if ep[slot] == -1:
                ep[slot] = rem_edges[e_fill]
                e_fill += 1

        return CubeState(
            cp=np.array(cp, dtype=np.int8),
            co=np.array(co, dtype=np.int8),
            ep=np.array(ep, dtype=np.int8),
            eo=np.array(eo, dtype=np.int8),
            centers=np.arange(6, dtype=np.int8),
        )


# -----------------------------------------------------------------------------
# Full Second Block Indexer (1,088,640 states)
# -----------------------------------------------------------------------------

class SBIndexer:
    """Bidirectional, zero-waste indexer for Roux Second Block states."""

    TOTAL_STATES: int = TOTAL_SB_STATES
    NUM_EDGE_CONFIGS: int = NUM_SB_EDGE_CONFIGS
    NUM_CORNER_CONFIGS: int = NUM_SB_CORNER_CONFIGS
    CANONICAL_SOLVED_INDEX: int = 0

    @classmethod
    def encode_corners(cls, dfr_slot: int, dfr_co: int, dbr_slot: int, dbr_co: int) -> int:
        """Encodes SB corner positions and orientations into [0, 269]."""
        if not (0 <= dfr_slot < 8 and 0 <= dbr_slot < 8):
            raise ValueError(f"Corner slot out of bounds [0..7]: dfr={dfr_slot}, dbr={dbr_slot}")
        if dfr_slot == dbr_slot:
            raise ValueError(f"Corners cannot occupy duplicate slot: {dfr_slot}")
        if not (0 <= dfr_co < 3 and 0 <= dbr_co < 3):
            raise ValueError(f"Corner orientation out of bounds [0..2]: dfr_co={dfr_co}, dbr_co={dbr_co}")

        c0 = SB_CORNER_SLOT_MAP[dfr_slot]
        c1 = SB_CORNER_SLOT_MAP[dbr_slot]
        if c0 == -1:
            raise ValueError(f"DFR corner occupies First Block slot {dfr_slot}")
        if c1 == -1:
            raise ValueError(f"DBR corner occupies First Block slot {dbr_slot}")

        c1_prime = c1 - 1 if c1 > c0 else c1
        pos_rank = c0 * 5 + c1_prime
        ori_rank = dfr_co * 3 + dbr_co
        return pos_rank * 9 + ori_rank

    @classmethod
    def decode_corners(cls, corner_index: int) -> Tuple[int, int, int, int]:
        """Decodes corner index [0, 269] into (dfr_slot, dfr_co, dbr_slot, dbr_co)."""
        if not (0 <= corner_index < NUM_SB_CORNER_CONFIGS):
            raise ValueError(f"Corner index out of bounds [0, 269]: {corner_index}")

        ori_rank = corner_index % 9
        pos_rank = corner_index // 9

        dfr_co = ori_rank // 3
        dbr_co = ori_rank % 3

        c0 = pos_rank // 5
        c1_prime = pos_rank % 5
        c1 = c1_prime + 1 if c1_prime >= c0 else c1_prime

        dfr_slot = SB_CORNER_SLOT_UNMAP[c0]
        dbr_slot = SB_CORNER_SLOT_UNMAP[c1]
        return dfr_slot, dfr_co, dbr_slot, dbr_co

    @classmethod
    def encode_edges(
        cls,
        dr_slot: int, dr_eo: int,
        fr_slot: int, fr_eo: int,
        br_slot: int, br_eo: int
    ) -> int:
        """Encodes SB edge positions and orientations into [0, 4031]."""
        if not (0 <= dr_slot < 12 and 0 <= fr_slot < 12 and 0 <= br_slot < 12):
            raise ValueError(f"Edge slot out of bounds [0..11]: dr={dr_slot}, fr={fr_slot}, br={br_slot}")
        if dr_slot == fr_slot or dr_slot == br_slot or fr_slot == br_slot:
            raise ValueError(f"Edges cannot occupy duplicate slots: dr={dr_slot}, fr={fr_slot}, br={br_slot}")
        if not (0 <= dr_eo < 2 and 0 <= fr_eo < 2 and 0 <= br_eo < 2):
            raise ValueError(f"Edge orientation out of bounds [0..1]: dr_eo={dr_eo}, fr_eo={fr_eo}, br_eo={br_eo}")

        m_dr = SB_EDGE_SLOT_MAP[dr_slot]
        m_fr = SB_EDGE_SLOT_MAP[fr_slot]
        m_br = SB_EDGE_SLOT_MAP[br_slot]

        if m_dr == -1:
            raise ValueError(f"DR edge occupies First Block slot {dr_slot}")
        if m_fr == -1:
            raise ValueError(f"FR edge occupies First Block slot {fr_slot}")
        if m_br == -1:
            raise ValueError(f"BR edge occupies First Block slot {br_slot}")

        # Sort mapped slots while tracking which piece (0=DR, 1=FR, 2=BR) is in each slot
        if m_dr < m_fr:
            if m_fr < m_br:
                slot_0, piece_0 = m_dr, 0; slot_1, piece_1 = m_fr, 1; slot_2, piece_2 = m_br, 2
            elif m_dr < m_br:
                slot_0, piece_0 = m_dr, 0; slot_1, piece_1 = m_br, 2; slot_2, piece_2 = m_fr, 1
            else:
                slot_0, piece_0 = m_br, 2; slot_1, piece_1 = m_dr, 0; slot_2, piece_2 = m_fr, 1
        else:
            if m_dr < m_br:
                slot_0, piece_0 = m_fr, 1; slot_1, piece_1 = m_dr, 0; slot_2, piece_2 = m_br, 2
            elif m_fr < m_br:
                slot_0, piece_0 = m_fr, 1; slot_1, piece_1 = m_br, 2; slot_2, piece_2 = m_dr, 0
            else:
                slot_0, piece_0 = m_br, 2; slot_1, piece_1 = m_fr, 1; slot_2, piece_2 = m_dr, 0

        # Combinadic rank for 3 slots out of 9 (range 0..83)
        comb = slot_0 + _C_T1_2[slot_1] + _C_T2_3[slot_2]

        # Factoradic / Lehmer rank for permutation of the 3 pieces (range 0..5)
        lehmer_0 = (1 if piece_1 < piece_0 else 0) + (1 if piece_2 < piece_0 else 0)
        lehmer_1 = (1 if piece_2 < piece_1 else 0)
        perm = lehmer_0 * 2 + lehmer_1

        # Binary orientation rank (range 0..7)
        ori = (dr_eo << 2) | (fr_eo << 1) | br_eo

        return ((comb * 6 + perm) << 3) | ori

    @classmethod
    def decode_edges(cls, edge_index: int) -> Tuple[int, int, int, int, int, int]:
        """Decodes edge index [0, 4031] into (dr_slot, dr_eo, fr_slot, fr_eo, br_slot, br_eo)."""
        if not (0 <= edge_index < NUM_SB_EDGE_CONFIGS):
            raise ValueError(f"Edge index out of bounds [0, 4031]: {edge_index}")

        ori = edge_index & 7
        dr_eo = (ori >> 2) & 1
        fr_eo = (ori >> 1) & 1
        br_eo = ori & 1

        rem = edge_index >> 3
        perm = rem % 6
        comb = rem // 6

        t0, t1, t2 = _SB_COMB_TO_SLOTS_3_OF_9[comb]
        p0, p1, p2 = _PERM_TO_PIECES[perm]

        mapped_slots = [0, 0, 0]
        mapped_slots[p0] = t0
        mapped_slots[p1] = t1
        mapped_slots[p2] = t2

        dr_slot = SB_EDGE_SLOT_UNMAP[mapped_slots[0]]
        fr_slot = SB_EDGE_SLOT_UNMAP[mapped_slots[1]]
        br_slot = SB_EDGE_SLOT_UNMAP[mapped_slots[2]]

        return dr_slot, dr_eo, fr_slot, fr_eo, br_slot, br_eo

    @classmethod
    def extract_placement(cls, cube: CubeState) -> SBPlacement:
        """Extracts the exact slots and orientations of canonical SB pieces from a CubeState."""
        cp = cube.cp.tolist()
        co = cube.co.tolist()
        ep = cube.ep.tolist()
        eo = cube.eo.tolist()

        dfr_slot = cp.index(Corner.DFR)
        dbr_slot = cp.index(Corner.DRB)
        dr_slot = ep.index(Edge.DR)
        fr_slot = ep.index(Edge.FR)
        br_slot = ep.index(Edge.BR)

        return SBPlacement(
            dr_slot=dr_slot,
            dr_eo=int(eo[dr_slot]),
            fr_slot=fr_slot,
            fr_eo=int(eo[fr_slot]),
            br_slot=br_slot,
            br_eo=int(eo[br_slot]),
            dfr_slot=dfr_slot,
            dfr_co=int(co[dfr_slot]),
            dbr_slot=dbr_slot,
            dbr_co=int(co[dbr_slot]),
        )

    @classmethod
    def encode_placement(cls, placement: SBPlacement) -> int:
        """Encodes an SBPlacement into a composite index in [0, 1,088,639]."""
        edge_idx = cls.encode_edges(
            placement.dr_slot, placement.dr_eo,
            placement.fr_slot, placement.fr_eo,
            placement.br_slot, placement.br_eo
        )
        corner_idx = cls.encode_corners(
            placement.dfr_slot, placement.dfr_co,
            placement.dbr_slot, placement.dbr_co
        )
        return edge_idx * NUM_SB_CORNER_CONFIGS + corner_idx

    @classmethod
    def encode(cls, cube: CubeState) -> int:
        """Maps any CubeState's Second Block pieces to a unique dense integer in [0, 1,088,639]."""
        return cls.encode_placement(cls.extract_placement(cube))

    @classmethod
    def decode_to_placement(cls, index: int) -> SBPlacement:
        """Decodes an index in [0, 1,088,639] into an SBPlacement."""
        if not (0 <= index < TOTAL_SB_STATES):
            raise ValueError(f"Index out of bounds [0, 1088639]: {index}")

        corner_idx = index % NUM_SB_CORNER_CONFIGS
        edge_idx = index // NUM_SB_CORNER_CONFIGS

        dfr_slot, dfr_co, dbr_slot, dbr_co = cls.decode_corners(corner_idx)
        dr_slot, dr_eo, fr_slot, fr_eo, br_slot, br_eo = cls.decode_edges(edge_idx)

        return SBPlacement(
            dr_slot=dr_slot,
            dr_eo=dr_eo,
            fr_slot=fr_slot,
            fr_eo=fr_eo,
            br_slot=br_slot,
            br_eo=br_eo,
            dfr_slot=dfr_slot,
            dfr_co=dfr_co,
            dbr_slot=dbr_slot,
            dbr_co=dbr_co,
        )

    @classmethod
    def decode(cls, index: int) -> CubeState:
        """Reconstructs a valid CubeState with solved FB and decoded SB from an index in [0, 1,088,639]."""
        placement = cls.decode_to_placement(index)

        cp = [int(Corner.DLF) if i == Corner.DLF else int(Corner.DBL) if i == Corner.DBL else -1 for i in range(8)]
        co = [0] * 8
        ep = [-1] * 12
        ep[Edge.DL] = int(Edge.DL)
        ep[Edge.FL] = int(Edge.FL)
        ep[Edge.BL] = int(Edge.BL)
        eo = [0] * 12

        # Place SB corners
        cp[placement.dfr_slot] = int(Corner.DFR)
        co[placement.dfr_slot] = placement.dfr_co
        cp[placement.dbr_slot] = int(Corner.DRB)
        co[placement.dbr_slot] = placement.dbr_co

        # Fill remaining non-FB, non-SB corners: UFL(0), ULB(1), UBR(2), URF(3)
        rem_corners = (int(Corner.UFL), int(Corner.ULB), int(Corner.UBR), int(Corner.URF))
        c_fill = 0
        for slot in range(8):
            if cp[slot] == -1:
                cp[slot] = rem_corners[c_fill]
                c_fill += 1

        # Place SB edges
        ep[placement.dr_slot] = int(Edge.DR)
        eo[placement.dr_slot] = placement.dr_eo
        ep[placement.fr_slot] = int(Edge.FR)
        eo[placement.fr_slot] = placement.fr_eo
        ep[placement.br_slot] = int(Edge.BR)
        eo[placement.br_slot] = placement.br_eo

        # Fill remaining non-FB, non-SB edges: UF(0), UL(1), UB(2), UR(3), DF(4), DB(6)
        rem_edges = (int(Edge.UF), int(Edge.UL), int(Edge.UB), int(Edge.UR), int(Edge.DF), int(Edge.DB))
        e_fill = 0
        for slot in range(12):
            if ep[slot] == -1:
                ep[slot] = rem_edges[e_fill]
                e_fill += 1

        return CubeState(
            cp=np.array(cp, dtype=np.int8),
            co=np.array(co, dtype=np.int8),
            ep=np.array(ep, dtype=np.int8),
            eo=np.array(eo, dtype=np.int8),
            centers=np.arange(6, dtype=np.int8),
        )

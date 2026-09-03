"""CubeState class representing 3D Rubik's Cube permutations and orientations."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Union, Tuple, Optional, Sequence, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .moves import Move

from .constants import (
    NUM_CORNERS, NUM_EDGES, NUM_CENTERS,
    CORNER_ORIENT_MOD, EDGE_ORIENT_MOD,
    Corner, Edge, Center
)


@dataclass
class CubeState:
    """Represents a 3D Rubik's Cube state in Cubie Coordinates.
    
    Attributes:
        cp: Corner Permutation (8 elements, array of 0..7)
        co: Corner Orientation (8 elements, array of 0..2)
        ep: Edge Permutation (12 elements, array of 0..11)
        eo: Edge Orientation (12 elements, array of 0..1)
        centers: Center Permutation (6 elements, array of 0..5, representing U, D, F, B, L, R)
    """
    cp: np.ndarray = field(default_factory=lambda: np.arange(NUM_CORNERS, dtype=np.int8))
    co: np.ndarray = field(default_factory=lambda: np.zeros(NUM_CORNERS, dtype=np.int8))
    ep: np.ndarray = field(default_factory=lambda: np.arange(NUM_EDGES, dtype=np.int8))
    eo: np.ndarray = field(default_factory=lambda: np.zeros(NUM_EDGES, dtype=np.int8))
    centers: np.ndarray = field(default_factory=lambda: np.arange(NUM_CENTERS, dtype=np.int8))

    def copy(self) -> CubeState:
        """Returns a deep copy of the CubeState."""
        return CubeState(
            cp=self.cp.copy(),
            co=self.co.copy(),
            ep=self.ep.copy(),
            eo=self.eo.copy(),
            centers=self.centers.copy()
        )

    def is_solved(self, allow_rotations: bool = False) -> bool:
        """Returns True if the cube is in a solved state.
        
        Args:
            allow_rotations: If True, checks if the cube is solved in any of the
                24 whole-cube rotation orientations. If False, checks strictly against
                the canonical identity orientation.
        """
        if not allow_rotations:
            return bool(
                np.array_equal(self.cp, np.arange(NUM_CORNERS, dtype=np.int8)) and
                np.all(self.co == 0) and
                np.array_equal(self.ep, np.arange(NUM_EDGES, dtype=np.int8)) and
                np.all(self.eo == 0) and
                np.array_equal(self.centers, np.arange(NUM_CENTERS, dtype=np.int8))
            )
        return self.to_bytes() in _get_solved_rotation_bytes()

    def is_fb_solved(self, white_bottom: bool = True) -> bool:
        """Checks if the canonical Left First Block (1x2x3 on L face) is solved.
        
        Left FB consists of:
        - 2 Corners: DFL (4 or 7 depending on orientation), DBL (5)
        - 3 Edges: DL (5), FL (8), BL (9)
        - L Center (4)
        """
        # Canonical White-bottom, Green-front, Orange-left
        # FB pieces: Corners DFL (DLF = 4), DBL (5)
        #            Edges DL (5), FL (8), BL (9)
        #            Center L (4)
        if white_bottom:
            if self.centers[Center.L] != Center.L:
                return False
            # Check corners DLF (4) and DBL (5)
            if self.cp[Corner.DLF] != Corner.DLF or self.co[Corner.DLF] != 0:
                return False
            if self.cp[Corner.DBL] != Corner.DBL or self.co[Corner.DBL] != 0:
                return False
            # Check edges DL (5), FL (8), BL (9)
            if self.ep[Edge.DL] != Edge.DL or self.eo[Edge.DL] != 0:
                return False
            if self.ep[Edge.FL] != Edge.FL or self.eo[Edge.FL] != 0:
                return False
            if self.ep[Edge.BL] != Edge.BL or self.eo[Edge.BL] != 0:
                return False
            return True
        return False

    def to_bytes(self) -> bytes:
        """Serializes the state into a compact 42-byte binary representation."""
        return (
            self.cp.tobytes() +
            self.co.tobytes() +
            self.ep.tobytes() +
            self.eo.tobytes() +
            self.centers.tobytes()
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> CubeState:
        """Deserializes a CubeState from a 42-byte binary representation."""
        cp = np.frombuffer(b[0:8], dtype=np.int8).copy()
        co = np.frombuffer(b[8:16], dtype=np.int8).copy()
        ep = np.frombuffer(b[16:28], dtype=np.int8).copy()
        eo = np.frombuffer(b[28:40], dtype=np.int8).copy()
        centers = np.frombuffer(b[40:46], dtype=np.int8).copy()
        return cls(cp=cp, co=co, ep=ep, eo=eo, centers=centers)

    def apply_move(self, move: Union[str, 'Move']) -> CubeState:
        """Applies a single move to the cube state in-place and returns self."""
        from .moves import apply_move as _apply_move
        return _apply_move(self, move)

    def apply_moves(self, moves: Union[str, Sequence[Union[str, 'Move']]]) -> CubeState:
        """Applies a sequence of moves to the cube state in-place and returns self."""
        from .moves import apply_moves as _apply_moves
        return _apply_moves(self, moves)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CubeState):
            return False
        return (
            np.array_equal(self.cp, other.cp) and
            np.array_equal(self.co, other.co) and
            np.array_equal(self.ep, other.ep) and
            np.array_equal(self.eo, other.eo) and
            np.array_equal(self.centers, other.centers)
        )

    def __hash__(self) -> int:
        return hash(self.to_bytes())

    def __repr__(self) -> str:
        return (
            f"CubeState(\n"
            f"  cp={self.cp.tolist()},\n"
            f"  co={self.co.tolist()},\n"
            f"  ep={self.ep.tolist()},\n"
            f"  eo={self.eo.tolist()},\n"
            f"  centers={self.centers.tolist()}\n"
            f")"
        )


_SOLVED_ROTATION_BYTES: Optional[set[bytes]] = None


def _get_solved_rotation_bytes() -> set[bytes]:
    """Returns the set of 42-byte binary representations for all 24 rotated solved states."""
    global _SOLVED_ROTATION_BYTES
    if _SOLVED_ROTATION_BYTES is None:
        rotation_sequences = [
            "", "x", "x2", "x'",
            "y", "y x", "y x2", "y x'",
            "y2", "y2 x", "y2 x2", "y2 x'",
            "y'", "y' x", "y' x2", "y' x'",
            "z", "z x", "z x2", "z x'",
            "z'", "z' x", "z' x2", "z' x'"
        ]
        s = set()
        for seq in rotation_sequences:
            c = CubeState()
            if seq:
                c.apply_moves(seq)
            s.add(c.to_bytes())
        _SOLVED_ROTATION_BYTES = s
    return _SOLVED_ROTATION_BYTES

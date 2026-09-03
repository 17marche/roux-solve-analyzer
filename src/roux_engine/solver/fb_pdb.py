"""First Block Pattern Database (PDB) runtime loader and query interface.

Provides memory-mapped access to the 5.32M state First Block PDB with < 5ms startup
and < 1us single-query heuristic distance lookups.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
from pathlib import Path
import numpy as np

from ..core.cube import CubeState
from .fb_indexer import FBIndexer, TOTAL_STATES
from .pdb_generator import (
    FB_MOVESET,
    get_default_pdb_path,
)

FILE_SIZE_BYTES: int = TOTAL_STATES // 2  # 2,661,120 bytes


class FBPDB:
    """Runtime memory-mapped First Block Pattern Database."""

    TOTAL_STATES: int = TOTAL_STATES
    FILE_SIZE_BYTES: int = FILE_SIZE_BYTES

    def __init__(
        self,
        pdb_path: Optional[Union[str, Path]] = None,
        mmap: bool = True,
    ) -> None:
        self.path: Path = Path(pdb_path) if pdb_path is not None else get_default_pdb_path()
        if not self.path.exists():
            raise FileNotFoundError(
                f"First Block PDB binary not found at '{self.path}'. "
                f"Run 'uv run python -m roux_engine.cli generate-fb-pdb' to generate it."
            )

        file_size = self.path.stat().st_size
        if file_size != self.FILE_SIZE_BYTES:
            raise ValueError(
                f"Invalid PDB file size: {file_size} bytes (expected exactly {self.FILE_SIZE_BYTES} bytes)"
            )

        if mmap:
            self._data: np.ndarray = np.memmap(self.path, dtype=np.uint8, mode="r")
        else:
            self._data = np.fromfile(self.path, dtype=np.uint8)

    @property
    def size(self) -> int:
        """Total number of states indexed by the database (5,322,240)."""
        return self.TOTAL_STATES

    @property
    def file_size_bytes(self) -> int:
        """Size of the packed binary database on disk (2,661,120 bytes)."""
        return self.FILE_SIZE_BYTES

    @property
    def is_loaded(self) -> bool:
        """Whether the database is currently mapped and ready for queries."""
        return self._data is not None

    def get_distance_by_index(self, index: int) -> int:
        """Looks up the exact shortest distance for a First Block state index in [0, 5322239].

        Optimized branchless bit-shift execution with < 1 microsecond query latency.
        """
        if not (0 <= index < self.TOTAL_STATES):
            raise IndexError(f"Index out of bounds [0, {self.TOTAL_STATES - 1}]: {index}")

        byte_val = int(self._data[index >> 1])
        return (byte_val >> (4 * (index & 1))) & 0x0F

    def get_distance(self, target: Union[CubeState, int]) -> int:
        """Returns the exact shortest distance to solved First Block for a CubeState or index."""
        if isinstance(target, int):
            return self.get_distance_by_index(target)
        elif isinstance(target, CubeState):
            return self.get_distance_by_index(FBIndexer.encode(target))
        else:
            raise TypeError(f"Expected CubeState or int index, got {type(target).__name__}")

    def get_distances_by_indices(self, indices: Union[Sequence[int], np.ndarray]) -> np.ndarray:
        """Vectorized lookup for multiple integer indices."""
        indices_arr = np.asarray(indices, dtype=np.int32)
        if np.any((indices_arr < 0) | (indices_arr >= self.TOTAL_STATES)):
            raise IndexError(f"Some indices out of bounds [0, {self.TOTAL_STATES - 1}]")

        byte_vals = self._data[indices_arr >> 1]
        shifts = (indices_arr & 1) << 2
        return ((byte_vals >> shifts) & 0x0F).astype(np.uint8)

    def get_distances(self, targets: Sequence[Union[CubeState, int]]) -> np.ndarray:
        """Batch lookup for a sequence of CubeStates or integer indices."""
        indices = np.empty(len(targets), dtype=np.int32)
        for i, t in enumerate(targets):
            if isinstance(t, int):
                indices[i] = t
            elif isinstance(t, CubeState):
                indices[i] = FBIndexer.encode(t)
            else:
                raise TypeError(f"Expected CubeState or int index at position {i}, got {type(t).__name__}")
        return self.get_distances_by_indices(indices)


__all__ = [
    "FBPDB",
    "FB_MOVESET",
    "FILE_SIZE_BYTES",
    "get_default_pdb_path",
]

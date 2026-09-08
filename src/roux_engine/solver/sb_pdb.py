"""Second Block and Right Square Pattern Database (PDB) runtime loaders.

Provides memory-mapped access to the 1.08M state Second Block PDB,
5.2k state Right Back Square PDB, and 5.2k state Right Front Square PDB
with < 5ms startup and < 1us single-query heuristic distance lookups.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Type
from pathlib import Path
import numpy as np

from ..core.cube import CubeState
from .sb_indexer import (
    SBIndexer,
    TOTAL_SB_STATES,
    RightBackSquareIndexer,
    TOTAL_RBS_STATES,
    RightFrontSquareIndexer,
    TOTAL_RFS_STATES,
)
from .sb_pdb_generator import (
    SB_MOVESET,
    get_default_sb_pdb_path,
    get_default_rbs_pdb_path,
    get_default_rfs_pdb_path,
)

SB_FILE_SIZE_BYTES: int = TOTAL_SB_STATES // 2    # 544,320 bytes
RBS_FILE_SIZE_BYTES: int = TOTAL_RBS_STATES // 2  # 2,592 bytes
RFS_FILE_SIZE_BYTES: int = TOTAL_RFS_STATES // 2  # 2,592 bytes


class _PackedPDBBase:
    """Base runtime memory-mapped 4-bit nibble Pattern Database loader."""

    TOTAL_STATES: int
    FILE_SIZE_BYTES: int
    _name: str
    _indexer_cls: Type

    def __init__(
        self,
        pdb_path: Optional[Union[str, Path]] = None,
        mmap: bool = True,
    ) -> None:
        self.path: Path = Path(pdb_path) if pdb_path is not None else self._get_default_path()
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self._name} binary not found at '{self.path}'. "
                f"Run 'uv run python -m roux_engine.cli generate-sb-pdb' to generate it."
            )

        file_size = self.path.stat().st_size
        if file_size != self.FILE_SIZE_BYTES:
            raise ValueError(
                f"Invalid {self._name} file size: {file_size} bytes "
                f"(expected exactly {self.FILE_SIZE_BYTES} bytes)"
            )

        if mmap:
            self._data: np.ndarray = np.memmap(self.path, dtype=np.uint8, mode="r")
        else:
            self._data = np.fromfile(self.path, dtype=np.uint8)

    def _get_default_path(self) -> Path:
        raise NotImplementedError

    @property
    def size(self) -> int:
        """Total number of states indexed by the database."""
        return self.TOTAL_STATES

    @property
    def file_size_bytes(self) -> int:
        """Size of the packed binary database on disk."""
        return self.FILE_SIZE_BYTES

    @property
    def is_loaded(self) -> bool:
        """Whether the database is currently mapped and ready for queries."""
        return self._data is not None

    def get_distance_by_index(self, index: int) -> int:
        """Looks up exact shortest distance for state index in [0, TOTAL_STATES - 1]."""
        if not (0 <= index < self.TOTAL_STATES):
            raise IndexError(f"Index out of bounds [0, {self.TOTAL_STATES - 1}]: {index}")

        byte_val = int(self._data[index >> 1])
        return (byte_val >> (4 * (index & 1))) & 0x0F

    def get_distance(self, target: Union[CubeState, int]) -> int:
        """Returns the shortest distance to target goal for CubeState or index."""
        if isinstance(target, int):
            return self.get_distance_by_index(target)
        elif isinstance(target, CubeState):
            return self.get_distance_by_index(self._indexer_cls.encode(target))
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
                indices[i] = self._indexer_cls.encode(t)
            else:
                raise TypeError(f"Expected CubeState or int index at position {i}, got {type(t).__name__}")
        return self.get_distances_by_indices(indices)


class SBPDB(_PackedPDBBase):
    """Runtime memory-mapped Full Second Block Pattern Database (1,088,640 states)."""

    TOTAL_STATES: int = TOTAL_SB_STATES
    FILE_SIZE_BYTES: int = SB_FILE_SIZE_BYTES
    _name: str = "Full Second Block PDB"
    _indexer_cls = SBIndexer

    def _get_default_path(self) -> Path:
        return get_default_sb_pdb_path()


class RightBackSquarePDB(_PackedPDBBase):
    """Runtime memory-mapped Right Back Square Pattern Database (5,184 states)."""

    TOTAL_STATES: int = TOTAL_RBS_STATES
    FILE_SIZE_BYTES: int = RBS_FILE_SIZE_BYTES
    _name: str = "Right Back Square PDB"
    _indexer_cls = RightBackSquareIndexer

    def _get_default_path(self) -> Path:
        return get_default_rbs_pdb_path()


class RightFrontSquarePDB(_PackedPDBBase):
    """Runtime memory-mapped Right Front Square Pattern Database (5,184 states)."""

    TOTAL_STATES: int = TOTAL_RFS_STATES
    FILE_SIZE_BYTES: int = RFS_FILE_SIZE_BYTES
    _name: str = "Right Front Square PDB"
    _indexer_cls = RightFrontSquareIndexer

    def _get_default_path(self) -> Path:
        return get_default_rfs_pdb_path()


__all__ = [
    "SBPDB",
    "RightBackSquarePDB",
    "RightFrontSquarePDB",
    "SB_FILE_SIZE_BYTES",
    "RBS_FILE_SIZE_BYTES",
    "RFS_FILE_SIZE_BYTES",
    "SB_MOVESET",
]

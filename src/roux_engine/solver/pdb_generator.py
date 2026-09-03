"""First Block Pattern Database (PDB) BFS generator and packed storage.

Explores all reachable First Block states using the ergonomic moveset
<U, D, R, F, B, r, M> (ADR-0001, strictly excluding physical L turns),
and serializes exact shortest distances into packed 4-bit nibbles.
"""

from __future__ import annotations
from typing import Optional, Union, Callable, Tuple, Sequence
from pathlib import Path
import numpy as np

from ..core.moves import MOVES
from .fb_indexer import (
    FBIndexer,
    NUM_CORNER_CONFIGS,
    NUM_EDGE_CONFIGS,
    TOTAL_STATES,
)

# Canonical ergonomic First Block moveset per ADR-0001 (strictly omitting physical L face turns)
FB_MOVESET: Tuple[str, ...] = (
    "U", "U2", "U'",
    "D", "D2", "D'",
    "R", "R2", "R'",
    "F", "F2", "F'",
    "B", "B2", "B'",
    "r", "r2", "r'",
    "M", "M2", "M'",
)


def build_transition_tables(
    moveset: Sequence[str] = FB_MOVESET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes corner and edge transition tables for the given moveset.

    Returns:
        Tuple of (corner_trans, edge_trans) where:
        - corner_trans has shape (NUM_CORNER_CONFIGS, len(moveset)), dtype int32
        - edge_trans has shape (NUM_EDGE_CONFIGS, len(moveset)), dtype int32
    """
    num_moves = len(moveset)

    # Precompute move permutations and orientations once for all moves
    moves_inv_cp: list[list[int]] = []
    moves_co_ori: list[list[int]] = []
    moves_inv_ep: list[list[int]] = []
    moves_eo_ori: list[list[int]] = []

    for m_name in moveset:
        m = MOVES[m_name]
        inv_cp = [0] * 8
        for dst in range(8):
            inv_cp[int(m.cp_perm[dst])] = dst
        moves_inv_cp.append(inv_cp)
        moves_co_ori.append([int(x) for x in m.co_ori])

        inv_ep = [0] * 12
        for dst in range(12):
            inv_ep[int(m.ep_perm[dst])] = dst
        moves_inv_ep.append(inv_ep)
        moves_eo_ori.append([int(x) for x in m.eo_ori])

    corner_trans = np.zeros((NUM_CORNER_CONFIGS, num_moves), dtype=np.int32)
    for c_idx in range(NUM_CORNER_CONFIGS):
        dlf_slot, dlf_co, dbl_slot, dbl_co = FBIndexer.decode_corners(c_idx)
        for m_idx in range(num_moves):
            inv_cp = moves_inv_cp[m_idx]
            co_ori = moves_co_ori[m_idx]

            new_dlf = inv_cp[dlf_slot]
            new_dlf_co = (dlf_co + co_ori[new_dlf]) % 3
            new_dbl = inv_cp[dbl_slot]
            new_dbl_co = (dbl_co + co_ori[new_dbl]) % 3
            corner_trans[c_idx, m_idx] = FBIndexer.encode_corners(
                new_dlf, new_dlf_co, new_dbl, new_dbl_co
            )

    edge_trans = np.zeros((NUM_EDGE_CONFIGS, num_moves), dtype=np.int32)
    for e_idx in range(NUM_EDGE_CONFIGS):
        dl_slot, dl_eo, fl_slot, fl_eo, bl_slot, bl_eo = FBIndexer.decode_edges(e_idx)
        for m_idx in range(num_moves):
            inv_ep = moves_inv_ep[m_idx]
            eo_ori = moves_eo_ori[m_idx]

            new_dl = inv_ep[dl_slot]
            new_dl_eo = (dl_eo + eo_ori[new_dl]) % 2
            new_fl = inv_ep[fl_slot]
            new_fl_eo = (fl_eo + eo_ori[new_fl]) % 2
            new_bl = inv_ep[bl_slot]
            new_bl_eo = (bl_eo + eo_ori[new_bl]) % 2
            edge_trans[e_idx, m_idx] = FBIndexer.encode_edges(
                new_dl, new_dl_eo, new_fl, new_fl_eo, new_bl, new_bl_eo
            )

    return corner_trans, edge_trans



def pack_distances_nibbles(distances: np.ndarray) -> np.ndarray:
    """Packs a uint8 array of distances in [0..15] into 4-bit nibbles (2 values per byte).

    Even indices are stored in the lower 4 bits (bits 0..3).
    Odd indices are stored in the upper 4 bits (bits 4..7).
    """
    if len(distances) % 2 != 0:
        raise ValueError(f"Input distances array must have an even length, got {len(distances)}")

    if np.any(distances > 15):
        raise ValueError("Distance value exceeds 4-bit nibble maximum of 15")

    low = distances[0::2].astype(np.uint8) & 0x0F
    high = distances[1::2].astype(np.uint8) & 0x0F
    packed = low | (high << 4)
    return packed.astype(np.uint8)


def unpack_distances_nibbles(packed: np.ndarray, num_states: Optional[int] = None) -> np.ndarray:
    """Unpacks a uint8 array of 4-bit nibbles into full distance values."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    total_len = len(packed) * 2
    unpacked = np.empty(total_len, dtype=np.uint8)
    unpacked[0::2] = low
    unpacked[1::2] = high

    if num_states is not None:
        unpacked = unpacked[:num_states]
    return unpacked


def get_default_pdb_path() -> Path:
    """Returns the default binary file path for First Block PDB."""
    return Path(__file__).resolve().parent.parent / "data" / "fb_pdb.bin"


def generate_fb_pdb(
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
    chunk_size: int = 500_000,
) -> Path:
    """Generates the full 5,322,240-state First Block Pattern Database via BFS.

    Explores all reachable states using the ergonomic moveset <U, D, R, F, B, r, M>
    (21 moves, strictly omitting physical L turns per ADR-0001).

    Packs distances 0..15 into 4-bit nibbles (2,661,120 bytes) and writes to disk.

    Args:
        output_path: Target binary file path. Defaults to src/roux_engine/data/fb_pdb.bin.
        progress_callback: Optional callback receiving (depth, new_states, total_states, elapsed_sec).
        chunk_size: Batch size for frontier expansion to bound memory usage.

    Returns:
        Path to the written binary file.
    """
    import time

    out = Path(output_path) if output_path is not None else get_default_pdb_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    corner_trans, edge_trans = build_transition_tables(FB_MOVESET)

    dist = np.full(TOTAL_STATES, 255, dtype=np.uint8)
    dist[0] = 0
    frontier = np.array([0], dtype=np.int32)
    total_states = 1

    if progress_callback:
        progress_callback(0, 1, 1, 0.0)

    depth = 0
    while len(frontier) > 0:
        depth += 1
        t_depth_start = time.time()
        frontier_edges = frontier // NUM_CORNER_CONFIGS
        frontier_corners = frontier % NUM_CORNER_CONFIGS

        unvisited_chunks = []
        for i in range(0, len(frontier), chunk_size):
            edges_chunk = frontier_edges[i : i + chunk_size]
            corners_chunk = frontier_corners[i : i + chunk_size]

            new_edges = edge_trans[edges_chunk]
            new_corners = corner_trans[corners_chunk]
            new_indices = new_edges * NUM_CORNER_CONFIGS + new_corners
            flat_indices = np.unique(new_indices.ravel())
            unvisited_mask = dist[flat_indices] == 255
            unvisited = flat_indices[unvisited_mask]
            dist[unvisited] = depth
            unvisited_chunks.append(unvisited)

        if len(unvisited_chunks) > 1:
            frontier = np.concatenate(unvisited_chunks)
        elif len(unvisited_chunks) == 1:
            frontier = unvisited_chunks[0]
        else:
            frontier = np.array([], dtype=np.int32)

        num_new = len(frontier)
        total_states += num_new
        elapsed = time.time() - t_depth_start

        if progress_callback:
            progress_callback(depth, num_new, total_states, elapsed)

    unvisited_count = int(np.count_nonzero(dist == 255))
    if unvisited_count > 0:
        raise RuntimeError(f"BFS generation incomplete: {unvisited_count} states unreachable")

    packed = pack_distances_nibbles(dist)
    packed.tofile(out)

    return out



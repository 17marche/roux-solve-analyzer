"""Second Block and Right Square Pattern Database (PDB) BFS generator.

Explores reachable Second Block, Right Back Square, and Right Front Square states
under the rotationless <R, U, r, M> generator (12 moves, strictly preserving First Block),
and serializes exact shortest distances into packed 4-bit nibbles.
"""

from __future__ import annotations
from typing import Optional, Union, Callable, Tuple, Sequence
from pathlib import Path
import time
import numpy as np

from ..core.moves import MOVES
from .pdb_generator import pack_distances_nibbles, unpack_distances_nibbles
from .sb_indexer import (
    SBIndexer,
    NUM_SB_CORNER_CONFIGS,
    NUM_SB_EDGE_CONFIGS,
    TOTAL_SB_STATES,
    RightBackSquareIndexer,
    NUM_RBS_CORNER_CONFIGS,
    NUM_RBS_EDGE_CONFIGS,
    TOTAL_RBS_STATES,
    RightFrontSquareIndexer,
    NUM_RFS_CORNER_CONFIGS,
    NUM_RFS_EDGE_CONFIGS,
    TOTAL_RFS_STATES,
)

# Canonical Second Block moveset under <R, U, r, M> (strictly preserving First Block)
SB_MOVESET: Tuple[str, ...] = (
    "R", "R2", "R'",
    "U", "U2", "U'",
    "r", "r2", "r'",
    "M", "M2", "M'",
)


def get_default_sb_pdb_path() -> Path:
    """Returns the default binary file path for full Second Block PDB."""
    return Path(__file__).resolve().parent.parent / "data" / "sb_pdb.bin"


def get_default_rbs_pdb_path() -> Path:
    """Returns the default binary file path for Right Back Square PDB."""
    return Path(__file__).resolve().parent.parent / "data" / "rbs_pdb.bin"


def get_default_rfs_pdb_path() -> Path:
    """Returns the default binary file path for Right Front Square PDB."""
    return Path(__file__).resolve().parent.parent / "data" / "rfs_pdb.bin"


def _precompute_move_inverses(
    moveset: Sequence[str],
) -> Tuple[list[list[int]], list[list[int]], list[list[int]], list[list[int]]]:
    """Precomputes inverted corner/edge permutations and orientation arrays for moves."""
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

    return moves_inv_cp, moves_co_ori, moves_inv_ep, moves_eo_ori


def build_sb_transition_tables(
    moveset: Sequence[str] = SB_MOVESET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes corner and edge transition tables for full Second Block.

    Returns:
        Tuple of (corner_trans, edge_trans) where:
        - corner_trans has shape (270, len(moveset)), dtype int32
        - edge_trans has shape (4032, len(moveset)), dtype int32
    """
    num_moves = len(moveset)
    moves_inv_cp, moves_co_ori, moves_inv_ep, moves_eo_ori = _precompute_move_inverses(moveset)

    corner_trans = np.zeros((NUM_SB_CORNER_CONFIGS, num_moves), dtype=np.int32)
    for c_idx in range(NUM_SB_CORNER_CONFIGS):
        dfr, dfr_co, dbr, dbr_co = SBIndexer.decode_corners(c_idx)
        for m_idx in range(num_moves):
            new_dfr = moves_inv_cp[m_idx][dfr]
            new_dfr_co = (dfr_co + moves_co_ori[m_idx][new_dfr]) % 3
            new_dbr = moves_inv_cp[m_idx][dbr]
            new_dbr_co = (dbr_co + moves_co_ori[m_idx][new_dbr]) % 3
            corner_trans[c_idx, m_idx] = SBIndexer.encode_corners(
                new_dfr, new_dfr_co, new_dbr, new_dbr_co
            )

    edge_trans = np.zeros((NUM_SB_EDGE_CONFIGS, num_moves), dtype=np.int32)
    for e_idx in range(NUM_SB_EDGE_CONFIGS):
        dr, dr_eo, fr, fr_eo, br, br_eo = SBIndexer.decode_edges(e_idx)
        for m_idx in range(num_moves):
            new_dr = moves_inv_ep[m_idx][dr]
            new_dr_eo = (dr_eo + moves_eo_ori[m_idx][new_dr]) % 2
            new_fr = moves_inv_ep[m_idx][fr]
            new_fr_eo = (fr_eo + moves_eo_ori[m_idx][new_fr]) % 2
            new_br = moves_inv_ep[m_idx][br]
            new_br_eo = (br_eo + moves_eo_ori[m_idx][new_br]) % 2
            edge_trans[e_idx, m_idx] = SBIndexer.encode_edges(
                new_dr, new_dr_eo, new_fr, new_fr_eo, new_br, new_br_eo
            )

    return corner_trans, edge_trans


def build_rbs_transition_tables(
    moveset: Sequence[str] = SB_MOVESET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes corner and edge transition tables for Right Back Square.

    Returns:
        Tuple of (corner_trans, edge_trans) where:
        - corner_trans has shape (18, len(moveset)), dtype int32
        - edge_trans has shape (288, len(moveset)), dtype int32
    """
    num_moves = len(moveset)
    moves_inv_cp, moves_co_ori, moves_inv_ep, moves_eo_ori = _precompute_move_inverses(moveset)

    corner_trans = np.zeros((NUM_RBS_CORNER_CONFIGS, num_moves), dtype=np.int32)
    for c_idx in range(NUM_RBS_CORNER_CONFIGS):
        dbr, dbr_co = RightBackSquareIndexer.decode_corner(c_idx)
        for m_idx in range(num_moves):
            new_dbr = moves_inv_cp[m_idx][dbr]
            new_dbr_co = (dbr_co + moves_co_ori[m_idx][new_dbr]) % 3
            corner_trans[c_idx, m_idx] = RightBackSquareIndexer.encode_corner(new_dbr, new_dbr_co)

    edge_trans = np.zeros((NUM_RBS_EDGE_CONFIGS, num_moves), dtype=np.int32)
    for e_idx in range(NUM_RBS_EDGE_CONFIGS):
        dr, dr_eo, br, br_eo = RightBackSquareIndexer.decode_edges(e_idx)
        for m_idx in range(num_moves):
            new_dr = moves_inv_ep[m_idx][dr]
            new_dr_eo = (dr_eo + moves_eo_ori[m_idx][new_dr]) % 2
            new_br = moves_inv_ep[m_idx][br]
            new_br_eo = (br_eo + moves_eo_ori[m_idx][new_br]) % 2
            edge_trans[e_idx, m_idx] = RightBackSquareIndexer.encode_edges(
                new_dr, new_dr_eo, new_br, new_br_eo
            )

    return corner_trans, edge_trans


def build_rfs_transition_tables(
    moveset: Sequence[str] = SB_MOVESET,
) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputes corner and edge transition tables for Right Front Square.

    Returns:
        Tuple of (corner_trans, edge_trans) where:
        - corner_trans has shape (18, len(moveset)), dtype int32
        - edge_trans has shape (288, len(moveset)), dtype int32
    """
    num_moves = len(moveset)
    moves_inv_cp, moves_co_ori, moves_inv_ep, moves_eo_ori = _precompute_move_inverses(moveset)

    corner_trans = np.zeros((NUM_RFS_CORNER_CONFIGS, num_moves), dtype=np.int32)
    for c_idx in range(NUM_RFS_CORNER_CONFIGS):
        dfr, dfr_co = RightFrontSquareIndexer.decode_corner(c_idx)
        for m_idx in range(num_moves):
            new_dfr = moves_inv_cp[m_idx][dfr]
            new_dfr_co = (dfr_co + moves_co_ori[m_idx][new_dfr]) % 3
            corner_trans[c_idx, m_idx] = RightFrontSquareIndexer.encode_corner(new_dfr, new_dfr_co)

    edge_trans = np.zeros((NUM_RFS_EDGE_CONFIGS, num_moves), dtype=np.int32)
    for e_idx in range(NUM_RFS_EDGE_CONFIGS):
        dr, dr_eo, fr, fr_eo = RightFrontSquareIndexer.decode_edges(e_idx)
        for m_idx in range(num_moves):
            new_dr = moves_inv_ep[m_idx][dr]
            new_dr_eo = (dr_eo + moves_eo_ori[m_idx][new_dr]) % 2
            new_fr = moves_inv_ep[m_idx][fr]
            new_fr_eo = (fr_eo + moves_eo_ori[m_idx][new_fr]) % 2
            edge_trans[e_idx, m_idx] = RightFrontSquareIndexer.encode_edges(
                new_dr, new_dr_eo, new_fr, new_fr_eo
            )

    return corner_trans, edge_trans


def generate_rbs_pdb(
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
) -> Path:
    """Generates the 5,184-state Right Back Square PDB via BFS and packs into 2,592 bytes."""
    out = Path(output_path) if output_path is not None else get_default_rbs_pdb_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    corner_trans, edge_trans = build_rbs_transition_tables()

    dist = np.full(TOTAL_RBS_STATES, 255, dtype=np.uint8)
    dist[0] = 0
    frontier = np.array([0], dtype=np.int32)
    total_states = 1

    if progress_callback:
        progress_callback(0, 1, 1, 0.0)

    depth = 0
    while len(frontier) > 0:
        depth += 1
        t_depth_start = time.time()
        f_e = frontier // NUM_RBS_CORNER_CONFIGS
        f_c = frontier % NUM_RBS_CORNER_CONFIGS

        new_indices = (edge_trans[f_e] * NUM_RBS_CORNER_CONFIGS + corner_trans[f_c]).ravel()
        flat_indices = np.unique(new_indices)
        unvisited = flat_indices[dist[flat_indices] == 255]
        dist[unvisited] = depth
        frontier = unvisited
        total_states += len(frontier)
        elapsed = time.time() - t_depth_start

        if progress_callback:
            progress_callback(depth, len(frontier), total_states, elapsed)

    unvisited_count = int(np.count_nonzero(dist == 255))
    if unvisited_count > 0:
        raise RuntimeError(f"RBS BFS incomplete: {unvisited_count} states unreachable")

    packed = pack_distances_nibbles(dist)
    packed.tofile(out)
    return out


def generate_rfs_pdb(
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
) -> Path:
    """Generates the 5,184-state Right Front Square PDB via BFS and packs into 2,592 bytes."""
    out = Path(output_path) if output_path is not None else get_default_rfs_pdb_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    corner_trans, edge_trans = build_rfs_transition_tables()

    dist = np.full(TOTAL_RFS_STATES, 255, dtype=np.uint8)
    dist[0] = 0
    frontier = np.array([0], dtype=np.int32)
    total_states = 1

    if progress_callback:
        progress_callback(0, 1, 1, 0.0)

    depth = 0
    while len(frontier) > 0:
        depth += 1
        t_depth_start = time.time()
        f_e = frontier // NUM_RFS_CORNER_CONFIGS
        f_c = frontier % NUM_RFS_CORNER_CONFIGS

        new_indices = (edge_trans[f_e] * NUM_RFS_CORNER_CONFIGS + corner_trans[f_c]).ravel()
        flat_indices = np.unique(new_indices)
        unvisited = flat_indices[dist[flat_indices] == 255]
        dist[unvisited] = depth
        frontier = unvisited
        total_states += len(frontier)
        elapsed = time.time() - t_depth_start

        if progress_callback:
            progress_callback(depth, len(frontier), total_states, elapsed)

    unvisited_count = int(np.count_nonzero(dist == 255))
    if unvisited_count > 0:
        raise RuntimeError(f"RFS BFS incomplete: {unvisited_count} states unreachable")

    packed = pack_distances_nibbles(dist)
    packed.tofile(out)
    return out


def generate_sb_pdb(
    output_path: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[int, int, int, float], None]] = None,
    chunk_size: int = 500_000,
) -> Path:
    """Generates the 1,088,640-state Full Second Block PDB via BFS and packs into 544,320 bytes."""
    out = Path(output_path) if output_path is not None else get_default_sb_pdb_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    corner_trans, edge_trans = build_sb_transition_tables()

    dist = np.full(TOTAL_SB_STATES, 255, dtype=np.uint8)
    dist[0] = 0
    frontier = np.array([0], dtype=np.int32)
    total_states = 1

    if progress_callback:
        progress_callback(0, 1, 1, 0.0)

    depth = 0
    while len(frontier) > 0:
        depth += 1
        t_depth_start = time.time()
        frontier_edges = frontier // NUM_SB_CORNER_CONFIGS
        frontier_corners = frontier % NUM_SB_CORNER_CONFIGS

        unvisited_chunks = []
        for i in range(0, len(frontier), chunk_size):
            edges_chunk = frontier_edges[i : i + chunk_size]
            corners_chunk = frontier_corners[i : i + chunk_size]

            new_edges = edge_trans[edges_chunk]
            new_corners = corner_trans[corners_chunk]
            new_indices = new_edges * NUM_SB_CORNER_CONFIGS + new_corners
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
        raise RuntimeError(f"SB BFS incomplete: {unvisited_count} states unreachable")

    packed = pack_distances_nibbles(dist)
    packed.tofile(out)
    return out


def generate_all_sb_pdbs(
    output_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable[[str, int, int, int, float], None]] = None,
) -> Tuple[Path, Path, Path]:
    """Generates all three Second Block binary lookup tables deterministically.

    Returns:
        Tuple of (sb_path, rbs_path, rfs_path)
    """
    if output_dir is not None:
        out_dir = Path(output_dir)
        sb_path = out_dir / "sb_pdb.bin"
        rbs_path = out_dir / "rbs_pdb.bin"
        rfs_path = out_dir / "rfs_pdb.bin"
    else:
        sb_path = get_default_sb_pdb_path()
        rbs_path = get_default_rbs_pdb_path()
        rfs_path = get_default_rfs_pdb_path()

    def make_cb(name: str):
        if not progress_callback:
            return None
        return lambda depth, new_s, total, el: progress_callback(name, depth, new_s, total, el)

    p_sb = generate_sb_pdb(output_path=sb_path, progress_callback=make_cb("sb"))
    p_rbs = generate_rbs_pdb(output_path=rbs_path, progress_callback=make_cb("rbs"))
    p_rfs = generate_rfs_pdb(output_path=rfs_path, progress_callback=make_cb("rfs"))

    return p_sb, p_rbs, p_rfs


__all__ = [
    "SB_MOVESET",
    "get_default_sb_pdb_path",
    "get_default_rbs_pdb_path",
    "get_default_rfs_pdb_path",
    "build_sb_transition_tables",
    "build_rbs_transition_tables",
    "build_rfs_transition_tables",
    "generate_sb_pdb",
    "generate_rbs_pdb",
    "generate_rfs_pdb",
    "generate_all_sb_pdbs",
]

"""Top-K Candidate Search IDA* First Block solver module.

Provides multi-path IDA* search finding the top-K shortest candidate paths
for First Block across dual-neutral orientations and M-slice pitch angles,
omitting physical L turns per ADR-0001.
"""

from __future__ import annotations

from .fb_solver import (
    FBSolution,
    FBSolver,
    solve_fb,
    normalize_rotations,
    _build_allowed_moves,
    _MOVE_FAMILIES,
    PITCH_ANGLES,
)

__all__ = [
    "FBSolution",
    "FBSolver",
    "solve_fb",
    "normalize_rotations",
]

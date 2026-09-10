"""End-to-end Roux Scramble Solver Pipeline.

Orchestrates the complete 4-phase Roux method heuristic search:
1. First Block (FB): Top-K IDA* heuristic search using the 5.32M state FB PDB.
2. Second Block (SB): Center-aligned IDA* heuristic search using the 1.08M state SB PDB
   (supporting Free Blockbuilding, Classical Standard, and Square + Pair paradigms).
3. CMLL: Instant 42-case classification table with AUF alignment.
4. Last Six Edges (LSE): Optimal <M, U> graph search for Steps 4a, 4b, and 4c.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import urllib.parse

from ..core.cube import CubeState
from ..core.orientation import (
    RouxOrientation,
    get_orientation,
    get_dual_neutral_orientations,
)
from ..core.parser import MoveParser
from ..segmenter.cmll_classifier import CMLLClassifier, CMLL_ALGS
from .fb_solver import FBSolution, FBSolver
from .sb_solver import SBSolution, solve_sb
from .lse_solver import LSESolution, solve_lse


@dataclass(frozen=True)
class FullSolveResult:
    """Structured result of an end-to-end Roux solve."""
    scramble: str
    fb: FBSolution
    sb: SBSolution
    cmll_case: str
    cmll_group: str
    cmll_pre_auf: str
    cmll_alg: str
    cmll_post_auf: str
    cmll_moves: List[str]
    cmll_stm: int
    lse: LSESolution
    full_moves: List[str]
    full_moves_str: str
    total_stm: int
    duration_ms: float
    style: str
    is_valid: bool = True

    @property
    def alg_cubing_url(self) -> str:
        """Generates an interactive 3D visualization URL on alg.cubing.net."""
        params = {
            "setup": self.scramble,
            "alg": self.full_moves_str,
        }
        return f"https://alg.cubing.net/?{urllib.parse.urlencode(params)}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result into a clean dictionary."""
        return {
            "scramble": self.scramble,
            "is_valid": self.is_valid,
            "style": self.style,
            "total_stm": self.total_stm,
            "duration_ms": round(self.duration_ms, 2),
            "full_solution": self.full_moves_str,
            "alg_cubing_url": self.alg_cubing_url,
            "fb": {
                "moves": self.fb.moves,
                "moves_str": " ".join(self.fb.moves),
                "move_count": self.fb.move_count,
                "inspection_rotation": self.fb.inspection_rotation,
                "orientation": self.fb.orientation,
            },
            "sb": {
                "moves": list(self.sb.moves),
                "moves_str": " ".join(self.sb.moves),
                "move_count": self.sb.move_count,
                "style": self.sb.style,
                "order": self.sb.order,
            },
            "cmll": {
                "case": self.cmll_case,
                "group": self.cmll_group,
                "pre_auf": self.cmll_pre_auf,
                "alg": self.cmll_alg,
                "post_auf": self.cmll_post_auf,
                "moves": self.cmll_moves,
                "moves_str": " ".join(self.cmll_moves),
                "move_count": self.cmll_stm,
            },
            "lse": {
                "target": self.lse.target,
                "moves": self.lse.moves,
                "moves_str": " ".join(self.lse.moves),
                "move_count": self.lse.move_count,
                "case_name": self.lse.case_name,
            },
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes result to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


class RouxScrambleSolver:
    """Heuristic search orchestrator for end-to-end Roux method solving."""

    _INSTANCE: Optional[RouxScrambleSolver] = None

    @classmethod
    def get_instance(cls) -> RouxScrambleSolver:
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def solve(
        self,
        scramble: str,
        style: str = "free",
        orientation: Optional[str] = None,
    ) -> FullSolveResult:
        """Solves a scramble completely from scratch using the 4-tier Roux pipeline.

        Args:
            scramble: Scramble move sequence string.
            style: Second Block solving paradigm ('free', 'classical', or 'square_pair').
            orientation: Optional orientation constraint (e.g. '' for canonical Yellow-bottom/Orange-left).

        Returns:
            FullSolveResult containing structured phase solutions, movecounts, and full solution string.
        """
        t0 = time.perf_counter()
        cube = CubeState().apply_moves(scramble)

        # 1. First Block Search across dual-neutral orientations
        fb_solver = FBSolver.get_instance()
        dn_orientations = (
            [get_orientation(orientation)]
            if orientation is not None
            else get_dual_neutral_orientations()
        )

        best_fb: Optional[FBSolution] = None
        best_fb_ori: Optional[RouxOrientation] = None

        for ori in dn_orientations:
            rot = ori.rotations
            candidates = fb_solver.solve(cube, k=10, orientation=rot)
            # Find candidate using standard inspection rotation (preserving U/D axis)
            for cand in candidates:
                if cand.inspection_rotation == rot:
                    if best_fb is None or cand.move_count < best_fb.move_count:
                        best_fb = cand
                        best_fb_ori = ori
                    break

        if best_fb is None or best_fb_ori is None:
            raise RuntimeError(f"Could not find valid First Block solution for scramble: {scramble}")

        # Apply FB to working simulation cube
        sim = cube.copy()
        if best_fb.inspection_rotation:
            sim.apply_moves(best_fb.inspection_rotation)
        if best_fb.moves:
            sim.apply_moves(" ".join(best_fb.moves))

        # 2. Second Block Search
        sb_sols = solve_sb(sim, k=1, style=style)
        if not sb_sols:
            # Fallback to free if chosen style had no valid transition
            sb_sols = solve_sb(sim, k=1, style="free")
            if not sb_sols:
                raise RuntimeError(f"Could not find valid Second Block solution after FB: {best_fb.moves}")

        best_sb = sb_sols[0]
        if best_sb.moves:
            sim.apply_moves(" ".join(best_sb.moves))

        # 3. CMLL Recognition & Algorithm
        cmll_case, cmll_group, cmll_pre_auf = CMLLClassifier.classify_state(sim, block=best_fb_ori)
        cmll_alg = ""
        for cid, grp, alg in CMLL_ALGS:
            if cid == cmll_case:
                cmll_alg = alg
                break

        if cmll_pre_auf:
            sim.apply_move(cmll_pre_auf)
        if cmll_alg:
            sim.apply_moves(cmll_alg)

        # Compute Post-CMLL AUF to align U corners with FB & SB
        ref_cube = CubeState()
        if best_fb_ori.rotations:
            ref_cube.apply_moves(best_fb_ori.rotations)
        target_ufl = int(ref_cube.cp[0])

        cmll_post_auf = ""
        for auf in ("", "U", "U2", "U'"):
            test_c = sim.copy()
            if auf:
                test_c.apply_move(auf)
            if int(test_c.cp[0]) == target_ufl:
                cmll_post_auf = auf
                break

        if cmll_post_auf:
            sim.apply_move(cmll_post_auf)

        # Assemble CMLL moves
        cmll_parts = [p for p in [cmll_pre_auf, cmll_alg, cmll_post_auf] if p]
        cmll_moves = " ".join(cmll_parts).split() if cmll_parts else []
        cmll_stm = len(cmll_moves)

        # 4. Last Six Edges (LSE)
        lse_sols = solve_lse(sim, target="1look", orientation=best_fb_ori)
        if not lse_sols:
            lse_sols = solve_lse(sim, target="all", orientation=best_fb_ori)
            if not lse_sols:
                raise RuntimeError("Could not find valid LSE solution")

        best_lse = lse_sols[0]
        if best_lse.moves:
            sim.apply_moves(" ".join(best_lse.moves))

        # 5. Full Move Assembly & Verification
        full_moves: List[str] = []
        if best_fb.inspection_rotation:
            full_moves.extend(best_fb.inspection_rotation.split())
        full_moves.extend(best_fb.moves)
        full_moves.extend(best_sb.moves)
        full_moves.extend(cmll_moves)
        full_moves.extend(best_lse.moves)

        full_moves_str = " ".join(full_moves)

        # Invert inspection rotation to test cube identity in world frame
        verify_cube = sim.copy()
        if best_fb.inspection_rotation:
            inv_rot = MoveParser.invert_moves(best_fb.inspection_rotation)
            verify_cube.apply_moves(" ".join(inv_rot))

        is_valid = verify_cube.is_solved()
        total_stm = best_fb.move_count + best_sb.move_count + cmll_stm + best_lse.move_count
        duration_ms = (time.perf_counter() - t0) * 1000.0

        return FullSolveResult(
            scramble=scramble,
            fb=best_fb,
            sb=best_sb,
            cmll_case=cmll_case,
            cmll_group=cmll_group,
            cmll_pre_auf=cmll_pre_auf,
            cmll_alg=cmll_alg,
            cmll_post_auf=cmll_post_auf,
            cmll_moves=cmll_moves,
            cmll_stm=cmll_stm,
            lse=best_lse,
            full_moves=full_moves,
            full_moves_str=full_moves_str,
            total_stm=total_stm,
            duration_ms=duration_ms,
            style=style,
            is_valid=is_valid,
        )


def solve_scramble(
    scramble: str,
    style: str = "free",
    orientation: Optional[str] = None,
) -> FullSolveResult:
    """Convenience functional interface for RouxScrambleSolver."""
    solver = RouxScrambleSolver.get_instance()
    return solver.solve(
        scramble=scramble,
        style=style,
        orientation=orientation,
    )

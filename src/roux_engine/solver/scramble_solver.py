"""End-to-end Roux Scramble Solver Pipeline.

Orchestrates the complete 4-phase Roux method heuristic search:
1. First Block (FB): Top-K IDA* heuristic search using the 5.32M state FB PDB.
2. Second Block (SB): Center-aligned IDA* heuristic search using the 1.08M state SB PDB
   (supporting Free Blockbuilding, Classical Standard, and Square + Pair paradigms).
3. CMLL: Instant 42-case classification table with AUF alignment.
4. Last Six Edges (LSE): Optimal <M, U> graph search for Steps 4a, 4b, and 4c.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Union, Literal
import urllib.parse

from ..core.cube import CubeState
from ..core.orientation import (
    RouxOrientation,
    get_orientation,
    get_dual_neutral_orientations,
)
from ..core.parser import MoveParser
from ..segmenter.cmll_classifier import CMLLClassifier, CMLL_ALGS
from ..ergonomics.flow_scorer import FlowScorer
from ..ergonomics.models import HandProfile
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
    cmll_e_stm: float = 0.0
    total_e_stm: float = 0.0
    kinematic_efficiency: float = 0.0
    rank_by: str = "e_stm"
    profile: str = "2H"
    m_slice_hand: str = "right"
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
            "rank_by": self.rank_by,
            "profile": self.profile,
            "m_slice_hand": self.m_slice_hand,
            "total_stm": self.total_stm,
            "total_e_stm": round(self.total_e_stm, 4),
            "kinematic_efficiency": round(self.kinematic_efficiency, 2),
            "duration_ms": round(self.duration_ms, 2),
            "full_solution": self.full_moves_str,
            "alg_cubing_url": self.alg_cubing_url,
            "fb": {
                "moves": self.fb.moves,
                "moves_str": " ".join(self.fb.moves),
                "move_count": self.fb.move_count,
                "e_stm": self.fb.e_stm,
                "inspection_rotation": self.fb.inspection_rotation,
                "orientation": self.fb.orientation,
            },
            "sb": {
                "moves": list(self.sb.moves),
                "moves_str": " ".join(self.sb.moves),
                "move_count": self.sb.move_count,
                "e_stm": self.sb.e_stm,
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
                "e_stm": self.cmll_e_stm,
            },
            "lse": {
                "target": self.lse.target,
                "moves": self.lse.moves,
                "moves_str": " ".join(self.lse.moves),
                "move_count": self.lse.move_count,
                "e_stm": self.lse.e_stm,
                "case_name": self.lse.case_name,
            },
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes result to JSON."""
        return json.dumps(self.to_dict(), indent=indent)


def _ensure_phase_e_stm(solution: Any, flow_scorer: FlowScorer, profile: HandProfile) -> Any:
    """Ensures a phase solution record has its e_stm score populated."""
    if solution.e_stm is not None:
        return solution
    moves = list(solution.moves)
    e_val = round(flow_scorer.score_moves(moves, profile=profile).e_stm, 4) if moves else 0.0
    return replace(solution, e_stm=e_val)


class RouxScrambleSolver:
    """Heuristic search orchestrator for end-to-end Roux method solving."""

    _INSTANCE: Optional[RouxScrambleSolver] = None

    def __init__(self) -> None:
        self._flow_scorer: Optional[FlowScorer] = None

    @property
    def flow_scorer(self) -> FlowScorer:
        if self._flow_scorer is None:
            self._flow_scorer = FlowScorer()
        return self._flow_scorer

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
        rank_by: Literal["stm", "e_stm"] = "e_stm",
        profile: Optional[Union[str, HandProfile]] = None,
        m_slice_hand: Optional[str] = None,
        hand_profile: Optional[HandProfile] = None,
    ) -> FullSolveResult:
        """Solves a scramble completely from scratch using the 4-tier Roux pipeline.

        Args:
            scramble: Scramble move sequence string.
            style: Second Block solving paradigm ('free', 'classical', or 'square_pair').
            orientation: Optional orientation constraint (e.g. '' for canonical Yellow-bottom/Orange-left).
            rank_by: Candidate ranking metric ('e_stm' or 'stm', default 'e_stm').
            profile: Optional solving style profile ('2H', 'OH', or HandProfile).
            m_slice_hand: Hand preference for M-slice flicking ('right' or 'left').
            hand_profile: Optional HandProfile domain model instance.

        Returns:
            FullSolveResult with structured phase solutions, movecounts, ergonomic metrics, and full solution string.
        """
        t0 = time.perf_counter()
        cube = CubeState().apply_moves(scramble)

        if hand_profile is not None:
            resolved_profile = hand_profile
        elif isinstance(profile, HandProfile):
            if m_slice_hand is not None and profile.m_slice_hand != m_slice_hand:
                resolved_profile = HandProfile(
                    solving_mode=profile.solving_mode,
                    m_slice_hand=m_slice_hand,
                    dominant_hand=profile.dominant_hand,
                )
            else:
                resolved_profile = profile
        else:
            solving_mode = (profile or "2H").strip().upper()
            m_hand = m_slice_hand or "right"
            resolved_profile = HandProfile(solving_mode=solving_mode, m_slice_hand=m_hand)

        # 1. First Block Search across dual-neutral orientations
        fb_solver = FBSolver.get_instance()
        best_fb: Optional[FBSolution] = None
        best_fb_ori: Optional[RouxOrientation] = None

        if orientation is not None:
            target_ori = get_orientation(orientation)
            candidates = fb_solver.solve(
                cube,
                k=10,
                orientation=target_ori.rotations,
                rank_by=rank_by,
                hand_profile=resolved_profile,
            )
            if candidates:
                best_fb = candidates[0]
                best_fb_ori = get_orientation(best_fb.orientation)
        else:
            candidates = fb_solver.solve(
                cube,
                k=10,
                rank_by=rank_by,
                hand_profile=resolved_profile,
            )
            if candidates:
                best_fb = candidates[0]
                best_fb_ori = get_orientation(best_fb.orientation)

        if best_fb is None or best_fb_ori is None:
            raise RuntimeError(f"Could not find valid First Block solution for scramble: {scramble}")

        best_fb = _ensure_phase_e_stm(best_fb, self.flow_scorer, resolved_profile)

        # Apply FB to working simulation cube
        sim = cube.copy()
        if best_fb.inspection_rotation:
            sim.apply_moves(best_fb.inspection_rotation)
        if best_fb.moves:
            sim.apply_moves(" ".join(best_fb.moves))

        # 2. Second Block Search
        sb_sols = solve_sb(sim, k=1, style=style, rank_by=rank_by, hand_profile=resolved_profile)
        if not sb_sols:
            # Fallback to free if chosen style had no valid transition
            sb_sols = solve_sb(sim, k=1, style="free", rank_by=rank_by, hand_profile=resolved_profile)
            if not sb_sols:
                raise RuntimeError(f"Could not find valid Second Block solution after FB: {best_fb.moves}")

        best_sb = _ensure_phase_e_stm(sb_sols[0], self.flow_scorer, resolved_profile)

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
        cmll_e_stm = round(
            self.flow_scorer.score_moves(cmll_moves, profile=resolved_profile).e_stm, 4
        ) if cmll_moves else 0.0

        # 4. Last Six Edges (LSE)
        lse_sols = solve_lse(
            sim,
            target="1look",
            orientation=best_fb_ori,
            rank_by=rank_by,
            hand_profile=resolved_profile,
        )
        if not lse_sols:
            lse_sols = solve_lse(
                sim,
                target="all",
                orientation=best_fb_ori,
                rank_by=rank_by,
                hand_profile=resolved_profile,
            )
            if not lse_sols:
                raise RuntimeError("Could not find valid LSE solution")

        best_lse = _ensure_phase_e_stm(lse_sols[0], self.flow_scorer, resolved_profile)

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

        # Verify that applying full solution yields a solved cube
        is_valid = sim.is_solved(allow_rotations=True)
        total_stm = best_fb.move_count + best_sb.move_count + cmll_stm + best_lse.move_count
        duration_ms = (time.perf_counter() - t0) * 1000.0

        total_e_stm = round(
            (best_fb.e_stm or 0.0)
            + (best_sb.e_stm or 0.0)
            + cmll_e_stm
            + (best_lse.e_stm or 0.0),
            4,
        )
        if total_e_stm > 0:
            kinematic_efficiency = round(min(100.0, (total_stm / total_e_stm * 100.0)), 2)
        else:
            kinematic_efficiency = 0.0

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
            cmll_e_stm=cmll_e_stm,
            total_e_stm=total_e_stm,
            kinematic_efficiency=kinematic_efficiency,
            rank_by=rank_by,
            profile=resolved_profile.solving_mode,
            m_slice_hand=resolved_profile.m_slice_hand,
            is_valid=is_valid,
        )


def solve_scramble(
    scramble: str,
    style: str = "free",
    orientation: Optional[str] = None,
    rank_by: Literal["stm", "e_stm"] = "e_stm",
    profile: Optional[Union[str, HandProfile]] = None,
    m_slice_hand: Optional[str] = None,
    hand_profile: Optional[HandProfile] = None,
) -> FullSolveResult:
    """Convenience functional interface for RouxScrambleSolver."""
    solver = RouxScrambleSolver.get_instance()
    return solver.solve(
        scramble=scramble,
        style=style,
        orientation=orientation,
        rank_by=rank_by,
        profile=profile,
        m_slice_hand=m_slice_hand,
        hand_profile=hand_profile,
    )

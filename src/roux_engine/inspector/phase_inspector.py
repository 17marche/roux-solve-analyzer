"""Biomechanical Phase Inspector and Candidate Evaluator (`roux inspect`).

Provides isolated inspection and turn-by-turn physical evaluation for individual Roux
phases (First Block, Second Block, Last Six Edges, and CMLL). Formats candidates with
raw Slice Turn Metric (STM), Effective STM (E-STM), Kinematic Flow Efficiency,
and turn-by-turn wrist grip telemetry with explicit regrip markers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple

from ..core.cube import CubeState
from ..core.constants import Color
from ..core.orientation import (
    RouxOrientation,
    get_orientation,
    get_dual_neutral_orientations,
    get_all_orientations,
)
from ..core.parser import MoveParser
from ..segmenter.cmll_classifier import CMLLClassifier, CMLL_ALGS
from ..ergonomics.models import GripState, HandProfile
from ..ergonomics.flow_scorer import FlowScorer
from ..ergonomics.grip_tracker import GripTracker, GripTrackingResult
from ..solver.fb_solver import FBSolution, solve_fb
from ..solver.sb_solver import SBSolution, solve_sb
from ..solver.lse_solver import LSESolution, LSEPath, solve_lse, solve_lse_paths
from ..solver.symmetry import is_fb_solved_for_symmetry, get_all_symmetries


@dataclass(frozen=True)
class RegripEvent:
    """Represents a forced wrist regrip event during move execution."""
    move_index: int
    prev_move: Optional[str]
    curr_move: str
    from_grip: str
    to_grip: str

    def format(self) -> str:
        """Formats the regrip event into canonical human-readable telemetry."""
        if self.prev_move is not None:
            return (
                f"Regrip between move {self.move_index} ({self.prev_move}) and "
                f"move {self.move_index + 1} ({self.curr_move}): from {self.from_grip} to {self.to_grip}"
            )
        return (
            f"Regrip before move 1 ({self.curr_move}): from {self.from_grip} to {self.to_grip}"
        )


@dataclass(frozen=True)
class GripTelemetry:
    """Structured kinematic grip telemetry for a candidate move sequence."""
    start_grip: str
    end_grip: str
    regrip_count: int
    regrips: List[RegripEvent] = field(default_factory=list)

    def summary(self) -> str:
        """Returns concise single-line grip summary."""
        regrip_str = f"{self.regrip_count} regrip" if self.regrip_count == 1 else f"{self.regrip_count} regrips"
        return f"Start in {self.start_grip} -> End in {self.end_grip} ({regrip_str})"


def compute_grip_telemetry(
    moves: Union[str, Sequence[str]],
    tracker: Optional[GripTracker] = None,
) -> GripTelemetry:
    """Computes wrist grip orientation and forced regrip events for a move sequence."""
    if tracker is None:
        tracker = GripTracker()

    tokens = moves.split() if isinstance(moves, str) else list(moves)
    if not tokens:
        return GripTelemetry(
            start_grip=GripState.HOME.value,
            end_grip=GripState.HOME.value,
            regrip_count=0,
            regrips=[],
        )

    res: GripTrackingResult = tracker.track(tokens, initial_grip=GripState.HOME)
    regrip_events: List[RegripEvent] = []

    for idx, step in enumerate(res.steps):
        if step.regrip:
            prev_token = res.steps[idx - 1].move if idx > 0 else None
            regrip_events.append(
                RegripEvent(
                    move_index=idx,
                    prev_move=prev_token,
                    curr_move=step.move,
                    from_grip=step.grip_before.value,
                    to_grip=step.grip_during.value,
                )
            )

    return GripTelemetry(
        start_grip=res.initial_grip.value,
        end_grip=res.final_grip.value,
        regrip_count=res.regrip_count,
        regrips=regrip_events,
    )


def format_grip_telemetry(
    moves: Union[str, Sequence[str]],
    tracker: Optional[GripTracker] = None,
) -> List[str]:
    """Generates formatted telemetry lines for display."""
    telem = compute_grip_telemetry(moves, tracker=tracker)
    lines = [f"Grip: {telem.summary()}"]
    for ev in telem.regrips:
        lines.append(f"  {ev.format()}")
    return lines


@dataclass(frozen=True)
class CandidateInspection:
    """Evaluated solver candidate sequence with biomechanical and kinematic telemetry."""
    rank: int
    moves: List[str]
    moves_str: str
    stm: int
    e_stm: float
    flow_efficiency: float
    orientation: str
    grip: GripTelemetry
    label: Optional[str] = None
    substeps: Optional[Dict[str, int]] = None
    inspection_rotation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes candidate into a dictionary."""
        d: Dict[str, Any] = {
            "rank": self.rank,
            "moves": self.moves,
            "moves_str": self.moves_str,
            "stm": self.stm,
            "e_stm": round(self.e_stm, 4),
            "flow_efficiency": round(self.flow_efficiency, 2),
            "orientation": self.orientation,
            "grip": {
                "start": self.grip.start_grip,
                "end": self.grip.end_grip,
                "regrip_count": self.grip.regrip_count,
                "regrips": [
                    {
                        "move_index": r.move_index,
                        "prev_move": r.prev_move,
                        "curr_move": r.curr_move,
                        "from_grip": r.from_grip,
                        "to_grip": r.to_grip,
                    }
                    for r in self.grip.regrips
                ],
            },
        }
        if self.inspection_rotation:
            d["inspection_rotation"] = self.inspection_rotation
        if self.label:
            d["label"] = self.label
        if self.substeps:
            d["substeps"] = self.substeps
        return d


@dataclass(frozen=True)
class PhaseInspectionResult:
    """Complete result of a phase inspection run."""
    phase: str
    scramble: str
    profile: str
    m_slice_hand: str
    setup_moves: Dict[str, List[str]]
    is_comparison: bool
    candidates: List[CandidateInspection]
    comparison_table: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes inspection result into a dictionary."""
        res: Dict[str, Any] = {
            "phase": self.phase,
            "scramble": self.scramble,
            "profile": self.profile,
            "m_slice_hand": self.m_slice_hand,
            "setup_moves": self.setup_moves,
            "is_comparison": self.is_comparison,
            "candidates": [c.to_dict() for c in self.candidates],
        }
        if self.comparison_table is not None:
            res["comparison_table"] = self.comparison_table
        return res

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes inspection result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class PhaseInspector:
    """Inspection engine evaluating solver candidate move sequences per Roux phase."""

    def __init__(
        self,
        flow_scorer: Optional[FlowScorer] = None,
        grip_tracker: Optional[GripTracker] = None,
    ) -> None:
        self.grip_tracker = grip_tracker or GripTracker()
        self.flow_scorer = flow_scorer or FlowScorer(grip_tracker=self.grip_tracker)

    def _validate_tokens(self, move_str: str, name: str) -> List[str]:
        """Parses and validates cube move tokens strictly."""
        if not move_str.strip():
            return []
        try:
            events = MoveParser.parse_string(move_str, strict=True)
        except Exception as e:
            raise ValueError(f"Invalid cube moves in {name}: {e}") from e
        return [e.move for e in events]

    def _is_fb_solved(self, cube: CubeState) -> bool:
        """Verifies if First Block is solved in any dual-neutral orientation."""
        for candidate in get_dual_neutral_orientations():
            if candidate.is_fb_solved(cube):
                return True
        for s in get_all_symmetries():
            if is_fb_solved_for_symmetry(cube, s, inspected=False):
                return True
        return False

    def _is_sb_solved(self, cube: CubeState) -> bool:
        """Verifies if both First Block and Second Block are solved in any orientation."""
        for candidate in get_all_orientations():
            if candidate.is_fb_solved(cube) and candidate.is_sb_solved(cube):
                return True
            if candidate.rotations:
                c_rot = cube.copy().apply_moves(candidate.rotations)
                if candidate.is_fb_solved(c_rot) and candidate.is_sb_solved(c_rot):
                    return True
        return False

    def inspect(
        self,
        scramble: str,
        phase: str,
        top_k: int = 5,
        style: Optional[str] = None,
        order: Optional[str] = None,
        target: Optional[str] = None,
        fb_override: Optional[str] = None,
        sb_override: Optional[str] = None,
        moves_override: Optional[str] = None,
        profile: Union[str, HandProfile] = "2H",
        m_slice_hand: str = "right",
    ) -> PhaseInspectionResult:
        """Executes targeted phase inspection and candidate evaluation."""
        norm_phase = phase.lower().strip()
        if norm_phase not in ("fb", "sb", "lse", "cmll"):
            raise ValueError(f"Unknown phase '{phase}'. Supported phases: 'fb', 'sb', 'lse', 'cmll'.")

        if isinstance(profile, HandProfile):
            hand_prof = HandProfile.resolve(profile)
        else:
            hand_prof = HandProfile(solving_mode=str(profile), m_slice_hand=m_slice_hand)
        assert hand_prof is not None

        # Base scrambled simulation state
        cube = CubeState()
        scramble_tokens = self._validate_tokens(scramble, "scramble")
        if scramble_tokens:
            cube.apply_moves(" ".join(scramble_tokens))

        if norm_phase == "fb":
            return self._inspect_fb(cube, scramble, top_k, hand_prof)

        if norm_phase == "sb":
            return self._inspect_sb(
                cube, scramble, top_k, style, order, fb_override, moves_override, hand_prof
            )

        if norm_phase == "lse":
            return self._inspect_lse(
                cube, scramble, top_k, target, fb_override, sb_override, moves_override, hand_prof
            )

        if norm_phase == "cmll":
            return self._inspect_cmll(
                cube, scramble, top_k, fb_override, sb_override, moves_override, hand_prof
            )

        raise ValueError(f"Unsupported phase: {norm_phase}")

    def _evaluate_candidate(
        self,
        rank: int,
        moves: Sequence[str],
        orientation: str,
        profile: HandProfile,
        label: Optional[str] = None,
        substeps: Optional[Dict[str, int]] = None,
        inspection_rotation: Optional[str] = None,
    ) -> CandidateInspection:
        """Scores candidate moves through flow scoring and grip tracking."""
        move_list = list(moves)
        score = self.flow_scorer.score_moves(move_list, profile=profile)
        telem = compute_grip_telemetry(move_list, tracker=self.grip_tracker)
        return CandidateInspection(
            rank=rank,
            moves=move_list,
            moves_str=" ".join(move_list),
            stm=score.raw_stm,
            e_stm=score.e_stm,
            flow_efficiency=score.kinematic_efficiency,
            orientation=orientation,
            grip=telem,
            label=label,
            substeps=substeps,
            inspection_rotation=inspection_rotation,
        )

    def _inspect_fb(
        self,
        cube: CubeState,
        scramble: str,
        top_k: int,
        profile: HandProfile,
    ) -> PhaseInspectionResult:
        """Inspects First Block candidate solutions."""
        solutions = solve_fb(cube, k=top_k, rank_by="e_stm", hand_profile=profile, timeout_ms=None)
        candidates: List[CandidateInspection] = []
        for rank, sol in enumerate(solutions, 1):
            cand = self._evaluate_candidate(
                rank=rank,
                moves=sol.moves,
                orientation=sol.orientation,
                profile=profile,
                inspection_rotation=sol.inspection_rotation if sol.inspection_rotation else None,
            )
            candidates.append(cand)

        return PhaseInspectionResult(
            phase="fb",
            scramble=scramble,
            profile=profile.solving_mode,
            m_slice_hand=profile.m_slice_hand,
            setup_moves={},
            is_comparison=False,
            candidates=candidates,
        )

    def _inspect_sb(
        self,
        cube: CubeState,
        scramble: str,
        top_k: int,
        style: Optional[str],
        order: Optional[str],
        fb_override: Optional[str],
        moves_override: Optional[str],
        profile: HandProfile,
    ) -> PhaseInspectionResult:
        """Inspects Second Block candidates with cascading or multi-paradigm comparison."""
        sim = cube.copy()
        setup_moves: Dict[str, List[str]] = {}

        # 1. State pre-conditioning
        if moves_override is not None:
            toks = self._validate_tokens(moves_override, "--moves")
            sim.apply_moves(" ".join(toks))
            setup_moves["moves"] = toks
            if not self._is_fb_solved(sim):
                raise ValueError("First Block is not solved on the cube after --moves setup. FB must be solved before SB.")
        elif fb_override is not None:
            toks = self._validate_tokens(fb_override, "--fb")
            sim.apply_moves(" ".join(toks))
            setup_moves["fb"] = toks
            if not self._is_fb_solved(sim):
                raise ValueError("First Block is not solved on the cube after --fb setup. FB must be solved before SB.")
        else:
            # Auto-cascade optimal FB
            fb_sols = solve_fb(cube, k=1, rank_by="e_stm", hand_profile=profile, timeout_ms=None)
            if not fb_sols:
                raise RuntimeError("Failed to auto-solve First Block for Second Block inspection.")
            best_fb = fb_sols[0]
            fb_moves: List[str] = []
            if best_fb.inspection_rotation:
                fb_moves.extend(best_fb.inspection_rotation.split())
            fb_moves.extend(best_fb.moves)
            sim.apply_moves(" ".join(fb_moves))
            setup_moves["fb"] = fb_moves

        # 2. Paradigm comparison vs style drill-down
        if style is None:
            # Multi-paradigm comparison table
            # Paradigms: Free, Classical (Back-first), Classical (Front-first), Square + Pair, Macro-Trigger
            paradigms = [
                ("Free Blockbuilding", "free", "best", False),
                ("Classical (Back-first)", "classical", "back_first", False),
                ("Classical (Front-first)", "classical", "front_first", False),
                ("Square + Pair", "square_pair", "best", False),
                ("Macro-Trigger", "free", "best", True),
            ]
            comp_candidates: List[CandidateInspection] = []
            comp_table: Dict[str, Any] = {}

            for idx, (label, s_name, o_name, allow_macro) in enumerate(paradigms, 1):
                sols = solve_sb(
                    sim,
                    k=1,
                    style=s_name,
                    order=o_name,
                    allow_macro_triggers=allow_macro,
                    rank_by="e_stm",
                    hand_profile=profile,
                )
                if sols:
                    cand = self._evaluate_candidate(
                        rank=idx,
                        moves=sols[0].moves,
                        orientation=sols[0].orientation,
                        profile=profile,
                        label=label,
                    )
                    comp_candidates.append(cand)
                    comp_table[label] = cand.to_dict()

            # Rank by E-STM ascending
            comp_candidates.sort(key=lambda c: (c.e_stm, c.stm))
            ranked_candidates = [
                self._evaluate_candidate(
                    rank=r + 1,
                    moves=c.moves,
                    orientation=c.orientation,
                    profile=profile,
                    label=c.label,
                )
                for r, c in enumerate(comp_candidates)
            ]

            return PhaseInspectionResult(
                phase="sb",
                scramble=scramble,
                profile=profile.solving_mode,
                m_slice_hand=profile.m_slice_hand,
                setup_moves=setup_moves,
                is_comparison=True,
                candidates=ranked_candidates,
                comparison_table=comp_table,
            )

        # Style drill-down
        style_norm = style.lower().replace("-", "_")
        allow_macro = False
        if style_norm in ("macro", "macro_trigger"):
            style_norm = "free"
            allow_macro = True

        order_norm = (order or "best").lower().replace("-", "_")
        solutions = solve_sb(
            sim,
            k=top_k,
            style=style_norm,
            order=order_norm,
            allow_macro_triggers=allow_macro,
            rank_by="e_stm",
            hand_profile=profile,
        )
        candidates = [
            self._evaluate_candidate(
                rank=r + 1,
                moves=sol.moves,
                orientation=sol.orientation,
                profile=profile,
                label=sol.style,
            )
            for r, sol in enumerate(solutions)
        ]

        return PhaseInspectionResult(
            phase="sb",
            scramble=scramble,
            profile=profile.solving_mode,
            m_slice_hand=profile.m_slice_hand,
            setup_moves=setup_moves,
            is_comparison=False,
            candidates=candidates,
        )

    def _inspect_lse(
        self,
        cube: CubeState,
        scramble: str,
        top_k: int,
        target: Optional[str],
        fb_override: Optional[str],
        sb_override: Optional[str],
        moves_override: Optional[str],
        profile: HandProfile,
    ) -> PhaseInspectionResult:
        """Inspects Last Six Edges candidates with cascading or multi-path comparison."""
        sim = cube.copy()
        setup_moves: Dict[str, List[str]] = {}

        # 1. State pre-conditioning
        if moves_override is not None:
            toks = self._validate_tokens(moves_override, "--moves")
            sim.apply_moves(" ".join(toks))
            setup_moves["moves"] = toks
            if not self._is_sb_solved(sim):
                raise ValueError(
                    "First Block and Second Block are not solved on the cube after --moves setup. "
                    "Both blocks must be solved before inspecting Last Six Edges."
                )
        else:
            fb_toks: List[str] = []
            sb_toks: List[str] = []
            if fb_override is not None:
                fb_toks = self._validate_tokens(fb_override, "--fb")
                sim.apply_moves(" ".join(fb_toks))
                setup_moves["fb"] = fb_toks
                if not self._is_fb_solved(sim):
                    raise ValueError("First Block is not solved after --fb setup.")
            else:
                # Auto-solve FB
                fb_sols = solve_fb(cube, k=1, rank_by="e_stm", hand_profile=profile, timeout_ms=None)
                if not fb_sols:
                    raise RuntimeError("Failed to auto-solve First Block for LSE cascade.")
                best_fb = fb_sols[0]
                if best_fb.inspection_rotation:
                    fb_toks.extend(best_fb.inspection_rotation.split())
                fb_toks.extend(best_fb.moves)
                sim.apply_moves(" ".join(fb_toks))
                setup_moves["fb"] = fb_toks

            if sb_override is not None:
                sb_toks = self._validate_tokens(sb_override, "--sb")
                sim.apply_moves(" ".join(sb_toks))
                setup_moves["sb"] = sb_toks
                if not self._is_sb_solved(sim):
                    raise ValueError("Second Block is not solved after --sb setup.")
            else:
                # Auto-solve SB
                sb_sols = solve_sb(sim, k=1, style="free", rank_by="e_stm", hand_profile=profile)
                if not sb_sols:
                    raise RuntimeError("Failed to auto-solve Second Block for LSE cascade.")
                sb_toks = list(sb_sols[0].moves)
                sim.apply_moves(" ".join(sb_toks))
                setup_moves["sb"] = sb_toks

            # Auto-solve CMLL if corners not solved
            # Find orientation
            ori: Optional[RouxOrientation] = None
            for candidate in get_dual_neutral_orientations():
                if candidate.is_fb_solved(sim) and candidate.is_sb_solved(sim):
                    ori = candidate
                    break
            if ori is not None:
                case_id, group, pre_auf = CMLLClassifier.classify_state(sim, block=ori)
                if case_id != "solved":
                    alg = ""
                    for cid, grp, a in CMLL_ALGS:
                        if cid == case_id:
                            alg = a
                            break
                    cmll_toks: List[str] = []
                    if pre_auf:
                        sim.apply_move(pre_auf)
                        cmll_toks.append(pre_auf)
                    if alg:
                        sim.apply_moves(alg)
                        cmll_toks.extend(alg.split())
                    # Post-AUF
                    ref_c = CubeState()
                    if ori.rotations:
                        ref_c.apply_moves(ori.rotations)
                    target_ufl = int(ref_c.cp[0])
                    for auf in ("", "U", "U2", "U'"):
                        tc = sim.copy()
                        if auf:
                            tc.apply_move(auf)
                        if int(tc.cp[0]) == target_ufl:
                            if auf:
                                sim.apply_move(auf)
                                cmll_toks.append(auf)
                            break
                    if cmll_toks:
                        setup_moves["cmll"] = cmll_toks

        # 2. Multi-path comparison vs micro-step drill-down
        target_norm = (target or "paths").lower().replace("-", "_")

        if target_norm == "paths":
            # Multi-path comparison table (Standard EO, EOLR Aligned, EOLR Misoriented, EOLR-b)
            path_dict = solve_lse_paths(sim, rank_by="e_stm", hand_profile=profile)
            label_map = {
                "standard": "Standard EO",
                "eolr": "EOLR Aligned",
                "eolr_misoriented": "EOLR Misoriented",
                "eolr_b": "EOLR-b",
            }
            comp_candidates: List[CandidateInspection] = []
            comp_table: Dict[str, Any] = {}

            for key in ("standard", "eolr", "eolr_misoriented", "eolr_b"):
                if key in path_dict:
                    p = path_dict[key]
                    cand = self._evaluate_candidate(
                        rank=len(comp_candidates) + 1,
                        moves=p.total_moves,
                        orientation=p.orientation,
                        profile=profile,
                        label=label_map.get(key, key),
                        substeps={
                            "4a": p.step_4a.move_count,
                            "4b": p.step_4b.move_count,
                            "4c": p.step_4c.move_count,
                        },
                    )
                    comp_candidates.append(cand)
                    cand_dict = cand.to_dict()
                    cand_dict["total_stm"] = p.total_move_count
                    cand_dict["total_e_stm"] = round(cand.e_stm, 4)
                    comp_table[label_map.get(key, key)] = cand_dict

            # Rank by E-STM ascending
            comp_candidates.sort(key=lambda c: (c.e_stm, c.stm))
            ranked_candidates = [
                self._evaluate_candidate(
                    rank=r + 1,
                    moves=c.moves,
                    orientation=c.orientation,
                    profile=profile,
                    label=c.label,
                    substeps=c.substeps,
                )
                for r, c in enumerate(comp_candidates)
            ]

            return PhaseInspectionResult(
                phase="lse",
                scramble=scramble,
                profile=profile.solving_mode,
                m_slice_hand=profile.m_slice_hand,
                setup_moves=setup_moves,
                is_comparison=True,
                candidates=ranked_candidates,
                comparison_table=comp_table,
            )

        # Micro-step drill-down
        sols = solve_lse(
            sim,
            target=target_norm,
            top_k=top_k,
            rank_by="e_stm",
            hand_profile=profile,
        )
        candidates = [
            self._evaluate_candidate(
                rank=r + 1,
                moves=sol.moves,
                orientation=sol.orientation,
                profile=profile,
                label=sol.case_name,
            )
            for r, sol in enumerate(sols)
        ]

        return PhaseInspectionResult(
            phase="lse",
            scramble=scramble,
            profile=profile.solving_mode,
            m_slice_hand=profile.m_slice_hand,
            setup_moves=setup_moves,
            is_comparison=False,
            candidates=candidates,
        )

    def _inspect_cmll(
        self,
        cube: CubeState,
        scramble: str,
        top_k: int,
        fb_override: Optional[str],
        sb_override: Optional[str],
        moves_override: Optional[str],
        profile: HandProfile,
    ) -> PhaseInspectionResult:
        """Inspects CMLL case and algorithm candidate."""
        sim = cube.copy()
        setup_moves: Dict[str, List[str]] = {}

        if moves_override is not None:
            toks = self._validate_tokens(moves_override, "--moves")
            sim.apply_moves(" ".join(toks))
            setup_moves["moves"] = toks
        else:
            fb_toks: List[str] = []
            if fb_override is not None:
                fb_toks = self._validate_tokens(fb_override, "--fb")
                sim.apply_moves(" ".join(fb_toks))
                setup_moves["fb"] = fb_toks
            else:
                fb_sols = solve_fb(cube, k=1, rank_by="e_stm", hand_profile=profile, timeout_ms=None)
                if fb_sols:
                    if fb_sols[0].inspection_rotation:
                        fb_toks.extend(fb_sols[0].inspection_rotation.split())
                    fb_toks.extend(fb_sols[0].moves)
                    sim.apply_moves(" ".join(fb_toks))
                    setup_moves["fb"] = fb_toks

            sb_toks: List[str] = []
            if sb_override is not None:
                sb_toks = self._validate_tokens(sb_override, "--sb")
                sim.apply_moves(" ".join(sb_toks))
                setup_moves["sb"] = sb_toks
            else:
                sb_sols = solve_sb(sim, k=1, style="free", rank_by="e_stm", hand_profile=profile)
                if sb_sols:
                    sb_toks = list(sb_sols[0].moves)
                    sim.apply_moves(" ".join(sb_toks))
                    setup_moves["sb"] = sb_toks

        # Classify CMLL
        ori: Optional[RouxOrientation] = None
        for candidate in get_dual_neutral_orientations():
            if candidate.is_fb_solved(sim) and candidate.is_sb_solved(sim):
                ori = candidate
                break

        case_id, group, pre_auf = CMLLClassifier.classify_state(sim, block=ori)
        alg = ""
        for cid, grp, a in CMLL_ALGS:
            if cid == case_id:
                alg = a
                break

        cmll_moves: List[str] = []
        if pre_auf:
            cmll_moves.append(pre_auf)
        if alg:
            cmll_moves.extend(alg.split())

        cand = self._evaluate_candidate(
            rank=1,
            moves=cmll_moves,
            orientation=ori.rotations if ori else "",
            profile=profile,
            label=f"{group} ({case_id})",
        )

        return PhaseInspectionResult(
            phase="cmll",
            scramble=scramble,
            profile=profile.solving_mode,
            m_slice_hand=profile.m_slice_hand,
            setup_moves=setup_moves,
            is_comparison=False,
            candidates=[cand],
        )


def inspect_phase(
    scramble: str,
    phase: str,
    top_k: int = 5,
    style: Optional[str] = None,
    order: Optional[str] = None,
    target: Optional[str] = None,
    fb_override: Optional[str] = None,
    sb_override: Optional[str] = None,
    moves_override: Optional[str] = None,
    profile: Union[str, HandProfile] = "2H",
    m_slice_hand: str = "right",
) -> PhaseInspectionResult:
    """Convenience functional interface for PhaseInspector.inspect."""
    inspector = PhaseInspector()
    return inspector.inspect(
        scramble=scramble,
        phase=phase,
        top_k=top_k,
        style=style,
        order=order,
        target=target,
        fb_override=fb_override,
        sb_override=sb_override,
        moves_override=moves_override,
        profile=profile,
        m_slice_hand=m_slice_hand,
    )


def format_inspect_report(result: PhaseInspectionResult) -> str:
    """Formats PhaseInspectionResult into a clean speedcubing terminal report."""
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("                      ROUX ERGONOMIC PHASE INSPECTOR")
    lines.append("=" * 80)
    phase_names = {
        "fb": "First Block (FB)",
        "sb": "Second Block (SB)",
        "lse": "Last Six Edges (LSE)",
        "cmll": "CMLL",
    }
    lines.append(f"  Phase:      {phase_names.get(result.phase, result.phase.upper())}")
    lines.append(f"  Scramble:   {result.scramble}")
    m_hand_desc = f" ({result.m_slice_hand}-handed M-slice)" if result.profile == "2H" else ""
    lines.append(f"  Profile:    {result.profile}{m_hand_desc}")
    lines.append("  Ranking:    Effective STM (E-STM)")

    if result.setup_moves:
        lines.append("-" * 80)
        lines.append("  Preceding Setup Moves:")
        for k, toks in result.setup_moves.items():
            if k == "fb":
                k_name = "First Block (FB)"
            elif k == "sb":
                k_name = "Second Block (SB)"
            elif k == "cmll":
                k_name = "CMLL"
            else:
                k_name = "Setup"
            lines.append(f"    • {f'{k_name}:':<22} {' '.join(toks)}")

    lines.append("-" * 80)

    # Multi-Paradigm Comparison Table for SB
    if result.is_comparison and result.phase == "sb":
        lines.append("Second Block Multi-Paradigm Comparison:")
        lines.append("-" * 80)
        lines.append("  * indicates lowest E-STM candidate")
        lines.append("")
        hdr_sb = f"  {'':<2} {'Paradigm':<26} | {'STM':<4} | {'E-STM':<7} | {'Flow Eff':<8} | {'Regrips':<7} | {'Moves'}"
        lines.append(hdr_sb)
        lines.append("  " + "-" * 76)
        min_estm = min(c.e_stm for c in result.candidates) if result.candidates else 0.0
        for c in result.candidates:
            star = "* " if abs(c.e_stm - min_estm) < 1e-4 else "  "
            p_label = c.label or "Paradigm"
            eff_pct = f"{c.flow_efficiency:>7.1f}%"
            lines.append(
                f"  {star}{p_label:<26} | {c.stm:>4} | {c.e_stm:>7.2f} | {eff_pct} | {c.grip.regrip_count:>7} | {c.moves_str}"
            )
        lines.append("-" * 80)
        lines.append("Detailed Telemetry by Paradigm:")
        lines.append("-" * 80)
        for c in result.candidates:
            p_label = c.label or f"Candidate #{c.rank}"
            lines.append(f"• {p_label}:")
            eff_str = f"{c.flow_efficiency:.1f}% Flow Eff"
            lines.append(f"  Moves: {c.moves_str} ({c.stm} STM // {c.e_stm:.2f} E-STM // {eff_str})")
            lines.append(f"  Grip:  {c.grip.summary()}")
            for r in c.grip.regrips:
                lines.append(f"    - {r.format()}")
            lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines).strip()

    # Multi-Path Comparison Table for LSE
    if result.is_comparison and result.phase == "lse":
        lines.append("Last Six Edges Multi-Path Comparison:")
        lines.append("-" * 80)
        lines.append("  * indicates lowest E-STM path")
        lines.append("")
        hdr_lse = (
            f"  {'':<2} {'Path':<20} | {'Total STM':<9} | {'Total E-STM':<11} | "
            f"{'Flow Eff':<8} | {'Sub-steps (4a/4b/4c)'}"
        )
        lines.append(hdr_lse)
        lines.append("  " + "-" * 76)
        min_estm = min(c.e_stm for c in result.candidates) if result.candidates else 0.0
        for c in result.candidates:
            star = "* " if abs(c.e_stm - min_estm) < 1e-4 else "  "
            p_label = c.label or "Path"
            sub_str = ""
            if c.substeps:
                sub_str = f"4a: {c.substeps.get('4a', 0)} | 4b: {c.substeps.get('4b', 0)} | 4c: {c.substeps.get('4c', 0)}"
            eff_pct = f"{c.flow_efficiency:>7.1f}%"
            lines.append(
                f"  {star}{p_label:<20} | {c.stm:>9} | {c.e_stm:>11.2f} | {eff_pct} | {sub_str}"
            )
        lines.append("-" * 80)
        lines.append("Detailed Telemetry by Path:")
        lines.append("-" * 80)
        for c in result.candidates:
            p_label = c.label or f"Candidate #{c.rank}"
            lines.append(f"• {p_label}:")
            eff_str = f"{c.flow_efficiency:.1f}% Flow Eff"
            lines.append(f"  Moves: {c.moves_str} ({c.stm} STM // {c.e_stm:.2f} E-STM // {eff_str})")
            lines.append(f"  Grip:  {c.grip.summary()}")
            for r in c.grip.regrips:
                lines.append(f"    - {r.format()}")
            lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines).strip()

    # Standard Top-K Candidate List
    lines.append(f"Candidate Solutions (Top {len(result.candidates)}):")
    lines.append("-" * 80)
    for c in result.candidates:
        lines.append(f"#{c.rank}: {c.moves_str}")
        lines.append(f"    STM: {c.stm} | E-STM: {c.e_stm:.4f} | Flow Efficiency: {c.flow_efficiency:.1f}%")
        if c.orientation:
            lines.append(f"    Orientation: {c.orientation}")
        lines.append(f"    Grip: {c.grip.summary()}")
        for r in c.grip.regrips:
            lines.append(f"    {r.format()}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines).strip()

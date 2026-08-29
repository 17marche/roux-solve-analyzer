"""Master Roux Phase Segmenter Pipeline."""

from __future__ import annotations
from typing import List, Optional, Dict, Any, Union, Sequence

from ..core.cube import CubeState
from ..core.parser import MoveEvent, MoveParser
from .models import SegmentedSolve, FBPhase, SBPhase, CMLLPhase, LSEPhase
from .fb_detector import FBDetector
from .sb_detector import SBDetector
from .cmll_classifier import CMLLClassifier
from .lse_classifier import LSEClassifier


class RouxSegmenter:
    """Master phase segmenter for the Roux method."""

    def __init__(self, full_color_neutral: bool = False) -> None:
        self.full_color_neutral = full_color_neutral

    def segment(self, scramble: str, solution: str, time_sec: Optional[float] = None) -> SegmentedSolve:
        """Segments a text-based solve into Roux micro-phases."""
        events = MoveParser.parse_string(solution)
        return self.segment_events(scramble, events, time_sec=time_sec)

    def segment_stream(self, scramble: str, stream: List[Dict[str, Any]]) -> SegmentedSolve:
        """Segments a timestamped smart-cube stream into Roux micro-phases."""
        events = MoveParser.parse_smart_cube_stream(stream)
        return self.segment_events(scramble, events)

    def segment_events(
        self,
        scramble: str,
        events: Sequence[MoveEvent],
        time_sec: Optional[float] = None
    ) -> SegmentedSolve:
        """Executes phase segmentation across parsed MoveEvents."""
        initial_state = CubeState()
        initial_state.apply_moves(scramble)

        total_stm = len([e for e in events if not e.move.startswith(('x', 'y', 'z'))])
        total_time_ms = None
        if time_sec is not None and time_sec > 0:
            total_time_ms = int(round(time_sec * 1000.0))
        elif events and events[-1].timestamp_ms is not None:
            first_t = events[0].timestamp_ms or 0
            last_t = events[-1].timestamp_ms
            total_time_ms = max(0, last_t - first_t) if last_t is not None else None

        tps = None
        if total_time_ms and total_time_ms > 0:
            tps = round(total_stm / (total_time_ms / 1000.0), 2)

        # 1. First Block Detection
        fb_res = FBDetector.detect_fb(
            initial_state=initial_state,
            events=events,
            full_color_neutral=self.full_color_neutral
        )
        if fb_res is None:
            return SegmentedSolve(
                scramble=scramble,
                solution_moves_count=len(events),
                total_moves_stm=total_stm,
                total_time_ms=total_time_ms,
                tps=tps,
                is_valid=False,
                warnings=["First Block (FB) completion not found in solve."]
            )

        fb_phase, ori_rotations, fb_state, matched_block = fb_res

        # 2. Second Block Detection
        sb_res = SBDetector.detect_sb(
            fb_state=fb_state,
            events=events,
            fb_end_idx=fb_phase.end_move_idx,
            block=matched_block
        )
        if sb_res is None:
            return SegmentedSolve(
                scramble=scramble,
                solution_moves_count=len(events),
                total_moves_stm=total_stm,
                total_time_ms=total_time_ms,
                tps=tps,
                is_valid=False,
                fb=fb_phase,
                warnings=["Second Block (SB) completion not found in solve."]
            )

        sb_phase, sb_state = sb_res

        # 3. CMLL Detection
        cmll_res = CMLLClassifier.detect_cmll(
            sb_state=sb_state,
            events=events,
            sb_end_idx=sb_phase.end_move_idx,
            block=matched_block
        )
        if cmll_res is None:
            return SegmentedSolve(
                scramble=scramble,
                solution_moves_count=len(events),
                total_moves_stm=total_stm,
                total_time_ms=total_time_ms,
                tps=tps,
                is_valid=False,
                fb=fb_phase,
                sb=sb_phase,
                warnings=["CMLL corner orientation/permutation completion not found in solve."]
            )

        cmll_phase, cmll_state = cmll_res

        # 4. LSE Detection
        lse_res = LSEClassifier.detect_lse(
            cmll_state=cmll_state,
            events=events,
            cmll_end_idx=cmll_phase.end_move_idx,
            block=matched_block
        )
        if lse_res is None:
            return SegmentedSolve(
                scramble=scramble,
                solution_moves_count=len(events),
                total_moves_stm=total_stm,
                total_time_ms=total_time_ms,
                tps=tps,
                is_valid=False,
                fb=fb_phase,
                sb=sb_phase,
                cmll=cmll_phase,
                warnings=["LSE completion / fully solved cube state not found in solve."]
            )

        lse_phase, final_state = lse_res

        return SegmentedSolve(
            scramble=scramble,
            solution_moves_count=len(events),
            total_moves_stm=total_stm,
            total_time_ms=total_time_ms,
            tps=tps,
            is_valid=True,
            fb=fb_phase,
            sb=sb_phase,
            cmll=cmll_phase,
            lse=lse_phase
        )

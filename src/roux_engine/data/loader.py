"""Dataset Loader, Ingestion, and Quality Validation Pipeline for Solve Archives."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import json
import os

from ..core.cube import CubeState
from ..core.parser import MoveParser
from ..segmenter.segmenter import RouxSegmenter


@dataclass
class SolveRecord:
    """Represents a Rubik's cube solve record with metadata and validation status."""
    id: str | int
    scramble: str
    solution: str
    solver: Optional[str] = None
    puzzle: Optional[str] = "3x3"
    result: Optional[str] = None
    competition: Optional[str] = None
    date: Optional[str] = None
    tps: Optional[float] = None
    is_valid: bool = True
    validation_status: str = "VALID"  # "VALID", "UNSOLVABLE", "NON_ROUX_OR_ANOMALY"
    invalid_reason: Optional[str] = None
    raw_reconstruction: Optional[str] = None
    human_splits: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecoDatasetLoader:
    """Loads and validates solve datasets from reco.nz archives or JSON files."""

    def __init__(self, segmenter: Optional[RouxSegmenter] = None) -> None:
        self.segmenter = segmenter or RouxSegmenter()
        self._fcn_segmenter = RouxSegmenter(full_color_neutral=True)

    def validate_solve(
        self,
        scramble: str,
        solution: str,
        allow_rotations: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """Validates that a solve physically solves the cube and segments into Roux phases."""
        is_valid, _, reason = self.validate_solve_detailed(
            scramble, solution, allow_rotations=allow_rotations
        )
        return is_valid, reason

    def validate_solve_detailed(
        self,
        scramble: str,
        solution: str,
        allow_rotations: bool = True
    ) -> Tuple[bool, str, Optional[str]]:
        """Detailed multi-stage solve validator returning (is_valid, status, reason)."""
        if not scramble or not solution:
            return False, "UNSOLVABLE", "Empty scramble or solution"

        # 1. Parseability test
        try:
            scramble_events = MoveParser.parse_string(scramble)
            solution_events = MoveParser.parse_string(solution)
        except Exception as e:
            return False, "UNSOLVABLE", f"Notation parsing error: {e}"

        if not scramble_events:
            return False, "UNSOLVABLE", "No valid moves in scramble"
        if not solution_events:
            return False, "UNSOLVABLE", "No valid moves in solution"

        # 2. Solvability test (Scramble + Solution == Solved state)
        sim = CubeState()
        try:
            for ev in scramble_events:
                sim.apply_move(ev.move)
            for ev in solution_events:
                sim.apply_move(ev.move)
        except Exception as e:
            return False, "UNSOLVABLE", f"Move application error: {e}"

        if not sim.is_solved(allow_rotations=allow_rotations):
            return False, "UNSOLVABLE", "Solution does not restore cube to solved state"

        # 3. Roux Phase Progression test (Dual-Neutral with automatic FCN fallback)
        try:
            seg = self.segmenter.segment_events(scramble, solution_events)
            if not seg.is_valid and not self.segmenter.full_color_neutral:
                fcn_seg = self._fcn_segmenter.segment_events(scramble, solution_events)
                if fcn_seg.is_valid:
                    seg = fcn_seg

            if not seg.is_valid:
                reason = seg.warnings[0] if seg.warnings else "Roux segmentation failed"
                return False, "NON_ROUX_OR_ANOMALY", reason
        except Exception as e:
            return False, "NON_ROUX_OR_ANOMALY", f"Segmentation exception: {e}"

        return True, "VALID", None

    def load_from_json(self, file_path: str, revalidate: bool = False) -> List[SolveRecord]:
        """Loads solve records from a JSON file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        records: List[SolveRecord] = []
        for item in raw_data:
            rec_id = item.get("id", len(records))
            scramble = item.get("scramble", "")
            solution = item.get("solution", "")
            solver = item.get("solver")
            puzzle = item.get("puzzle", "3x3")
            result = item.get("result")
            competition = item.get("competition")
            date = item.get("date")
            tps_val = float(item["tps"]) if item.get("tps") else None
            raw_recon = item.get("raw_reconstruction")
            human_splits = item.get("human_splits")

            if revalidate or "is_valid" not in item:
                if scramble and solution:
                    is_valid, status, reason = self.validate_solve_detailed(scramble, solution)
                else:
                    is_valid, status, reason = False, "UNSOLVABLE", "Missing scramble or solution"
            else:
                is_valid = item.get("is_valid", True)
                status = item.get("validation_status", "VALID" if is_valid else "NON_ROUX_OR_ANOMALY")
                reason = item.get("invalid_reason")

            records.append(SolveRecord(
                id=rec_id,
                scramble=scramble,
                solution=solution,
                solver=solver,
                puzzle=puzzle,
                result=result,
                competition=competition,
                date=date,
                tps=tps_val,
                is_valid=is_valid,
                validation_status=status,
                invalid_reason=reason,
                raw_reconstruction=raw_recon,
                human_splits=human_splits
            ))

        return records

    def filter_valid(self, records: List[SolveRecord]) -> List[SolveRecord]:
        """Filters a list of solves returning only verified, valid Roux solves."""
        return [r for r in records if r.is_valid]

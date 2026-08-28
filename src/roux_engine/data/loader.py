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
    invalid_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RecoDatasetLoader:
    """Loads and validates solve datasets from reco.nz archives or JSON files."""

    def __init__(self, segmenter: Optional[RouxSegmenter] = None) -> None:
        self.segmenter = segmenter or RouxSegmenter()

    def validate_solve(self, scramble: str, solution: str) -> Tuple[bool, Optional[str]]:
        """Validates that a solve physically solves the cube and segments into Roux phases."""
        if not scramble or not solution:
            return False, "Empty scramble or solution"

        # 1. Parseability test
        try:
            scramble_events = MoveParser.parse_string(scramble)
            solution_events = MoveParser.parse_string(solution)
        except Exception as e:
            return False, f"Notation parsing error: {e}"

        if not scramble_events:
            return False, "No valid moves in scramble"
        if not solution_events:
            return False, "No valid moves in solution"

        # 2. Solvability test (Scramble + Solution == Identity)
        sim = CubeState()
        try:
            for ev in scramble_events:
                sim.apply_move(ev.move)
            for ev in solution_events:
                sim.apply_move(ev.move)
        except Exception as e:
            return False, f"Move application error: {e}"

        if not sim.is_solved():
            return False, "Solution does not restore cube to solved state"

        # 3. Roux Phase Progression test
        try:
            seg = self.segmenter.segment_events(scramble, solution_events)
            if not seg.is_valid:
                reason = seg.warnings[0] if seg.warnings else "Roux segmentation failed"
                return False, reason
        except Exception as e:
            return False, f"Segmentation exception: {e}"

        return True, None

    def load_from_json(self, file_path: str) -> List[SolveRecord]:
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

            # Validate if scramble and solution are present
            if scramble and solution:
                is_valid, reason = self.validate_solve(scramble, solution)
            else:
                is_valid = False
                reason = "Missing scramble or solution"

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
                invalid_reason=reason
            ))

        return records

    def filter_valid(self, records: List[SolveRecord]) -> List[SolveRecord]:
        """Filters a list of solves returning only verified, valid Roux solves."""
        return [r for r in records if r.is_valid]

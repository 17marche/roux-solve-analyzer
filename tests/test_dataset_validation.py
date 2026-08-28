"""Tests for the RecoDatasetLoader and dataset validation pipeline."""

import pytest
import json
import tempfile
import os
from roux_engine.core.parser import MoveParser
from roux_engine.data.loader import RecoDatasetLoader, SolveRecord


def test_dataset_validator_clean_solve():
    """Verify that a valid Roux solve passes the validation pipeline."""
    solution = "U' F' L2 B' L' U' R U' R2 U R' U2 R U' R' R U R' U' R' F R F' M' U M' U' M2 U M' U2 M'"
    scramble = " ".join(MoveParser.invert_moves(solution))

    loader = RecoDatasetLoader()
    is_valid, reason = loader.validate_solve(scramble, solution)

    assert is_valid
    assert reason is None


def test_dataset_validator_unsolvable_solve():
    """Verify that a mismatched scramble/solution is correctly flagged and rejected."""
    scramble = "R U R' U'"
    solution = "R U R' U'"  # Does not solve the cube!

    loader = RecoDatasetLoader()
    is_valid, reason = loader.validate_solve(scramble, solution)

    assert not is_valid
    assert reason == "Solution does not restore cube to solved state"


def test_dataset_validator_non_roux_solve():
    """Verify that a solve that does not complete Roux stages is flagged."""
    scramble = "D' F' L2 D B r U R' U R"
    solution = "F B D L"

    loader = RecoDatasetLoader()
    is_valid, reason = loader.validate_solve(scramble, solution)

    assert not is_valid
    assert reason is not None


def test_dataset_load_and_filter_json():
    """Verify loading and filtering solves from a JSON file."""
    sol1 = "U' F' L2 B' L' U' R U' R2 U R' U2 R U' R' R U R' U' R' F R F' M' U M' U' M2 U M' U2 M'"
    scram1 = " ".join(MoveParser.invert_moves(sol1))

    sample_data = [
        {
            "id": 1,
            "solver": "Sean Patrick Villanueva",
            "scramble": scram1,
            "solution": sol1,
            "tps": "6.5"
        },
        {
            "id": 2,
            "solver": "Bad Solver",
            "scramble": "R U R' U'",
            "solution": "R U R' U'",  # Invalid
            "tps": "4.0"
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_data, f)
        temp_path = f.name

    try:
        loader = RecoDatasetLoader()
        records = loader.load_from_json(temp_path)
        assert len(records) == 2
        assert records[0].is_valid is True
        assert records[1].is_valid is False

        valid_records = loader.filter_valid(records)
        assert len(valid_records) == 1
        assert valid_records[0].id == 1
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

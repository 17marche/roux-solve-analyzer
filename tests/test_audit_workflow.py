"""Tests for the Ergonomic Audit Workflow and Persistence."""

import json
from pathlib import Path
import pytest

from roux_engine.inspector.phase_inspector import inspect_phase
from roux_engine.inspector.audit import (
    AuditRecord,
    record_audit_entry,
    DEFAULT_AUDIT_MD_PATH,
    DEFAULT_AUDIT_JSONL_PATH,
)


@pytest.fixture
def sample_scramble() -> str:
    return "D2 F2 R2 B2 U L2 U2 F2 D R2 B2 R B U2 L B2 D2 F R2 B2"


class TestAuditPersistence:
    """Slice 1: Audit Record Serialization and File Persistence."""

    def test_record_audit_agree_minimal_log(self, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        record = AuditRecord(
            timestamp="2026-10-09T10:00:00Z",
            verdict="agree",
            phase="fb",
            scramble="R U R' U'",
            profile="2H",
            m_slice_hand="right",
            top_candidate={
                "moves": ["U", "R", "U'", "R'"],
                "moves_str": "U R U' R'",
                "stm": 4,
                "e_stm": 4.12,
                "flow_efficiency": 94.5,
            },
            domain_notes="Looks solid",
        )

        record_audit_entry(record, md_path=md_file, jsonl_path=jsonl_file)

        # Verify Markdown file contains a single compact confirmation line
        assert md_file.exists()
        md_content = md_file.read_text().strip().splitlines()
        # Header + 1 line
        assert any(l.startswith("- [AGREE]") for l in md_content)
        agree_line = [l for l in md_content if l.startswith("- [AGREE]")][0]
        assert "Phase: FB" in agree_line
        assert "Top pick: U R U' R'" in agree_line
        assert "4 STM" in agree_line
        assert "Looks solid" in agree_line
        # Confirm no verbose failure block
        assert "### Solver Candidate Rankings" not in md_file.read_text()

        # Verify JSONL file contains single JSON line
        assert jsonl_file.exists()
        jsonl_lines = jsonl_file.read_text().strip().splitlines()
        assert len(jsonl_lines) == 1
        data = json.loads(jsonl_lines[0])
        assert data["verdict"] == "agree"
        assert data["phase"] == "fb"
        assert data["top_candidate"]["stm"] == 4
        assert data["domain_notes"] == "Looks solid"

    def test_record_audit_disagree_candidate_inversion(self, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        record = AuditRecord(
            timestamp="2026-10-09T10:05:00Z",
            verdict="disagree",
            phase="sb",
            scramble="D2 F2 R2 B2 U",
            profile="2H",
            m_slice_hand="right",
            failure_category="candidate_inversion",
            human_preference="Candidate #2 (r U' r')",
            flagged_transitions="R U2 -> M'",
            domain_notes="Candidate #2 flows much better in physical 2H execution despite slightly higher model cost.",
            setup_moves={"fb": ["r", "U", "R'"]},
            candidates=[
                {
                    "rank": 1,
                    "moves_str": "R U R' U2 R' U'",
                    "stm": 6,
                    "e_stm": 6.1,
                    "flow_efficiency": 92.0,
                    "grip": {"summary": "Start in HOME -> End in HOME (0 regrips)", "regrips": []},
                },
                {
                    "rank": 2,
                    "moves_str": "r U' r'",
                    "stm": 3,
                    "e_stm": 6.3,
                    "flow_efficiency": 89.0,
                    "grip": {"summary": "Start in HOME -> End in HOME (0 regrips)", "regrips": []},
                },
            ],
        )

        record_audit_entry(record, md_path=md_file, jsonl_path=jsonl_file)

        # Verify Markdown file contains detailed diagnostic case block
        assert md_file.exists()
        md_text = md_file.read_text()
        assert "## [DISAGREE: Candidate Inversion]" in md_text
        assert "- **Scramble**: `D2 F2 R2 B2 U`" in md_text
        assert "- **Phase**: Second Block (SB)" in md_text
        assert "- **Preceding Setup Moves**: First Block (FB): r U R'" in md_text
        assert "- **Failure Category**: Candidate Inversion" in md_text
        assert "- **Human Preference**: Candidate #2 (r U' r')" in md_text
        assert "- **Flagged Transitions**: R U2 -> M'" in md_text
        assert "Candidate #2 flows much better" in md_text
        assert "### Solver Candidate Rankings:" in md_text
        assert "1. `R U R' U2 R' U'` | STM: 6 | E-STM: 6.1000" in md_text
        assert "2. `r U' r'` | STM: 3 | E-STM: 6.3000" in md_text

        # Verify JSONL contains complete structured record
        assert jsonl_file.exists()
        jsonl_lines = jsonl_file.read_text().strip().splitlines()
        assert len(jsonl_lines) == 1
        data = json.loads(jsonl_lines[0])
        assert data["verdict"] == "disagree"
        assert data["failure_category"] == "candidate_inversion"
        assert data["human_preference"] == "Candidate #2 (r U' r')"
        assert data["flagged_transitions"] == "R U2 -> M'"
        assert len(data["candidates"]) == 2
        assert data["setup_moves"]["fb"] == ["r", "U", "R'"]


class TestInteractiveAuditWorkflow:
    """Slice 2: Interactive Prompting Engine (perform_audit)."""

    def test_perform_audit_agree_with_notes(self, sample_scramble: str, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        result = inspect_phase(sample_scramble, phase="fb", top_k=2)

        # Simulated user inputs: "1" (Agree), then "Felt very fast"
        inputs = iter(["1", "Felt very fast"])

        from roux_engine.inspector.audit import perform_audit

        record = perform_audit(
            result=result,
            input_func=lambda prompt: next(inputs),
            md_path=md_file,
            jsonl_path=jsonl_file,
            timestamp="2026-10-09T10:10:00Z",
        )

        assert record.verdict == "agree"
        assert record.domain_notes == "Felt very fast"
        assert record.top_candidate is not None
        assert md_file.exists()
        assert jsonl_file.exists()

        md_text = md_file.read_text()
        assert "- [AGREE]" in md_text
        assert "Felt very fast" in md_text
        assert "### Solver Candidate Rankings" not in md_text

    def test_perform_audit_disagree_candidate_inversion_rank(self, sample_scramble: str, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        result = inspect_phase(sample_scramble, phase="fb", top_k=3)

        # Inputs: "2" (Candidate Inversion), "2" (Candidate #2), "U2 -> M'", "Candidate #2 has better ergonomics"
        inputs = iter(["2", "2", "U2 -> M'", "Candidate #2 has better ergonomics"])

        from roux_engine.inspector.audit import perform_audit

        record = perform_audit(
            result=result,
            input_func=lambda prompt: next(inputs),
            md_path=md_file,
            jsonl_path=jsonl_file,
            timestamp="2026-10-09T10:12:00Z",
        )

        assert record.verdict == "disagree"
        assert record.failure_category == "candidate_inversion"
        assert "Candidate #2" in str(record.human_preference)
        assert record.flagged_transitions == "U2 -> M'"
        assert record.domain_notes == "Candidate #2 has better ergonomics"
        assert len(record.candidates) == 3

        md_text = md_file.read_text()
        assert "## [DISAGREE: Candidate Inversion]" in md_text
        assert "Candidate #2" in md_text
        assert "### Solver Candidate Rankings:" in md_text

        jsonl_lines = jsonl_file.read_text().strip().splitlines()
        assert len(jsonl_lines) == 1
        data = json.loads(jsonl_lines[0])
        assert data["failure_category"] == "candidate_inversion"
        assert len(data["candidates"]) == 3

    def test_perform_audit_disagree_regrip_error(self, sample_scramble: str, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        result = inspect_phase(sample_scramble, phase="fb", top_k=2)

        # Inputs: "3" (Regrip Error), "move 2", "1" (false-positive), "", "False regrip detected"
        inputs = iter(["3", "move 2", "1", "", "False regrip detected"])

        from roux_engine.inspector.audit import perform_audit

        record = perform_audit(
            result=result,
            input_func=lambda prompt: next(inputs),
            md_path=md_file,
            jsonl_path=jsonl_file,
            timestamp="2026-10-09T10:14:00Z",
        )

        assert record.verdict == "disagree"
        assert record.failure_category == "regrip_error"
        assert "move 2" in str(record.flagged_transitions)
        assert "false-positive" in str(record.flagged_transitions)
        assert record.domain_notes == "False regrip detected"

    def test_perform_audit_disagree_transition_matrix_error(self, sample_scramble: str, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        result = inspect_phase(sample_scramble, phase="fb", top_k=2)

        # Inputs: "4" (Transition Matrix Error), "U2 M'", "r U R'", "Penalty is distorted"
        inputs = iter(["4", "U2 M'", "r U R'", "Penalty is distorted"])

        from roux_engine.inspector.audit import perform_audit

        record = perform_audit(
            result=result,
            input_func=lambda prompt: next(inputs),
            md_path=md_file,
            jsonl_path=jsonl_file,
            timestamp="2026-10-09T10:16:00Z",
        )

        assert record.verdict == "disagree"
        assert record.failure_category == "transition_matrix_error"
        assert record.flagged_transitions == "U2 M'"
        assert record.human_preference == "r U R'"
        assert record.domain_notes == "Penalty is distorted"

    def test_perform_audit_eof_raises_aborted_error(self, sample_scramble: str, tmp_path: Path):
        md_file = tmp_path / "ergonomic_audit.md"
        jsonl_file = tmp_path / "ergonomic_audit.jsonl"

        result = inspect_phase(sample_scramble, phase="fb", top_k=2)

        def eof_input(prompt: str) -> str:
            raise EOFError()

        from roux_engine.inspector.audit import perform_audit, AuditAbortedError

        with pytest.raises(AuditAbortedError):
            perform_audit(
                result=result,
                input_func=eof_input,
                md_path=md_file,
                jsonl_path=jsonl_file,
            )


"""Human-in-the-loop Ergonomic Audit Workflow and File Persistence.

Provides interactive evaluation prompting and persistent logging to Markdown
(`.scratch/ergonomic_audit.md`) and JSONL (`.scratch/ergonomic_audit.jsonl`)
for human domain expert evaluations on physical cubes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import datetime
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .phase_inspector import PhaseInspectionResult, CandidateInspection

DEFAULT_AUDIT_MD_PATH = Path(".scratch/ergonomic_audit.md")
DEFAULT_AUDIT_JSONL_PATH = Path(".scratch/ergonomic_audit.jsonl")

PHASE_LABELS: Dict[str, str] = {
    "fb": "First Block (FB)",
    "sb": "Second Block (SB)",
    "lse": "Last Six Edges (LSE)",
    "cmll": "CMLL",
}

FAILURE_CATEGORY_LABELS: Dict[str, str] = {
    "candidate_inversion": "Candidate Inversion",
    "regrip_error": "Regrip Error",
    "transition_matrix_error": "Transition Matrix Error",
}


@dataclass
class AuditRecord:
    """Structured record of an on-cube ergonomic physical audit."""
    timestamp: str
    verdict: str  # "agree" or "disagree"
    phase: str
    scramble: str
    profile: str = "2H"
    m_slice_hand: str = "right"
    top_candidate: Optional[Dict[str, Any]] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    setup_moves: Dict[str, List[str]] = field(default_factory=dict)
    failure_category: Optional[str] = None
    human_preference: Optional[str] = None
    flagged_transitions: Optional[str] = None
    domain_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the audit record to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "verdict": self.verdict,
            "phase": self.phase,
            "scramble": self.scramble,
            "profile": self.profile,
            "m_slice_hand": self.m_slice_hand,
        }
        if self.verdict == "agree":
            if self.top_candidate is not None:
                d["top_candidate"] = self.top_candidate
            if self.domain_notes:
                d["domain_notes"] = self.domain_notes
            return d

        # Disagreement / Flaw payload
        d["failure_category"] = self.failure_category
        d["human_preference"] = self.human_preference
        d["flagged_transitions"] = self.flagged_transitions
        d["domain_notes"] = self.domain_notes
        d["setup_moves"] = self.setup_moves
        d["candidates"] = self.candidates
        return d

    def format_markdown(self) -> str:
        """Formats the audit record for markdown appending."""
        phase_label = PHASE_LABELS.get(self.phase.lower(), self.phase.upper())

        if self.verdict == "agree":
            notes_str = f" | Notes: {self.domain_notes}" if self.domain_notes else ""
            if self.top_candidate:
                moves_str = self.top_candidate.get("moves_str") or " ".join(self.top_candidate.get("moves", []))
                stm = self.top_candidate.get("stm", 0)
                e_stm = self.top_candidate.get("e_stm", 0.0)
                top_str = f"Top pick: {moves_str} ({stm} STM, {e_stm:.2f} E-STM)"
            else:
                top_str = "Top pick: Confirmed"
            return (
                f"- [AGREE] [{self.timestamp}] Phase: {self.phase.upper()} | "
                f"Scramble: {self.scramble} | {top_str}{notes_str}\n"
            )

        # Disagree case block
        cat_label = FAILURE_CATEGORY_LABELS.get(self.failure_category or "", self.failure_category or "Flaw")
        m_hand_desc = f"{self.m_slice_hand}-handed M-slice"

        setup_parts: List[str] = []
        if self.setup_moves:
            for k, toks in self.setup_moves.items():
                k_name = PHASE_LABELS.get(k, k.capitalize())
                setup_parts.append(f"{k_name}: {' '.join(toks)}")
        setup_str = ", ".join(setup_parts) if setup_parts else "None"

        lines: List[str] = []
        lines.append(f"## [DISAGREE: {cat_label}] {self.timestamp} - Phase: {phase_label}")
        lines.append("")
        lines.append(f"- **Scramble**: `{self.scramble}`")
        lines.append(f"- **Phase**: {phase_label}")
        lines.append(f"- **Profile**: `{self.profile}` ({m_hand_desc})")
        lines.append(f"- **Preceding Setup Moves**: {setup_str}")
        lines.append(f"- **Failure Category**: {cat_label}")
        lines.append(f"- **Flagged Transitions**: {self.flagged_transitions or 'None'}")
        lines.append(f"- **Human Preference**: {self.human_preference or 'None'}")
        lines.append("- **Domain Notes**:")
        lines.append(f"  > {self.domain_notes or 'None'}")
        lines.append("")

        if self.candidates:
            lines.append("### Solver Candidate Rankings:")
            for idx, c in enumerate(self.candidates, 1):
                m_str = c.get("moves_str") or " ".join(c.get("moves", []))
                stm = c.get("stm", 0)
                e_stm = c.get("e_stm", 0.0)
                eff = c.get("flow_efficiency", 0.0)
                grip = c.get("grip", {})
                grip_summary = format_grip_summary_from_data(grip)
                lines.append(f"{idx}. `{m_str}` | STM: {stm} | E-STM: {e_stm:.4f} | Flow Eff: {eff:.1f}% | Grip: {grip_summary}")
                regrips = grip.get("regrips", []) if isinstance(grip, dict) else []
                for r in regrips:
                    if isinstance(r, dict):
                        from_g = r.get("from_grip", "")
                        to_g = r.get("to_grip", "")
                        p_m = r.get("prev_move")
                        c_m = r.get("curr_move", "")
                        m_idx = r.get("move_index", 0)
                        if p_m:
                            lines.append(f"   - Regrip between move {m_idx} ({p_m}) and move {m_idx + 1} ({c_m}): from {from_g} to {to_g}")
                        else:
                            lines.append(f"   - Regrip before move 1 ({c_m}): from {from_g} to {to_g}")
                    else:
                        lines.append(f"   - {r}")
            lines.append("")

        return "\n".join(lines) + "\n"


def record_audit_entry(
    record: AuditRecord,
    md_path: Union[str, Path] = DEFAULT_AUDIT_MD_PATH,
    jsonl_path: Union[str, Path] = DEFAULT_AUDIT_JSONL_PATH,
) -> None:
    """Appends an audit record to markdown and JSONL persistence files."""
    target_md = Path(md_path)
    target_jsonl = Path(jsonl_path)

    # Ensure parent directories exist
    target_md.parent.mkdir(parents=True, exist_ok=True)
    target_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Markdown header initialization if file does not exist
    if not target_md.exists() or target_md.stat().st_size == 0:
        header = (
            "# Roux Ergonomic Physical Audit Log\n\n"
            "Persistent audit log documenting human domain expert evaluations on physical speedcubes.\n\n"
        )
        target_md.write_text(header, encoding="utf-8")

    # Append to markdown
    with open(target_md, "a", encoding="utf-8") as f:
        f.write(record.format_markdown())

    # Append to JSONL
    with open(target_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")


class AuditAbortedError(Exception):
    """Raised when the human audit workflow is interrupted or cancelled."""
    pass


def format_grip_summary_from_data(grip: Any) -> str:
    """Extracts or formats a human-readable grip summary string."""
    if isinstance(grip, dict):
        if "summary" in grip and grip["summary"]:
            return str(grip["summary"])
        start = grip.get("start", "HOME")
        end = grip.get("end", "HOME")
        regrip_cnt = grip.get("regrip_count", 0)
        regrip_str = f"{regrip_cnt} regrip" if regrip_cnt == 1 else f"{regrip_cnt} regrips"
        return f"Start in {start} -> End in {end} ({regrip_str})"
    return str(grip)


def _resolve_candidate_pref(pref_input: str, candidates: Sequence[Any]) -> Optional[str]:
    """Resolves a human preference input to formatted text, handling rank numbers."""
    text = pref_input.strip()
    if not text:
        return None
    if text.isdigit():
        rank = int(text)
        if 1 <= rank <= len(candidates):
            c = candidates[rank - 1]
            if hasattr(c, "moves_str"):
                m_str = c.moves_str
            elif isinstance(c, dict):
                m_str = c.get("moves_str") or " ".join(c.get("moves", []))
            else:
                m_str = ""
            if m_str:
                return f"Candidate #{rank} ({m_str})"
            return f"Candidate #{rank}"
        return f"Candidate #{rank}"
    return text


def _prompt(prompt: str, input_func: Callable[[str], str]) -> str:
    """Prompts for input, converting interruption or I/O errors into AuditAbortedError."""
    try:
        return input_func(prompt).strip()
    except (EOFError, KeyboardInterrupt, OSError) as e:
        raise AuditAbortedError("Audit aborted: no interactive input provided.") from e


def perform_audit(
    result: PhaseInspectionResult,
    input_func: Optional[Callable[[str], str]] = None,
    print_func: Optional[Callable[..., None]] = None,
    md_path: Union[str, Path] = DEFAULT_AUDIT_MD_PATH,
    jsonl_path: Union[str, Path] = DEFAULT_AUDIT_JSONL_PATH,
    timestamp: Optional[str] = None,
) -> AuditRecord:
    """Executes the interactive human audit workflow and appends evaluation records."""
    if input_func is None:
        input_func = input
    if print_func is None:
        print_func = print

    if timestamp is None:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print_func("\n" + "=" * 80)
    print_func("              ERGONOMIC HUMAN AUDIT EVALUATION (--audit)")
    print_func("=" * 80)
    print_func("Evaluate solver recommendations on physical cube:")
    print_func("  [1] Agree (Top candidate is physically best)")
    print_func("  [2] Disagree - Candidate Inversion (Another candidate or custom line is superior)")
    print_func("  [3] Disagree - Regrip Error (False-positive or missed regrip)")
    print_func("  [4] Disagree - Transition Matrix Error (Specific move pair over/under-penalized)")
    print_func("-" * 80)

    raw_choice = _prompt("Verdict [1-4 or Agree/Disagree]: ", input_func).lower()
    if not raw_choice:
        raise AuditAbortedError("No evaluation input provided.")

    verdict = "agree"
    failure_category: Optional[str] = None

    if raw_choice in ("1", "agree", "a", "yes", "y"):
        verdict = "agree"
    elif raw_choice in ("2", "inversion", "candidate inversion"):
        verdict = "disagree"
        failure_category = "candidate_inversion"
    elif raw_choice in ("3", "regrip", "regrip error"):
        verdict = "disagree"
        failure_category = "regrip_error"
    elif raw_choice in ("4", "transition", "transition matrix", "transition matrix error", "matrix"):
        verdict = "disagree"
        failure_category = "transition_matrix_error"
    elif raw_choice in ("disagree", "d", "no", "n"):
        verdict = "disagree"
        print_func("Select failure category:")
        print_func("  [1] Candidate Inversion (Preferred alternative)")
        print_func("  [2] Regrip Error (False-positive or missed regrip)")
        print_func("  [3] Transition Matrix Error (Distorted transition cost)")
        cat_choice = _prompt("Category [1-3]: ", input_func).lower()

        if cat_choice in ("1", "inversion"):
            failure_category = "candidate_inversion"
        elif cat_choice in ("2", "regrip"):
            failure_category = "regrip_error"
        elif cat_choice in ("3", "transition", "matrix"):
            failure_category = "transition_matrix_error"
        else:
            raise AuditAbortedError(f"Invalid category selection: '{cat_choice}'.")
    else:
        raise AuditAbortedError(f"Invalid evaluation choice: '{raw_choice}'.")

    if verdict == "agree":
        notes_raw = _prompt("Domain notes / physical feel (optional, press Enter to skip): ", input_func)
        top_cand_dict = result.candidates[0].to_dict() if result.candidates else None
        record = AuditRecord(
            timestamp=timestamp,
            verdict="agree",
            phase=result.phase,
            scramble=result.scramble,
            profile=result.profile,
            m_slice_hand=result.m_slice_hand,
            top_candidate=top_cand_dict,
            domain_notes=notes_raw if notes_raw else None,
        )
        record_audit_entry(record, md_path=md_path, jsonl_path=jsonl_path)
        print_func(f"\nAudit logged: Agree (saved to {md_path} and {jsonl_path})")
        return record

    # Disagree workflow
    human_preference: Optional[str] = None
    flagged_transitions: Optional[str] = None
    domain_notes: Optional[str] = None

    if failure_category == "candidate_inversion":
        pref_raw = _prompt("Preferred candidate (enter rank # or custom moves): ", input_func)
        human_preference = _resolve_candidate_pref(pref_raw, result.candidates)
        flagged_raw = _prompt("Flagged transitions / bigrams (optional, press Enter to skip): ", input_func)
        flagged_transitions = flagged_raw if flagged_raw else None
        notes_raw = _prompt("Domain notes / rationale: ", input_func)
        domain_notes = notes_raw if notes_raw else None

    elif failure_category == "regrip_error":
        flagged_move = _prompt("Flagged transition or move index (e.g. 'move 2' or 'R -> R'): ", input_func)
        subtype_in = _prompt("Regrip error type ([1] false-positive, [2] missed): ", input_func).lower()
        subtype = "missed" if ("2" in subtype_in or "miss" in subtype_in) else "false-positive"
        flagged_transitions = f"{flagged_move} ({subtype})" if flagged_move else subtype
        pref_raw = _prompt("Preferred candidate or custom moves (optional): ", input_func)
        human_preference = _resolve_candidate_pref(pref_raw, result.candidates)
        notes_raw = _prompt("Domain notes / physical explanation: ", input_func)
        domain_notes = notes_raw if notes_raw else None

    elif failure_category == "transition_matrix_error":
        flagged_raw = _prompt("Flagged transition pair / bigram (e.g. 'U2 -> M''): ", input_func)
        flagged_transitions = flagged_raw if flagged_raw else None
        pref_raw = _prompt("Preferred candidate or custom moves (optional): ", input_func)
        human_preference = _resolve_candidate_pref(pref_raw, result.candidates)
        notes_raw = _prompt("Domain notes / transition cost feedback: ", input_func)
        domain_notes = notes_raw if notes_raw else None

    candidates_serialized = [c.to_dict() for c in result.candidates]
    top_cand_dict = candidates_serialized[0] if candidates_serialized else None

    record = AuditRecord(
        timestamp=timestamp,
        verdict="disagree",
        phase=result.phase,
        scramble=result.scramble,
        profile=result.profile,
        m_slice_hand=result.m_slice_hand,
        top_candidate=top_cand_dict,
        candidates=candidates_serialized,
        setup_moves=result.setup_moves,
        failure_category=failure_category,
        human_preference=human_preference,
        flagged_transitions=flagged_transitions,
        domain_notes=domain_notes,
    )
    record_audit_entry(record, md_path=md_path, jsonl_path=jsonl_path)
    cat_label = FAILURE_CATEGORY_LABELS.get(failure_category or "", failure_category or "Flaw")
    print_func(f"\nAudit logged: Disagree ({cat_label}) (saved to {md_path} and {jsonl_path})")
    return record

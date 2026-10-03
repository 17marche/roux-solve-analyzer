#!/usr/bin/env python3
"""Calibrate the Two-Handed (2H) Bigram Transition Matrix.

Reads the uncleaned baseline dataset from `src/roux_engine/data/transitions/matrix_2h_raw.json`
and systematically produces the calibrated production matrix at
`src/roux_engine/data/transitions/matrix_2h.json` according to
`docs/TRANSITION_MATRIX_CALIBRATION_SPEC.md`.

Multi-Pass Pipeline:
  Pass 1: Anchor Extraction - Retain clean empirical baseline transitions.
  Pass 2: Rule 2 - Bilateral Mirroring for Left-Hand (L) turns.
  Pass 3: Rule 1 - Rotational Drag Derivation for Wide (r) turns.
  Pass 4: Rules 4, 5 & 6 - M-Slice, Decoupled Opposing Faces, and Biomechanical Fallback.
  Pass 5: Rule 3 - Double-Move Monotonicity Enforcement.

Usage:
  uv run python scripts/calibrate_matrix_2h.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

from roux_engine.core.parser import MoveParser

# File locations
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_MATRIX_PATH = REPO_ROOT / "src" / "roux_engine" / "data" / "transitions" / "matrix_2h_raw.json"
CALIBRATED_MATRIX_PATH = REPO_ROOT / "src" / "roux_engine" / "data" / "transitions" / "matrix_2h.json"

# Biomechanical scaling factors
WIDE_DRAG_FACTOR = 1.12
NON_DOMINANT_DRAG_FACTOR = 1.15
DOUBLE_MOVE_FACTOR = 1.15

# Bilateral mirror mapping across the sagittal plane (Rule 2)
BILATERAL_MIRROR = {
    "R": "L'", "R'": "L", "R2": "L2",
    "L": "R'", "L'": "R", "L2": "R2",
    "U": "U'", "U'": "U", "U2": "U2",
    "D": "D'", "D'": "D", "D2": "D2",
    "F": "F'", "F'": "F", "F2": "F2",
    "B": "B'", "B'": "B", "B2": "B2",
    "M": "M", "M'": "M'", "M2": "M2",
}

# Wide to outer mapping (Rule 1)
WIDE_TO_OUTER = {
    "r": "R", "r'": "R'", "r2": "R2",
}

FACES = ["U", "D", "F", "B", "L", "R", "M", "r"]
ALL_MOVES = [f for face in FACES for f in [face, f"{face}'", f"{face}2"]]


import re

TOKEN_PATTERN = re.compile(r"^([UDFBLRMr]['2]?)(?:([UDFBLRMr]['2]?))?$")


def parse_token(key: str) -> Tuple[Optional[str], str]:
    """Splits a key into (prev_move, curr_move)."""
    m = TOKEN_PATTERN.match(key)
    if not m:
        raise ValueError(f"Cannot parse token: {key}")
    if m.group(2) is None:
        return None, m.group(1)
    return m.group(1), m.group(2)


def is_default_ceiling(val: float) -> bool:
    """Detects whether a value falls in the synthetic default ceiling (>= 2.88)."""
    return val >= 2.88


def is_floor_clamp(val: float) -> bool:
    """Detects whether a value is jammed at the ~0.50 ms sensor clamp floor."""
    return 0.48 <= val <= 0.52


def compute_biomechanical_fallback(
    prev_move: str,
    curr_move: str,
    single_efforts: Dict[str, float],
) -> float:
    """Computes Rule 6 fallback for unmeasured or corrupted bigrams."""
    prev_single = single_efforts.get(prev_move, 1.0)
    curr_single = single_efforts.get(curr_move, 1.0)
    avg_effort = (prev_single + curr_single) / 2.0

    prev_face = prev_move[0].upper()
    curr_face = curr_move[0].upper()

    # Determine anatomical awkwardness modifier
    # Decoupled independent hands (e.g. L with R, or U with D)
    if (prev_face in "LR" and curr_face in "LR") or (prev_face in "UD" and curr_face in "UD"):
        modifier = 0.95
    elif prev_face in "FB" and curr_face in "FB":
        # Opposing front/back face reach conflict
        modifier = 1.30
    elif (prev_face == "R" and curr_face in "FB") or (prev_face in "FB" and curr_face == "R"):
        # Right wrist rotation contention with front/back flick
        modifier = 1.35
    elif prev_face in "FB" or curr_face in "FB":
        modifier = 1.20
    elif prev_face == curr_face:
        modifier = 1.10
    else:
        modifier = 1.00

    effort = round(avg_effort * modifier, 3)
    # Rule 4: Two-handed M-slice transitions should not exceed 2.0 unless same-finger collision
    if (prev_face == "M" or curr_face == "M") and effort > 2.0:
        effort = 1.85
    # Ensure it never accidentally lands in the synthetic default range (>= 2.88)
    if effort >= 2.88:
        effort = 2.75
    return effort


def calibrate() -> Dict[str, float]:
    """Executes the 5-pass calibration pipeline."""
    with open(RAW_MATRIX_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_transitions: Dict[str, float] = raw_data["transitions"]
    calibrated: Dict[str, float] = dict(raw_transitions)

    # -------------------------------------------------------------------------
    # PASS 1: Single Moves Calibration & Anchor Extraction
    # -------------------------------------------------------------------------
    # Rule 4: Recalibrate single M
    calibrated["M"] = 1.45
    # M' (0.498) and M2 (0.826) are retained from raw
    # Rule 1: Calibrate wide single moves from R
    calibrated["r"] = round(calibrated["R"] * WIDE_DRAG_FACTOR, 3)     # 0.889
    calibrated["r'"] = round(calibrated["R'"] * WIDE_DRAG_FACTOR, 3)   # 0.660
    calibrated["r2"] = round(calibrated["R2"] * WIDE_DRAG_FACTOR, 3)   # 0.559
    # Rule 2: Calibrate left single moves from R
    calibrated["L'"] = round(calibrated["R"] * NON_DOMINANT_DRAG_FACTOR, 3)   # 0.913
    calibrated["L2"] = round(calibrated["R2"] * NON_DOMINANT_DRAG_FACTOR, 3)  # 0.574
    # Single L (0.643) is retained clean

    single_efforts = {m: calibrated[m] for m in ALL_MOVES if m in calibrated}

    # -------------------------------------------------------------------------
    # PASS 2: Bilateral Mirroring for Left-Hand (L) Turns (Rule 2)
    # -------------------------------------------------------------------------
    for key in list(raw_transitions.keys()):
        prev, curr = parse_token(key)
        if prev is None:
            continue  # Single moves already handled
        if "L" in prev or "L" in curr:
            # Find mirror counterpart
            if prev in BILATERAL_MIRROR and curr in BILATERAL_MIRROR:
                m_prev = BILATERAL_MIRROR[prev]
                m_curr = BILATERAL_MIRROR[curr]
                m_key = f"{m_prev}{m_curr}"
                if m_key in raw_transitions:
                    m_val = raw_transitions[m_key]
                    if not is_default_ceiling(m_val):
                        calibrated[key] = round(m_val * NON_DOMINANT_DRAG_FACTOR, 3)

    # -------------------------------------------------------------------------
    # PASS 3: Wide-Turn Derivation (Rule 1)
    # -------------------------------------------------------------------------
    for key in list(raw_transitions.keys()):
        prev, curr = parse_token(key)
        if prev is None:
            continue
        if "r" in prev or "r" in curr:
            r_prev = WIDE_TO_OUTER.get(prev, prev)
            r_curr = WIDE_TO_OUTER.get(curr, curr)
            r_key = f"{r_prev}{r_curr}"
            if r_key in calibrated:
                base_val = calibrated[r_key]
                # If base value was in default ceiling, use fallback for base first
                if is_default_ceiling(base_val):
                    base_val = compute_biomechanical_fallback(r_prev, r_curr, single_efforts)
                    calibrated[r_key] = base_val
                calibrated[key] = round(base_val * WIDE_DRAG_FACTOR, 3)

    # -------------------------------------------------------------------------
    # PASS 4: Rules 4, 5 & 6 - M-Slice, Decoupled Opposing Faces & Fallbacks
    # -------------------------------------------------------------------------
    # Rule 4: M-slice decoupled pairs with U
    calibrated["MU"] = 1.45
    calibrated["UM"] = 1.55
    calibrated["MU'"] = 1.331
    calibrated["U'M"] = 0.805

    # Rule 5: Decoupled Opposing Faces (D <-> U)
    calibrated["D'U"] = 0.751
    calibrated["DU'"] = 0.478
    calibrated["D2U"] = 1.15
    calibrated["U2D"] = 1.15
    calibrated["DU"] = 1.35
    calibrated["UD"] = 1.35
    calibrated["D'U'"] = 1.393
    calibrated["U'D'"] = 1.393
    calibrated["D2U'"] = 1.223
    calibrated["DU2"] = 1.538
    calibrated["D'U2"] = 1.40
    calibrated["D2U2"] = 1.65

    # Specific awkward reach pairs that must not be in default ceiling
    # RF is a physical wrist reach (~2.65 effort)
    calibrated["RF"] = 2.65

    # Recalibrate ANY remaining entry in synthetic default ceiling [2.88, 3.12]
    # or physically impossible floor clamps (e.g. UB2, F2B, F2R')
    for key, val in list(calibrated.items()):
        prev, curr = parse_token(key)
        if prev is None:
            continue
        if is_default_ceiling(val):
            calibrated[key] = compute_biomechanical_fallback(prev, curr, single_efforts)
        elif is_floor_clamp(val):
            # Clamp floor checks for double moves or reach conflicts
            if curr.endswith("2") or (prev[0] in "FB" and curr[0] in "FB") or (prev[0] == "F" and curr[0] == "R"):
                calibrated[key] = compute_biomechanical_fallback(prev, curr, single_efforts)
        elif ("M" in prev or "M" in curr) and val > 2.0:
            # Rule 4: Decoupled two-handed M-slice transitions must not exceed 2.0
            calibrated[key] = compute_biomechanical_fallback(prev, curr, single_efforts)

    # -------------------------------------------------------------------------
    # PASS 5: Rule 3 - Double-Move Monotonicity Enforcement
    # -------------------------------------------------------------------------
    # Single moves monotonicity
    for face in FACES:
        x = face
        xp = f"{face}'"
        x2 = f"{face}2"
        if x in calibrated and xp in calibrated and x2 in calibrated:
            min_single = min(calibrated[x], calibrated[xp])
            calibrated[x2] = max(calibrated[x2], round(min_single * DOUBLE_MOVE_FACTOR, 3))

    # Bigrams monotonicity
    for face in FACES:
        x = face
        xp = f"{face}'"
        x2 = f"{face}2"
        for a in ALL_MOVES:
            if a.startswith(face):
                continue
            pair_x = f"{a}{x}"
            pair_xp = f"{a}{xp}"
            pair_x2 = f"{a}{x2}"
            if pair_x in calibrated and pair_xp in calibrated and pair_x2 in calibrated:
                min_flick = min(calibrated[pair_x], calibrated[pair_xp])
                calibrated[pair_x2] = max(
                    calibrated[pair_x2],
                    round(min_flick * DOUBLE_MOVE_FACTOR, 3),
                )

    return calibrated


def main() -> None:
    print(f"Reading raw baseline matrix from {RAW_MATRIX_PATH}...")
    calibrated_transitions = calibrate()

    with open(RAW_MATRIX_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Update metadata
    metadata = dict(raw_data.get("_metadata", {}))
    metadata.update({
        "source": "onionhoney/roux-trainers",
        "attribution": (
            "Ported from onionhoney/roux-trainers (two_gram_v1.json) under MIT license. "
            "Calibrated with physically grounded biomechanical model (TRANSITION_MATRIX_CALIBRATION_SPEC.md)."
        ),
        "calibration_spec": "docs/TRANSITION_MATRIX_CALIBRATION_SPEC.md",
        "calibration_version": "2.0.0",
        "pair_count": len(calibrated_transitions),
    })

    output_data = {
        "_metadata": metadata,
        "transitions": calibrated_transitions,
    }

    print(f"Writing calibrated matrix ({len(calibrated_transitions)} entries) to {CALIBRATED_MATRIX_PATH}...")
    with open(CALIBRATED_MATRIX_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("Calibration complete successfully.")


if __name__ == "__main__":
    main()

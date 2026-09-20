"""Closed-Loop Macro Triggers registry, invariant verification, and chunking pattern matcher."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Union

from ..core.cube import CubeState
from ..core.orientation import RouxOrientation, get_orientation, get_dual_neutral_orientations
from ..core.parser import MoveParser, MoveEvent


@dataclass(frozen=True)
class MacroTrigger:
    """Closed-loop macro trigger compound move sequence.
    
    Temporarily disturbs First Block pieces during intermediate turns but leaves
    First Block 100% intact at completion.
    """
    name: str
    moves: Tuple[str, ...]
    stm: int = 4


SLEDGEHAMMER = MacroTrigger(
    name="sledgehammer",
    moves=("R'", "F", "R", "F'"),
    stm=4,
)

HEDGE = MacroTrigger(
    name="hedge",
    moves=("F", "R'", "F'", "R"),
    stm=4,
)

MACRO_TRIGGERS: Dict[str, MacroTrigger] = {
    "sledgehammer": SLEDGEHAMMER,
    "hedge": HEDGE,
}


def verify_first_block_preservation(
    trigger: MacroTrigger,
    orientation: Optional[Union[str, RouxOrientation]] = None,
) -> bool:
    """Automated verification of the First Block preservation invariant for a macro trigger.
    
    Verifies that executing the trigger in the solve frame leaves First Block
    (DL, FL, BL edges; DFL, DBL corners; Left center) 100% intact.
    """
    orientations: Sequence[RouxOrientation]
    if orientation is not None:
        orientations = [get_orientation(orientation)]
    else:
        orientations = get_dual_neutral_orientations()

    for ori in orientations:
        cube = CubeState()
        if ori.rotations:
            cube.apply_moves(ori.rotations)
        if not ori.is_fb_solved(cube):
            return False
        cube.apply_moves(" ".join(trigger.moves))
        if not ori.is_fb_solved(cube):
            return False

    return True


@dataclass(frozen=True)
class MacroTriggerMatch:
    """Represents a matched macro trigger instance in a move sequence."""
    trigger: MacroTrigger
    start_index: int
    end_index: int  # exclusive index in the move sequence
    moves: Tuple[str, ...]


def match_macro_triggers(
    moves: Union[str, Sequence[Union[str, MoveEvent]]],
) -> List[MacroTriggerMatch]:
    """Identifies closed-loop macro triggers in arbitrary move sequences.
    
    Args:
        moves: Move sequence as a space-delimited string or list of tokens / MoveEvents.
        
    Returns:
        List of MacroTriggerMatch records in order of occurrence.
    """
    if isinstance(moves, str):
        events = MoveParser.parse_string(moves)
        tokens = [e.move for e in events]
    else:
        tokens = [
            m.move if isinstance(m, MoveEvent) else MoveParser.normalize_token(str(m))
            for m in moves
        ]

    # Preserve all tokens (including rotations) so that macro triggers must be contiguous
    raw_tokens = [t for t in tokens if t]

    matches: List[MacroTriggerMatch] = []
    i = 0
    n = len(raw_tokens)

    while i < n:
        matched = False
        for trigger in MACRO_TRIGGERS.values():
            trig_len = len(trigger.moves)
            if i + trig_len <= n:
                candidate = tuple(raw_tokens[i + k] for k in range(trig_len))
                if candidate == trigger.moves:
                    matches.append(
                        MacroTriggerMatch(
                            trigger=trigger,
                            start_index=i,
                            end_index=i + trig_len,
                            moves=candidate,
                        )
                    )
                    i += trig_len
                    matched = True
                    break
        if not matched:
            i += 1

    return matches


def evaluate_macro_triggers(
    moves: Union[str, Sequence[Union[str, MoveEvent]]],
) -> Tuple[List[MacroTriggerMatch], int]:
    """Identifies macro triggers and evaluates them with 0 regrips within the macro triggers.
    
    Returns:
        Tuple of (matches, regrip_count_within_macros), where regrip_count_within_macros is always 0.
    """
    matches = match_macro_triggers(moves)
    return matches, 0

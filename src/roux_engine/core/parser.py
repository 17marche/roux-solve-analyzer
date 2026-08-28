"""MoveParser for tokenizing, normalizing, and parsing move sequences and smart-cube streams."""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union


@dataclass
class MoveEvent:
    """Represents a single parsed move event with optional timestamp metadata."""
    move: str                    # Canonical normalized token (e.g. "R", "r'", "M2", "U")
    raw_token: str               # The original token as parsed
    timestamp_ms: Optional[int] = None
    delta_ms: Optional[int] = None  # Inter-move transition time in milliseconds

    def __repr__(self) -> str:
        if self.timestamp_ms is not None:
            return f"MoveEvent('{self.move}', t={self.timestamp_ms}ms, Δt={self.delta_ms}ms)"
        return f"MoveEvent('{self.move}')"


class MoveParser:
    """Parser and normalizer for Rubik's cube move sequences."""

    # Canonical move normalizations mapping
    _ALIAS_MAP: Dict[str, str] = {
        # Wide moves
        "Rw": "r", "Rw'": "r'", "Rw2": "r2", "Rw2'": "r2",
        "Lw": "l", "Lw'": "l'", "Lw2": "l2", "Lw2'": "l2",
        "Uw": "u", "Uw'": "u'", "Uw2": "u2", "Uw2'": "u2",
        "Dw": "d", "Dw'": "d'", "Dw2": "d2", "Dw2'": "d2",
        "Fw": "f", "Fw'": "f'", "Fw2": "f2", "Fw2'": "f2",
        "Bw": "b", "Bw'": "b'", "Bw2": "b2", "Bw2'": "b2",
        # Prefix wide notations (2R -> r)
        "2R": "r", "2R'": "r'", "2R2": "r2",
        "2L": "l", "2L'": "l'", "2L2": "l2",
        "2U": "u", "2U'": "u'", "2U2": "u2",
        "2D": "d", "2D'": "d'", "2D2": "d2",
        "2F": "f", "2F'": "f'", "2F2": "f2",
        "2B": "b", "2B'": "b'", "2B2": "b2",
        # Reverse primes / prime variations
        "U2'": "U2", "D2'": "D2", "F2'": "F2", "B2'": "B2", "L2'": "L2", "R2'": "R2",
        "M2'": "M2", "E2'": "E2", "S2'": "S2",
        "u2'": "u2", "d2'": "d2", "f2'": "f2", "b2'": "b2", "l2'": "l2", "r2'": "r2",
        "x2'": "x2", "y2'": "y2", "z2'": "z2",
    }

    @classmethod
    def normalize_token(cls, token: str) -> str:
        """Normalizes a single move token into canonical Roux/WCA notation."""
        cleaned = token.strip().replace("’", "'").replace("‘", "'").replace("`", "'")
        
        # Strip surrounding brackets or parentheses around a single move (e.g. [U2] -> U2, (M') -> M')
        if (cleaned.startswith("[") and cleaned.endswith("]")) or (cleaned.startswith("(") and cleaned.endswith(")")):
            cleaned = cleaned[1:-1].strip()

        if cleaned in cls._ALIAS_MAP:
            return cls._ALIAS_MAP[cleaned]
        return cleaned

    @classmethod
    def is_valid_move_token(cls, token: str) -> bool:
        """Checks if a string token looks like a valid speedcube turn."""
        norm = cls.normalize_token(token)
        from .moves import MOVES
        return norm in MOVES

    @classmethod
    def parse_string(cls, moves_str: str) -> List[MoveEvent]:
        """Parses a space-separated or formatted move string into MoveEvent objects."""
        # Strip C-style comments (/* ... */) and line comments (// ...)
        text = re.sub(r"//.*", "", moves_str)
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        
        # Replace brackets and parentheses with spaces if they contain whitespace or are wrapper brackets
        # But allow tokens like [U2] or (M') to be handled cleanly
        text = text.replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ")
        
        # Tokenize by whitespace and commas
        raw_tokens = re.split(r"[\s,]+", text.strip())
        events: List[MoveEvent] = []

        for raw in raw_tokens:
            if not raw:
                continue
            normalized = cls.normalize_token(raw)
            # Only add if it's a recognized move
            if cls.is_valid_move_token(normalized):
                events.append(MoveEvent(move=normalized, raw_token=raw))

        return events

    @classmethod
    def parse_smart_cube_stream(cls, stream: List[Dict[str, Any]]) -> List[MoveEvent]:
        """Parses a smart-cube stream array into MoveEvents with delta times calculated."""
        events: List[MoveEvent] = []
        prev_t: Optional[int] = None

        for item in stream:
            raw_move = item.get("move") or item.get("m") or ""
            if not raw_move:
                continue
            
            t_ms = item.get("timestamp_ms") or item.get("t_ms") or item.get("t")
            if t_ms is not None:
                t_ms = int(t_ms)

            delta_ms = None
            if t_ms is not None and prev_t is not None:
                delta_ms = max(0, t_ms - prev_t)
            prev_t = t_ms

            normalized = cls.normalize_token(raw_move)
            events.append(MoveEvent(
                move=normalized,
                raw_token=raw_move,
                timestamp_ms=t_ms,
                delta_ms=delta_ms
            ))

        return events

    @staticmethod
    def invert_move(move_name: str) -> str:
        """Returns the algebraic inverse of a move (e.g. R -> R', R' -> R, R2 -> R2)."""
        norm = MoveParser.normalize_token(move_name)
        if not norm:
            return ""
        if norm.endswith("2"):
            return norm
        if norm.endswith("'"):
            return norm[:-1]
        return f"{norm}'"

    @staticmethod
    def invert_moves(moves: Union[str, List[Union[str, MoveEvent]]]) -> List[str]:
        """Inverts and reverses a sequence of moves (for undoing scrambles)."""
        if isinstance(moves, str):
            tokens = [e.move for e in MoveParser.parse_string(moves)]
        else:
            tokens = [m.move if isinstance(m, MoveEvent) else str(m) for m in moves]
        
        return [MoveParser.invert_move(t) for t in reversed(tokens)]

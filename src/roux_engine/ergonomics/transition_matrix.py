"""Calibrated Bigram Transition Matrices and Smart-Cube Ingestion Pipeline.

Includes Two-Handed (2H) and One-Handed (OH) transition models.
Ported with explicit attribution to onionhoney/roux-trainers (two_gram_v1.json).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, Sequence
from dataclasses import dataclass, field

from ..core.parser import MoveParser, MoveEvent
from .models import HandProfile


class TransitionMatrix:
    """Empirical Bigram Transition Matrix mapping move transitions to dimensionless effort multipliers.

    Ported with attribution from onionhoney/roux-trainers (two_gram_v1.json) under MIT License.
    Effort values are normalized relative to a baseline turn (effort = 1.0 for a baseline turn).
    """

    DEFAULT_BASELINE_LATENCY_SEC: float = 0.10

    def __init__(
        self,
        transitions: Dict[str, float],
        profile: Optional[HandProfile] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.transitions = dict(transitions)
        self.profile = profile or HandProfile()
        self.metadata = dict(metadata or {})

    @classmethod
    def _get_default_data_path(cls, solving_mode: str) -> Path:
        """Locates the packaged default transition matrix JSON."""
        mode = solving_mode.upper()
        filename = "matrix_oh.json" if mode == "OH" else "matrix_2h.json"
        return Path(__file__).resolve().parents[1] / "data" / "transitions" / filename

    @classmethod
    def load(
        cls,
        path: Optional[Union[str, Path]] = None,
        profile: Optional[HandProfile] = None,
    ) -> TransitionMatrix:
        """Loads a TransitionMatrix from file or packaged defaults for the given profile."""
        prof = profile or HandProfile()
        target_path = Path(path) if path is not None else cls._get_default_data_path(prof.solving_mode)

        if not target_path.is_file():
            raise FileNotFoundError(f"Transition matrix file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        metadata: Dict[str, Any] = {}
        transitions: Dict[str, float] = {}

        if "transitions" in raw and isinstance(raw["transitions"], dict):
            transitions = {k: float(v) for k, v in raw["transitions"].items()}
            metadata = raw.get("_metadata", {})
        else:
            for k, v in raw.items():
                if k.startswith("_"):
                    metadata[k] = v
                else:
                    transitions[k] = float(v)

        return cls(transitions=transitions, profile=prof, metadata=metadata)

    @classmethod
    def load_2h(cls, path: Optional[Union[str, Path]] = None) -> TransitionMatrix:
        """Convenience loader for Two-Handed baseline matrix."""
        return cls.load(path=path, profile=HandProfile(solving_mode="2H"))

    @classmethod
    def load_oh(cls, path: Optional[Union[str, Path]] = None) -> TransitionMatrix:
        """Convenience loader for One-Handed baseline matrix."""
        return cls.load(path=path, profile=HandProfile(solving_mode="OH"))

    def get_effort(self, prev_move: Optional[str], move: str) -> float:
        """Returns the dimensionless relative effort multiplier for a move or bigram transition."""
        norm_curr = MoveParser.normalize_token(move)
        # Whole-cube rotations take zero physical turn effort
        if norm_curr.startswith(("x", "y", "z")):
            return 0.0

        if prev_move is None:
            # Single move lookup
            if norm_curr in self.transitions:
                return self.transitions[norm_curr]
            return 1.0

        norm_prev = MoveParser.normalize_token(prev_move)
        if norm_prev.startswith(("x", "y", "z")):
            return self.get_effort(None, norm_curr)

        pair_key = f"{norm_prev}{norm_curr}"
        if pair_key in self.transitions:
            return self.transitions[pair_key]

        # Fallback interpolation for unobserved bigrams
        return self._interpolate_fallback(norm_prev, norm_curr)

    def __getitem__(self, key: Union[str, Tuple[Optional[str], str]]) -> float:
        """Convenience indexing supporting matrix['RU'], matrix[('R', 'U')], or matrix['R']."""
        if isinstance(key, tuple):
            if len(key) == 2:
                return self.get_effort(key[0], key[1])
            raise KeyError(f"Invalid tuple key length for TransitionMatrix: {key}")

        if not isinstance(key, str):
            raise TypeError(f"Key must be str or 2-tuple, got {type(key)}")

        norm = MoveParser.normalize_token(key)
        # Direct match in transitions dict
        if norm in self.transitions:
            return self.transitions[norm]

        # Parse string keys that may be compound bigrams (e.g. "M2b'") or single moves
        parsed = MoveParser.parse_string(key)
        if len(parsed) == 2:
            return self.get_effort(parsed[0].move, parsed[1].move)
        if len(parsed) == 1:
            return self.get_effort(None, parsed[0].move)

        return self.get_effort(None, norm)

    def __contains__(self, key: object) -> bool:
        """Checks if a transition or single move is explicitly present in the matrix."""
        if isinstance(key, tuple) and len(key) == 2:
            prev, curr = key
            norm_curr = MoveParser.normalize_token(curr)
            if prev is None:
                return norm_curr in self.transitions
            norm_prev = MoveParser.normalize_token(prev)
            return f"{norm_prev}{norm_curr}" in self.transitions

        if isinstance(key, str):
            norm = MoveParser.normalize_token(key)
            return norm in self.transitions

        return False

    def __len__(self) -> int:
        """Returns the number of explicit transitions stored."""
        return len(self.transitions)

    def _interpolate_fallback(self, prev_move: str, curr_move: str) -> float:
        """Interpolates effort for unobserved bigrams using single-move marginals."""
        is_prev_valid = MoveParser.is_valid_move_token(prev_move)
        is_curr_valid = MoveParser.is_valid_move_token(curr_move)

        if not is_prev_valid and not is_curr_valid:
            return 1.0

        single_prev = self.transitions.get(prev_move, 1.0)
        single_curr = self.transitions.get(curr_move, 1.0)

        # Baseline average of marginal efforts
        avg_effort = (single_prev + single_curr) / 2.0

        # Awkwardness modifier if consecutive turns involve awkward face combinations
        if is_prev_valid and is_curr_valid:
            face_prev = prev_move[0].upper()
            face_curr = curr_move[0].upper()
            if face_prev in "FB" or face_curr in "FB":
                avg_effort *= 1.2
            elif face_prev == face_curr:
                # Same face consecutive turns without cancellation (e.g. U then U')
                avg_effort *= 1.1

        return round(avg_effort, 4)

    def save(self, path: Union[str, Path]) -> None:
        """Saves matrix to JSON file."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "_metadata": self.metadata,
            "transitions": self.transitions,
        }
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class TransitionMatrixBuilder:
    """Ingests Bluetooth smart-cube move streams and calibrates empirical transition matrices."""

    def __init__(self, max_delta_ms: int = 1500) -> None:
        self.max_delta_ms = max_delta_ms
        self._samples: Dict[str, List[float]] = {}
        self._single_samples: Dict[str, List[float]] = {}
        self._total_events_ingested: int = 0

    def ingest_stream(
        self,
        stream: Union[str, Sequence[Union[str, MoveEvent, Dict[str, Any]]]],
    ) -> int:
        """Ingests a smart-cube stream, extracting inter-move transition times and filtering pause outliers."""
        events: List[MoveEvent] = []

        if isinstance(stream, str):
            try:
                parsed_json = json.loads(stream)
                if isinstance(parsed_json, list) and len(parsed_json) > 0 and isinstance(parsed_json[0], dict):
                    events = MoveParser.parse_smart_cube_stream(parsed_json)
                else:
                    events = MoveParser.parse_string(stream)
            except Exception:
                events = MoveParser.parse_string(stream)
        elif len(stream) > 0 and isinstance(stream[0], dict):
            # Sequence of dicts
            events = MoveParser.parse_smart_cube_stream(list(stream))  # type: ignore[arg-type]
        else:
            for item in stream:
                if isinstance(item, MoveEvent):
                    events.append(item)
                else:
                    events.append(MoveEvent(move=MoveParser.normalize_token(str(item)), raw_token=str(item)))

        # Normalize inter-move deltas across all events
        prev_t: Optional[int] = None
        for ev in events:
            if ev.delta_ms is None and ev.timestamp_ms is not None and prev_t is not None:
                ev.delta_ms = max(0, ev.timestamp_ms - prev_t)
            if ev.timestamp_ms is not None:
                prev_t = ev.timestamp_ms

        # Process consecutive events and record transition latencies
        ingested_count = 0
        prev_event: Optional[MoveEvent] = None

        for ev in events:
            norm_curr = ev.move
            # Whole cube rotations reset timing context
            if norm_curr.startswith(("x", "y", "z")):
                prev_event = None
                continue

            if prev_event is not None and ev.delta_ms is not None:
                delta = float(ev.delta_ms)
                if delta <= self.max_delta_ms:
                    pair_key = f"{prev_event.move}{norm_curr}"
                    if pair_key not in self._samples:
                        self._samples[pair_key] = []
                    self._samples[pair_key].append(delta)

                    if norm_curr not in self._single_samples:
                        self._single_samples[norm_curr] = []
                    self._single_samples[norm_curr].append(delta)

                    ingested_count += 1

            prev_event = ev

        self._total_events_ingested += ingested_count
        return ingested_count

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calculates descriptive empirical statistics for all observed bigrams."""
        import statistics

        stats: Dict[str, Dict[str, float]] = {}
        for key, deltas in self._samples.items():
            if not deltas:
                continue
            count = len(deltas)
            mean_val = float(statistics.mean(deltas))
            med_val = float(statistics.median(deltas))
            std_val = float(statistics.stdev(deltas)) if count >= 2 else 0.0
            min_val = float(min(deltas))
            max_val = float(max(deltas))

            stats[key] = {
                "count": float(count),
                "mean_ms": round(mean_val, 4),
                "median_ms": round(med_val, 4),
                "std_ms": round(std_val, 4),
                "min_ms": round(min_val, 4),
                "max_ms": round(max_val, 4),
            }

        return stats

    def build(
        self,
        baseline_latency_ms: Optional[float] = None,
        profile: Optional[HandProfile] = None,
        fallback_matrix: Optional[TransitionMatrix] = None,
        min_samples: int = 1,
    ) -> TransitionMatrix:
        """Calibrates a custom TransitionMatrix from ingested smart-cube statistics."""
        import statistics

        prof = profile or HandProfile()
        fallback = fallback_matrix or TransitionMatrix.load(profile=prof)

        # Determine baseline turn latency (effort = 1.0)
        if baseline_latency_ms is None:
            all_deltas: List[float] = []
            for deltas in self._samples.values():
                all_deltas.extend(deltas)
            if all_deltas:
                baseline_latency_ms = float(statistics.median(all_deltas))
            else:
                baseline_latency_ms = fallback.DEFAULT_BASELINE_LATENCY_SEC * 1000.0

        # Start with fallback transitions
        calibrated_transitions = dict(fallback.transitions)

        # Calibrate observed bigrams
        unique_calibrated = 0
        for key, deltas in self._samples.items():
            if len(deltas) >= min_samples:
                med = float(statistics.median(deltas))
                effort = round(med / baseline_latency_ms, 4)
                calibrated_transitions[key] = effort
                unique_calibrated += 1

        # Calibrate observed single turns
        for move, deltas in self._single_samples.items():
            if len(deltas) >= min_samples:
                med = float(statistics.median(deltas))
                effort = round(med / baseline_latency_ms, 4)
                calibrated_transitions[move] = effort

        metadata = dict(fallback.metadata)
        metadata.update({
            "calibrated_from_stream": True,
            "baseline_latency_ms": baseline_latency_ms,
            "total_events_ingested": self._total_events_ingested,
            "unique_transitions_calibrated": unique_calibrated,
            "max_delta_ms": self.max_delta_ms,
        })

        return TransitionMatrix(
            transitions=calibrated_transitions,
            profile=prof,
            metadata=metadata,
        )


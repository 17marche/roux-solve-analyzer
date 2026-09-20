"""Ergonomics package for biomechanical flow scoring, transition matrices, and grip tracking."""

from .models import GripState, HandProfile, MoveAnalysis, FlowScore, PauseType
from .grip_tracker import GripTracker, GripStep, GripTrackingResult
from .flow_scorer import FlowScorer
from .transition_matrix import TransitionMatrix, TransitionMatrixBuilder
from .pause_detector import PauseEvent, StreamMetrics, StreamPauseDetector
from .macro_triggers import (
    MacroTrigger,
    MacroTriggerMatch,
    SLEDGEHAMMER,
    HEDGE,
    MACRO_TRIGGERS,
    verify_first_block_preservation,
    match_macro_triggers,
    evaluate_macro_triggers,
)

__all__ = [
    "GripState",
    "HandProfile",
    "MoveAnalysis",
    "FlowScore",
    "PauseType",
    "GripTracker",
    "GripStep",
    "GripTrackingResult",
    "FlowScorer",
    "TransitionMatrix",
    "TransitionMatrixBuilder",
    "PauseEvent",
    "StreamMetrics",
    "StreamPauseDetector",
    "MacroTrigger",
    "MacroTriggerMatch",
    "SLEDGEHAMMER",
    "HEDGE",
    "MACRO_TRIGGERS",
    "verify_first_block_preservation",
    "match_macro_triggers",
    "evaluate_macro_triggers",
]



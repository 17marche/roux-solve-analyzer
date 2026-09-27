"""Tests for StreamPauseDetector, turning ratio, rhythm CV, and pause classification."""

import pytest
from roux_engine.core.parser import MoveParser, MoveEvent
from roux_engine.ergonomics.models import GripState
from roux_engine.ergonomics.pause_detector import (
    PauseType,
    PauseEvent,
    StreamPauseDetector,
)


class TestStreamPauseDetectorMetrics:
    """Seam 3.1 & 3.2: Turning ratio, rhythm CV, and pace invariance."""

    def test_empty_stream(self):
        detector = StreamPauseDetector()
        metrics = detector.calculate_metrics([])
        assert metrics.turning_ratio == 0.0
        assert metrics.rhythm_cv == 0.0
        assert metrics.stream_flow_index == 0.0
        assert metrics.pauses == []

    def test_single_move_stream(self):
        detector = StreamPauseDetector()
        stream = [{"move": "R", "timestamp_ms": 100}]
        metrics = detector.calculate_metrics(stream)
        assert metrics.turning_ratio == 1.0
        assert metrics.rhythm_cv == 0.0
        assert metrics.stream_flow_index == 100.0
        assert metrics.pauses == []

    def test_perfectly_consistent_stream_without_pauses(self):
        detector = StreamPauseDetector()
        # 5 moves, every interval exactly 120ms
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
            {"move": "R'", "timestamp_ms": 340},
            {"move": "U'", "timestamp_ms": 460},
            {"move": "R", "timestamp_ms": 580},
        ]
        metrics = detector.calculate_metrics(stream)
        assert metrics.rhythm_cv == 0.0
        assert metrics.turning_ratio == 1.0
        assert metrics.stream_flow_index == 100.0
        assert len(metrics.pauses) == 0

    def test_pace_invariance_7s_vs_14s_solver(self):
        """Scaling all timestamps by 2x must yield identical turning_ratio, rhythm_cv, and stream_flow_index."""
        detector = StreamPauseDetector()

        # Fast 7s solver rhythm with 1 pause
        stream_fast = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 120},
            {"move": "R'", "timestamp_ms": 240},
            {"move": "U'", "timestamp_ms": 360},
            {"move": "R", "timestamp_ms": 1200},  # Pause
            {"move": "U", "timestamp_ms": 1320},
        ]

        # 14s solver: exactly 2x slower intervals
        stream_slow = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 240},
            {"move": "R'", "timestamp_ms": 480},
            {"move": "U'", "timestamp_ms": 720},
            {"move": "R", "timestamp_ms": 2400},  # Pause
            {"move": "U", "timestamp_ms": 2640},
        ]

        m_fast = detector.calculate_metrics(stream_fast)
        m_slow = detector.calculate_metrics(stream_slow)

        assert round(m_fast.rhythm_cv, 4) == round(m_slow.rhythm_cv, 4)
        assert round(m_fast.turning_ratio, 4) == round(m_slow.turning_ratio, 4)
        assert round(m_fast.stream_flow_index, 2) == round(m_slow.stream_flow_index, 2)
        assert len(m_fast.pauses) == len(m_slow.pauses) == 1
        assert m_fast.pauses[0].pause_type == m_slow.pauses[0].pause_type


class TestStreamPauseClassification:
    """Seam 3.3: Classifying pauses into PHYSICAL_REGRIP, COGNITIVE_HESITATION, and EXECUTION_LOCKUP."""

    def test_physical_regrip_classification(self):
        detector = StreamPauseDetector(pause_threshold_ms=250.0, pace_adaptive=False)
        # R -> R forces a kinematic regrip (R is blocked in R_AWAY)
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "R", "timestamp_ms": 450},  # delta = 350ms > 250ms, forced regrip
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 1
        assert pauses[0].move == "R"
        assert pauses[0].pause_type == PauseType.PHYSICAL_REGRIP
        assert pauses[0].pause_type == "PHYSICAL_REGRIP"

    def test_execution_lockup_classification(self):
        detector = StreamPauseDetector(
            pause_threshold_ms=250.0,
            cognitive_threshold_ms=500.0,
            pace_adaptive=False,
        )
        # R -> U requires NO regrip, but took 350ms (between 250ms and 500ms)
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 450},  # delta = 350ms, lockup
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 1
        assert pauses[0].move == "U"
        assert pauses[0].pause_type == PauseType.EXECUTION_LOCKUP
        assert pauses[0].pause_type == "EXECUTION_LOCKUP"

    def test_cognitive_hesitation_classification(self):
        detector = StreamPauseDetector(
            pause_threshold_ms=250.0,
            cognitive_threshold_ms=500.0,
            pace_adaptive=False,
        )
        # R -> U requires NO regrip, and took 900ms (> 500ms cognitive threshold)
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 1000},  # delta = 900ms, cognitive hesitation
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 1
        assert pauses[0].move == "U"
        assert pauses[0].pause_type == PauseType.COGNITIVE_HESITATION
        assert pauses[0].pause_type == "COGNITIVE_HESITATION"

    def test_cognitive_hesitation_takes_precedence_over_regrip_on_long_pause(self):
        detector = StreamPauseDetector(
            pause_threshold_ms=250.0,
            cognitive_threshold_ms=500.0,
            pace_adaptive=False,
        )
        # R -> R forces a kinematic regrip, but took 1200ms (>= 500ms cognitive threshold)
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "R", "timestamp_ms": 1300},  # delta = 1200ms
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 1
        assert pauses[0].move == "R"
        assert pauses[0].pause_type == PauseType.COGNITIVE_HESITATION

    def test_mixed_stream_with_all_pause_types(self):
        detector = StreamPauseDetector(
            pause_threshold_ms=250.0,
            cognitive_threshold_ms=500.0,
            pace_adaptive=False,
        )
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "R", "timestamp_ms": 450},   # 350ms: R->R forced regrip -> PHYSICAL_REGRIP
            {"move": "R", "timestamp_ms": 570},   # 120ms: fluent turn (no pause)
            {"move": "U", "timestamp_ms": 920},   # 350ms: no regrip, 350ms -> EXECUTION_LOCKUP
            {"move": "R'", "timestamp_ms": 1040}, # 120ms: fluent turn (no pause)
            {"move": "U'", "timestamp_ms": 2040}, # 1000ms: no regrip, 1000ms -> COGNITIVE_HESITATION
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 3

        types = [p.pause_type for p in pauses]
        assert types == [
            PauseType.PHYSICAL_REGRIP,
            PauseType.EXECUTION_LOCKUP,
            PauseType.COGNITIVE_HESITATION,
        ]

        breakdown = detector.get_pause_breakdown(stream)
        assert breakdown["PHYSICAL_REGRIP"] == 1
        assert breakdown["EXECUTION_LOCKUP"] == 1
        assert breakdown["COGNITIVE_HESITATION"] == 1


class TestStreamPauseDetectorPropertiesAndEdgeCases:
    """Seam 3.5: Scaling invariants, input formats, rotations, and bounds."""

    @pytest.mark.parametrize("scale", [0.5, 1.5, 2.0, 3.0, 5.0, 10.0])
    def test_pace_invariance_across_multiple_scaling_factors(self, scale: float):
        detector = StreamPauseDetector()
        base_stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
            {"move": "R'", "timestamp_ms": 340},
            {"move": "U'", "timestamp_ms": 460},
            {"move": "R", "timestamp_ms": 1200},  # Pause
            {"move": "U", "timestamp_ms": 1320},
            {"move": "R'", "timestamp_ms": 1440},
        ]

        scaled_stream = [
            {"move": item["move"], "timestamp_ms": int(item["timestamp_ms"] * scale)}
            for item in base_stream
        ]

        m_base = detector.calculate_metrics(base_stream)
        m_scaled = detector.calculate_metrics(scaled_stream)

        assert round(m_base.turning_ratio, 3) == round(m_scaled.turning_ratio, 3)
        assert round(m_base.rhythm_cv, 3) == round(m_scaled.rhythm_cv, 3)
        assert round(m_base.stream_flow_index, 1) == round(m_scaled.stream_flow_index, 1)
        assert len(m_base.pauses) == len(m_scaled.pauses)
        if m_base.pauses:
            assert m_base.pauses[0].pause_type == m_scaled.pauses[0].pause_type

    def test_json_string_input(self):
        import json
        detector = StreamPauseDetector()
        stream_data = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 220},
            {"move": "R'", "timestamp_ms": 340},
        ]
        json_str = json.dumps(stream_data)
        metrics = detector.calculate_metrics(json_str)
        assert metrics.turning_ratio == 1.0
        assert metrics.rhythm_cv == 0.0
        assert metrics.stream_flow_index == 100.0

    def test_stream_with_rotations(self):
        detector = StreamPauseDetector()
        stream = [
            {"move": "x", "timestamp_ms": 50},
            {"move": "R", "timestamp_ms": 150},
            {"move": "y", "timestamp_ms": 250},
            {"move": "U", "timestamp_ms": 370},
        ]
        metrics = detector.calculate_metrics(stream)
        assert metrics.total_time_ms == 320
        assert metrics.turning_ratio == 1.0

    def test_pause_event_attributes(self):
        detector = StreamPauseDetector(pause_threshold_ms=250.0, pace_adaptive=False)
        stream = [
            {"move": "R", "timestamp_ms": 100},
            {"move": "U", "timestamp_ms": 500},
        ]
        pauses = detector.detect_pauses(stream)
        assert len(pauses) == 1
        p = pauses[0]
        assert p.move_index == 1
        assert p.move == "U"
        assert p.prev_move == "R"
        assert p.delta_ms == 400
        assert p.timestamp_ms == 500
        assert p.expected_ms > 0.0
        assert p.ratio > 0.0
        assert isinstance(p.pause_type, PauseType)

    def test_metrics_bounds(self):
        detector = StreamPauseDetector()
        stream = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 1000},
            {"move": "R'", "timestamp_ms": 1100},
            {"move": "U'", "timestamp_ms": 5000},
        ]
        metrics = detector.calculate_metrics(stream)
        assert 0.0 <= metrics.turning_ratio <= 1.0
        assert metrics.rhythm_cv >= 0.0
        assert 0.0 <= metrics.stream_flow_index <= 100.0



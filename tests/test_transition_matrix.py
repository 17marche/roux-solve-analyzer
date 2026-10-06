"""Tests for TransitionMatrix and TransitionMatrixBuilder (Issue 02)."""

import pytest
from roux_engine.ergonomics.models import HandProfile
from roux_engine.ergonomics.transition_matrix import TransitionMatrix, TransitionMatrixBuilder


class TestTransitionMatrixLoadingAndBaselines:
    """Seam 1: TransitionMatrix loading, metadata attribution, and baseline lookups."""

    def test_load_2h_matrix_attribution_and_scale(self):
        matrix = TransitionMatrix.load_2h()
        assert matrix.profile.solving_mode == "2H"
        # Explicit attribution to onionhoney/roux-trainers
        assert "onionhoney" in matrix.metadata.get("attribution", "").lower()
        assert "roux-trainers" in matrix.metadata.get("attribution", "").lower()

        # At least 528 calibrated pairs from onionhoney
        assert len(matrix.transitions) >= 528

        # Fluid triggers should be sub-1.0 effort
        effort_ru = matrix.get_effort("R", "U")
        assert 0.45 < effort_ru < 0.55

        # Awkward reaches should be > 2.5 effort
        effort_rf = matrix.get_effort("R", "F")
        assert effort_rf > 2.5

        # Single move lookup
        effort_r = matrix.get_effort(None, "R")
        assert 0.6 < effort_r < 1.0

    def test_load_oh_matrix_mechanics(self):
        matrix_2h = TransitionMatrix.load_2h()
        matrix_oh = TransitionMatrix.load_oh()
        assert matrix_oh.profile.solving_mode == "OH"

        # In OH, table and finger mechanics make F, B, and M moves significantly more costly than in 2H
        assert matrix_oh.get_effort(None, "F") > matrix_2h.get_effort(None, "F")
        assert matrix_oh.get_effort(None, "B") > matrix_2h.get_effort(None, "B")
        assert matrix_oh.get_effort(None, "M") > matrix_2h.get_effort(None, "M")

        # Fluid R/U triggers remain relatively low effort in OH
        effort_ru_oh = matrix_oh.get_effort("R", "U")
        assert effort_ru_oh < 1.0

    def test_rotations_have_zero_effort_and_reset_context(self):
        matrix = TransitionMatrix.load_2h()
        assert matrix.get_effort(None, "x") == 0.0
        assert matrix.get_effort("R", "y") == 0.0
        assert matrix.get_effort("x", "U") == matrix.get_effort(None, "U")


class TestMatrixQueryRobustnessAndFallback:
    """Seam 2: Fallback interpolation, dictionary protocol, and robustness."""

    def test_dict_like_access_and_membership(self):
        matrix = TransitionMatrix.load_2h()
        # String key
        assert matrix["RU"] == matrix.get_effort("R", "U")
        # Tuple key
        assert matrix[("R", "U")] == matrix.get_effort("R", "U")
        assert matrix[(None, "R")] == matrix.get_effort(None, "R")
        assert matrix["R"] == matrix.get_effort(None, "R")

        # In operator
        assert "RU" in matrix
        assert ("R", "U") in matrix
        assert "R" in matrix
        assert "NONEXISTENT_KEY" not in matrix

        # Length
        assert len(matrix) >= 528

    def test_fallback_interpolation_unseen_bigrams(self):
        matrix = TransitionMatrix.load_2h()
        # Create an artificial transition not in the matrix
        effort = matrix.get_effort("M2", "b'")
        assert effort > 0.0
        # Should not default to 0 or crash
        assert isinstance(effort, float)
        assert 0.5 <= effort <= 5.0
        # Accessing via string key directly parses compound bigram
        assert matrix["M2 b'"] == effort

    def test_unknown_tokens_gracefully_default(self):
        # Empty matrix with no transitions
        sparse_matrix = TransitionMatrix(transitions={})
        effort = sparse_matrix.get_effort("UNKNOWN_1", "UNKNOWN_2")
        assert effort == 1.0
        assert sparse_matrix.get_effort(None, "UNKNOWN") == 1.0


class TestTransitionMatrixBuilderIngestionAndCalibration:
    """Seam 3: Smart-cube stream ingestion, pause filtering, statistics, and matrix calibration."""

    def test_ingest_smart_cube_stream_and_pause_filtering(self):
        builder = TransitionMatrixBuilder(max_delta_ms=1500)
        stream = [
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 80},     # delta 80ms (RU)
            {"move": "R'", "timestamp_ms": 160},   # delta 80ms (UR')
            {"move": "U'", "timestamp_ms": 3000},  # delta 2840ms (> 1500ms micro-pause)
            {"move": "R", "timestamp_ms": 3120},   # delta 120ms (U'R)
        ]
        ingested = builder.ingest_stream(stream)
        assert ingested == 3  # RU, UR', and U'R (R'U' dropped due to micro-pause)

        stats = builder.get_statistics()
        assert "RU" in stats
        assert stats["RU"]["count"] == 1
        assert stats["RU"]["median_ms"] == 80.0

        assert "UR'" in stats
        assert stats["UR'"]["count"] == 1
        assert stats["UR'"]["median_ms"] == 80.0

        assert "U'R" in stats
        assert stats["U'R"]["count"] == 1
        assert stats["U'R"]["median_ms"] == 120.0

        # Long pause between R' and U' must NOT be in bigram transitions
        assert "R'U'" not in stats

    def test_empirical_statistics_multiple_samples(self):
        builder = TransitionMatrixBuilder()
        # Ingest 3 distinct solves with "RU" transitions
        builder.ingest_stream([{"move": "R", "t": 0}, {"move": "U", "t": 70}])
        builder.ingest_stream([{"move": "R", "t": 100}, {"move": "U", "t": 190}])
        builder.ingest_stream([{"move": "R", "t": 200}, {"move": "U", "t": 320}])

        stats = builder.get_statistics()
        ru_stats = stats["RU"]
        assert ru_stats["count"] == 3
        # deltas: 70, 90, 120
        assert ru_stats["median_ms"] == 90.0
        assert round(ru_stats["mean_ms"], 1) == 93.3
        assert ru_stats["min_ms"] == 70.0
        assert ru_stats["max_ms"] == 120.0
        assert ru_stats["std_ms"] > 0.0

    def test_build_calibrated_matrix_with_fallback(self):
        builder = TransitionMatrixBuilder()
        builder.ingest_stream([
            {"move": "R", "timestamp_ms": 0},
            {"move": "U", "timestamp_ms": 60},
            {"move": "R'", "timestamp_ms": 110},
        ])

        # Calibrate with 100ms baseline (effort = 1.0 for 100ms)
        matrix = builder.build(baseline_latency_ms=100.0)
        assert isinstance(matrix, TransitionMatrix)

        # RU latency = 60ms -> effort = 0.6
        assert matrix["RU"] == 0.6
        # UR' latency = 50ms -> effort = 0.5
        assert matrix["UR'"] == 0.5

        # Unobserved transitions must be inherited from default baseline matrix
        assert "RF" in matrix
        assert matrix["RF"] > 2.5
        assert matrix.get_effort(None, "B") > 1.0

        # Metadata records calibration details
        assert matrix.metadata.get("calibrated_from_stream") is True
        assert matrix.metadata.get("baseline_latency_ms") == 100.0


class TestFlowScorerTransitionMatrixIntegration:
    """Seam 4: FlowScorer integration with TransitionMatrix and HandProfile."""

    def test_flow_scorer_evaluates_oh_profile_differently(self):
        from roux_engine.ergonomics.flow_scorer import FlowScorer
        scorer_2h = FlowScorer(profile=HandProfile(solving_mode="2H"))
        scorer_oh = FlowScorer(profile=HandProfile(solving_mode="OH"))

        # In OH, M-slice and F-turn heavy sequences are significantly harder
        seq = "M U M' U2 M' U2 M"
        score_2h = scorer_2h.score_moves(seq)
        score_oh = scorer_oh.score_moves(seq)

        assert score_2h.raw_stm == score_oh.raw_stm
        assert score_oh.e_stm > score_2h.e_stm
        assert score_oh.kinematic_efficiency < score_2h.kinematic_efficiency

    def test_flow_scorer_accepts_transition_matrix_instance(self):
        from roux_engine.ergonomics.flow_scorer import FlowScorer
        custom_tm = TransitionMatrix(
            transitions={"RU": 0.20, "UR'": 0.20, "R'U'": 0.20, "U'R": 0.20},
            profile=HandProfile(solving_mode="2H"),
        )
        scorer = FlowScorer(transition_matrix=custom_tm)
        score = scorer.score_moves("R U R' U'")
        # With effort 0.20 per transition, E-STM should be very low
        assert score.e_stm < 2.0


class TestTransitionMatrixProperties:
    """Property-based tests verifying invariant behaviors across domains."""

    def test_all_base_moves_have_positive_finite_effort(self):
        from roux_engine.core.moves import MOVES
        matrix_2h = TransitionMatrix.load_2h()
        matrix_oh = TransitionMatrix.load_oh()

        for move in MOVES:
            if move.startswith(("x", "y", "z")):
                continue
            effort_2h = matrix_2h.get_effort(None, move)
            effort_oh = matrix_oh.get_effort(None, move)
            assert effort_2h > 0.0
            assert effort_oh > 0.0
            assert effort_2h < 20.0
            assert effort_oh < 20.0

    def test_baseline_latency_scaling_property(self):
        # Property: Doubling the baseline latency exactly halves calibrated efforts
        builder = TransitionMatrixBuilder()
        builder.ingest_stream([
            {"move": "R", "t": 0},
            {"move": "U", "t": 120},
            {"move": "R'", "t": 200},
        ])

        m100 = builder.build(baseline_latency_ms=100.0)
        m200 = builder.build(baseline_latency_ms=200.0)

        assert abs(m100["RU"] - 2.0 * m200["RU"]) < 1e-4
        assert abs(m100["UR'"] - 2.0 * m200["UR'"]) < 1e-4

    def test_rotations_invariance_across_all_axes(self):
        matrix = TransitionMatrix.load_2h()
        rotations = ["x", "x'", "x2", "y", "y'", "y2", "z", "z'", "z2"]
        for rot in rotations:
            assert matrix.get_effort(None, rot) == 0.0
            assert matrix.get_effort("R", rot) == 0.0
            assert matrix.get_effort(rot, "U") == matrix.get_effort(None, "U")

    def test_save_and_reload_matrix(self, tmp_path):
        matrix = TransitionMatrix.load_2h()
        target_file = tmp_path / "test_matrix.json"
        matrix.save(target_file)
        assert target_file.is_file()

        loaded = TransitionMatrix.load(path=target_file)
        assert loaded.profile.solving_mode == "2H"
        assert loaded["RU"] == matrix["RU"]
        assert len(loaded) == len(matrix)


class TestCalibrated2HMatrixInvariants:
    """Verification suite for the calibrated 2H bigram transition matrix (TRANSITION_MATRIX_CALIBRATION_SPEC.md)."""

    def test_calibrated_matrix_has_no_synthetic_3_defaults(self):
        matrix = TransitionMatrix.load_2h()
        # In the calibrated matrix, all unmeasured ~3.0 defaults (>= 2.88) must be eliminated.
        defaults = [k for k, v in matrix.transitions.items() if v >= 2.88]
        assert len(defaults) == 0, f"Found synthetic defaults: {defaults[:10]}"

    def test_double_move_monotonicity_invariant(self):
        matrix = TransitionMatrix.load_2h()
        # Rule 3: For all X2, effort(A, X2) >= min(effort(A, X), effort(A, X')) * 1.15
        faces = ["U", "D", "F", "B", "L", "R", "M", "r"]
        all_moves = [f for face in faces for f in [face, f"{face}'", f"{face}2"]]

        for face in faces:
            # Single move monotonicity
            effort_x = matrix.get_effort(None, face)
            effort_xp = matrix.get_effort(None, f"{face}'")
            effort_x2 = matrix.get_effort(None, f"{face}2")
            min_single = min(effort_x, effort_xp)
            assert effort_x2 >= min_single * 1.15 - 0.005, (
                f"Single move monotonicity paradox for {face}2: {effort_x2} < {min_single} * 1.15"
            )

            # Pair monotonicity: for all preceding moves A
            for a in all_moves:
                if a.startswith(face):
                    continue  # Same face does not appear in bigrams
                pair_x = f"{a}{face}"
                pair_xp = f"{a}{face}'"
                pair_x2 = f"{a}{face}2"
                if pair_x in matrix and pair_xp in matrix and pair_x2 in matrix:
                    min_flick = min(matrix[pair_x], matrix[pair_xp])
                    assert matrix[pair_x2] >= min_flick * 1.15 - 0.005, (
                        f"Pair monotonicity paradox for {pair_x2}: {matrix[pair_x2]} < {min_flick} * 1.15"
                    )

    def test_wide_turn_ratio_invariant(self):
        matrix = TransitionMatrix.load_2h()
        # Rule 1: Wide r inherits from outer R with ~1.12x drag factor
        # Check single moves
        for r_move, r_base in [("r", "R"), ("r'", "R'"), ("r2", "R2")]:
            ratio = matrix.get_effort(None, r_move) / matrix.get_effort(None, r_base)
            assert 1.05 <= ratio <= 1.20, f"Single wide ratio for {r_move}/{r_base} was {ratio:.3f}"

        # Check key bigrams from spec
        assert 0.50 <= matrix["rU"] <= 0.65
        assert 0.50 <= matrix["r'U"] <= 0.65

    def test_bilateral_mirror_invariant(self):
        matrix = TransitionMatrix.load_2h()
        # Rule 2: Left-hand turns mirror clean right-hand pairs with 1.15x drag factor
        assert abs(matrix.get_effort(None, "L'") - matrix.get_effort(None, "R") * 1.15) < 0.05
        assert abs(matrix.get_effort(None, "L2") - matrix.get_effort(None, "R2") * 1.15) < 0.05
        # Key pairs
        assert 0.50 <= matrix["U'L"] <= 0.65
        assert 0.50 <= matrix["L'U'"] <= 0.65

    def test_m_slice_recalibration_invariant(self):
        matrix = TransitionMatrix.load_2h()
        # Rule 4: Single M is calibrated to 1.45 (not 2.994)
        assert abs(matrix.get_effort(None, "M") - 1.45) < 0.05
        # Decoupled two-handed M/U pairs sit between 0.75 and 1.30
        assert 1.15 <= matrix["MU"] <= 1.30
        assert 0.90 <= matrix["MU'"] <= 1.05

        # Biomechanical invariants for 2H LSE (left hand U, right hand M):
        # Index pull (U') is faster than fingernail push (U)
        assert matrix["U'M'"] < matrix["U'M"]
        assert matrix["U'M'"] < matrix["UM'"]
        assert matrix["M2U'"] < matrix["M2U"]
        assert matrix["MU'"] < matrix["MU"]
        assert matrix["M'U'"] < matrix["M'U"]

        # Corrections for empirical raw artifacts:
        # AUF recognition pause corrected: U'M' is fast primary trigger
        assert 0.75 <= matrix["U'M'"] <= 0.82
        # Sensor clamp floor corrected: U2M' >= 0.80
        assert matrix["U2M'"] >= 0.80
        # H-perm algorithm drill bias corrected: M2U' < M2U
        assert matrix["M2U'"] == 0.820

    def test_fallback_interpolation_rule_6(self):
        matrix = TransitionMatrix.load_2h()
        # Fallback for arbitrary unobserved wide turn should strip wide notation and apply drag
        effort_r = matrix._interpolate_fallback("r", "B")
        effort_R = matrix._interpolate_fallback("R", "B")
        # Should be scaled by roughly 1.12
        assert effort_r > effort_R
        assert 1.05 <= (effort_r / effort_R) <= 1.20

    def test_left_m_slice_adaptation(self):
        matrix_right = TransitionMatrix.load(profile=HandProfile(m_slice_hand="right"))
        matrix_left = TransitionMatrix.load(profile=HandProfile(m_slice_hand="left"))

        # M* -> U* pairs must invert U <-> U'
        assert matrix_left["M'U"] == matrix_right["M'U'"] == 0.831
        assert matrix_left["M'U'"] == matrix_right["M'U"] == 1.063
        assert matrix_left["M2U"] == matrix_right["M2U'"] == 0.820
        assert matrix_left["M2U'"] == matrix_right["M2U"] == 0.884
        assert matrix_left["MU"] == matrix_right["MU'"] == 0.980
        assert matrix_left["MU'"] == matrix_right["MU"] == 1.220

        # U* -> M* pairs must invert U <-> U'
        assert matrix_left["UM'"] == matrix_right["U'M'"] == 0.780
        assert matrix_left["U'M'"] == matrix_right["UM'"] == 1.020
        assert matrix_left["UM"] == matrix_right["U'M"] == 0.980
        assert matrix_left["U'M"] == matrix_right["UM"] == 1.250
        assert matrix_left["UM2"] == matrix_right["U'M2"] == 0.900
        assert matrix_left["U'M2"] == matrix_right["UM2"] == 1.180

        # Invariance: U2 pairs must be identical
        assert matrix_left["M'U2"] == matrix_right["M'U2"]
        assert matrix_left["M2U2"] == matrix_right["M2U2"]
        assert matrix_left["MU2"] == matrix_right["MU2"]
        assert matrix_left["U2M'"] == matrix_right["U2M'"]
        assert matrix_left["U2M"] == matrix_right["U2M"]
        assert matrix_left["U2M2"] == matrix_right["U2M2"]

        # Invariance: non-M outer layer pairs must be completely identical
        assert matrix_left["RU"] == matrix_right["RU"]
        assert matrix_left["RU'"] == matrix_right["RU'"]
        assert matrix_left["R'U"] == matrix_right["R'U"]
        assert matrix_left["R'U'"] == matrix_right["R'U'"]
        assert matrix_left["LU"] == matrix_right["LU"]
        assert matrix_left["L'U'"] == matrix_right["L'U'"]
        assert matrix_left["FU"] == matrix_right["FU"]

    def test_fallback_interpolation_wide_turns(self):
        matrix = TransitionMatrix.load_2h()
        # Wide l inherits from outer L with 1.12x wide drag factor
        effort_l_u = matrix.get_effort("l", "U")
        effort_L_u = matrix.get_effort("L", "U")
        assert abs(effort_l_u - round(effort_L_u * 1.12, 4)) < 1e-4

        # Wide u inherits from outer U with 1.12x wide drag factor
        effort_u_r = matrix.get_effort("u", "R")
        effort_U_r = matrix.get_effort("U", "R")
        assert abs(effort_u_r - round(effort_U_r * 1.12, 4)) < 1e-4

        # Dual wide turns l and r apply 1.12 * 1.12 drag
        effort_l_r = matrix.get_effort("l", "r")
        effort_L_R = matrix.get_effort("L", "R")
        assert abs(effort_l_r - round(effort_L_R * 1.12 * 1.12, 4)) < 1e-4

        # Cube rotations do not inherit wide drag and maintain zero effort
        assert matrix.get_effort("x", "U") == matrix.get_effort(None, "U")
        assert matrix.get_effort("R", "y") == 0.0


class TestCalibratedOHMatrixInvariants:
    """Verification suite for One-Handed calibrated transition matrix invariants."""

    def test_left_m_slice_adaptation_does_not_affect_oh(self):
        matrix_oh_right = TransitionMatrix.load(profile=HandProfile(solving_mode="OH", m_slice_hand="right"))
        matrix_oh_left = TransitionMatrix.load(profile=HandProfile(solving_mode="OH", m_slice_hand="left"))
        assert matrix_oh_left.transitions == matrix_oh_right.transitions

    def test_calibrated_oh_matrix_lse_m_u_invariants(self):
        matrix_oh = TransitionMatrix.load_oh()
        # AUF pause corrected in OH (should be ~1.33, not 2.39)
        assert matrix_oh["U'M'"] < 1.50
        # Index pull faster than push in OH
        assert matrix_oh["U'M'"] < matrix_oh["UM'"]
        assert matrix_oh["M2U'"] < matrix_oh["M2U"]
        # Sensor clamp floor corrected in OH (should be ~1.45, not 0.85)
        assert matrix_oh["U2M'"] > 1.20







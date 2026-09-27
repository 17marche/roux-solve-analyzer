"""Tests for GripTracker kinematic transitions, contortions, and optimal regrip minimization."""

import pytest
from roux_engine.ergonomics.models import GripState
from roux_engine.ergonomics.grip_tracker import GripTracker


class TestGripTrackerTransitionsAndContortions:
    """Seam 2.1: Single move transitions and impossible contortions."""

    def test_r_family_transitions_from_home(self):
        tracker = GripTracker()
        assert tracker.is_valid_turn("R", GripState.HOME)
        assert tracker.get_next_grip("R", GripState.HOME) == GripState.R_AWAY

        assert tracker.is_valid_turn("R'", GripState.HOME)
        assert tracker.get_next_grip("R'", GripState.HOME) == GripState.R_PRIME_AWAY

        # R2 from HOME requires anticipatory regrip and is blocked from HOME directly
        assert not tracker.is_valid_turn("R2", GripState.HOME)
        assert not tracker.is_valid_turn("r2", GripState.HOME)

        # Wide r turns follow the same kinematics
        assert tracker.get_next_grip("r", GripState.HOME) == GripState.R_AWAY
        assert tracker.get_next_grip("r'", GripState.HOME) == GripState.R_PRIME_AWAY

    def test_r_family_transitions_from_r_away(self):
        tracker = GripTracker()
        # Continuing R or r from R_AWAY past +90° is an anatomical limit
        assert not tracker.is_valid_turn("R", GripState.R_AWAY)
        assert not tracker.is_valid_turn("r", GripState.R_AWAY)

        # R' and r' from R_AWAY returns to HOME
        assert tracker.is_valid_turn("R'", GripState.R_AWAY)
        assert tracker.get_next_grip("R'", GripState.R_AWAY) == GripState.HOME
        assert tracker.is_valid_turn("r'", GripState.R_AWAY)
        assert tracker.get_next_grip("r'", GripState.R_AWAY) == GripState.HOME

        # R2 and r2 from R_AWAY moves to R_PRIME_AWAY
        assert tracker.is_valid_turn("R2", GripState.R_AWAY)
        assert tracker.get_next_grip("R2", GripState.R_AWAY) == GripState.R_PRIME_AWAY
        assert tracker.is_valid_turn("r2", GripState.R_AWAY)
        assert tracker.get_next_grip("r2", GripState.R_AWAY) == GripState.R_PRIME_AWAY

    def test_r_family_transitions_from_r_prime_away(self):
        tracker = GripTracker()
        # Continuing R' or r' from R_PRIME_AWAY past -90° is an anatomical limit
        assert not tracker.is_valid_turn("R'", GripState.R_PRIME_AWAY)
        assert not tracker.is_valid_turn("r'", GripState.R_PRIME_AWAY)

        # R and r from R_PRIME_AWAY returns to HOME
        assert tracker.is_valid_turn("R", GripState.R_PRIME_AWAY)
        assert tracker.get_next_grip("R", GripState.R_PRIME_AWAY) == GripState.HOME
        assert tracker.is_valid_turn("r", GripState.R_PRIME_AWAY)
        assert tracker.get_next_grip("r", GripState.R_PRIME_AWAY) == GripState.HOME

        # R2 and r2 from R_PRIME_AWAY moves to R_AWAY
        assert tracker.is_valid_turn("R2", GripState.R_PRIME_AWAY)
        assert tracker.get_next_grip("R2", GripState.R_PRIME_AWAY) == GripState.R_AWAY
        assert tracker.is_valid_turn("r2", GripState.R_PRIME_AWAY)
        assert tracker.get_next_grip("r2", GripState.R_PRIME_AWAY) == GripState.R_AWAY

    def test_blocked_moves_from_home(self):
        tracker = GripTracker()
        # B moves, F2, and R2/r2 are blocked from HOME without regrip
        assert not tracker.is_valid_turn("B", GripState.HOME)
        assert not tracker.is_valid_turn("B'", GripState.HOME)
        assert not tracker.is_valid_turn("B2", GripState.HOME)
        assert not tracker.is_valid_turn("F2", GripState.HOME)
        assert not tracker.is_valid_turn("R2", GripState.HOME)
        assert not tracker.is_valid_turn("r2", GripState.HOME)

    def test_unblocked_moves_in_non_home_grips(self):
        tracker = GripTracker()
        # In R_AWAY, F moves and left-hand U moves are unblocked
        assert tracker.is_valid_turn("F", GripState.R_AWAY)
        assert tracker.is_valid_turn("F'", GripState.R_AWAY)
        assert tracker.is_valid_turn("U", GripState.R_AWAY)
        assert tracker.is_valid_turn("U'", GripState.R_AWAY)
        assert tracker.is_valid_turn("D", GripState.R_AWAY)

        # In R_PRIME_AWAY, F and U moves (including left index U/U') are unblocked
        assert tracker.is_valid_turn("F", GripState.R_PRIME_AWAY)
        assert tracker.is_valid_turn("F'", GripState.R_PRIME_AWAY)
        assert tracker.is_valid_turn("U", GripState.R_PRIME_AWAY)
        assert tracker.is_valid_turn("U'", GripState.R_PRIME_AWAY)
        assert tracker.is_valid_turn("D", GripState.R_PRIME_AWAY)

    def test_non_r_turns_preserve_grip_state(self):
        tracker = GripTracker()
        # Valid non-R turns maintain the current grip state
        assert tracker.is_valid_turn("U", GripState.HOME)
        assert tracker.get_next_grip("U", GripState.HOME) == GripState.HOME

        assert tracker.is_valid_turn("U'", GripState.R_AWAY)
        assert tracker.get_next_grip("U'", GripState.R_AWAY) == GripState.R_AWAY

        assert tracker.is_valid_turn("F", GripState.R_PRIME_AWAY)
        assert tracker.get_next_grip("F", GripState.R_PRIME_AWAY) == GripState.R_PRIME_AWAY

        assert tracker.is_valid_turn("M'", GripState.HOME)
        assert tracker.get_next_grip("M'", GripState.HOME) == GripState.HOME


class TestGripTrackerDPMinimization:
    """Seam 2.2: Dynamic programming optimal regrip minimization."""

    def test_empty_sequence_evaluates_to_zero_regrips(self):
        tracker = GripTracker()
        res = tracker.track("")
        assert res.regrip_count == 0
        assert len(res.steps) == 0
        assert res.initial_grip == GripState.HOME
        assert res.final_grip == GripState.HOME

    def test_pure_home_grip_sequence_has_zero_regrips(self):
        tracker = GripTracker()
        # Sexy move: R U R' U' flows naturally with 0 regrips
        res = tracker.track("R U R' U'")
        assert res.regrip_count == 0
        assert len(res.steps) == 4
        # Check step by step
        # Step 0: R executed from HOME -> ends in R_AWAY, no regrip
        assert res.steps[0].move == "R"
        assert res.steps[0].grip_during == GripState.HOME
        assert res.steps[0].grip_after == GripState.R_AWAY
        assert not res.steps[0].regrip

        # Step 1: U executed from R_AWAY -> ends in R_AWAY, no regrip
        assert res.steps[1].move == "U"
        assert res.steps[1].grip_during == GripState.R_AWAY
        assert res.steps[1].grip_after == GripState.R_AWAY
        assert not res.steps[1].regrip

        # Step 2: R' executed from R_AWAY -> ends in HOME, no regrip
        assert res.steps[2].move == "R'"
        assert res.steps[2].grip_during == GripState.R_AWAY
        assert res.steps[2].grip_after == GripState.HOME
        assert not res.steps[2].regrip

        # Step 3: U' executed from HOME -> ends in HOME, no regrip
        assert res.steps[3].move == "U'"
        assert res.steps[3].grip_during == GripState.HOME
        assert res.steps[3].grip_after == GripState.HOME
        assert not res.steps[3].regrip

        assert res.final_grip == GripState.HOME

    def test_roux_rotationless_blockbuilding_flow_has_zero_regrips(self):
        tracker = GripTracker()
        # Common Roux trigger: r U R' U' r' R U
        res = tracker.track("r U R' U' r' R U")
        assert res.regrip_count == 0

    def test_fluid_r_f_in_r_away(self):
        tracker = GripTracker()
        # R puts hand in R_AWAY. F is unblocked in R_AWAY (flicked like a D turn from home).
        res = tracker.track("R F")
        assert res.regrip_count == 0
        assert len(res.steps) == 2
        assert not res.steps[0].regrip
        assert not res.steps[1].regrip

    def test_r2_from_home_forces_anticipatory_regrip(self):
        tracker = GripTracker()
        # R2 from HOME cannot be done directly; solver does an anticipatory regrip
        res = tracker.track("R2")
        assert res.regrip_count == 1
        assert len(res.steps) == 1
        assert res.steps[0].regrip
        assert res.steps[0].grip_before == GripState.HOME
        assert res.steps[0].grip_during in (GripState.R_AWAY, GripState.R_PRIME_AWAY)

    def test_double_r_forces_regrip_due_to_anatomical_dead_end(self):
        tracker = GripTracker()
        # R -> R_AWAY (+90°). Second R is an anatomical dead end, forcing a regrip
        res = tracker.track("R R")
        assert res.regrip_count == 1
        assert not res.steps[0].regrip
        assert res.steps[1].regrip

    def test_double_r_prime_forces_regrip_due_to_anatomical_dead_end(self):
        tracker = GripTracker()
        # R' -> R_PRIME_AWAY (-90°). Second R' is an anatomical dead end, forcing a regrip
        res = tracker.track("R' R'")
        assert res.regrip_count == 1
        assert not res.steps[0].regrip
        assert res.steps[1].regrip

    def test_fluid_r_prime_u_moves(self):
        tracker = GripTracker()
        # R' puts right hand in R_PRIME_AWAY. Both U and U' are executed with 0 regrips
        res_u = tracker.track("R' U")
        assert res_u.regrip_count == 0
        res_u_prime = tracker.track("R' U'")
        assert res_u_prime.regrip_count == 0

    def test_custom_initial_grip(self):
        tracker = GripTracker()
        # If solver starts already at R_AWAY, an R' brings them to HOME without regrip
        res = tracker.track("R'", initial_grip=GripState.R_AWAY)
        assert res.regrip_count == 0
        assert res.final_grip == GripState.HOME

    def test_m_slice_lse_flow_preserves_home_grip(self):
        tracker = GripTracker()
        # Standard Roux LSE moves: M' U M U' M2
        res = tracker.track("M' U M U' M2")
        assert res.regrip_count == 0
        assert res.final_grip == GripState.HOME

    def test_initial_regrip_when_first_move_impossible_from_start_grip(self):
        tracker = GripTracker()
        # B is blocked from HOME. Solver must regrip immediately before move 0.
        res = tracker.track("B", initial_grip=GripState.HOME)
        assert res.regrip_count == 1
        assert len(res.steps) == 1
        assert res.steps[0].regrip
        assert res.steps[0].grip_before == GripState.HOME
        assert res.steps[0].grip_during in (GripState.R_AWAY, GripState.R_PRIME_AWAY)




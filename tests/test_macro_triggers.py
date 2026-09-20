"""Tests for Closed-Loop Macro Triggers in ergonomics and SB solver."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.orientation import get_dual_neutral_orientations, get_orientation
from roux_engine.ergonomics.macro_triggers import (
    MacroTrigger,
    SLEDGEHAMMER,
    HEDGE,
    MACRO_TRIGGERS,
    verify_first_block_preservation,
)


class TestMacroTriggerRegistryAndInvariant:
    """Seam 1: MacroTrigger Registry & FB Invariant Verification."""

    def test_macro_trigger_registry_definitions(self):
        assert "sledgehammer" in MACRO_TRIGGERS
        assert "hedge" in MACRO_TRIGGERS

        sledge = MACRO_TRIGGERS["sledgehammer"]
        assert sledge.name == "sledgehammer"
        assert sledge.moves == ("R'", "F", "R", "F'")
        assert sledge.stm == 4

        hedge = MACRO_TRIGGERS["hedge"]
        assert hedge.name == "hedge"
        assert hedge.moves == ("F", "R'", "F'", "R")
        assert hedge.stm == 4

        assert SLEDGEHAMMER == sledge
        assert HEDGE == hedge

    def test_verify_first_block_preservation_for_sledgehammer_and_hedge(self):
        # Must preserve First Block across all dual-neutral orientations
        assert verify_first_block_preservation(SLEDGEHAMMER) is True
        assert verify_first_block_preservation(HEDGE) is True

        for ori in get_dual_neutral_orientations():
            assert verify_first_block_preservation(SLEDGEHAMMER, orientation=ori) is True
            assert verify_first_block_preservation(HEDGE, orientation=ori) is True

    def test_verify_first_block_preservation_rejects_non_preserving_triggers(self):
        # An invalid trigger that disturbs First Block
        bad_trigger = MacroTrigger(name="bad_trigger", moves=("R", "U", "R'", "F"))
        assert verify_first_block_preservation(bad_trigger) is False

        disturbing_trigger = MacroTrigger(name="disturbing", moves=("L", "U", "L'", "U'"))
        assert verify_first_block_preservation(disturbing_trigger) is False


        # Single F turn disturbs FL and DFL
        single_f = MacroTrigger(name="single_f", moves=("F",))
        assert verify_first_block_preservation(single_f) is False


class TestMacroTriggerChunkingAndErgonomics:
    """Seam 2: Chunking Pattern Matcher & Ergonomic Evaluation."""

    def test_match_macro_triggers_sledgehammer_in_sequence(self):
        from roux_engine.ergonomics.macro_triggers import match_macro_triggers, MacroTriggerMatch

        matches = match_macro_triggers("U R' F R F' U2")
        assert len(matches) == 1
        m = matches[0]
        assert m.trigger == SLEDGEHAMMER
        assert m.start_index == 1
        assert m.end_index == 5
        assert m.moves == ("R'", "F", "R", "F'")

    def test_match_macro_triggers_hedge_in_sequence(self):
        from roux_engine.ergonomics.macro_triggers import match_macro_triggers

        matches = match_macro_triggers("F R' F' R U")
        assert len(matches) == 1
        m = matches[0]
        assert m.trigger == HEDGE
        assert m.start_index == 0
        assert m.end_index == 4
        assert m.moves == ("F", "R'", "F'", "R")

    def test_match_multiple_macro_triggers(self):
        from roux_engine.ergonomics.macro_triggers import match_macro_triggers

        matches = match_macro_triggers("R' F R F' U F R' F' R")
        assert len(matches) == 2
        assert matches[0].trigger == SLEDGEHAMMER
        assert matches[1].trigger == HEDGE

    def test_match_no_triggers(self):
        from roux_engine.ergonomics.macro_triggers import match_macro_triggers

        matches = match_macro_triggers("R U R' U' M2 U M U2 M'")
        assert len(matches) == 0

    def test_flow_score_captures_macro_triggers_with_zero_regrips(self):
        from roux_engine.ergonomics.flow_scorer import FlowScorer

        scorer = FlowScorer()
        score = scorer.score_moves("U R' F R F' U2")
        assert score.macro_triggers == ["sledgehammer"]
        assert score.regrip_count == 0
        # The moves within the macro trigger (indices 1..4) should have 0 regrips
        for step in score.per_move_analysis[1:5]:
            assert step.regrip is False

    def test_match_macro_triggers_contiguous_only(self):
        from roux_engine.ergonomics.macro_triggers import match_macro_triggers

        # A rotation in the middle breaks the contiguous 4-STM trigger
        matches = match_macro_triggers("R' F y R F'")
        assert len(matches) == 0

    def test_evaluate_macro_triggers(self):
        from roux_engine.ergonomics.macro_triggers import evaluate_macro_triggers

        matches, regrips = evaluate_macro_triggers("U R' F R F' U2")
        assert len(matches) == 1
        assert matches[0].trigger == SLEDGEHAMMER
        assert regrips == 0


class TestMacroTriggersSBSolver:
    """Seam 3: Second Block Solver Support (allow_macro_triggers=True)."""

    def test_default_sb_solver_does_not_use_macro_triggers(self):
        from roux_engine.solver.sb_solver import solve_sb

        c = CubeState().apply_moves("R2 R' F R F'")
        sols = solve_sb(c, k=3, style="free")
        assert len(sols) > 0
        for s in sols:
            assert "F" not in s.moves
            assert "F'" not in s.moves

    def test_sb_solver_discovers_hedge_and_preserves_fb(self):
        from roux_engine.solver.sb_solver import solve_sb

        ori = get_orientation("")
        # Scramble where pure <R, U, r, M> requires 7 moves, while Hedge + R2 solves in 5 moves
        c = CubeState().apply_moves("R2 R' F R F'")
        assert ori.is_fb_solved(c)

        sols = solve_sb(c, k=3, style="free", allow_macro_triggers=True)
        assert len(sols) > 0

        # At least one solution should use the macro trigger
        macro_sols = [s for s in sols if "F" in s.moves or "F'" in s.moves]
        assert len(macro_sols) > 0

        best_macro_sol = macro_sols[0]
        # Macro solution move count should be <= 5 (beats pure 7 moves)
        assert best_macro_sol.move_count <= 5

        # Verify FB is 100% intact after applying solution
        c_after = c.copy().apply_moves(" ".join(best_macro_sol.moves))
        assert ori.is_fb_solved(c_after)
        assert ori.is_center_aligned_sb_solved(c_after)

    def test_sb_solver_discovers_sledgehammer_beating_pure_moves(self):
        from roux_engine.solver.sb_solver import solve_sb

        ori = get_orientation("")
        # Scramble where pure <R, U, r, M> requires 9 moves
        c = CubeState().apply_moves("R2 F R' F' R")
        assert ori.is_fb_solved(c)

        sols_pure = solve_sb(c, k=1, style="free", allow_macro_triggers=False)
        assert sols_pure[0].move_count >= 7

        sols_macro = solve_sb(c, k=3, style="free", allow_macro_triggers=True)
        assert len(sols_macro) > 0

        best_macro = sols_macro[0]
        assert best_macro.move_count < sols_pure[0].move_count

        c_after = c.copy().apply_moves(" ".join(best_macro.moves))
        assert ori.is_fb_solved(c_after)
        assert ori.is_center_aligned_sb_solved(c_after)

"""Tests for Top-K Ergonomic Re-Ranking in FBSolver."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.ergonomics.flow_scorer import FlowScorer
from roux_engine.solver.fb_solver import FBSolution, FBSolver, solve_fb


class TestFBTopKErgonomicRanking:
    """Seam 1: FBSolver Top-K Candidate Search with rank_by='e_stm' | 'stm'."""

    def test_fb_solution_has_optional_e_stm(self):
        sol = FBSolution(
            moves=["R", "U"],
            move_count=2,
            orientation="WHITE-ORANGE",
            inspection_rotation="",
            e_stm=2.5,
        )
        assert sol.e_stm == 2.5

        sol_default = FBSolution(
            moves=["R", "U"],
            move_count=2,
            orientation="WHITE-ORANGE",
            inspection_rotation="",
        )
        assert sol_default.e_stm is None

    def test_solve_fb_accepts_rank_by_and_top_k(self):
        # A simple 2-move scramble
        scramble = "U' R'"
        solutions_stm = solve_fb(scramble, top_k=3, rank_by="stm")
        assert len(solutions_stm) > 0
        assert solutions_stm[0].move_count <= 2

        solutions_estm = solve_fb(scramble, top_k=3, rank_by="e_stm")
        assert len(solutions_estm) > 0
        assert solutions_estm[0].e_stm is not None
        # Must be sorted by e_stm ascending
        for i in range(len(solutions_estm) - 1):
            assert solutions_estm[i].e_stm <= solutions_estm[i + 1].e_stm

    def test_fbsolver_class_method_solve_fb(self):
        scramble = "U' R'"
        sols = FBSolver.solve_fb(scramble, top_k=2, rank_by="e_stm")
        assert len(sols) > 0
        assert sols[0].e_stm is not None

    def test_ergonomic_line_preferred_over_awkward_line(self):
        """Verify that an ergonomic 6-move line is preferred over an awkward 6-move line."""
        scorer = FlowScorer()
        ergo_moves = ["r", "U", "R'", "U2", "R'", "U'"]
        awkward_moves = ["B'", "D2", "F'", "R2", "D'", "F"]

        score_ergo = scorer.score_moves(ergo_moves)
        score_awkward = scorer.score_moves(awkward_moves)

        assert score_ergo.raw_stm == 6
        assert score_awkward.raw_stm == 6
        assert score_ergo.regrip_count == 0
        assert score_awkward.regrip_count >= 1
        assert score_ergo.e_stm < score_awkward.e_stm

        sol_ergo = FBSolution(
            moves=ergo_moves,
            move_count=6,
            orientation="YELLOW-BLUE",
            inspection_rotation="",
            e_stm=score_ergo.e_stm,
        )
        sol_awkward = FBSolution(
            moves=awkward_moves,
            move_count=6,
            orientation="YELLOW-BLUE",
            inspection_rotation="",
            e_stm=score_awkward.e_stm,
        )

        ranked = sorted([sol_awkward, sol_ergo], key=lambda s: (s.e_stm, s.move_count))
        assert ranked[0] == sol_ergo
        assert ranked[1] == sol_awkward

    def test_solve_fb_ranks_candidates_by_e_stm(self):
        # Scramble that has multiple optimal or near-optimal solutions
        scramble = "F R U R' U' F' B D"
        sols = solve_fb(scramble, top_k=5, rank_by="e_stm")
        assert len(sols) >= 2
        for s in sols:
            assert s.e_stm is not None
        # Assert monotonic order
        for i in range(len(sols) - 1):
            assert sols[i].e_stm <= sols[i + 1].e_stm

    def test_solve_fb_prefers_lower_e_stm_candidate(self):
        # Scramble with multiple FB solutions
        scramble = "B D F R"
        sols_estm = solve_fb(scramble, top_k=5, rank_by="e_stm")
        assert len(sols_estm) >= 2
        assert sols_estm[0].e_stm is not None
        assert all(sols_estm[0].e_stm <= s.e_stm for s in sols_estm if s.e_stm is not None)

    def test_solve_fb_expands_to_depth_l_plus_2(self):
        # A scramble that triggers multi-depth candidate collection up to L+2
        scramble = "D F2 U B2"
        sols = solve_fb(scramble, top_k=10, rank_by="e_stm")
        assert len(sols) > 0
        for s in sols:
            assert s.e_stm is not None


class TestSBTopKErgonomicRanking:
    """Seam 5: SBSolver Top-K Candidate Search with rank_by='e_stm' | 'stm'."""

    def test_sb_solution_has_optional_e_stm(self):
        from roux_engine.solver.sb_solver import SBSolution

        sol = SBSolution(
            moves=("R", "U", "R'"),
            move_count=3,
            e_stm=2.1,
        )
        assert sol.e_stm == 2.1

        sol_default = SBSolution(
            moves=("R", "U", "R'"),
            move_count=3,
        )
        assert sol_default.e_stm is None

    def test_solve_sb_accepts_rank_by_and_top_k(self):
        from roux_engine.solver.sb_solver import solve_sb

        c = CubeState().apply_moves("R U' R2 U r U' r' M2")
        solutions_stm = solve_sb(c, top_k=3, rank_by="stm")
        assert len(solutions_stm) > 0

        solutions_estm = solve_sb(c, top_k=3, rank_by="e_stm")
        assert len(solutions_estm) > 0
        assert solutions_estm[0].e_stm is not None
        # Monotonic order by e_stm
        for i in range(len(solutions_estm) - 1):
            assert solutions_estm[i].e_stm <= solutions_estm[i + 1].e_stm

    def test_sbsolver_class_method_solve_sb(self):
        from roux_engine.solver.sb_solver import SBSolver

        c = CubeState().apply_moves("r U r' U' R U2 R'")
        sols = SBSolver.solve_sb(c, top_k=2, rank_by="e_stm")
        assert len(sols) > 0
        assert sols[0].e_stm is not None

    def test_solve_sb_with_different_styles_respects_rank_by(self):
        from roux_engine.solver.sb_solver import solve_sb

        c = CubeState().apply_moves("R U' R2 U r U' r' M2")
        for style in ("free", "classical", "square_pair", "all"):
            sols = solve_sb(c, top_k=3, style=style, rank_by="e_stm")
            assert len(sols) > 0
            for s in sols:
                assert s.e_stm is not None
            for i in range(len(sols) - 1):
                assert sols[i].e_stm <= sols[i + 1].e_stm



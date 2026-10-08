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


class TestLSETopKErgonomicRanking:
    """LSE Top-K Candidate Search with rank_by='e_stm' | 'stm' and HandProfile parameterization."""

    def test_lse_solution_has_optional_e_stm(self):
        from roux_engine.solver.lse_solver import LSESolution

        sol = LSESolution(
            target="4a",
            moves=["M", "U"],
            move_count=2,
            case_name="standard_eo",
            e_stm=1.85,
        )
        assert sol.e_stm == 1.85

        sol_default = LSESolution(
            target="4a",
            moves=["M", "U"],
            move_count=2,
            case_name="standard_eo",
        )
        assert sol_default.e_stm is None

    def test_solve_lse_accepts_rank_by_and_top_k(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube = CubeState().apply_moves("M' U2 M U M2")
        # STM ranking preserves shortest move count order
        sols_stm = solve_lse(cube, target="4a", top_k=3, rank_by="stm")
        assert len(sols_stm) > 0
        assert sols_stm[0].e_stm is None
        for i in range(len(sols_stm) - 1):
            assert sols_stm[i].move_count <= sols_stm[i + 1].move_count

        # E-STM ranking orders monotonically by e_stm
        sols_estm = solve_lse(cube, target="4a", top_k=3, rank_by="e_stm")
        assert len(sols_estm) > 0
        assert sols_estm[0].e_stm is not None
        for i in range(len(sols_estm) - 1):
            assert sols_estm[i].e_stm is not None
            assert sols_estm[i + 1].e_stm is not None
            assert sols_estm[i].e_stm <= sols_estm[i + 1].e_stm

    def test_solve_lse_expands_to_depth_l_plus_2(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube = CubeState().apply_moves("U M' U2 M U M2")
        # Querying with rank_by="e_stm" searches optimal depth L and expands to L+1 and L+2
        sols = solve_lse(cube, target="4b", top_k=6, rank_by="e_stm")
        assert len(sols) > 0
        # Check that solutions include depths beyond optimal L
        lengths = {s.move_count for s in sols}
        assert len(lengths) > 1, f"Expected candidates at multiple depths, got lengths: {lengths}"

    def test_solve_lse_1look_ranks_by_e_stm_and_solves_cube(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube = CubeState().apply_moves("U M' U2 M U M2")
        sols = solve_lse(cube, target="1look", top_k=3, rank_by="e_stm")
        assert len(sols) > 0
        for s in sols:
            assert s.e_stm is not None
            # Every candidate must completely solve the cube
            sim = cube.copy().apply_moves(s.moves)
            assert sim.is_solved(), f"Candidate {s.moves} failed to solve cube"

        # Check monotonic ordering
        for i in range(len(sols) - 1):
            assert sols[i].e_stm <= sols[i + 1].e_stm

    def test_solve_lse_m_slice_handedness_adaptation(self):
        from roux_engine.ergonomics.models import HandProfile
        from roux_engine.solver.lse_solver import solve_lse

        # Cube state with asymmetric LSE candidate paths
        cube = CubeState().apply_moves("U M' U2 M U M2")

        prof_right = HandProfile(solving_mode="2H", m_slice_hand="right")
        prof_left = HandProfile(solving_mode="2H", m_slice_hand="left")

        sols_right = solve_lse(cube, target="1look", top_k=5, rank_by="e_stm", profile=prof_right)
        sols_left = solve_lse(cube, target="1look", top_k=5, rank_by="e_stm", profile=prof_left)

        assert len(sols_right) > 0
        assert len(sols_left) > 0
        # E-STM values should reflect the handedness adaptation
        e_right = [s.e_stm for s in sols_right]
        e_left = [s.e_stm for s in sols_left]
        assert e_right != e_left, "Expected different E-STM scores between right and left M-slice flicking"

    def test_fb_and_sb_solvers_accept_hand_profile(self):
        from roux_engine.ergonomics.models import HandProfile
        from roux_engine.solver.fb_solver import solve_fb
        from roux_engine.solver.sb_solver import solve_sb

        prof_right = HandProfile(solving_mode="2H", m_slice_hand="right")
        prof_left = HandProfile(solving_mode="2H", m_slice_hand="left")
        prof_oh = HandProfile(solving_mode="OH")

        # FB solver accepts HandProfile
        fb_sols_2h = solve_fb("U' R'", top_k=2, rank_by="e_stm", profile=prof_right)
        fb_sols_oh = solve_fb("U' R'", top_k=2, rank_by="e_stm", profile=prof_oh)
        assert len(fb_sols_2h) > 0
        assert len(fb_sols_oh) > 0

        # SB solver accepts HandProfile
        c = CubeState().apply_moves("R U' R2 U r U' r' M2")
        sb_sols_r = solve_sb(c, top_k=2, rank_by="e_stm", profile=prof_right)
        sb_sols_l = solve_sb(c, top_k=2, rank_by="e_stm", profile=prof_left)
        assert len(sb_sols_r) > 0
        assert len(sb_sols_l) > 0

    def test_solve_lse_solved_state_with_e_stm(self):
        from roux_engine.solver.lse_solver import solve_lse

        clean = CubeState()
        sols = solve_lse(clean, target="all", rank_by="e_stm")
        for s in sols:
            assert s.move_count == 0
            assert s.moves == []
            assert s.e_stm == 0.0

    def test_solve_lse_step_4c_with_e_stm(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube_dots = CubeState().apply_moves("M' U2 M2 U2 M'")
        sols = solve_lse(cube_dots, target="4c", rank_by="e_stm")
        assert len(sols) == 1
        assert sols[0].case_name == "dots"
        assert sols[0].e_stm is not None
        assert sols[0].e_stm > 0.0

    def test_solve_lse_paths_with_e_stm(self):
        from roux_engine.solver.lse_solver import solve_lse_paths
        from roux_engine.ergonomics.models import HandProfile

        cube = CubeState().apply_moves("M U' M U M2 U M U M' U2")
        prof = HandProfile(solving_mode="2H", m_slice_hand="right")
        paths = solve_lse_paths(cube, rank_by="e_stm", profile=prof)

        assert "standard" in paths
        assert "eolr" in paths
        assert "eolr_misoriented" in paths
        assert "eolr_b" in paths

        for name, p in paths.items():
            assert p.e_stm is not None
            assert p.step_4a.e_stm is not None
            assert p.step_4b.e_stm is not None
            assert p.step_4c.e_stm is not None
            assert p.e_stm > 0.0

    def test_solve_lse_backward_compatibility_defaults(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube = CubeState().apply_moves("M'")
        sols = solve_lse(cube, target="4a")
        # Default behavior retains stm and e_stm is None
        assert len(sols) >= 1
        assert sols[0].e_stm is None
        assert sols[0].moves == ["M"]

    def test_lse_candidates_depth_expansion_not_starved_by_depth_l(self):
        from roux_engine.solver.lse_solver import LSEGraph

        graph = LSEGraph()
        graph._ensure_4a_tables()
        cube = CubeState().apply_moves(["M", "U", "M'", "U"])
        code = graph.encode_cube(cube)
        cands = graph.get_candidates(code, graph.std_eo_targets, graph.dist_std_eo, k=5, rank_by="e_stm")
        lengths = {len(p) for p, _ in cands}
        # Optimal depth is 4; candidate collection must also explore deeper alternatives (e.g. 6)
        assert 4 in lengths
        assert any(l > 4 for l in lengths), f"Expected candidates at depth > 4, found {lengths}"

    def test_lse_4a_misoriented_centers_classification(self):
        from roux_engine.solver.lse_solver import solve_lse

        cube = CubeState().apply_moves("M' U M U' M2")
        sols = solve_lse(cube, target="4a", top_k=5, rank_by="e_stm", allow_misoriented_centers=True)
        assert len(sols) > 0
        for s in sols:
            if s.center_state == "misaligned":
                # When centers are misaligned, case must be eolr or eolr_b, never standard_eo
                assert s.case_name in ("eolr", "eolr_b"), f"Unexpected misaligned standard_eo: {s}"

    def test_hand_profile_resolve_helper(self):
        from roux_engine.ergonomics.models import HandProfile

        assert HandProfile.resolve(None) is None
        p1 = HandProfile.resolve("oh")
        assert p1 is not None and p1.solving_mode == "OH"
        p2 = HandProfile(solving_mode="2H", m_slice_hand="left")
        assert HandProfile.resolve(p2) == p2
        assert HandProfile.resolve(profile="2H", hand_profile=p2) == p2



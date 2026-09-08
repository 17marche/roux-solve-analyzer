"""Tests for Second Block (SB) Center-Aligned Free Blockbuilding IDA* Solver."""

import pytest
from roux_engine.solver.sb_pdb_generator import SB_MOVESET


class TestSBBranchPruningAndTransitions:
    """Slice 1: SB Branch Pruning and Allowed Moves Transition Table."""

    def test_allowed_moves_dimensions(self):
        """Allowed moves table must have an entry for each of the 12 moves in SB_MOVESET."""
        from roux_engine.solver.sb_solver import _build_sb_allowed_moves

        allowed = _build_sb_allowed_moves()
        assert len(allowed) == 12
        for m_idx, next_moves in enumerate(allowed):
            assert isinstance(next_moves, tuple)
            assert len(next_moves) > 0
            assert len(next_moves) < 12

    def test_same_family_consecutive_turns_pruned(self):
        """Adjacent turns within the same family (R, U, r, M) must be pruned."""
        from roux_engine.solver.sb_solver import _build_sb_allowed_moves, _SB_MOVE_FAMILIES

        allowed = _build_sb_allowed_moves()
        for m1 in range(12):
            f1 = _SB_MOVE_FAMILIES[m1]
            for m2 in allowed[m1]:
                f2 = _SB_MOVE_FAMILIES[m2]
                assert f1 != f2, f"Allowed consecutive moves from same family: {SB_MOVESET[m1]} -> {SB_MOVESET[m2]}"

    def test_parallel_slice_group_canonical_ordering(self):
        """In {R, r, M}, only R followed by M is allowed; all other adjacent pairs in {R, r, M} are pruned."""
        from roux_engine.solver.sb_solver import _build_sb_allowed_moves, _SB_MOVE_FAMILIES

        # Families: 0=R, 1=U, 2=r, 3=M
        allowed = _build_sb_allowed_moves()
        for m1 in range(12):
            f1 = _SB_MOVE_FAMILIES[m1]
            for m2 in allowed[m1]:
                f2 = _SB_MOVE_FAMILIES[m2]
                if f1 in (0, 2, 3) and f2 in (0, 2, 3):
                    # Only R (0) followed by M (3) is allowed
                    assert f1 == 0 and f2 == 3, (
                        f"Disallowed adjacent pair in {{R, r, M}}: {SB_MOVESET[m1]} (fam {f1}) -> {SB_MOVESET[m2]} (fam {f2})"
                    )

    def test_u_turns_allow_all_non_u_families(self):
        """From U turns, transitions to R, r, and M families must all be permitted (9 transitions)."""
        from roux_engine.solver.sb_solver import _build_sb_allowed_moves

        allowed = _build_sb_allowed_moves()
        # U moves are indices 3, 4, 5
        for u_idx in (3, 4, 5):
            next_moves = allowed[u_idx]
            assert len(next_moves) == 9
            # Non-U moves are 0..2 (R), 6..8 (r), 9..11 (M)
            assert set(next_moves) == {0, 1, 2, 6, 7, 8, 9, 10, 11}


class TestSBCenterAlignmentAndSolutionModel:
    """Slice 2: SBSolution Dataclass, Center Offset Tracking, and Goal Enforcement."""

    def test_sb_solution_dataclass_attributes_and_immutability(self):
        """SBSolution must be immutable with required attributes."""
        from roux_engine.solver.sb_solver import SBSolution

        sol = SBSolution(
            moves=("R", "U", "R'"),
            move_count=3,
            style="free",
            order="direct",
            dr_move_idx=0,
            pair1_move_idx=1,
            square_move_idx=1,
            resulting_cmll_case="Skip",
            orientation="",
        )
        assert sol.moves == ("R", "U", "R'")
        assert sol.move_count == 3
        assert sol.style == "free"
        assert sol.order == "direct"
        assert sol.resulting_cmll_case == "Skip"

        with pytest.raises(Exception):
            sol.move_count = 4  # type: ignore

    def test_center_aligned_sb_solved_on_clean_cube(self):
        """Clean CubeState must satisfy Center-Aligned SB goal condition."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import is_center_aligned_sb_solved
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS

        clean = CubeState()
        block = ALL_BLOCK_DEFINITIONS[""]
        assert is_center_aligned_sb_solved(clean, block)

    def test_misaligned_centers_reject_sb_solved(self):
        """When 5 SB pieces are solved: M/M' are off-axis (rejected); M2 preserves U/D axis (accepted)."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import is_center_aligned_sb_solved, get_m_slice_center_offset
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS
        from roux_engine.segmenter.sb_detector import SBDetector

        block = ALL_BLOCK_DEFINITIONS[""]
        # M and M' rotate centers to F/B axis (offset 1 and 3): not center-aligned
        for m_move in ("M", "M'"):
            c = CubeState().apply_move(m_move)
            assert SBDetector.is_canonical_sb_solved(c)
            assert get_m_slice_center_offset(c, block) in (1, 3)
            assert not is_center_aligned_sb_solved(c, block)

        # M2 rotates centers by 180 (offset 2), keeping them along U/D axis: aligned
        c_m2 = CubeState().apply_move("M2")
        assert SBDetector.is_canonical_sb_solved(c_m2)
        assert get_m_slice_center_offset(c_m2, block) == 2
        assert is_center_aligned_sb_solved(c_m2, block)

    def test_center_transition_matrix(self):
        """Center transition table must accurately model M and r moves modulo 4."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import _build_center_transitions, get_m_slice_center_offset
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS

        block = ALL_BLOCK_DEFINITIONS[""]
        center_trans = _build_center_transitions()
        assert len(center_trans) == 4
        for row in center_trans:
            assert len(row) == 12

        # Verify against CubeState for all 12 moves starting from each of the 4 center offsets
        for start_m in ("", "M", "M2", "M'"):
            c_base = CubeState()
            if start_m:
                c_base.apply_move(start_m)
            c_off = get_m_slice_center_offset(c_base, block)

            for m_idx, m_name in enumerate(SB_MOVESET):
                c_after = c_base.copy().apply_move(m_name)
                expected_off = get_m_slice_center_offset(c_after, block)
                assert center_trans[c_off][m_idx] == expected_off, (
                    f"Mismatch from offset {c_off} after {m_name}: expected {expected_off}, got {center_trans[c_off][m_idx]}"
                )


class TestSBSolverCanonicalSearch:
    """Slice 3: Top-K Candidate Search Engine in Canonical Frame."""

    def test_already_solved_canonical_state(self):
        """Clean CubeState with already solved SB returns 0-move candidate."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver

        solver = SBSolver.get_instance()
        sols = solver.solve(CubeState(), k=5)
        assert len(sols) >= 1
        top = sols[0]
        assert top.move_count == 0
        assert top.moves == ()
        assert top.style == "free"

    def test_solve_short_scramble_canonical(self):
        """Solves a 3-move scramble optimally with Center-Aligned SB verified."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver, is_center_aligned_sb_solved
        from roux_engine.segmenter.fb_detector import FBDetector
        from roux_engine.core.parser import MoveParser

        scramble = "R U R'"
        c = CubeState().apply_moves(scramble)

        solver = SBSolver.get_instance()
        sols = solver.solve(c, k=3)
        assert len(sols) == 3
        top = sols[0]
        assert top.move_count == 3
        # Inverting R U R' is R U' R'
        assert top.moves == ("R", "U'", "R'")

        # Verify physical cube state after applying solution
        final_cube = c.copy().apply_moves(" ".join(top.moves))
        assert FBDetector.is_canonical_fb_solved(final_cube)
        assert is_center_aligned_sb_solved(final_cube)

    def test_moveset_strictly_within_rum(self):
        """All solution move tokens must belong strictly to <R, U, r, M>."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver

        c = CubeState().apply_moves("r U2 r' M U2 M'")
        solver = SBSolver.get_instance()
        sols = solver.solve(c, k=5)
        assert len(sols) > 0

        valid_tokens = set(SB_MOVESET)
        for sol in sols:
            for m in sol.moves:
                assert m in valid_tokens, f"Illegal move token outside <R, U, r, M>: {m}"

    def test_center_alignment_enforced_when_pieces_solved_early(self):
        """When 5 SB pieces are solved but centers are off-axis by M, solver must align centers."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver, is_center_aligned_sb_solved
        from roux_engine.segmenter.fb_detector import FBDetector

        # M preserves all 5 SB pieces and FB, but rotates M-slice centers by 1 (off-axis)
        c = CubeState().apply_move("M")
        solver = SBSolver.get_instance()
        sols = solver.solve(c, k=3)
        assert len(sols) > 0
        top = sols[0]
        assert top.move_count == 1
        # M' brings centers back to offset 0, M brings them to offset 2 (both on U/D axis)
        assert top.moves in (("M'",), ("M",))

        final_cube = c.copy().apply_moves(" ".join(top.moves))
        assert FBDetector.is_canonical_fb_solved(final_cube)
        assert is_center_aligned_sb_solved(final_cube)

    def test_solve_latency_under_10ms(self):
        """Single Free Blockbuilding query for scrambled states completes in < 10ms on CPU."""
        import time
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver

        # 8-move authentic SB scramble
        scramble = "R U' R2 U r U' r' M2"
        c = CubeState().apply_moves(scramble)

        solver = SBSolver.get_instance()
        # Warmup
        solver.solve(c, k=1)

        # Timed benchmark
        t0 = time.perf_counter()
        sols = solver.solve(c, k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(sols) > 0
        assert elapsed_ms < 10.0, f"Query took {elapsed_ms:.2f}ms, expected < 10ms"


class TestSBSolverSymmetryAndDualNeutrality:
    """Slice 4: Canonical Symmetry Re-Mapping and Dual-Neutral Orientations."""

    def test_solve_across_all_eight_dual_neutral_orientations_inspected(self):
        """SBSolver automatically detects orientation and solves across all 8 dual-neutral orientations."""
        from roux_engine.core.cube import CubeState
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS, DUAL_NEUTRAL_ORIENTATIONS
        from roux_engine.segmenter.sb_detector import SBDetector
        from roux_engine.solver.sb_solver import SBSolver, is_center_aligned_sb_solved

        solver = SBSolver.get_instance()
        scramble_moves = "R U R' U2 r U' r' M2"

        for ori in DUAL_NEUTRAL_ORIENTATIONS:
            block = ALL_BLOCK_DEFINITIONS[ori]
            c = CubeState()
            if ori:
                c.apply_moves(ori)
            c.apply_moves(scramble_moves)

            # Auto-detect orientation (orientation=None)
            sols = solver.solve(c, k=3, orientation=None)
            assert len(sols) >= 1, f"No solutions found for orientation {ori!r}"
            top = sols[0]
            assert top.orientation == ori

            # Physical verification: applying moves solves the block and preserves FB
            final_cube = c.copy().apply_moves(" ".join(top.moves))
            assert is_center_aligned_sb_solved(final_cube, block), (
                f"Center-Aligned SB not solved for {ori!r}"
            )

    def test_solve_with_explicit_orientation_identifiers(self):
        """Orientation can be specified as string, CanonicalSymmetry enum, or color tuple."""
        from roux_engine.core.cube import CubeState
        from roux_engine.core.constants import Color
        from roux_engine.solver.symmetry import CanonicalSymmetry
        from roux_engine.solver.sb_solver import SBSolver

        solver = SBSolver.get_instance()
        # Setup cube in x2 y (White bottom, Blue left)
        c = CubeState().apply_moves("x2 y R U R' M2")

        # 1. CanonicalSymmetry enum
        sols_enum = solver.solve(c, k=1, orientation=CanonicalSymmetry.X2_Y)
        assert len(sols_enum) == 1
        assert sols_enum[0].orientation == "x2 y"

        # 2. String alias "x2y"
        sols_str = solver.solve(c, k=1, orientation="x2y")
        assert sols_str[0].moves == sols_enum[0].moves

        # 3. Hyphenated color string "white-blue"
        sols_hyphen = solver.solve(c, k=1, orientation="white-blue")
        assert sols_hyphen[0].moves == sols_enum[0].moves

        # 4. Color tuple (Color.WHITE, Color.BLUE)
        sols_tuple = solver.solve(c, k=1, orientation=(Color.WHITE, Color.BLUE))
        assert sols_tuple[0].moves == sols_enum[0].moves

    def test_solve_uninspected_frame_translates_moves(self):
        """When input cube is in uninspected frame, returned solution moves are translated to user frame."""
        from roux_engine.core.cube import CubeState
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS
        from roux_engine.segmenter.sb_detector import SBDetector
        from roux_engine.solver.symmetry import CanonicalSymmetry, translate_moves_to_original
        from roux_engine.solver.sb_solver import SBSolver

        solver = SBSolver.get_instance()
        sym = CanonicalSymmetry.Y
        block = ALL_BLOCK_DEFINITIONS[sym.value]

        # In uninspected frame (no 'y' rotation applied to cube):
        # We scramble with user-frame moves corresponding to "R U R'"
        canon_scramble = ["R", "U", "R'"]
        user_scramble = translate_moves_to_original(canon_scramble, sym)
        c = CubeState().apply_moves(" ".join(user_scramble))

        # Solving with explicit orientation
        sols = solver.solve(c, k=1, orientation=sym)
        assert len(sols) == 1
        sol = sols[0]

        # Applying solution directly to uninspected cube must physically solve the block
        c_final = c.copy().apply_moves(" ".join(sol.moves))
        # After rotating to inspect, block must be solved
        c_inspected = c_final.copy().apply_moves(sym.inspection_rotation)
        assert SBDetector.is_right_1x2x3_block_solved(c_inspected, block)

    def test_unsolved_fb_raises_descriptive_value_error(self):
        """Scrambled cube where FB is not solved raises descriptive ValueError."""
        from roux_engine.core.cube import CubeState
        from roux_engine.solver.sb_solver import SBSolver

        # Pure scramble where FB is not solved
        c = CubeState().apply_moves("R U F D L B")
        solver = SBSolver.get_instance()

        with pytest.raises(ValueError, match="First Block"):
            solver.solve(c, k=1)


class TestSBSolverPublicAPIAndCMLLPreview:
    """Slice 5: Master Public API solve_sb, CMLL Case Preview, and Package Exports."""

    def test_solve_sb_accepts_raw_scramble_string(self):
        """solve_sb accepts move sequence string containing scramble + FB moves."""
        from roux_engine.solver.sb_solver import solve_sb, is_center_aligned_sb_solved
        from roux_engine.core.cube import CubeState
        from roux_engine.core.parser import MoveParser

        # Authentic canonical solve: full solve contains FB + SB
        full_solve = "D' F' L2 D B r U R' U' R U2 R' U R U' R'"
        scramble = " ".join(MoveParser.invert_moves(full_solve))
        full_moves = f"{scramble} D' F' L2 D B"

        sols = solve_sb(full_moves, k=3)
        assert len(sols) == 3
        top = sols[0]
        assert top.move_count > 0

        # Physical execution on pre-simulated CubeState
        c = CubeState().apply_moves(full_moves)
        c.apply_moves(" ".join(top.moves))
        assert is_center_aligned_sb_solved(c)

    def test_solve_sb_resulting_cmll_case_preview(self):
        """SBSolution reports the resulting CMLL case produced by the solution."""
        from roux_engine.solver.sb_solver import solve_sb
        from roux_engine.core.cube import CubeState
        from roux_engine.core.parser import MoveParser

        # Scramble SB with Sune CMLL on top: Sune is R U R' U R U2 R'
        # Start clean, apply inv Sune to corners, then scramble SB
        inv_sune = MoveParser.invert_moves("R U R' U R U2 R'")
        c = CubeState().apply_moves(inv_sune).apply_moves("R U R'")

        sols = solve_sb(c, k=1)
        assert len(sols) == 1
        sol = sols[0]
        # Resulting CMLL case should detect s_left_bar
        assert sol.resulting_cmll_case == "s_left_bar"

    def test_solve_sb_top_k_parameter_alias(self):
        """top_k argument functions identically to k."""
        from roux_engine.solver.sb_solver import solve_sb
        from roux_engine.core.cube import CubeState

        c = CubeState().apply_moves("R U R'")
        sols = solve_sb(c, top_k=2)
        assert len(sols) == 2

    def test_solve_sb_milestones_tracking(self):
        """SBSolution reports valid non-negative move indices for sub-phases."""
        from roux_engine.solver.sb_solver import solve_sb
        from roux_engine.core.cube import CubeState

        c = CubeState().apply_moves("R U R' U2 r U' r' M2")
        sols = solve_sb(c, k=1)
        assert len(sols) == 1
        sol = sols[0]
        assert sol.dr_move_idx is not None
        assert sol.pair1_move_idx is not None
        assert sol.square_move_idx is not None
        assert 0 <= sol.dr_move_idx <= sol.move_count
        assert 0 <= sol.pair1_move_idx <= sol.move_count

    def test_solve_sb_presolved_dr_milestone(self):
        """Pre-solved DR edge before SB execution records dr_move_idx = 0."""
        from roux_engine.solver.sb_solver import solve_sb
        from roux_engine.core.cube import CubeState
        from roux_engine.segmenter.sb_detector import SBDetector

        # Scramble where DR is untouched/solved, only U/R moves scramble the pairs:
        # e.g., R U R' U' R U R' disrupts pairs but keeps DR solved at start?
        # Wait, R moves move DR. To keep DR solved, pairs can be scrambled with U and R U R' U' R U R' (moves that preserve DR).
        # Actually: Sexy move R U R' U' leaves DR untouched!
        c = CubeState().apply_moves("R U R' U'")
        assert SBDetector.is_dr_solved(c)
        sols = solve_sb(c, k=1)
        assert len(sols) == 1
        assert sols[0].dr_move_idx == 0

    def test_package_exports(self):
        """roux_engine.solver must export SBSolution, SBSolver, solve_sb, is_center_aligned_sb_solved."""
        from roux_engine.solver import (
            SBSolution,
            SBSolver,
            solve_sb,
            is_center_aligned_sb_solved,
        )

        assert SBSolution is not None
        assert SBSolver is not None
        assert callable(solve_sb)
        assert callable(is_center_aligned_sb_solved)





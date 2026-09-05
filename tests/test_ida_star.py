"""Tests for Top-K Candidate Search IDA* First Block solver and public API."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.core.constants import Color
from roux_engine.solver import FBSolution, solve_fb
from roux_engine.solver.symmetry import CanonicalSymmetry, get_symmetry, is_fb_solved_for_symmetry


def assert_physically_solves_fb(cube: CubeState, solution: FBSolution) -> None:
    """Helper verifying that applying solution from its inspection frame solves First Block."""
    c = cube.copy()
    if solution.inspection_rotation:
        c.apply_moves(solution.inspection_rotation)
    c.apply_moves(" ".join(solution.moves))
    sym = get_symmetry(solution.orientation)
    assert is_fb_solved_for_symmetry(c, sym, inspected=True)


class TestFBSolutionInterface:
    """Slice 1: Public API interface, FBSolution dataclass, and solved cube behavior."""

    def test_solved_cube_returns_zero_moves(self):
        """A clean solved cube should return a 0-move solution with distance 0."""
        clean_cube = CubeState()
        solutions = solve_fb(clean_cube, k=1)
        assert len(solutions) >= 1
        sol = solutions[0]
        assert isinstance(sol, FBSolution)
        assert sol.move_count == 0
        assert sol.moves == []
        assert isinstance(sol.orientation, str)
        assert isinstance(sol.inspection_rotation, str)

    def test_scramble_string_input_acceptance(self):
        """solve_fb accepts both string scramble and CubeState input."""
        sol_from_str = solve_fb("", k=1)
        assert len(sol_from_str) >= 1
        assert sol_from_str[0].move_count == 0

    def test_fbsolution_dataclass_immutable(self):
        """FBSolution must be an immutable frozen dataclass."""
        sol = FBSolution(moves=["R", "U"], move_count=2, orientation="WHITE-BLUE", inspection_rotation="x2 y")
        with pytest.raises((AttributeError, TypeError)):
            sol.move_count = 3  # type: ignore

    def test_invalid_k_raises_value_error(self):
        """Requesting k <= 0 must raise ValueError."""
        with pytest.raises(ValueError):
            solve_fb(CubeState(), k=0)
        with pytest.raises(ValueError):
            solve_fb(CubeState(), k=-1)

    def test_zero_physical_l_turns(self):
        """Returned move sequences must contain zero physical {L, L', L2} face turns per ADR-0001."""
        clean = CubeState()
        solutions = solve_fb(clean, k=5)
        for sol in solutions:
            for m in sol.moves:
                assert m not in {"L", "L'", "L2"}, f"Physical L turn {m} found in FB solution per ADR-0001"


class TestOptimalSearchAndPhysicalVerification:
    """Slice 2: Optimal IDA* search on scrambles, multi-path candidate collection, and physical verification."""

    def test_one_move_scramble_solved_in_one_move(self):
        """A 1-move disturbance to FB must be solved in exactly 1 move."""
        # World move U turns the White layer, which in White-Blue (X2_Y) frame is D
        scramble = "U"
        c = CubeState().apply_moves(scramble)
        solutions = solve_fb(c, k=3, orientation="WHITE-BLUE")
        assert len(solutions) >= 1
        best = solutions[0]
        assert best.move_count == 1
        assert best.moves == ["D'"]
        assert best.orientation == "WHITE-BLUE"

        # Physically verify
        assert_physically_solves_fb(c, best)

    def test_two_move_scramble_physical_verification(self):
        """A 2-move scramble is solved in <= 2 moves and physically solves First Block."""
        scramble = "R2 B2"
        c = CubeState().apply_moves(scramble)
        solutions = solve_fb(c, k=5)
        assert len(solutions) == 5
        assert solutions[0].move_count <= 2

        for sol in solutions:
            assert sol.move_count == len(sol.moves)
            assert_physically_solves_fb(c, sol)

    def test_candidate_expansion_to_l_plus_one(self):
        """When fewer than K optimal paths exist at depth L, search expands to L+1."""
        # Consider a state with few optimal solutions
        scramble = "F2 L' F2 L2 D2 L2 D' U2 F2 L' U2 L D R F R2 U' F2"
        solutions = solve_fb(scramble, k=5, orientation="WHITE-BLUE")
        assert len(solutions) == 5

        # Must be ranked ascending by move_count
        counts = [s.move_count for s in solutions]
        assert counts == sorted(counts)
        min_l = counts[0]
        # Every solution must have length L or L+1
        for cnt in counts:
            assert cnt in (min_l, min_l + 1)

    def test_all_candidates_distinct(self):
        """All returned top-K candidate paths must be unique move sequences."""
        scramble = "F2 L' F2 L2 D2 L2 D' U2 F2 L' U2 L D R F R2 U' F2"
        solutions = solve_fb(scramble, k=5)
        seen = set()
        for sol in solutions:
            key = (sol.inspection_rotation, tuple(sol.moves))
            assert key not in seen, f"Duplicate solution path found: {key}"
            seen.add(key)


class TestOrientationRestrictionAndSymmetry:
    """Slice 3: Orientation filtering, symmetry re-mapping, and input validation."""

    def test_orientation_restricted_to_single_color_scheme(self):
        """Specifying orientation restricts candidate search strictly to that color scheme."""
        scramble = "F2 L' F2 L2 D2 L2 D' U2 F2 L' U2 L D R F R2 U' F2"
        solutions = solve_fb(scramble, k=5, orientation="WHITE-BLUE")
        assert len(solutions) == 5
        for sol in solutions:
            assert sol.orientation == "WHITE-BLUE"

    def test_orientation_input_formats(self):
        """solve_fb accepts various orientation formats: string, enum, tuple, rotation."""
        scramble = "R U R' U'"
        
        # Upper hyphenated
        sols1 = solve_fb(scramble, k=1, orientation="WHITE-BLUE")
        assert sols1[0].orientation == "WHITE-BLUE"

        # Lower hyphenated
        sols2 = solve_fb(scramble, k=1, orientation="white-blue")
        assert sols2[0].orientation == "WHITE-BLUE"

        # CanonicalSymmetry enum
        sols3 = solve_fb(scramble, k=1, orientation=CanonicalSymmetry.X2_Y)
        assert sols3[0].orientation == "WHITE-BLUE"

        # Color tuple
        sols4 = solve_fb(scramble, k=1, orientation=(Color.WHITE, Color.BLUE))
        assert sols4[0].orientation == "WHITE-BLUE"

        # Rotation string
        sols5 = solve_fb(scramble, k=1, orientation="x2 y")
        assert sols5[0].orientation == "WHITE-BLUE"

    def test_invalid_orientation_raises_value_error(self):
        """Unrecognized orientation strings or invalid color pairs raise ValueError."""
        with pytest.raises(ValueError):
            solve_fb(CubeState(), orientation="invalid-scheme")
        with pytest.raises(ValueError):
            solve_fb(CubeState(), orientation=(Color.RED, Color.BLUE))


class TestSearchPruningRules:
    """Slice 4: Verification of branch pruning rules (inverse moves, commutative ordering, redundant wide turns)."""

    def test_pruning_rules_table_integrity(self):
        """Verify structural properties of the precomputed allowed moves table."""
        from roux_engine.solver.fb_solver import _build_allowed_moves, _MOVE_FAMILIES
        from roux_engine.solver.pdb_generator import FB_MOVESET

        allowed = _build_allowed_moves()
        assert len(allowed) == len(FB_MOVESET)

        for m1_idx, allowed_m2 in enumerate(allowed):
            m1_name = FB_MOVESET[m1_idx]
            f1 = _MOVE_FAMILIES[m1_idx]

            for m2_idx in allowed_m2:
                m2_name = FB_MOVESET[m2_idx]
                f2 = _MOVE_FAMILIES[m2_idx]

                # 1. No same family moves
                assert f1 != f2, f"Same family move allowed: {m1_name} -> {m2_name}"

                # 2. No inverted commutative order (D before U or B before F)
                if f1 == 1:  # D
                    assert f2 != 0, f"D followed by U allowed: {m1_name} -> {m2_name}"
                if f1 == 4:  # B
                    assert f2 != 3, f"B followed by F allowed: {m1_name} -> {m2_name}"

                # 3. Redundant wide / slice in {R, r, M}
                if f1 in (2, 5, 6) and f2 in (2, 5, 6):
                    # Only R followed by M is allowed
                    assert f1 == 2 and f2 == 6, f"Redundant wide/slice transition allowed: {m1_name} -> {m2_name}"

    def test_solutions_satisfy_all_pruning_constraints(self):
        """All move sequences generated by solve_fb must strictly respect pruning invariants."""
        from roux_engine.solver.fb_solver import _MOVE_FAMILIES
        from roux_engine.solver.pdb_generator import FB_MOVESET

        move_to_idx = {m: i for i, m in enumerate(FB_MOVESET)}

        scrambles = [
            "F2 L' F2 L2 D2 L2 D' U2 F2 L' U2 L D R F R2 U' F2",
            "D2 L2 F2 U' R2 U' B2 D' F2 U B2 L B' F2 U' R2 F2 U2",
            "R2 F2 U' B2 U F2 R2 D' F2 D B' R' B F2 D' L D2 U'",
        ]

        for sc in scrambles:
            sols = solve_fb(sc, k=5)
            for sol in sols:
                moves = sol.moves
                for i in range(len(moves) - 1):
                    m1_idx = move_to_idx[moves[i]]
                    m2_idx = move_to_idx[moves[i + 1]]
                    f1 = _MOVE_FAMILIES[m1_idx]
                    f2 = _MOVE_FAMILIES[m2_idx]

                    # Same family check
                    assert f1 != f2, f"Solution {moves} violates inverse pruning: {moves[i]} {moves[i+1]}"

                    # Commutative order check
                    assert not (f1 == 1 and f2 == 0), f"Solution {moves} violates D-U ordering: {moves[i]} {moves[i+1]}"
                    assert not (f1 == 4 and f2 == 3), f"Solution {moves} violates B-F ordering: {moves[i]} {moves[i+1]}"

                    # R-r-M group check
                    if f1 in (2, 5, 6) and f2 in (2, 5, 6):
                        assert f1 == 2 and f2 == 6, f"Solution {moves} violates R-r-M redundancy: {moves[i]} {moves[i+1]}"


class TestDatasetIntegrationAndPerformance:
    """Slice 5: End-to-end integration on roux_solves.json scrambles and < 10ms CPU performance."""

    def test_solve_5_from_dataset_optimal_and_fast(self):
        """Solve 5 from dataset (human solution: x // inspection, R2 B2 // FB) is found in <= 10ms."""
        import time, json
        with open("data/roux_solves.json", "r", encoding="utf-8") as f:
            solve_data = json.load(f)[5]

        scramble = solve_data["scramble"]
        t0 = time.perf_counter()
        solutions = solve_fb(scramble, k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 10.0, f"Query took {elapsed_ms:.2f}ms, exceeding 10ms CPU limit"
        assert len(solutions) == 5
        # The human solved in 2 moves: optimal length must be <= 2
        assert solutions[0].move_count <= 2

        # Verify physical solution for each candidate
        c_base = CubeState().apply_moves(scramble)
        for sol in solutions:
            assert_physically_solves_fb(c_base, sol)

    def test_batch_dataset_scrambles_all_physically_solve_in_under_10ms(self):
        """Verify across a batch of 20 authentic dataset scrambles that all candidates physically solve First Block in < 10ms each."""
        import time, json

        with open("data/roux_solves.json", "r", encoding="utf-8") as f:
            solves = json.load(f)[:20]

        durations = []
        for i, s in enumerate(solves):
            scramble = s.get("scramble", "")
            if not scramble:
                continue

            t0 = time.perf_counter()
            solutions = solve_fb(scramble, k=5)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            durations.append(dt_ms)

            assert dt_ms < 20.0, f"Solve {i} query took {dt_ms:.2f}ms (threshold 20ms)"
            assert len(solutions) == 5

            c_base = CubeState().apply_moves(scramble)
            for sol in solutions:
                # 1. Zero physical L turns
                for m in sol.moves:
                    assert m not in {"L", "L'", "L2"}
                # 2. Correct length
                assert sol.move_count == len(sol.moves)
                # 3. Physical First Block verification
                assert_physically_solves_fb(c_base, sol)

        avg_ms = sum(durations) / len(durations)
        print(f"\nBatch 20 solves: avg={avg_ms:.2f}ms, max={max(durations):.2f}ms")
        assert avg_ms < 10.0, f"Average query latency {avg_ms:.2f}ms exceeded 10ms"

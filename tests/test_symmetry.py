"""Unit and property tests for Canonical Symmetry Re-Mapping (Milestone 3)."""

import pytest
from roux_engine.core.constants import Color, Edge, Corner
from roux_engine.core.cube import CubeState
from roux_engine.core.moves import MOVES
from roux_engine.core.parser import MoveParser
from roux_engine.segmenter.fb_detector import FBDetector, ALL_BLOCK_DEFINITIONS
from roux_engine.solver.fb_indexer import FBIndexer, FBPlacement
from roux_engine.solver.fb_pdb import FBPDB
from roux_engine.solver.symmetry import (
    CanonicalSymmetry,
    get_symmetry,
    get_all_symmetries,
    translate_moves,
    translate_moves_to_original,
    translate_moves_to_canonical,
    conjugate_cube,
    extract_canonical_placement,
    is_fb_solved_for_symmetry,
)


class TestGroupStructureAndSymmetries:
    """Slice 1: Group G structure, elements, inverses, and symmetry lookup."""

    def test_group_has_eight_elements(self):
        """Group G must have exactly 8 elements representing dual-neutral First Blocks."""
        symmetries = get_all_symmetries()
        assert len(symmetries) == 8
        assert len(CanonicalSymmetry) == 8

    def test_dual_neutral_color_coverage(self):
        """All 8 symmetries must cover 4 White-bottom and 4 Yellow-bottom First Blocks with 4 side colors."""
        symmetries = get_all_symmetries()
        bottom_colors = [s.bottom_color for s in symmetries]
        assert bottom_colors.count(Color.YELLOW) == 4
        assert bottom_colors.count(Color.WHITE) == 4

        expected_pairs = {
            (Color.YELLOW, Color.ORANGE),
            (Color.YELLOW, Color.GREEN),
            (Color.YELLOW, Color.RED),
            (Color.YELLOW, Color.BLUE),
            (Color.WHITE, Color.ORANGE),
            (Color.WHITE, Color.BLUE),
            (Color.WHITE, Color.RED),
            (Color.WHITE, Color.GREEN),
        }
        actual_pairs = {(s.bottom_color, s.left_color) for s in symmetries}
        assert actual_pairs == expected_pairs

    def test_symmetry_inverses(self):
        """Every symmetry g in G must have a valid inverse g^-1 in G satisfying g * g^-1 == I."""
        for s in get_all_symmetries():
            inv = s.inverse
            assert isinstance(inv, CanonicalSymmetry)
            # The inverse of the inverse is the original symmetry
            assert inv.inverse == s

        # Check known algebra:
        assert CanonicalSymmetry.I.inverse == CanonicalSymmetry.I
        assert CanonicalSymmetry.Y.inverse == CanonicalSymmetry.Y_PRIME
        assert CanonicalSymmetry.Y_PRIME.inverse == CanonicalSymmetry.Y
        assert CanonicalSymmetry.Y2.inverse == CanonicalSymmetry.Y2
        assert CanonicalSymmetry.X2.inverse == CanonicalSymmetry.X2
        assert CanonicalSymmetry.X2_Y.inverse == CanonicalSymmetry.X2_Y
        assert CanonicalSymmetry.X2_Y2.inverse == CanonicalSymmetry.X2_Y2
        assert CanonicalSymmetry.X2_Y_PRIME.inverse == CanonicalSymmetry.X2_Y_PRIME

    def test_get_symmetry_by_string_and_aliases(self):
        """Lookup by rotation strings with and without spaces or aliases must succeed."""
        assert get_symmetry("") == CanonicalSymmetry.I
        assert get_symmetry("I") == CanonicalSymmetry.I
        assert get_symmetry("identity") == CanonicalSymmetry.I
        assert get_symmetry("y") == CanonicalSymmetry.Y
        assert get_symmetry("y2") == CanonicalSymmetry.Y2
        assert get_symmetry("y'") == CanonicalSymmetry.Y_PRIME
        assert get_symmetry("x2") == CanonicalSymmetry.X2
        assert get_symmetry("x2 y") == CanonicalSymmetry.X2_Y
        assert get_symmetry("x2y") == CanonicalSymmetry.X2_Y
        assert get_symmetry("x2 y2") == CanonicalSymmetry.X2_Y2
        assert get_symmetry("x2y2") == CanonicalSymmetry.X2_Y2
        assert get_symmetry("x2 y'") == CanonicalSymmetry.X2_Y_PRIME
        assert get_symmetry("x2y'") == CanonicalSymmetry.X2_Y_PRIME

    def test_get_symmetry_by_colors(self):
        """Lookup by (bottom_color, left_color) tuple or Color enum."""
        assert get_symmetry((Color.WHITE, Color.BLUE)) == CanonicalSymmetry.X2_Y
        assert get_symmetry((0, 3)) == CanonicalSymmetry.X2_Y
        assert get_symmetry((Color.YELLOW, Color.ORANGE)) == CanonicalSymmetry.I
        assert get_symmetry((1, 4)) == CanonicalSymmetry.I
        assert get_symmetry((Color.WHITE, Color.GREEN)) == CanonicalSymmetry.X2_Y_PRIME
        assert get_symmetry("WHITE-BLUE") == CanonicalSymmetry.X2_Y
        assert get_symmetry("yellow-green") == CanonicalSymmetry.Y

    def test_get_symmetry_invalid_raises(self):
        """Invalid symmetry strings or unrecognized color pairs must raise ValueError or KeyError."""
        with pytest.raises((ValueError, KeyError)):
            get_symmetry("z")
        with pytest.raises((ValueError, KeyError)):
            get_symmetry("x")
        with pytest.raises((ValueError, KeyError)):
            get_symmetry((Color.RED, Color.BLUE))


class TestMoveTranslation:
    """Slice 2: Move translation and inverse automorphism mapping."""

    def test_identity_moves_invariant(self):
        """Under identity symmetry I, moves translate to themselves."""
        moves = ["R", "U", "R'", "F'", "r2", "M"]
        assert translate_moves_to_original(moves, CanonicalSymmetry.I) == moves
        assert translate_moves_to_canonical(moves, CanonicalSymmetry.I) == moves
        assert translate_moves(moves, CanonicalSymmetry.I, inverse=True) == moves
        assert translate_moves(moves, CanonicalSymmetry.I, inverse=False) == moves

    def test_empty_moves_translation(self):
        """Translating empty sequence returns empty list."""
        for sym in get_all_symmetries():
            assert translate_moves_to_original([], sym) == []
            assert translate_moves_to_canonical([], sym) == []
            assert translate_moves("", sym) == []

    def test_string_and_list_input_equivalence(self):
        """String input (e.g. 'R U R'') produces identical output to list input ['R', 'U', 'R']."""
        for sym in get_all_symmetries():
            res_str = translate_moves_to_original("R U R'", sym)
            res_list = translate_moves_to_original(["R", "U", "R'"], sym)
            assert res_str == res_list

    def test_move_translation_bijection_and_roundtrip(self):
        """Translating canonical moves to original and back to canonical is an exact bijection."""
        base_moves = [k for k in MOVES.keys() if not k.endswith("w") and not k.endswith("w'") and not k.endswith("w2")]

        for sym in get_all_symmetries():
            for m in base_moves:
                orig = translate_moves_to_original([m], sym)
                assert len(orig) == 1
                assert orig[0] in MOVES

                back_canon = translate_moves_to_canonical(orig, sym)
                assert back_canon == [m]

                # Vice-versa
                canon = translate_moves_to_canonical([m], sym)
                assert len(canon) == 1
                assert canon[0] in MOVES

                back_orig = translate_moves_to_original(canon, sym)
                assert back_orig == [m]

    def test_move_translation_algebraic_physical_correctness(self):
        """Translating m_canon to m_orig satisfies: CubeState().apply_move(m_orig) == g * m_canon * g^-1."""
        test_moves = ["U", "U'", "U2", "D", "D'", "D2", "R", "R'", "R2", "F", "F'", "F2", "B", "B'", "B2", "r", "r'", "r2", "M", "M'", "M2"]

        for sym in get_all_symmetries():
            g = sym.inspection_rotation
            inv_g = " ".join(MoveParser.invert_moves(g)) if g else ""

            for m in test_moves:
                # Inverse automorphism: canonical -> original (g * m * inv_g)
                orig = translate_moves_to_original([m], sym)[0]
                c_trans = CubeState().apply_move(orig)
                c_expected = CubeState().apply_moves(f"{g} {m} {inv_g}".strip())
                assert c_trans == c_expected, f"Failed for move {m} under symmetry {sym.name}"

                # Forward automorphism: original -> canonical (inv_g * m * g)
                canon = translate_moves_to_canonical([m], sym)[0]
                c_trans_canon = CubeState().apply_move(canon)
                c_expected_canon = CubeState().apply_moves(f"{inv_g} {m} {g}".strip())
                assert c_trans_canon == c_expected_canon, f"Failed forward automorphism for {m} under {sym.name}"

    def test_x2_conjugation_known_moves(self):
        """Under x2: U <-> D, F <-> B, L <-> L, R <-> R, M <-> M, r <-> r."""
        assert translate_moves_to_original(["U"], CanonicalSymmetry.X2) == ["D"]
        assert translate_moves_to_original(["D"], CanonicalSymmetry.X2) == ["U"]
        assert translate_moves_to_original(["F"], CanonicalSymmetry.X2) == ["B"]
        assert translate_moves_to_original(["B"], CanonicalSymmetry.X2) == ["F"]
        assert translate_moves_to_original(["R"], CanonicalSymmetry.X2) == ["R"]
        assert translate_moves_to_original(["L"], CanonicalSymmetry.X2) == ["L"]
        assert translate_moves_to_original(["M"], CanonicalSymmetry.X2) == ["M"]
        assert translate_moves_to_original(["r"], CanonicalSymmetry.X2) == ["r"]


class TestCubeConjugation:
    """Slice 3: Cube state conjugation to canonical frame."""

    @pytest.fixture
    def fb_pdb(self):
        return FBPDB()

    def test_solved_cube_maps_to_canonical_zero(self, fb_pdb):
        """On a clean/solved cube, conjugating under any of the 8 symmetries produces canonical index 0 and distance 0."""
        for sym in get_all_symmetries():
            clean = CubeState()
            c_canon = conjugate_cube(clean, sym)
            assert FBDetector.is_canonical_fb_solved(c_canon), f"Failed for symmetry {sym.name}"
            assert FBIndexer.encode(c_canon) == 0
            assert fb_pdb.get_distance(c_canon) == 0

    def test_inspected_solved_cube_maps_to_canonical_zero(self, fb_pdb):
        """On a cube already rotated into inspection frame, conjugating produces canonical index 0 and distance 0."""
        for sym in get_all_symmetries():
            inspected = CubeState()
            if sym.inspection_rotation:
                inspected.apply_moves(sym.inspection_rotation)
            c_canon = conjugate_cube(inspected, sym, inspected=True)
            assert FBDetector.is_canonical_fb_solved(c_canon)
            assert FBIndexer.encode(c_canon) == 0
            assert fb_pdb.get_distance(c_canon) == 0

    def test_is_fb_solved_for_symmetry(self):
        """is_fb_solved_for_symmetry detects solved block accurately across orientations."""
        for sym in get_all_symmetries():
            c = CubeState()
            if sym.inspection_rotation:
                c.apply_moves(sym.inspection_rotation)
            assert is_fb_solved_for_symmetry(c, sym, inspected=True)

            # In original frame
            c_clean = CubeState()
            assert is_fb_solved_for_symmetry(c_clean, sym, inspected=False)

            # Move affecting FB should break solved condition
            c_disturbed = c.copy().apply_move("D")
            assert not is_fb_solved_for_symmetry(c_disturbed, sym, inspected=True)

    def test_one_move_distance_consistency(self, fb_pdb):
        """1 move affecting FB must yield canonical distance 1; 1 move preserving FB must yield distance 0."""
        for sym in get_all_symmetries():
            for m in ["D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]:
                m_orig = translate_moves_to_original([m], sym)[0]
                c_orig = CubeState().apply_move(m_orig)
                c_canon = conjugate_cube(c_orig, sym, inspected=False)
                d = fb_pdb.get_distance(c_canon)
                assert d == 1, f"Move {m} ({m_orig}) under {sym.name} should have distance 1, got {d}"

            for m in ["R", "R'", "R2", "U", "U'", "U2", "r", "r'", "r2", "M", "M'", "M2"]:
                m_orig = translate_moves_to_original([m], sym)[0]
                c_orig = CubeState().apply_move(m_orig)
                c_canon = conjugate_cube(c_orig, sym, inspected=False)
                d = fb_pdb.get_distance(c_canon)
                assert d == 0, f"Move {m} ({m_orig}) under {sym.name} should preserve FB (distance 0), got {d}"

    def test_extract_canonical_placement_solved(self):
        """extract_canonical_placement produces solved coordinates for any solved orientation."""
        for sym in get_all_symmetries():
            c = CubeState()
            p = extract_canonical_placement(c, sym)
            assert p.dl_slot == Edge.DL and p.dl_eo == 0
            assert p.fl_slot == Edge.FL and p.fl_eo == 0
            assert p.bl_slot == Edge.BL and p.bl_eo == 0
            assert p.dlf_slot == Corner.DLF and p.dlf_co == 0
            assert p.dbl_slot == Corner.DBL and p.dbl_co == 0
            assert FBIndexer.encode_placement(p) == 0


class TestEndToEndSolutionVerification:
    """Slice 4: End-to-end solution verification across all 8 dual-neutral orientations."""

    @pytest.fixture
    def fb_pdb(self):
        return FBPDB()

    def test_random_walk_solution_verification_all_eight_orientations(self, fb_pdb):
        """For any of the 8 dual-neutral First Blocks, applying translated solution moves

        from its inspection rotation solves the specified First Block, and heuristic h(s) is admissible.
        """
        import random
        from roux_engine.solver.pdb_generator import FB_MOVESET

        rng = random.Random(999)

        for sym in get_all_symmetries():
            for depth in [1, 2, 3, 4, 5]:
                for trial in range(3):
                    # Pick canonical solution moves in FB_MOVESET
                    sol_moves = [rng.choice(FB_MOVESET) for _ in range(depth)]
                    sol_str = " ".join(sol_moves)
                    inv_sol_str = " ".join(MoveParser.invert_moves(sol_str))

                    # Create scramble in the original frame:
                    # Scramble in world coordinates such that inspecting with sym yields cube needing sol_moves:
                    # scramble = sym.inspection_rotation * inv_sol_str * inv_sym
                    g = sym.inspection_rotation
                    inv_g = " ".join(MoveParser.invert_moves(g)) if g else ""
                    scramble = f"{g} {inv_sol_str} {inv_g}".strip()

                    c_scramble = CubeState().apply_moves(scramble)

                    # 1. Verify heuristic admissibility on canonical conjugate
                    c_canon = conjugate_cube(c_scramble, sym, inspected=False)
                    h = fb_pdb.get_distance(c_canon)
                    assert h <= depth, f"Heuristic admissibility violated: h({h}) > depth({depth}) for {sym.name}"

                    # 2. Applying translated solution moves from inspection rotation solves specified First Block
                    c_inspected = c_scramble.copy()
                    if g:
                        c_inspected.apply_moves(g)
                    c_inspected.apply_moves(sol_str)
                    assert is_fb_solved_for_symmetry(c_inspected, sym, inspected=True), (
                        f"Applying moves from inspection rotation failed to solve block {sym.name}"
                    )

                    # 3. Applying translated solution moves directly to original frame solves specified First Block
                    m_orig = translate_moves_to_original(sol_moves, sym)
                    c_orig_solved = c_scramble.copy().apply_moves(" ".join(m_orig))
                    assert is_fb_solved_for_symmetry(c_orig_solved, sym, inspected=False), (
                        f"Applying translated moves in original frame failed to solve block {sym.name}"
                    )

    def test_reco_benchmark_dataset_solves(self, fb_pdb):
        """Authentic Roux solves from dataset correctly conjugate and verify across dual-neutral orientations."""
        import json, os
        from roux_engine.segmenter import RouxSegmenter

        dataset_path = "data/roux_solves.json"
        if not os.path.exists(dataset_path):
            pytest.skip("Dataset not found")

        with open(dataset_path, "r", encoding="utf-8") as f:
            solves = json.load(f)[:50]

        segmenter = RouxSegmenter()

        checked_count = 0
        for item in solves:
            scramble = item.get("scramble", "")
            solution = item.get("solution", "")
            if not scramble or not solution:
                continue

            segmented = segmenter.segment(scramble, solution)
            if not segmented.is_valid or segmented.fb is None or segmented.fb.orientation is None:
                continue

            try:
                sym = get_symmetry((segmented.fb.orientation.bottom_color, segmented.fb.orientation.left_color))
            except ValueError:
                # Full color neutral solve outside x2y group, skip for dual-neutral test
                continue

            # Replay solution from scramble and verify FB is solved
            c_scramble = CubeState().apply_moves(scramble)
            c_test = c_scramble.copy()
            c_test.apply_moves(segmented.fb.moves_str)
            assert is_fb_solved_for_symmetry(c_test, sym, inspected=True)

            # For solves using canonical FB moveset and standard orientation, verify heuristic admissibility
            from roux_engine.solver.pdb_generator import FB_MOVESET
            events = MoveParser.parse_string(segmented.fb.moves_str)
            exec_moves = [e.move for e in events if not e.move.startswith(('x', 'y', 'z'))]
            rot_moves = [e.move for e in events if e.move.startswith(('x', 'y', 'z'))]

            c_actual_inspect = c_scramble.copy()
            if rot_moves:
                c_actual_inspect.apply_moves(" ".join(rot_moves))

            if all(m in FB_MOVESET for m in exec_moves):
                if (c_actual_inspect.centers[1] == sym.bottom_color.value and
                    c_actual_inspect.centers[4] == sym.left_color.value):
                    c_canon = conjugate_cube(c_scramble, sym, inspected=False)
                    h = fb_pdb.get_distance(c_canon)
                    human_stm = segmented.fb.move_count_stm
                    assert h <= human_stm, f"Admissibility violated on reco solve: h({h}) > human_stm({human_stm})"

            checked_count += 1

        assert checked_count > 0, "Expected at least one valid dual-neutral solve in sample"


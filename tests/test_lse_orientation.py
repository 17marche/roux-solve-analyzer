"""Unit tests for LSE orientation resolution and frame conjugation machinery."""

import pytest
from roux_engine.core.constants import Color, Center, Edge
from roux_engine.core.cube import CubeState
from roux_engine.core.orientation import (
    CanonicalSymmetry,
    RouxOrientation,
    CANONICAL_ORIENTATION,
    get_dual_neutral_orientations,
    get_orientation,
)
from roux_engine.solver.lse_solver import resolve_lse_orientation


class TestLSEOrientationAutoDetectionInspected:
    """Slice 1: Auto-detection of solve orientation in inspected hand frames."""

    def test_solved_cube_defaults_to_canonical_inspected(self):
        """A clean/solved cube defaults to canonical orientation and inspected frame."""
        clean = CubeState()
        ori, is_inspected = resolve_lse_orientation(clean)
        assert ori == CANONICAL_ORIENTATION
        assert ori.rotations == ""
        assert is_inspected is True

    def test_auto_detect_all_eight_dual_neutral_inspected(self):
        """Auto-detects each of the 8 dual-neutral orientations when FB and SB are solved in inspected hand frame."""
        for expected_ori in get_dual_neutral_orientations():
            # Build cube with FB and SB solved in expected_ori inspection frame
            c = CubeState()
            if expected_ori.rotations:
                c.apply_moves(expected_ori.rotations)

            # Apply some LSE moves so the cube is not fully solved, but FB & SB remain intact
            c.apply_moves("M' U2 M U M2")
            assert expected_ori.is_fb_solved(c)
            assert expected_ori.is_sb_solved(c)

            detected_ori, is_inspected = resolve_lse_orientation(c)
            assert detected_ori == expected_ori, f"Failed for orientation {expected_ori.rotations!r}"
            assert is_inspected is True

    def test_undetermined_cube_defaults_to_canonical(self):
        """A fully scrambled cube with no solved blocks defaults to canonical orientation."""
        scrambled = CubeState().apply_moves("R U F D L B")
        ori, is_inspected = resolve_lse_orientation(scrambled)
        assert ori == CANONICAL_ORIENTATION
        assert is_inspected is True


class TestLSEOrientationExplicitOverrides:
    """Slice 2: Explicit orientation overrides and validation."""

    def test_explicit_override_by_rotation_string(self):
        """Accepts explicit rotation strings (e.g. 'x2', 'y', 'x2 y')."""
        c = CubeState().apply_moves("x2 M' U2 M")
        ori, is_inspected = resolve_lse_orientation(c, orientation="x2")
        assert ori.rotations == "x2"
        assert ori.bottom_color == Color.WHITE
        assert is_inspected is True

    def test_explicit_override_by_color_pair(self):
        """Accepts explicit (bottom_color, left_color) tuple."""
        # White bottom, Blue left -> "x2 y"
        c = CubeState().apply_moves("x2 y M U M'")
        ori, is_inspected = resolve_lse_orientation(c, orientation=(Color.WHITE, Color.BLUE))
        assert ori.rotations == "x2 y"
        assert ori.bottom_color == Color.WHITE
        assert ori.left_color == Color.BLUE
        assert is_inspected is True

    def test_explicit_override_by_enum_and_instance(self):
        """Accepts CanonicalSymmetry enum and RouxOrientation instance."""
        c = CubeState().apply_moves("y M2 U2 M2")
        # CanonicalSymmetry enum
        ori_enum, is_inspected = resolve_lse_orientation(c, orientation=CanonicalSymmetry.Y)
        assert ori_enum.rotations == "y"
        assert is_inspected is True

        # RouxOrientation instance
        ori_obj = get_orientation("y")
        ori_res, is_inspected = resolve_lse_orientation(c, orientation=ori_obj)
        assert ori_res == ori_obj
        assert is_inspected is True

    def test_explicit_override_on_solved_cube(self):
        """Explicit orientation on solved cube correctly identifies inspected vs uninspected frame."""
        clean = CubeState()
        # Canonical orientation on clean cube is inspected (blocks on Left/Right)
        ori_canon, is_inspected_canon = resolve_lse_orientation(clean, orientation="")
        assert ori_canon.rotations == ""
        assert is_inspected_canon is True

        # Non-canonical orientation on clean cube has blocks in uninspected world frame
        ori_world, is_inspected_world = resolve_lse_orientation(clean, orientation="x2 y2")
        assert ori_world.rotations == "x2 y2"
        assert is_inspected_world is False

        # If cube is inspected for "x2 y2", is_inspected is True
        c_inspected = CubeState().apply_moves("x2 y2")
        ori_insp, is_inspected_insp = resolve_lse_orientation(c_inspected, orientation="x2 y2")
        assert ori_insp.rotations == "x2 y2"
        assert is_inspected_insp is True

    def test_explicit_override_mismatch_raises_value_error(self):
        """Explicit orientation that doesn't match the solved blocks on the cube raises ValueError."""
        # Cube is oriented for "x2", but caller asserts orientation "y"
        c = CubeState().apply_moves("x2 M' U2 M")
        with pytest.raises(ValueError, match="First Block and Second Block are not solved"):
            resolve_lse_orientation(c, orientation="y")

    def test_invalid_orientation_identifier_raises(self):
        """Invalid orientation identifier strings or impossible color pairs raise ValueError."""
        c = CubeState()
        with pytest.raises(ValueError):
            resolve_lse_orientation(c, orientation="invalid_rotation_xyz")
        with pytest.raises(ValueError):
            resolve_lse_orientation(c, orientation=(Color.WHITE, Color.YELLOW))


class TestLSEOrientationUninspectedWorldFrames:
    """Slice 3: Identification of uninspected world frames (omitted inspection rotations)."""

    def test_auto_detect_uninspected_world_frames_front_back(self):
        """When solve is performed without inspection rotation, detects orientation and flags is_inspected=False."""
        from roux_engine.core.orientation import translate_moves_to_original

        # Test Dual-Neutral symmetries that rotate blocks from Left/Right to Front/Back
        for sym in [CanonicalSymmetry.Y, CanonicalSymmetry.X2_Y]:
            exp_ori = get_orientation(sym)

            # In uninspected world frame, cuber solves FB and SB in world coordinates
            # and leaves cube in LSE state by executing translated LSE moves:
            world_lse_moves = translate_moves_to_original(["M'", "U2", "M", "U", "M2"], sym)
            c_world = CubeState().apply_moves(" ".join(world_lse_moves))

            # On c_world, blocks are on Front/Back, NOT on Left/Right:
            assert not exp_ori.is_fb_solved(c_world)

            # Auto-detection must discover orientation and flag uninspected:
            detected_ori, is_inspected = resolve_lse_orientation(c_world)
            assert detected_ori == exp_ori, f"Failed auto-detecting uninspected {sym.name}"
            assert is_inspected is False, f"Expected is_inspected=False for {sym.name}"

    def test_explicit_override_uninspected_frames_all_dual_neutral(self):
        """Explicit override on uninspected cubes correctly validates and flags is_inspected=False."""
        from roux_engine.core.orientation import translate_moves_to_original

        test_symmetries = [
            CanonicalSymmetry.Y,
            CanonicalSymmetry.Y2,
            CanonicalSymmetry.Y_PRIME,
            CanonicalSymmetry.X2,
            CanonicalSymmetry.X2_Y,
            CanonicalSymmetry.X2_Y2,
            CanonicalSymmetry.X2_Y_PRIME,
        ]

        for sym in test_symmetries:
            exp_ori = get_orientation(sym)
            world_lse_moves = translate_moves_to_original(["M'", "U2", "M", "U", "M2"], sym)
            c_world = CubeState().apply_moves(" ".join(world_lse_moves))

            # If not in inspected hand frame for this orientation:
            if not (exp_ori.is_fb_solved(c_world) and exp_ori.is_sb_solved(c_world)):
                detected_exp, is_inspected_exp = resolve_lse_orientation(c_world, orientation=sym)
                assert detected_exp == exp_ori
                assert is_inspected_exp is False


class TestLSECubeConjugation:
    """Slice 4: Canonical re-mapping and conjugation of cube states into the canonical frame."""

    def test_conjugate_clean_cube_yields_canonical_solved(self):
        """Conjugating an oriented solved cube yields a canonical solved cube."""
        from roux_engine.solver.lse_solver import conjugate_lse_cube

        for ori in get_dual_neutral_orientations():
            c = CubeState()
            if ori.rotations:
                c.apply_moves(ori.rotations)
            c_canon = conjugate_lse_cube(c, orientation=ori, is_inspected=True)
            assert c_canon.is_solved(), f"Failed for {ori.rotations!r}"

    def test_conjugate_all_eight_dual_neutral_in_lse_graph(self):
        """Conjugated states across all 8 dual-neutral orientations are present in LSEGraph."""
        from roux_engine.solver.lse_solver import conjugate_lse_cube, LSEGraph

        graph = LSEGraph.get_instance()
        test_moves = [
            "M'",
            "M2 U2 M2",
            "M U M' U'",
            "M' U2 M U M2",
            "U M2 U' M U2 M' U2",
        ]

        for ori in get_dual_neutral_orientations():
            for m in test_moves:
                c = CubeState()
                if ori.rotations:
                    c.apply_moves(ori.rotations)
                c.apply_moves(m)

                c_canon = conjugate_lse_cube(c, orientation=ori, is_inspected=True)
                assert graph.contains_state(c_canon), f"State not in graph for {ori.rotations!r} with {m}"

    def test_conjugate_uninspected_frame_matches_inspected(self):
        """Conjugating an uninspected cube with is_inspected=False yields identical canonical state to inspected."""
        from roux_engine.solver.lse_solver import conjugate_lse_cube
        from roux_engine.core.orientation import translate_moves_to_original

        for sym in [CanonicalSymmetry.Y, CanonicalSymmetry.X2_Y, CanonicalSymmetry.X2_Y2]:
            ori = get_orientation(sym)

            # Inspected cube
            c_ins = CubeState()
            if ori.rotations:
                c_ins.apply_moves(ori.rotations)
            c_ins.apply_moves("M' U2 M U M2")

            # Uninspected cube (world frame)
            world_moves = translate_moves_to_original(["M'", "U2", "M", "U", "M2"], sym)
            c_world = CubeState().apply_moves(" ".join(world_moves))

            canon_ins = conjugate_lse_cube(c_ins, orientation=ori, is_inspected=True)
            canon_world = conjugate_lse_cube(c_world, orientation=ori, is_inspected=False)

            assert canon_ins == canon_world, f"Conjugation mismatch for {sym.name}"

    def test_conjugate_preserves_misoriented_centers(self):
        """Misoriented centers (on F/B axis) are mapped to canonical misoriented center values."""
        from roux_engine.solver.lse_solver import conjugate_lse_cube

        # Scramble ending with misoriented centers
        for ori in get_dual_neutral_orientations():
            c = CubeState()
            if ori.rotations:
                c.apply_moves(ori.rotations)
            # M puts front/back center on U
            c.apply_move("M")
            c_canon = conjugate_lse_cube(c, orientation=ori, is_inspected=True)
            # In canonical, misoriented centers on U are Green (2) or Blue (3)
            assert c_canon.centers[0] in (2, 3), f"Failed for {ori.rotations!r}"

    def test_conjugate_preserves_corner_auf(self):
        """Corner AUF on the U layer is preserved during canonical conjugation."""
        from roux_engine.solver.lse_solver import conjugate_lse_cube

        for ori in get_dual_neutral_orientations():
            for auf in ("U", "U2", "U'"):
                c = CubeState()
                if ori.rotations:
                    c.apply_moves(ori.rotations)
                c.apply_move(auf)
                c_canon = conjugate_lse_cube(c, orientation=ori, is_inspected=True)
                # Comparing with clean cube rotated by auf
                expected_canon = CubeState().apply_move(auf)
                assert c_canon.cp[0] == expected_canon.cp[0]


class TestLSEMoveTranslation:
    """Slice 5: Move translation to user execution frame."""

    def test_translate_inspected_moves_pass_through(self):
        """For inspected hand frames, canonical <M, U> moves pass through unchanged."""
        from roux_engine.solver.lse_solver import translate_lse_moves

        test_moves = ["M", "U", "M'", "U2", "M2"]
        for ori in get_dual_neutral_orientations():
            translated = translate_lse_moves(test_moves, orientation=ori, is_inspected=True)
            assert translated == test_moves

    def test_translate_accepts_string_and_list(self):
        """Accepts either space-separated move string or list of move tokens."""
        from roux_engine.solver.lse_solver import translate_lse_moves

        res_str = translate_lse_moves("M' U2 M", orientation="", is_inspected=True)
        res_list = translate_lse_moves(["M'", "U2", "M"], orientation="", is_inspected=True)
        assert res_str == ["M'", "U2", "M"]
        assert res_str == res_list

    def test_translate_uninspected_moves_dual_neutral(self):
        """For uninspected frames, translates <M, U> moves to caller's world frame."""
        from roux_engine.solver.lse_solver import translate_lse_moves

        # Y symmetry: M -> S, M' -> S', M2 -> S2, U -> U
        trans_y = translate_lse_moves(["M", "U", "M'"], orientation="y", is_inspected=False)
        assert trans_y == ["S", "U", "S'"]

        # X2 symmetry: M -> M, U -> D, U' -> D', U2 -> D2
        trans_x2 = translate_lse_moves(["M", "U", "U'"], orientation="x2", is_inspected=False)
        assert trans_x2 == ["M", "D", "D'"]

    def test_translate_algebraic_physical_correctness(self):
        """Applying translated moves directly to world cube is physically equivalent to applying canonical moves to inspected cube."""
        from roux_engine.solver.lse_solver import translate_lse_moves

        canon_moves = ["M'", "U2", "M", "U", "M2"]

        for ori in get_dual_neutral_orientations():
            world_moves = translate_lse_moves(canon_moves, orientation=ori, is_inspected=False)

            # Applying canonical moves to inspected cube:
            c_ins = CubeState()
            if ori.rotations:
                c_ins.apply_moves(ori.rotations)
            c_ins.apply_moves(canon_moves)

            # Applying world moves to clean cube, then rotating into inspected frame:
            c_world = CubeState().apply_moves(world_moves)
            if ori.rotations:
                c_world.apply_moves(ori.rotations)

            assert c_ins == c_world, f"Algebraic mismatch for orientation {ori.rotations!r}"


class TestLSERoundTripConjugationAllDualNeutral:
    """Slice 6: Full round-trip conjugation and move translation verification across all 8 Dual-Neutral orientations."""

    SCRAMBLES = [
        "M'",
        "M2 U2 M2",
        "M U M' U'",
        "U2 M U M' U2 M2 U",
        "M' U2 M U M2 U' M'",
        "M U2 M U2 M' U2 M",
        "U M2 U' M U2 M' U2",
    ]

    def test_round_trip_all_eight_dual_neutral_inspected(self):
        """For all 8 dual-neutral orientations in inspected frame: resolve -> conjugate -> solve -> translate -> apply -> solved."""
        from roux_engine.solver.lse_solver import (
            resolve_lse_orientation,
            conjugate_lse_cube,
            translate_lse_moves,
            solve_lse,
        )

        for ori in get_dual_neutral_orientations():
            for sc in self.SCRAMBLES:
                # 1. Cube in inspected hand frame for orientation
                c = CubeState()
                if ori.rotations:
                    c.apply_moves(ori.rotations)
                c.apply_moves(sc)

                # 2. Resolve orientation
                det_ori, is_inspected = resolve_lse_orientation(c)
                assert det_ori == ori
                assert is_inspected is True

                # 3. Conjugate to canonical frame
                c_canon = conjugate_lse_cube(c, orientation=det_ori, is_inspected=is_inspected)

                # 4. Solve in canonical frame
                sols = solve_lse(c_canon, target="lse")
                assert len(sols) == 1
                m_canon = sols[0].moves

                # 5. Translate moves back to caller's execution frame
                m_user = translate_lse_moves(m_canon, orientation=det_ori, is_inspected=is_inspected)

                # 6. Apply to original cube
                c_solved = c.copy().apply_moves(m_user)

                # Verify all phases are fully solved for orientation
                assert det_ori.is_fb_solved(c_solved), f"FB not solved for {ori.rotations} with {sc}"
                assert det_ori.is_sb_solved(c_solved), f"SB not solved for {ori.rotations} with {sc}"
                assert det_ori.is_eo_solved(c_solved), f"EO not solved for {ori.rotations} with {sc}"
                assert det_ori.is_ul_ur_solved(c_solved), f"UL/UR not solved for {ori.rotations} with {sc}"
                assert det_ori.get_m_slice_center_offset(c_solved) == 0, f"Centers not aligned for {ori.rotations} with {sc}"

                # And when rotated back to canonical inspection, it is completely solved
                c_clean_check = c_solved.copy()
                from roux_engine.core.parser import MoveParser
                inv_rot = " ".join(MoveParser.invert_moves(det_ori.rotations))
                if inv_rot:
                    c_clean_check.apply_moves(inv_rot)
                assert c_clean_check.is_solved(), f"Cube not completely solved for {ori.rotations} with {sc}"

    def test_round_trip_all_eight_dual_neutral_uninspected(self):
        """For all 8 dual-neutral orientations in uninspected world frame: resolve -> conjugate -> solve -> translate -> apply -> solved."""
        from roux_engine.solver.lse_solver import (
            resolve_lse_orientation,
            conjugate_lse_cube,
            translate_lse_moves,
            solve_lse,
        )
        from roux_engine.core.orientation import translate_moves_to_original

        for sym in CanonicalSymmetry:
            ori = get_orientation(sym)
            for sc in self.SCRAMBLES:
                # 1. Cube in uninspected world frame:
                # Solution moves were performed in world frame without inspection rotation
                c_world = CubeState()
                world_sc = translate_moves_to_original(sc.split(), sym)
                c_world.apply_moves(world_sc)

                # 2. Resolve orientation: explicit orientation works across all 8 dual-neutral orientations
                det_ori, is_inspected = resolve_lse_orientation(c_world, orientation=sym)
                assert det_ori == ori
                expected_inspected = (sym == CanonicalSymmetry.I)
                assert is_inspected is expected_inspected

                # 3. Conjugate to canonical frame
                c_canon = conjugate_lse_cube(c_world, orientation=det_ori, is_inspected=is_inspected)

                # 4. Solve in canonical frame
                sols = solve_lse(c_canon, target="lse")
                assert len(sols) == 1
                m_canon = sols[0].moves

                # 5. Translate moves back to caller's execution frame
                m_user = translate_lse_moves(m_canon, orientation=det_ori, is_inspected=is_inspected)

                # 6. Apply to original uninspected cube
                c_solved = c_world.copy().apply_moves(m_user)

                # In the world frame, c_solved must be physically 100% solved!
                assert c_solved.is_solved(), f"Uninspected cube not solved for {sym.name} with {sc}"







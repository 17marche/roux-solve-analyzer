"""Tests for RouxOrientation domain entity and symmetry integration."""

import pytest
from roux_engine.core.constants import Color, Edge, Corner, Center
from roux_engine.core.cube import CubeState
from roux_engine.core.orientation import (
    RouxOrientation,
    CanonicalSymmetry,
    SBPlacement,
    get_orientation,
    get_all_orientations,
    get_dual_neutral_orientations,
    translate_moves,
    translate_moves_to_original,
    translate_moves_to_canonical,
)


class TestOrientationResolutionAndRegistry:
    """Seam 1: Orientation Resolution & Registry."""

    def test_all_24_orientations_registered(self):
        orientations = get_all_orientations()
        assert len(orientations) == 24
        # All rotations strings are unique
        rotations = [o.rotations for o in orientations]
        assert len(set(rotations)) == 24

    def test_dual_neutral_orientations_count(self):
        dn = get_dual_neutral_orientations()
        assert len(dn) == 8
        expected_dn_rotations = {
            "", "y", "y2", "y'",
            "x2", "x2 y", "x2 y2", "x2 y'"
        }
        assert {o.rotations for o in dn} == expected_dn_rotations

    def test_resolve_by_rotation_string(self):
        o_identity = get_orientation("")
        assert o_identity.rotations == ""
        assert o_identity.bottom_color == Color.YELLOW
        assert o_identity.left_color == Color.ORANGE

        o_y = get_orientation("y")
        assert o_y.rotations == "y"
        assert o_y.bottom_color == Color.YELLOW
        assert o_y.left_color == Color.GREEN

        o_x2 = get_orientation("x2")
        assert o_x2.rotations == "x2"
        assert o_x2.bottom_color == Color.WHITE
        assert o_x2.left_color == Color.ORANGE

    def test_resolve_by_color_pair(self):
        # Yellow bottom, Orange left -> Identity ""
        o1 = get_orientation((Color.YELLOW, Color.ORANGE))
        assert o1.rotations == ""

        # White bottom, Blue left -> "x2 y" (or equivalent)
        o2 = get_orientation((Color.WHITE, Color.BLUE))
        assert o2.bottom_color == Color.WHITE
        assert o2.left_color == Color.BLUE

    def test_resolve_identity_passthrough(self):
        o = get_orientation("")
        assert get_orientation(o) is o

    def test_invalid_resolution_raises_value_error(self):
        with pytest.raises(ValueError):
            get_orientation("invalid_rotations")

        with pytest.raises(ValueError):
            get_orientation((Color.WHITE, Color.YELLOW))  # Opposite faces cannot be bottom and left


class TestPieceIdentityAndSymmetryIntegration:
    """Seam 2: Piece Identity & Dual-Neutral Symmetry Integration."""

    def test_piece_identities_across_all_24_orientations(self):
        for o in get_all_orientations():
            c = CubeState()
            if o.rotations:
                c.apply_moves(o.rotations)

            # Check colors
            assert o.left_color == Color(int(c.centers[Center.L]))
            assert o.bottom_color == Color(int(c.centers[Center.D]))
            assert o.front_color == Color(int(c.centers[Center.F]))
            assert o.back_color == Color(int(c.centers[Center.B]))
            assert o.right_color == Color(int(c.centers[Center.R]))
            assert o.top_color == Color(int(c.centers[Center.U]))

            # Check FB piece identities
            assert o.dl_piece == int(c.ep[Edge.DL])
            assert o.dl_eo == int(c.eo[Edge.DL])
            assert o.fl_piece == int(c.ep[Edge.FL])
            assert o.fl_eo == int(c.eo[Edge.FL])
            assert o.bl_piece == int(c.ep[Edge.BL])
            assert o.bl_eo == int(c.eo[Edge.BL])
            assert o.dlf_piece == int(c.cp[Corner.DLF])
            assert o.dlf_co == int(c.co[Corner.DLF])
            assert o.dbl_piece == int(c.cp[Corner.DBL])
            assert o.dbl_co == int(c.co[Corner.DBL])

            # Check SB piece identities
            assert o.dr_piece == int(c.ep[Edge.DR])
            assert o.dr_eo == int(c.eo[Edge.DR])
            assert o.fr_piece == int(c.ep[Edge.FR])
            assert o.fr_eo == int(c.eo[Edge.FR])
            assert o.br_piece == int(c.ep[Edge.BR])
            assert o.br_eo == int(c.eo[Edge.BR])
            assert o.dfr_piece == int(c.cp[Corner.DFR])
            assert o.dfr_co == int(c.co[Corner.DFR])
            assert o.drb_piece == int(c.cp[Corner.DRB])
            assert o.drb_co == int(c.co[Corner.DRB])

            # Check LSE piece identities
            assert o.ul_piece == int(c.ep[Edge.UL])
            assert o.ul_eo == int(c.eo[Edge.UL])
            assert o.ur_piece == int(c.ep[Edge.UR])
            assert o.ur_eo == int(c.eo[Edge.UR])
            assert o.uf_piece == int(c.ep[Edge.UF])
            assert o.uf_eo == int(c.eo[Edge.UF])
            assert o.ub_piece == int(c.ep[Edge.UB])
            assert o.ub_eo == int(c.eo[Edge.UB])
            assert o.df_piece == int(c.ep[Edge.DF])
            assert o.df_eo == int(c.eo[Edge.DF])
            assert o.db_piece == int(c.ep[Edge.DB])
            assert o.db_eo == int(c.eo[Edge.DB])

    def test_dual_neutral_symmetry_property_and_resolution(self):
        dn_orientations = get_dual_neutral_orientations()
        assert len(dn_orientations) == 8
        for o in dn_orientations:
            assert o.is_dual_neutral is True
            assert isinstance(o.symmetry, CanonicalSymmetry)
            assert o.symmetry.value == o.rotations
            # Resolution by CanonicalSymmetry
            resolved = get_orientation(o.symmetry)
            assert resolved == o
            # Inspection rotation
            assert o.symmetry.inspection_rotation == o.rotations

    def test_fcn_orientations_have_no_symmetry(self):
        all_orientations = get_all_orientations()
        dn_orientations = set(get_dual_neutral_orientations())
        non_dn = [o for o in all_orientations if o not in dn_orientations]
        assert len(non_dn) == 16
        for o in non_dn:
            assert o.is_dual_neutral is False
            assert o.symmetry is None

    def test_canonical_symmetry_inverses_and_move_translation(self):
        for sym in CanonicalSymmetry:
            assert sym.inverse.inverse == sym

        assert CanonicalSymmetry.I.inverse == CanonicalSymmetry.I
        assert CanonicalSymmetry.Y.inverse == CanonicalSymmetry.Y_PRIME
        assert CanonicalSymmetry.Y_PRIME.inverse == CanonicalSymmetry.Y
        assert CanonicalSymmetry.X2.inverse == CanonicalSymmetry.X2

        # Move translations
        orig_moves = ["R", "U", "R'"]
        canon_moves = translate_moves_to_canonical(orig_moves, CanonicalSymmetry.Y)
        back_moves = translate_moves_to_original(canon_moves, CanonicalSymmetry.Y)
        assert back_moves == orig_moves


class TestBlockAndPhaseCompletionQueries:
    """Seam 3: Block & Phase Completion Queries across all 24 orientations."""

    def test_all_queries_on_solved_orientations(self):
        """A cube oriented for any of the 24 orientations must pass all phase queries."""
        for o in get_all_orientations():
            c = CubeState()
            if o.rotations:
                c.apply_moves(o.rotations)

            assert o.is_fb_solved(c) is True, f"FB failed for {o.rotations}"
            assert o.is_dr_solved(c) is True, f"DR failed for {o.rotations}"
            assert o.is_back_pair_solved(c) is True, f"Back pair failed for {o.rotations}"
            assert o.is_front_pair_solved(c) is True, f"Front pair failed for {o.rotations}"
            assert o.is_sb_solved(c) is True, f"SB failed for {o.rotations}"
            assert o.get_m_slice_center_offset(c) == 0, f"M offset failed for {o.rotations}"
            assert o.is_center_aligned_sb_solved(c) is True, f"Center aligned SB failed for {o.rotations}"
            assert o.is_center_axis_aligned(c) is True, f"Center axis aligned failed for {o.rotations}"
            assert o.count_bad_edges(c) == 0, f"Count bad edges failed for {o.rotations}"
            assert o.is_eo_solved(c) is True, f"EO failed for {o.rotations}"
            assert o.is_ul_ur_solved(c) is True, f"UL/UR failed for {o.rotations}"

    def test_fb_disturbance_rejections(self):
        o = get_orientation("")
        c = CubeState()
        assert o.is_fb_solved(c) is True

        # Corner twist at DLF
        c_twist = c.copy()
        c_twist.co[Corner.DLF] = 1
        assert o.is_fb_solved(c_twist) is False

        # Edge flip at DL
        c_flip = c.copy()
        c_flip.eo[Edge.DL] = 1
        assert o.is_fb_solved(c_flip) is False

        # Alter L center
        c_center = c.copy()
        c_center.centers[Center.L] = Center.R
        assert o.is_fb_solved(c_center) is False

    def test_sb_piece_queries_and_rejections(self):
        o = get_orientation("")
        c = CubeState()

        # DR query
        assert o.is_dr_solved(c) is True
        c_dr_flip = c.copy()
        c_dr_flip.eo[Edge.DR] = 1
        assert o.is_dr_solved(c_dr_flip) is False
        assert o.is_sb_solved(c_dr_flip) is False

        # Back pair query
        assert o.is_back_pair_solved(c) is True
        c_bp_flip = c.copy()
        c_bp_flip.eo[Edge.BR] = 1
        assert o.is_back_pair_solved(c_bp_flip) is False
        assert o.is_sb_solved(c_bp_flip) is False

        # Front pair query
        assert o.is_front_pair_solved(c) is True
        c_fp_twist = c.copy()
        c_fp_twist.co[Corner.DFR] = 2
        assert o.is_front_pair_solved(c_fp_twist) is False
        assert o.is_sb_solved(c_fp_twist) is False

    def test_center_aligned_sb_and_m_slice_offsets(self):
        o = get_orientation("")
        c = CubeState()

        # Offset 0: Aligned
        assert o.get_m_slice_center_offset(c) == 0
        assert o.is_center_aligned_sb_solved(c) is True

        # Apply M2: Offset 2 (still center-aligned SB)
        c_m2 = c.copy()
        c_m2.apply_move("M2")
        # In M2, centers swap U/D and F/B, but SB pieces (DR, FR, BR, DFR, DRB) were on R, unaffected by M!
        assert o.is_sb_solved(c_m2) is True
        assert o.get_m_slice_center_offset(c_m2) == 2
        assert o.is_center_aligned_sb_solved(c_m2) is True

        # Apply M: Offset 1 (U/D axis misaligned)
        c_m = c.copy()
        c_m.apply_move("M")
        assert o.is_sb_solved(c_m) is True
        assert o.get_m_slice_center_offset(c_m) in (1, 3)
        assert o.is_center_aligned_sb_solved(c_m) is False

    def test_lse_eo_and_ul_ur_queries(self):
        o = get_orientation("")
        c = CubeState()

        # Solved state
        assert o.count_bad_edges(c) == 0
        assert o.is_eo_solved(c) is True
        assert o.is_ul_ur_solved(c) is True

        # UL/UR solved up to AUF (U, U2, U')
        for auf in ("U", "U2", "U'"):
            c_auf = c.copy()
            c_auf.apply_move(auf)
            assert o.is_ul_ur_solved(c_auf) is True

        # Swapping UL and UR (via M2 U2 M2 U2 on solved cube)
        c_swap = c.copy()
        c_swap.apply_moves("M2 U2 M2 U2")
        # In this state, UL and UR are in their slots but let's check
        # Wait, M2 swaps UF/UB and DF/DB; UL and UR are on U face sides, unaffected by M2.
        # Let's swap UL and UR explicitly:
        c_swapped_lr = c.copy()
        c_swapped_lr.ep[Edge.UL], c_swapped_lr.ep[Edge.UR] = (
            c_swapped_lr.ep[Edge.UR].copy(),
            c_swapped_lr.ep[Edge.UL].copy(),
        )
        assert o.is_ul_ur_solved(c_swapped_lr) is False


class TestSBPlacementExtraction:
    """Seam 4: Second Block Placement Extraction across orientations."""

    def test_extract_placement_solved_cube(self):
        o = get_orientation("")
        c = CubeState()
        placement = o.extract_sb_placement(c)

        assert isinstance(placement, SBPlacement)
        assert placement.dr_slot == Edge.DR
        assert placement.dr_eo == 0
        assert placement.fr_slot == Edge.FR
        assert placement.fr_eo == 0
        assert placement.br_slot == Edge.BR
        assert placement.br_eo == 0
        assert placement.dfr_slot == Corner.DFR
        assert placement.dfr_co == 0
        assert placement.dbr_slot == Corner.DRB
        assert placement.dbr_co == 0

    def test_extract_placement_across_all_24_orientations(self):
        """When an oriented cube has SB solved, extract_sb_placement returns solved slot indices and deltas 0."""
        for o in get_all_orientations():
            c = CubeState()
            if o.rotations:
                c.apply_moves(o.rotations)

            placement = o.extract_sb_placement(c)
            assert placement.dr_slot == Edge.DR
            assert placement.dr_eo == 0
            assert placement.fr_slot == Edge.FR
            assert placement.fr_eo == 0
            assert placement.br_slot == Edge.BR
            assert placement.br_eo == 0
            assert placement.dfr_slot == Corner.DFR
            assert placement.dfr_co == 0
            assert placement.dbr_slot == Corner.DRB
            assert placement.dbr_co == 0

    def test_extract_placement_with_scramble(self):
        """Move applied to SB pieces must be reflected in the extracted slots and orientation deltas."""
        o = get_orientation("")
        c = CubeState()
        # R moves DRB corner, DFR corner, FR edge, BR edge, but not DR edge (DR is on D face, R turn does not affect DR unless r is used)
        # Wait: R face has URF, UBR, DRB, DFR corners; UR, BR, DR?, wait!
        # Let's check R face pieces in constants:
        # Corners on R: URF(3), UBR(2), DRB(6), DFR(7)
        # Edges on R: UR(3), BR(10), DR(7), FR(11)
        # So R turn moves DR to FR, FR to UR, etc.
        c.apply_move("R")
        p = o.extract_sb_placement(c)
        assert p.dr_slot != Edge.DR or p.fr_slot != Edge.FR

    def test_parity_with_existing_sb_solver_extractor(self):
        """Extract placement produces byte-exact matching results with legacy sb_solver extractor."""
        from roux_engine.solver.sb_solver import extract_sb_placement as legacy_extract
        from roux_engine.segmenter.fb_detector import ALL_BLOCK_DEFINITIONS

        test_scrambles = [
            "",
            "R U R' U'",
            "R2 U2 R2 U2",
            "M2 U' M U2 M' U' M2",
            "r U R' U' r' F R F'",
        ]
        for ori in ["", "y", "y2", "y'", "x2", "x2 y", "x2 y2", "x2 y'"]:
            o = get_orientation(ori)
            legacy_block = ALL_BLOCK_DEFINITIONS[ori]
            for scramble in test_scrambles:
                c = CubeState()
                if ori:
                    c.apply_moves(ori)
                if scramble:
                    c.apply_moves(scramble)

                new_p = o.extract_sb_placement(c)
                old_p = legacy_extract(c, legacy_block)

                assert new_p.dr_slot == old_p.dr_slot
                assert new_p.dr_eo == old_p.dr_eo
                assert new_p.fr_slot == old_p.fr_slot
                assert new_p.fr_eo == old_p.fr_eo
                assert new_p.br_slot == old_p.br_slot
                assert new_p.br_eo == old_p.br_eo
                assert new_p.dfr_slot == old_p.dfr_slot
                assert new_p.dfr_co == old_p.dfr_co
                assert new_p.dbr_slot == old_p.dbr_slot
                assert new_p.dbr_co == old_p.dbr_co

    def test_canonical_corner_naming_aliases(self):
        """Verify dfl_piece/dfl_co and dbr_piece/dbr_co aliases match dlf and drb properties."""
        for o in get_all_orientations():
            assert o.dfl_piece == o.dlf_piece
            assert o.dfl_co == o.dlf_co
            assert o.dbr_piece == o.drb_piece
            assert o.dbr_co == o.drb_co

    def test_cross_module_symmetry_resolution(self):
        """Verify get_orientation resolves solver.symmetry.CanonicalSymmetry seamlessly."""
        from roux_engine.solver.symmetry import CanonicalSymmetry as SolverSymmetry
        for s in SolverSymmetry:
            o = get_orientation(s)
            assert o.rotations == s.value




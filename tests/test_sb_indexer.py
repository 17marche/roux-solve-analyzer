"""Unit and property tests for Second Block State Indexers."""

import pytest
import numpy as np
from roux_engine.core.constants import Corner, Edge
from roux_engine.core.cube import CubeState
from roux_engine.solver.sb_indexer import (
    SBIndexer,
    SBPlacement,
    RightBackSquareIndexer,
    RightBackSquarePlacement,
    RightFrontSquareIndexer,
    RightFrontSquarePlacement,
)


class TestSBCornerIndexing:
    """Slice 1: SB corner indexing and unranking (270 states)."""

    def test_solved_corners_map_to_zero(self):
        """Canonical solved SB corner positions and orientations must map to corner index 0."""
        idx = SBIndexer.encode_corners(dfr_slot=Corner.DFR, dfr_co=0, dbr_slot=Corner.DRB, dbr_co=0)
        assert idx == 0

    def test_corner_decoding_zero_yields_solved_corners(self):
        """Corner index 0 must decode back to canonical solved corner positions and orientations."""
        dfr_slot, dfr_co, dbr_slot, dbr_co = SBIndexer.decode_corners(0)
        assert dfr_slot == Corner.DFR
        assert dfr_co == 0
        assert dbr_slot == Corner.DRB
        assert dbr_co == 0

    def test_corner_indexing_bijection(self):
        """All 30 allowed position pairs * 9 orientations = 270 configurations must be strictly bijective in [0, 269]."""
        allowed_slots = (Corner.UFL, Corner.ULB, Corner.UBR, Corner.URF, Corner.DRB, Corner.DFR)
        seen_indices = set()
        for s0 in allowed_slots:
            for s1 in allowed_slots:
                if s0 == s1:
                    continue
                for o0 in range(3):
                    for o1 in range(3):
                        idx = SBIndexer.encode_corners(dfr_slot=s0, dfr_co=o0, dbr_slot=s1, dbr_co=o1)
                        assert 0 <= idx < 270
                        assert idx not in seen_indices
                        seen_indices.add(idx)

                        u_s0, u_o0, u_s1, u_o1 = SBIndexer.decode_corners(idx)
                        assert (u_s0, u_o0, u_s1, u_o1) == (s0, o0, s1, o1)

        assert len(seen_indices) == 270
        assert min(seen_indices) == 0
        assert max(seen_indices) == 269

    def test_corner_index_invalid_inputs(self):
        """Invalid slots, duplicate slots, FB-occupied slots, or out-of-bounds orientations must raise ValueError."""
        # Duplicate slots
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=Corner.DFR, dfr_co=0, dbr_slot=Corner.DFR, dbr_co=0)
        # FB slot occupation (slots 4 and 5 are FB)
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=Corner.DLF, dfr_co=0, dbr_slot=Corner.DRB, dbr_co=0)
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=Corner.DFR, dfr_co=0, dbr_slot=Corner.DBL, dbr_co=0)
        # Out of bounds slot
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=8, dfr_co=0, dbr_slot=Corner.DRB, dbr_co=0)
        # Out of bounds orientation
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=Corner.DFR, dfr_co=3, dbr_slot=Corner.DRB, dbr_co=0)
        with pytest.raises(ValueError):
            SBIndexer.encode_corners(dfr_slot=Corner.DFR, dfr_co=0, dbr_slot=Corner.DRB, dbr_co=-1)
        # Decode out of bounds
        with pytest.raises(ValueError):
            SBIndexer.decode_corners(-1)
        with pytest.raises(ValueError):
            SBIndexer.decode_corners(270)


class TestSBEdgeIndexing:
    """Slice 2: SB edge indexing and unranking (4,032 states)."""

    def test_solved_edges_map_to_zero(self):
        """Canonical solved SB edge positions and orientations must map to edge index 0."""
        idx = SBIndexer.encode_edges(
            dr_slot=Edge.DR, dr_eo=0,
            fr_slot=Edge.FR, fr_eo=0,
            br_slot=Edge.BR, br_eo=0
        )
        assert idx == 0

    def test_edge_decoding_zero_yields_solved_edges(self):
        """Edge index 0 must decode back to canonical solved edge positions and orientations."""
        dr_slot, dr_eo, fr_slot, fr_eo, br_slot, br_eo = SBIndexer.decode_edges(0)
        assert dr_slot == Edge.DR
        assert dr_eo == 0
        assert fr_slot == Edge.FR
        assert fr_eo == 0
        assert br_slot == Edge.BR
        assert br_eo == 0

    def test_edge_indexing_bijection(self):
        """All 504 position triplets * 8 orientations = 4,032 configurations must be strictly bijective in [0, 4031]."""
        allowed_slots = (
            Edge.UF, Edge.UL, Edge.UB, Edge.UR,
            Edge.DF, Edge.DB, Edge.DR,
            Edge.BR, Edge.FR
        )
        seen_indices = set()
        for s0 in allowed_slots:
            for s1 in allowed_slots:
                if s1 == s0:
                    continue
                for s2 in allowed_slots:
                    if s2 == s0 or s2 == s1:
                        continue
                    for o0 in range(2):
                        for o1 in range(2):
                            for o2 in range(2):
                                idx = SBIndexer.encode_edges(
                                    dr_slot=s0, dr_eo=o0,
                                    fr_slot=s1, fr_eo=o1,
                                    br_slot=s2, br_eo=o2
                                )
                                assert 0 <= idx < 4032
                                assert idx not in seen_indices
                                seen_indices.add(idx)

                                u_s0, u_o0, u_s1, u_o1, u_s2, u_o2 = SBIndexer.decode_edges(idx)
                                assert (u_s0, u_o0, u_s1, u_o1, u_s2, u_o2) == (s0, o0, s1, o1, s2, o2)

        assert len(seen_indices) == 4032
        assert min(seen_indices) == 0
        assert max(seen_indices) == 4031

    def test_edge_index_invalid_inputs(self):
        """Invalid slots, duplicate slots, FB-occupied slots, or out-of-bounds orientations must raise ValueError."""
        # Duplicate slots
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.DR, 0, Edge.BR, 0)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.FR, 0, Edge.FR, 0)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.FR, 0, Edge.DR, 0)
        # FB-occupied slots: DL (5), FL (8), BL (9)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DL, 0, Edge.FR, 0, Edge.BR, 0)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.FL, 0, Edge.BR, 0)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.FR, 0, Edge.BL, 0)
        # Slot out of bounds
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(12, 0, Edge.FR, 0, Edge.BR, 0)
        # Orientation out of bounds
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 2, Edge.FR, 0, Edge.BR, 0)
        with pytest.raises(ValueError):
            SBIndexer.encode_edges(Edge.DR, 0, Edge.FR, -1, Edge.BR, 0)
        # Decode out of bounds
        with pytest.raises(ValueError):
            SBIndexer.decode_edges(-1)
        with pytest.raises(ValueError):
            SBIndexer.decode_edges(4032)


class TestCompositeSBIndexing:
    """Slice 3: Composite Second Block indexing and placement extraction."""

    def test_canonical_solved_cube_maps_to_index_zero(self):
        """Canonical solved cube state must map to composite index 0."""
        cube = CubeState()
        idx = SBIndexer.encode(cube)
        assert idx == 0

    def test_extract_placement_solved_cube(self):
        """Extracting placement from a clean cube gives canonical solved coordinates."""
        cube = CubeState()
        p = SBIndexer.extract_placement(cube)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.fr_slot == Edge.FR and p.fr_eo == 0
        assert p.br_slot == Edge.BR and p.br_eo == 0
        assert p.dfr_slot == Corner.DFR and p.dfr_co == 0
        assert p.dbr_slot == Corner.DRB and p.dbr_co == 0
        assert SBIndexer.encode_placement(p) == 0

    def test_decode_to_placement_zero(self):
        """Index 0 decodes to canonical solved placement."""
        p = SBIndexer.decode_to_placement(0)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.fr_slot == Edge.FR and p.fr_eo == 0
        assert p.br_slot == Edge.BR and p.br_eo == 0
        assert p.dfr_slot == Corner.DFR and p.dfr_co == 0
        assert p.dbr_slot == Corner.DRB and p.dbr_co == 0

    def test_decode_zero_yields_solved_sb_and_fb_cube(self):
        """Decoding index 0 produces a CubeState where both FB and SB are solved."""
        from roux_engine.segmenter.sb_detector import SBDetector
        from roux_engine.segmenter.fb_detector import FBDetector
        cube = SBIndexer.decode(0)
        assert SBDetector.is_canonical_sb_solved(cube)
        assert FBDetector.is_canonical_fb_solved(cube)
        assert SBIndexer.encode(cube) == 0

    def test_composite_bounds_and_invalid_index(self):
        """Index must be strictly within [0, 1,088,639]; out of bounds must raise ValueError."""
        with pytest.raises(ValueError):
            SBIndexer.decode(-1)
        with pytest.raises(ValueError):
            SBIndexer.decode(1_088_640)
        with pytest.raises(ValueError):
            SBIndexer.decode_to_placement(-1)
        with pytest.raises(ValueError):
            SBIndexer.decode_to_placement(1_088_640)

    def test_boundary_max_index_roundtrip(self):
        """Boundary index 1,088,639 decodes and encodes back identically."""
        max_idx = 1_088_639
        p = SBIndexer.decode_to_placement(max_idx)
        assert SBIndexer.encode_placement(p) == max_idx
        c = SBIndexer.decode(max_idx)
        assert SBIndexer.encode(c) == max_idx

    def test_random_indices_roundtrip(self):
        """Sampled random indices across the full 1.08M state space roundtrip identically."""
        import random
        rng = random.Random(42)
        sample_size = 500
        for _ in range(sample_size):
            idx = rng.randint(0, SBIndexer.TOTAL_STATES - 1)
            cube = SBIndexer.decode(idx)
            assert len(set(cube.cp.tolist())) == 8
            assert len(set(cube.ep.tolist())) == 12
            assert np.all(cube.co >= 0) and np.all(cube.co < 3)
            assert np.all(cube.eo >= 0) and np.all(cube.eo < 2)

            assert SBIndexer.encode(cube) == idx

    def test_l_moves_preserve_sb_index(self):
        """Moves on the L face do not affect DR, FR, BR, DFR, DBR, so SB index must be invariant."""
        cube = CubeState()
        initial_idx = SBIndexer.encode(cube)
        assert initial_idx == 0

        l_moves = ["L", "L'", "L2", "L", "L2", "L'"]
        for m in l_moves:
            cube.apply_move(m)
            assert SBIndexer.encode(cube) == 0

    def test_ru_scrambled_cubes_roundtrip(self):
        """Scrambles using <R, U> preserve FB and roundtrip cleanly through decode(encode(cube))."""
        import random
        rng = random.Random(1337)
        moves = ["R", "U"]
        modifiers = ["", "'", "2"]

        for _ in range(50):
            scramble = " ".join(rng.choice(moves) + rng.choice(modifiers) for _ in range(30))
            cube = CubeState().apply_moves(scramble)
            idx = SBIndexer.encode(cube)
            assert 0 <= idx < SBIndexer.TOTAL_STATES

            decoded_cube = SBIndexer.decode(idx)
            orig_p = SBIndexer.extract_placement(cube)
            dec_p = SBIndexer.extract_placement(decoded_cube)
            assert orig_p == dec_p
            assert SBIndexer.encode(decoded_cube) == idx


class TestRightBackSquareIndexing:
    """Slice 4: Right Back Square indexing and unranking (5,184 states)."""

    def test_solved_rbs_maps_to_zero(self):
        """Canonical solved RBS positions and orientations must map to index 0."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer, RightBackSquarePlacement
        cube = CubeState()
        assert RightBackSquareIndexer.encode(cube) == 0

        p = RightBackSquareIndexer.extract_placement(cube)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.br_slot == Edge.BR and p.br_eo == 0
        assert p.dbr_slot == Corner.DRB and p.dbr_co == 0
        assert RightBackSquareIndexer.encode_placement(p) == 0

    def test_rbs_decoding_zero_yields_solved_rbs(self):
        """Index 0 must decode to solved RBS placement and CubeState."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer
        from roux_engine.segmenter.sb_detector import SBDetector
        from roux_engine.segmenter.fb_detector import FBDetector

        p = RightBackSquareIndexer.decode_to_placement(0)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.br_slot == Edge.BR and p.br_eo == 0
        assert p.dbr_slot == Corner.DRB and p.dbr_co == 0

        cube = RightBackSquareIndexer.decode(0)
        assert FBDetector.is_canonical_fb_solved(cube)
        assert SBDetector.is_dr_solved(cube)
        assert SBDetector.is_back_pair_solved(cube)
        assert RightBackSquareIndexer.encode(cube) == 0

    def test_rbs_corner_bijection(self):
        """All 6 slots * 3 orientations = 18 corner configurations must be bijective in [0, 17]."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer
        allowed_slots = (Corner.UFL, Corner.ULB, Corner.UBR, Corner.URF, Corner.DRB, Corner.DFR)
        seen = set()
        for slot in allowed_slots:
            for co in range(3):
                idx = RightBackSquareIndexer.encode_corner(slot, co)
                assert 0 <= idx < 18
                assert idx not in seen
                seen.add(idx)

                u_slot, u_co = RightBackSquareIndexer.decode_corner(idx)
                assert (u_slot, u_co) == (slot, co)

        assert len(seen) == 18
        assert min(seen) == 0
        assert max(seen) == 17

    def test_rbs_edge_bijection(self):
        """All 72 slot pairs * 4 orientations = 288 edge configurations must be bijective in [0, 287]."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer
        allowed_slots = (
            Edge.UF, Edge.UL, Edge.UB, Edge.UR,
            Edge.DF, Edge.DB, Edge.DR,
            Edge.BR, Edge.FR
        )
        seen = set()
        for s0 in allowed_slots:
            for s1 in allowed_slots:
                if s0 == s1:
                    continue
                for o0 in range(2):
                    for o1 in range(2):
                        idx = RightBackSquareIndexer.encode_edges(s0, o0, s1, o1)
                        assert 0 <= idx < 288
                        assert idx not in seen
                        seen.add(idx)

                        u_s0, u_o0, u_s1, u_o1 = RightBackSquareIndexer.decode_edges(idx)
                        assert (u_s0, u_o0, u_s1, u_o1) == (s0, o0, s1, o1)

        assert len(seen) == 288
        assert min(seen) == 0
        assert max(seen) == 287

    def test_rbs_exhaustive_composite_bijection(self):
        """All 5,184 composite states roundtrip cleanly through placement and cube decode/encode."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer
        for idx in range(RightBackSquareIndexer.TOTAL_STATES):
            p = RightBackSquareIndexer.decode_to_placement(idx)
            assert RightBackSquareIndexer.encode_placement(p) == idx

            cube = RightBackSquareIndexer.decode(idx)
            assert RightBackSquareIndexer.encode(cube) == idx

    def test_rbs_invalid_inputs(self):
        """Invalid slots, duplicate slots, FB conflicts, or out of bounds values raise ValueError."""
        from roux_engine.solver.sb_indexer import RightBackSquareIndexer
        with pytest.raises(ValueError):
            RightBackSquareIndexer.encode_corner(Corner.DLF, 0)  # FB corner
        with pytest.raises(ValueError):
            RightBackSquareIndexer.encode_corner(Corner.DRB, 3)  # Orientation OOB
        with pytest.raises(ValueError):
            RightBackSquareIndexer.encode_edges(Edge.DR, 0, Edge.DR, 0)  # Duplicate slot
        with pytest.raises(ValueError):
            RightBackSquareIndexer.encode_edges(Edge.DL, 0, Edge.BR, 0)  # FB edge
        with pytest.raises(ValueError):
            RightBackSquareIndexer.encode_edges(Edge.DR, 0, Edge.FL, 0)  # FB edge
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode(-1)
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode(5184)
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode_corner(-1)
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode_corner(18)
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode_edges(-1)
        with pytest.raises(ValueError):
            RightBackSquareIndexer.decode_edges(288)


class TestRightFrontSquareIndexing:
    """Slice 5: Right Front Square indexing and unranking (5,184 states)."""

    def test_solved_rfs_maps_to_zero(self):
        """Canonical solved RFS positions and orientations must map to index 0."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer, RightFrontSquarePlacement
        cube = CubeState()
        assert RightFrontSquareIndexer.encode(cube) == 0

        p = RightFrontSquareIndexer.extract_placement(cube)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.fr_slot == Edge.FR and p.fr_eo == 0
        assert p.dfr_slot == Corner.DFR and p.dfr_co == 0
        assert RightFrontSquareIndexer.encode_placement(p) == 0

    def test_rfs_decoding_zero_yields_solved_rfs(self):
        """Index 0 must decode to solved RFS placement and CubeState."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer
        from roux_engine.segmenter.sb_detector import SBDetector
        from roux_engine.segmenter.fb_detector import FBDetector

        p = RightFrontSquareIndexer.decode_to_placement(0)
        assert p.dr_slot == Edge.DR and p.dr_eo == 0
        assert p.fr_slot == Edge.FR and p.fr_eo == 0
        assert p.dfr_slot == Corner.DFR and p.dfr_co == 0

        cube = RightFrontSquareIndexer.decode(0)
        assert FBDetector.is_canonical_fb_solved(cube)
        assert SBDetector.is_dr_solved(cube)
        assert SBDetector.is_front_pair_solved(cube)
        assert RightFrontSquareIndexer.encode(cube) == 0

    def test_rfs_corner_bijection(self):
        """All 6 slots * 3 orientations = 18 corner configurations must be bijective in [0, 17]."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer
        allowed_slots = (Corner.UFL, Corner.ULB, Corner.UBR, Corner.URF, Corner.DRB, Corner.DFR)
        seen = set()
        for slot in allowed_slots:
            for co in range(3):
                idx = RightFrontSquareIndexer.encode_corner(slot, co)
                assert 0 <= idx < 18
                assert idx not in seen
                seen.add(idx)

                u_slot, u_co = RightFrontSquareIndexer.decode_corner(idx)
                assert (u_slot, u_co) == (slot, co)

        assert len(seen) == 18
        assert min(seen) == 0
        assert max(seen) == 17

    def test_rfs_edge_bijection(self):
        """All 72 slot pairs * 4 orientations = 288 edge configurations must be bijective in [0, 287]."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer
        allowed_slots = (
            Edge.UF, Edge.UL, Edge.UB, Edge.UR,
            Edge.DF, Edge.DB, Edge.DR,
            Edge.BR, Edge.FR
        )
        seen = set()
        for s0 in allowed_slots:
            for s1 in allowed_slots:
                if s0 == s1:
                    continue
                for o0 in range(2):
                    for o1 in range(2):
                        idx = RightFrontSquareIndexer.encode_edges(s0, o0, s1, o1)
                        assert 0 <= idx < 288
                        assert idx not in seen
                        seen.add(idx)

                        u_s0, u_o0, u_s1, u_o1 = RightFrontSquareIndexer.decode_edges(idx)
                        assert (u_s0, u_o0, u_s1, u_o1) == (s0, o0, s1, o1)

        assert len(seen) == 288
        assert min(seen) == 0
        assert max(seen) == 287

    def test_rfs_exhaustive_composite_bijection(self):
        """All 5,184 composite states roundtrip cleanly through placement and cube decode/encode."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer
        for idx in range(RightFrontSquareIndexer.TOTAL_STATES):
            p = RightFrontSquareIndexer.decode_to_placement(idx)
            assert RightFrontSquareIndexer.encode_placement(p) == idx

            cube = RightFrontSquareIndexer.decode(idx)
            assert RightFrontSquareIndexer.encode(cube) == idx

    def test_rfs_invalid_inputs(self):
        """Invalid slots, duplicate slots, FB conflicts, or out of bounds values raise ValueError."""
        from roux_engine.solver.sb_indexer import RightFrontSquareIndexer
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.encode_corner(Corner.DLF, 0)  # FB corner
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.encode_corner(Corner.DFR, 3)  # Orientation OOB
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.encode_edges(Edge.DR, 0, Edge.DR, 0)  # Duplicate slot
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.encode_edges(Edge.DL, 0, Edge.FR, 0)  # FB edge
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.encode_edges(Edge.DR, 0, Edge.BL, 0)  # FB edge
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode(-1)
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode(5184)
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode_corner(-1)
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode_corner(18)
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode_edges(-1)
        with pytest.raises(ValueError):
            RightFrontSquareIndexer.decode_edges(288)


class TestSBIndexerIntegrationAndPerformance:
    """Slice 6: Integration, convenience bindings, and performance benchmarks."""

    def test_convenience_bindings_on_sb_indexer(self):
        """SBIndexer convenience methods map directly to sub-indexers."""
        cube = CubeState()
        assert SBIndexer.encode_back_square(cube) == 0
        assert SBIndexer.encode_front_square(cube) == 0

        rbs_cube = SBIndexer.decode_back_square(0)
        assert SBIndexer.encode_back_square(rbs_cube) == 0

        rfs_cube = SBIndexer.decode_front_square(0)
        assert SBIndexer.encode_front_square(rfs_cube) == 0

        assert SBIndexer.BackSquare is RightBackSquareIndexer
        assert SBIndexer.FrontSquare is RightFrontSquareIndexer

    def test_encode_cube_with_fb_conflicts_raises_error(self):
        """Cubes with SB pieces in FB slots raise descriptive ValueError."""
        # Cube where DFR is in DLF slot (slot 4)
        cube = CubeState()
        cube.cp[Corner.DLF] = Corner.DFR
        cube.cp[Corner.DFR] = Corner.DLF
        with pytest.raises(ValueError, match="DFR corner occupies First Block slot 4"):
            SBIndexer.encode(cube)
        with pytest.raises(ValueError, match="DFR corner occupies First Block slot 4"):
            RightFrontSquareIndexer.encode(cube)

        # Cube where DR is in DL slot (slot 5)
        cube = CubeState()
        cube.ep[Edge.DL] = Edge.DR
        cube.ep[Edge.DR] = Edge.DL
        with pytest.raises(ValueError, match="DR edge occupies First Block slot 5"):
            SBIndexer.encode(cube)
        with pytest.raises(ValueError, match="DR edge occupies First Block slot 5"):
            RightBackSquareIndexer.encode(cube)
        with pytest.raises(ValueError, match="DR edge occupies First Block slot 5"):
            RightFrontSquareIndexer.encode(cube)

    def test_reco_benchmark_scrambles(self):
        """Scrambles followed by FB moves from dataset encode within range and roundtrip."""
        import json, os
        from roux_engine.segmenter.segmenter import RouxSegmenter
        dataset_path = "data/roux_solves.json"
        if not os.path.exists(dataset_path):
            pytest.skip("Dataset not found")

        with open(dataset_path, "r", encoding="utf-8") as f:
            solves = json.load(f)[:15]

        segmenter = RouxSegmenter()
        for item in solves:
            scramble = item.get("scramble", "")
            solution = item.get("solution", "")
            if not scramble or not solution:
                continue

            segmented = segmenter.segment(scramble, solution)
            if not segmented or not segmented.fb:
                continue

            # Apply scramble + FB moves to reach state where FB is solved
            cube = CubeState().apply_moves(scramble).apply_moves(segmented.fb.moves_str)

            # If this solve is canonical FB (white bottom, blue left), SB pieces are outside FB
            from roux_engine.segmenter.fb_detector import FBDetector
            if FBDetector.is_canonical_fb_solved(cube):
                idx = SBIndexer.encode(cube)
                assert 0 <= idx < SBIndexer.TOTAL_STATES
                dec = SBIndexer.decode(idx)
                assert SBIndexer.encode(dec) == idx

                rbs_idx = RightBackSquareIndexer.encode(cube)
                assert 0 <= rbs_idx < RightBackSquareIndexer.TOTAL_STATES
                rbs_dec = RightBackSquareIndexer.decode(rbs_idx)
                assert RightBackSquareIndexer.encode(rbs_dec) == rbs_idx

                rfs_idx = RightFrontSquareIndexer.encode(cube)
                assert 0 <= rfs_idx < RightFrontSquareIndexer.TOTAL_STATES
                rfs_dec = RightFrontSquareIndexer.decode(rfs_idx)
                assert RightFrontSquareIndexer.encode(rfs_dec) == rfs_idx

    def test_encoding_performance(self):
        """Encoding must be fast enough for IDA* search (< 20us per encode, > 50,000 ops/sec)."""
        import time
        cube = CubeState().apply_moves("R U R' U' R' U2 R2 U R2 U' R2")
        n_iters = 10_000

        # Full SB encode
        start = time.perf_counter()
        for _ in range(n_iters):
            SBIndexer.encode(cube)
        dur = time.perf_counter() - start
        sb_us = (dur / n_iters) * 1e6
        assert sb_us < 20.0, f"SBIndexer.encode too slow: {sb_us:.2f} us"

        # Right Back Square encode
        start = time.perf_counter()
        for _ in range(n_iters):
            RightBackSquareIndexer.encode(cube)
        dur = time.perf_counter() - start
        rbs_us = (dur / n_iters) * 1e6
        assert rbs_us < 20.0, f"RightBackSquareIndexer.encode too slow: {rbs_us:.2f} us"

        # Right Front Square encode
        start = time.perf_counter()
        for _ in range(n_iters):
            RightFrontSquareIndexer.encode(cube)
        dur = time.perf_counter() - start
        rfs_us = (dur / n_iters) * 1e6
        assert rfs_us < 20.0, f"RightFrontSquareIndexer.encode too slow: {rfs_us:.2f} us"






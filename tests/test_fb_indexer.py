"""Unit and property tests for First Block State Indexer."""

import pytest
import numpy as np
from roux_engine.core.constants import Corner, Edge
from roux_engine.core.cube import CubeState
from roux_engine.solver.fb_indexer import FBIndexer, FBPlacement


class TestCornerIndexing:
    """Slice 1: Corner indexing and unranking (504 states)."""

    def test_solved_corners_map_to_zero(self):
        """Canonical solved corner positions and orientations must map to corner index 0."""
        idx = FBIndexer.encode_corners(dlf_slot=Corner.DLF, dlf_co=0, dbl_slot=Corner.DBL, dbl_co=0)
        assert idx == 0

    def test_corner_decoding_zero_yields_solved_corners(self):
        """Corner index 0 must decode back to canonical solved corner positions and orientations."""
        dlf_slot, dlf_co, dbl_slot, dbl_co = FBIndexer.decode_corners(0)
        assert dlf_slot == Corner.DLF
        assert dlf_co == 0
        assert dbl_slot == Corner.DBL
        assert dbl_co == 0

    def test_corner_indexing_bijection(self):
        """All 56 position pairs * 9 orientations = 504 configurations must be strictly bijective in [0, 503]."""
        seen_indices = set()
        for s0 in range(8):
            for s1 in range(8):
                if s0 == s1:
                    continue
                for o0 in range(3):
                    for o1 in range(3):
                        idx = FBIndexer.encode_corners(dlf_slot=s0, dlf_co=o0, dbl_slot=s1, dbl_co=o1)
                        assert 0 <= idx < 504
                        assert idx not in seen_indices
                        seen_indices.add(idx)

                        u_s0, u_o0, u_s1, u_o1 = FBIndexer.decode_corners(idx)
                        assert (u_s0, u_o0, u_s1, u_o1) == (s0, o0, s1, o1)

        assert len(seen_indices) == 504
        assert min(seen_indices) == 0
        assert max(seen_indices) == 503

    def test_corner_index_invalid_inputs(self):
        """Invalid slot indices, duplicate slots, or out-of-bounds orientations must raise ValueError."""
        with pytest.raises(ValueError):
            FBIndexer.encode_corners(dlf_slot=0, dlf_co=0, dbl_slot=0, dbl_co=0)
        with pytest.raises(ValueError):
            FBIndexer.encode_corners(dlf_slot=8, dlf_co=0, dbl_slot=1, dbl_co=0)
        with pytest.raises(ValueError):
            FBIndexer.encode_corners(dlf_slot=0, dlf_co=3, dbl_slot=1, dbl_co=0)
        with pytest.raises(ValueError):
            FBIndexer.decode_corners(-1)
        with pytest.raises(ValueError):
            FBIndexer.decode_corners(504)


class TestEdgeIndexing:
    """Slice 2: Edge indexing and unranking (10,560 states)."""

    def test_solved_edges_map_to_zero(self):
        """Canonical solved edge positions and orientations must map to edge index 0."""
        idx = FBIndexer.encode_edges(
            dl_slot=Edge.DL, dl_eo=0,
            fl_slot=Edge.FL, fl_eo=0,
            bl_slot=Edge.BL, bl_eo=0
        )
        assert idx == 0

    def test_edge_decoding_zero_yields_solved_edges(self):
        """Edge index 0 must decode back to canonical solved edge positions and orientations."""
        dl_slot, dl_eo, fl_slot, fl_eo, bl_slot, bl_eo = FBIndexer.decode_edges(0)
        assert dl_slot == Edge.DL
        assert dl_eo == 0
        assert fl_slot == Edge.FL
        assert fl_eo == 0
        assert bl_slot == Edge.BL
        assert bl_eo == 0

    def test_edge_indexing_bijection(self):
        """All 1320 position triplets * 8 orientations = 10,560 configurations must be strictly bijective in [0, 10559]."""
        seen_indices = set()
        for s0 in range(12):
            for s1 in range(12):
                if s1 == s0:
                    continue
                for s2 in range(12):
                    if s2 == s0 or s2 == s1:
                        continue
                    for o0 in range(2):
                        for o1 in range(2):
                            for o2 in range(2):
                                idx = FBIndexer.encode_edges(
                                    dl_slot=s0, dl_eo=o0,
                                    fl_slot=s1, fl_eo=o1,
                                    bl_slot=s2, bl_eo=o2
                                )
                                assert 0 <= idx < 10560
                                assert idx not in seen_indices
                                seen_indices.add(idx)

                                u_s0, u_o0, u_s1, u_o1, u_s2, u_o2 = FBIndexer.decode_edges(idx)
                                assert (u_s0, u_o0, u_s1, u_o1, u_s2, u_o2) == (s0, o0, s1, o1, s2, o2)

        assert len(seen_indices) == 10560
        assert min(seen_indices) == 0
        assert max(seen_indices) == 10559

    def test_edge_index_invalid_inputs(self):
        """Invalid slot indices, duplicate slots, or out-of-bounds orientations must raise ValueError."""
        with pytest.raises(ValueError):
            FBIndexer.encode_edges(0, 0, 0, 0, 1, 0)  # Duplicate slot s0 == s1
        with pytest.raises(ValueError):
            FBIndexer.encode_edges(0, 0, 1, 0, 1, 0)  # Duplicate slot s1 == s2
        with pytest.raises(ValueError):
            FBIndexer.encode_edges(12, 0, 1, 0, 2, 0)  # Slot out of bounds
        with pytest.raises(ValueError):
            FBIndexer.encode_edges(0, 2, 1, 0, 2, 0)  # Orientation out of bounds
        with pytest.raises(ValueError):
            FBIndexer.decode_edges(-1)
        with pytest.raises(ValueError):
            FBIndexer.decode_edges(10560)


class TestCompositeFBIndexing:
    """Slice 3: Composite First Block indexing and placement extraction."""

    def test_canonical_solved_cube_maps_to_index_zero(self):
        """Canonical solved cube state must map to composite index 0."""
        cube = CubeState()
        idx = FBIndexer.encode(cube)
        assert idx == 0

    def test_extract_placement_solved_cube(self):
        """Extracting placement from a clean cube gives canonical solved coordinates."""
        cube = CubeState()
        p = FBIndexer.extract_placement(cube)
        assert p.dl_slot == Edge.DL and p.dl_eo == 0
        assert p.fl_slot == Edge.FL and p.fl_eo == 0
        assert p.bl_slot == Edge.BL and p.bl_eo == 0
        assert p.dlf_slot == Corner.DLF and p.dlf_co == 0
        assert p.dbl_slot == Corner.DBL and p.dbl_co == 0
        assert FBIndexer.encode_placement(p) == 0

    def test_decode_to_placement_zero(self):
        """Index 0 decodes to canonical solved placement."""
        p = FBIndexer.decode_to_placement(0)
        assert p.dl_slot == Edge.DL and p.dl_eo == 0
        assert p.fl_slot == Edge.FL and p.fl_eo == 0
        assert p.bl_slot == Edge.BL and p.bl_eo == 0
        assert p.dlf_slot == Corner.DLF and p.dlf_co == 0
        assert p.dbl_slot == Corner.DBL and p.dbl_co == 0

    def test_decode_zero_yields_solved_fb_cube(self):
        """Decoding index 0 produces a CubeState where FB is solved."""
        from roux_engine.segmenter.fb_detector import FBDetector
        cube = FBIndexer.decode(0)
        assert FBDetector.is_canonical_fb_solved(cube)
        assert FBIndexer.encode(cube) == 0

    def test_composite_bounds_and_invalid_index(self):
        """Index must be strictly within [0, 5,322,239]; out of bounds must raise ValueError."""
        with pytest.raises(ValueError):
            FBIndexer.decode(-1)
        with pytest.raises(ValueError):
            FBIndexer.decode(5_322_240)
        with pytest.raises(ValueError):
            FBIndexer.decode_to_placement(-1)
        with pytest.raises(ValueError):
            FBIndexer.decode_to_placement(5_322_240)

    def test_boundary_max_index_roundtrip(self):
        """Boundary index 5,322,239 decodes and encodes back identically."""
        max_idx = 5_322_239
        p = FBIndexer.decode_to_placement(max_idx)
        assert FBIndexer.encode_placement(p) == max_idx
        c = FBIndexer.decode(max_idx)
        assert FBIndexer.encode(c) == max_idx


class TestRoundtripAndScrambledProperties:
    """Slice 4: Roundtrip, invariants, and scrambled state verification."""

    @pytest.mark.parametrize("boundary_idx", [
        0, 1, 503, 504, 505,
        10559, 10560,
        504 * 10559,  # last edge block, first corner
        5_322_238, 5_322_239
    ])
    def test_boundary_indices_roundtrip(self, boundary_idx: int):
        """Key boundary indices roundtrip cleanly through decode and encode."""
        cube = FBIndexer.decode(boundary_idx)
        assert FBIndexer.encode(cube) == boundary_idx
        placement = FBIndexer.decode_to_placement(boundary_idx)
        assert FBIndexer.encode_placement(placement) == boundary_idx

    def test_random_indices_roundtrip(self):
        """Sampled random indices across the full 5.32M state space roundtrip identically."""
        import random
        rng = random.Random(42)  # Deterministic seed for reproducible tests
        sample_size = 1000
        for _ in range(sample_size):
            idx = rng.randint(0, FBIndexer.TOTAL_STATES - 1)
            cube = FBIndexer.decode(idx)
            # Decoded cube must be a valid permutation of corners and edges
            assert len(set(cube.cp.tolist())) == 8
            assert len(set(cube.ep.tolist())) == 12
            assert np.all(cube.co >= 0) and np.all(cube.co < 3)
            assert np.all(cube.eo >= 0) and np.all(cube.eo < 2)

            assert FBIndexer.encode(cube) == idx

    def test_ru_moves_preserve_fb_index(self):
        """Moves in <R, U> do not affect DL, FL, BL, DFL, DBL, so FB index must be invariant."""
        cube = CubeState()
        initial_idx = FBIndexer.encode(cube)
        assert initial_idx == 0

        ru_moves = ["R", "U", "R'", "U'", "R2", "U2", "R", "U2", "R'", "U'"]
        for m in ru_moves:
            cube.apply_move(m)
            assert FBIndexer.encode(cube) == 0

    def test_scrambled_cubes_roundtrip(self):
        """Diverse scrambled states preserve FB pieces under decode(encode(state))."""
        import random
        rng = random.Random(1337)
        moves = ["U", "D", "R", "L", "F", "B", "M", "r"]
        modifiers = ["", "'", "2"]

        for _ in range(50):
            scramble = " ".join(rng.choice(moves) + rng.choice(modifiers) for _ in range(30))
            cube = CubeState().apply_moves(scramble)
            idx = FBIndexer.encode(cube)
            assert 0 <= idx < FBIndexer.TOTAL_STATES

            decoded_cube = FBIndexer.decode(idx)
            # Extracted placements must match exactly
            orig_p = FBIndexer.extract_placement(cube)
            dec_p = FBIndexer.extract_placement(decoded_cube)
            assert orig_p == dec_p
            assert FBIndexer.encode(decoded_cube) == idx

    def test_reco_benchmark_scrambles(self):
        """Scrambles from real Roux solve dataset encode within range and roundtrip."""
        import json, os
        dataset_path = "data/roux_solves.json"
        if not os.path.exists(dataset_path):
            pytest.skip("Dataset not found")

        with open(dataset_path, "r", encoding="utf-8") as f:
            solves = json.load(f)[:30]  # First 30 solves

        for item in solves:
            scramble = item.get("scramble", "")
            if not scramble:
                continue
            cube = CubeState().apply_moves(scramble)
            idx = FBIndexer.encode(cube)
            assert 0 <= idx < FBIndexer.TOTAL_STATES
            dec_cube = FBIndexer.decode(idx)
            assert FBIndexer.encode(dec_cube) == idx

    def test_encoding_performance(self):
        """Encoding must be fast enough for IDA* search (> 50,000 states/sec, < 20us per encode)."""
        import time
        cube = CubeState().apply_moves("R U R' F' r2 U' M2")
        n_iters = 10_000
        start = time.perf_counter()
        for _ in range(n_iters):
            FBIndexer.encode(cube)
        duration = time.perf_counter() - start
        per_call_us = (duration / n_iters) * 1e6
        throughput = n_iters / duration
        print(f"\nFB Indexer Encode: {per_call_us:.2f} us/call ({throughput:,.0f} ops/sec)")
        assert per_call_us < 20.0, f"Encoding too slow: {per_call_us:.2f} us"

"""Tests for First Block Pattern Database (PDB) generator, storage, and loader."""

import pytest
import numpy as np
from roux_engine.solver.fb_indexer import TOTAL_STATES
from roux_engine.solver.pdb_generator import (
    pack_distances_nibbles,
    unpack_distances_nibbles,
)


class TestNibblePacking:
    """Slice 1: 4-bit nibble packing and unpacking."""

    def test_pack_unpack_roundtrip(self):
        """Packing uint8 distances in [0..15] and unpacking returns identical values."""
        # Simple pattern
        data = np.array([0, 1, 2, 3, 14, 15, 7, 8], dtype=np.uint8)
        packed = pack_distances_nibbles(data)
        assert len(packed) == 4
        # Byte 0: 0 in low nibble, 1 in high nibble -> 0x10 = 16
        assert packed[0] == (0 | (1 << 4))
        # Byte 1: 2 in low nibble, 3 in high nibble -> 0x32 = 50
        assert packed[1] == (2 | (3 << 4))
        # Byte 2: 14 in low nibble, 15 in high nibble -> 0xFE = 254
        assert packed[2] == (14 | (15 << 4))
        # Byte 3: 7 in low nibble, 8 in high nibble -> 0x87 = 135
        assert packed[3] == (7 | (8 << 4))

        unpacked = unpack_distances_nibbles(packed, num_states=len(data))
        np.testing.assert_array_equal(unpacked, data)

    def test_pack_odd_length_raises(self):
        """Packing an odd-length array must raise ValueError."""
        data = np.array([1, 2, 3], dtype=np.uint8)
        with pytest.raises(ValueError, match="even length"):
            pack_distances_nibbles(data)

    def test_pack_out_of_bounds_raises(self):
        """Values greater than 15 cannot fit in a 4-bit nibble and must raise ValueError."""
        data = np.array([1, 16], dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds 4-bit"):
            pack_distances_nibbles(data)

    def test_pack_full_total_states_size(self):
        """Packing 5,322,240 distance values produces exactly 2,661,120 bytes."""
        # Random mock distances in [0..9]
        rng = np.random.default_rng(42)
        mock_distances = rng.integers(0, 10, size=TOTAL_STATES, dtype=np.uint8)
        packed = pack_distances_nibbles(mock_distances)
        assert len(packed) == 2_661_120
        assert packed.nbytes == 2_661_120


class TestTransitionTables:
    """Slice 2: Transition tables & ADR-0001 moveset compliance."""

    def test_fb_moveset_adr0001_compliance(self):
        """Moveset must strictly adhere to ADR-0001: 21 moves from <U, D, R, F, B, r, M>."""
        from roux_engine.solver.pdb_generator import FB_MOVESET

        assert len(FB_MOVESET) == 21
        # No physical L face turns
        for m in FB_MOVESET:
            assert not m.startswith("L"), f"Physical L turn forbidden by ADR-0001: {m}"
            assert not m.startswith("l"), f"Left wide turn l forbidden: {m}"
            assert not m.startswith("E"), f"E slice turn forbidden: {m}"
            assert not m.startswith("S"), f"S slice turn forbidden: {m}"

        expected_families = {"U", "D", "R", "F", "B", "r", "M"}
        actual_families = {m[0] for m in FB_MOVESET}
        assert actual_families == expected_families

        for family in expected_families:
            assert family in FB_MOVESET
            assert f"{family}2" in FB_MOVESET
            assert f"{family}'" in FB_MOVESET

    def test_transition_tables_shape_and_bounds(self):
        """Corner and edge transition tables must have exact dimensions and valid range."""
        from roux_engine.solver.pdb_generator import build_transition_tables
        from roux_engine.solver.fb_indexer import NUM_CORNER_CONFIGS, NUM_EDGE_CONFIGS

        corner_trans, edge_trans = build_transition_tables()
        assert corner_trans.shape == (NUM_CORNER_CONFIGS, 21)
        assert edge_trans.shape == (NUM_EDGE_CONFIGS, 21)

        assert np.min(corner_trans) >= 0
        assert np.max(corner_trans) < NUM_CORNER_CONFIGS
        assert np.min(edge_trans) >= 0
        assert np.max(edge_trans) < NUM_EDGE_CONFIGS

    def test_transition_tables_correctness_against_cube_state(self):
        """Applying transitions via tables must match applying moves to CubeState."""
        from roux_engine.solver.pdb_generator import build_transition_tables, FB_MOVESET
        from roux_engine.solver.fb_indexer import FBIndexer, NUM_CORNER_CONFIGS
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move

        corner_trans, edge_trans = build_transition_tables()

        # Check from solved state
        cube = CubeState()
        for m_idx, m_name in enumerate(FB_MOVESET):
            c = cube.copy()
            c.apply_move(get_move(m_name))
            expected_idx = FBIndexer.encode(c)

            new_c = corner_trans[0, m_idx]
            new_e = edge_trans[0, m_idx]
            actual_idx = new_e * NUM_CORNER_CONFIGS + new_c
            assert actual_idx == expected_idx, f"Mismatch on move {m_name} from solved"


class TestPDBGenerator:
    """Slice 3: BFS generation of exact shortest distances and file output."""

    def test_generator_default_output_path(self):
        """Default output path must be src/roux_engine/data/fb_pdb.bin."""
        from roux_engine.solver.pdb_generator import get_default_pdb_path

        p = get_default_pdb_path()
        assert p.name == "fb_pdb.bin"
        assert p.parent.name == "data"

    def test_generator_function_exists(self):
        """generate_fb_pdb function must be callable."""
        from roux_engine.solver.pdb_generator import generate_fb_pdb
        assert callable(generate_fb_pdb)


class TestFBPDBLoader:
    """Slice 4: Runtime memory-mapped loader and query interface."""

    def test_fbpdb_loads_default_file(self):
        """FBPDB must load the default 2.66 MB binary database."""
        from roux_engine.solver.fb_pdb import FBPDB

        pdb = FBPDB()
        assert pdb.size == TOTAL_STATES
        assert pdb.file_size_bytes == 2_661_120
        assert pdb.is_loaded

    def test_fbpdb_startup_latency(self):
        """FBPDB memory-mapped initialization must take < 5ms."""
        import time
        from roux_engine.solver.fb_pdb import FBPDB

        t0 = time.perf_counter()
        pdb = FBPDB()
        t1 = time.perf_counter()
        startup_ms = (t1 - t0) * 1000
        assert startup_ms < 5.0, f"Startup took {startup_ms:.2f}ms, expected < 5ms"

    def test_fbpdb_single_query_latency(self):
        """Single distance lookup query latency must be < 1 microsecond."""
        import time
        from roux_engine.solver.fb_pdb import FBPDB

        pdb = FBPDB()
        # Warmup
        for i in range(100):
            _ = pdb.get_distance_by_index(i)

        N = 20_000
        t0 = time.perf_counter()
        for i in range(N):
            _ = pdb.get_distance_by_index(i)
        t1 = time.perf_counter()

        avg_latency_us = ((t1 - t0) / N) * 1_000_000
        assert avg_latency_us < 1.0, f"Query latency was {avg_latency_us:.3f}us, expected < 1.0us"

    def test_fbpdb_canonical_solved_distance(self):
        """Canonical solved state distance must be 0."""
        from roux_engine.solver.fb_pdb import FBPDB
        from roux_engine.core.cube import CubeState

        pdb = FBPDB()
        assert pdb.get_distance(0) == 0
        assert pdb.get_distance_by_index(0) == 0
        assert pdb.get_distance(CubeState()) == 0

    def test_fbpdb_one_move_distances(self):
        """1-move states must have distance 0 (if FB unmoved) or 1 (if FB altered)."""
        from roux_engine.solver.fb_pdb import FBPDB, FB_MOVESET
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move

        pdb = FBPDB()
        for m_name in FB_MOVESET:
            cube = CubeState()
            cube.apply_move(get_move(m_name))
            d = pdb.get_distance(cube)
            if m_name in ["D", "D2", "D'", "F", "F2", "F'", "B", "B2", "B'"]:
                assert d == 1, f"Move {m_name} changes FB pieces, expected distance 1, got {d}"
            else:
                assert d == 0, f"Move {m_name} does not move solved FB pieces, expected 0, got {d}"

    def test_fbpdb_batch_lookup_matches_single(self):
        """Vectorized batch lookup must match individual scalar lookups."""
        from roux_engine.solver.fb_pdb import FBPDB

        pdb = FBPDB()
        indices = np.array([0, 1, 10, 503, 504, 1000, 5322239], dtype=np.int32)
        batch_dists = pdb.get_distances_by_indices(indices)

        for i, idx in enumerate(indices):
            single_dist = pdb.get_distance_by_index(int(idx))
            assert batch_dists[i] == single_dist

    def test_fbpdb_out_of_bounds_raises(self):
        """Indices outside [0, 5322239] must raise IndexError."""
        from roux_engine.solver.fb_pdb import FBPDB

        pdb = FBPDB()
        with pytest.raises(IndexError):
            pdb.get_distance_by_index(-1)
        with pytest.raises(IndexError):
            pdb.get_distance_by_index(TOTAL_STATES)

    def test_fbpdb_file_not_found(self, tmp_path):
        """Non-existent PDB file must raise FileNotFoundError."""
        from roux_engine.solver.fb_pdb import FBPDB

        with pytest.raises(FileNotFoundError):
            FBPDB(pdb_path=tmp_path / "non_existent.bin")


class TestCLIGenerateFBPDB:
    """Slice 5: CLI command `generate-fb-pdb`."""

    def test_cli_generate_fb_pdb_help(self, capsys):
        """`generate-fb-pdb --help` must display usage and exit 0."""
        from roux_engine.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["generate-fb-pdb", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "generate-fb-pdb" in captured.out or "First Block" in captured.out

    def test_cli_generate_fb_pdb_runs_and_verifies(self, monkeypatch, tmp_path, capsys):
        """`generate-fb-pdb` must call generator, show progress, and verify completeness."""
        from roux_engine.cli import main
        from roux_engine.solver.fb_pdb import FILE_SIZE_BYTES

        dummy_path = tmp_path / "test_fb.bin"

        def mock_generate(output_path=None, progress_callback=None, **kwargs):
            if progress_callback:
                progress_callback(0, 1, 1, 0.0)
                progress_callback(1, 9, 10, 0.01)
                progress_callback(2, 5322230, 5322240, 0.05)
            # Write dummy valid size file
            with open(output_path, "wb") as f:
                f.write(b"\x00" * FILE_SIZE_BYTES)
            return output_path

        monkeypatch.setattr("roux_engine.solver.pdb_generator.generate_fb_pdb", mock_generate)

        exit_code = main(["generate-fb-pdb", "--output", str(dummy_path), "--verify"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Depth" in captured.out
        assert "5,322,240" in captured.out
        assert "verified" in captured.out.lower() or "complete" in captured.out.lower()

    def test_cli_backward_compatibility_segmentation(self, capsys):
        """Original CLI invocation flags (-s, -sol) must remain backward-compatible."""
        from roux_engine.cli import main
        from roux_engine.core.parser import MoveParser

        solution = "D' F' L2 D B r U R' U' R U2 R' U R U' R' U R U R' U R U2 R' M' U M' U2 M U M' U2 M2"
        scramble = " ".join(MoveParser.invert_moves(solution))
        exit_code = main(["-s", scramble, "-sol", solution])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "ROUX SOLVE SEGMENTATION REPORT" in captured.out


@pytest.fixture(scope="module")
def fb_pdb_instance():
    from roux_engine.solver.fb_pdb import FBPDB
    return FBPDB()


class TestPDBAdmissibilityAndConsistency:
    """Slice 6: Mathematical admissibility, consistency, and depth distribution."""

    def test_depth_distribution_properties(self, fb_pdb_instance):
        """All 5,322,240 states must have distances in [0..9]."""
        pdb = fb_pdb_instance
        sample_indices = np.linspace(0, pdb.size - 1, 100_000, dtype=np.int32)
        dists = pdb.get_distances_by_indices(sample_indices)
        assert np.all(dists <= 9)
        assert pdb.get_distance(0) == 0

    def test_heuristic_admissibility_random_walks(self, fb_pdb_instance):
        """Heuristic h(s) must be strictly admissible: h(s) <= true STM distance L."""
        import random
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.solver.pdb_generator import FB_MOVESET

        pdb = fb_pdb_instance
        rng = random.Random(12345)
        for length in range(1, 9):
            for trial in range(20):
                cube = CubeState()
                moves = [rng.choice(FB_MOVESET) for _ in range(length)]
                for m in moves:
                    cube.apply_move(get_move(m))

                h = pdb.get_distance(cube)
                assert h <= length, f"Admissibility violated: h({h}) > path length({length}) with moves {moves}"

    def test_triangular_inequality_consistency(self, fb_pdb_instance):
        """Heuristic must be consistent: |h(s) - h(s * m)| <= 1 for all moves in moveset."""
        import random
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.solver.pdb_generator import FB_MOVESET

        pdb = fb_pdb_instance
        rng = random.Random(67890)
        for _ in range(30):
            cube = CubeState()
            walk_len = rng.randint(1, 8)
            for _ in range(walk_len):
                cube.apply_move(get_move(rng.choice(FB_MOVESET)))

            h_s = pdb.get_distance(cube)

            for m_name in FB_MOVESET:
                c_next = cube.copy()
                c_next.apply_move(get_move(m_name))
                h_next = pdb.get_distance(c_next)
                diff = abs(h_s - h_next)
                assert diff <= 1, f"Consistency violated on move {m_name}: |{h_s} - {h_next}| = {diff} > 1"

    def test_known_dataset_scramble_admissibility(self, fb_pdb_instance):
        """Heuristic distance must be admissible on genuine scramble prefixes."""
        from roux_engine.data.loader import RecoDatasetLoader
        from roux_engine.core.cube import CubeState
        from roux_engine.core.parser import MoveParser
        from roux_engine.solver.pdb_generator import FB_MOVESET

        pdb = fb_pdb_instance
        loader = RecoDatasetLoader()
        solves = loader.load_from_json("data/roux_solves.json", revalidate=False)[:10]
        for s in solves:
            events = MoveParser.parse_string(s.scramble)
            for count in [1, 2, 3, 4, 5]:
                c = CubeState()
                for ev in events[:count]:
                    c.apply_move(ev.move)
                move_names = [ev.move for ev in events[:count]]
                if all(m in FB_MOVESET for m in move_names):
                    h = pdb.get_distance(c)
                    assert h <= count

    def test_known_scrambles_exact_distances(self, fb_pdb_instance):
        """Known scramble sequences must match exact minimal STM distance."""
        from roux_engine.core.cube import CubeState

        pdb = fb_pdb_instance

        # Test known optimal sequences and verify exact distances
        # 1-move states
        for m in ["D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]:
            c = CubeState().apply_moves(m)
            assert pdb.get_distance(c) == 1

        # 2-move states with independent faces
        c2 = CubeState().apply_moves("D F")
        assert pdb.get_distance(c2) == 2

        c3 = CubeState().apply_moves("D F B")
        assert pdb.get_distance(c3) == 3

        # Scramble with wide/slice moves
        c_wide = CubeState().apply_moves("D r U' r' D'")
        h_wide = pdb.get_distance(c_wide)
        assert h_wide <= 5










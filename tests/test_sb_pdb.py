"""Tests for Second Block and Right Square Pattern Databases (PDB)."""

import pytest
import numpy as np


class TestSBMovesetAndTransitions:
    """Slice 1: SB Moveset compliance and transition tables."""

    def test_sb_moveset_compliance(self):
        """Moveset must strictly consist of 12 moves from <R, U, r, M>."""
        from roux_engine.solver.sb_pdb_generator import SB_MOVESET

        assert len(SB_MOVESET) == 12
        expected_moves = (
            "R", "R2", "R'",
            "U", "U2", "U'",
            "r", "r2", "r'",
            "M", "M2", "M'",
        )
        assert SB_MOVESET == expected_moves

    def test_sb_moveset_preserves_first_block(self):
        """All 12 moves in SB_MOVESET must leave all First Block pieces untouched."""
        from roux_engine.solver.sb_pdb_generator import SB_MOVESET
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.core.constants import Corner, Edge, Center

        for m_name in SB_MOVESET:
            cube = CubeState()
            cube.apply_move(get_move(m_name))

            # FB corners: DLF (4), DBL (5)
            assert cube.cp[Corner.DLF] == Corner.DLF, f"{m_name} moved DLF corner"
            assert cube.co[Corner.DLF] == 0, f"{m_name} oriented DLF corner"
            assert cube.cp[Corner.DBL] == Corner.DBL, f"{m_name} moved DBL corner"
            assert cube.co[Corner.DBL] == 0, f"{m_name} oriented DBL corner"

            # FB edges: DL (5), FL (8), BL (9)
            assert cube.ep[Edge.DL] == Edge.DL, f"{m_name} moved DL edge"
            assert cube.eo[Edge.DL] == 0, f"{m_name} oriented DL edge"
            assert cube.ep[Edge.FL] == Edge.FL, f"{m_name} moved FL edge"
            assert cube.eo[Edge.FL] == 0, f"{m_name} oriented FL edge"
            assert cube.ep[Edge.BL] == Edge.BL, f"{m_name} moved BL edge"
            assert cube.eo[Edge.BL] == 0, f"{m_name} oriented BL edge"

            # FB center: L (4)
            assert cube.centers[Center.L] == Center.L, f"{m_name} moved L center"

    def test_sb_transition_tables_shape_and_bounds(self):
        """Corner and edge transition tables for full SB must have exact dimensions and valid ranges."""
        from roux_engine.solver.sb_pdb_generator import build_sb_transition_tables
        from roux_engine.solver.sb_indexer import NUM_SB_CORNER_CONFIGS, NUM_SB_EDGE_CONFIGS

        c_trans, e_trans = build_sb_transition_tables()
        assert c_trans.shape == (NUM_SB_CORNER_CONFIGS, 12)
        assert e_trans.shape == (NUM_SB_EDGE_CONFIGS, 12)

        assert np.min(c_trans) >= 0
        assert np.max(c_trans) < NUM_SB_CORNER_CONFIGS
        assert np.min(e_trans) >= 0
        assert np.max(e_trans) < NUM_SB_EDGE_CONFIGS

    def test_sb_transition_tables_correctness_against_cube_state(self):
        """Applying transitions via SB tables must match applying moves to CubeState."""
        from roux_engine.solver.sb_pdb_generator import build_sb_transition_tables, SB_MOVESET
        from roux_engine.solver.sb_indexer import SBIndexer, NUM_SB_CORNER_CONFIGS
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move

        c_trans, e_trans = build_sb_transition_tables()

        # Check from solved state
        cube = CubeState()
        for m_idx, m_name in enumerate(SB_MOVESET):
            c = cube.copy()
            c.apply_move(get_move(m_name))
            expected_idx = SBIndexer.encode(c)

            new_c = c_trans[0, m_idx]
            new_e = e_trans[0, m_idx]
            actual_idx = new_e * NUM_SB_CORNER_CONFIGS + new_c
            assert actual_idx == expected_idx, f"Mismatch on move {m_name} from solved"

    def test_rbs_and_rfs_transition_tables_shape_and_correctness(self):
        """Transition tables for RBS and RFS must match cube transitions."""
        from roux_engine.solver.sb_pdb_generator import (
            build_rbs_transition_tables,
            build_rfs_transition_tables,
            SB_MOVESET,
        )
        from roux_engine.solver.sb_indexer import (
            RightBackSquareIndexer,
            NUM_RBS_CORNER_CONFIGS,
            NUM_RBS_EDGE_CONFIGS,
            RightFrontSquareIndexer,
            NUM_RFS_CORNER_CONFIGS,
            NUM_RFS_EDGE_CONFIGS,
        )
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move

        rbs_c, rbs_e = build_rbs_transition_tables()
        assert rbs_c.shape == (NUM_RBS_CORNER_CONFIGS, 12)
        assert rbs_e.shape == (NUM_RBS_EDGE_CONFIGS, 12)

        rfs_c, rfs_e = build_rfs_transition_tables()
        assert rfs_c.shape == (NUM_RFS_CORNER_CONFIGS, 12)
        assert rfs_e.shape == (NUM_RFS_EDGE_CONFIGS, 12)

        cube = CubeState()
        for m_idx, m_name in enumerate(SB_MOVESET):
            c = cube.copy()
            c.apply_move(get_move(m_name))

            expected_rbs = RightBackSquareIndexer.encode(c)
            actual_rbs = rbs_e[0, m_idx] * NUM_RBS_CORNER_CONFIGS + rbs_c[0, m_idx]
            assert actual_rbs == expected_rbs, f"RBS mismatch on move {m_name}"

            expected_rfs = RightFrontSquareIndexer.encode(c)
            actual_rfs = rfs_e[0, m_idx] * NUM_RFS_CORNER_CONFIGS + rfs_c[0, m_idx]
            assert actual_rfs == expected_rfs, f"RFS mismatch on move {m_name}"


class TestSBPDBGenerators:
    """Slice 2: Generation of packed binary tables via BFS."""

    def test_default_file_paths(self):
        """Default binary file paths must point to src/roux_engine/data/."""
        from roux_engine.solver.sb_pdb_generator import (
            get_default_sb_pdb_path,
            get_default_rbs_pdb_path,
            get_default_rfs_pdb_path,
        )

        sb_path = get_default_sb_pdb_path()
        rbs_path = get_default_rbs_pdb_path()
        rfs_path = get_default_rfs_pdb_path()

        assert sb_path.name == "sb_pdb.bin"
        assert sb_path.parent.name == "data"

        assert rbs_path.name == "rbs_pdb.bin"
        assert rbs_path.parent.name == "data"

        assert rfs_path.name == "rfs_pdb.bin"
        assert rfs_path.parent.name == "data"

    def test_generate_rbs_pdb_creates_valid_file(self, tmp_path):
        """Generating RBS PDB creates a 2,592-byte packed binary table with 0 unvisited states."""
        from roux_engine.solver.sb_pdb_generator import generate_rbs_pdb
        from roux_engine.solver.pdb_generator import unpack_distances_nibbles
        from roux_engine.solver.sb_indexer import TOTAL_RBS_STATES

        out_file = tmp_path / "rbs_test.bin"
        result_path = generate_rbs_pdb(output_path=out_file)
        assert result_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size == TOTAL_RBS_STATES // 2  # 2,592 bytes

        raw = np.fromfile(out_file, dtype=np.uint8)
        dists = unpack_distances_nibbles(raw, num_states=TOTAL_RBS_STATES)
        assert dists[0] == 0  # Solved state
        assert np.max(dists) <= 10
        assert np.all(dists != 15)  # No unvisited states

    def test_generate_rfs_pdb_creates_valid_file(self, tmp_path):
        """Generating RFS PDB creates a 2,592-byte packed binary table with 0 unvisited states."""
        from roux_engine.solver.sb_pdb_generator import generate_rfs_pdb
        from roux_engine.solver.pdb_generator import unpack_distances_nibbles
        from roux_engine.solver.sb_indexer import TOTAL_RFS_STATES

        out_file = tmp_path / "rfs_test.bin"
        result_path = generate_rfs_pdb(output_path=out_file)
        assert result_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size == TOTAL_RFS_STATES // 2  # 2,592 bytes

        raw = np.fromfile(out_file, dtype=np.uint8)
        dists = unpack_distances_nibbles(raw, num_states=TOTAL_RFS_STATES)
        assert dists[0] == 0  # Solved state
        assert np.max(dists) <= 10
        assert np.all(dists != 15)

    def test_generate_sb_pdb_creates_valid_file(self, tmp_path):
        """Generating full SB PDB creates a 544,320-byte packed binary table."""
        from roux_engine.solver.sb_pdb_generator import generate_sb_pdb
        from roux_engine.solver.pdb_generator import unpack_distances_nibbles
        from roux_engine.solver.sb_indexer import TOTAL_SB_STATES

        out_file = tmp_path / "sb_test.bin"
        progress_calls = []

        def callback(depth, new_s, total_s, elapsed):
            progress_calls.append((depth, new_s, total_s))

        result_path = generate_sb_pdb(output_path=out_file, progress_callback=callback)
        assert result_path == out_file
        assert out_file.exists()
        assert out_file.stat().st_size == TOTAL_SB_STATES // 2  # 544,320 bytes
        assert len(progress_calls) > 0
        assert progress_calls[-1][2] == TOTAL_SB_STATES

        raw = np.fromfile(out_file, dtype=np.uint8)
        dists = unpack_distances_nibbles(raw, num_states=TOTAL_SB_STATES)
        assert dists[0] == 0  # Solved state
        assert np.max(dists) <= 14
        assert np.all(dists != 15)

    def test_generate_all_sb_pdbs_total_footprint(self, tmp_path):
        """Total combined on-disk footprint of all three PDB assets is strictly under 600 KB."""
        from roux_engine.solver.sb_pdb_generator import generate_all_sb_pdbs

        paths = generate_all_sb_pdbs(output_dir=tmp_path)
        assert len(paths) == 3

        total_bytes = sum(p.stat().st_size for p in paths)
        assert total_bytes == 544_320 + 2_592 + 2_592  # Exactly 549,504 bytes
        assert total_bytes < 600 * 1024  # Strictly under 600 KB


class TestSBPDBLoaders:
    """Slice 3: Runtime memory-mapped loaders and query interfaces."""

    def test_sb_pdb_loads_default_file(self):
        """SBPDB must load the default 544 KB binary database."""
        from roux_engine.solver.sb_pdb import SBPDB
        from roux_engine.solver.sb_indexer import TOTAL_SB_STATES

        pdb = SBPDB()
        assert pdb.size == TOTAL_SB_STATES
        assert pdb.file_size_bytes == 544_320
        assert pdb.is_loaded

    def test_rbs_and_rfs_pdb_load_default_files(self):
        """RightBackSquarePDB and RightFrontSquarePDB must load their 2.6 KB tables."""
        from roux_engine.solver.sb_pdb import RightBackSquarePDB, RightFrontSquarePDB
        from roux_engine.solver.sb_indexer import TOTAL_RBS_STATES, TOTAL_RFS_STATES

        rbs_pdb = RightBackSquarePDB()
        assert rbs_pdb.size == TOTAL_RBS_STATES
        assert rbs_pdb.file_size_bytes == 2_592
        assert rbs_pdb.is_loaded

        rfs_pdb = RightFrontSquarePDB()
        assert rfs_pdb.size == TOTAL_RFS_STATES
        assert rfs_pdb.file_size_bytes == 2_592
        assert rfs_pdb.is_loaded

    def test_sb_pdb_startup_latency(self):
        """SBPDB memory-mapped initialization must take < 5ms."""
        import time
        from roux_engine.solver.sb_pdb import SBPDB

        t0 = time.perf_counter()
        pdb = SBPDB()
        t1 = time.perf_counter()
        startup_ms = (t1 - t0) * 1000
        assert startup_ms < 5.0, f"Startup took {startup_ms:.2f}ms, expected < 5ms"

    def test_sb_pdb_single_query_latency(self):
        """Single distance lookup query latency must be < 1 microsecond."""
        import time
        from roux_engine.solver.sb_pdb import SBPDB

        pdb = SBPDB()
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

    def test_canonical_solved_distances(self):
        """Canonical solved states must return distance 0 across all loaders."""
        from roux_engine.solver.sb_pdb import SBPDB, RightBackSquarePDB, RightFrontSquarePDB
        from roux_engine.core.cube import CubeState

        sb_pdb = SBPDB()
        rbs_pdb = RightBackSquarePDB()
        rfs_pdb = RightFrontSquarePDB()
        cube = CubeState()

        assert sb_pdb.get_distance(0) == 0
        assert sb_pdb.get_distance(cube) == 0
        assert sb_pdb.get_distance_by_index(0) == 0

        assert rbs_pdb.get_distance(0) == 0
        assert rbs_pdb.get_distance(cube) == 0
        assert rbs_pdb.get_distance_by_index(0) == 0

        assert rfs_pdb.get_distance(0) == 0
        assert rfs_pdb.get_distance(cube) == 0
        assert rfs_pdb.get_distance_by_index(0) == 0

    def test_batch_lookup_matches_single(self):
        """Vectorized batch lookups must match individual scalar lookups."""
        from roux_engine.solver.sb_pdb import SBPDB, RightBackSquarePDB, RightFrontSquarePDB

        sb_pdb = SBPDB()
        indices = np.array([0, 1, 10, 503, 1000, 50000, 1088639], dtype=np.int32)
        batch = sb_pdb.get_distances_by_indices(indices)
        for i, idx in enumerate(indices):
            assert batch[i] == sb_pdb.get_distance_by_index(int(idx))

        rbs_pdb = RightBackSquarePDB()
        rbs_indices = np.array([0, 1, 100, 500, 5183], dtype=np.int32)
        rbs_batch = rbs_pdb.get_distances_by_indices(rbs_indices)
        for i, idx in enumerate(rbs_indices):
            assert rbs_batch[i] == rbs_pdb.get_distance_by_index(int(idx))

    def test_out_of_bounds_raises_index_error(self):
        """Querying out of bounds indices must raise IndexError."""
        from roux_engine.solver.sb_pdb import SBPDB, RightBackSquarePDB, RightFrontSquarePDB

        sb = SBPDB()
        with pytest.raises(IndexError):
            sb.get_distance_by_index(-1)
        with pytest.raises(IndexError):
            sb.get_distance_by_index(sb.size)

        rbs = RightBackSquarePDB()
        with pytest.raises(IndexError):
            rbs.get_distance_by_index(-1)
        with pytest.raises(IndexError):
            rbs.get_distance_by_index(rbs.size)

    def test_file_not_found_raises(self, tmp_path):
        """Missing binary file must raise FileNotFoundError with actionable instructions."""
        from roux_engine.solver.sb_pdb import SBPDB

        with pytest.raises(FileNotFoundError, match="generate-sb-pdb"):
            SBPDB(pdb_path=tmp_path / "nonexistent.bin")


class TestCLIGenerateSBPDB:
    """Slice 4: CLI command `generate-sb-pdb`."""

    def test_cli_generate_sb_pdb_help(self, capsys):
        """`generate-sb-pdb --help` must display usage and exit 0."""
        from roux_engine.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["generate-sb-pdb", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "generate-sb-pdb" in captured.out or "Second Block" in captured.out

    def test_cli_generate_sb_pdb_runs_and_verifies(self, monkeypatch, tmp_path, capsys):
        """`generate-sb-pdb` must call generator, report progress, and verify tables."""
        from roux_engine.cli import main
        from roux_engine.solver.sb_pdb import (
            SB_FILE_SIZE_BYTES,
            RBS_FILE_SIZE_BYTES,
            RFS_FILE_SIZE_BYTES,
        )

        def mock_generate_all(output_dir=None, progress_callback=None):
            out_dir = Path(output_dir) if output_dir else tmp_path
            p_sb = out_dir / "sb_pdb.bin"
            p_rbs = out_dir / "rbs_pdb.bin"
            p_rfs = out_dir / "rfs_pdb.bin"

            if progress_callback:
                progress_callback("sb", 0, 1, 1, 0.0)
                progress_callback("sb", 1, 3, 4, 0.01)
                progress_callback("rbs", 0, 1, 1, 0.0)
                progress_callback("rfs", 0, 1, 1, 0.0)

            with open(p_sb, "wb") as f:
                f.write(b"\x00" * SB_FILE_SIZE_BYTES)
            with open(p_rbs, "wb") as f:
                f.write(b"\x00" * RBS_FILE_SIZE_BYTES)
            with open(p_rfs, "wb") as f:
                f.write(b"\x00" * RFS_FILE_SIZE_BYTES)

            return p_sb, p_rbs, p_rfs

        from pathlib import Path
        monkeypatch.setattr("roux_engine.solver.sb_pdb_generator.generate_all_sb_pdbs", mock_generate_all)

        exit_code = main(["generate-sb-pdb", "--output-dir", str(tmp_path), "--verify"])
        assert exit_code == 0

        captured = capsys.readouterr()
        assert "Second Block" in captured.out or "SB" in captured.out
        assert "verified" in captured.out.lower() or "complete" in captured.out.lower()

    def test_cli_generate_sb_pdb_verify_only(self, capsys):
        """`generate-sb-pdb --verify-only` verifies existing default tables and exits 0."""
        from roux_engine.cli import main

        exit_code = main(["generate-sb-pdb", "--verify-only"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "verified" in captured.out.lower() or "complete" in captured.out.lower()


@pytest.fixture(scope="module")
def sb_pdb_suite():
    from roux_engine.solver.sb_pdb import SBPDB, RightBackSquarePDB, RightFrontSquarePDB
    return SBPDB(), RightBackSquarePDB(), RightFrontSquarePDB()


class TestPDBAdmissibilityAndConsistency:
    """Slice 5: Mathematical admissibility, triangular consistency, and bounds."""

    def test_depth_distributions(self, sb_pdb_suite):
        """All reachable states must fall strictly within verified God's numbers."""
        sb_pdb, rbs_pdb, rfs_pdb = sb_pdb_suite

        # Sample 100k indices for SB
        sample_indices = np.linspace(0, sb_pdb.size - 1, 100_000, dtype=np.int32)
        sb_dists = sb_pdb.get_distances_by_indices(sample_indices)
        assert np.all(sb_dists <= 14)
        assert sb_pdb.get_distance(0) == 0

        # Full check for 5,184 states of RBS and RFS
        rbs_all = rbs_pdb.get_distances_by_indices(np.arange(rbs_pdb.size, dtype=np.int32))
        assert np.all(rbs_all <= 10)
        assert rbs_pdb.get_distance(0) == 0

        rfs_all = rfs_pdb.get_distances_by_indices(np.arange(rfs_pdb.size, dtype=np.int32))
        assert np.all(rfs_all <= 10)
        assert rfs_pdb.get_distance(0) == 0

    def test_heuristic_admissibility_random_walks(self, sb_pdb_suite):
        """Heuristic h(s) must be strictly admissible: h(s) <= true STM move distance."""
        import random
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.solver.sb_pdb import SB_MOVESET

        sb_pdb, rbs_pdb, rfs_pdb = sb_pdb_suite
        rng = random.Random(4242)

        for walk_len in range(1, 10):
            for _ in range(15):
                cube = CubeState()
                moves = [rng.choice(SB_MOVESET) for _ in range(walk_len)]
                for m in moves:
                    cube.apply_move(get_move(m))

                h_sb = sb_pdb.get_distance(cube)
                h_rbs = rbs_pdb.get_distance(cube)
                h_rfs = rfs_pdb.get_distance(cube)

                assert h_sb <= walk_len, f"SB admissibility violated: h={h_sb} > len={walk_len}"
                assert h_rbs <= walk_len, f"RBS admissibility violated: h={h_rbs} > len={walk_len}"
                assert h_rfs <= walk_len, f"RFS admissibility violated: h={h_rfs} > len={walk_len}"

    def test_heuristic_hierarchy_sb_dominates_squares(self, sb_pdb_suite):
        """Full SB distance must always be greater than or equal to sub-square distances."""
        import random
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.solver.sb_pdb import SB_MOVESET

        sb_pdb, rbs_pdb, rfs_pdb = sb_pdb_suite
        rng = random.Random(1337)

        for _ in range(50):
            cube = CubeState()
            walk_len = rng.randint(1, 12)
            for _ in range(walk_len):
                cube.apply_move(get_move(rng.choice(SB_MOVESET)))

            h_sb = sb_pdb.get_distance(cube)
            h_rbs = rbs_pdb.get_distance(cube)
            h_rfs = rfs_pdb.get_distance(cube)

            assert h_sb >= h_rbs, f"Hierarchy violated: h_sb({h_sb}) < h_rbs({h_rbs})"
            assert h_sb >= h_rfs, f"Hierarchy violated: h_sb({h_sb}) < h_rfs({h_rfs})"

    def test_triangular_inequality_consistency(self, sb_pdb_suite):
        """Heuristic must be consistent: |h(s) - h(s * m)| <= 1 for all moves in moveset."""
        import random
        from roux_engine.core.cube import CubeState
        from roux_engine.core.moves import get_move
        from roux_engine.solver.sb_pdb import SB_MOVESET

        sb_pdb, rbs_pdb, rfs_pdb = sb_pdb_suite
        rng = random.Random(9999)

        for _ in range(25):
            cube = CubeState()
            for _ in range(rng.randint(1, 8)):
                cube.apply_move(get_move(rng.choice(SB_MOVESET)))

            h_sb_0 = sb_pdb.get_distance(cube)
            h_rbs_0 = rbs_pdb.get_distance(cube)
            h_rfs_0 = rfs_pdb.get_distance(cube)

            for m_name in SB_MOVESET:
                next_c = cube.copy()
                next_c.apply_move(get_move(m_name))

                assert abs(h_sb_0 - sb_pdb.get_distance(next_c)) <= 1
                assert abs(h_rbs_0 - rbs_pdb.get_distance(next_c)) <= 1
                assert abs(h_rfs_0 - rfs_pdb.get_distance(next_c)) <= 1

    def test_dataset_scramble_prefixes_admissibility(self, sb_pdb_suite):
        """Prefixes of actual dataset scrambles restricted to <R, U, r, M> are strictly admissible."""
        from roux_engine.data.loader import RecoDatasetLoader
        from roux_engine.core.cube import CubeState
        from roux_engine.core.parser import MoveParser
        from roux_engine.solver.sb_pdb import SB_MOVESET

        sb_pdb, rbs_pdb, rfs_pdb = sb_pdb_suite
        loader = RecoDatasetLoader()
        solves = loader.load_from_json("data/roux_solves.json", revalidate=False)[:15]

        for s in solves:
            events = MoveParser.parse_string(s.scramble)
            cube = CubeState()
            count = 0
            for ev in events:
                if ev.move not in SB_MOVESET:
                    break
                cube.apply_move(ev.move)
                count += 1
                if count >= 1:
                    assert sb_pdb.get_distance(cube) <= count
                    assert rbs_pdb.get_distance(cube) <= count
                    assert rfs_pdb.get_distance(cube) <= count
                if count >= 6:
                    break

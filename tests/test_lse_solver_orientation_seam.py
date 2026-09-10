"""Unit tests for the deep orientation-agnostic solve_lse and solve_lse_paths seam."""

import pytest
from roux_engine.core.cube import CubeState
from roux_engine.solver.lse_solver import (
    LSESolution,
    LSEPath,
    solve_lse,
    solve_lse_paths,
)


class TestLSEDataclassOrientationProvenance:
    """Slice 1: Orientation provenance field on LSESolution and LSEPath."""

    def test_lse_solution_orientation_default_empty(self):
        """LSESolution includes orientation field defaulting to empty string."""
        sol = LSESolution(
            target="4a",
            moves=["M"],
            move_count=1,
            case_name="standard_eo",
            center_state="axis_aligned",
        )
        assert hasattr(sol, "orientation")
        assert sol.orientation == ""

    def test_lse_solution_orientation_explicit(self):
        """LSESolution accepts explicit orientation provenance string."""
        sol = LSESolution(
            target="4a",
            moves=["M"],
            move_count=1,
            case_name="standard_eo",
            center_state="axis_aligned",
            orientation="x2",
        )
        assert sol.orientation == "x2"

    def test_lse_path_orientation_default_empty(self):
        """LSEPath includes orientation field defaulting to empty string."""
        sol_4a = LSESolution(target="4a", moves=["M"], move_count=1, case_name="standard_eo")
        sol_4b = LSESolution(target="4b", moves=["U"], move_count=1, case_name="ul_ur")
        sol_4c = LSESolution(target="4c", moves=["M2"], move_count=1, case_name="opp_opp")
        path = LSEPath(
            name="standard",
            step_4a=sol_4a,
            step_4b=sol_4b,
            step_4c=sol_4c,
            total_moves=["M", "U", "M2"],
            total_move_count=3,
        )
        assert hasattr(path, "orientation")
        assert path.orientation == ""

    def test_lse_path_orientation_explicit(self):
        """LSEPath accepts explicit orientation provenance string."""
        sol = LSESolution(target="4a", moves=[], move_count=0, case_name="solved", orientation="y")
        path = LSEPath(
            name="standard",
            step_4a=sol,
            step_4b=sol,
            step_4c=sol,
            total_moves=[],
            total_move_count=0,
            orientation="y",
        )
        assert path.orientation == "y"


class TestLSESolverInputTypes:
    """Slice 2: Public solver functions accept either move sequence strings or instantiated cube states."""

    def test_solve_lse_accepts_move_string(self):
        """solve_lse accepts a move sequence string directly."""
        sols = solve_lse("M'", target="4a")
        assert len(sols) >= 1
        std_sol = [s for s in sols if s.case_name == "standard_eo"][0]
        assert std_sol.moves == ["M"]

    def test_solve_lse_accepts_cube_state(self):
        """solve_lse accepts an instantiated CubeState object."""
        cube = CubeState().apply_move("M'")
        sols = solve_lse(cube, target="4a")
        assert len(sols) >= 1
        std_sol = [s for s in sols if s.case_name == "standard_eo"][0]
        assert std_sol.moves == ["M"]

    def test_solve_lse_accepts_scramble_or_cube_and_cube_kwargs(self):
        """solve_lse accepts scramble_or_cube or legacy cube keyword argument."""
        cube = CubeState().apply_move("M'")
        sols1 = solve_lse(scramble_or_cube="M'", target="4a")
        sols2 = solve_lse(cube=cube, target="4a")
        assert len(sols1) >= 1
        assert len(sols2) >= 1

    def test_solve_lse_paths_accepts_move_string(self):
        """solve_lse_paths accepts a move sequence string directly."""
        paths = solve_lse_paths("M2 U2 M2")
        assert "standard" in paths
        assert "eolr" in paths
        assert "eolr_misoriented" in paths
        assert "eolr_b" in paths

    def test_solve_lse_paths_accepts_cube_state(self):
        """solve_lse_paths accepts an instantiated CubeState object."""
        cube = CubeState().apply_moves("M2 U2 M2")
        paths = solve_lse_paths(cube)
        assert "standard" in paths

    def test_solve_lse_paths_accepts_kwargs(self):
        """solve_lse_paths accepts scramble_or_cube or legacy cube keyword argument."""
        cube = CubeState().apply_moves("M2 U2 M2")
        p1 = solve_lse_paths(scramble_or_cube="M2 U2 M2")
        p2 = solve_lse_paths(cube=cube)
        assert "standard" in p1
        assert "standard" in p2


class TestLSEOrientationAutoDetectAndOverrideInSolveLSE:
    """Slice 3: Orientation auto-detection and explicit override in solve_lse."""

    def test_solve_lse_auto_detects_dual_neutral_inspected(self):
        """solve_lse auto-detects dual-neutral orientations and returns moves in caller frame."""
        from roux_engine.core.orientation import get_orientation, CanonicalSymmetry

        # Yellow-bottom, Green-front, Blue-left ('x2')
        cube_x2 = CubeState().apply_moves("x2 M' U2 M U M2")
        sols = solve_lse(cube_x2, target="lse")
        assert len(sols) == 1
        sol = sols[0]
        assert sol.orientation == "x2"
        # Applying sol.moves to cube_x2 must solve the cube in the x2 frame
        sim = cube_x2.copy().apply_moves(sol.moves)
        ori_x2 = get_orientation("x2")
        assert ori_x2.is_eo_solved(sim)
        assert ori_x2.is_ul_ur_solved(sim)
        assert ori_x2.get_m_slice_center_offset(sim) == 0

    def test_solve_lse_auto_detects_uninspected_world_frame(self):
        """solve_lse auto-detects uninspected world frames and returns translated world moves."""
        from roux_engine.core.orientation import CanonicalSymmetry, translate_moves_to_original

        # World solve for Y symmetry
        world_scramble = translate_moves_to_original(["M'", "U2", "M", "U", "M2"], CanonicalSymmetry.Y)
        cube_world = CubeState().apply_moves(" ".join(world_scramble))

        sols = solve_lse(cube_world, target="lse")
        assert len(sols) == 1
        sol = sols[0]
        assert sol.orientation == "y"
        # Applying sol.moves to cube_world directly solves the physical cube
        sim = cube_world.copy().apply_moves(sol.moves)
        assert sim.is_solved()

    def test_solve_lse_explicit_orientation_override(self):
        """solve_lse accepts explicit orientation overrides and validates them."""
        cube_x2 = CubeState().apply_moves("x2 M' U2 M")
        sols = solve_lse(cube_x2, target="lse", orientation="x2")
        assert len(sols) == 1
        assert sols[0].orientation == "x2"

        # Explicit override mismatch raises ValueError
        with pytest.raises(ValueError, match="First Block and Second Block are not solved"):
            solve_lse(cube_x2, target="lse", orientation="y")

    def test_solve_lse_fcn_orientation(self):
        """solve_lse auto-detects and solves Full Color Neutral orientations."""
        from roux_engine.core.orientation import get_orientation

        # FCN orientation: e.g. "z" rotation
        ori_z = get_orientation("z")
        cube_z = CubeState().apply_moves("z M U2 M' U2")
        sols = solve_lse(cube_z, target="lse")
        assert len(sols) == 1
        sol = sols[0]
        assert sol.orientation == ori_z.rotations
        sim = cube_z.copy().apply_moves(sol.moves)
        assert ori_z.is_fb_solved(sim)
        assert ori_z.is_sb_solved(sim)
        assert ori_z.is_eo_solved(sim)
        assert ori_z.is_ul_ur_solved(sim)
        assert ori_z.get_m_slice_center_offset(sim) == 0


class TestLSEComparativePathsAcrossOrientations:
    """Slice 4: All 4 comparative LSE paths in solve_lse_paths across orientations."""

    def test_solve_lse_paths_dual_neutral_inspected(self):
        """solve_lse_paths produces 4 valid solving paths on inspected dual-neutral cube."""
        from roux_engine.core.orientation import get_orientation

        cube_x2 = CubeState().apply_moves("x2 M U' M U M2 U M U M' U2")
        paths = solve_lse_paths(cube_x2)

        assert "standard" in paths
        assert "eolr" in paths
        assert "eolr_misoriented" in paths
        assert "eolr_b" in paths

        ori_x2 = get_orientation("x2")

        for name, path in paths.items():
            assert path.orientation == "x2"
            assert path.step_4a.orientation == "x2"
            assert path.step_4b.orientation == "x2"
            assert path.step_4c.orientation == "x2"

            # Total moves must solve the cube in x2 frame
            sim_total = cube_x2.copy().apply_moves(path.total_moves)
            assert ori_x2.is_fb_solved(sim_total)
            assert ori_x2.is_sb_solved(sim_total)
            assert ori_x2.is_eo_solved(sim_total)
            assert ori_x2.is_ul_ur_solved(sim_total)
            assert ori_x2.get_m_slice_center_offset(sim_total) == 0

            # Sub-step progression: 4a -> 4b -> 4c
            sim_4a = cube_x2.copy().apply_moves(path.step_4a.moves)
            if path.step_4a.center_state == "axis_aligned":
                assert ori_x2.is_eo_solved(sim_4a), f"EO failed on 4a for {name}"
            else:
                # Misoriented EOLR: centers are on F/B axis (offset 1 or 3)
                assert ori_x2.get_m_slice_center_offset(sim_4a) in (1, 3), f"Misaligned center offset failed on 4a for {name}"

            sim_4b = sim_4a.copy().apply_moves(path.step_4b.moves)
            assert ori_x2.is_eo_solved(sim_4b), f"EO failed on 4b for {name}"
            assert ori_x2.is_ul_ur_solved(sim_4b), f"UL/UR failed on 4b for {name}"

            sim_4c = sim_4b.copy().apply_moves(path.step_4c.moves)
            assert ori_x2.get_m_slice_center_offset(sim_4c) == 0, f"Centers failed on 4c for {name}"

    def test_solve_lse_paths_uninspected_world_frame(self):
        """solve_lse_paths produces 4 valid paths directly executable in world frame."""
        from roux_engine.core.orientation import CanonicalSymmetry, translate_moves_to_original

        # Uninspected world frame for Y symmetry
        world_scramble = translate_moves_to_original(["M", "U'", "M", "U", "M2", "U", "M", "U", "M'", "U2"], CanonicalSymmetry.Y)
        cube_world = CubeState().apply_moves(" ".join(world_scramble))

        paths = solve_lse_paths(cube_world)
        assert len(paths) == 4

        for name, path in paths.items():
            assert path.orientation == "y"
            # Total moves applied to world cube must fully solve the physical cube
            sim = cube_world.copy().apply_moves(path.total_moves)
            assert sim.is_solved(), f"Path {name} failed to solve world cube"

    def test_solve_lse_paths_explicit_orientation(self):
        """solve_lse_paths respects and validates explicit orientation override."""
        cube_x2 = CubeState().apply_moves("x2 M2 U2 M2")
        paths = solve_lse_paths(cube_x2, orientation="x2")
        for path in paths.values():
            assert path.orientation == "x2"

        with pytest.raises(ValueError, match="First Block and Second Block are not solved"):
            solve_lse_paths(cube_x2, orientation="y")


class TestLSEAdvancedEOLRMisorientedCenters:
    """Slice 5: Advanced EOLR with misoriented centers across orientations."""

    SCRAMBLE = "M U' M U M2 U M U M' U2"

    def test_eolr_misoriented_centers_all_dual_neutral_inspected(self):
        """Advanced EOLR with misoriented centers works accurately across all 8 dual-neutral orientations."""
        from roux_engine.core.orientation import get_dual_neutral_orientations

        for ori in get_dual_neutral_orientations():
            cube = CubeState()
            if ori.rotations:
                cube.apply_moves(ori.rotations)
            cube.apply_moves(self.SCRAMBLE)

            # 1. Aligned only
            sols_aligned = solve_lse(cube, target="4a", allow_misoriented_centers=False)
            assert sols_aligned[0].move_count == 7
            assert sols_aligned[0].center_state == "axis_aligned"
            assert sols_aligned[0].orientation == ori.rotations

            # 2. Misoriented centers enabled
            sols_std = solve_lse(cube, target="standard_eo", allow_misoriented_centers=True)
            assert sols_std[0].move_count == 7
            assert sols_std[0].center_state == "axis_aligned"
            assert sols_std[0].orientation == ori.rotations

            sols_eolr = solve_lse(cube, target="eolr", allow_misoriented_centers=True)
            sol_mis = sols_eolr[0]
            assert sol_mis.move_count == 5
            assert sol_mis.center_state == "misaligned"
            assert sol_mis.orientation == ori.rotations

            # In inspected frame, moves are <M, U>
            assert sol_mis.moves == ["M'", "U'", "M'", "U", "M'"]

            sim_mis = cube.copy().apply_moves(sol_mis.moves)
            # Centers are on F/B axis (offset 1 or 3)
            assert ori.get_m_slice_center_offset(sim_mis) in (1, 3)

    def test_eolr_misoriented_centers_uninspected_world_frame(self):
        """Advanced EOLR with misoriented centers works in uninspected world frames."""
        from roux_engine.core.orientation import CanonicalSymmetry, translate_moves_to_original, get_orientation

        sym = CanonicalSymmetry.Y
        ori = get_orientation(sym)
        world_scramble = translate_moves_to_original(self.SCRAMBLE.split(), sym)
        cube_world = CubeState().apply_moves(" ".join(world_scramble))

        sols_eolr = solve_lse(cube_world, target="eolr", allow_misoriented_centers=True)
        sol_mis = sols_eolr[0]
        assert sol_mis.move_count == 5
        assert sol_mis.center_state == "misaligned"
        assert sol_mis.orientation == "y"

        # In Y symmetry, M' translates to S'
        assert sol_mis.moves == ["S'", "U'", "S'", "U", "S'"]

        sim_mis = cube_world.copy().apply_moves(sol_mis.moves)
        # World cube center offset when rotated to inspection frame
        sim_ins = sim_mis.copy().apply_moves(ori.rotations)
        assert ori.get_m_slice_center_offset(sim_ins) in (1, 3)


class TestLSEGraphSingletonPreservation:
    """Slice 6: Underlying in-memory graph singleton remains strictly canonical and single-instance."""

    def test_singleton_identity_preserved_across_orientations(self):
        """LSEGraph singleton identity and state count are invariant across multi-orientation solving."""
        from roux_engine.solver.lse_solver import LSEGraph
        from roux_engine.core.orientation import get_dual_neutral_orientations

        graph_before = LSEGraph.get_instance()
        assert graph_before.num_states == 7680
        assert len(graph_before) == 7680

        # Solve across all dual-neutral orientations
        for ori in get_dual_neutral_orientations():
            cube = CubeState()
            if ori.rotations:
                cube.apply_moves(ori.rotations)
            cube.apply_moves("M' U2 M U M2")
            solve_lse(cube, target="all")
            solve_lse_paths(cube)

        graph_after = LSEGraph.get_instance()
        assert graph_before is graph_after
        assert graph_after.num_states == 7680
        assert len(graph_after) == 7680

    def test_canonical_methods_unaffected_by_orientation_solving(self):
        """Low-level canonical methods on LSEGraph remain strictly canonical."""
        from roux_engine.solver.lse_solver import LSEGraph

        graph = LSEGraph.get_instance()
        clean = CubeState()
        clean_code = graph.encode_cube(clean)
        assert clean_code == (1 << 12) | (3 << 9) | (0 << 3) | (1 << 2) | 0
        assert graph.contains_state(clean)
        assert graph.state_to_id[clean_code] == 0

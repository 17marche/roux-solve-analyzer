"""Unit and property tests for Last Six Edges (LSE) graph and solver."""

import time
import pytest
from roux_engine.core.cube import CubeState
from roux_engine.solver.lse_solver import LSEGraph, LSESolution, solve_lse


def test_lse_graph_initialization_and_state_count():
    """Verify that LSEGraph generates exactly 7,680 states in < 50ms upon initialization."""
    t0 = time.perf_counter()
    graph = LSEGraph()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # 1. Verification of exact state space size
    assert len(graph) == 7680
    assert graph.num_states == 7680

    # 2. Performance benchmark: must initialize in < 50ms
    assert elapsed_ms < 50.0, f"Graph initialization took {elapsed_ms:.2f}ms (expected < 50ms)"

    # 3. Canonical solved state must be present in graph
    solved = CubeState()
    assert graph.contains_state(solved)


def test_lse_solved_state():
    """Verify solve_lse on an already solved cube returns 0-move solutions for all sub-steps."""
    clean_cube = CubeState()
    solutions = solve_lse(clean_cube, target="all")

    targets = {sol.target: sol for sol in solutions}
    assert "4a" in targets
    assert "4b" in targets
    assert "4c" in targets
    assert "lse" in targets

    assert targets["4a"].move_count == 0
    assert targets["4a"].moves == []
    assert targets["4b"].move_count == 0
    assert targets["4b"].moves == []
    assert targets["4c"].move_count == 0
    assert targets["4c"].moves == []
    assert targets["4c"].case_name == "solved"
    assert targets["lse"].move_count == 0
    assert targets["lse"].moves == []


def test_lse_step_4a_variants():
    """Verify solve_lse correctly identifies and solves standard EO, EOLR, and EOLR-b targets."""
    from roux_engine.segmenter.lse_classifier import LSEClassifier

    # 1. Scramble with 4 bad edges (M'): standard EO target
    cube_4bad = CubeState().apply_move("M'")
    assert not LSEClassifier.is_eo_solved(cube_4bad)

    sols_4a = solve_lse(cube_4bad, target="4a")
    assert len(sols_4a) >= 1
    # Check standard EO solution:
    std_sols = [s for s in sols_4a if s.case_name == "standard_eo"]
    assert len(std_sols) == 1
    std_sol = std_sols[0]
    assert std_sol.move_count == 1
    assert std_sol.moves == ["M"]

    sim_std = cube_4bad.copy().apply_moves(std_sol.moves)
    assert LSEClassifier.is_eo_solved(sim_std)
    assert LSEClassifier.is_center_axis_aligned(sim_std)

    # 2. Test EOLR target: UL and UR must end up in bottom slots DF and DB
    # Test on a state where EOLR is reachable
    cube_arrow = CubeState().apply_moves("U M' U' M")
    eolr_sols = solve_lse(cube_arrow, target="eolr")
    assert len(eolr_sols) >= 1
    eolr_sol = eolr_sols[0]
    sim_eolr = cube_arrow.copy().apply_moves(eolr_sol.moves)
    assert LSEClassifier.is_eo_solved(sim_eolr)
    assert LSEClassifier.is_center_axis_aligned(sim_eolr)
    assert LSEClassifier.classify_4a_variant(sim_eolr) in ("eolr", "eolr_b")

    # 3. Test EOLR-b target: UL and UR must end up directly in top slots
    eolrb_sols = solve_lse(cube_arrow, target="eolr_b")
    assert len(eolrb_sols) >= 1
    eolrb_sol = eolrb_sols[0]
    sim_eolrb = cube_arrow.copy().apply_moves(eolrb_sol.moves)
    assert LSEClassifier.is_eo_solved(sim_eolrb)
    assert LSEClassifier.is_center_axis_aligned(sim_eolrb)
    assert LSEClassifier.is_ul_ur_solved(sim_eolrb)
    assert LSEClassifier.classify_4a_variant(sim_eolrb) == "eolr_b"


def test_lse_step_4b_ul_ur_placement():
    """Verify solve_lse correctly solves Step 4b (UL/UR edge placement into top slots)."""
    from roux_engine.segmenter.lse_classifier import LSEClassifier

    # 1. Start from EOLR state where EO is solved and UL/UR are in DF/DB
    # e.g. "U M2 U'" puts UL/UR into DF/DB with EO solved
    cube_eolr = CubeState().apply_moves("U M2 U'")
    assert LSEClassifier.is_eo_solved(cube_eolr)
    assert not LSEClassifier.is_ul_ur_solved(cube_eolr)

    sols_4b = solve_lse(cube_eolr, target="4b")
    assert len(sols_4b) >= 1
    sol_4b = sols_4b[0]
    assert sol_4b.target == "4b"
    assert sol_4b.case_name == "ul_ur"
    assert sol_4b.move_count > 0

    sim_4b = cube_eolr.copy().apply_moves(sol_4b.moves)
    assert LSEClassifier.is_eo_solved(sim_4b)
    assert LSEClassifier.is_center_axis_aligned(sim_4b)
    assert LSEClassifier.is_ul_ur_solved(sim_4b)

    # 2. If already solved 4b, returns 0 moves
    cube_4b_solved = CubeState().apply_moves("M2 U2 M2 U2") # opp_opp: 4b is solved, 4c unsolved
    assert LSEClassifier.is_ul_ur_solved(cube_4b_solved)
    sols_4b_skip = solve_lse(cube_4b_solved, target="4b")
    assert len(sols_4b_skip) == 1
    assert sols_4b_skip[0].move_count == 0
    assert sols_4b_skip[0].moves == []


def test_lse_step_4c_cases_and_solutions():
    """Verify solve_lse correctly classifies and solves Step 4c cases (Dots, Bars, Column, Opp-Opp, Solved)."""
    # 1. Solved
    cube_solved = CubeState()
    sols_solved = solve_lse(cube_solved, target="4c")
    assert len(sols_solved) == 1
    assert sols_solved[0].case_name == "solved"
    assert sols_solved[0].move_count == 0

    # 2. Dots: UF<->DF and UB<->DB swapped across M-slice (M' U2 M2 U2 M')
    cube_dots = CubeState().apply_moves("M' U2 M2 U2 M'")
    sols_dots = solve_lse(cube_dots, target="4c")
    assert len(sols_dots) == 1
    sol_dots = sols_dots[0]
    assert sol_dots.target == "4c"
    assert sol_dots.case_name == "dots"
    assert sol_dots.move_count > 0
    sim_dots = cube_dots.copy().apply_moves(sol_dots.moves)
    assert sim_dots.is_solved()

    # 3. Bars: Single horizontal bar
    cube_bars = CubeState().apply_moves("U2 M U2 M2 U2 M U2 M2")
    sols_bars = solve_lse(cube_bars, target="4c")
    assert len(sols_bars) == 1
    sol_bars = sols_bars[0]
    assert sol_bars.target == "4c"
    assert sol_bars.case_name == "bars"
    assert sol_bars.move_count > 0
    sim_bars = cube_bars.copy().apply_moves(sol_bars.moves)
    assert sim_bars.is_solved()

    # 4. Opp-Opp: Horizontal bars on both top and bottom (M2 U2 M2 U2)
    cube_opp_opp = CubeState().apply_moves("M2 U2 M2 U2")
    sols_opp_opp = solve_lse(cube_opp_opp, target="4c")
    assert len(sols_opp_opp) == 1
    sol_opp_opp = sols_opp_opp[0]
    assert sol_opp_opp.target == "4c"
    assert sol_opp_opp.case_name == "opp_opp"
    assert sol_opp_opp.move_count > 0
    sim_opp_opp = cube_opp_opp.copy().apply_moves(sol_opp_opp.moves)
    assert sim_opp_opp.is_solved()

    # 5. Column (Columns / 3-cycle: M U2 M' U2)
    cube_column = CubeState().apply_moves("M U2 M' U2")
    sols_column = solve_lse(cube_column, target="4c")
    assert len(sols_column) == 1
    sol_column = sols_column[0]
    assert sol_column.target == "4c"
    assert sol_column.case_name in ("column", "columns", "3_cycle")
    assert sol_column.move_count > 0
    sim_column = cube_column.copy().apply_moves(sol_column.moves)
    assert sim_column.is_solved()


def test_lse_1look_global_solution_and_correctness():
    """Verify 1-look global LSE solutions directly solve arbitrary LSE states."""
    scrambles = [
        [],  # solved
        ["M'"],
        ["M2", "U2", "M2"],
        ["M", "U", "M'", "U'"],
        ["U2", "M", "U", "M'", "U2", "M2", "U"],
        ["M'", "U2", "M", "U", "M2", "U'", "M'"],
        ["M", "U2", "M", "U2", "M'", "U2", "M"],
        ["U", "M2", "U'", "M", "U2", "M'", "U2"],
    ]

    for scramble in scrambles:
        cube = CubeState()
        if scramble:
            cube.apply_moves(scramble)

        # Query 1-look LSE solution
        sols_lse = solve_lse(cube, target="lse")
        assert len(sols_lse) == 1
        sol = sols_lse[0]
        assert sol.target == "lse"
        assert sol.case_name in ("solved", "1look")

        # Applying moves must fully solve the cube
        sim = cube.copy().apply_moves(sol.moves)
        assert sim.is_solved(), f"Failed to solve for scramble: {' '.join(scramble)}"

        # Query all targets
        all_sols = solve_lse(cube, target="all")
        target_map = {s.target: s for s in all_sols}
        assert "4a" in target_map
        assert "4b" in target_map
        assert "lse" in target_map
        sim_all = cube.copy().apply_moves(target_map["lse"].moves)
        assert sim_all.is_solved()


def test_lse_query_latency_microsecond():
    """Verify single LSE queries execute in < 1µs in O(1) time."""
    graph = LSEGraph.get_instance()
    cube = CubeState().apply_moves("M' U2 M U M2")
    code = graph.encode_cube(cube)

    # Warm-up tables
    graph.query_move(code, target="4a")

    iterations = 50000
    t0 = time.perf_counter()
    for _ in range(iterations):
        move = graph.query_move(code, target="4a")
    t1 = time.perf_counter()

    avg_latency_us = ((t1 - t0) / iterations) * 1e6
    assert avg_latency_us < 1.0, f"Average query latency {avg_latency_us:.3f}µs exceeded 1µs limit"






"""Tests verifying architectural tier inversion severing in roux_engine.solver.

Verifies that sb_solver.py and symmetry.py import directly from core.orientation
rather than segmenter.fb_detector or segmenter.sb_detector, and that all Second Block
and symmetry operations delegate to the core orientation domain model.
"""

import ast
import inspect
import pytest

import roux_engine.solver.sb_solver as sb_solver_mod
import roux_engine.solver.symmetry as symmetry_mod
from roux_engine.core.constants import Color, Edge, Corner
from roux_engine.core.cube import CubeState
from roux_engine.core.orientation import (
    CanonicalSymmetry as CoreCanonicalSymmetry,
    RouxOrientation,
    SBPlacement,
    get_orientation,
    get_all_orientations,
    get_dual_neutral_orientations,
)
from roux_engine.solver.symmetry import (
    CanonicalSymmetry as SolverCanonicalSymmetry,
    get_symmetry,
    get_all_symmetries,
    is_fb_solved_for_symmetry,
    extract_canonical_placement,
)
from roux_engine.solver.sb_solver import (
    SBSolver,
    is_center_aligned_sb_solved,
    extract_sb_placement,
    get_m_slice_center_offset,
    solve_sb,
)


class TestArchitecturalTierSeparation:
    """Verifies that search tier modules do not import from phase segmentation detectors."""

    def _get_imported_module_names(self, module) -> list[str]:
        source = inspect.getsource(module)
        tree = ast.parse(source)
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                dots = "." * node.level
                full_mod = f"{dots}{mod}" if dots else mod
                imported.append(full_mod)
                for alias in node.names:
                    imported.append(f"{full_mod}.{alias.name}")
        return imported

    def test_symmetry_has_no_segmenter_imports(self):
        """symmetry.py must have zero imports from the segmenter tier."""
        imports = self._get_imported_module_names(symmetry_mod)
        segmenter_imports = [imp for imp in imports if "segmenter" in imp or "fb_detector" in imp or "sb_detector" in imp]
        assert segmenter_imports == [], f"Found forbidden segmenter imports in symmetry.py: {segmenter_imports}"

    def test_sb_solver_has_no_detector_imports(self):
        """sb_solver.py must not import from fb_detector or sb_detector in segmenter."""
        imports = self._get_imported_module_names(sb_solver_mod)
        detector_imports = [
            imp for imp in imports
            if "fb_detector" in imp or "sb_detector" in imp
        ]
        assert detector_imports == [], f"Found forbidden detector imports in sb_solver.py: {detector_imports}"

    def test_canonical_symmetry_identity(self):
        """CanonicalSymmetry in solver.symmetry must be the exact enum from core.orientation."""
        assert SolverCanonicalSymmetry is CoreCanonicalSymmetry


class TestSolverDelegationToCoreOrientation:
    """Verifies that solver functions delegate to core.orientation."""

    def test_extract_sb_placement_delegation(self):
        """extract_sb_placement must return core.orientation.SBPlacement and match core method."""
        c = CubeState().apply_moves("R U R' U'")
        for ori in get_dual_neutral_orientations():
            solver_p = extract_sb_placement(c, ori)
            core_p = ori.extract_sb_placement(c)
            assert solver_p == core_p
            assert isinstance(solver_p, SBPlacement)

    def test_extract_sb_placement_string_resolution(self):
        """extract_sb_placement resolves rotation strings and symmetry enums."""
        c = CubeState().apply_moves("r U R' U' r' F R F'")
        p_str = extract_sb_placement(c, "y")
        p_sym = extract_sb_placement(c, CoreCanonicalSymmetry.Y)
        p_ori = extract_sb_placement(c, get_orientation("y"))
        assert p_str == p_sym == p_ori

    def test_is_center_aligned_sb_solved_delegation(self):
        """is_center_aligned_sb_solved must match RouxOrientation.is_center_aligned_sb_solved."""
        clean = CubeState()
        for ori in get_dual_neutral_orientations():
            c = clean.copy()
            if ori.rotations:
                c.apply_moves(ori.rotations)
            assert is_center_aligned_sb_solved(c, ori) is True
            assert is_center_aligned_sb_solved(c, ori.rotations) is True

            # Disrupt M-slice by M
            c_m = c.copy().apply_move("M")
            assert is_center_aligned_sb_solved(c_m, ori) is False

            # Disrupt M-slice by M2 (preserves U/D axis)
            c_m2 = c.copy().apply_move("M2")
            assert is_center_aligned_sb_solved(c_m2, ori) is True

    def test_get_m_slice_center_offset_delegation(self):
        """get_m_slice_center_offset must match RouxOrientation.get_m_slice_center_offset."""
        clean = CubeState()
        for ori in get_dual_neutral_orientations():
            c = clean.copy()
            if ori.rotations:
                c.apply_moves(ori.rotations)
            assert get_m_slice_center_offset(c, ori) == 0
            assert get_m_slice_center_offset(c.copy().apply_move("M"), ori) == 1
            assert get_m_slice_center_offset(c.copy().apply_move("M2"), ori) == 2
            assert get_m_slice_center_offset(c.copy().apply_move("M'"), ori) == 3

    def test_is_fb_solved_for_symmetry_dual_neutral(self):
        """is_fb_solved_for_symmetry validates across all 8 dual-neutral orientations without segmenter."""
        for sym in get_all_symmetries():
            c = CubeState()
            if sym == CoreCanonicalSymmetry.I:
                assert is_fb_solved_for_symmetry(c, sym, inspected=False) is True
            c_inspected = CubeState()
            if sym.inspection_rotation:
                c_inspected.apply_moves(sym.inspection_rotation)
            assert is_fb_solved_for_symmetry(c_inspected, sym, inspected=True) is True

    def test_sb_solver_auto_detect_dual_neutral_orientations(self):
        """SBSolver.solve auto-detects orientation using core.orientation across dual-neutral set."""
        solver = SBSolver.get_instance()
        for sym in get_all_symmetries():
            c = CubeState()
            if sym.inspection_rotation:
                c.apply_moves(sym.inspection_rotation)
            c.apply_moves("R U R' U'")
            sols = solver.solve(c, k=1)
            assert len(sols) >= 1
            sol = sols[0]
            assert sol.orientation == sym.value
            c_solved = c.copy().apply_moves(" ".join(sol.moves))
            assert is_center_aligned_sb_solved(c_solved, sym.value) is True

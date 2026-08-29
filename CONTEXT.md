# Roux Solve Analyzer

Core domain model and canonical terminology for the Roux AI Speedcube Coach and solve critique engine.

## Language

### Core Entities

**SolveRecord**:
A structured record of a speedcube solve containing metadata, scramble sequence, executed solution moves, and validation state.
_Avoid_: SolveItem, RawSolve, SolveData

**Reconstruction**:
A human-documented transcript of a competitive speedcube solve containing move-by-move turns, inspection rotations, and phase comments.
_Avoid_: SolveText, AlgTrace, ForumPost

**Human Phase Split**:
The step-by-step phase boundaries, move counts, and split timings recorded by a human reconstructor.
_Avoid_: Manual split, Annotation marker, Step split

**SegmentedSolve**:
A solve whose move sequence has been deterministically partitioned into Roux micro-phases with classified sub-steps.
_Avoid_: ParsedSolve, SlicedSolve, SplitResult

---

### Roux Phases

**First Block (FB)**:
A solved 1x2x3 block on the Left face consisting of 5 moving cubies (DL, FL, BL edges and DFL, DBL corners) plus the Left center.
_Avoid_: Left block, First square, FB1

**Second Block (SB)**:
A solved 1x2x3 block on the Right face (DR, FR, BR edges and DFR, DBR corners) built without disturbing the First Block.
_Avoid_: Right block, F2L-2, SB2

**CMLL**:
Corners of the Last Layer; the algorithmic step orienting and permuting all four U-layer corners simultaneously while preserving FB and SB.
_Avoid_: CLL, COLL, Corner OLL/PLL

**Last Six Edges (LSE)**:
The final phase of Roux solving the remaining 6 edges (UL, UR, UF, UB, DF, DB) and M/U/D centers across micro-steps 4a, 4b, and 4c.
_Avoid_: L6E, Step 4, M-slice phase

**Step 4a (EO / EOLR)**:
Sub-step of LSE that orients all 6 remaining edges (EO) and optionally influences or places UL/UR into bottom slots (EOLR) or top slots (EOLR-b).
_Avoid_: Edge orientation, EO step

**Step 4b (UL/UR Insertion)**:
Sub-step of LSE that positions UL and UR edges into their respective top left and top right positions.
_Avoid_: LR insertion, Wing insertion

**Step 4c (M-Permutation)**:
Sub-step of LSE that permutes the final four M-slice edges (UF, UB, DF, DB) and aligns centers into their solved state.
_Avoid_: 4c, EP, Center alignment, M-slice finish

---

### Mechanics & Notation

**Slice Turn Metric (STM)**:
The primary movecount metric for the Roux method where single face turns, slice turns (M, E, S), and wide turns (r, l, u, d, f, b) count as 1 move, while whole-cube rotations (x, y, z) count as 0 moves.
_Avoid_: HTM, ETM, QTM, Move count

**Dual-Neutrality**:
Solving with either White or Yellow on the bottom (D) face with First Block on the Left (L) face (8 total First Block symmetries).
_Avoid_: Half-neutral, Yellow-white neutral

**Full Color Neutrality (FCN)**:
Solving First Block on any of the 6 faces and with any color on bottom (24 total First Block orientations).
_Avoid_: Total neutrality, Global neutrality

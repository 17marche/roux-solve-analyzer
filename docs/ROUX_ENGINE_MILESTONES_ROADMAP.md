# Roux AI Speedcube Coach: Comprehensive Milestones & Execution Roadmap

**Document Version:** 1.0.0  
**Project:** Roux AI Speedcube Coach & Critique Engine (`roux-solve-analyzer`)  
**Target Platform:** Python 3.10+  

---

## 1. Roadmap Architecture Overview

The development of the Roux AI Speedcube Coach is broken down into **6 sequential, test-driven milestones**. Each milestone produces a fully functional, tested, and decoupled module.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 1: Virtual Cube Engine & STM Move Parser                          │
│ -> 3D Cubie Coordinate Representation, Move Transforms, Normalizer          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 2: Deterministic Roux Phase Segmenter & Validation Suite          │
│ -> FB (5 pieces), SB (DR/Pairs/Rotations), CMLL (42 cases), LSE (4a/4b/4c)  │
│ -> Benchmark across 1,181 reco.nz solves                                    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 3: Pattern Databases (PDB) & IDA* Heuristic Solver                │
│ -> 5.32M State FB PDB (~2.66MB), LSE Table (7,680 states), Top-K Candidates │
│ -> Canonical Symmetry Re-Mapping (Evaluates all 8 First Blocks)             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 4: Empirical Transition Matrix & Biomechanical Flow Scorer        │
│ -> Bigram Latency Tables (2H & OH), Regrip & Micro-Pause Detector           │
│ -> Ingestion & Calibration of 528-pair 2-gram transition datasets           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 5: Neural Policy Model (Behavioral Cloning)                       │
│ -> 8x Automorphism Data Pipeline, PyTorch Pre-training & Fine-Tuning        │
│ -> State-Aware Lookahead & Style Ranker (ONNX runtime inference)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Milestone 6: Diagnostic Engine & Structured Coaching Output                 │
│ -> Master `analyze_solve()` Pipeline, JSON Diagnostic Report Generator      │
│ -> LLM Prompt Templates & Longitudinal Profile Integration                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Milestone Specifications

---

### Milestone 1: Virtual Cube Simulator & STM Move Parser
* **Goal:** Build a fast, deterministic 3D Rubik's cube state simulator and move parser in pure Python.
* **Key Modules:**
  * `src/roux_engine/core/constants.py`: Piece enums (`Corner`, `Edge`, `Center`, `Color`).
  * `src/roux_engine/core/cube.py`: `CubeState` dataclass storing `cp` (8), `co` (8), `ep` (12), `eo` (12), and `centers` (6) in NumPy arrays.
  * `src/roux_engine/core/moves.py`: 18 face turns ($U, D, F, B, L, R$ + modifiers), 6 slice turns ($M, E, S$), 6 wide turns ($r, l, u, d, f, b$), and 3 cube rotations ($x, y, z$).
  * `src/roux_engine/core/parser.py`: `MoveParser` normalizing notation aliases (`Rw'`, `2R`, `r'`, `M2`) and parsing both space-delimited text and timestamped JSON streams.
* **Reference Integration:** Port standard facelet and permutation definitions from `onionhoney/roux-trainers/src/lib/Defs.tsx`.
* **Testing & Verification:**
  * Identity tests: $M \cdot M' = I$, $r \cdot r' = I$, $x \cdot x' = I$.
  * Move orders: $U^4 = I$, $M^4 = I$, $x^4 = I$.
  * Commutator consistency: Sexy move $(R\ U\ R'\ U')^6 = I$, Sune, T-Perm, H-Perm.
* **Acceptance Criteria:** Move simulation executes $> 50,000\text{ moves/second}$ in Python with 100% state accuracy.

---

### Milestone 2: Deterministic Roux Phase Segmenter & Validation Suite
* **Goal:** Automatically and deterministically slice any solve into exact Roux micro-phases.
* **Key Modules:**
  * `src/roux_engine/segmenter/fb_detector.py`: Detects Left 1x2x3 block (5 moving pieces: $DL, FL, BL$ edges + $DFL, DBL$ corners) + checks for concurrent progress (e.g. SB pieces/squares formed during FB).
  * `src/roux_engine/segmenter/sb_detector.py`: Detects Right 1x2x3 block ($DR, FR, BR$ edges + $DFR, DBR$ corners) + tracks $DR$ placement timing, pair 1 vs pair 2 sequence, and counts cube rotations ($y, x, z$).
  * `src/roux_engine/segmenter/cmll_classifier.py`: Recognizes all **42 CMLL cases** (Sune, Antisune, H, Pi, T, U, L, Skip) + isolates pre-AUF, post-AUF, and recognition pauses.
  * `src/roux_engine/segmenter/lse_classifier.py`: 
    * Step 4a: Classifies Standard EO vs. EOLR ($UL/UR$ influenced into bottom) vs. EOLR-b (direct top insertion).
    * Step 4b: Tracks $UL/UR$ insertion into top slots.
    * Step 4c: Classifies final $M$-slice and center cases (*Dots, Bars, Column, Opp-Opp*).
  * `src/roux_engine/segmenter/segmenter.py`: Master pipeline coordinating all phase detectors.
* **Reference Integration:** Port 42 CMLL definitions, algorithms, and setups directly from `onionhoney/roux-trainers/src/lib/Algs.tsx`.
* **Testing & Verification:**
  * Run segmenter across all 1,181 real human solves from `roux_solves.json`.
  * Validate step-split timings against human reconstructors' splits.
* **Acceptance Criteria:** **100% segmentation success** on all valid reconstructed solves in the dataset.

---

### Milestone 3: Pattern Databases (PDB) & IDA* Heuristic Solver
* **Goal:** Precompute mathematical lower-bound lookup tables and build an $IDA^*$ search engine to generate top-$K$ candidate solutions.
* **Key Modules:**
  * `src/roux_engine/solver/pdb_generator.py`: Generates the **5.32M state canonical FB PDB** via Breadth-First Search (BFS) in NumPy and serializes to packed bytes (`fb_pdb.npy`, ~2.66 MB).
  * `src/roux_engine/solver/symmetry.py`: Canonical symmetry re-mapping applying the $x2y$ automorphism group $G = \{I, y, y2, y', x2, x2y, x2y2, x2y'\}$ to query all 8 dual-neutral First Blocks from a single database.
  * `src/roux_engine/solver/lse_solver.py`: Complete 7,680-state in-memory LSE graph lookup table for instant optimal EOLR and 4c solutions.
  * `src/roux_engine/solver/ida_star.py`: Multi-path $IDA^*$ search finding the top-$K$ shortest candidate paths for any given phase.
* **Reference Integration:** Adapt state indexing techniques from `onionhoney/roux-trainers/src/lib/Pruner.tsx` and `Solver.tsx`.
* **Testing & Verification:**
  * Verify PDB lookups are strictly admissible ($h(s) \le \text{true distance}$).
  * Verify optimal FB search finds 5-move and 6-move solutions in $< 1\text{ms}$.
* **Acceptance Criteria:** Single lookup latency $< 1\mu\text{s}$; total PDB memory $< 3\text{ MB}$.

---

### Milestone 4: Empirical Transition Matrix & Biomechanical Flow Scorer
* **Goal:** Model real-world physical fingertrick speeds, regrips, and execution flow from human smart-cube data.
* **Key Modules:**
  * `src/roux_engine/ergonomics/transition_matrix.py`: Ingests and normalizes 2-gram bigram transition latencies for Two-Handed (2H) and One-Handed (OH) profiles.
  * `src/roux_engine/ergonomics/regrip_detector.py`: Identifies physical regrips and execution pauses based on inter-move timestamps ($\Delta t$) and move mechanics ($R \rightarrow F$, $r2 \rightarrow U$).
  * `src/roux_engine/ergonomics/hand_profile.py`: Conditions LSE advice based on user hand preference (`m_slice_hand`: `"right"` vs `"left"`).
* **Reference Integration:** Ingest `onionhoney/roux-trainers/src/lib/two_gram_v1.json` (528 empirical transition pairs) as our calibrated baseline matrix.
* **Testing & Verification:**
  * Generate transition latency heatmaps comparing home-grip flow ($\langle R, U, r, M \rangle$) against regrip moves ($F, B, D, y$).
  * Test candidate ranking: verify smooth fingertrick sequences score higher than awkward regrip sequences of equal movecount.
* **Acceptance Criteria:** Accurate detection of pauses and regrips across smart-cube solve streams.

---

### Milestone 5: Neural Policy Model (Behavioral Cloning)
* **Goal:** Train a lightweight neural network to score human style, pair lookahead, and piece preservation across 3D cube states.
* **Key Modules:**
  * `src/roux_engine/policy/dataset.py`: Training data pipeline applying 8x $x2y$ automorphism data augmentation to elite human reconstructions (expanding ~18,000 transitions to ~144,000 samples).
  * `src/roux_engine/policy/model.py`: Lightweight PyTorch Policy Network (Transformer / MLP) outputting probability distribution $P(a_t \in \mathcal{A} \mid s_t)$.
  * `src/roux_engine/policy/train.py`: Two-stage training pipeline (Stage 1: Synthetic pre-training on $IDA^*$ solves $\rightarrow$ Stage 2: Fine-tuning on human reconstructions).
  * `src/roux_engine/policy/ranker.py`: Evaluates candidate paths:
    $$S(\text{path}) = \sum_t \log P(a_t \mid s_t)$$
  * `src/roux_engine/policy/export.py`: Exports trained weights to ONNX for fast, lightweight local CPU inference.
* **Testing & Verification:**
  * Benchmark top-1 and top-3 next-move prediction accuracy on held-out human solves.
  * Verify model favors lookahead-preserving pair choices over lookahead-destroying choices.
* **Acceptance Criteria:** Model runs inference in $< 5\text{ms}$ on CPU via ONNX runtime.

---

### Milestone 6: Diagnostic Engine & Structured Coaching Output
* **Goal:** Tie all modules together into a unified `analyze_solve()` Python API producing structured diagnostic reports and coaching critiques.
* **Key Modules:**
  * `src/roux_engine/diagnostics/engine.py`: Master analysis pipeline integrating Segmentation + Deterministic Search + Transition Scoring + Policy Ranking.
  * `src/roux_engine/diagnostics/reporter.py`: Formats analysis into clean, standardized JSON payloads (`diagnostic_report.json`) detailing phase metrics, flow breakdowns, detected mistakes, and actionable alternatives.
  * `src/roux_engine/diagnostics/llm_prompter.py`: Generates structured prompt payloads for downstream LLM mentors.
* **Testing & Verification:**
  * End-to-end integration tests analyzing diverse solves (world-record singles, intermediate solves with rotations, smart-cube streams with pauses).
* **Acceptance Criteria:** Full 50-move solve analysis completes in $< 250\text{ms}$ on CPU.

---

## 3. Recommended Git Tracking Strategy

To keep your repository clean, professional, and easy to maintain:

### Files that MUST be Committed:
1. **Specification & Plan Documents:**
   * `ROUX_AI_COACH_SPEC_AND_PLAN.md` (Master Specification)
   * `ROUX_ENGINE_MILESTONES_ROADMAP.md` (This Execution Roadmap)
   * `milestone_1_plan.md` (Milestone 1 Implementation Plan)
   * `roux_analyzer.md` (Original Concept Document)
   * `roux_method_report.md` (Statistical Analysis Report)
2. **Datasets & References:**
   * `roux_solves.json` (The 1,181 reco.nz solve dataset)
   * `data/transitions/matrix_2h.json` (Empirical transition matrix)
3. **Source Code & Tests:**
   * Everything under `src/roux_engine/`
   * Everything under `tests/`
   * `pyproject.toml` / `requirements.txt` / `.gitignore`

### Files that are Ignored (Via `.gitignore`):
* `.venv/` (Local Python virtual environment)
* `__pycache__/` and `*.pyc` (Python compiled bytecode)
* `.pytest_cache/` (Test runner cache)
* `build/`, `dist/`, `*.egg-info/` (Build artifacts)

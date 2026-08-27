# Roux AI Speedcube Coach & Critique Engine (`roux-solve-analyzer`)

An intelligent, automated speedcubing critique and analytics engine specifically tailored to the **Roux method**.

---

## Architecture Overview

The engine evaluates Roux solves through a **3-Tier Hierarchy**:
1. **Tier 1 — Deterministic Search:** Candidate generation using Pattern Databases (PDB) and $IDA^*$ search.
2. **Tier 2 — Empirical Transition Matrix:** Biomechanical execution flow and regrip detection from human smart-cube datasets.
3. **Tier 3 — Neural Policy Model:** State-aware lookahead, pair preservation, and human solving habits via Behavioral Cloning.

---

## Getting Started

### Installation & Environment
```bash
# Using uv (recommended)
uv venv
uv pip install pytest numpy
```

### Running Automated Tests
```bash
uv run pytest -v
```

---

## Documentation
* **Master Specification & Plan:** [`docs/ROUX_AI_COACH_SPEC_AND_PLAN.md`](docs/ROUX_AI_COACH_SPEC_AND_PLAN.md)
* **Milestones Execution Roadmap:** [`docs/ROUX_ENGINE_MILESTONES_ROADMAP.md`](docs/ROUX_ENGINE_MILESTONES_ROADMAP.md)

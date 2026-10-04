# Ergonomic Transition Matrix Calibration Specification

**Document Version:** 1.0.0  
**Status:** Approved Specification  
**Target Files:**  
- [`src/roux_engine/data/transitions/matrix_2h.json`](../src/roux_engine/data/transitions/matrix_2h.json)  
- [`src/roux_engine/data/transitions/matrix_oh.json`](../src/roux_engine/data/transitions/matrix_oh.json)  
- [`src/roux_engine/ergonomics/transition_matrix.py`](../src/roux_engine/ergonomics/transition_matrix.py)  
- [`src/roux_engine/ergonomics/grip_tracker.py`](../src/roux_engine/ergonomics/grip_tracker.py)  

---

## 1. Executive Summary & Purpose

This document serves as the authoritative architectural specification and historical record for the calibration of the Two-Handed (2H) Bigram Transition Matrix in the `roux-solve-analyzer` engine.

### The Problem
The original 2H baseline transition matrix ([`matrix_2h.json`](../src/roux_engine/data/transitions/matrix_2h.json)) was imported directly from `onionhoney/roux-trainers` (`two_gram_v1.json`). Detailed empirical analysis revealed that **74.4% of all transitions (393 out of 528)** suffered from critical flaws:
1. **Synthetic Default Ceiling:** 59.3% (313 entries) were unmeasured transitions initialized to an arbitrary $\approx 3.0$ penalty (equivalent to a 300 ms pause).
2. **Artificial Clamp Floor:** 40 entries were clamped to $\approx 50\text{ ms}$ (`0.48–0.52`), creating impossible physical feats (e.g. `UB2 = 0.494`, `F2B = 0.499`).
3. **Double-Move Monotonicity Paradox:** 37 double turns ($X2$) were recorded as faster than single flicks ($X$), because $X2$ occurred in drilled algorithms at max TPS while single turns were measured during lookahead pauses.
4. **Massive Wide-Turn Bias:** Wide turns ($r$) were penalized by 600% (`rU = 3.036` vs `RU = 0.498`), causing First Block (FB) and Second Block (SB) solvers to reject optimal ergonomic solutions.
5. **Left-Hand Vacuum:** 125 left-hand ($L$) transitions and single moves (`L'`, `L2`) were dumped into the 3.0 default bucket because Roux trainers rarely drilled $L$ turns.

### The Solution
A unified, physically grounded architecture:
* **Separation of Concerns:** The transition matrix records **pure fingertrick execution latency** assuming fingers are in position. The kinematic [`GripTracker`](../src/roux_engine/ergonomics/grip_tracker.py) handles physical wrist states and penalizes forced regrips (+2.0 E-STM), eliminating double-penalization.
* **Decoupled Two-Hand Model:** Recognizes that the left and right hands act in parallel. Two-handed transitions ($M\ U$, $U\ D$) are naturally fluid.
* **Systematic Physical Scaling:** Clean empirical anchors (135 entries) are preserved; wide turns inherit a $1.12\times$ drag factor; left-hand turns inherit clean $R$ pairs via bilateral mirroring with a $1.15\times$ non-dominant drag factor; double turns enforce strict execution monotonicity ($\ge 1.15\times$ single flick).

---

## 2. Historical Provenance & Data Definition

### 2.1 Origin
The transition data originated from open-source web training tools by Jeffrey Sun ([`onionhoney/roux-trainers`](https://github.com/onionhoney/roux-trainers), specifically `src/lib/two_gram_v1.json`), developed to train algorithm execution for CMLL and LSE.

### 2.2 Mathematical Definition of the Values
In [`TransitionMatrixBuilder`](../src/roux_engine/ergonomics/transition_matrix.py#L228-L250), each entry represents the normalized inter-move delta:
$$\Delta t = t_B - t_A$$
$$\text{effort}(A, B) = \frac{\text{median}(\Delta t)}{\text{baseline\_latency\_ms}}$$
where $\text{baseline\_latency\_ms} = 100\text{ ms}$ (`DEFAULT_BASELINE_LATENCY_SEC = 0.10`).

Therefore:
* A value of **`1.0`** represents a baseline turn taking **100 ms** (10 TPS).
* A value of **`0.50`** represents a fluid trigger taking **50 ms** (20 TPS).
* A value of **`3.0`** represents a slow/unmeasured turn taking **300 ms** (3.3 TPS).

$\Delta t$ measures the **elapsed time from the completion of Move $A$ to the completion of Move $B$**. It encompasses:
$$\Delta t = (\text{inter-move pause / transition delay}) + (\text{execution time of Move } B)$$
*(or less if Move $B$ was initiated while Move $A$ was completing, i.e., overlapping turning).*

---

## 3. Deep-Dive Audit: The 5 Failure Modes of the Raw Matrix

An audit of all 528 entries in [`matrix_2h.json`](../src/roux_engine/data/transitions/matrix_2h.json) revealed five distinct failure modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW MATRIX FAILURE MODES                        │
├────────────────────────────────────────┬───────┬───────────────────────┤
│ Failure Mode                           │ Count │ Percentage of Matrix  │
├────────────────────────────────────────┼───────┼───────────────────────┤
│ 1. Synthetic ~3.0 Default Ceiling      │  313  │ 59.3%                 │
│ 2. Artificial 50 ms Clamp Floor        │   40  │  7.6%                 │
│ 3. Double-Move Monotonicity Paradox    │   37  │  7.0%                 │
│ 4. Noisy / Divergent Wide Turns        │   38  │  7.2%                 │
│ 5. Broken Single Moves                 │    4  │  0.8%                 │
├────────────────────────────────────────┼───────┼───────────────────────┤
│ Total Broken Entries (with overlap)    │  393  │ 74.4%                 │
│ Clean Empirical Entries (Retained)     │  135  │ 25.6%                 │
└────────────────────────────────────────┴───────┴───────────────────────┘
```

### 3.1 Failure Mode 1: The Synthetic $\approx 3.0$ Default Ceiling (313 entries)
* **Statistical Proof:** 313 entries (59.3%) form an artificial Gaussian distribution centered on 3.00:
  * **Mean:** `2.9998`
  * **Median:** `2.9990`
  * **Standard Deviation:** `0.0638`
  * **Range:** `[2.882, 3.108]`
* **Cause:** The original trainer populated any unmeasured bigram with a default 300 ms baseline (`3.0`) plus pseudo-random noise.
* **Breakdown:**
  * **125 entries in $L$-family:** $L$ is rarely turned after First Block in Roux, so almost all $L$ transitions were defaulted (`L'`, `L2`, `U'L`, `L'U'`, `DL`, etc.).
  * **84 entries in wide $r$:** Standard wide-turn bigrams were omitted from CMLL training sets (`rU`, `r'U`, `r'U'`, `Ur`, etc.).
  * **104 entries in other gaps:** Decoupled transitions like $D \to U$ (`D'U`, `DU'`, `D2U`), opposing face turns, and obscure slice transitions.

### 3.2 Failure Mode 2: The Artificial 50 ms Clamp Floor (40 entries)
* **Description:** 40 transitions are jammed at `0.48 – 0.52` (50 ms).
* **Cause:** In smart-cube stream processing, when moves are overlapped (or register within the same sensor polling frame), the inter-move delta hits a hard minimum clamp floor of ~50 ms.
* **Physical Impossibilities Produced:**
  * `UB2 = 0.494`: Completing a 180° double-flick on the back layer in 49 ms after finishing $U$.
  * `F2B = 0.499`: Finishing a 180° front-face wrist twist and registering a back-face turn in 49 ms.
  * `F2R' = 0.497`: Recovering from an $F2$ wrist turn into an $R'$ in 49 ms without hand reset.

### 3.3 Failure Mode 3: Double-Move Monotonicity Paradox (37 entries)
* **Description:** Executing a 180° double turn ($X2$) is recorded as taking less time than executing a 90° single flick ($X$ or $X'$) on the same face from the same starting move:
  $$\text{effort}(A, X2) < \text{effort}(A, X) \quad \text{or} \quad \text{effort}(A, X2) < \text{effort}(A, X')$$
* **Prominent Violations:**
  * `M' U2` (`0.497`) vs `M' U'` (`0.831`): A double-flick scored as **half the effort** of a single flick!
  * `R2 U2` (`1.272`) vs `R2 U` (`1.784`): Double-turn $U2$ scored 51 ms faster than single $U$!
  * `FR2` (`0.498`) vs `FR'` (`2.749`): Double-turn scored 5.5× faster than single turn!
* **Cause (Context Bias):**
  * Double-turn pairs (like `R2 U2` or `M' U2`) almost exclusively occurred inside **heavily drilled CMLL/LSE algorithms** executed at maximum TPS without pausing.
  * Single-turn pairs (like `R2 U` or `FR'`) occurred during **First/Second Block building**, where solvers paused to track pieces or regrip before turning. The data recorded **human lookahead pauses**, not fingertrick execution speed.

### 3.4 Failure Mode 4: Wide-Turn Inconsistencies (84 defaults + 38 noisy pairs)
* **Description:** Wide turns ($r, r', r2$) share the exact same wrist biomechanics as outer-layer turns ($R, R', R2$). In physical reality, turning two layers adds ~10%–15% rotational inertia/friction.
* In the raw data:
  * Single `r` was `3.056` (vs `R` at `0.794`).
  * `rU` was `3.036` (vs `RU` at `0.498`).
  * When $r$ happened to be measured, small sample sizes created wild noise: `D2r' = 1.786` vs `D2R' = 1.340` (+33.3% penalty); `Fr2 = 2.416` vs `FR2 = 0.498` (+385% penalty).

### 3.5 Failure Mode 5: The Grip-Confounded Measurement Trap
* **Description:** The raw dataset was collected from full solves without tracking wrist orientation.
* **The Contradiction:**
  * When a solver did an A-perm ending in `... R' B2`, the right hand was in `R_PRIME_AWAY` (fingers naturally resting on the B face). Flicking $B2$ from this grip is as effortless as flicking $D2$ from `HOME` grip $\to$ `UB2` registered as `0.494`.
  * But when a solver did `UB` from neutral `HOME` grip, reaching B was awkward $\to$ `UB` registered as `2.564`.
* **The Double-Penalization Bug:**
  In [`FlowScorer`](../src/roux_engine/ergonomics/flow_scorer.py#L156-L158), total score is:
  $$\text{E-STM} = \sum \text{transition\_effort} + 2.0 \times \text{regrip\_count}$$
  If `UB` is in the matrix at `2.564` (including an awkward reach pause), AND [`GripTracker`](../src/roux_engine/ergonomics/grip_tracker.py) flags `B` from `HOME` as a forced regrip (+2.0), the move is scored at **`4.564 E-STM`** (double-penalized for the same physical constraint).

---

## 4. Biomechanical Foundations: Decoupled Two-Hand Model

To calibrate the matrix accurately, the engine relies on a physically correct speedcubing model:

### 4.1 Independent Hand Operations
In two-handed Roux solving, the hands are **decoupled**:
* **Left Hand:** Holds and stabilizes the First Block (FB). Left fingers are permanently available to execute:
  * $U' / U$ (left index pull / push)
  * $D / D'$ (left ring / pinky flick)
  * $L / L' / L2$ (left wrist rotations)
* **Right Hand:** Executes Second Block, CMLL, and M-slice:
  * $U / U'$ (right index pull / push)
  * $R / R' / R2$ and $r / r' / r2$ (right wrist rotations)
  * $M' / M2 / M$ (right ring/middle slice flicks)
  * $D'$ (right ring flick)
  * $F / F'$ (right index push / pull)

### 4.2 Resource Contention vs. Hand Independence
* **Decoupled (Parallel / Zero Contention):**
  When Move $A$ uses Hand 1 and Move $B$ uses Hand 2, inter-move latency is naturally low ($\Delta t \approx 0.50\text{--}0.75$):
  * $U\ D'$: Right index $U$, left ring $D'$.
  * $M\ U$: Right ring/middle $M$, left index $U$.
  * $U\ B2$ (from `R_PRIME_AWAY`): Left index $U$, right ring/middle $B2$.
* **Coupled (Serial / Same-Hand Contention):**
  When Move $A$ and Move $B$ contend for the same fingers or require a wrist reset on the same hand, transition latency increases:
  * $R \to F$: Right wrist rotates away, pulling the right index back; pushing $F$ requires repositioning.
  * $F2 \to R'$: Right wrist rotates 180° on $F$; resetting the hand to turn $R'$ requires a grip recovery.

### 4.3 Separation of Concerns: Matrix vs. GripTracker
* **Transition Matrix:** Represents **pure fingertrick execution latency** assuming the hand is in position to execute the move.
* **GripTracker:** Tracks the 3 right-wrist kinematic states:
  * `HOME` (neutral, thumb on Front, fingers on Back)
  * `R_AWAY` (+90° clockwise, thumb on Up, fingers on Down)
  * `R_PRIME_AWAY` (-90° counter-clockwise, thumb on Down, fingers on Up/Back)
  Whenever an anatomical dead-end occurs (e.g. attempting $B2$ from `HOME`, or $R$ from `R_AWAY`), [`GripTracker`](../src/roux_engine/ergonomics/grip_tracker.py) injects a flat **+2.0 E-STM (200 ms) regrip penalty**.

---

## 5. Calibration Rules & Mathematical Formulations

### 5.1 Accounting Overview
* **Retained Clean Entries:** **135 entries (25.6%)** remain identical to preserve genuine human timing anchors.
* **Calibrated / Replaced Entries:** **393 entries (74.4%)** are systematically derived using the rules below.

---

### 5.2 Rule 1: Wide-Turn Derivation ($r$ and $l$)
Wide turns inherit from outer-layer turns with a **$1.12\times$ rotational drag factor**:
$$\text{effort}(A, r) = \text{effort}(A, R) \times 1.12$$
$$\text{effort}(r, B) = \text{effort}(R, B) \times 1.12$$
$$\text{effort}(r) = \text{effort}(R) \times 1.12 = 0.794 \times 1.12 = \mathbf{0.889}$$
$$\text{effort}(r') = \text{effort}(R') \times 1.12 = 0.589 \times 1.12 = \mathbf{0.660}$$
$$\text{effort}(r2) = \text{effort}(R2) \times 1.12 = 0.499 \times 1.12 = \mathbf{0.559}$$

*Wide $l$ turns inherit from $L$ following the exact same rule: $\text{effort}(l) = \text{effort}(L) \times 1.12$.*

**Key Outcomes:**
* `rU` drops from `3.036` to $0.498 \times 1.12 = \mathbf{0.558}$.
* `r'U` drops from `3.069` to $0.498 \times 1.12 = \mathbf{0.558}$.
* `D2r'` is normalized from `1.786` down to $1.340 \times 1.12 = \mathbf{1.501}$ (eliminating empirical noise).

---

### 5.3 Rule 2: Left-Hand Bilateral Mirroring ($L$)
Left-hand turns mirror clean right-hand pairs with a **$1.15\times$ non-dominant drag factor**:

#### Bilateral Mirror Mapping:
| Move $X$ | Bilateral Mirror $X_{\text{mirror}}$ |
| :--- | :--- |
| **`R`** (right wrist up) | **`L'`** (left wrist up) |
| **`R'`** (right wrist down) | **`L`** (left wrist down) |
| **`R2`** (right wrist 180°) | **`L2`** (left wrist 180°) |
| **`U`** (right index pull) | **`U'`** (left index pull) |
| **`U'`** (left index pull) | **`U`** (right index pull) |
| **`D'`** (right ring flick) | **`D`** (left ring flick) |
| **`D`** (left ring flick) | **`D'`** (right ring flick) |
| **`F`** (right index push) | **`F'`** (left index push) |
| **`F'`** (right index pull) | **`F`** (left index pull) |
| **`B'`** (right ring pull) | **`B`** (left ring pull) |
| **`B`** (left ring pull) | **`B'`** (right ring pull) |

#### Mathematical Scaling:
$$\text{effort}(A_{\text{mirror}}, B_{\text{mirror}}) = \text{effort}(A, B) \times 1.15$$

**Key Outcomes:**
* `U' L` mirrors `U R'`: $0.496 \times 1.15 = \mathbf{0.570}$.
* `L' U'` mirrors `R U`: $0.498 \times 1.15 = \mathbf{0.573}$.
* `L U'` mirrors `R' U`: $0.498 \times 1.15 = \mathbf{0.573}$.
* Single `L'` = $R \times 1.15 = 0.794 \times 1.15 = \mathbf{0.913}$ (replaces `3.062`).
* Single `L2` = $R2 \times 1.15 = 0.499 \times 1.15 = \mathbf{0.574}$ (replaces `2.966`).
* Single `L` = `0.643` (retained, matches $R' \times 1.092$).

---

### 5.4 Rule 3: Double-Move Monotonicity Enforcement (Option A)
Executing a 180° double-flick ($X2$) must always require at least **15% more execution effort** than the fastest single flick ($X$ or $X'$) on that face from the same preceding move:
$$\text{effort}(A, X2) = \max\Big(\text{effort}(A, X2),\ \min\big(\text{effort}(A, X),\ \text{effort}(A, X')\big) \times 1.15\Big)$$

**Key Outcomes:**
* `M' U2`: Raised from the `0.497` floor clamp to $0.831 \times 1.15 = \mathbf{0.956}$.
* `FR2`: Raised from the `0.498` floor clamp to $0.499 \times 1.15 = \mathbf{0.574}$.
* `UB2`: Raised from `0.494` to $0.750 \times 1.15 = \mathbf{0.863}$.
* `U'R2`: Raised from `0.498` to $0.498 \times 1.15 = \mathbf{0.573}$.

---

### 5.5 Rule 4: $M$-Slice & $U$-Layer Calibration (2H LSE Biomechanics)

During 2H Roux Last Six Edges (LSE), the solver's hands are decoupled: the right hand operates the $M$ slice while the left hand operates the $U$ layer (assuming standard right-handed slice flicking).

#### Fingertrick Biomechanics:
* **Left Hand ($U$ Layer):**
  * $U'$: Left index pull with the finger pad $\implies$ fast, ergonomic baseline ($1.00\times$).
  * $U$: Left index push using the fingernail / back of the finger $\implies$ biomechanically awkward and slower ($\approx 1.25\times$ penalty).
  * $U2$: Left double flick $\implies$ strictly $\ge 1.15\times$ single flick effort.
* **Right Hand ($M$ Slice):**
  * $M'$: Downward ring flick from DB $\implies$ primary fluid slice baseline ($1.00\times$). Single $M' = 0.498$ is retained.
  * $M$: Upward push from DF $\implies$ slower than downward flick ($\approx 1.25\times$). Single $M = 1.45$ (replaces synthetic $2.994$).
  * $M2$: Ring $\to$ middle double flick from DB $\implies \ge 1.15\times$ single $M'$. Single $M2 = 0.826$ is retained.

#### Calibrated $M^* \to U^*$ Transitions:
| Bigram | Calibrated Effort | Biomechanical Rationale |
| :--- | :--- | :--- |
| **`M'U'`** | **`0.831`** | Empirical index pull baseline. |
| **`M'U`** | **`1.063`** | Empirical fingernail push baseline. |
| **`M'U2`** | **`0.956`** | Double-flick monotonicity ($0.831 \times 1.15$). |
| **`M2U'`** | **`0.820`** | Index pull; faster than push, corrects H-perm drill bias. |
| **`M2U`** | **`0.884`** | Empirical fingernail push from drilled LSE. |
| **`M2U2`** | **`0.980`** | Double-flick monotonicity ($\ge 0.820 \times 1.15$). |
| **`MU'`** | **`0.980`** | Index pull after upward push ($\approx 1.25 \times 0.831$). |
| **`MU`** | **`1.220`** | Fingernail push after upward push ($\approx 1.25 \times 0.980$). |
| **`MU2`** | **`1.150`** | Double-flick monotonicity ($\ge 0.980 \times 1.15$). |

#### Calibrated $U^* \to M^*$ Transitions:
| Bigram | Calibrated Effort | Biomechanical Rationale |
| :--- | :--- | :--- |
| **`U'M'`** | **`0.780`** | Primary fluid trigger; corrects AUF recognition pause (`1.401` in raw). |
| **`U'M`** | **`0.980`** | Upward push after index pull ($\approx 1.25 \times 0.780$). |
| **`U'M2`** | **`0.900`** | Double-flick monotonicity ($\ge 0.780 \times 1.15$). |
| **`UM'`** | **`1.020`** | Right ring flick after awkward fingernail push ($\approx 1.25 \times 0.780$). |
| **`UM`** | **`1.250`** | Right upward push after awkward fingernail push ($\approx 1.25 \times 0.980$). |
| **`UM2`** | **`1.180`** | Right double flick after awkward fingernail push ($\ge 1.020 \times 1.15$). |
| **`U2M'`** | **`0.850`** | Right ring flick after left double flick; corrects 49ms sensor clamp floor (`0.495` in raw). |
| **`U2M`** | **`1.150`** | Right upward push after left double flick. |
| **`U2M2`** | **`0.980`** | Empirical double-flick transition retained. |

#### Left-Handed $M$-Slice Adaptation (`m_slice_hand = "left"`):
When a solver operates the $M$ slice with the left hand, the right hand operates the $U$ layer. This inverts which $U$ direction is a fluid pull vs awkward fingernail push:
* $U$ becomes the right-index pull (pad) $\implies$ fluid trigger.
* $U'$ becomes the right-index push (nail) $\implies$ awkward trigger.

The engine adapts dynamically by swapping $U \longleftrightarrow U'$ across all 6 coupled $M \leftrightarrow U$ bigrams ($M^* U \longleftrightarrow M^* U'$ and $U M^* \longleftrightarrow U' M^*$), while preserving $U2$ double turns and all non-$M$ transitions (`RU`, `R'U'`, `FU`, etc.).


---

### 5.6 Rule 5: Decoupled Opposing Face Pairs ($D \leftrightarrow U$)
For unmeasured defaults between opposing decoupled faces ($D \to U$):
* $D \to U$ is symmetric to $U \to D$ across the two hands:
  * `D' U` mirrors `U D'`: **`0.751`** (left ring $D'$, right index $U$).
  * `D U'` mirrors `U' D`: **`0.478`**.
  * `D2 U` and `U2 D`: **`1.15`**.

---

### 5.7 Rule 6: Biomechanical Fallback Interpolation in Code
In [`src/roux_engine/ergonomics/transition_matrix.py`](../src/roux_engine/ergonomics/transition_matrix.py#L152):
When an arbitrary or novel bigram is queried that is not explicitly present in the matrix, `_interpolate_fallback(prev_move, curr_move)` must:
1. Strip wide notation ($r \to R$, $l \to L$).
2. Query the underlying outer-layer transition.
3. Apply the appropriate drag factor ($1.12\times$ for $r$, $1.15\times$ for $L/l$).
4. Never fall back to an uncalibrated default penalty.

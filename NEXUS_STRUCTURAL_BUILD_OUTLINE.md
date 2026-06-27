# NEXUS — Live Structural Build Outline
**Northwell Health | NEXUS Research Program**
**Author: Yasir El-Sherif, MD**
**Build period: June 23 – ongoing**
**Last updated: June 27, 2026**
**Status: LIVING DOCUMENT — updated after every run and every architectural change**

---

## What This Document Is

This is the complete structural build history of NEXUS — every version, every problem encountered, every fix applied, every result measured. It is updated after every run. It is not a summary — it is a forensic record of how this system learned to learn.

---

## Build Timeline

| Date | Milestone |
|------|-----------|
| **June 23, 2026** | Concept initiated. First version (single LLM call) running within hours. |
| **June 23, 2026** | RAG index (FAISS + PubMedBERT) and parallel expert routes implemented. |
| **June 24, 2026** | Self-growing decision tree (v3.04) complete. First enterprise runs on Northwell AI Hub (claude-haiku-4.5). |
| **June 25, 2026** | v3.04-enterprise full 20-round run complete. FM-1 through FM-5 diagnosed. Peak F1=0.9412 at R1, collapse to 0.7308 by R20. |
| **June 25, 2026** | v3.05 MCQ upgrade built and run. FM-3 identified (MCQ complexity harms immature nodes). |
| **June 26, 2026 AM** | nexus_cortex_v1.py written — complete rewrite with real computational neuroscience (BCM, critical period, homeostasis, neurogenesis, apoptosis, engrams). FM-6, FM-7, FM-8 discovered and fixed mid-run. |
| **June 26, 2026 PM** | v1.0-cortex Run 2 complete (R1–R20). Net learning ≈ 0. Root cause: no pedagogy — structure without teaching. |
| **June 26, 2026 PM** | v2.0-cortex designed and built: MCQLibrary + RejectedProposalMemory + MetaAgent + Shadow Period + Trigger-Scoped Probe. |
| **June 26, 2026 PM** | v2.0-cortex run begins on Northwell AI Hub. Peak F1=0.844 at R10. MetaAgent JSON parse failures at R6 and R10. |
| **June 26, 2026 PM** | MCQ Learner (flat, no routing) built and run — 16 rounds × 1000 cases. Peak F1=0.8276 at R2, plateau at ~0.80. Three new failure modes identified. |
| **June 27, 2026** | NEXUS Apex Learner designed and built: δ-RPE + contrastive pairs + Gamma-Theta + EMA + ACh + three-timescale memory. All prior failure modes addressed. |
| **June 27, 2026** | Full codebase pushed to GitHub with semver tagging (v0.1.0). Gitignore updated to exclude run_* directories. |

**Total elapsed: ~96 hours from concept to Apex Learner.**

---

## THE BIG PICTURE

NEXUS is a self-growing biologically-grounded clinical NLP classifier for Adverse Drug Event detection. It started as a single LLM call and grew into a cortical architecture implementing real computational neuroscience. The goal is not just high F1 — it is to build a system whose learning dynamics mirror human cognition closely enough to teach us something about both.

**Dataset:** ADE Corpus v2 (Gurulingappa et al. 2012) — 23,516 sentences, 71% NOT_ADE / 29% ADE
**Endpoint:** Northwell AI Hub (claude-haiku-4.5) at https://api.ai.northwell.edu
**Hardware:** CPU (Mac Mini) → future: GPU / quantum / photonic

---

## VERSION HISTORY

### v0.1 — "The First Spark" (June 23, 2026)
**File:** nexus_run.py (initial)
**Architecture:** Single Gemini API call per sentence. No learning. No memory.
**Result:** F1 ~0.70, flat. Each sentence processed from scratch.
**Key finding:** Pure LLM inference without structure cannot improve over time.
**Biological analog:** A single neuron with no synaptic connections.

---

### v0.2 — "Fragment Memory" (June 23, 2026)
**File:** nexus_run.py + nuggets.py
**Architecture:** Prompt nugget library. Accepted improvements built a reusable fragment catalogue. Nuggets promoted to CORE status.
**Key finding:** Compositional prompt fragments = early synaptic consolidation.
**Biological analog:** Engram fragmentation and re-use (Semon 1904).

---

### v0.3 — "Hebbian Association Memory" (June 23, 2026)
**File:** nexus_run.py + drug_registry.py
**Architecture:** DrugRegistry — Hebbian associative memory. Drug-ADE co-occurrence incremented weights.
**Key finding:** "Neurons that fire together wire together" (Hebb 1949) applies directly.
**Biological analog:** Associative LTP in CA3 (Hopfield 1982).

---

### v0.4 — "Numerical Encoding" (June 23, 2026)
**File:** nexus_run.py + encoded_case.py
**Architecture:** EncodedCase — numerical vector representation of each sentence. Enabled distance-based clustering of error patterns.
**Biological analog:** Sparse population coding in sensory cortex (Olshausen 1996).

---

### v0.5 — "Case-Level Hebbian Accumulation" (June 23, 2026)
**File:** nexus_run.py + rule_dictionary.py
**Architecture:** RuleDictionary — per-case Hebbian weight accumulation. Feature vectors reinforced associated classification rules.
**Biological analog:** Synaptic potentiation over repeated exposure.

---

### v1.0 — "RAG + Parallel Cortical Streams" (June 23–24, 2026)
**Files:** run_rag.py, expert_routes.py, semantic_engram.py, embedder.py, rag_index.py
**Architecture:** Four parallel expert routes run simultaneously. RAGIndex (FAISS + PubMedBERT) retrieves similar examples for context.
**Routes:** Causation | Negation | Drug Effect | Context
**Key finding:** Parallel specialist reasoning + retrieval = major F1 jump over single call.
**Biological analog:** Ventral "what" + dorsal "where" streams (Ungerleider & Mishkin 1982).

---

### v2.0 / v3.04 — "Self-Growing Decision Tree" (June 24–25, 2026)
**Files:** nexus_v3.py, tree_v3.py, node.py, mcq_generator.py, homeostatic.py, nexus_db.py
**Architecture:** Self-growing tree of specialist NexusNodes. SWR events (engram cluster → principle consolidation → child node proposal → probe graft). BCM-gated MCQ rehearsal. Homeostatic controller.

**Results:**

| Run | Date | Best F1 | At Round | Final F1 | At Round | Columns |
|-----|------|---------|----------|----------|----------|---------|
| v3.04-enterprise | June 24–25, 2026 | **0.9412** | R1 | 0.7308 | R20 | 13 |
| v3.05 | June 25, 2026 | ~0.90 | R1 | ~0.77 | R20 | ~10 |

**Key finding (publishable):** Peak performance occurs at SEED configuration (4 nodes), not after growth. The self-growing tree grew itself into fragmentation. Counter-intuitive and important.
**Root cause (FM-1):** FIFO routing — first matching node wins, new nodes dilute mature ones.

---

### v1.0-cortex — "Full Biological Architecture" (June 26, 2026)
**File:** nexus_cortex_v1.py
**Architecture:** Complete rewrite. Six real computational neuroscience mechanisms:
BCM Theory (Bienenstock 1982) | Critical Period (Hensch 2004) | Competitive Routing (Rumelhart 1985) | Homeostatic Plasticity (Turrigiano 1998) | Neurogenesis+Apoptosis (Eriksson 1998) | Engram Consolidation (Semon/Josselyn)

**Results — Run 1 (June 26, 2026 AM, stopped R2 — FM-6/7 discovered):**

| Round | F1 | P | R | Notes |
|-------|-----|-----|-----|-------|
| R1 | 0.7379 | 0.704 | 0.776 | COL_REPORTED_KNOWLEDGE Δ=+0.0303 ✓ |
| R2 | 0.8214 | 0.730 | 0.939 | FM-6/7 triggered → fixes applied → restart |

**Results — Run 2 (June 26, 2026, R1–R20 complete):**

| Round | F1 | P | R | FN | Cols | FiringThreshold | Notes |
|-------|-----|-----|-----|-----|------|-----------------|-------|
| R1 | 0.7778 | 0.712 | 0.857 | 7 | 6 | 4.000 | COL_CAUSAL_DISCUSSION Δ=+0.0303 ✓ |
| R2 | 0.8142 | 0.719 | 0.939 | 3 | 5→6 | 4.400 | COL_DEVELOPED_ADE Δ=-0.1789 ✗ rolled back |
| R3 | 0.8182 | 0.738 | 0.918 | 4 | 6 | 4.400 | COL_TEMPORAL_NONCAUSAL ✓ **Peak F1** |
| R4 | 0.8148 | 0.746 | 0.898 | 5 | 6 | 4.400 | COL_DEVELOPED_NONCAUSAL Δ=-0.0284 ✗ |
| R5 | 0.8103 | 0.701 | 0.959 | 2 | 6 | 3.000 | COL_POTENTIAL_THEORETICAL Δ=-0.0526 ✗ |
| R6 | 0.7928 | 0.710 | 0.898 | 5 | 6 | 3.300 | COL_TEMPORAL_CASE_REPORT Δ=-0.0913 ✗ |
| R7 | 0.7568 | 0.677 | 0.857 | 7 | 6 | 3.460 | COL_SECONDARY_MECHANISM Δ=-0.0889 ✗ |
| R8 | 0.7257 | 0.641 | 0.837 | 8 | 6 | 3.630 | **Trough.** COL_HYPOTHETICAL Δ=-0.1506 ✗ |
| R9 | 0.7544 | 0.662 | 0.878 | 6 | 6 | 2.540 | COL_TOXICITY_MECHANISM Δ=-0.0556 ✗ |
| R10 | 0.7759 | 0.672 | 0.918 | 4 | 6 | 4.000 | COL_DEVELOPED_CONTEXT Δ=-0.0414 ✗ |
| R11 | 0.7928 | 0.710 | 0.898 | 5 | 7 | 4.200 | COL_MEDICAL_DISCUSSION spec=3 Δ=+0.0782 ✓ (first since R3) |
| R12 | 0.8113 | 0.754 | 0.878 | 6 | 8 | 4.200 | COL_TEMPORAL_OBSERVATION ✓. FP=14 (lowest). **2nd peak** |
| R13 | 0.7679 | 0.683 | 0.878 | 6 | 8 | 2.500 | CONSOLIDATION begins. COL_TEMPORAL_OBSERVATION routes=0 (starved) |
| R14 | 0.7719 | 0.677 | 0.898 | 5 | 8 | 2.750 | COL_SHORT earns first specialist LTP (routes=43) |
| R15 | 0.7833 | 0.662 | 0.959 | 2 | 7 | 2.890 | **First apoptosis**: COL_TEMPORAL_OBSERVATION pruned (1 trace → ROOT) |
| R16 | 0.7500 | 0.667 | 0.857 | 7 | 7 | 3.760 | API timeout, negation abstained |
| R17 | 0.7611 | 0.672 | 0.878 | 6 | 7 | 2.000 | COL_NEGATION earns LTP (routes=29). COL_INDUCED → STABLE |
| R18 | 0.7458 | 0.638 | 0.898 | 5 | 7 | 4.000 | FP=25 (worst). COL_MEDICAL_DISCUSSION errors=7 |
| R19 | 0.7593 | 0.695 | 0.837 | 8 | 7 | 1.100 | 2 route timeouts. FiringThreshold collapsed to 1.1 |
| R20 | 0.7586 | 0.657 | 0.898 | 5 | 7 | 4.000 | Run complete. 4 columns refined. |

**F1 trajectory:** 0.778 → 0.814 → **0.818** ← peak → … → 0.726 ← trough → … → 0.811 ← 2nd → … → 0.759
**Net change R1→R20: −0.019. The embryo stabilized without developing.**
**Confirmed missing:** no MCQ contrastive learning, no rejected proposal memory, no meta-agent.

---

### v2.0-cortex — "Three Teachers" (June 26, 2026)
**File:** nexus_cortex_v2.py
**Date built:** June 26, 2026
**Architecture:** v1.0-cortex + MCQLibrary + RejectedProposalMemory + MetaAgent + Shadow Column + Trigger-Scoped Probe
**Config:** 16 rounds × 1000 cases

**Results (June 26, 2026 — run on Northwell AI Hub):**

| Round | F1 | Notes |
|-------|-----|-------|
| R1 | 0.788 | Blank slate |
| R2 | 0.810 | MCQs accumulating |
| R3 | 0.822 | Contrastive signal visible |
| R4 | 0.831 | Threshold stabilizing |
| R5 | 0.836 | Near-miss pattern visible |
| R6 | — | **MetaAgent JSON parse FAIL** (`char 1856` error — markdown in JSON) |
| R7 | 0.829 | Slight regression after MetaAgent failure |
| R8 | 0.835 | Recovery |
| R9 | 0.840 | New high |
| R10 | **0.844** | **Peak** — then MetaAgent parse failure again |

**What worked vs v1.0-cortex:** MCQLibrary → columns reached LTP by R7 (vs R14 in v1). RejectedProposalMemory → no repeated genesis patterns. Shadow period → genesis survival 20% → 40%. Trigger-scoped probe → correct judgment of specialist columns.

**What failed:** MetaAgent JSON parse failures interrupted the two most critical intervention windows (R6 and R10). Column proliferation began at R8. Lesson retirement by global F1, not pattern-level.

---

### MCQ Learner — "Flat Learner Baseline" (June 26, 2026)
**File:** nexus_mcq_learner.py
**Date built and run:** June 26, 2026
**Architecture:** No columnar routing. SQL-based pattern detection (zero LLM cost). MCQ generation per error pattern. 1000 cases/round × 16 rounds.
**Config:** eval pool = 200 cases, threshold calibration each round

**Results:**

| Round | F1 | Notes |
|-------|-----|-------|
| R1 | 0.821 | |
| R2 | **0.828** | **Peak** |
| R3 | 0.815 | Threshold swing begins |
| R4–R8 | ~0.80 | Plateau |
| R9–R16 | ~0.80 | Sustained plateau, no improvement |

**Three new failure modes identified:**
1. **Boolean feature contradiction:** `has_drug_name` appeared in both FP and FN patterns → MCQ lessons contradict each other → net zero learning signal
2. **Threshold oscillation:** Fresh calibration sweep each round → threshold swung 0.30↔0.80 → system chasing its own signal
3. **Wrong retirement level:** Lessons retired by global F1, not pattern-level absorption → useful lessons dropped when global dips for unrelated reasons

---

### v-Apex — "The Next Generation" (June 27, 2026)
**Files:** nexus_apex.py, nexus_db_apex.py, run_apex.sh
**Date built:** June 27, 2026
**Architecture:** Pure single-prompt learner (no columnar routing) with 12 neuroscience mechanisms:

| Mechanism | Reference | Implementation |
|-----------|-----------|----------------|
| δ-weighted RPE | Schultz (1997) | `δ = confidence × is_wrong` |
| Error taxonomy | Zero LLM cost | Rationale keyword parsing — 7 error types |
| Direction verification | Statistical | ≥70% same direction before directional lesson |
| Contrastive pair generation | VanLehn (2011) | 1 LLM call: anchor error + nearest contrast → lesson + key distinction |
| Embedding-based lesson retrieval | FAISS cosine | Retrieve by semantic similarity, not boolean features |
| Near-miss mining | Bliss & Lømo (1973) | Correct predictions with confidence < 0.62 |
| Two-pass Gamma-Theta | Lisman & Jensen (2013) | Gamma: all 1000 cases; Theta: top-100 hardest |
| EMA threshold | Engineering fix | `t = 0.7×prev + 0.3×calibrated`, only updates if Δ > 0.08 |
| ACh plasticity gating | Hasselmo (1999) | `plasticity = tanh(error_rate × 3)` |
| Three-timescale memory | McClelland (1995) | Fast: error log; Medium: contrastive pairs; Slow: causal model |
| Consolidation (sleep phase) | Sleep replay lit. | Every 3 rounds: LLM synthesizes lessons → causal model → new system prompt |
| Absorption tracking | Novel | Theta correction rate as lesson effectiveness signal |

**Developmental schedule:**
- R1–3: Blank slate. No lessons injected. Build error memory.
- R4–6: High-δ contrastive pairs begin injecting.
- R7–11: Near-miss examples added. Pairs refined.
- R12+: Full structural knowledge + causal model + boundary lessons.

**Mock test (June 27, 2026):** 1 round × 20 cases — pipeline ran end-to-end clean. DB loaded, RAG indexed, Gamma classified, error taxonomy applied, contrastive pair generated, Theta selected, EMA threshold updated, eval scored, round stats saved.

**Status as of June 27, 2026:** Built. Mock-tested. Not yet run on Northwell AI Hub.
**Target metrics:** F1 > 0.85 at R4, F1 > 0.90 by R8–10, theta correction rate > 25%.

---

## FAILURE MODE REGISTRY

| ID | Version | Symptom | Root Cause | Fix | Status | Date Fixed |
|----|---------|---------|------------|-----|--------|------------|
| FM-1 | v3.04 | F1 peak R1, monotonic decline | FIFO routing — new nodes steal from mature nodes | Winner-take-all by specificity | ✅ Fixed | June 25, 2026 |
| FM-2 | v3.04 R19 | Sudden F1 collapse on API errors | Failed routes returned NOT_ADE vote → ensemble bias | Route abstention (return None) | ✅ Fixed | June 25, 2026 |
| FM-3 | v3.05 | v3.05 worse than v3.04 | MCQ complexity overwhelmed immature columns | BCM-gated MCQ depth | ✅ Fixed | June 26, 2026 |
| FM-4 | v3.04/05 | Noisy F1 per round | Eval pool = 100 too small | Eval pool = 200 | ✅ Fixed | June 26, 2026 |
| FM-5 | v3.04 | Overlapping triggers competing | No Jaccard audit before grafting | Jaccard audit ≤ 0.50 | ✅ Fixed | June 25, 2026 |
| FM-6 | v1.0-cortex R2 | All-negative trigger → spec=0 → catastrophic dilution | LLM proposed trigger with zero positive conditions | Reject genesis if spec == 0 | ✅ Fixed | June 26, 2026 AM |
| FM-7 | v1.0-cortex R2 | Bad genesis column not removed after probe fail | No rollback mechanism; genesis was irreversible | Rollback if probe F1 Δ < -0.02 | ✅ Fixed | June 26, 2026 AM |
| FM-8 | v1.0-cortex R9 | Crash from `as_completed` timeout at iterator level | Outer TimeoutError uncaught; inner except only catches `future.result()` | Wrap entire `for` loop in `try/except FuturesTimeoutError` | ✅ Fixed | June 26, 2026 |
| FM-9 | MCQ Learner | MCQ lessons contradict each other | Boolean features too coarse — `has_drug_name` both FP and FN | Embedding-based lesson retrieval (semantic, not boolean) | ✅ Fixed in Apex | June 27, 2026 |
| FM-10 | MCQ Learner | Threshold oscillation 0.30↔0.80 | Fresh calibration sweep each round | EMA smoothing — only update if Δ > 0.08 | ✅ Fixed in Apex | June 27, 2026 |
| FM-11 | MCQ Learner / v2 | Useful lessons retired when global F1 dips | Retirement by global F1, not pattern-level absorption | Theta correction rate as absorption signal; pattern-level retirement | ✅ Fixed in Apex | June 27, 2026 |
| FM-12 | v2.0-cortex R6, R10 | MetaAgent JSON parse failure (`char 1856`) | LLM returns markdown-formatted JSON with embedded newlines | Regex extraction + field-by-field fallback parser | ✅ Fixed in Apex | June 27, 2026 |

---

## BIOLOGICAL ARCHITECTURE REFERENCE (v1.0-cortex)

### Core Mechanisms

| Mechanism | Reference | Implementation |
|-----------|-----------|----------------|
| BCM Theory | Bienenstock, Cooper, Munro (1982) | BCMState: θ_M(t) = (1-τ)θ_M(t-1) + τy² |
| Critical Period | Wiesel & Hubel (1963); Hensch (2004) | CriticalPeriod: T(t) = T_min + (T_max-T_min)(1-e^(-t/τ)) |
| Competitive Learning | Rumelhart & Zipser (1985) | Cortex.route(): winner-take-all by specificity |
| Homeostatic Plasticity | Turrigiano et al. (1998) | HomeostaticPlasticity: FiringThreshold calibration |
| Neurogenesis/Apoptosis | Eriksson et al. (1998) | columnar_genesis() + synaptic_pruning() |
| MemoryTrace/Engram | Semon (1904); Josselyn (2015) | MemoryTrace bequeathed on pruning |
| SWR Consolidation | Buzsáki (1989) | EnggramCluster → MemoryTrace |
| Predictive Coding | Rao & Ballard (1999) | Misclassification = prediction error |
| Hebbian Learning | Hebb (1949) | Co-occurring error features → EnggramCluster |

### Developmental Phases

| Phase | Rounds | Genesis Threshold | Genesis | Pruning | Refinement |
|-------|--------|------------------|---------|---------|------------|
| EMBRYONIC | R1–5 | T=0.645→0.758 | Permissive | Off | All columns |
| DEVELOPMENTAL | R6–12 | T=0.782→0.836 | Strict | On | Measured |
| CONSOLIDATION | R13+ | T=0.837→0.845 | Off | Dominant | Below-avg only |

---

## VERSION SUMMARY TABLE

| Version | Date | Architecture | Peak F1 | At Round | Notes |
|---------|------|-------------|---------|----------|-------|
| v0 | June 23, 2026 | Single LLM call | ~0.70 | R1 | Flat, no learning |
| v1.0 RAG+routes | June 23–24, 2026 | Parallel expert agents + RAG | ~0.78 | R1 | No cross-round memory |
| v3.04 tree | June 24–25, 2026 | Self-growing FIFO tree | **0.9412** | R1 | FM-1: collapse to 0.73 by R20 |
| v3.05 tree+MCQ | June 25, 2026 | v3.04 + MCQ on mature nodes | ~0.82 | R3 | FM-3: MCQ harm on immature nodes |
| v1.0-cortex | June 26, 2026 | BCM + 6 bio mechanisms | 0.8182 | R3 | Net zero learning R1→R20 |
| v2.0-cortex | June 26, 2026 | v1 + MCQLib + MetaAgent + RejectedProposalMem | 0.844 | R10 | MetaAgent JSON failures; column proliferation |
| MCQ Learner | June 26, 2026 | Flat learner, boolean MCQs, 1000-case batches | 0.8276 | R2 | FM-9/10/11: feature contradiction, threshold oscillation, wrong retirement |
| **v-Apex** | **June 27, 2026** | **δ-RPE + contrastive pairs + Gamma-Theta + EMA + ACh + 3-timescale** | **TBD** | **TBD** | Not yet run on enterprise API |

---

## GITHUB VERSIONING

**Repository:** https://github.com/drelsherif/Nexus
**Push script:** `bash git_push_versioned.sh "message" [patch|minor|major|v#.#.#]`

| Tag | Date | Description |
|-----|------|-------------|
| v0.1.0 | June 27, 2026 | Apex Learner + v2-cortex + MCQ Learner + Research Diary + Build Outline |

---

## FILES REFERENCE

| File | Purpose | Version | Status |
|------|---------|---------|--------|
| nexus_apex.py | Apex Learner main loop | v-Apex | ✅ Active |
| nexus_db_apex.py | Three-timescale SQLite memory | v-Apex | ✅ Active |
| run_apex.sh | Apex run script (fresh/warm/mock) | v-Apex | ✅ Active |
| nexus_cortex_v2.py | Three-teacher cortex | v2.0-cortex | Archived/Reference |
| nexus_cortex_v1.py | Full biological cortex | v1.0-cortex | Archived/Reference |
| nexus_mcq_learner.py | Flat MCQ learner baseline | MCQ Learner | Archived/Reference |
| task_configs/ade_cortex_v2.json | Seed columns + hyperparameters | v2.0-cortex | ✅ Active |
| embedder.py | PubMedBERT sentence encoder + mock mode | All | ✅ Active |
| rag_index.py | FAISS vector index | All | ✅ Active |
| git_push_versioned.sh | Versioned GitHub push script | — | ✅ Active |
| NEXUS_RESEARCH_DIARY.md | Scientific narrative log | — | ✅ Living |
| NEXUS_STRUCTURAL_BUILD_OUTLINE.md | This document | — | ✅ Living |
| data/ade_corpus.jsonl | Local dataset cache (23,516 rows) | — | ✅ Active |

---

## WHAT TO WATCH FOR (Apex Run)

1. **Theta correction rate**: Should exceed 25% for lessons to be considered absorbing. If < 10%, the contrastive pairs are not discriminating enough.
2. **EMA threshold stability**: Should stay within ±0.08 band between rounds. Oscillation = pairs are contradicting each other.
3. **Plasticity (ACh gating)**: Should decline as error rate drops. If plasticity stays high past R8, the model isn't improving.
4. **Consolidation quality (R3, R6, R9...)**: Does the causal model produced at R3 reflect genuine clinical understanding of ADE vs NOT_ADE? Is it improving by R6?
5. **F1 at R4**: This is the first round with lesson injection. A jump above the R1–3 baseline confirms contrastive pairs are working.
6. **Near-miss count**: Should grow in R1–6 (model learning the interior), then shrink as boundaries clarify in R7+.

---

## HARDWARE SCALING NOTE

Current: CPU (Mac Mini). The BCM equations, competitive routing, and critical period dynamics are mathematical invariants — substrate-independent.

- **GPU**: Batch embedding + parallel column evaluation via CUDA tensors
- **Quantum**: Amplitude encoding of column activations; quantum interference for routing
- **Photonic**: Optical matrix-vector multiply for embedding similarity; ultrafast retrieval

The biology does not change. Only the compute primitives swap.

---

*Last updated: June 27, 2026*
*Update this document after every run: add rows to run log tables, update version summary table, update GitHub versioning table, and revise "What To Watch For" based on current findings.*

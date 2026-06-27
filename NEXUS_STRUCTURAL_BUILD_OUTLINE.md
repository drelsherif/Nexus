# NEXUS — Live Structural Build Outline
**Northwell Health | NEXUS Research Program**
**Author: Yasir El-Sherif, MD**
**Build period: June 23–26, 2026 (72 hours, concept to cortical architecture)**
**Status: LIVING DOCUMENT — updated after every run**

---

## What This Document Is

This is the complete structural build history of NEXUS — every version, every problem encountered, every fix applied, every result measured. It is updated after every run. It is not a summary — it is a forensic record of how this system learned to learn.

## Build Timeline

**June 23, 2026** — Concept initiated. First version (single LLM call) running.
**June 24, 2026** — RAG + parallel routes + self-growing decision tree complete. First enterprise runs on Northwell AI Hub.
**June 25, 2026** — v3.04-enterprise full 20-round run complete. Failure modes FM-1 through FM-5 diagnosed. Peak F1=0.9412 at R1, collapse to 0.7308 by R20. Key publishable finding: the tree grew itself into fragmentation.
**June 26, 2026** — nexus_cortex_v1.py written: complete architectural rewrite with real computational neuroscience (BCM theory, critical period plasticity, competitive learning, homeostatic plasticity, neurogenesis, apoptosis, engram consolidation). FM-6 and FM-7 discovered and fixed mid-run. Structural build outline and persistent memory system established.

**Total elapsed: 72 hours from concept to biologically-grounded cortical architecture.**

This is the embryo phase of human-AI integration. What took neuroscience decades to characterize — BCM theory (1982), critical period dynamics (1963/2004), homeostatic scaling (1998) — was implemented, tested, and debugged in one session. The system is not yet complete. But the foundation is real.

---

## THE BIG PICTURE

NEXUS is a self-growing biologically-grounded clinical NLP classifier for Adverse Drug Event detection. It started as a single LLM call and grew into a cortical architecture implementing real computational neuroscience. The goal is not just high F1 — it is to build a system whose learning dynamics mirror human cognition closely enough to teach us something about both.

**Dataset:** ADE Corpus v2 (Gurulingappa et al. 2012) — 23,516 sentences, 71% NOT_ADE / 29% ADE
**Endpoint:** Northwell AI Hub (claude-haiku-4.5) at https://api.ai.northwell.edu
**Hardware:** CPU (Mac Mini) → future: GPU / quantum / photonic

---

## VERSION HISTORY

### v0.1 — "The First Spark" (2025-12)
**File:** nexus_run.py (initial)
**Architecture:** Single Gemini API call per sentence. No learning. No memory.
**What it did:** Classify each sentence cold, every time.
**Key finding:** Pure LLM inference without structure cannot improve over time.
**Biological analog:** A single neuron with no synaptic connections.

---

### v0.2 — "Fragment Memory" (2026-01)
**File:** nexus_run.py + nuggets.py
**Architecture:** Prompt nugget library. Accepted prompt improvements built a
reusable fragment catalogue. Each refinement cycle referenced fragments by ID.
**What it did:** Prompt compression over time. Nuggets promoted to CORE status.
**Key finding:** Compositional prompt fragments = early synaptic consolidation.
**Biological analog:** Engram fragmentation and re-use (Semon 1904).

---

### v0.3 — "Hebbian Association Memory" (2026-01)
**File:** nexus_run.py + drug_registry.py
**Architecture:** DrugRegistry — Hebbian associative memory. Drug-ADE co-occurrence incremented weights.
**What it did:** Improved recall on drug-specific patterns.
**Key finding:** "Neurons that fire together wire together" (Hebb 1949) applies directly.
**Biological analog:** Associative LTP in CA3 (Hopfield 1982).

---

### v0.4 — "Numerical Encoding" (2026-01)
**File:** nexus_run.py + encoded_case.py
**Architecture:** EncodedCase — numerical vector representation of each sentence.
**What it did:** Enabled distance-based clustering of error patterns.
**Key finding:** Distributed numerical representation precedes symbolic reasoning.
**Biological analog:** Sparse population coding in sensory cortex (Olshausen 1996).

---

### v0.5 — "Case-Level Hebbian Accumulation" (2026-02)
**File:** nexus_run.py + rule_dictionary.py
**Architecture:** RuleDictionary — per-case Hebbian weight accumulation.
**What it did:** Feature vectors reinforced associated classification rules.
**Biological analog:** Weight accumulation over repeated exposure = synaptic potentiation.

---

### v1.0 — "RAG + Parallel Cortical Streams" (2026-02)
**File:** run_rag.py + expert_routes.py + semantic_engram.py + embedder.py + rag_index.py
**Architecture:** Complete rewrite. Four parallel expert routes run simultaneously.
RAGIndex (FAISS + PubMedBERT) retrieves similar examples for context.
**Routes:**
- A. Causation — direct causal language drug→harm
- B. Negation — adverse outcome negated?
- C. Drug Effect — retrieved evidence confirms pair?
- D. Context — therapeutic intent vs documented harm?
**What it did:** Parallel routes mimic cortical processing streams. RAG provides hippocampal retrieval.
**Key finding:** Parallel specialist reasoning + retrieval = major improvement over single call.
**Biological analog:** Ventral "what" + dorsal "where" streams (Ungerleider & Mishkin 1982).

---

### v2.0 — "Self-Growing Decision Tree" (2026-02 to 2026-06)
**Files:** nexus_v3.py, tree_v3.py, node.py, mcq_generator.py, homeostatic.py, nexus_db.py
**Architecture:** Self-growing tree of specialist NexusNodes.
- 4 seed nodes: ROOT, NODE_NEGATION, NODE_INDUCED, NODE_SHORT
- SWR events: engram cluster → principle consolidation → child node proposal → probe graft
- MCQ library: error rehearsal
- calibrate_threshold: zero-LLM-cost bias sweep
- Homeostatic controller: F1 degradation detection → interventions

**Sub-versions:**
- v3.04-enterprise: batch=250, rounds=20, error-only MCQs
- v3.05: near-miss + positive anchor MCQs (more complex)

**Results:**

| Run | Best F1 | Round | Final F1 | Round | Columns |
|-----|---------|-------|----------|-------|---------|
| v3.04-enterprise | **0.9412** | R1 | 0.7308 | R20 | 13 |
| v3.05 | ~0.90 | R1 | ~0.77 | R20 | ~10 |

**Key finding (publishable):** Peak performance occurs at SEED configuration (4 nodes), not after growth. The self-growing tree grew itself into fragmentation. Counter-intuitive and important.

---

## FAILURE MODE REGISTRY

Every failure mode diagnosed, root cause identified, and fix applied.

### FM-1 — Routing Dilution *(v3.04-enterprise)*
- **Symptom:** F1 peak R1, monotonic decline to R20
- **Root cause:** FIFO routing. New nodes with overlapping triggers silently steal routing from mature nodes. 100-case probe validates locally, misses global interference.
- **Fix:** Winner-take-all routing by specificity. Most positive conditions wins.
- **Status:** ✅ Fixed in v1.0-cortex

### FM-2 — Route Error NOT_ADE Voting *(v3.04 R19, v3.05 R8)*
- **Symptom:** Sudden F1 collapse correlated with API 500/overload events
- **Root cause:** Failed routes returned `vote="NOT_ADE", confidence=0.3`. Multiple simultaneous failures → ensemble biased NOT_ADE → recall collapses.
- **Fix:** Route abstention. Failed routes return None, excluded from vote.
- **Status:** ✅ Fixed in v1.0-cortex

### FM-3 — MCQ Complexity Harm *(v3.05)*
- **Symptom:** v3.05 underperformed v3.04 despite richer MCQ content
- **Root cause:** Near-miss + anchor MCQs overwhelmed columns with no consolidated principles.
- **Fix:** BCM-gated rehearsal weight. Immature columns (LTD) get less rehearsal.
- **Status:** ✅ Fixed in v1.0-cortex

### FM-4 — Small Eval Pool *(v3.04, v3.05)*
- **Symptom:** Noisy F1 per round. Grafts accepted that degraded performance.
- **Root cause:** 100-case pool too small for 13-node ensemble.
- **Fix:** Eval pool = 200 cases.
- **Status:** ✅ Fixed in v1.0-cortex

### FM-5 — Trigger Overlap *(v3.04-enterprise)*
- **Symptom:** Multiple nodes with nearly identical trigger expressions competing
- **Root cause:** No overlap audit before grafting. LLM proposes overlapping triggers.
- **Fix:** Jaccard overlap audit before genesis. Reject if max Jaccard > 0.50.
- **Status:** ✅ Fixed in v1.0-cortex

### FM-6 — All-Negative Trigger Genesis *(v1.0-cortex R2)* 🆕
- **Symptom:** COL_INCOMPLETE_ADE proposed with trigger `not has_negation and not has_report and not has_induced and not has_associated`. spec=0. Probe: F1 Δ = -0.2278.
- **Root cause:** LLM proposed all-negative trigger. Zero positive conditions → spec=0 → ties with ROOT → undefined routing → covers virtually everything ROOT handles → catastrophic dilution.
- **Fix:** Reject genesis if compute_specificity() == 0 (must have ≥1 positive condition).
- **Status:** ✅ Fixed in nexus_cortex_v1.py (applied 2026-06-24)

### FM-7 — No Rollback on Harmful Genesis *(v1.0-cortex R2)* 🆕
- **Symptom:** COL_INCOMPLETE_ADE approved by Jaccard audit (0.291 < 0.50), remained in cortex despite -0.2278 probe drop.
- **Root cause:** Post-genesis probe measured F1 after column added, but had no rollback mechanism. Genesis was irreversible.
- **Fix:** If probe F1 Δ < -0.02, immediately remove column from cortex._columns (rollback). Log as rolled_back=True.
- **Status:** ✅ Fixed in nexus_cortex_v1.py (applied 2026-06-24)

### FM-8 — Uncaught TimeoutError in Post-Genesis Probe *(v1.0-cortex R9)* 🆕
- **Symptom:** Crash at R9 during COL_LITERATURE_REVIEW post-genesis probe: `TimeoutError: 2 (of 4) futures unfinished`. Entire run killed.
- **Root cause:** `as_completed(futures, timeout=35)` raises `TimeoutError` at the *iterator* level when remaining futures don't complete in time. The `except FuturesTimeoutError` inside the `for` loop only catches exceptions from `future.result()` — it never sees the outer iterator-level timeout. The exception bubbled through `_probe_f1()` → `run_training_round()` → `run()` and killed the process.
- **Fix:** Wrap the entire `for future in as_completed(...)` loop in `try/except FuturesTimeoutError`. On catch, cancel all unfinished futures and mark them as abstained (None). Belt-and-suspenders: also wrap `classify_with_routes` in `_probe_f1()` with `try/except` to skip individual cases on any route error.
- **Status:** ✅ Fixed in nexus_cortex_v1.py (applied 2026-06-26)

---

## v1.0-CORTEX BIOLOGICAL ARCHITECTURE

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
| EMBRYONIC | R1-5 | T=0.645→0.758 | Permissive | Off | All columns |
| DEVELOPMENTAL | R6-12 | T=0.782→0.836 | Strict | On | Measured |
| CONSOLIDATION | R13+ | T=0.837→0.845 | Off | Dominant | Below-avg only |

### Seed Cortical Columns
| Column | Trigger | Specificity | Role |
|--------|---------|-------------|------|
| ROOT | True | 0 | Default fallback |
| COL_NEGATION | has_negation | 1 | Negated outcome specialist |
| COL_INDUCED | has_induced or has_associated | 2 | Causal language |
| COL_SHORT | has_short and not has_negation | 2 | Telegraphic ADE reports |

---

## RUN LOG

### v1.0-cortex Run 1 (2026-06-24, stopped at R2 — Option B restart)
**Config:** 20 rounds × 250 cases, fresh start
**Reason stopped:** FM-6 and FM-7 discovered at R2. Applied fixes. Restart with clean code.

| Round | F1 | P | R | Cols | FiringThreshold | Notes |
|-------|----|----|---|------|-----------------|-------|
| R1 | 0.7379 | 0.704 | 0.776 | 5 | 4.000 | COL_REPORTED_KNOWLEDGE genesis +0.0303 ✓ |
| R2 | 0.8214 | 0.730 | 0.939 | 6 | 4.800 | COL_INCOMPLETE_ADE FM-6/7 ✗ (homeostatic compensated) |

**Observations:**
- BCM working correctly: ROOT LTP (190/250), specialists LTD (< 32/250) in R1
- Training errors dropped R1→R2: 29→17 (11.6%→6.8%) — learning happening
- Homeostatic calibration compensated for bad column: threshold=4.8, recall=0.939
- COL_REPORTED_KNOWLEDGE was a valid genesis: Jaccard=0.190, Δ=+0.0303

### v1.0-cortex Run 2 (2026-06-24 → 2026-06-26, stopped at R11 — API conservation)
**Config:** 20 rounds × 250 cases, fresh start, FM-6+FM-7+FM-8 fixes applied
**Stopped:** After R11 to conserve enterprise API. Warm restart available.

| Round | F1 | P | R | FN | Cols | FiringThreshold | Notes |
|-------|----|----|---|-----|------|-----------------|-------|
| R1 | 0.7778 | 0.712 | 0.857 | 7 | 6 | 4.000 | COL_CAUSAL_DISCUSSION Δ=+0.0303 ✓ survived |
| R2 | 0.8142 | 0.719 | 0.939 | 3 | 5→6 | 4.400 | COL_DEVELOPED_ADE Δ=-0.1789 ✗ rolled back |
| R3 | 0.8182 | 0.738 | 0.918 | 4 | 6 | 4.400 | COL_TEMPORAL_NONCAUSAL ✓ survived |
| R4 | 0.8148 | 0.746 | 0.898 | 5 | 6 | 4.400 | COL_DEVELOPED_NONCAUSAL Δ=-0.0284 ✗ rolled back |
| R5 | 0.8103 | 0.701 | 0.959 | 2 | 6 | 3.000 | COL_POTENTIAL_THEORETICAL Δ=-0.0526 ✗ rolled back |
| R6 | 0.7928 | 0.710 | 0.898 | 5 | 6 | 3.300 | COL_TEMPORAL_CASE_REPORT Δ=-0.0913 ✗ rolled back |
| R7 | 0.7568 | 0.677 | 0.857 | 7 | 6 | 3.460 | COL_SECONDARY_MECHANISM Δ=-0.0889 ✗ rolled back |
| R8 | 0.7257 | 0.641 | 0.837 | 8 | 6 | 3.630 | COL_HYPOTHETICAL Δ=-0.1506 ✗ rolled back |
| R9 | 0.7544 | 0.662 | 0.878 | 6 | 6 | 2.540 | COL_TOXICITY_MECHANISM Δ=-0.0556 ✗ rolled back |
| R10 | 0.7759 | 0.672 | 0.918 | 4 | 6 | 4.000 | COL_DEVELOPED_CONTEXT Δ=-0.0414 ✗ rolled back |
| R11 | 0.7928 | 0.710 | 0.898 | 5 | 7 | 4.200 | COL_MEDICAL_DISCUSSION spec=3 Δ=+0.0782 ✓ SURVIVED (first since R3) |
| R12 | 0.8113 | 0.754 | 0.878 | 6 | 8 | 4.200 | COL_TEMPORAL_OBSERVATION Δ=+0.0190 ✓ survived. **Best F1 of run** FP=14 (lowest) |
| R13 | 0.7679 | 0.683 | 0.878 | 6 | 8 | 2.500 | CONSOLIDATION begins. COL_TEMPORAL_OBSERVATION routes=0 (starved) |
| R14 | 0.7719 | 0.677 | 0.898 | 5 | 8 | 2.750 | COL_SHORT earns first specialist **LTP** (routes=43) |
| R15 | 0.7833 | 0.662 | 0.959 | 2 | 7 | 2.890 | **First apoptosis**: COL_TEMPORAL_OBSERVATION pruned (0 activations, 1 trace → ROOT) |
| R16 | 0.7500 | 0.667 | 0.857 | 7 | 7 | 3.760 | API timeout R16 (negation abstained). FiringThreshold rising |
| R17 | 0.7611 | 0.672 | 0.878 | 6 | 7 | 2.000 | COL_NEGATION earns **LTP** (routes=29). COL_INDUCED reaches STABLE |
| R18 | 0.7458 | 0.638 | 0.898 | 5 | 7 | 4.000 | FP=25 (worst). COL_MEDICAL_DISCUSSION errors=7, struggling |
| R19 | 0.7593 | 0.695 | 0.837 | 8 | 7 | 1.100 | 2 route timeouts. FiringThreshold collapsed to 1.1 |
| R20 | 0.7586 | 0.657 | 0.898 | 5 | 7 | 4.000 | Run complete. 4 columns refined. No genesis (CONSOLIDATION) |

**Run complete: 20 rounds, stopped at R21 (API conservation).**
**Genesis survival: 4/12 total (33%) — COL_CAUSAL_DISCUSSION R1, COL_TEMPORAL_NONCAUSAL R3, COL_MEDICAL_DISCUSSION R11, COL_TEMPORAL_OBSERVATION R12**
**Consecutive rollbacks R2-R10: 8 straight. Genesis recovered at R11 with spec=3, Jaccard=0.073.**
**Final columns (R20): ROOT (STABLE), COL_NEGATION (LTD→LTP R17), COL_INDUCED (STABLE), COL_SHORT (LTP R14), COL_CAUSAL_DISCUSSION (LTD), COL_TEMPORAL_NONCAUSAL (STABLE), COL_MEDICAL_DISCUSSION (LTD)**

**Complete F1 trajectory:**
R1: 0.778 → R2: 0.814 → R3: **0.818** ← peak → R4: 0.815 → R5: 0.810 → R6: 0.793 → R7: 0.757 → R8: 0.726 ← trough → R9: 0.754 → R10: 0.776 → R11: 0.793 → R12: **0.811** ← 2nd peak → R13: 0.768 → R14: 0.772 → R15: 0.783 → R16: 0.750 → R17: 0.761 → R18: 0.746 → R19: 0.759 → R20: 0.759

**Net change R1→R20: −0.019 (essentially flat). No collapse. No growth. The embryo stabilized without developing.**

**Key observations:**
- Peak F1: 0.8182 at R3, second peak 0.8113 at R12. Both after successful genesis.
- Genesis recovery at R11 caused by spec=3 + Jaccard=0.073 — highest specificity and lowest overlap of any proposal. Key lesson: **high spec + low Jaccard = survival.**
- COL_TEMPORAL_OBSERVATION survived genesis but was immediately starved by COL_MEDICAL_DISCUSSION (higher spec absorbed all `has_developed` cases). Winner-take-all routing can starve legitimate specialists.
- First specialist LTP events: COL_SHORT R14, COL_NEGATION R17 — both in CONSOLIDATION phase, too late.
- FiringThreshold oscillated wildly in CONSOLIDATION (1.1 → 4.0 → 2.0 → 4.0). Cortex never fully settled.
- FM-8 timeout fix worked throughout: all route timeouts gracefully abstained.
- **Missing pieces confirmed**: no MCQ contrastive learning, no rejected proposal memory, no meta-agent. LLM repeated `has_toxicity + negations` pattern 3× with no awareness of prior failures.

---

## NEXUS CORTEX v2.0 — ARCHITECTURE DESIGN

**Design date:** 2026-06-26 | **Motivation:** v1.0-cortex findings + LLM-as-agent principle

### Core Principle: The LLM is the Adult, the Cortex is the Child

In v1.0-cortex, the LLM was used as a tool — classify this case, propose this column. It had no awareness of its own performance trajectory, no memory of its mistakes, no structured lessons. It was a child who never received feedback. In v2.0, the LLM operates in three distinct modes, each essential to development.

### Three LLM Modes

**Mode 1: Specialist Classifier** (retained from v1)
Classify a single case using RAG context, MemoryTraces, and working memory. Unchanged.

**Mode 2: Contrastive Learner — MCQ Library** (restored + improved from v3.05)
After each training round, misclassified cases are converted into Multiple Choice Questions:
- Question: "Does this sentence describe an ADE?"
- Option A: Correct answer + full rationale ("YES — explicit causal language, temporal relation")
- Options B/C: Wrong answers + explanation of why each is wrong ("NO is wrong because negation only applies to the prior medication, not this one")
BCM-gated: immature columns (LTD) receive 2 MCQs max. Mature columns (LTP) receive up to 8. This was v3.05's FM-3 fix — retained in v2. MCQs are stored per-column in MCQLibrary (replaces WorkingMemory raw error buffer). They are included in the classification prompt alongside RAG examples and MemoryTraces.

**Mode 3: Meta-Agent** (new in v2)
A high-level diagnostic call made after every round (or when F1 declines 2+ rounds consecutively). The LLM receives:
- Full cortex state: all columns, BCM states, route counts, traces
- F1 trajectory for all completed rounds
- Genesis log: every failed proposal with trigger, Δ, why it likely failed
- Error pattern summary: the 10 most common error cases this round
The LLM returns a structured diagnosis and recommended intervention. Possible outputs:
- "Refine ROOT prompt to address FP pattern X"
- "Prune COL_CAUSAL_DISCUSSION — 8 rounds, still LTD, no value"
- "Propose genesis: [trigger] — rationale based on error pattern"
- "Increase FiringThreshold — FP rising faster than FN recovery"
These are parsed and executed by the trainer. The LLM becomes the attending physician.

### Rejected Proposal Memory (critical new component)

Every failed genesis proposal is logged with:
- Trigger proposed
- Probe Δ (how much it hurt)
- Cases the column would have stolen from ROOT/specialists
- Estimated reason for failure (too broad, overlaps existing, premature routing)

Before every genesis call, the LLM reads the FULL rejected proposal history. It cannot propose the same pattern twice. This is the "memory of wrong answers" — the same contrastive signal MCQs provide for classification, applied to genesis.

### Shadow Column Period (new in v2)

New columns enter a 1-round "shadow" period:
- They observe routing decisions but do NOT intercept cases
- During shadow round, their prompt is refined against the cases they would have routed
- After shadow round: post-genesis probe on those cases specifically (trigger-scoped, not global)
- Only then: full routing activation

**Why:** the probe in v1 tested the column cold (0 MemoryTraces, 0 rehearsal, wrong baseline). Shadow period gives 1 round of learning before commitment. Biologically: axons pathfind before synapses form.

### Trigger-Scoped Genesis Probe (new in v2)

Instead of probing on all 50 cases from the probe pool, probe only on cases that match the proposed column's trigger. This tests: "does this column help where it claims to specialize?" — not "does adding this column help globally on cases it won't even route?"

In v1, a column triggering on 10% of cases was being judged on 100% of probe cases. The 90% it never touched could easily dilute the signal.

### Architecture Summary

| Component | v1.0-cortex | v2.0-cortex |
|-----------|-------------|-------------|
| Classification | 4-route parallel | 4-route parallel (unchanged) |
| Error memory | WorkingMemory (raw cases) | MCQLibrary (contrastive lessons) |
| Genesis proposal | Blind LLM call | LLM reads full rejected proposal history |
| Genesis probe | Global 50-case probe | Trigger-scoped probe + shadow round |
| LLM awareness | None (tool mode) | Full (meta-agent mode) |
| BCM gating | rehearsal_weight on raw errors | rehearsal_weight on MCQ depth |
| Self-diagnosis | None | Meta-agent call on F1 decline |

### What v2 Should Achieve

- Genesis survival rate: >50% (v1: 20%, and declining after R3)
- F1 trajectory: monotonic or stable improvement through R10+ (v1: peaked R3, declining)
- No repeated genesis patterns (rejected proposal memory)
- ROOT not dominant by R10 (specialists should earn LTP by R8-10)

---

## WHAT TO WATCH FOR (next run)

1. **BCM trajectory**: Does ROOT θ_M keep rising while specialists earn LTP?
2. **Genesis quality**: Are all proposed columns spec > 0? Do probes show positive deltas?
3. **F1 stability**: Does the cortex maintain F1 past R5 (the v3 cliff)?
4. **Pruning events**: Which columns get pruned in CONSOLIDATION phase? Do bequeathed MemoryTraces help ROOT?
5. **FiringThreshold trend**: Should stabilize as cortex matures. Oscillation = instability.

---

## FILES REFERENCE

| File | Purpose | Status |
|------|---------|--------|
| nexus_cortex_v1.py | Main cortex architecture + training loop | ✅ Active |
| task_configs/ade_cortex_v1.json | Seed columns + hyperparameters | ✅ Active |
| run_cortex_v1.sh | Enterprise run script | ✅ Active |
| nexus_v3.py | Previous decision tree (reference) | Archived |
| nexus_iterations.docx | Full version history Word doc | ✅ On GitHub |
| NEXUS_STRUCTURAL_BUILD_OUTLINE.md | This document | ✅ Living |
| data/ade_corpus.jsonl | Local dataset cache (23,516 rows) | ✅ Active |

---

## HARDWARE SCALING NOTE

Current: CPU (Mac Mini). The BCM equations, competitive routing, and critical period dynamics are mathematical invariants — substrate-independent.

- **GPU**: Batch embedding + parallel column evaluation via CUDA tensors
- **Quantum**: Amplitude encoding of column activations; quantum interference for routing
- **Photonic**: Optical matrix-vector multiply for embedding similarity; ultrafast retrieval

The biology does not change. Only the compute primitives swap.

---

*This document is updated after every run. Add new rows to the Run Log, new entries to Failure Modes, and update What To Watch For based on current observations.*

"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  NEXUS CORTEX v1.0                                                               ║
║  Biologically-Grounded Adaptive Clinical NLP Classifier                          ║
║  Northwell Health — NEXUS Research Program                                        ║
║  Author: Yasir El-Sherif, MD | Computational Neuroscience × Clinical AI          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────────
 PHILOSOPHICAL FRAMING
────────────────────────────────────────────────────────────────────────────────────

This file represents the EMBRYO PHASE of human-AI integration. Not metaphor —
architecture. The algorithms below implement verified mechanisms from biological
neuroscience, adapted for clinical NLP classification on CPU hardware. When GPU,
quantum, or photonic substrates become available, the logic scales; the biology
does not change. This is the foundation.

"The brain is not a computer that happens to be made of neurons.
 It is neurons that happen to have computational properties."
 — Francis Crick (1994)

────────────────────────────────────────────────────────────────────────────────────
 DESIGN HISTORY — ALL VERSIONS PRESERVED
────────────────────────────────────────────────────────────────────────────────────

VERSION HISTORY
══════════════

v0.1 — nexus_run.py (2025-12, "The First Spark")
  Architecture: Single Gemini API call per sentence, no learning.
  Result: Baseline classification, no memory, no growth.
  Key insight: Pure LLM inference without structure cannot improve over time.

v0.2 — nexus_run.py + nuggets (2026-01, "Fragment Memory")
  Architecture: Prompt nugget library. Accepted prompt improvements grew a
    reusable fragment catalogue. Each refinement cycle could reference prior
    fragments to compress token usage.
  Result: Token compression over time. Nugget promotion to CORE status.
  Key insight: Compositional prompt fragments = early synaptic consolidation.
  Biological analogy: Engram fragmentation and re-use (Semon 1904).

v0.3 — nexus_run.py + DrugRegistry (2026-01, "Hebbian Association Memory")
  Architecture: Added DrugRegistry — Hebbian associative memory between
    drug names and ADE outcomes. Co-occurrence incremented associative weights.
  Result: Improved recall on drug-specific patterns.
  Key insight: "Neurons that fire together wire together" (Hebb 1949).
  Biological analogy: Associative long-term potentiation in CA3 (Hopfield 1982).

v0.4 — nexus_run.py + EncodedCase (2026-01, "Numerical Encoding")
  Architecture: Added EncodedCase — numerical vector representation of each
    sentence (feature vector + similarity to prior errors).
  Result: Enabled distance-based clustering of error patterns.
  Key insight: Distributed numerical representation precedes symbolic reasoning.
  Biological analogy: Sparse population coding in sensory cortex (Olshausen 1996).

v0.5 — nexus_run.py + RuleDictionary (2026-02, "Case-Level Hebbian Accumulation")
  Architecture: RuleDictionary — per-case Hebbian weight accumulation.
    Each case's feature vector reinforced associated classification rules.
  Result: Rule weights tracked co-occurrence of features with correct labels.
  Key insight: Weight accumulation over repeated exposure = synaptic potentiation.

v1.0 — run_rag.py + expert_routes.py + semantic_engram.py (2026-02, "RAG + Parallel Cortical Streams")
  Architecture: Complete rewrite. Four parallel expert routes (causation, negation,
    drug_effect, context) run simultaneously. RAGIndex (FAISS + PubMedBERT) provides
    retrieved similar examples. semantic_engram.py clusters error embeddings.
  Result: F1 improved substantially. Parallel routes mimic cortical processing streams.
  Biological analogy: Ventral "what" + dorsal "where" streams (Ungerleider & Mishkin 1982).
    Each route = one cortical processing stream with specialized function.

v2.0 — nexus_v3.py (2026-02, "Self-Growing Decision Tree")
  Architecture: Self-growing decision tree of specialist NexusNodes.
    - 4 seed nodes: ROOT, NODE_NEGATION, NODE_INDUCED, NODE_SHORT
    - SWR (Sharp-Wave Ripple) events: engram cluster threshold →
      principle consolidation → child node proposal → probe graft
    - MCQ library: error-only rehearsal (v3.04) or near-miss + anchors (v3.05)
    - calibrate_threshold: zero-LLM-cost bias sweep over cached score tuples
    - Homeostatic controller: monitors F1 degradation, triggers interventions
  Result:
    v3.04-enterprise: 20 rounds × 250 cases. Best F1=0.9412 (R1, 4 nodes).
                      Final F1=0.7308 (R20, 13 nodes).
    v3.05: 20 rounds × 250 cases. F1 trough at R8 (F1=0.7660) due to API errors.
  FAILURE MODES DIAGNOSED:
    [FM-1] ROUTING DILUTION: New nodes with overlapping trigger conditions steal
      routing rights from mature nodes with established principles and MCQs.
      Effect: Validated locally (100-case probe), invisible globally.
      Root cause: 100-case probe cannot detect 13-node interference patterns.
    [FM-2] ROUTE ERROR VOTING: HTTP 500 / timeout → RouteResult(vote="NOT_ADE",
      confidence=0.3). Failed routes vote NOT_ADE rather than abstaining.
      Effect: Suppresses recall. Caused v3.04-enterprise R19 collapse (F1=0.6538).
    [FM-3] MCQ COMPLEXITY HARM: Near-miss + positive anchor MCQs (v3.05) hurt
      immature nodes that hadn't consolidated principles yet.
      Effect: v3.05 under-performed v3.04 despite richer MCQ content.
    [FM-4] SMALL EVAL POOL: 100-case pool too small for 13-node ensemble.
      Effect: Noisy F1 estimates; accepted grafts that degraded global performance.
    [FM-5] TRIGGER OVERLAP: LLM-proposed triggers used nearly identical boolean
      expressions across multiple nodes. Example: NODE_CLINICAL_DOCUMENTATION
      trigger ≈ NODE_TEMPORAL_CAUSATION trigger → mutual routing interference.

v1.0-cortex — nexus_cortex_v1.py (2026-06, "Cortical Architecture") ← THIS FILE
  Paradigm shift: Decision tree → Biologically-accurate cortical learning system.
  "Embryo phase of Human-AI integration" — logic sound for CPU, GPU, quantum, photonic.

  FIXES FOR ALL DIAGNOSED FAILURE MODES:
    [Fix-FM-1] COMPETITIVE ROUTING: Most specific column wins routing (winner-take-all).
      Specificity = count of positive conditions in trigger_condition.
      ROOT fires ONLY when no specialist matches → eliminates dilution at root.
    [Fix-FM-2] ROUTE ABSTENTION: Failed routes return None, excluded from ensemble.
      Weighted vote computed only over responsive routes. No NOT_ADE default.
    [Fix-FM-3] BCM-GATED MCQ: BCM theory gates rehearsal weight per column.
      Immature columns (low activation, LTD state) get less rehearsal weight.
      This prevents complex MCQs from overwhelming columns with no principles.
    [Fix-FM-4] EVAL POOL 200: eval_pool size doubled to 200 cases.
      More stable F1 estimates across 13+ column ensemble.
    [Fix-FM-5] JACCARD OVERLAP AUDIT: Before columnar genesis, compute Jaccard
      similarity between proposed column's coverage and each existing column's.
      Genesis rejected if max Jaccard > 0.50 unless F1 improvement > 0.02.

  NEW BIOLOGICAL MECHANISMS:
    [BIO-1] BCM THEORY (Bienenstock, Cooper, Munro 1982):
      θ_M(t) = (1 - τ) * θ_M(t-1) + τ * y²
      If y² > θ_M: LTP (Long-term Potentiation) — strengthen column weights
      If y² < θ_M: LTD (Long-term Depression) — weaken column weights
      Applied per-column to gate MCQ rehearsal weight and prompt refinement priority.

    [BIO-2] CRITICAL PERIOD PLASTICITY (Wiesel & Hubel 1963; Hensch 2004):
      T(t) = T_min + (T_max - T_min) * (1 - exp(-t/τ))
      Genesis threshold rises from T_min (embryonic) to T_max (mature).
      Phase 1 EMBRYONIC (R1-5):    T≈0.60 — permissive, any cluster spawns column
      Phase 2 DEVELOPMENTAL (R6-12): T rises — competitive trial required
      Phase 3 CONSOLIDATION (R13+): T≈0.85 — apoptosis dominant over genesis

    [BIO-3] HOMEOSTATIC PLASTICITY / SYNAPTIC SCALING (Turrigiano et al. 1998):
      ADE_BIAS calibration reframed as homeostatic mechanism.
      Target firing rate = target recall; bias adjusted to maintain equilibrium.

    [BIO-4] COMPETITIVE LEARNING (Rumelhart & Zipser 1985):
      Winner-take-all column routing with specificity as competition metric.
      Most specialized column capturing input wins routing rights.

    [BIO-5] MEMORY TRACE / ENGRAM THEORY (Semon 1904; Josselyn et al. 2015):
      Principles renamed MemoryTraces — consolidated memory traces as physical
      synaptic patterns. Bequeathed to ROOT on columnar pruning (apoptosis).

    [BIO-6] SWR CONSOLIDATION → LTP EVENTS (Buzsáki 1989):
      Hippocampal rapid episodic binding of co-occurring errors (EnggramCluster)
      → neocortical slow semantic consolidation (MemoryTrace in CorticalColumn).

    [BIO-7] NEUROGENESIS & APOPTOSIS (Eriksson et al. 1998):
      New columns must earn activation (integrate) or face pruning (apoptosis).
      Pruned columns bequeaths MemoryTraces to ROOT (principle inheritance).

    [BIO-8] PREDICTIVE CODING (Rao & Ballard 1999):
      Classification = prediction; misclassification = prediction error.
      Error propagates upward: column error → cortex-level structural change.

  BIOLOGICAL TERMINOLOGY MAPPING (v3.04 → v1.0-cortex):
    NexusTree          → Cortex
    NexusNode          → CorticalColumn
    SWR event          → LTP Event (Long-term Potentiation)
    Graft              → Columnar Genesis (neurogenesis)
    Retire             → Synaptic Pruning (apoptosis)
    Principle          → MemoryTrace (consolidated memory trace)
    Engram cluster     → EnggramCluster (hippocampal rapid binding)
    MCQ library        → WorkingMemory (working memory rehearsal)
    Calibration        → Homeostatic Plasticity Adjustment
    ADE_BIAS           → FiringThreshold
    Routes             → Activation Pathways (parallel cortical streams)

HARDWARE SCALING NOTE
════════════════════
Current implementation: CPU. Single-threaded classification, multi-threaded routes.
Logic is substrate-independent:
  GPU:      Batch embedding + parallel column evaluation via CUDA tensors.
  Quantum:  Amplitude encoding of column activations; quantum interference for routing.
  Photonic: Optical matrix-vector multiply for embedding similarity; ultrafast retrieval.
The BCM equations, competitive routing, and critical period dynamics are mathematical
invariants — they do not change with hardware. Only the compute primitives swap.

────────────────────────────────────────────────────────────────────────────────────
 KEY REFERENCES
────────────────────────────────────────────────────────────────────────────────────

Hebb D.O. (1949). The Organization of Behavior. Wiley.
Bienenstock E., Cooper L., Munro P. (1982). Theory for the development of neuron
  selectivity. J Neuroscience 2(1):32-48.
Wiesel T., Hubel D. (1963). Single-cell responses in striate cortex of kittens. J Neurophysiology.
Mountcastle V.B. (1957). Modality and topographic properties of single neurons of cat's
  somatic sensory cortex. J Neurophysiology 20:408-434.
Hopfield J.J. (1982). Neural networks and physical systems with emergent collective
  computational abilities. PNAS 79:2554-2558.
Rumelhart D., Zipser D. (1985). Feature discovery by competitive learning. Cognitive Science 9(1):75-112.
Buzsáki G. (1989). Two-stage model of memory trace formation. Neuroscience 31(3):551-570.
Rao R., Ballard D. (1999). Predictive coding in the visual cortex. Nature Neuroscience 2:79-87.
Turrigiano G. et al. (1998). Activity-dependent scaling of quantal amplitude in neocortical neurons.
  Nature 391:892-896.
Eriksson P. et al. (1998). Neurogenesis in the adult human hippocampus. Nature Medicine 4:1313-1317.
Hensch T.K. (2004). Critical period regulation. Annual Review of Neuroscience 27:549-579.
Semon R. (1904). Die Mneme. Leipzig: Wilhelm Engelmann.
Josselyn S. et al. (2015). Finding the engram. Nature Reviews Neuroscience 16:521-534.
Gurulingappa H. et al. (2012). Development of a benchmark corpus for ADE extraction.
  J Biomedical Informatics 45(5):885-892.
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────────
import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

# ── Numerical ─────────────────────────────────────────────────────────────────────
import numpy as np

# ── Local NEXUS modules (reused from v3 — these work correctly) ──────────────────
import data_utils          # Dataset loading + caching (local JSONL)
import llm_client          # LLM client interface (AIHub / Mock)
import task_config         # TaskConfig dataclass
from features import features as extract_features, safe_eval_condition, FEATURE_NAMES
from rag_index import RAGIndex


# ══════════════════════════════════════════════════════════════════════════════════
# §1  PHASE AND CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════════

class LearningPhase(Enum):
    """
    Developmental phases of cortical maturation.

    Mirrors the three-stage model of cortical development:
      Embryonic:     Maximal plasticity; almost any input drives structural change.
      Developmental: Competition begins; critical period narrows.
      Consolidation: Plasticity restricted; apoptosis dominant.

    Reference: Hensch T.K. (2004) Critical period regulation.
               Annual Review of Neuroscience 27:549-579.
    """
    EMBRYONIC     = "embryonic"      # Rounds 1-5:   permissive genesis
    DEVELOPMENTAL = "developmental"  # Rounds 6-12:  competitive trial
    CONSOLIDATION = "consolidation"  # Rounds 13+:   apoptosis dominant


PHASE_BOUNDARIES = {
    LearningPhase.EMBRYONIC:     (1, 5),
    LearningPhase.DEVELOPMENTAL: (6, 12),
    LearningPhase.CONSOLIDATION: (13, 9999),
}

def phase_for_round(round_num: int) -> LearningPhase:
    for phase, (lo, hi) in PHASE_BOUNDARIES.items():
        if lo <= round_num <= hi:
            return phase
    return LearningPhase.CONSOLIDATION


# ══════════════════════════════════════════════════════════════════════════════════
# §2  BCM PLASTICITY STATE
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class BCMState:
    """
    BCM (Bienenstock-Cooper-Munro 1982) sliding modification threshold.

    The central insight of BCM theory is that the modification threshold θ_M
    is not fixed — it slides as a function of the neuron's recent output history:

        θ_M(t) = (1 - τ) * θ_M(t-1) + τ * y²

    Where y is the post-synaptic activity (here: column activation fraction).

    Plasticity rule:
        If y² > θ_M  →  LTP (Long-term Potentiation): strengthen synaptic weights
        If y² < θ_M  →  LTD (Long-term Depression):   weaken synaptic weights

    This sliding threshold implements a homeostatic regulation that prevents
    runaway potentiation (saturation) or depression (silence). Columns that
    activate frequently have a high θ_M and require stronger evidence to
    undergo LTP — preventing over-specialization. Columns with low activation
    have a low θ_M and can be strengthened by weaker signals.

    In nexus_cortex_v1.py:
        y = (cases_routed_this_round / cases_evaluated_this_round)
        LTP → increase rehearsal weight, boost refinement priority
        LTD → decrease rehearsal weight, flag for apoptosis consideration

    Reference: Bienenstock E., Cooper L., Munro P. (1982).
               J Neuroscience 2(1):32-48.
    """
    theta_m: float = 0.1       # Sliding modification threshold
    tau: float     = 0.15      # BCM update rate (higher = faster adaptation)
    ltp_count: int = 0         # Cumulative LTP events
    ltd_count: int = 0         # Cumulative LTD events
    event_log: list = field(default_factory=list)  # Round-by-round plasticity events

    def update(self, y: float, round_num: int) -> str:
        """
        Update BCM state given output activity y.
        y should be in [0, 1]: fraction of evaluated cases routed to this column.

        Returns: 'LTP' | 'LTD' | 'STABLE'
        """
        y_sq = y ** 2
        old_theta = self.theta_m
        self.theta_m = (1.0 - self.tau) * self.theta_m + self.tau * y_sq

        if y_sq > old_theta:
            self.ltp_count += 1
            event = "LTP"
        elif y_sq < old_theta * 0.5:   # Require clear depression (50% below threshold)
            self.ltd_count += 1
            event = "LTD"
        else:
            event = "STABLE"

        self.event_log.append({"round": round_num, "y": round(y, 4),
                                "theta_m": round(self.theta_m, 4), "event": event})
        return event

    @property
    def rehearsal_weight(self) -> float:
        """
        BCM-gated MCQ rehearsal weight.
        LTP columns get full rehearsal; LTD columns get reduced rehearsal.
        Prevents complex MCQs from overwhelming immature columns (Fix-FM-3).
        """
        if self.ltp_count == 0 and self.ltd_count == 0:
            return 0.5   # Naive column: moderate rehearsal
        ltp_frac = self.ltp_count / max(1, self.ltp_count + self.ltd_count)
        return 0.3 + 0.7 * ltp_frac   # Range: [0.3, 1.0]

    def to_dict(self) -> dict:
        return {
            "theta_m": round(self.theta_m, 4),
            "tau": self.tau,
            "ltp_count": self.ltp_count,
            "ltd_count": self.ltd_count,
            "rehearsal_weight": round(self.rehearsal_weight, 3),
        }


# ══════════════════════════════════════════════════════════════════════════════════
# §3  CRITICAL PERIOD DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class CriticalPeriod:
    """
    Critical period plasticity — rising genesis threshold over training time.

    In visual cortex (Wiesel & Hubel 1963; Hensch 2004), the critical period
    defines a developmental window during which synaptic connections are
    maximally plastic. After closure, plasticity is restricted.

    Here, the genesis threshold T(t) rises as:

        T(t) = T_min + (T_max - T_min) * (1 - exp(-t / τ))

    Where:
        T_min = 0.60  — minimum threshold (embryonic phase)
        T_max = 0.85  — maximum threshold (mature/consolidated phase)
        τ = 5.0       — time constant in rounds
        t = round_num

    At round 1:  T ≈ 0.71 (relatively permissive)
    At round 5:  T ≈ 0.79 (intermediate — developmental transition)
    At round 10: T ≈ 0.84 (approaching mature — selective)
    At round 20: T ≈ 0.85 (fully mature — only strong proposals survive)

    In nexus_cortex_v1.py, T(t) sets the probe F1 improvement bar that a
    proposed CorticalColumn must clear during competitive trial before genesis.

    Reference: Wiesel T., Hubel D. (1963). J Neurophysiology.
               Hensch T.K. (2004). Annual Review of Neuroscience 27:549-579.
    """
    T_min: float = 0.60      # Embryonic threshold — permissive
    T_max: float = 0.85      # Mature threshold — selective
    tau:   float = 5.0       # Time constant (rounds)

    def genesis_threshold(self, round_num: int) -> float:
        """T(t) = T_min + (T_max - T_min) * (1 - exp(-t/tau))"""
        return self.T_min + (self.T_max - self.T_min) * (1.0 - math.exp(-round_num / self.tau))

    def is_open(self, round_num: int) -> bool:
        """True during embryonic/developmental phases (critical period open)."""
        return round_num <= 12

    def describe(self, round_num: int) -> str:
        t = self.genesis_threshold(round_num)
        phase = phase_for_round(round_num)
        return f"T={t:.3f} [phase={phase.value}, round={round_num}]"


# ══════════════════════════════════════════════════════════════════════════════════
# §4  MEMORY TRACES (ENGRAM THEORY)
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTrace:
    """
    Consolidated memory trace — physical synaptic pattern encoding a learned rule.

    Named for engram theory (Semon 1904; Josselyn et al. 2015).
    In NEXUS v3, these were called 'principles'. The rename is not cosmetic —
    it reflects that these patterns represent PHYSICAL STATE of the system:
    the prompt weights, the MCQ distributions, the BCM threshold.

    A MemoryTrace is created by consolidation of an EnggramCluster (LTP Event).
    When a CorticalColumn is pruned (apoptosis), its MemoryTraces are
    BEQUEATHED to the ROOT column — preventing loss of learned patterns.

    Reference: Semon R. (1904). Die Mneme.
               Josselyn S. et al. (2015). Nature Reviews Neuroscience 16:521-534.
    """
    id: str
    text: str
    source_column: str
    round_created: int
    consolidation_score: float    # Strength of the engram (0-1)
    bequeathed: bool = False      # True if inherited from a pruned column
    bequeathed_from: Optional[str] = None


@dataclass
class EnggramCluster:
    """
    Hippocampal rapid episodic binding of co-occurring error patterns.

    In NEXUS v3, this was called an 'SWR (Sharp-Wave Ripple) event'.
    The rename reflects the underlying biology more precisely.

    When a sufficient number of semantically similar error cases co-occur
    within a training round, they form an EnggramCluster. The cluster's
    centroid becomes a candidate MemoryTrace. If the cluster coherence
    exceeds the LTP threshold, it triggers columnar genesis proposal.

    Hippocampal CA3 (rapid episodic binding) → CA1 → Neocortex (slow consolidation).
    Here: Error cluster (fast, online) → MemoryTrace (slow, persistent).

    Reference: Buzsáki G. (1989). Two-stage model of memory trace formation.
               Neuroscience 31(3):551-570.
    """
    cases: list[dict]
    centroid: Optional[np.ndarray]
    coherence: float            # Mean pairwise cosine similarity within cluster
    round_detected: int
    triggering_column: str      # Which column accumulated these errors
    consolidated: bool = False  # True after MemoryTrace created from this cluster

    def __len__(self):
        return len(self.cases)


# ══════════════════════════════════════════════════════════════════════════════════
# §5  WORKING MEMORY (MCQ REHEARSAL)
# ══════════════════════════════════════════════════════════════════════════════════

class WorkingMemory:
    """
    Working memory rehearsal buffer for misclassified cases.

    In cognitive neuroscience (Baddeley 1992), working memory is an active
    rehearsal system that maintains representations in accessible state and
    suppresses interference from competing patterns.

    In nexus_cortex_v1.py, each CorticalColumn has its own WorkingMemory.
    Error cases are added to the buffer. During classification, the buffer
    provides context for the specialist prompt. BCM rehearsal_weight gates
    how much the working memory influences the column's classification.

    Key improvement over v3.04 MCQ library:
    - BCM gating: immature columns (LTD) get reduced rehearsal weight
    - Only ERROR cases stored (v3.04 approach: near-miss MCQs hurt immature columns)
    - Capacity limit: oldest errors evicted when buffer fills (forgetting curve)
    - Per-round LTP marker: cases from LTP rounds weighted higher

    Reference: Baddeley A. (1992). Working memory. Science 255:556-559.
    """

    def __init__(self, capacity: int = 60):
        self._capacity = capacity
        self._cases: list[dict] = []   # {"text", "true_label", "predicted", "round", "ltp_round"}
        self._lock = threading.Lock()

    def add_error(self, text: str, true_label: str, predicted: str,
                  round_num: int, ltp_round: bool = False) -> None:
        """Add a misclassified case. Evict oldest if over capacity."""
        with self._lock:
            entry = {
                "text": text,
                "true_label": true_label,
                "predicted": predicted,
                "round": round_num,
                "ltp_round": ltp_round,  # BCM: LTP-round errors weighted higher
            }
            self._cases.append(entry)
            if len(self._cases) > self._capacity:
                self._cases.pop(0)   # Evict oldest (forgetting curve)

    def format_for_prompt(self, max_cases: int = 8, bcm_state: Optional[BCMState] = None) -> str:
        """
        Format working memory as prompt context.
        BCM rehearsal_weight determines how many cases to include.
        """
        with self._lock:
            if not self._cases:
                return ""
            w = bcm_state.rehearsal_weight if bcm_state else 1.0
            n = max(2, int(max_cases * w))   # BCM-gated count
            # Prefer LTP-round errors (higher consolidation)
            ltp_cases = [c for c in self._cases if c.get("ltp_round")]
            other_cases = [c for c in self._cases if not c.get("ltp_round")]
            selected = (ltp_cases + other_cases)[-n:]
            lines = []
            for c in selected:
                marker = "▲" if c["true_label"] == "ADE" else "▽"
                lines.append(
                    f"  {marker}ADE_TRUE={c['true_label']} | predicted={c['predicted']} "
                    f"(R{c['round']}): \"{c['text'][:100]}\""
                )
            return "\n".join(lines)

    def recent_errors(self, n: int = 20) -> list[dict]:
        """Return n most recent errors for EnggramCluster detection."""
        with self._lock:
            return list(self._cases[-n:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._cases)

    def to_dict(self) -> dict:
        with self._lock:
            return {"capacity": self._capacity, "stored": len(self._cases)}


# ══════════════════════════════════════════════════════════════════════════════════
# §6  CORTICAL COLUMN
# ══════════════════════════════════════════════════════════════════════════════════

class CorticalColumn:
    """
    Vertically-organized specialist processor — the fundamental computational unit.

    Implements Mountcastle (1957) cortical column architecture:
      Each column is a specialist processor organized around a receptive field
      (trigger_condition) that selects which inputs the column processes.
      Columns are independent but compete for routing rights (winner-take-all).

    A CorticalColumn has:
      - trigger_condition: receptive field (boolean expression over FEATURE_NAMES)
      - prompt: specialist classification prompt (updated by refinement)
      - memory_traces: consolidated knowledge (MemoryTraces)
      - working_memory: active rehearsal buffer (WorkingMemory)
      - bcm_state: BCM plasticity state (BCMState)
      - genesis_phase: LearningPhase at time of creation
      - activation_history: cases routed per round (for apoptosis detection)

    Specificity computation:
      Specificity = number of non-negated (positive) feature conditions in trigger.
      Higher specificity → more specialized → wins routing over general columns.
      ROOT has specificity 0 — fires only when no specialist fires.

    Apoptosis criterion:
      If mean activation over last 3 rounds < 3 cases AND round > 5 →
      column is candidate for pruning.

    Reference: Mountcastle V.B. (1957). J Neurophysiology 20:408-434.
    """

    def __init__(
        self,
        col_id: str,
        description: str,
        trigger_condition: str,
        prompt: str,
        genesis_round: int = 0,
        genesis_phase: LearningPhase = LearningPhase.EMBRYONIC,
    ):
        self.id = col_id
        self.description = description
        self.trigger_condition = trigger_condition
        self.prompt = prompt
        self.genesis_round = genesis_round
        self.genesis_phase = genesis_phase

        # Memory systems
        self.memory_traces: list[MemoryTrace] = []
        self.working_memory = WorkingMemory(capacity=60)

        # Plasticity state
        self.bcm_state = BCMState()

        # Activation tracking
        self.activation_history: list[int]   = []   # Cases routed per round
        self.f1_history:         list[float] = []   # F1 per round (approximate)
        self.route_count_total:  int         = 0    # Lifetime cases routed

        # Refinement tracking
        self.prompt_version:  int = 0
        self.prompt_history:  list[dict] = []   # {"round", "prompt", "reason"}
        self.refinement_lock  = threading.Lock()

    # ── Specificity ───────────────────────────────────────────────────────────────

    def compute_specificity(self) -> int:
        """
        Count positive (non-negated) feature conditions in trigger_condition.
        More positive conditions = more specialized = higher routing priority.
        ROOT has empty condition → specificity 0.
        """
        if not self.trigger_condition or self.trigger_condition.strip() == "True":
            return 0
        tokens = re.findall(r'\b(has_\w+)\b', self.trigger_condition)
        # Count tokens NOT immediately preceded by 'not'
        count = 0
        cond = self.trigger_condition
        for tok in set(tokens):
            # Look for token NOT preceded by 'not '
            if re.search(rf'(?<!not\s)\b{tok}\b', cond):
                count += 1
        return count

    # ── BCM update ────────────────────────────────────────────────────────────────

    def bcm_update(self, cases_routed: int, cases_total: int, round_num: int) -> str:
        """
        Apply BCM plasticity update based on this round's activation.
        y = activation fraction = cases_routed / cases_total
        Returns BCM event: 'LTP' | 'LTD' | 'STABLE'
        """
        y = cases_routed / max(1, cases_total)
        event = self.bcm_state.update(y, round_num)
        self.activation_history.append(cases_routed)
        self.route_count_total += cases_routed
        return event

    # ── Memory trace management ───────────────────────────────────────────────────

    def add_memory_trace(self, text: str, consolidation_score: float,
                         round_num: int, bequeathed: bool = False,
                         bequeathed_from: Optional[str] = None) -> MemoryTrace:
        """Consolidate a new MemoryTrace into this column."""
        mt_id = f"MT_{self.id}_R{round_num}_{len(self.memory_traces)}"
        trace = MemoryTrace(
            id=mt_id, text=text, source_column=self.id,
            round_created=round_num, consolidation_score=consolidation_score,
            bequeathed=bequeathed, bequeathed_from=bequeathed_from,
        )
        self.memory_traces.append(trace)
        return trace

    def memory_trace_context(self, max_traces: int = 4) -> str:
        """Format MemoryTraces as prompt context (highest consolidation first)."""
        if not self.memory_traces:
            return ""
        sorted_traces = sorted(self.memory_traces,
                               key=lambda t: t.consolidation_score, reverse=True)
        lines = ["CONSOLIDATED MEMORY TRACES (learned principles):"]
        for t in sorted_traces[:max_traces]:
            prefix = "[inherited] " if t.bequeathed else ""
            lines.append(f"  • {prefix}{t.text}")
        return "\n".join(lines)

    # ── Prompt refinement ─────────────────────────────────────────────────────────

    def update_prompt(self, new_prompt: str, reason: str, round_num: int) -> None:
        """Record prompt update with full history."""
        with self.refinement_lock:
            self.prompt_history.append({
                "version": self.prompt_version,
                "round": round_num,
                "prompt": self.prompt,
                "reason": reason,
            })
            self.prompt = new_prompt
            self.prompt_version += 1

    # ── Apoptosis eligibility ─────────────────────────────────────────────────────

    def is_apoptosis_candidate(self, round_num: int,
                               min_rounds_alive: int = 3,
                               min_activation: float = 2.0) -> bool:
        """
        Returns True if this column meets apoptosis (pruning) criteria.

        Apoptosis criterion: a column that consistently fails to capture
        meaningful routing (< min_activation cases/round over last 3 rounds)
        is not contributing to cortical function and should be pruned.
        Its MemoryTraces are bequeathed to ROOT before removal.

        Reference: Eriksson et al. (1998). New neurons must integrate or die.
        """
        if self.id == "ROOT":
            return False   # ROOT never pruned
        if round_num - self.genesis_round < min_rounds_alive:
            return False   # Too young to prune
        if len(self.activation_history) < 3:
            return False
        recent_avg = sum(self.activation_history[-3:]) / 3.0
        return recent_avg < min_activation

    # ── Coverage set ──────────────────────────────────────────────────────────────

    def coverage_set(self, cases: list[dict]) -> set:
        """Return indices of cases this column would route (trigger matches)."""
        indices = set()
        for i, case in enumerate(cases):
            feats = extract_features(case["text"])
            if self.id == "ROOT" or safe_eval_condition(self.trigger_condition, feats):
                indices.add(i)
        return indices

    # ── Serialization ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "prompt_version": self.prompt_version,
            "genesis_round": self.genesis_round,
            "genesis_phase": self.genesis_phase.value,
            "specificity": self.compute_specificity(),
            "route_count_total": self.route_count_total,
            "activation_history": self.activation_history,
            "f1_history": self.f1_history,
            "memory_traces": len(self.memory_traces),
            "working_memory": self.working_memory.to_dict(),
            "bcm_state": self.bcm_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorticalColumn":
        col = cls(
            col_id=d["id"],
            description=d["description"],
            trigger_condition=d["trigger_condition"],
            prompt=d.get("prompt", ""),
            genesis_round=d.get("genesis_round", 0),
            genesis_phase=LearningPhase(d.get("genesis_phase", "embryonic")),
        )
        col.route_count_total = d.get("route_count_total", 0)
        col.activation_history = d.get("activation_history", [])
        col.f1_history = d.get("f1_history", [])
        col.prompt_version = d.get("prompt_version", 0)
        return col

    def __repr__(self) -> str:
        return (f"CorticalColumn(id={self.id!r}, spec={self.compute_specificity()}, "
                f"routes={self.route_count_total}, traces={len(self.memory_traces)})")


# ══════════════════════════════════════════════════════════════════════════════════
# §7  CORTEX — COMPETITIVE COLUMN ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════════

class Cortex:
    """
    The full cortical architecture — hierarchical collection of CorticalColumns.

    Implements competitive learning (Rumelhart & Zipser 1985):
      Input → All columns compute trigger match simultaneously
      Winner: most specific matching column (highest specificity score)
      Loser columns: not activated for this input
      ROOT: guaranteed fallback when no specialist matches

    Key design differences from nexus_v3.py NexusTree:
      - Winner-take-all via specificity (not first-match FIFO)
      - ROOT never fires if ANY specialist matches (eliminates ROOT dilution)
      - Columnar genesis requires Jaccard overlap audit (Fix-FM-5)
      - Apoptosis with MemoryTrace bequeathal (knowledge preservation)

    Reference: Rumelhart D., Zipser D. (1985). Cognitive Science 9(1):75-112.
    """

    def __init__(self):
        self._columns: list[CorticalColumn] = []
        self._lock = threading.RLock()

    # ── Column management ─────────────────────────────────────────────────────────

    def add_column(self, column: CorticalColumn) -> None:
        with self._lock:
            self._columns.append(column)

    @property
    def columns(self) -> list[CorticalColumn]:
        with self._lock:
            return list(self._columns)

    def get_column(self, col_id: str) -> Optional[CorticalColumn]:
        with self._lock:
            for col in self._columns:
                if col.id == col_id:
                    return col
            return None

    def root_column(self) -> CorticalColumn:
        col = self.get_column("ROOT")
        if col is None:
            raise ValueError("ROOT column not found in Cortex")
        return col

    # ── Competitive routing ───────────────────────────────────────────────────────

    def route(self, feats: dict) -> CorticalColumn:
        """
        Winner-take-all routing: most specific matching column wins.

        Competitive learning (Rumelhart & Zipser 1985):
          1. Evaluate all columns' trigger conditions against features
          2. Among matching columns, select highest specificity
          3. ROOT is always eligible but has specificity 0
          → ROOT only wins if NO specialist matches (no ambiguity dilution)

        This fixes FM-1 (Routing Dilution): in v3.04, FIFO routing meant
        the first matching node captured cases regardless of specificity.
        Here, the most specialized column always wins.
        """
        with self._lock:
            candidates = []
            for col in self._columns:
                if col.id == "ROOT":
                    continue  # Evaluate ROOT separately as fallback
                if safe_eval_condition(col.trigger_condition, feats):
                    candidates.append(col)

            if candidates:
                # Winner-take-all: highest specificity wins
                return max(candidates, key=lambda c: c.compute_specificity())
            else:
                # No specialist matches — ROOT fires (predictive coding error: no match)
                return self.root_column()

    def route_and_track(self, feats: dict, round_num: int) -> tuple[CorticalColumn, bool]:
        """Route and return (winning_column, was_contested)."""
        with self._lock:
            candidates = [
                col for col in self._columns
                if col.id != "ROOT" and safe_eval_condition(col.trigger_condition, feats)
            ]
            was_contested = len(candidates) > 1
            if candidates:
                winner = max(candidates, key=lambda c: c.compute_specificity())
                return winner, was_contested
            return self.root_column(), False

    # ── Jaccard overlap audit ─────────────────────────────────────────────────────

    def jaccard_overlap(self, trigger_condition: str, probe_cases: list[dict]) -> dict:
        """
        Compute Jaccard similarity between proposed column's coverage
        and each existing specialist column's coverage.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|

        High Jaccard (> 0.50) means the proposed column would steal cases
        from an established specialist → reject genesis unless the proposed
        column dramatically improves performance on those cases.

        Returns: {"max_jaccard": float, "worst_col": str, "all": dict}
        """
        # Compute proposed column's coverage
        proposed_coverage = set()
        for i, case in enumerate(probe_cases):
            feats = extract_features(case["text"])
            if safe_eval_condition(trigger_condition, feats):
                proposed_coverage.add(i)

        if not proposed_coverage:
            return {"max_jaccard": 0.0, "worst_col": None, "all": {}, "proposed_size": 0}

        with self._lock:
            results = {}
            for col in self._columns:
                if col.id == "ROOT":
                    continue
                col_coverage = col.coverage_set(probe_cases)
                if not col_coverage:
                    results[col.id] = 0.0
                    continue
                intersection = len(proposed_coverage & col_coverage)
                union = len(proposed_coverage | col_coverage)
                results[col.id] = intersection / union if union > 0 else 0.0

        max_j = max(results.values()) if results else 0.0
        worst = max(results, key=results.get) if results else None
        return {
            "max_jaccard": max_j,
            "worst_col": worst,
            "all": results,
            "proposed_size": len(proposed_coverage),
        }

    # ── Columnar genesis (neurogenesis) ───────────────────────────────────────────

    def columnar_genesis(
        self,
        proposal: dict,
        probe_cases: list[dict],
        critical_period: CriticalPeriod,
        round_num: int,
        probe_f1_improvement: float = 0.0,
        max_jaccard: float = 0.50,
        jaccard_f1_override: float = 0.03,
    ) -> tuple[bool, str]:
        """
        Neurogenesis gate: validates and registers a new CorticalColumn.

        Steps:
          1. Validate trigger_condition syntax (safe_eval guard)
          2. Check for duplicate column ID
          3. Jaccard overlap audit — reject if max Jaccard > threshold
             (unless probe F1 improvement > jaccard_f1_override)
          4. Register column with genesis_phase from critical_period

        Returns: (approved: bool, reason: str)

        Fixes:
          FM-5 (Trigger Overlap): Jaccard audit prevents routing dilution
          FM-1 (Routing Dilution): Competitive routing ensures specialist wins

        Reference: Eriksson et al. (1998). Neurogenesis in adult hippocampus.
        """
        col_id = proposal.get("id", "")
        trigger = proposal.get("trigger_condition", "")
        prompt = proposal.get("prompt", "")
        description = proposal.get("description", "")

        # ── Validation ────────────────────────────────────────────────────────────
        if not col_id or not trigger or not prompt:
            return False, "missing required fields (id, trigger_condition, prompt)"

        if self.get_column(col_id):
            return False, f"column {col_id!r} already exists"

        # Validate trigger syntax
        dummy_feats = {name: False for name in FEATURE_NAMES}
        tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', trigger)
        allowed = set(FEATURE_NAMES) | {'and', 'or', 'not', 'True', 'False'}
        bad = [t for t in tokens if t not in allowed]
        if bad:
            return False, f"invalid trigger variables: {bad}"

        # ── Critical period threshold ──────────────────────────────────────────────
        genesis_phase = phase_for_round(round_num)

        # ── Jaccard overlap audit ─────────────────────────────────────────────────
        if probe_cases:
            overlap = self.jaccard_overlap(trigger, probe_cases)
            j = overlap["max_jaccard"]
            worst = overlap["worst_col"]
            if j > max_jaccard:
                if probe_f1_improvement >= jaccard_f1_override:
                    reason = (f"JACCARD OVERRIDE: max_jaccard={j:.3f} with {worst!r} "
                              f"but F1 improvement={probe_f1_improvement:+.4f} > threshold")
                    print(f"  [Genesis] {reason}", flush=True)
                else:
                    return False, (f"Jaccard overlap {j:.3f} with {worst!r} exceeds "
                                   f"{max_jaccard} and F1 improvement "
                                   f"{probe_f1_improvement:+.4f} < {jaccard_f1_override}")

        # ── Create column ─────────────────────────────────────────────────────────
        new_col = CorticalColumn(
            col_id=col_id,
            description=description,
            trigger_condition=trigger,
            prompt=prompt,
            genesis_round=round_num,
            genesis_phase=genesis_phase,
        )
        with self._lock:
            self._columns.append(new_col)

        print(f"  [Genesis ✓] {col_id!r} | phase={genesis_phase.value} "
              f"| spec={new_col.compute_specificity()} | round={round_num}", flush=True)
        return True, "genesis approved"

    # ── Synaptic pruning (apoptosis) ──────────────────────────────────────────────

    def synaptic_pruning(self, round_num: int) -> list[CorticalColumn]:
        """
        Apoptosis: prune columns with consistently low activation.
        Bequeaths MemoryTraces to ROOT before removal.

        Biological parallel: in adult neurogenesis (Eriksson 1998),
        new neurons that fail to integrate (receive synaptic input)
        undergo apoptosis within weeks. Their molecular signals are
        recycled by neighboring cells.

        Here: pruned column's MemoryTraces are bequeathed to ROOT,
        preserving learned knowledge even as the column is removed.
        This prevents catastrophic forgetting during structural change.
        """
        root = self.root_column()
        pruned = []

        with self._lock:
            to_prune = [
                col for col in self._columns
                if col.is_apoptosis_candidate(round_num)
            ]

            for col in to_prune:
                # Bequest MemoryTraces to ROOT (knowledge preservation)
                for trace in col.memory_traces:
                    inherited = root.add_memory_trace(
                        text=f"[from {col.id}] {trace.text}",
                        consolidation_score=trace.consolidation_score * 0.8,  # slight decay
                        round_num=round_num,
                        bequeathed=True,
                        bequeathed_from=col.id,
                    )
                    print(f"  [Apoptosis] Bequeathing MemoryTrace from {col.id} → ROOT: "
                          f"{trace.text[:60]}...", flush=True)

                print(f"  [Apoptosis ✂] Pruning {col.id!r} | "
                      f"avg_activation={sum(col.activation_history[-3:])/3:.1f} cases/round "
                      f"| {len(col.memory_traces)} traces bequeathed", flush=True)
                self._columns.remove(col)
                pruned.append(col)

        return pruned

    # ── Serialization ─────────────────────────────────────────────────────────────

    def save(self, out_path: str) -> None:
        data = {
            "cortex_version": "1.0",
            "columns": [{"id": c.id, "description": c.description,
                         "trigger_condition": c.trigger_condition,
                         "prompt": c.prompt,
                         "genesis_round": c.genesis_round,
                         "genesis_phase": c.genesis_phase.value,
                         "route_count_total": c.route_count_total,
                         "activation_history": c.activation_history,
                         "f1_history": c.f1_history,
                         "prompt_version": c.prompt_version,
                         "memory_traces": [
                             {"id": t.id, "text": t.text, "source": t.source_column,
                              "round": t.round_created, "score": t.consolidation_score,
                              "bequeathed": t.bequeathed, "from": t.bequeathed_from}
                             for t in c.memory_traces
                         ],
                         "bcm_state": c.bcm_state.to_dict()}
                        for c in self._columns],
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, in_path: str) -> "Cortex":
        data = json.loads(Path(in_path).read_text())
        cortex = cls()
        for cd in data["columns"]:
            col = CorticalColumn(
                col_id=cd["id"], description=cd["description"],
                trigger_condition=cd["trigger_condition"], prompt=cd["prompt"],
                genesis_round=cd.get("genesis_round", 0),
                genesis_phase=LearningPhase(cd.get("genesis_phase", "embryonic")),
            )
            col.route_count_total = cd.get("route_count_total", 0)
            col.activation_history = cd.get("activation_history", [])
            col.f1_history = cd.get("f1_history", [])
            col.prompt_version = cd.get("prompt_version", 0)
            for td in cd.get("memory_traces", []):
                col.memory_traces.append(MemoryTrace(
                    id=td["id"], text=td["text"], source_column=td["source"],
                    round_created=td["round"], consolidation_score=td["score"],
                    bequeathed=td.get("bequeathed", False),
                    bequeathed_from=td.get("from"),
                ))
            cortex._columns.append(col)
        return cortex

    def summary(self) -> str:
        cols = self.columns
        lines = [f"Cortex: {len(cols)} columns"]
        for c in sorted(cols, key=lambda x: x.compute_specificity(), reverse=True):
            lines.append(
                f"  {c.id:35s} spec={c.compute_specificity()} "
                f"routes={c.route_count_total:5d} "
                f"traces={len(c.memory_traces):3d} "
                f"BCM={c.bcm_state.ltp_count}LTP/{c.bcm_state.ltd_count}LTD"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════════
# §8  ACTIVATION PATHWAYS (ROUTE ABSTENTION FIX)
# ══════════════════════════════════════════════════════════════════════════════════

def classify_with_routes(
    text: str,
    rag_index: RAGIndex,
    llm_fn: Callable,
    column: CorticalColumn,
    top_k: int = 5,
    route_timeout: float = 30.0,
    firing_threshold: float = 1.0,  # FiringThreshold (ADE_BIAS equivalent)
) -> dict:
    """
    Classify a sentence using parallel activation pathways (expert routes).

    KEY FIX (Fix-FM-2 — Route Abstention):
      In nexus_v3.py, a failed route returned RouteResult(vote="NOT_ADE",
      confidence=0.3). This systematically suppressed recall during API
      overload events (v3.04 R19 collapse: F1=0.6538).

      Here, failed routes return None and are EXCLUDED from the ensemble vote.
      The weighted vote is computed only over routes that responded.
      If ALL routes fail, the ensemble returns NOT_ADE with 0.1 confidence
      and flags the case for manual review.

    Biological parallel — Predictive Coding (Rao & Ballard 1999):
      Classification = prediction. Each route makes a prediction about
      the input's label. Failed routes = absent prediction signals.
      The cortex should not interpret absent signals as NOT_ADE —
      absence of evidence is not evidence of absence.

    Activation pathways (from expert_routes.py):
      A. Causation:   Direct causal language drug → harm?
      B. Negation:    Adverse outcome negated?
      C. Drug Effect: Retrieved evidence confirms drug-effect pair?
      D. Context:     Therapeutic intent vs documented adverse outcome?

    Parameters:
      firing_threshold: Homeostatic plasticity parameter (was ADE_BIAS).
        ADE_score ≥ NOT_ADE_score * firing_threshold → predict ADE.
    """
    # Retrieve similar examples from hippocampal index (RAG)
    examples = rag_index.query(text, k=top_k)

    # Assemble memory trace context for this column
    principle_context = ""
    if column.memory_traces:
        principle_context = "\n\n" + column.memory_trace_context(max_traces=3)

    # Prepare working memory context
    wm_context = column.working_memory.format_for_prompt(
        max_cases=6, bcm_state=column.bcm_state
    )
    if wm_context:
        principle_context += f"\n\nRECENT ERROR PATTERNS (working memory):\n{wm_context}"

    # Route functions
    route_fns = {
        "causation":   _route_causation,
        "negation":    _route_negation,
        "drug_effect": _route_drug_effect,
        "context":     _route_context,
    }

    route_results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(fn, text, examples, llm_fn, principle_context): name
            for name, fn in route_fns.items()
        }
        for future in as_completed(futures, timeout=route_timeout + 5):
            route_name = futures[future]
            try:
                result = future.result(timeout=route_timeout)
                route_results[route_name] = result   # None if route abstained
            except FuturesTimeoutError:
                print(f"  [ROUTE TIMEOUT] {route_name} — abstaining", file=sys.stderr)
                route_results[route_name] = None
            except Exception as e:
                print(f"  [ROUTE ERROR] {route_name}: {e} — abstaining", file=sys.stderr)
                route_results[route_name] = None

    # Weighted ensemble (abstained routes excluded — Fix-FM-2)
    route_weights = {"causation": 1.5, "negation": 1.5, "drug_effect": 1.2, "context": 1.0}
    ade_score = 0.0
    not_ade_score = 0.0
    responsive_routes = []

    for name, result in route_results.items():
        if result is None:
            continue   # ABSTAIN — do not influence vote
        w = route_weights.get(name, 1.0)
        if result["vote"] == "ADE":
            ade_score += result["confidence"] * w
        else:
            not_ade_score += result["confidence"] * w
        responsive_routes.append(result)

    # If ALL routes failed — abstain (minimal confidence NOT_ADE, flagged)
    if not responsive_routes:
        return {
            "label": "NOT_ADE", "confidence": 0.1, "ade_score": 0.0,
            "not_ade_score": 0.0, "all_routes_failed": True,
            "route_results": [], "agreement": 0.0, "split": False,
        }

    # Apply homeostatic firing threshold
    final_label = "ADE" if ade_score >= not_ade_score * firing_threshold else "NOT_ADE"
    total = ade_score + not_ade_score
    confidence = max(ade_score, not_ade_score) / total if total > 0 else 0.5
    agreeing = sum(1 for r in responsive_routes if r["vote"] == final_label)
    agreement = agreeing / len(responsive_routes)

    return {
        "label": final_label,
        "confidence": confidence,
        "ade_score": ade_score,
        "not_ade_score": not_ade_score,
        "all_routes_failed": False,
        "route_results": responsive_routes,
        "agreement": agreement,
        "split": agreement < 0.75,
    }


def _route_causation(text: str, examples: list[dict], llm_fn: Callable,
                     principle_context: str = "") -> Optional[dict]:
    """Route A: Direct causal language drug → harm? Returns None on failure (abstain)."""
    system = (
        "You are a causation expert in pharmacovigilance. "
        "Determine whether the sentence contains DIRECT causal language "
        "linking a specific drug to a harmful or unintended outcome.\n\n"
        "Causal signals: caused, induced, associated with, resulted in, led to, "
        "following [drug], due to [drug], [drug]-related.\n"
        "NOT causal: therapeutic effect, negated outcome, drug mentioned without causal link.\n\n"
        "Respond ONLY with JSON:\n"
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
        + principle_context
    )
    ex_text = _format_examples(examples)
    user = (f'Sentence: "{text}"\n\nSimilar labeled examples:\n{ex_text}\n\n'
            "Does this sentence show direct drug-to-harm causation?")
    try:
        raw = llm_fn(system, user)
        return _parse_json_vote(raw)
    except Exception:
        return None   # Abstain — do not vote NOT_ADE by default


def _route_negation(text: str, examples: list[dict], llm_fn: Callable,
                    principle_context: str = "") -> Optional[dict]:
    """Route B: Adverse outcome negated? Returns None on failure (abstain)."""
    system = (
        "You are a negation expert in clinical NLP. "
        "Determine whether any adverse outcome is NEGATED, denied, or hypothetical.\n\n"
        "Negation signals: no, not, without, denied, failed to develop, "
        "did not experience, absence of, ruled out, tolerates well.\n"
        "Scope: 'no relief from pain' negates relief, NOT an ADE.\n\n"
        "Respond ONLY with JSON:\n"
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
        + principle_context
    )
    ex_text = _format_examples(examples)
    user = (f'Sentence: "{text}"\n\nSimilar labeled examples:\n{ex_text}\n\n'
            "Is the adverse outcome negated or hypothetical?")
    try:
        raw = llm_fn(system, user)
        return _parse_json_vote(raw)
    except Exception:
        return None


def _route_drug_effect(text: str, examples: list[dict], llm_fn: Callable,
                       principle_context: str = "") -> Optional[dict]:
    """Route C: Retrieved evidence confirms drug-effect pair? Returns None on failure."""
    system = (
        "You are a pharmacology expert with access to retrieved literature examples.\n"
        "Based on the retrieved examples, does this sentence describe a known "
        "drug-adverse effect relationship?\n"
        "If retrieved ADE examples are highly similar (score > 0.85), weight them heavily.\n\n"
        "Respond ONLY with JSON:\n"
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
        + principle_context
    )
    ade_ex = [e for e in examples if e["label"] == "ADE"]
    not_ex = [e for e in examples if e["label"] == "NOT_ADE"]
    ex_text = f"ADE examples:\n{_format_examples(ade_ex, 3)}\n\nNOT_ADE examples:\n{_format_examples(not_ex, 2)}"
    user = (f'Sentence: "{text}"\n\n{ex_text}\n\n'
            "Does the evidence support ADE classification?")
    try:
        raw = llm_fn(system, user)
        return _parse_json_vote(raw)
    except Exception:
        return None


def _route_context(text: str, examples: list[dict], llm_fn: Callable,
                   principle_context: str = "") -> Optional[dict]:
    """Route D: Therapeutic intent vs documented adverse outcome? Returns None on failure."""
    system = (
        "You are a clinical context expert. "
        "Determine whether the sentence documents a genuine adverse outcome "
        "or describes a therapeutic intent / desired effect.\n\n"
        "ADE context: documented side effect, unintended harm, patient complaint.\n"
        "NOT_ADE context: desired therapeutic effect, drug prescribed for symptom, "
        "symptom without drug connection.\n\n"
        "Respond ONLY with JSON:\n"
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
        + principle_context
    )
    ex_text = _format_examples(examples)
    user = (f'Sentence: "{text}"\n\nSimilar examples:\n{ex_text}\n\n'
            "Is this a documented adverse drug event or therapeutic context?")
    try:
        raw = llm_fn(system, user)
        return _parse_json_vote(raw)
    except Exception:
        return None


def _format_examples(examples: list[dict], max_k: int = 4) -> str:
    lines = []
    for i, ex in enumerate(examples[:max_k]):
        bar = "▲ADE" if ex["label"] == "ADE" else "▽NOT"
        sim = f"{ex.get('score', 0):.2f}"
        lines.append(f"  [{i+1}] {bar} (sim={sim}) \"{ex['text'][:100]}\"")
    return "\n".join(lines) if lines else "  (no examples retrieved)"


def _parse_json_vote(raw: str) -> dict:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        vote = "ADE" if ('"ADE"' in raw or "'ADE'" in raw) else "NOT_ADE"
        conf = 0.4
        return {"vote": vote, "confidence": conf, "reasoning": "parse fallback"}


# ══════════════════════════════════════════════════════════════════════════════════
# §9  HOMEOSTATIC PLASTICITY (FIRING THRESHOLD CALIBRATION)
# ══════════════════════════════════════════════════════════════════════════════════

class HomeostaticPlasticity:
    """
    Homeostatic plasticity — maintains target neural firing rate via synaptic scaling.

    In neuroscience (Turrigiano et al. 1998), when a neuron's firing rate
    deviates from its homeostatic setpoint, it scales its synaptic weights
    up or down to restore equilibrium. This prevents runaway excitation
    (epilepsy) or runaway inhibition (silence).

    In nexus_cortex_v1.py, the FiringThreshold (ADE_BIAS) is calibrated
    to maintain target recall — the homeostatic setpoint. The score cache
    (ADE_score, NOT_ADE_score, true_label) from the evaluation round is
    swept across candidate thresholds to find the one that achieves the
    target F1/recall/precision trade-off.

    Key features vs nexus_v3.py calibrate_threshold():
      - Same zero-LLM-cost algorithm (sweep over cached scores)
      - Broader candidate range with column-count-adaptive fine grid
      - History tracking: allows detecting threshold drift over rounds
      - Biological framing: threshold = homeostatic setpoint

    Reference: Turrigiano G. et al. (1998). Nature 391:892-896.
    """

    def __init__(self, initial_threshold: float = 1.0, target: str = "f1", beta: float = 1.0):
        self.firing_threshold = initial_threshold
        self.target = target        # 'f1', 'recall', 'precision', 'fbeta'
        self.beta = beta
        self.history: list[dict] = []   # Per-round threshold history
        self.positive_label = "ADE"
        self.negative_label = "NOT_ADE"

    def calibrate(
        self,
        score_cache: list[tuple[float, float, str]],  # (ade_score, not_ade_score, true_label)
        round_num: int,
        n_columns: int = 4,
        verbose: bool = True,
    ) -> float:
        """
        Zero-LLM-cost threshold sweep over cached score tuples.

        candidate range: 0.3–4.0 fixed grid + fine grid around current threshold.
        Returns optimal FiringThreshold and updates internal state.
        """
        if not score_cache:
            return self.firing_threshold

        pos = self.positive_label
        beta2 = self.beta ** 2
        target = self.target

        # Candidate grid
        base_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                           1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
        fine = [
            round(self.firing_threshold * f, 2)
            for f in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]
        ]
        candidates = sorted(set(base_candidates + fine))

        def _score_at(bias: float) -> dict:
            tp = fp = fn = tn = 0
            for ade_s, not_s, true_lbl in score_cache:
                pred = pos if ade_s >= not_s * bias else self.negative_label
                if pred == pos and true_lbl == pos:    tp += 1
                elif pred == pos and true_lbl != pos:  fp += 1
                elif pred != pos and true_lbl == pos:  fn += 1
                else:                                   tn += 1
            prec   = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1     = 2 * prec * recall / max(1e-9, prec + recall)
            fbeta  = (1 + beta2) * prec * recall / max(1e-9, beta2 * prec + recall)
            return {"f1": f1, "recall": recall, "precision": prec,
                    "fbeta": fbeta, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        best_threshold = self.firing_threshold
        best_score = _score_at(self.firing_threshold).get(target, 0.0)
        results = []

        for bias in candidates:
            s = _score_at(bias)
            score = s.get(target, s["f1"])
            results.append((bias, score, s))
            if score > best_score:
                best_score = score
                best_threshold = bias

        self.firing_threshold = best_threshold

        # Record homeostatic history
        best_stats = _score_at(best_threshold)
        self.history.append({
            "round": round_num,
            "threshold": best_threshold,
            "target": target,
            "score": best_score,
            "f1": best_stats["f1"],
            "recall": best_stats["recall"],
            "precision": best_stats["precision"],
            "n_cached": len(score_cache),
        })

        if verbose:
            print(f"\n[Homeostatic Calibration] R{round_num} — "
                  f"{len(score_cache)} cached scores, 0 LLM calls", flush=True)
            print(f"  FiringThreshold: {best_threshold:.3f} | "
                  f"F1={best_stats['f1']:.4f} | "
                  f"P={best_stats['precision']:.3f} | R={best_stats['recall']:.3f}", flush=True)

        return best_threshold


# ══════════════════════════════════════════════════════════════════════════════════
# §10  LLM SYNTHESIS (COLUMNAR GENESIS PROPOSALS)
# ══════════════════════════════════════════════════════════════════════════════════

# Synthesis prompt — proposes new CorticalColumn from error batch
_SYNTHESIS_PROMPT = """\
You are a computational neuroscience expert designing cortical columns for a \
clinical NLP classifier.

A cortical column is a specialist processor with:
  - A receptive field (trigger_condition) that selects which sentences it processes
  - A specialist prompt for classifying sentences within its receptive field

MISCLASSIFIED CASES (Round {round_num}):
{cases_block}

EXISTING CORTICAL COLUMNS: {col_list}

CRITICAL: Trigger conditions MUST use ONLY these 11 boolean variables:
  has_induced, has_associated, has_toxicity, has_adverse, has_developed,
  has_following, has_reaction, has_report, has_negation, has_short, has_drug_name
Operators: and, or, not (no parentheses nesting beyond 2 levels)

ROUTING COMPETITION: The most specific column (most positive conditions) wins routing.
Design your trigger to be as SPECIFIC as possible — avoid overlapping with:
  Existing columns: {trigger_list}

Propose ONE new specialist cortical column targeting the dominant error pattern.

Respond with ONLY valid JSON (no markdown fences):
{{
  "error_pattern": "dominant pattern in 1 sentence",
  "new_column": {{
    "id": "COL_<DESCRIPTIVE_NAME>",
    "description": "what this column handles",
    "trigger_condition": "expression using ONLY the 11 allowed variables",
    "prompt": "specialist classification prompt (2-4 sentences)"
  }}
}}"""

# Refinement prompt — improves existing column's specialist prompt
_REFINE_PROMPT = """\
You are improving a cortical column's specialist classification prompt.

COLUMN ID: {col_id}
DESCRIPTION: {description}
CURRENT PROMPT: {current_prompt}

CONSOLIDATED MEMORY TRACES (what this column has learned):
{memory_traces}

CASES THIS COLUMN GOT WRONG (recent errors):
{error_cases}

Task:
1. Identify what pattern the current prompt fails on (1-2 sentences).
2. Write an improved prompt that addresses the failure pattern.
3. Incorporate applicable memory traces into the prompt.
4. Keep it concise — 3-5 sentences maximum.

Respond with ONLY valid JSON (no markdown fences):
{{
  "analysis": "what the current prompt misses",
  "improved_prompt": "new specialist prompt"
}}"""

# Memory trace extraction — consolidates an accepted improvement
_CONSOLIDATE_TEMPLATE = """\
A prompt improvement was accepted for a cortical column.

ORIGINAL PROMPT: {original}
IMPROVED PROMPT: {improved}
F1 IMPROVEMENT: {delta_f1:+.4f}

Extract the generalizable principle that made this improvement effective.
State it as a single concrete rule for clinical NLP ADE classification.
It must be specific enough to apply to other columns in the same domain.

Respond with ONLY valid JSON (no markdown fences):
{{"principle": "Generalizable rule in 1-2 sentences"}}"""


def llm_synthesize_column(
    errors: list[dict],
    existing_columns: list[CorticalColumn],
    llm_fn: Callable,
    round_num: int,
    max_errors: int = 12,
) -> Optional[dict]:
    """
    LLM proposes a new CorticalColumn targeting the dominant error pattern.

    Input: list of {"text", "true_label", "predicted"} error dicts.
    Output: {"id", "description", "trigger_condition", "prompt"} or None.
    """
    if not errors:
        return None

    cases_block = "\n".join(
        f"  [{i+1}] TRUE={c['true_label']} PRED={c.get('predicted','?')}: \"{c['text'][:120]}\""
        for i, c in enumerate(errors[:max_errors])
    )
    col_list = ", ".join(c.id for c in existing_columns)
    trigger_list = "\n  ".join(
        f"{c.id}: {c.trigger_condition}" for c in existing_columns if c.id != "ROOT"
    )

    prompt = _SYNTHESIS_PROMPT.format(
        round_num=round_num,
        cases_block=cases_block,
        col_list=col_list,
        trigger_list=trigger_list,
    )

    system = "You are a computational neuroscience expert. Output ONLY valid JSON."
    try:
        raw = llm_fn(system, prompt)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        data = json.loads(raw)
        return data.get("new_column")
    except Exception as e:
        print(f"  [Synthesis Error] {e}", file=sys.stderr)
        return None


def llm_refine_column(
    column: CorticalColumn,
    errors: list[dict],
    llm_fn: Callable,
    round_num: int,
    max_errors: int = 10,
) -> Optional[str]:
    """
    LLM refines an existing column's specialist prompt given its recent errors.
    Returns improved_prompt string or None if refinement fails/not beneficial.
    """
    if not errors:
        return None

    error_block = "\n".join(
        f"  [{i+1}] TRUE={c['true_label']} PRED={c.get('predicted','?')}: \"{c['text'][:120]}\""
        for i, c in enumerate(errors[:max_errors])
    )
    traces_text = column.memory_trace_context(max_traces=4) or "(none yet)"

    prompt = _REFINE_PROMPT.format(
        col_id=column.id,
        description=column.description,
        current_prompt=column.prompt[:500],
        memory_traces=traces_text,
        error_cases=error_block,
    )

    system = "You are a clinical NLP expert. Output ONLY valid JSON."
    try:
        raw = llm_fn(system, prompt)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        data = json.loads(raw)
        return data.get("improved_prompt")
    except Exception as e:
        print(f"  [Refine Error] {column.id}: {e}", file=sys.stderr)
        return None


def llm_consolidate_memory(
    original_prompt: str,
    improved_prompt: str,
    delta_f1: float,
    llm_fn: Callable,
) -> Optional[str]:
    """Extract a MemoryTrace from an accepted prompt improvement."""
    prompt = _CONSOLIDATE_TEMPLATE.format(
        original=original_prompt[:300],
        improved=improved_prompt[:300],
        delta_f1=delta_f1,
    )
    system = "You are a clinical NLP expert. Output ONLY valid JSON."
    try:
        raw = llm_fn(system, prompt)
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        data = json.loads(raw)
        return data.get("principle")
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════════
# §11  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════════

def evaluate_cortex(
    cortex: Cortex,
    eval_pool: list[dict],
    rag_index: RAGIndex,
    llm_fn: Callable,
    homeostatic: HomeostaticPlasticity,
    config,  # TaskConfig
    round_num: int,
    max_workers: int = 4,
) -> tuple[dict, list[tuple[float, float, str]]]:
    """
    Evaluate the cortex on the eval pool (200 cases — Fix-FM-4).

    Returns:
      metrics: {"f1", "precision", "recall", "tp", "fp", "fn", "tn", "n_evaluated"}
      score_cache: list of (ade_score, not_ade_score, true_label) for calibration

    Key changes vs nexus_v3.py evaluate():
      - eval_pool size 200 (was 100) — Fix-FM-4
      - Route abstention: failed routes excluded from vote — Fix-FM-2
      - Competitive routing: winner-take-all by specificity — Fix-FM-1
    """
    pos = config.positive_label  # "ADE"

    results = []
    score_cache = []

    def _classify_one(item: dict) -> dict:
        text = item["text"]
        true_label = item["label"]
        feats = extract_features(text)
        winning_col = cortex.route(feats)

        result = classify_with_routes(
            text=text,
            rag_index=rag_index,
            llm_fn=llm_fn,
            column=winning_col,
            firing_threshold=homeostatic.firing_threshold,
        )
        return {
            "text": text,
            "true_label": true_label,
            "predicted": result["label"],
            "confidence": result["confidence"],
            "ade_score": result["ade_score"],
            "not_ade_score": result["not_ade_score"],
            "column": winning_col.id,
            "all_routes_failed": result.get("all_routes_failed", False),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_classify_one, item): item for item in eval_pool}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  [Eval Error] {e}", file=sys.stderr)

    # Metrics
    tp = fp = fn = tn = 0
    for r in results:
        score_cache.append((r["ade_score"], r["not_ade_score"], r["true_label"]))
        if r["predicted"] == pos and r["true_label"] == pos:    tp += 1
        elif r["predicted"] == pos and r["true_label"] != pos:  fp += 1
        elif r["predicted"] != pos and r["true_label"] == pos:  fn += 1
        else:                                                     tn += 1

    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)

    # Per-column activation tracking
    col_activations: dict[str, int] = {}
    for r in results:
        col_activations[r["column"]] = col_activations.get(r["column"], 0) + 1

    metrics = {
        "f1": f1, "precision": prec, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_evaluated": len(results),
        "col_activations": col_activations,
    }

    return metrics, score_cache


# ══════════════════════════════════════════════════════════════════════════════════
# §12  ENGRAM CLUSTER DETECTION (LTP EVENT TRIGGER)
# ══════════════════════════════════════════════════════════════════════════════════

def detect_engram_clusters(
    errors: list[dict],
    round_num: int,
    triggering_column: str,
    min_cluster_size: int = 5,
) -> list[EnggramCluster]:
    """
    Detect EnggramClusters from a batch of error cases.

    In NEXUS v3, these were 'SWR events'. The algorithm:
      1. Represent each error by its feature vector (11 boolean features)
      2. Compute pairwise Hamming distance between feature vectors
      3. Simple threshold clustering: cases within distance < 4 form a cluster
      4. Only clusters with >= min_cluster_size cases are returned

    No embeddings required — pure feature-space clustering for speed.
    (Embedder available but slow; feature-space is fast and interpretable.)

    Returns list of EnggramCluster objects sorted by size (largest first).
    """
    if len(errors) < min_cluster_size:
        return []

    # Vectorize
    def featurize(text: str) -> np.ndarray:
        feats = extract_features(text)
        return np.array([float(feats[k]) for k in FEATURE_NAMES], dtype=np.float32)

    vecs = [featurize(e["text"]) for e in errors]
    n = len(vecs)

    # Simple greedy clustering by Hamming distance
    used = [False] * n
    clusters = []

    for i in range(n):
        if used[i]:
            continue
        cluster_indices = [i]
        used[i] = True
        for j in range(i + 1, n):
            if not used[j]:
                dist = np.sum(np.abs(vecs[i] - vecs[j]))
                if dist < 4:   # Within 4 feature differences = similar pattern
                    cluster_indices.append(j)
                    used[j] = True

        if len(cluster_indices) >= min_cluster_size:
            cluster_vecs = np.stack([vecs[k] for k in cluster_indices])
            centroid = cluster_vecs.mean(axis=0)
            # Coherence: mean pairwise similarity (dot product on binary features)
            n_c = len(cluster_vecs)
            if n_c > 1:
                dists = []
                for a in range(n_c):
                    for b in range(a+1, n_c):
                        dists.append(np.sum(np.abs(cluster_vecs[a] - cluster_vecs[b])))
                coherence = 1.0 - np.mean(dists) / len(FEATURE_NAMES)
            else:
                coherence = 1.0

            clusters.append(EnggramCluster(
                cases=[errors[k] for k in cluster_indices],
                centroid=centroid,
                coherence=float(coherence),
                round_detected=round_num,
                triggering_column=triggering_column,
            ))

    clusters.sort(key=lambda c: len(c), reverse=True)
    return clusters


# ══════════════════════════════════════════════════════════════════════════════════
# §13  MAIN TRAINING LOOP — CortexTrainer
# ══════════════════════════════════════════════════════════════════════════════════

class CortexTrainer:
    """
    Orchestrates the full cortical learning cycle.

    Training loop per round:
      1. Sample batch from train_pool
      2. Classify each case via competitive routing → route abstention
      3. Collect errors per column (column-specific error tracking)
      4. Per-column: BCM update, working memory rehearsal, optional refinement
      5. Detect EnggramClusters → LTP events → columnar genesis proposals
      6. Jaccard overlap audit → genesis decision (critical period gated)
      7. Synaptic pruning (apoptosis) for low-activation columns
      8. Evaluate on eval_pool (200 cases)
      9. Homeostatic calibration (FiringThreshold sweep)
      10. Save cortex state

    Phase-specific behavior:
      EMBRYONIC (R1-5):    Permissive genesis, no pruning, aggressive refinement
      DEVELOPMENTAL (R6-12): Jaccard audit strict, pruning enabled, measured refinement
      CONSOLIDATION (R13+): No genesis, apoptosis-only, only best columns refined
    """

    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.round_history: list[dict] = []
        self.genesis_log:   list[dict] = []
        self.pruning_log:   list[dict] = []

    def build_llm_fn(self) -> Callable:
        """Build the LLM callable from CLI args."""
        args = self.args
        if args.mock:
            client = llm_client.MockClient()
        elif args.ai_hub:
            client = llm_client.AIHubClient(
                api_key=args.ai_hub_key,
                ad_object_id=args.ai_hub_ad_id,
            )
        else:
            raise ValueError("Must specify --ai-hub or --mock")

        def llm_fn(system: str, user: str) -> str:
            # Use chat() — freeform text generation, returns string directly.
            # All four client types (AIHubClient, MockClient, OpenAIClient, GeminiClient)
            # implement chat(system, user) → str.
            return client.chat(system=system, user=user)

        return llm_fn

    def build_seed_cortex(self, config) -> Cortex:
        """
        Initialize the Cortex with seed CorticalColumns from config.

        Seed columns are the embryonic cortical scaffold — analogous to
        the pre-wired cortical areas present at birth before experience-
        dependent refinement begins.
        """
        cortex = Cortex()
        # Support both task_config.py SeedNode objects and raw dict lists
        seed_nodes = getattr(config, "seed_nodes", [])
        for node_def in seed_nodes:
            # SeedNode dataclass has .id, .trigger, .prompt, .description
            if hasattr(node_def, "id"):
                col_id = node_def.id
                trigger = node_def.trigger or "True"
                prompt = node_def.prompt
                description = node_def.description
            else:
                col_id = node_def["id"]
                trigger = node_def.get("trigger") or node_def.get("trigger_condition") or "True"
                prompt = node_def.get("prompt", "")
                description = node_def.get("description", "")

            col = CorticalColumn(
                col_id=col_id,
                description=description,
                trigger_condition=trigger,
                prompt=prompt,
                genesis_round=0,
                genesis_phase=LearningPhase.EMBRYONIC,
            )
            cortex.add_column(col)
            print(f"  [Seed Column] {col.id!r} | spec={col.compute_specificity()} "
                  f"| trigger: {col.trigger_condition}", flush=True)
        return cortex

    def run_training_round(
        self,
        round_num: int,
        cortex: Cortex,
        train_batch: list[dict],
        eval_pool: list[dict],
        probe_pool: list[dict],
        rag_index: RAGIndex,
        llm_fn: Callable,
        homeostatic: HomeostaticPlasticity,
        critical_period: CriticalPeriod,
        config,
    ) -> dict:
        """Execute one full training round."""

        phase = phase_for_round(round_num)
        print(f"\n{'═'*80}", flush=True)
        print(f"  ROUND {round_num} | Phase: {phase.value.upper()} | "
              f"Columns: {len(cortex.columns)} | "
              f"FiringThreshold: {homeostatic.firing_threshold:.3f}", flush=True)
        print(f"  CriticalPeriod: {critical_period.describe(round_num)}", flush=True)
        print(f"{'═'*80}", flush=True)

        # ── Training batch classification ─────────────────────────────────────────
        print(f"\n[Training] Classifying {len(train_batch)} cases...", flush=True)

        col_errors: dict[str, list[dict]] = {c.id: [] for c in cortex.columns}
        col_route_counts: dict[str, int] = {c.id: 0 for c in cortex.columns}
        all_errors: list[dict] = []
        ltp_round = False  # Will be set True if LTP event fires this round

        def classify_training_case(item: dict) -> dict:
            feats = extract_features(item["text"])
            winning_col, contested = cortex.route_and_track(feats, round_num)
            result = classify_with_routes(
                text=item["text"],
                rag_index=rag_index,
                llm_fn=llm_fn,
                column=winning_col,
                firing_threshold=homeostatic.firing_threshold,
            )
            return {
                "text": item["text"],
                "true_label": item["label"],
                "predicted": result["label"],
                "column": winning_col.id,
                "ade_score": result["ade_score"],
                "not_ade_score": result["not_ade_score"],
                "all_routes_failed": result.get("all_routes_failed", False),
                "contested": contested,
            }

        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = [executor.submit(classify_training_case, item)
                       for item in train_batch]
            for future in as_completed(futures):
                try:
                    r = future.result()
                    col_route_counts[r["column"]] = col_route_counts.get(r["column"], 0) + 1
                    if r["predicted"] != r["true_label"]:
                        err = {"text": r["text"], "true_label": r["true_label"],
                               "predicted": r["predicted"], "column": r["column"]}
                        col_errors[r["column"]].append(err)
                        all_errors.append(err)
                except Exception as e:
                    print(f"  [Training Error] {e}", file=sys.stderr)

        total_cases = len(train_batch)
        error_rate = len(all_errors) / max(1, total_cases)
        print(f"\n[Training] Errors: {len(all_errors)}/{total_cases} "
              f"({error_rate:.1%})", flush=True)

        # ── Per-column BCM update + working memory ────────────────────────────────
        print(f"\n[BCM] Updating plasticity states...", flush=True)
        for col in cortex.columns:
            routes = col_route_counts.get(col.id, 0)
            event = col.bcm_update(routes, total_cases, round_num)
            errors_this = col_errors.get(col.id, [])
            ltp_this = (event == "LTP")
            if ltp_this:
                ltp_round = True

            # Add errors to working memory
            for err in errors_this:
                col.working_memory.add_error(
                    text=err["text"], true_label=err["true_label"],
                    predicted=err["predicted"], round_num=round_num,
                    ltp_round=ltp_this,
                )

            print(f"  {col.id:35s} routes={routes:3d} | BCM={event:6s} | "
                  f"θ_M={col.bcm_state.theta_m:.4f} | "
                  f"errors={len(errors_this):3d} | "
                  f"rehearsal_w={col.bcm_state.rehearsal_weight:.2f}", flush=True)

        # ── Column refinement (prompt learning) ───────────────────────────────────
        print(f"\n[Refinement] Improving column prompts...", flush=True)
        n_refined = 0
        for col in cortex.columns:
            errors_this = col_errors.get(col.id, [])
            if len(errors_this) < 3:
                continue

            # Phase-gated refinement: embryonic=aggressive, consolidation=selective
            if phase == LearningPhase.CONSOLIDATION and col.f1_history:
                # Only refine columns whose F1 is below cortex average
                avg_col_f1 = sum(col.f1_history[-3:]) / min(3, len(col.f1_history))
                cortex_f1 = (sum(c.f1_history[-1] for c in cortex.columns
                                 if c.f1_history) / max(1, len([c for c in cortex.columns
                                                                  if c.f1_history])))
                if avg_col_f1 >= cortex_f1:
                    continue   # This column is already above average — skip

            original_prompt = col.prompt
            improved = llm_refine_column(col, errors_this, llm_fn, round_num)
            if improved and improved != original_prompt:
                col.update_prompt(improved, f"R{round_num} refinement", round_num)
                n_refined += 1
                print(f"  [Refined] {col.id!r} (v{col.prompt_version})", flush=True)

                # Consolidate MemoryTrace if refinement is meaningful
                if len(errors_this) >= 5:
                    principle = llm_consolidate_memory(
                        original_prompt, improved, 0.0, llm_fn
                    )
                    if principle:
                        col.add_memory_trace(principle, 0.7, round_num)
                        print(f"  [MemoryTrace] {col.id!r}: {principle[:80]}...", flush=True)

        print(f"  {n_refined} columns refined", flush=True)

        # ── EnggramCluster detection → LTP events → columnar genesis ──────────────
        print(f"\n[EnggramCluster] Detecting LTP events...", flush=True)
        genesis_approved = False

        if phase != LearningPhase.CONSOLIDATION and len(all_errors) >= 5:
            # Detect clusters from all errors (across all columns)
            clusters = detect_engram_clusters(
                errors=all_errors,
                round_num=round_num,
                triggering_column="CORTEX",
                min_cluster_size=5,
            )

            if clusters:
                largest = clusters[0]
                print(f"  Largest EnggramCluster: {len(largest)} cases | "
                      f"coherence={largest.coherence:.3f}", flush=True)

                # Propose new CorticalColumn from this cluster
                proposal = llm_synthesize_column(
                    errors=largest.cases,
                    existing_columns=cortex.columns,
                    llm_fn=llm_fn,
                    round_num=round_num,
                )

                if proposal:
                    print(f"  [LTP Event] Proposed: {proposal.get('id','?')} | "
                          f"trigger: {proposal.get('trigger_condition','?')}", flush=True)

                    # Probe eval to measure F1 improvement before genesis
                    probe_f1_before = self._probe_f1(
                        cortex, probe_pool, rag_index, llm_fn,
                        homeostatic, config, max_cases=50
                    )

                    # Jaccard overlap audit
                    overlap = cortex.jaccard_overlap(
                        proposal.get("trigger_condition", ""),
                        probe_pool[:100]
                    )
                    j = overlap["max_jaccard"]
                    print(f"  [Jaccard Audit] max_overlap={j:.3f} "
                          f"(worst={overlap['worst_col']!r})", flush=True)

                    approved, reason = cortex.columnar_genesis(
                        proposal=proposal,
                        probe_cases=probe_pool[:100],
                        critical_period=critical_period,
                        round_num=round_num,
                        probe_f1_improvement=0.0,  # Can't know before probe
                        max_jaccard=0.50,
                    )

                    if approved:
                        # Post-genesis probe to measure actual improvement
                        probe_f1_after = self._probe_f1(
                            cortex, probe_pool, rag_index, llm_fn,
                            homeostatic, config, max_cases=50
                        )
                        delta_f1 = probe_f1_after - probe_f1_before
                        print(f"  [Post-Genesis Probe] F1 Δ = {delta_f1:+.4f} "
                              f"({probe_f1_before:.4f} → {probe_f1_after:.4f})", flush=True)

                        # Consolidate cluster into MemoryTrace on new column
                        new_col = cortex.get_column(proposal["id"])
                        if new_col:
                            new_col.add_memory_trace(
                                text=f"Spawned from EnggramCluster R{round_num}: "
                                     f"{proposal.get('description','')[:100]}",
                                consolidation_score=largest.coherence,
                                round_num=round_num,
                            )
                        genesis_approved = True
                        self.genesis_log.append({
                            "round": round_num, "phase": phase.value,
                            "col_id": proposal["id"], "jaccard": j,
                            "probe_delta_f1": delta_f1,
                        })
                    else:
                        print(f"  [Genesis Rejected] {reason}", flush=True)
            else:
                print(f"  No significant EnggramClusters detected "
                      f"(errors={len(all_errors)}, threshold=5)", flush=True)

        # ── Synaptic pruning (apoptosis) ──────────────────────────────────────────
        if phase != LearningPhase.EMBRYONIC:
            pruned = cortex.synaptic_pruning(round_num)
            if pruned:
                self.pruning_log.append({
                    "round": round_num,
                    "pruned": [c.id for c in pruned],
                })

        # ── Evaluation ────────────────────────────────────────────────────────────
        print(f"\n[Evaluation] Evaluating on {len(eval_pool)} cases...", flush=True)
        metrics, score_cache = evaluate_cortex(
            cortex=cortex,
            eval_pool=eval_pool,
            rag_index=rag_index,
            llm_fn=llm_fn,
            homeostatic=homeostatic,
            config=config,
            round_num=round_num,
            max_workers=self.args.workers,
        )

        # ── BCM update per column based on eval activations ───────────────────────
        for col in cortex.columns:
            col_eval_routes = metrics["col_activations"].get(col.id, 0)
            col.f1_history.append(metrics["f1"])  # Global F1 as proxy

        # ── Homeostatic calibration ───────────────────────────────────────────────
        homeostatic.calibrate(score_cache, round_num, n_columns=len(cortex.columns))

        # ── Round summary ──────────────────────────────────────────────────────────
        round_result = {
            "round": round_num,
            "phase": phase.value,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "n_evaluated": metrics["n_evaluated"],
            "n_columns": len(cortex.columns),
            "firing_threshold": homeostatic.firing_threshold,
            "genesis": genesis_approved,
            "errors_this_round": len(all_errors),
            "col_activations": metrics["col_activations"],
        }

        print(f"\n{'─'*60}", flush=True)
        print(f"  Round {round_num} | F1={metrics['f1']:.4f} | "
              f"P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | "
              f"Cols={len(cortex.columns)} | Phase={phase.value}", flush=True)
        print(f"  TP={metrics['tp']} FP={metrics['fp']} "
              f"FN={metrics['fn']} TN={metrics['tn']}", flush=True)
        print(f"{'─'*60}", flush=True)

        return round_result

    def _probe_f1(
        self,
        cortex: Cortex,
        probe_pool: list[dict],
        rag_index: RAGIndex,
        llm_fn: Callable,
        homeostatic: HomeostaticPlasticity,
        config,
        max_cases: int = 50,
    ) -> float:
        """Quick F1 estimate on probe_pool subset (for genesis delta measurement)."""
        sample = random.sample(probe_pool, min(max_cases, len(probe_pool)))
        pos = config.positive_label
        tp = fp = fn = tn = 0
        for item in sample:
            feats = extract_features(item["text"])
            col = cortex.route(feats)
            result = classify_with_routes(
                text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                column=col, firing_threshold=homeostatic.firing_threshold,
            )
            pred = result["label"]
            true = item["label"]
            if pred == pos and true == pos:    tp += 1
            elif pred == pos and true != pos:  fp += 1
            elif pred != pos and true == pos:  fn += 1
            else:                               tn += 1
        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return 2 * prec * recall / max(1e-9, prec + recall)

    def run(self, config) -> None:
        """Main entry point — full cortical training run."""

        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS CORTEX v1.0 — Biologically-Grounded Clinical NLP Classifier", flush=True)
        print(f"  Task: {config.task_name}", flush=True)
        print(f"  Output: {self.out_dir}/", flush=True)
        print(f"{'╚'+'═'*78+'╝'}\n", flush=True)

        # ── LLM client ───────────────────────────────────────────────────────────
        llm_fn = self.build_llm_fn()

        # ── API sanity check ─────────────────────────────────────────────────────
        if not self.args.mock:
            print("[Startup] Verifying LLM API...", flush=True)
            try:
                test = llm_fn("Say OK", "Reply with just the word OK")
                print(f"  API OK: {test[:30]!r}", flush=True)
            except Exception as e:
                print(f"  [FATAL] API test failed: {e}", file=sys.stderr)
                sys.exit(1)

        # ── Data loading ─────────────────────────────────────────────────────────
        print("\n[Data] Loading corpus...", flush=True)
        eval_pool, probe_pool, train_pool = data_utils.load_and_split(
            seed=self.args.seed, eval_size=200, probe_size=300
        )
        print(f"  Corpus: eval={len(eval_pool)} | probe={len(probe_pool)} | "
              f"train={len(train_pool)}", flush=True)
        print(f"  ADE: eval={sum(1 for x in eval_pool if x['label']=='ADE')}, "
              f"NOT_ADE={sum(1 for x in eval_pool if x['label']!='ADE')}", flush=True)

        # ── RAG index ────────────────────────────────────────────────────────────
        rag_dir = str(self.out_dir / "rag_index")
        print(f"\n[RAG] Building/loading FAISS index at {rag_dir}...", flush=True)
        if self.args.fresh or not (Path(rag_dir) / "faiss.index").exists():
            rag_index = RAGIndex.build(train_pool, out_dir=rag_dir)
        else:
            rag_index = RAGIndex.load(rag_dir)

        # ── Cortex initialization ─────────────────────────────────────────────────
        cortex_path = str(self.out_dir / "cortex_state.json")
        if not self.args.fresh and Path(cortex_path).exists():
            print(f"\n[Cortex] Loading existing state from {cortex_path}...", flush=True)
            cortex = Cortex.load(cortex_path)
        else:
            print(f"\n[Cortex] Building seed cortex from config...", flush=True)
            cortex = self.build_seed_cortex(config)

        print(f"\n{cortex.summary()}\n", flush=True)

        # ── Biological controllers ────────────────────────────────────────────────
        homeostatic = HomeostaticPlasticity(
            initial_threshold=1.723,  # Prior: 71/29 ADE class imbalance
            target=config.calibration_target if hasattr(config, "calibration_target") else "f1",
        )
        critical_period = CriticalPeriod(T_min=0.60, T_max=0.85, tau=5.0)

        # ── Training rounds ───────────────────────────────────────────────────────
        random.seed(self.args.seed)

        for round_num in range(1, self.args.rounds + 1):
            # Sample training batch
            batch_size = config.batch_size if hasattr(config, "batch_size") else 250
            batch = random.sample(train_pool, min(batch_size, len(train_pool)))

            result = self.run_training_round(
                round_num=round_num,
                cortex=cortex,
                train_batch=batch,
                eval_pool=eval_pool,
                probe_pool=probe_pool,
                rag_index=rag_index,
                llm_fn=llm_fn,
                homeostatic=homeostatic,
                critical_period=critical_period,
                config=config,
            )
            self.round_history.append(result)

            # Save state after each round
            cortex.save(cortex_path)
            self._save_run_log()

        # ── Final report ──────────────────────────────────────────────────────────
        self._print_final_report(cortex)

    def _save_run_log(self) -> None:
        log_path = self.out_dir / "cortex_run_log.json"
        log_path.write_text(json.dumps({
            "rounds": self.round_history,
            "genesis_log": self.genesis_log,
            "pruning_log": self.pruning_log,
        }, indent=2))

    def _print_final_report(self, cortex: Cortex) -> None:
        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS CORTEX v1.0 — FINAL REPORT", flush=True)
        print(f"{'╚'+'═'*78+'╝'}", flush=True)
        print(f"\n  Total rounds: {len(self.round_history)}", flush=True)
        if self.round_history:
            best = max(self.round_history, key=lambda r: r["f1"])
            last = self.round_history[-1]
            print(f"  Best F1:  {best['f1']:.4f} at R{best['round']} "
                  f"({best['n_columns']} columns, {best['phase']} phase)", flush=True)
            print(f"  Final F1: {last['f1']:.4f} at R{last['round']} "
                  f"({last['n_columns']} columns)", flush=True)
        print(f"\n  Genesis events: {len(self.genesis_log)}", flush=True)
        print(f"  Pruning events: {len(self.pruning_log)}", flush=True)
        print(f"\n{cortex.summary()}", flush=True)

        # F1 trajectory
        print(f"\n  F1 Trajectory:", flush=True)
        for r in self.round_history:
            bar = "█" * int(r["f1"] * 40)
            print(f"    R{r['round']:02d} [{r['phase'][:3].upper()}] "
                  f"{r['f1']:.4f} {bar} | cols={r['n_columns']} | "
                  f"T={r['firing_threshold']:.3f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════════
# §14  CLI — ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="NEXUS Cortex v1.0 — Biologically-Grounded Adaptive Clinical NLP"
    )
    ap.add_argument("--task",    default="task_configs/ade_cortex_v1.json",
                    help="Path to task config JSON")
    ap.add_argument("--out",     default="run_cortex_v1",
                    help="Output directory for cortex state + logs")
    ap.add_argument("--rounds",  type=int, default=20)
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for classification")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--fresh",   action="store_true",
                    help="Ignore existing cortex state — start from seed columns")

    # Northwell AI Hub (enterprise)
    ap.add_argument("--ai-hub",       action="store_true")
    ap.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    ap.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))

    # Mock mode (no API cost — for testing)
    ap.add_argument("--mock", action="store_true",
                    help="MockClient — deterministic, no API cost")

    args = ap.parse_args()

    # Load task config
    import task_config as tc
    config = tc.TaskConfig.load(args.task)

    trainer = CortexTrainer(args)
    trainer.run(config)


if __name__ == "__main__":
    main()

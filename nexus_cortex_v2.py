"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  NEXUS CORTEX v2.0                                                               ║
║  Guided Developmental Cortical NLP Classifier                                    ║
║  Northwell Health — NEXUS Research Program                                        ║
║  Author: Yasir El-Sherif, MD | Computational Neuroscience × Clinical AI          ║
╚══════════════════════════════════════════════════════════════════════════════════╝

────────────────────────────────────────────────────────────────────────────────────
 WHAT CHANGED FROM v1.0-cortex AND WHY
────────────────────────────────────────────────────────────────────────────────────

v1.0-cortex ran 20 rounds and showed net zero learning (R1 F1=0.778, R20 F1=0.759).
The biological mechanisms (BCM, critical period, homeostasis, apoptosis) all fired
correctly. The cortex stabilized without developing. Three pieces were missing:

MISSING PIECE 1 — CONTRASTIVE LEARNING (MCQLibrary)
─────────────────────────────────────────────────────
WorkingMemory stored raw error cases: "you got this wrong."
MCQLibrary generates structured lessons: "here is right, here is wrong, here is
  WHY each wrong answer is wrong." The contrastive signal — seeing wrong answers
  with explanations — is how supervised learning actually works. A student who
  only knows their score never learns as fast as one who reviews the worked solution.

  MCQItem structure:
    text → the sentence
    correct_answer (ADE|NOT_ADE) + rationale (why this is correct)
    wrong_answers → list of {answer, explanation of why it is wrong}

  BCM-gated depth: LTD columns → max 2 MCQs. LTP columns → max 8 MCQs.
  This preserves Fix-FM-3 from v3.05: complexity is earned, not given.

MISSING PIECE 2 — REJECTED PROPOSAL MEMORY (RejectedProposalMemory)
─────────────────────────────────────────────────────────────────────
In v1, the LLM proposed "has_toxicity + negations" in R7, R9, and R10.
Three times. With no memory that it tried before or why it failed.

RejectedProposalMemory logs every failed genesis: trigger, probe Δ, estimated
reason, round. Before every new genesis call, the LLM reads the FULL history.
It cannot repeat a failed pattern without explanation.

MISSING PIECE 3 — ADULT GUIDANCE (MetaAgent)
──────────────────────────────────────────────
When F1 declines 2+ consecutive rounds, a meta-agent LLM call fires.
The LLM receives: full cortex state, F1 trajectory, error patterns, rejected
proposals. It returns a structured diagnosis with specific interventions:
  - prompt refinement guidance
  - pruning recommendations
  - genesis proposals with rationale grounded in actual error analysis

The LLM is the adult. The cortex is the child. In v1, the adult was mute.

NEW MECHANISM 4 — SHADOW COLUMN PERIOD
────────────────────────────────────────
New columns enter shadow_mode for 1 round.
During shadow: they observe which cases would have been routed to them, but do
NOT participate in routing. Their prompt is refined against those observed cases.
After shadow round: a TRIGGER-SCOPED probe — evaluated only on cases matching
their trigger, vs ROOT's performance on those same cases.
Only then: full routing activation (or rollback if trigger-scoped probe fails).

This solves the "cold column" problem: v1 probed new columns immediately after
genesis with 0 MemoryTraces and 0 rehearsal. The global 50-case probe included
cases the new column would never touch, diluting the signal. Every proposal
looked harmful even when it wasn't.

NEW MECHANISM 5 — TRIGGER-SCOPED GENESIS PROBE
────────────────────────────────────────────────
Genesis probe now evaluates ONLY on cases matching the proposed trigger.
Compares column performance vs ROOT baseline on the same trigger-matching subset.
Falls back to global probe if fewer than 10 trigger-matched cases exist.

────────────────────────────────────────────────────────────────────────────────────
 KEY REFERENCES (all v1 references retained; additions marked NEW)
────────────────────────────────────────────────────────────────────────────────────

Hebb D.O. (1949). The Organization of Behavior. Wiley.
Bienenstock E., Cooper L., Munro P. (1982). J Neuroscience 2(1):32-48.
Wiesel T., Hubel D. (1963). J Neurophysiology.
Rumelhart D., Zipser D. (1985). Cognitive Science 9(1):75-112.
Buzsáki G. (1989). Neuroscience 31(3):551-570.
Turrigiano G. et al. (1998). Nature 391:892-896.
Eriksson P. et al. (1998). Nature Medicine 4:1313-1317.
Hensch T.K. (2004). Annual Review of Neuroscience 27:549-579.
Semon R. (1904). Die Mneme.
Josselyn S. et al. (2015). Nature Reviews Neuroscience 16:521-534.
Gurulingappa H. et al. (2012). J Biomedical Informatics 45(5):885-892.
[NEW] Baddeley A. (1992). Working memory. Science 255:556-559.
[NEW] VanLehn K. (2011). The relative effectiveness of human tutoring,
  intelligent tutoring systems, and other tutoring systems. Educational Psychologist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import data_utils
import llm_client
from embedder import embed as embedder_embed, _load_model as _load_embedder
from features import features as extract_features, safe_eval_condition, FEATURE_NAMES
from rag_index import RAGIndex
from task_config import TaskConfig


# ══════════════════════════════════════════════════════════════════════════════════
# §1  DEVELOPMENTAL PHASES + PHASE ROUTER
# ══════════════════════════════════════════════════════════════════════════════════

class LearningPhase(Enum):
    EMBRYONIC      = "embryonic"       # R1-5:  permissive growth
    DEVELOPMENTAL  = "developmental"   # R6-12: competitive selection
    CONSOLIDATION  = "consolidation"   # R13+:  pruning dominant

def phase_for_round(round_num: int) -> LearningPhase:
    if round_num <= 5:   return LearningPhase.EMBRYONIC
    if round_num <= 12:  return LearningPhase.DEVELOPMENTAL
    return LearningPhase.CONSOLIDATION


# ══════════════════════════════════════════════════════════════════════════════════
# §2  BCM PLASTICITY STATE
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class BCMState:
    """BCM sliding modification threshold (Bienenstock, Cooper, Munro 1982).
    θ_M(t) = (1-τ)*θ_M(t-1) + τ*y²
    y²>θ_M → LTP | y²<0.5*θ_M → LTD | else STABLE
    """
    theta_m:   float = 0.1
    tau:       float = 0.15
    ltp_count: int   = 0
    ltd_count: int   = 0

    def update(self, y: float, round_num: int) -> str:
        y2 = y * y
        if y2 > self.theta_m:
            event = "LTP"; self.ltp_count += 1
        elif y2 < 0.5 * self.theta_m:
            event = "LTD"; self.ltd_count += 1
        else:
            event = "STABLE"
        self.theta_m = (1 - self.tau) * self.theta_m + self.tau * y2
        return event

    @property
    def rehearsal_weight(self) -> float:
        """BCM-gated rehearsal depth. LTD→0.30, LTP→1.0. Controls MCQ depth."""
        if self.ltp_count == 0 and self.ltd_count == 0:
            return 0.5
        ltp_frac = self.ltp_count / max(1, self.ltp_count + self.ltd_count)
        return 0.3 + 0.7 * ltp_frac

    def to_dict(self) -> dict:
        return {"theta_m": round(self.theta_m, 4), "tau": self.tau,
                "ltp_count": self.ltp_count, "ltd_count": self.ltd_count,
                "rehearsal_weight": round(self.rehearsal_weight, 3)}


# ══════════════════════════════════════════════════════════════════════════════════
# §3  CRITICAL PERIOD PLASTICITY
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class CriticalPeriod:
    """Critical period plasticity (Hensch 2004).
    T(t) = T_min + (T_max-T_min)*(1-exp(-t/τ))
    """
    T_min: float = 0.60
    T_max: float = 0.85
    tau:   float = 5.0

    def genesis_threshold(self, round_num: int) -> float:
        return self.T_min + (self.T_max - self.T_min) * (1 - math.exp(-round_num / self.tau))

    def describe(self, round_num: int) -> str:
        phase = phase_for_round(round_num)
        return (f"T={self.genesis_threshold(round_num):.3f} "
                f"[phase={phase.value}, round={round_num}]")


# ══════════════════════════════════════════════════════════════════════════════════
# §4  MEMORY TRACES + ENGRAM CLUSTERS
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryTrace:
    """Consolidated memory trace (engram) in a CorticalColumn."""
    id:                 str
    text:               str
    source_column:      str
    round_created:      int
    consolidation_score: float
    bequeathed:         bool  = False
    bequeathed_from:    Optional[str] = None

@dataclass
class EnggramCluster:
    """Hippocampal rapid-binding cluster (Buzsáki 1989): co-occurring error embeddings."""
    cases:     list[dict]
    centroid:  list[float]
    coherence: float


# ══════════════════════════════════════════════════════════════════════════════════
# §5  MCQ LIBRARY — CONTRASTIVE LEARNING (new in v2)
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class MCQItem:
    """
    A Multiple Choice Question generated from a misclassified case.

    Contrastive learning theory (VanLehn 2011): seeing wrong answers with
    explicit explanations of why they are wrong produces stronger learning
    signal than correct answers alone. A student who reads "B is wrong because
    negation only cancels the modifier, not the causal relationship" learns
    more than one who reads "A is correct."

    Fields:
        text:             The sentence being classified
        true_label:       ADE | NOT_ADE
        predicted_label:  What the system incorrectly predicted
        correct_rationale: Why the correct answer is correct
        wrong_answers:    [{"answer": "...", "explanation": "why wrong"}]
        round_created:    Training round when error occurred
        ltp_round:        Whether this was an LTP round (weighted higher in rehearsal)
    """
    text:              str
    true_label:        str
    predicted_label:   str
    correct_rationale: str
    wrong_answers:     list[dict]   # [{"answer": str, "explanation": str}]
    round_created:     int
    ltp_round:         bool = False

    def format_for_prompt(self, index: int) -> str:
        """Format as a prompt-ready MCQ block."""
        wa_lines = "\n".join(
            f"   {'✗'} {wa['answer']}: {wa['explanation']}"
            for wa in self.wrong_answers
        )
        return (
            f"[MCQ {index}] Sentence: \"{self.text}\"\n"
            f"   {'✓'} {self.true_label} (CORRECT): {self.correct_rationale}\n"
            f"{wa_lines}"
        )


class MCQLibrary:
    """
    Per-column contrastive learning library. Replaces WorkingMemory from v1.

    WorkingMemory (v1): stored raw error cases. The LLM saw "you got this wrong."
    MCQLibrary  (v2): stores structured MCQs. The LLM sees "here is right,
                       here is why, here are the wrong answers and why each is wrong."

    BCM-gated depth:
        LTD columns (rehearsal_weight≈0.30) → max 2 MCQs
        LTP columns (rehearsal_weight≈1.00) → max 8 MCQs
    This preserves Fix-FM-3 (BCM-gated complexity from v3.05 analysis).
    """

    def __init__(self, capacity: int = 40):
        self._capacity = capacity
        self._items: list[MCQItem] = []
        self._lock = threading.Lock()

    def generate_and_add(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        llm_fn: Callable,
        round_num: int,
        ltp_round: bool = False,
    ) -> Optional[MCQItem]:
        """Generate an MCQ from a misclassified case and add it to the library."""
        prompt_system = (
            "You are an expert in clinical NLP and adverse drug event (ADE) classification. "
            "Your job is to generate a structured teaching question from a misclassified sentence."
        )
        prompt_user = (
            f"A clinical text classifier misclassified the following sentence.\n\n"
            f"Sentence: \"{text}\"\n"
            f"Correct label: {true_label}\n"
            f"Predicted label: {predicted_label}\n\n"
            f"Generate a Multiple Choice Question (MCQ) teaching this case. "
            f"Return ONLY a JSON object with this exact structure:\n"
            f'{{\n'
            f'  "correct_rationale": "one sentence explaining why {true_label} is correct",\n'
            f'  "wrong_answers": [\n'
            f'    {{"answer": "{predicted_label}", "explanation": "why {predicted_label} is wrong here"}},\n'
            f'    {{"answer": "UNCERTAIN", "explanation": "why uncertainty is not appropriate here"}}\n'
            f'  ]\n'
            f'}}'
        )
        try:
            response = llm_fn(system=prompt_system, user=prompt_user)
            # Extract JSON from response
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group())
            mcq = MCQItem(
                text=text,
                true_label=true_label,
                predicted_label=predicted_label,
                correct_rationale=data.get("correct_rationale", ""),
                wrong_answers=data.get("wrong_answers", []),
                round_created=round_num,
                ltp_round=ltp_round,
            )
            with self._lock:
                self._items.append(mcq)
                if len(self._items) > self._capacity:
                    self._items.pop(0)
            return mcq
        except Exception as e:
            return None

    def format_for_prompt(self, bcm_state: Optional[BCMState] = None) -> str:
        """Format BCM-gated MCQ block for inclusion in classification prompt."""
        with self._lock:
            if not self._items:
                return ""
            w = bcm_state.rehearsal_weight if bcm_state else 1.0
            max_items = max(2, int(8 * w))
            # Prefer LTP-round items (stronger consolidation signal)
            ltp_items  = [m for m in self._items if m.ltp_round]
            base_items = [m for m in self._items if not m.ltp_round]
            selected = (ltp_items + base_items)[-max_items:]
            lines = ["STRUCTURED LESSONS FROM PAST ERRORS (study right and wrong):"]
            for i, item in enumerate(selected, 1):
                lines.append(item.format_for_prompt(i))
            return "\n".join(lines)

    def recent_errors(self, n: int = 20) -> list[MCQItem]:
        with self._lock:
            return list(self._items[-n:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def to_dict(self) -> dict:
        with self._lock:
            return {"capacity": self._capacity, "stored": len(self._items)}


# ══════════════════════════════════════════════════════════════════════════════════
# §6  REJECTED PROPOSAL MEMORY (new in v2)
# ══════════════════════════════════════════════════════════════════════════════════

@dataclass
class RejectedProposal:
    """Record of a failed columnar genesis attempt."""
    col_id:      str
    trigger:     str
    spec:        int
    probe_delta: float
    round:       int
    jaccard:     float
    reason:      str   # Estimated reason for failure

class RejectedProposalMemory:
    """
    Persistent log of every failed genesis proposal, fed to the LLM before
    every new genesis call.

    Without this, the LLM proposes the same failed pattern repeatedly.
    With this, the LLM reads: "I tried has_toxicity + negations 3 times.
    Each time probe Δ was negative. I need a different approach."

    This is the 'memory of wrong answers' for genesis — the same contrastive
    signal that MCQs provide for classification.
    """

    def __init__(self):
        self._proposals: list[RejectedProposal] = []
        self._lock = threading.Lock()

    def log(
        self,
        col_id: str,
        trigger: str,
        spec: int,
        probe_delta: float,
        round_num: int,
        jaccard: float,
        reason: str = "",
    ) -> None:
        """Record a failed genesis attempt."""
        # Estimate reason if not provided
        if not reason:
            if spec == 0:
                reason = "All-negative trigger (spec=0): no discriminative power over ROOT"
            elif spec == 1 and jaccard >= 0.40:
                reason = f"High overlap with existing column (Jaccard={jaccard:.3f}): routing competition"
            elif probe_delta < -0.10:
                reason = f"Large probe drop (Δ={probe_delta:+.4f}): trigger too broad, steals correct ROOT routes"
            elif probe_delta < -0.02:
                reason = f"Moderate probe drop (Δ={probe_delta:+.4f}): column not yet ready for these cases"
            else:
                reason = f"Probe drop (Δ={probe_delta:+.4f}): insufficient specialization"

        with self._lock:
            self._proposals.append(RejectedProposal(
                col_id=col_id, trigger=trigger, spec=spec,
                probe_delta=probe_delta, round=round_num,
                jaccard=jaccard, reason=reason,
            ))

    def format_for_genesis_context(self) -> str:
        """Format the full rejection history as LLM context for genesis proposals."""
        with self._lock:
            if not self._proposals:
                return ""
            lines = [
                "REJECTED GENESIS PROPOSALS (do NOT repeat these patterns):",
                "Study these failures before proposing a new column.\n",
            ]
            for p in self._proposals:
                lines.append(
                    f"  Round {p.round}: {p.col_id}\n"
                    f"    Trigger: {p.trigger}\n"
                    f"    Spec={p.spec}, Jaccard={p.jaccard:.3f}, Probe Δ={p.probe_delta:+.4f}\n"
                    f"    Why it failed: {p.reason}"
                )
            lines.append(
                "\nPATTERNS TO AVOID:\n"
                "- spec=1 triggers with mostly-negative conditions (has_toxicity + 4 negations)\n"
                "- Triggers that overlap existing columns (Jaccard > 0.30)\n"
                "- Triggers that cover ROOT's core cases (all broad positive features)\n"
                "WHAT TENDS TO SURVIVE:\n"
                "- spec ≥ 2 (multiple positive feature conditions)\n"
                "- Jaccard < 0.15 with all existing columns\n"
                "- Triggers based on rare but reliable positive signals"
            )
            return "\n".join(lines)

    def to_list(self) -> list[dict]:
        with self._lock:
            return [
                {"col_id": p.col_id, "trigger": p.trigger, "spec": p.spec,
                 "probe_delta": p.probe_delta, "round": p.round,
                 "jaccard": p.jaccard, "reason": p.reason}
                for p in self._proposals
            ]

    @classmethod
    def from_list(cls, data: list[dict]) -> "RejectedProposalMemory":
        mem = cls()
        for d in data:
            mem._proposals.append(RejectedProposal(
                col_id=d["col_id"], trigger=d["trigger"], spec=d.get("spec", 1),
                probe_delta=d["probe_delta"], round=d["round"],
                jaccard=d.get("jaccard", 0.0), reason=d.get("reason", ""),
            ))
        return mem

    def __len__(self) -> int:
        with self._lock:
            return len(self._proposals)


# ══════════════════════════════════════════════════════════════════════════════════
# §7  META-AGENT — LLM AS DIAGNOSTIC PHYSICIAN (new in v2)
# ══════════════════════════════════════════════════════════════════════════════════

class MetaAgent:
    """
    The LLM as attending physician for the developing cortex.

    When the cortex is stagnating (F1 declining 2+ consecutive rounds) or has
    accumulated too many failed genesis attempts, the MetaAgent fires a
    diagnostic LLM call. The LLM receives the full clinical picture:
      - All column states (BCM, routes, traces, errors)
      - F1 trajectory
      - All rejected genesis proposals
      - Most common error cases this round

    The LLM returns a structured diagnosis with specific interventions.
    The MetaAgent parses these and queues them for the trainer to execute.

    Biological analogy: a developing brain has adult supervision during
    critical periods — parental care, environmental scaffolding, correction.
    Without guidance, the child stabilizes but does not develop.
    """

    def __init__(self, consecutive_decline_threshold: int = 2):
        self.threshold = consecutive_decline_threshold
        self.last_fired_round: int = 0
        self.diagnosis_log: list[dict] = []

    def should_trigger(self, f1_history: list[float], current_round: int) -> bool:
        """Fire if F1 has declined for `threshold` consecutive rounds."""
        if len(f1_history) < self.threshold + 1:
            return False
        if current_round - self.last_fired_round < 3:
            return False  # Don't fire more than once every 3 rounds
        recent = f1_history[-self.threshold:]
        return all(recent[i] > recent[i+1] for i in range(len(recent)-1))

    def diagnose(
        self,
        cortex_summary: str,
        f1_history: list[float],
        rejected_proposals: RejectedProposalMemory,
        error_cases: list[dict],
        round_num: int,
        n_fp: int,
        n_fn: int,
        llm_fn: Callable,
    ) -> dict:
        """
        Fire a diagnostic LLM call. Returns parsed interventions.
        """
        self.last_fired_round = round_num

        # Format error cases for context
        error_context = ""
        if error_cases:
            sample = error_cases[:10]
            error_context = "RECENT MISCLASSIFIED CASES (this round):\n" + "\n".join(
                f"  [{c.get('true_label','?')}→{c.get('predicted','?')}] {c.get('text','')[:120]}"
                for c in sample
            )

        # Format F1 trajectory
        f1_str = " → ".join(f"{f:.4f}" for f in f1_history[-8:])
        trend = "DECLINING" if len(f1_history) >= 2 and f1_history[-1] < f1_history[-2] else "STABLE"

        prompt_system = (
            "You are a computational neuroscience expert advising on the development of "
            "an adaptive cortical NLP classifier. Your job is to diagnose why the system "
            "is not improving and recommend specific interventions."
        )

        prompt_user = (
            f"CORTEX STATUS REPORT — Round {round_num}\n"
            f"{'='*60}\n\n"
            f"COLUMN STATES:\n{cortex_summary}\n\n"
            f"F1 TRAJECTORY (last 8 rounds): {f1_str} [{trend}]\n"
            f"Current eval: FP={n_fp}, FN={n_fn}\n\n"
            f"{rejected_proposals.format_for_genesis_context()}\n\n"
            f"{error_context}\n\n"
            f"{'='*60}\n"
            f"DIAGNOSIS TASK:\n"
            f"1. Identify the primary reason for stagnation or decline.\n"
            f"2. Recommend up to 3 specific interventions.\n\n"
            f"Return ONLY a JSON object:\n"
            f'{{\n'
            f'  "diagnosis": "one paragraph explaining root cause",\n'
            f'  "interventions": [\n'
            f'    {{\n'
            f'      "type": "refine|prune|threshold_up|threshold_down|genesis_hint",\n'
            f'      "target": "column_id or null",\n'
            f'      "guidance": "specific instruction",\n'
            f'      "priority": 1\n'
            f'    }}\n'
            f'  ]\n'
            f'}}'
        )

        try:
            response = llm_fn(system=prompt_system, user=prompt_user)
            m = re.search(r'\{.*\}', response, re.DOTALL)
            if not m:
                return {"diagnosis": "Parse failed", "interventions": []}
            result = json.loads(m.group())
            result["round"] = round_num
            self.diagnosis_log.append(result)
            return result
        except Exception as e:
            print(f"  [MetaAgent] Diagnosis failed: {e}", file=sys.stderr)
            return {"diagnosis": str(e), "interventions": []}


# ══════════════════════════════════════════════════════════════════════════════════
# §8  CORTICAL COLUMN (updated for v2: MCQLibrary, shadow mode)
# ══════════════════════════════════════════════════════════════════════════════════

class CorticalColumn:
    """
    A cortical column: the fundamental processing unit of the neocortex.

    v2 changes:
    - WorkingMemory → MCQLibrary (contrastive lessons instead of raw errors)
    - shadow_mode: new columns observe for 1 round before routing
    - shadow_cases: cases observed during shadow round (for warm-up refinement)
    """

    def __init__(
        self,
        col_id:          str,
        description:     str,
        trigger_condition: str,
        prompt:          str,
        genesis_round:   int = 0,
        genesis_phase:   LearningPhase = LearningPhase.EMBRYONIC,
    ):
        self.id                = col_id
        self.description       = description
        self.trigger_condition = trigger_condition
        self.prompt            = prompt
        self.genesis_round     = genesis_round
        self.genesis_phase     = genesis_phase
        self.prompt_version    = 0

        # Plasticity
        self.bcm_state     = BCMState()
        self.memory_traces: list[MemoryTrace] = []
        self.mcq_library   = MCQLibrary(capacity=40)

        # Activation tracking
        self.activation_history: list[int]   = []
        self.f1_history:         list[float] = []
        self.route_count_total:  int         = 0

        # Shadow mode (v2 new): observe before routing
        self.shadow_mode:  bool      = False
        self.shadow_round: int       = 0
        self.shadow_cases: list[dict] = []  # Cases that WOULD have been routed

    def compute_specificity(self) -> int:
        """Count positive (non-negated) feature conditions in trigger."""
        if not self.trigger_condition or self.trigger_condition.strip() == "True":
            return 0
        pos = 0
        for feat in FEATURE_NAMES:
            pattern = rf'\b{re.escape(feat)}\b'
            matches = [m.start() for m in re.finditer(pattern, self.trigger_condition)]
            for idx in matches:
                prefix = self.trigger_condition[:idx].rstrip()
                if not (prefix.endswith("not") or prefix.endswith("not ")):
                    pos += 1
        return pos

    def matches_trigger(self, features: dict) -> bool:
        """True if this column's trigger fires for the given feature vector."""
        if self.trigger_condition.strip() == "True":
            return True
        return safe_eval_condition(self.trigger_condition, features)

    def bcm_update(self, cases_routed: int, cases_total: int, round_num: int) -> str:
        y = cases_routed / max(1, cases_total)
        event = self.bcm_state.update(y, round_num)
        self.activation_history.append(cases_routed)
        self.route_count_total += cases_routed
        return event

    def add_memory_trace(self, text: str, consolidation_score: float,
                         round_num: int, bequeathed: bool = False,
                         bequeathed_from: Optional[str] = None) -> MemoryTrace:
        trace_id = f"{self.id}_T{len(self.memory_traces)+1}_R{round_num}"
        t = MemoryTrace(id=trace_id, text=text, source_column=self.id,
                        round_created=round_num,
                        consolidation_score=consolidation_score,
                        bequeathed=bequeathed, bequeathed_from=bequeathed_from)
        self.memory_traces.append(t)
        return t

    def memory_trace_context(self, max_traces: int = 3) -> str:
        if not self.memory_traces:
            return ""
        traces = sorted(self.memory_traces,
                        key=lambda t: t.consolidation_score, reverse=True)[:max_traces]
        return "CONSOLIDATED KNOWLEDGE (memory traces):\n" + "\n".join(
            f"  [{t.round_created}] {t.text[:200]}" for t in traces
        )

    def coverage_set(self, cases: list[dict]) -> set[int]:
        indices = set()
        for i, case in enumerate(cases):
            feats = extract_features(case["text"])
            if self.id == "ROOT" or safe_eval_condition(self.trigger_condition, feats):
                indices.add(i)
        return indices

    def enter_shadow_mode(self, round_num: int) -> None:
        """Enter 1-round shadow period: observe but do not route."""
        self.shadow_mode  = True
        self.shadow_round = round_num
        self.shadow_cases = []

    def exit_shadow_mode(self) -> None:
        """Exit shadow mode; column is now active in routing."""
        self.shadow_mode  = False
        self.shadow_cases = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "trigger_condition": self.trigger_condition,
            "prompt": self.prompt,
            "prompt_version": self.prompt_version,
            "genesis_round": self.genesis_round,
            "genesis_phase": self.genesis_phase.value,
            "specificity": self.compute_specificity(),
            "route_count_total": self.route_count_total,
            "activation_history": self.activation_history,
            "f1_history": self.f1_history,
            "memory_traces": len(self.memory_traces),
            "mcq_library": self.mcq_library.to_dict(),
            "bcm_state": self.bcm_state.to_dict(),
            "shadow_mode": self.shadow_mode,
        }


# ══════════════════════════════════════════════════════════════════════════════════
# §9  CORTEX — COLUMN ENSEMBLE + COMPETITIVE ROUTING
# ══════════════════════════════════════════════════════════════════════════════════

class Cortex:
    """
    The neocortex: ensemble of CorticalColumns with competitive routing,
    genesis (neurogenesis), pruning (apoptosis), and engram consolidation.

    v2 changes:
    - Shadow columns are excluded from routing (observe only)
    - genesis is accompanied by shadow period before full activation
    - rejected_proposals persisted alongside column state
    """

    def __init__(self):
        self._columns: list[CorticalColumn] = []
        self._lock = threading.RLock()

    def add_column(self, col: CorticalColumn) -> None:
        with self._lock:
            self._columns.append(col)

    def get_column(self, col_id: str) -> Optional[CorticalColumn]:
        with self._lock:
            for c in self._columns:
                if c.id == col_id:
                    return c
            return None

    @property
    def columns(self) -> list[CorticalColumn]:
        with self._lock:
            return list(self._columns)

    @property
    def active_columns(self) -> list[CorticalColumn]:
        """Columns that are NOT in shadow mode (participate in routing)."""
        with self._lock:
            return [c for c in self._columns if not c.shadow_mode]

    @property
    def shadow_columns(self) -> list[CorticalColumn]:
        with self._lock:
            return [c for c in self._columns if c.shadow_mode]

    def route(self, features: dict) -> CorticalColumn:
        """
        Competitive winner-take-all routing by specificity (Fix-FM-1).
        Shadow-mode columns are EXCLUDED from routing.
        Returns the winning column (ROOT if no specialist matches).
        """
        active = self.active_columns
        root = next((c for c in active if c.id == "ROOT"), active[0])
        specialists = [c for c in active if c.id != "ROOT"]

        candidates = [c for c in specialists if c.matches_trigger(features)]
        if not candidates:
            return root
        return max(candidates, key=lambda c: c.compute_specificity())

    def observe_shadow(self, features: dict, case: dict) -> None:
        """
        For each shadow column whose trigger matches this case, record it
        as an observed case (for warm-up refinement after shadow round).
        """
        for col in self.shadow_columns:
            if col.matches_trigger(features):
                col.shadow_cases.append(case)

    def jaccard_overlap(self, trigger: str, probe_cases: list[dict]) -> dict:
        """Compute max Jaccard overlap between proposed trigger and existing columns."""
        tmp = CorticalColumn("__tmp__", "", trigger, "")
        proposed_set = tmp.coverage_set(probe_cases)
        max_j, worst = 0.0, ""
        for col in self.active_columns:
            if col.id == "ROOT":
                continue
            col_set = col.coverage_set(probe_cases)
            union = proposed_set | col_set
            if not union:
                continue
            j = len(proposed_set & col_set) / len(union)
            if j > max_j:
                max_j, worst = j, col.id
        return {"max_jaccard": max_j, "worst_col": worst}

    def columnar_genesis(
        self,
        proposal: dict,
        critical_period: CriticalPeriod,
        round_num: int,
        max_jaccard: float = 0.50,
        probe_cases: Optional[list[dict]] = None,
    ) -> tuple[bool, str]:
        """
        Validate and approve (or reject) a genesis proposal.
        On approval: column enters SHADOW MODE for 1 round before full routing.
        """
        col_id  = proposal.get("id", "")
        trigger = proposal.get("trigger_condition", "")
        prompt  = proposal.get("prompt", "")

        # Required fields
        if not col_id or not trigger or not prompt:
            return False, "Missing required fields (id, trigger_condition, prompt)"

        # No duplicate IDs
        if self.get_column(col_id):
            return False, f"Column '{col_id}' already exists"

        # Trigger syntax
        tmp = CorticalColumn("__tmp__", "", trigger, "")
        tokens = set(re.findall(r'\b[a-z_]+\b', trigger)) - {"and","or","not","True","False"}
        unknown = tokens - set(FEATURE_NAMES)
        if unknown:
            return False, f"Unknown feature names in trigger: {unknown}"

        # FM-6 fix: reject all-negative triggers
        if tmp.compute_specificity() == 0 and trigger.strip() not in ("True", ""):
            return False, f"Trigger has spec=0 (all-negative). Must have ≥1 positive condition."

        # FM-5 fix: Jaccard overlap audit
        if probe_cases:
            overlap = self.jaccard_overlap(trigger, probe_cases[:100])
            if overlap["max_jaccard"] > max_jaccard:
                return False, (
                    f"Trigger overlap too high: Jaccard={overlap['max_jaccard']:.3f} "
                    f"with '{overlap['worst_col']}' (max={max_jaccard})"
                )

        # Approve — create column in SHADOW MODE (v2 new)
        new_col = CorticalColumn(
            col_id=col_id,
            description=proposal.get("description", ""),
            trigger_condition=trigger,
            prompt=prompt,
            genesis_round=round_num,
            genesis_phase=phase_for_round(round_num),
        )
        new_col.enter_shadow_mode(round_num)   # Shadow period: observe before routing

        with self._lock:
            self._columns.append(new_col)

        return True, "approved"

    def activate_shadow_column(self, col_id: str) -> Optional[CorticalColumn]:
        """Exit shadow mode for a column after warm-up round."""
        col = self.get_column(col_id)
        if col and col.shadow_mode:
            col.exit_shadow_mode()
        return col

    def synaptic_pruning(self, round_num: int,
                         min_rounds: int = 3,
                         min_avg_activation: float = 2.0) -> list[CorticalColumn]:
        """
        Apoptosis: prune columns with zero or very low average activation.
        Bequeath their MemoryTraces to ROOT (principle inheritance).
        Only applies in DEVELOPMENTAL and CONSOLIDATION phases.
        """
        pruned = []
        root = self.get_column("ROOT")
        with self._lock:
            to_remove = []
            for col in self._columns:
                if col.id == "ROOT" or col.shadow_mode:
                    continue
                rounds_alive = round_num - col.genesis_round
                if rounds_alive < min_rounds:
                    continue
                avg = col.route_count_total / max(1, rounds_alive)
                if avg < min_avg_activation:
                    # Bequest MemoryTraces to ROOT
                    if root:
                        for t in col.memory_traces:
                            bequeathed = MemoryTrace(
                                id=f"BEQUEATHED_{t.id}",
                                text=t.text,
                                source_column=col.id,
                                round_created=round_num,
                                consolidation_score=t.consolidation_score * 0.8,
                                bequeathed=True,
                                bequeathed_from=col.id,
                            )
                            root.memory_traces.append(bequeathed)
                            print(f"  [Apoptosis] Bequeathing MemoryTrace from "
                                  f"{col.id} → ROOT: {t.text[:60]}...", flush=True)
                    print(f"  [Apoptosis ✂] Pruning '{col.id}' | "
                          f"avg_activation={avg:.1f} cases/round | "
                          f"{len(col.memory_traces)} traces bequeathed", flush=True)
                    to_remove.append(col)
            for col in to_remove:
                self._columns.remove(col)
                pruned.append(col)
        return pruned

    def summary(self) -> str:
        cols = self.columns
        lines = [f"Cortex: {len(cols)} columns "
                 f"({sum(1 for c in cols if c.shadow_mode)} in shadow)"]
        for c in sorted(cols, key=lambda x: x.compute_specificity(), reverse=True):
            shadow_tag = " [SHADOW]" if c.shadow_mode else ""
            lines.append(
                f"  {c.id:35s} spec={c.compute_specificity()} "
                f"routes={c.route_count_total:5d} "
                f"traces={len(c.memory_traces):3d} "
                f"mcqs={len(c.mcq_library):3d} "
                f"BCM={c.bcm_state.ltp_count}LTP/{c.bcm_state.ltd_count}LTD"
                f"{shadow_tag}"
            )
        return "\n".join(lines)

    def save(self, out_path: str, meta: dict | None = None) -> None:
        data = {
            "cortex_version": "2.0",
            **(meta or {}),
            "columns": [
                {
                    "id": c.id, "description": c.description,
                    "trigger_condition": c.trigger_condition,
                    "prompt": c.prompt,
                    "genesis_round": c.genesis_round,
                    "genesis_phase": c.genesis_phase.value,
                    "route_count_total": c.route_count_total,
                    "activation_history": c.activation_history,
                    "f1_history": c.f1_history,
                    "prompt_version": c.prompt_version,
                    "shadow_mode": c.shadow_mode,
                    "shadow_round": c.shadow_round,
                    "memory_traces": [
                        {"id": t.id, "text": t.text, "source": t.source_column,
                         "round": t.round_created, "score": t.consolidation_score,
                         "bequeathed": t.bequeathed, "from": t.bequeathed_from}
                        for t in c.memory_traces
                    ],
                    "bcm_state": c.bcm_state.to_dict(),
                }
                for c in self._columns
            ],
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
            col.route_count_total   = cd.get("route_count_total", 0)
            col.activation_history  = cd.get("activation_history", [])
            col.f1_history          = cd.get("f1_history", [])
            col.prompt_version      = cd.get("prompt_version", 0)
            col.shadow_mode         = cd.get("shadow_mode", False)
            col.shadow_round        = cd.get("shadow_round", 0)
            # Restore BCM state
            bcm_d = cd.get("bcm_state", {})
            if bcm_d:
                col.bcm_state.theta_m   = bcm_d.get("theta_m", 0.1)
                col.bcm_state.tau       = bcm_d.get("tau", 0.15)
                col.bcm_state.ltp_count = bcm_d.get("ltp_count", 0)
                col.bcm_state.ltd_count = bcm_d.get("ltd_count", 0)
            # Restore MemoryTraces
            for td in cd.get("memory_traces", []):
                col.memory_traces.append(MemoryTrace(
                    id=td["id"], text=td["text"], source_column=td["source"],
                    round_created=td["round"], consolidation_score=td["score"],
                    bequeathed=td.get("bequeathed", False),
                    bequeathed_from=td.get("from"),
                ))
            cortex._columns.append(col)
        return cortex


# ══════════════════════════════════════════════════════════════════════════════════
# §10  ACTIVATION PATHWAYS (4 parallel routes; FM-2 + FM-8 fixes retained)
# ══════════════════════════════════════════════════════════════════════════════════

def _route_causation(text: str, examples: list[dict], llm_fn: Callable,
                     context: str = "") -> Optional[dict]:
    system = (
        "You are a clinical NLP expert specializing in CAUSAL LANGUAGE analysis. "
        "Focus on whether the sentence contains explicit drug-to-harm causal language."
    )
    ex_text = "\n".join(
        f"  [{e['label']}] {e['text'][:100]}" for e in examples[:3]
    )
    user = (
        f"Classify this sentence for Adverse Drug Event (ADE).\n\n"
        f"SIMILAR EXAMPLES:\n{ex_text}\n\n"
        f"{context}\n\n"
        f"SENTENCE: {text}\n\n"
        f"Does this sentence describe a direct adverse drug event with causal language?\n"
        f"Reply with JSON: {{\"vote\": \"ADE\" or \"NOT_ADE\", \"confidence\": 0.0-1.0, "
        f"\"reason\": \"brief\"}}"
    )
    try:
        resp = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if d.get("vote") in ("ADE", "NOT_ADE"):
                return d
    except Exception:
        pass
    return None


def _route_negation(text: str, examples: list[dict], llm_fn: Callable,
                    context: str = "") -> Optional[dict]:
    system = (
        "You are a clinical NLP expert specializing in NEGATION DETECTION. "
        "Focus on whether adverse outcomes are negated, denied, or ruled out."
    )
    ex_text = "\n".join(
        f"  [{e['label']}] {e['text'][:100]}" for e in examples[:3]
    )
    user = (
        f"Classify for ADE with focus on negation.\n\n"
        f"SIMILAR EXAMPLES:\n{ex_text}\n\n"
        f"{context}\n\n"
        f"SENTENCE: {text}\n\n"
        f"Is any adverse outcome NEGATED or denied? If yes → NOT_ADE.\n"
        f"Reply with JSON: {{\"vote\": \"ADE\" or \"NOT_ADE\", \"confidence\": 0.0-1.0, "
        f"\"reason\": \"brief\"}}"
    )
    try:
        resp = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if d.get("vote") in ("ADE", "NOT_ADE"):
                return d
    except Exception:
        pass
    return None


def _route_drug_effect(text: str, examples: list[dict], llm_fn: Callable,
                       context: str = "") -> Optional[dict]:
    system = (
        "You are a clinical pharmacology expert. "
        "Determine if the sentence describes a documented drug-effect pair."
    )
    ex_text = "\n".join(
        f"  [{e['label']}] {e['text'][:100]}" for e in examples[:3]
    )
    user = (
        f"Classify for ADE — focus on drug-effect relationship.\n\n"
        f"SIMILAR EXAMPLES:\n{ex_text}\n\n"
        f"{context}\n\n"
        f"SENTENCE: {text}\n\n"
        f"Does this show a documented adverse drug-effect pair?\n"
        f"Reply with JSON: {{\"vote\": \"ADE\" or \"NOT_ADE\", \"confidence\": 0.0-1.0, "
        f"\"reason\": \"brief\"}}"
    )
    try:
        resp = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if d.get("vote") in ("ADE", "NOT_ADE"):
                return d
    except Exception:
        pass
    return None


def _route_context(text: str, examples: list[dict], llm_fn: Callable,
                   context: str = "") -> Optional[dict]:
    system = (
        "You are a clinical context expert. "
        "Determine if the medical context confirms an adverse drug event."
    )
    ex_text = "\n".join(
        f"  [{e['label']}] {e['text'][:100]}" for e in examples[:3]
    )
    user = (
        f"Classify for ADE — focus on clinical context.\n\n"
        f"SIMILAR EXAMPLES:\n{ex_text}\n\n"
        f"{context}\n\n"
        f"SENTENCE: {text}\n\n"
        f"Does the clinical context (therapeutic intent vs. documented harm) confirm ADE?\n"
        f"Reply with JSON: {{\"vote\": \"ADE\" or \"NOT_ADE\", \"confidence\": 0.0-1.0, "
        f"\"reason\": \"brief\"}}"
    )
    try:
        resp = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', resp, re.DOTALL)
        if m:
            d = json.loads(m.group())
            if d.get("vote") in ("ADE", "NOT_ADE"):
                return d
    except Exception:
        pass
    return None


def classify_with_routes(
    text: str,
    rag_index: RAGIndex,
    llm_fn: Callable,
    column: CorticalColumn,
    firing_threshold: float = 1.0,
    route_timeout: float = 30.0,
) -> dict:
    """
    Classify using 4 parallel routes with MCQ context (v2: MCQLibrary replaces WorkingMemory).
    FM-2 fix: abstention on route failure.
    FM-8 fix: outer as_completed TimeoutError caught.
    """
    examples = rag_index.query(text, k=5)
    principle_context = ""
    if column.memory_traces:
        principle_context = "\n\n" + column.memory_trace_context(max_traces=3)

    # v2: MCQ contrastive lessons replace raw working memory
    mcq_context = column.mcq_library.format_for_prompt(bcm_state=column.bcm_state)
    if mcq_context:
        principle_context += f"\n\n{mcq_context}"

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
        try:
            for future in as_completed(futures, timeout=route_timeout + 5):
                route_name = futures[future]
                try:
                    result = future.result(timeout=route_timeout)
                    route_results[route_name] = result
                except FuturesTimeoutError:
                    route_results[futures[future]] = None
                except Exception:
                    route_results[futures[future]] = None
        except FuturesTimeoutError:
            # FM-8 fix: outer iterator timeout — mark remaining as abstained
            for future, route_name in futures.items():
                if route_name not in route_results:
                    future.cancel()
                    route_results[route_name] = None
                    print(f"  [ROUTE TIMEOUT] {route_name} — as_completed expired, abstaining",
                          file=sys.stderr)

    route_weights = {"causation": 1.5, "negation": 1.5, "drug_effect": 1.2, "context": 1.0}
    ade_score, not_ade_score = 0.0, 0.0
    responsive_routes = []

    for name, result in route_results.items():
        if result is None:
            continue
        w = route_weights.get(name, 1.0)
        if result["vote"] == "ADE":
            ade_score += w * result.get("confidence", 0.7)
        else:
            not_ade_score += w * result.get("confidence", 0.7)
        responsive_routes.append(result)

    if not responsive_routes:
        return {
            "label": "NOT_ADE", "confidence": 0.1,
            "ade_score": 0.0, "not_ade_score": 1.0,
            "column_id": column.id,
            "route_results": [], "agreement": 0.0, "split": False,
        }

    final_label = "ADE" if ade_score >= not_ade_score * firing_threshold else "NOT_ADE"
    total = ade_score + not_ade_score
    confidence = max(ade_score, not_ade_score) / total if total > 0 else 0.5
    agreeing = sum(1 for r in responsive_routes if r["vote"] == final_label)
    agreement = agreeing / len(responsive_routes)

    return {
        "label": final_label, "confidence": confidence,
        "ade_score": ade_score, "not_ade_score": not_ade_score,
        "column_id": column.id,
        "route_results": responsive_routes,
        "agreement": agreement,
        "split": agreement < 0.6,
    }


# ══════════════════════════════════════════════════════════════════════════════════
# §11  ENGRAM CLUSTER DETECTION
# ══════════════════════════════════════════════════════════════════════════════════

def detect_engram_clusters(
    error_cases: list[dict],
    min_cluster_size: int = 5,
    similarity_threshold: float = 0.70,
) -> list[EnggramCluster]:
    if len(error_cases) < min_cluster_size:
        return []
    texts = [c["text"] for c in error_cases]
    embeddings = embedder_embed(texts)

    import numpy as np
    emb_array = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    emb_norm = emb_array / norms

    centroid = emb_norm.mean(axis=0)
    sims = emb_norm @ centroid
    mask = sims >= similarity_threshold
    cluster_cases = [error_cases[i] for i in range(len(error_cases)) if mask[i]]

    if len(cluster_cases) < min_cluster_size:
        return []

    coherence = float(sims[mask].mean()) if mask.any() else 0.0
    return [EnggramCluster(
        cases=cluster_cases,
        centroid=centroid.tolist(),
        coherence=coherence,
    )]


# ══════════════════════════════════════════════════════════════════════════════════
# §12  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════════

def evaluate_cortex(
    cortex: Cortex,
    eval_pool: list[dict],
    rag_index: RAGIndex,
    llm_fn: Callable,
    homeostatic,
    config,
    round_num: int,
    max_workers: int = 4,
) -> tuple[dict, list[tuple]]:
    pos = config.positive_label
    tp = fp = fn = tn = 0
    score_cache: list[tuple] = []
    col_activations: dict[str, int] = {}
    errors: list[dict] = []

    def _classify_one(item: dict) -> dict:
        feats = extract_features(item["text"])
        col   = cortex.route(feats)
        try:
            result = classify_with_routes(
                text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                column=col, firing_threshold=homeostatic.firing_threshold,
            )
        except Exception as e:
            print(f"  [EVAL ERROR] {e}", file=sys.stderr)
            result = {"label": "NOT_ADE", "confidence": 0.1,
                      "ade_score": 0.0, "not_ade_score": 1.0, "column_id": col.id,
                      "route_results": [], "agreement": 0.0, "split": False}
        return {"item": item, "result": result, "column": col}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_classify_one, item) for item in eval_pool]
        for fut in futs:
            try:
                out = fut.result(timeout=60)
            except Exception:
                continue
            item   = out["item"]
            result = out["result"]
            col    = out["column"]

            pred  = result["label"]
            true  = item["label"]
            col_activations[col.id] = col_activations.get(col.id, 0) + 1

            score_cache.append((result["ade_score"], result["not_ade_score"], true))

            if pred == pos and true == pos:     tp += 1
            elif pred == pos and true != pos:   fp += 1
            elif pred != pos and true == pos:   fn += 1
            else:                               tn += 1

            if pred != true:
                errors.append({
                    "text": item["text"],
                    "true_label": true,
                    "predicted": pred,
                    "column_id": col.id,
                    "confidence": result["confidence"],
                })

    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)

    return {
        "f1": f1, "precision": prec, "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_evaluated": len(eval_pool),
        "col_activations": col_activations,
        "errors": errors,
    }, score_cache


# ══════════════════════════════════════════════════════════════════════════════════
# §13  LLM GENESIS SYNTHESIS (updated: reads RejectedProposalMemory)
# ══════════════════════════════════════════════════════════════════════════════════

def llm_synthesize_column(
    errors: list[dict],
    existing_columns: list[CorticalColumn],
    llm_fn: Callable,
    round_num: int,
    rejected_memory: Optional[RejectedProposalMemory] = None,
) -> Optional[dict]:
    """
    Ask the LLM to propose a new CorticalColumn from an error cluster.
    v2: LLM reads the FULL rejected proposal history before proposing.
    """
    existing_info = "\n".join(
        f"  {c.id}: trigger={c.trigger_condition!r} (spec={c.compute_specificity()})"
        for c in existing_columns
    )
    error_texts = "\n".join(
        f"  [{c.get('true_label','?')}] {c.get('text','')[:120]}"
        for c in errors[:15]
    )

    # v2: Include rejected proposal history
    rejected_context = ""
    if rejected_memory and len(rejected_memory) > 0:
        rejected_context = f"\n\n{rejected_memory.format_for_genesis_context()}\n"

    system = (
        "You are a computational neuroscience expert designing cortical columns "
        "for a biologically-grounded NLP classifier. Your job is to propose a NEW "
        "specialist column that will correctly handle the given error cases.\n"
        "CRITICAL: Read the rejected proposals carefully. Do NOT repeat patterns that "
        "have already failed. Propose something genuinely different."
    )
    user = (
        f"CURRENT CORTEX — Round {round_num}:\n{existing_info}\n"
        f"{rejected_context}\n"
        f"ERROR CLUSTER ({len(errors)} cases these columns all got wrong):\n{error_texts}\n\n"
        f"AVAILABLE FEATURES (boolean): {', '.join(FEATURE_NAMES)}\n\n"
        f"Design a NEW specialist column. Requirements:\n"
        f"1. trigger_condition: boolean expression using ONLY the listed features\n"
        f"2. Must have ≥2 POSITIVE feature conditions (spec ≥ 2) to be effective\n"
        f"3. Jaccard overlap with existing triggers should be minimal (< 0.15 preferred)\n"
        f"4. Prompt: specialist classification prompt for this column\n\n"
        f"Return ONLY JSON:\n"
        f'{{\n'
        f'  "id": "COL_DESCRIPTIVE_NAME",\n'
        f'  "description": "what this column specializes in",\n'
        f'  "trigger_condition": "feature1 and feature2 and not feature3",\n'
        f'  "prompt": "You are a specialist in [X]. Classify..."\n'
        f'}}'
    )
    try:
        resp = llm_fn(system=system, user=user)
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group())
        if all(k in d for k in ("id", "trigger_condition", "prompt")):
            return d
    except Exception as e:
        print(f"  [Genesis LLM] Error: {e}", file=sys.stderr)
    return None


# ══════════════════════════════════════════════════════════════════════════════════
# §14  HOMEOSTATIC PLASTICITY (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════════

class HomeostaticPlasticity:
    """FiringThreshold calibration — zero LLM cost (Turrigiano et al. 1998)."""

    def __init__(self, initial_threshold: float = 1.0,
                 target: str = "f1", beta: float = 1.0):
        self.firing_threshold = initial_threshold
        self.target = target
        self.beta   = beta
        self.history: list[dict] = []
        self.positive_label = "ADE"

    def calibrate(self, score_cache: list[tuple], round_num: int,
                  n_columns: int = 1, verbose: bool = True) -> float:
        if not score_cache:
            return self.firing_threshold
        pos   = self.positive_label
        beta2 = self.beta ** 2
        target = self.target

        base_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
                           1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0]
        fine = [round(self.firing_threshold * f, 2)
                for f in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3]]
        candidates = sorted(set(base_candidates + fine))

        def _score_at(bias: float) -> dict:
            tp = fp = fn = tn = 0
            for ade_s, nlade_s, true_label in score_cache:
                pred = pos if ade_s >= nlade_s * bias else "NOT_ADE"
                if pred == pos and true_label == pos:     tp += 1
                elif pred == pos and true_label != pos:   fp += 1
                elif pred != pos and true_label == pos:   fn += 1
                else:                                     tn += 1
            prec   = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1     = 2 * prec * recall / max(1e-9, prec + recall)
            fbeta  = (1 + beta2) * prec * recall / max(1e-9, beta2 * prec + recall)
            return {"f1": f1, "recall": recall, "precision": prec,
                    "fbeta": fbeta, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        best_threshold = self.firing_threshold
        best_score     = _score_at(self.firing_threshold).get(target, 0.0)
        for bias in candidates:
            s = _score_at(bias)
            score = s.get(target, 0.0)
            if score > best_score:
                best_score = score
                best_threshold = bias

        self.firing_threshold = best_threshold
        best_stats = _score_at(best_threshold)
        self.history.append({
            "round": round_num, "threshold": best_threshold,
            "f1": best_stats["f1"], "precision": best_stats["precision"],
            "recall": best_stats["recall"],
        })
        if verbose:
            print(f"\n[Homeostatic Calibration] R{round_num} — "
                  f"{len(score_cache)} cached scores, 0 LLM calls", flush=True)
            print(f"  FiringThreshold: {best_threshold:.3f} | "
                  f"F1={best_stats['f1']:.4f} | "
                  f"P={best_stats['precision']:.3f} | R={best_stats['recall']:.3f}",
                  flush=True)
        return best_threshold


# ══════════════════════════════════════════════════════════════════════════════════
# §15  CORTEX TRAINER (updated for all v2 mechanisms)
# ══════════════════════════════════════════════════════════════════════════════════

class CortexTrainer:
    """
    Orchestrates the full developmental training loop.

    v2 additions:
    - MCQ generation after each training round (per-column, BCM-gated)
    - Shadow column warm-up + trigger-scoped probe before full activation
    - RejectedProposalMemory fed to every genesis call
    - MetaAgent diagnostic call on 2+ consecutive F1 declines
    """

    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.round_history: list[dict] = []
        self.genesis_log:   list[dict] = []
        self.pruning_log:   list[dict] = []
        self.meta_log:      list[dict] = []

    def build_llm_fn(self) -> Callable:
        args = self.args
        if args.mock:
            def llm_fn(system: str, user: str) -> str:
                # Inline mock: returns valid JSON for all expected call patterns
                import random as _r
                vote = _r.choice(["ADE", "NOT_ADE"])
                conf = round(_r.uniform(0.55, 0.90), 2)
                return json.dumps({
                    "vote": vote, "classification": vote,
                    "confidence": conf, "reason": "mock",
                    "rationale": "mock rationale",
                    "correct_rationale": "mock correct rationale",
                    "wrong_answers": [
                        {"answer": "NOT_ADE", "explanation": "mock explanation"}
                    ],
                    "diagnosis": "mock diagnosis — F1 declined due to mock data",
                    "interventions": [],
                    "id": "COL_MOCK", "description": "mock",
                    "trigger_condition": "has_induced and not has_negation",
                    "prompt": "You are a mock classifier. Classify as ADE or NOT_ADE.",
                })
            return llm_fn
        elif args.ai_hub:
            client = llm_client.AIHubClient(
                api_key=args.ai_hub_key, ad_object_id=args.ai_hub_ad_id,
            )
            def llm_fn(system: str, user: str) -> str:
                return client.chat(system=system, user=user)
            return llm_fn
        else:
            raise ValueError("Must specify --ai-hub or --mock")

    def build_seed_cortex(self, config) -> Cortex:
        cortex = Cortex()
        seed_nodes = getattr(config, "seed_nodes", [])
        for node_def in seed_nodes:
            if hasattr(node_def, "id"):
                col_id  = node_def.id
                trigger = node_def.trigger or "True"
                prompt  = node_def.prompt
                desc    = node_def.description
            else:
                col_id  = node_def["id"]
                trigger = node_def.get("trigger") or node_def.get("trigger_condition") or "True"
                prompt  = node_def.get("prompt", "")
                desc    = node_def.get("description", "")
            col = CorticalColumn(col_id=col_id, description=desc,
                                 trigger_condition=trigger, prompt=prompt,
                                 genesis_round=0, genesis_phase=LearningPhase.EMBRYONIC)
            cortex.add_column(col)
            print(f"  [Seed] {col.id!r} | spec={col.compute_specificity()} "
                  f"| trigger: {col.trigger_condition}", flush=True)
        return cortex

    def _probe_f1_global(
        self, cortex: Cortex, probe_pool: list[dict],
        rag_index: RAGIndex, llm_fn: Callable,
        homeostatic: HomeostaticPlasticity, config, max_cases: int = 50,
    ) -> float:
        """Global probe F1 (same as v1 — used for pre/post comparison)."""
        sample = random.sample(probe_pool, min(max_cases, len(probe_pool)))
        pos = config.positive_label
        tp = fp = fn = tn = 0
        for item in sample:
            feats = extract_features(item["text"])
            col   = cortex.route(feats)
            try:
                result = classify_with_routes(
                    text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                    column=col, firing_threshold=homeostatic.firing_threshold,
                )
            except Exception as e:
                print(f"  [PROBE ERROR] skipping: {e}", file=sys.stderr)
                continue
            pred = result["label"]; true = item["label"]
            if pred == pos and true == pos:    tp += 1
            elif pred == pos and true != pos:  fp += 1
            elif pred != pos and true == pos:  fn += 1
            else:                              tn += 1
        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        return 2 * prec * recall / max(1e-9, prec + recall)

    def _probe_f1_scoped(
        self, cortex: Cortex, probe_pool: list[dict],
        rag_index: RAGIndex, llm_fn: Callable,
        homeostatic: HomeostaticPlasticity, config,
        trigger: str, column: CorticalColumn,
        fallback_global: bool = True,
    ) -> tuple[float, float, int]:
        """
        Trigger-scoped genesis probe (v2 new).

        Evaluates the new column ONLY on cases that match its trigger,
        comparing its performance to ROOT's baseline on those same cases.

        Returns: (column_f1, root_f1, n_matched_cases)
        Falls back to global probe if fewer than 10 trigger-matched cases.
        """
        pos = config.positive_label

        # Find cases matching the trigger
        matched = []
        for item in probe_pool:
            feats = extract_features(item["text"])
            if safe_eval_condition(trigger, feats):
                matched.append(item)

        if len(matched) < 10:
            # Too few trigger-matched cases — use global probe as fallback
            if fallback_global:
                gf1 = self._probe_f1_global(cortex, probe_pool, rag_index, llm_fn,
                                            homeostatic, config, max_cases=50)
                return gf1, gf1, len(matched)
            return 0.5, 0.5, len(matched)

        # Sample up to 40 matched cases
        sample = random.sample(matched, min(40, len(matched)))
        root_col = cortex.get_column("ROOT")

        col_tp = col_fp = col_fn = col_tn = 0
        root_tp = root_fp = root_fn = root_tn = 0

        for item in sample:
            true = item["label"]
            # New column classification
            try:
                cr = classify_with_routes(
                    text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                    column=column, firing_threshold=homeostatic.firing_threshold,
                )
                pred = cr["label"]
            except Exception:
                pred = "NOT_ADE"
            if pred == pos and true == pos:    col_tp += 1
            elif pred == pos and true != pos:  col_fp += 1
            elif pred != pos and true == pos:  col_fn += 1
            else:                              col_tn += 1

            # ROOT baseline on same cases
            if root_col:
                try:
                    rr = classify_with_routes(
                        text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                        column=root_col, firing_threshold=homeostatic.firing_threshold,
                    )
                    rpred = rr["label"]
                except Exception:
                    rpred = "NOT_ADE"
                if rpred == pos and true == pos:    root_tp += 1
                elif rpred == pos and true != pos:  root_fp += 1
                elif rpred != pos and true == pos:  root_fn += 1
                else:                               root_tn += 1

        def _f1(tp, fp, fn):
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            return 2 * p * r / max(1e-9, p + r)

        return _f1(col_tp, col_fp, col_fn), _f1(root_tp, root_fp, root_fn), len(matched)

    def _generate_mcqs_for_round(
        self,
        cortex: Cortex,
        round_errors: list[dict],
        llm_fn: Callable,
        round_num: int,
        ltp_round: bool,
        max_mcqs_per_round: int = 6,
    ) -> int:
        """
        Generate MCQs from this round's errors and add to per-column MCQ libraries.
        BCM-gated: only generate for columns that actually had errors.
        Returns total MCQs generated.
        """
        total = 0
        # Group errors by column
        by_column: dict[str, list[dict]] = {}
        for err in round_errors:
            cid = err.get("column_id", "ROOT")
            by_column.setdefault(cid, []).append(err)

        for col_id, col_errors in by_column.items():
            col = cortex.get_column(col_id)
            if not col:
                continue
            # BCM-gated: how many MCQs this column earns
            w = col.bcm_state.rehearsal_weight
            n_mcqs = max(1, int(max_mcqs_per_round * w))
            sample = random.sample(col_errors, min(n_mcqs, len(col_errors)))

            for err in sample:
                try:
                    mcq = col.mcq_library.generate_and_add(
                        text=err["text"],
                        true_label=err["true_label"],
                        predicted_label=err["predicted"],
                        llm_fn=llm_fn,
                        round_num=round_num,
                        ltp_round=ltp_round,
                    )
                    if mcq:
                        total += 1
                except Exception as e:
                    print(f"  [MCQ] Generation failed for {col_id}: {e}", file=sys.stderr)

        return total

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
        rejected_memory: RejectedProposalMemory,
        meta_agent: MetaAgent,
        config,
    ) -> dict:
        """Execute one full training round — all v2 mechanisms."""

        phase = phase_for_round(round_num)
        print(f"\n{'═'*80}", flush=True)
        print(f"  ROUND {round_num} | Phase: {phase.value.upper()} | "
              f"Columns: {len(cortex.columns)} "
              f"({len(cortex.shadow_columns)} shadow) | "
              f"FiringThreshold: {homeostatic.firing_threshold:.3f}", flush=True)
        print(f"  CriticalPeriod: {critical_period.describe(round_num)}", flush=True)
        print(f"{'═'*80}", flush=True)

        # ── Shadow column warm-up ──────────────────────────────────────────────────
        # Shadow columns observe this round's training cases before activation
        for col in cortex.shadow_columns:
            print(f"  [Shadow] '{col.id}' observing round {round_num} "
                  f"(shadow since R{col.shadow_round})", flush=True)

        # ── Training batch classification ──────────────────────────────────────────
        print(f"\n[Training] Classifying {len(train_batch)} cases...", flush=True)
        _load_embedder()  # warm up embedding model
        all_errors: list[dict] = []
        col_route_counts: dict[str, int] = {}

        def _train_one(item: dict) -> dict:
            feats = extract_features(item["text"])
            # Shadow observation
            cortex.observe_shadow(feats, item)
            col = cortex.route(feats)
            try:
                result = classify_with_routes(
                    text=item["text"], rag_index=rag_index, llm_fn=llm_fn,
                    column=col, firing_threshold=homeostatic.firing_threshold,
                )
            except Exception:
                result = {"label": "NOT_ADE", "confidence": 0.1,
                          "ade_score": 0.0, "not_ade_score": 1.0, "column_id": col.id,
                          "route_results": [], "agreement": 0.0, "split": False}
            return {"item": item, "result": result, "column": col}

        with ThreadPoolExecutor(max_workers=self.args.workers) as ex:
            futs = [ex.submit(_train_one, item) for item in train_batch]
            for fut in futs:
                try:
                    out = fut.result(timeout=90)
                except Exception:
                    continue
                item   = out["item"]
                result = out["result"]
                col    = out["column"]
                col_route_counts[col.id] = col_route_counts.get(col.id, 0) + 1
                if result["label"] != item["label"]:
                    all_errors.append({
                        "text": item["text"],
                        "true_label": item["label"],
                        "predicted": result["label"],
                        "column_id": col.id,
                    })

        print(f"\n[Training] Errors: {len(all_errors)}/{len(train_batch)} "
              f"({100*len(all_errors)/len(train_batch):.1f}%)", flush=True)

        # ── BCM update ────────────────────────────────────────────────────────────
        print(f"\n[BCM] Updating plasticity states...", flush=True)
        ltp_this_round = False
        col_error_counts: dict[str, int] = {}
        for e in all_errors:
            col_error_counts[e["column_id"]] = col_error_counts.get(e["column_id"], 0) + 1

        for col in cortex.columns:
            routes = col_route_counts.get(col.id, 0)
            event  = col.bcm_update(routes, len(train_batch), round_num)
            col.f1_history.append(0.0)  # Placeholder; updated after eval
            n_err  = col_error_counts.get(col.id, 0)
            print(f"  {col.id:35s} routes={routes:4d} | BCM={event:6s} | "
                  f"θ_M={col.bcm_state.theta_m:.4f} | errors={n_err:3d} | "
                  f"rehearsal_w={col.bcm_state.rehearsal_weight:.2f}", flush=True)
            if event == "LTP":
                ltp_this_round = True

        # ── MCQ generation (v2 new) ───────────────────────────────────────────────
        if all_errors:
            print(f"\n[MCQ] Generating contrastive lessons from {len(all_errors)} errors...",
                  flush=True)
            n_mcqs = self._generate_mcqs_for_round(
                cortex=cortex, round_errors=all_errors, llm_fn=llm_fn,
                round_num=round_num, ltp_round=ltp_this_round,
            )
            print(f"  {n_mcqs} MCQs generated across {len(cortex.columns)} columns",
                  flush=True)

        # ── Shadow column probe + activation (v2 new) ─────────────────────────────
        for col in list(cortex.shadow_columns):
            if round_num > col.shadow_round:
                # Warm-up refinement of shadow column using observed cases
                if col.shadow_cases:
                    print(f"\n[Shadow → Active] Probing '{col.id}' on "
                          f"{len(col.shadow_cases)} observed cases...", flush=True)

                # Trigger-scoped probe
                col_f1, root_f1, n_matched = self._probe_f1_scoped(
                    cortex=cortex, probe_pool=probe_pool,
                    rag_index=rag_index, llm_fn=llm_fn,
                    homeostatic=homeostatic, config=config,
                    trigger=col.trigger_condition, column=col,
                )
                delta = col_f1 - root_f1
                print(f"  [Scoped Probe] '{col.id}': col_F1={col_f1:.4f} vs "
                      f"ROOT_F1={root_f1:.4f} Δ={delta:+.4f} "
                      f"({n_matched} trigger-matched cases)", flush=True)

                SHADOW_ROLLBACK = -0.05
                if delta < SHADOW_ROLLBACK:
                    with cortex._lock:
                        if col in cortex._columns:
                            cortex._columns.remove(col)
                    print(f"  [Shadow ROLLED BACK] '{col.id}' — "
                          f"scoped Δ={delta:+.4f} < {SHADOW_ROLLBACK}", flush=True)
                    rejected_memory.log(
                        col_id=col.id, trigger=col.trigger_condition,
                        spec=col.compute_specificity(), probe_delta=delta,
                        round_num=round_num, jaccard=0.0,
                        reason=f"Trigger-scoped probe: col_F1={col_f1:.4f} vs ROOT_F1={root_f1:.4f}",
                    )
                    self.genesis_log.append({
                        "round": round_num, "phase": phase.value,
                        "col_id": col.id, "scoped_delta": delta,
                        "col_f1": col_f1, "root_f1": root_f1,
                        "n_matched": n_matched, "rolled_back": True,
                        "probe_type": "trigger_scoped",
                    })
                else:
                    cortex.activate_shadow_column(col.id)
                    print(f"  [Shadow ACTIVATED] '{col.id}' — "
                          f"scoped Δ={delta:+.4f} ≥ {SHADOW_ROLLBACK}", flush=True)
                    self.genesis_log.append({
                        "round": round_num, "phase": phase.value,
                        "col_id": col.id, "scoped_delta": delta,
                        "col_f1": col_f1, "root_f1": root_f1,
                        "n_matched": n_matched, "rolled_back": False,
                        "probe_type": "trigger_scoped",
                    })

        # ── Prompt refinement ─────────────────────────────────────────────────────
        refined = 0
        cols_to_refine = cortex.active_columns
        if phase == LearningPhase.CONSOLIDATION:
            # CONSOLIDATION: only refine below-average columns
            avg_f1 = (sum(c.bcm_state.theta_m for c in cols_to_refine) /
                      max(1, len(cols_to_refine)))
            cols_to_refine = [c for c in cols_to_refine
                              if c.bcm_state.theta_m <= avg_f1 or c.id == "ROOT"]

        print(f"\n[Refinement] Improving column prompts...", flush=True)
        for col in cols_to_refine:
            col_errors = [e for e in all_errors if e["column_id"] == col.id]
            if not col_errors:
                continue
            err_context = "\n".join(
                f"  [{e['true_label']}→{e['predicted']}] {e['text'][:100]}"
                for e in col_errors[:8]
            )
            mcq_ctx = col.mcq_library.format_for_prompt(bcm_state=col.bcm_state)
            system = (
                f"You are refining the prompt for column '{col.id}' in a cortical NLP classifier. "
                f"Use the error cases and MCQ lessons to improve the prompt."
            )
            user = (
                f"Current prompt (v{col.prompt_version}):\n{col.prompt}\n\n"
                f"Errors this round:\n{err_context}\n\n"
                f"{mcq_ctx}\n\n"
                f"Provide an improved prompt. Return ONLY the improved prompt text."
            )
            try:
                new_prompt = llm_fn(system=system, user=user)
                if new_prompt and len(new_prompt) > 50:
                    col.prompt = new_prompt.strip()
                    col.prompt_version += 1
                    print(f"  [Refined] '{col.id}' (v{col.prompt_version})", flush=True)
                    # Distill key learning into a MemoryTrace
                    system2 = "Extract the single most important new insight from this prompt revision."
                    user2   = (f"Old prompt: {col.prompt[:200]}\n"
                               f"New prompt: {new_prompt[:200]}\n"
                               f"Key errors: {err_context[:200]}\n"
                               f"In one sentence, what did we learn?")
                    try:
                        insight = llm_fn(system=system2, user=user2)
                        if insight and len(insight) > 10:
                            col.add_memory_trace(
                                text=insight.strip()[:300],
                                consolidation_score=col.bcm_state.rehearsal_weight,
                                round_num=round_num,
                            )
                            print(f"  [MemoryTrace] '{col.id}': {insight[:70]}...", flush=True)
                    except Exception:
                        pass
                    refined += 1
            except Exception as e:
                print(f"  [Refinement Error] {col.id}: {e}", file=sys.stderr)

        print(f"  {refined} columns refined", flush=True)

        # ── Meta-Agent diagnostic (v2 new) ────────────────────────────────────────
        f1_history_so_far = [r["f1"] for r in self.round_history]
        if (phase != LearningPhase.EMBRYONIC and
                meta_agent.should_trigger(f1_history_so_far, round_num)):
            print(f"\n[MetaAgent] F1 declining — firing diagnostic call...", flush=True)
            # Get current eval metrics (approximate from training errors)
            n_fp_approx = sum(1 for e in all_errors if e["true_label"] == "NOT_ADE")
            n_fn_approx = sum(1 for e in all_errors if e["true_label"] == "ADE")
            diagnosis = meta_agent.diagnose(
                cortex_summary=cortex.summary(),
                f1_history=f1_history_so_far,
                rejected_proposals=rejected_memory,
                error_cases=all_errors,
                round_num=round_num,
                n_fp=n_fp_approx,
                n_fn=n_fn_approx,
                llm_fn=llm_fn,
            )
            print(f"  [MetaAgent Diagnosis] {diagnosis.get('diagnosis','')[:200]}",
                  flush=True)
            interventions = diagnosis.get("interventions", [])
            for iv in interventions:
                print(f"  [MetaAgent Intervention] type={iv.get('type')} "
                      f"target={iv.get('target')} | {iv.get('guidance','')[:100]}",
                      flush=True)
                # Apply threshold interventions immediately
                if iv.get("type") == "threshold_up":
                    homeostatic.firing_threshold *= 1.2
                    print(f"    → FiringThreshold raised to {homeostatic.firing_threshold:.3f}",
                          flush=True)
                elif iv.get("type") == "threshold_down":
                    homeostatic.firing_threshold *= 0.85
                    print(f"    → FiringThreshold lowered to {homeostatic.firing_threshold:.3f}",
                          flush=True)
            self.meta_log.append(diagnosis)

        # ── EnggramCluster → Genesis ───────────────────────────────────────────────
        genesis_approved = False
        if phase != LearningPhase.CONSOLIDATION:
            print(f"\n[EnggramCluster] Detecting LTP events...", flush=True)
            clusters = detect_engram_clusters(
                error_cases=all_errors, min_cluster_size=5,
            )
            if clusters:
                largest = clusters[0]
                print(f"  Largest EnggramCluster: {len(largest.cases)} cases | "
                      f"coherence={largest.coherence:.3f}", flush=True)

                proposal = llm_synthesize_column(
                    errors=largest.cases,
                    existing_columns=cortex.columns,
                    llm_fn=llm_fn,
                    round_num=round_num,
                    rejected_memory=rejected_memory,
                )

                if proposal:
                    print(f"  [LTP Event] Proposed: {proposal.get('id','?')} | "
                          f"trigger: {proposal.get('trigger_condition','?')}", flush=True)

                    overlap = cortex.jaccard_overlap(
                        proposal.get("trigger_condition", ""),
                        probe_pool[:100],
                    )
                    j = overlap["max_jaccard"]
                    print(f"  [Jaccard Audit] max_overlap={j:.3f} "
                          f"(worst={overlap['worst_col']!r})", flush=True)

                    approved, reason = cortex.columnar_genesis(
                        proposal=proposal,
                        probe_cases=probe_pool[:100],
                        critical_period=critical_period,
                        round_num=round_num,
                        max_jaccard=0.50,
                    )

                    if approved:
                        print(f"  [Genesis ✓] '{proposal['id']}' entered SHADOW MODE | "
                              f"round={round_num} | will probe next round", flush=True)
                        genesis_approved = True
                        # Log tentatively — final genesis logged after shadow probe
                        self.genesis_log.append({
                            "round": round_num, "phase": phase.value,
                            "col_id": proposal["id"], "jaccard": j,
                            "status": "shadow", "rolled_back": None,
                        })
                    else:
                        print(f"  [Genesis Rejected] {reason}", flush=True)
                        # Log rejected proposals for future LLM context
                        tmp = CorticalColumn("__tmp__", "", proposal.get("trigger_condition",""), "")
                        rejected_memory.log(
                            col_id=proposal.get("id","?"),
                            trigger=proposal.get("trigger_condition",""),
                            spec=tmp.compute_specificity(),
                            probe_delta=-0.0,
                            round_num=round_num,
                            jaccard=j,
                            reason=f"Rejected at genesis: {reason}",
                        )
            else:
                print(f"  No significant EnggramClusters (errors={len(all_errors)}, threshold=5)",
                      flush=True)

        # ── Synaptic pruning ───────────────────────────────────────────────────────
        if phase != LearningPhase.EMBRYONIC:
            pruned = cortex.synaptic_pruning(round_num)
            if pruned:
                self.pruning_log.append({
                    "round": round_num, "pruned": [c.id for c in pruned],
                })

        # ── Evaluation ─────────────────────────────────────────────────────────────
        print(f"\n[Evaluation] Evaluating on {len(eval_pool)} cases...", flush=True)
        metrics, score_cache = evaluate_cortex(
            cortex=cortex, eval_pool=eval_pool, rag_index=rag_index,
            llm_fn=llm_fn, homeostatic=homeostatic, config=config,
            round_num=round_num, max_workers=self.args.workers,
        )

        # Update F1 history on columns
        for col in cortex.columns:
            if col.f1_history:
                col.f1_history[-1] = metrics["f1"]

        # ── Homeostatic calibration ────────────────────────────────────────────────
        homeostatic.calibrate(score_cache, round_num, n_columns=len(cortex.columns))

        # ── Round summary ──────────────────────────────────────────────────────────
        round_result = {
            "round": round_num, "phase": phase.value,
            "f1": metrics["f1"], "precision": metrics["precision"],
            "recall": metrics["recall"],
            "tp": metrics["tp"], "fp": metrics["fp"],
            "fn": metrics["fn"], "tn": metrics["tn"],
            "n_evaluated": metrics["n_evaluated"],
            "n_columns": len(cortex.columns),
            "n_active": len(cortex.active_columns),
            "n_shadow": len(cortex.shadow_columns),
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

    def run(self, config) -> None:
        """Main entry — full v2 developmental training loop."""

        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS CORTEX v2.0 — Guided Developmental Cortical NLP Classifier",
              flush=True)
        print(f"  Task: {config.task_name}", flush=True)
        print(f"  Output: {self.out_dir}/", flush=True)
        print(f"{'╚'+'═'*78+'╝'}\n", flush=True)

        llm_fn = self.build_llm_fn()

        # API sanity check
        if not self.args.mock:
            print("[Startup] Verifying LLM API...", flush=True)
            try:
                test = llm_fn("Say OK", "Reply with just the word OK")
                print(f"  API OK: {test[:30]!r}", flush=True)
            except Exception as e:
                print(f"  [FATAL] API test failed: {e}", file=sys.stderr)
                sys.exit(1)

        # Data loading
        print("\n[Data] Loading corpus...", flush=True)
        eval_pool, probe_pool, train_pool = data_utils.load_and_split(
            seed=self.args.seed, eval_size=200, probe_size=300,
        )
        print(f"  Corpus: eval={len(eval_pool)} | probe={len(probe_pool)} | "
              f"train={len(train_pool)}", flush=True)
        print(f"  ADE: eval={sum(1 for x in eval_pool if x['label']=='ADE')}, "
              f"NOT_ADE={sum(1 for x in eval_pool if x['label']!='ADE')}", flush=True)

        # RAG index
        rag_dir = str(self.out_dir / "rag_index")
        print(f"\n[RAG] Building/loading FAISS index at {rag_dir}...", flush=True)
        if self.args.fresh or not (Path(rag_dir) / "faiss.index").exists():
            rag_index = RAGIndex.build(train_pool, out_dir=rag_dir)
        else:
            rag_index = RAGIndex.load(rag_dir)

        # Cortex initialization
        cortex_path    = str(self.out_dir / "cortex_state.json")
        rejected_path  = str(self.out_dir / "rejected_proposals.json")
        saved_threshold  = 1.723
        completed_rounds = 0

        if not self.args.fresh and Path(cortex_path).exists():
            print(f"\n[Cortex] Loading from {cortex_path}...", flush=True)
            saved_meta = json.loads(Path(cortex_path).read_text())
            saved_threshold  = saved_meta.get("firing_threshold", 1.723)
            completed_rounds = saved_meta.get("completed_rounds", 0)
            cortex = Cortex.load(cortex_path)
            print(f"  Warm restart: round {completed_rounds + 1} | "
                  f"FiringThreshold={saved_threshold:.3f}", flush=True)
        else:
            print(f"\n[Cortex] Building seed cortex...", flush=True)
            cortex = self.build_seed_cortex(config)

        # Load rejected proposal memory
        rejected_memory = RejectedProposalMemory()
        if not self.args.fresh and Path(rejected_path).exists():
            try:
                data = json.loads(Path(rejected_path).read_text())
                rejected_memory = RejectedProposalMemory.from_list(data)
                print(f"  Loaded {len(rejected_memory)} rejected proposals from disk",
                      flush=True)
            except Exception:
                pass

        print(f"\n{cortex.summary()}\n", flush=True)

        # Controllers
        homeostatic = HomeostaticPlasticity(
            initial_threshold=saved_threshold,
            target=getattr(config, "calibration_target", "f1"),
        )
        critical_period = CriticalPeriod(T_min=0.60, T_max=0.85, tau=5.0)
        meta_agent = MetaAgent(consecutive_decline_threshold=2)

        # Training loop
        random.seed(self.args.seed)
        start_round = completed_rounds + 1
        end_round   = completed_rounds + self.args.rounds

        for round_num in range(start_round, end_round + 1):
            batch_size = getattr(config, "batch_size", 250)
            batch = random.sample(train_pool, min(batch_size, len(train_pool)))

            result = self.run_training_round(
                round_num=round_num, cortex=cortex,
                train_batch=batch, eval_pool=eval_pool, probe_pool=probe_pool,
                rag_index=rag_index, llm_fn=llm_fn, homeostatic=homeostatic,
                critical_period=critical_period, rejected_memory=rejected_memory,
                meta_agent=meta_agent, config=config,
            )
            self.round_history.append(result)

            # Save state
            cortex.save(cortex_path, meta={
                "firing_threshold": homeostatic.firing_threshold,
                "completed_rounds": round_num,
            })
            # Save rejected proposals separately
            Path(rejected_path).write_text(
                json.dumps(rejected_memory.to_list(), indent=2)
            )
            self._save_run_log()

        self._print_final_report(cortex, rejected_memory)

    def _save_run_log(self) -> None:
        log_path = self.out_dir / "cortex_run_log.json"
        log_path.write_text(json.dumps({
            "rounds": self.round_history,
            "genesis_log": self.genesis_log,
            "pruning_log": self.pruning_log,
            "meta_log": self.meta_log,
        }, indent=2))

    def _print_final_report(self, cortex: Cortex,
                            rejected_memory: RejectedProposalMemory) -> None:
        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS CORTEX v2.0 — FINAL REPORT", flush=True)
        print(f"{'╚'+'═'*78+'╝'}", flush=True)
        print(f"\n  Total rounds: {len(self.round_history)}", flush=True)
        if self.round_history:
            best = max(self.round_history, key=lambda r: r["f1"])
            last = self.round_history[-1]
            print(f"  Best F1:  {best['f1']:.4f} at R{best['round']} "
                  f"({best['n_columns']} columns, {best['phase']})", flush=True)
            print(f"  Final F1: {last['f1']:.4f} at R{last['round']} "
                  f"({last['n_columns']} columns)", flush=True)
        print(f"\n  Genesis events: {len(self.genesis_log)}", flush=True)
        print(f"  Pruning events: {len(self.pruning_log)}", flush=True)
        print(f"  Meta-agent calls: {len(self.meta_log)}", flush=True)
        print(f"  Rejected proposals: {len(rejected_memory)}", flush=True)
        print(f"\n{cortex.summary()}", flush=True)
        print(f"\n  F1 Trajectory:", flush=True)
        for r in self.round_history:
            bar = "█" * int(r["f1"] * 40)
            shadow = f" [{r.get('n_shadow',0)}s]" if r.get("n_shadow") else ""
            print(f"    R{r['round']:02d} [{r['phase'][:3].upper()}] "
                  f"{r['f1']:.4f} {bar} | cols={r['n_columns']}{shadow} | "
                  f"T={r['firing_threshold']:.3f}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════════
# §16  CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NEXUS Cortex v2.0")
    parser.add_argument("--config",       required=True)
    parser.add_argument("--out",          required=True)
    parser.add_argument("--rounds",       type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--workers",      type=int, default=4)
    parser.add_argument("--fresh",        action="store_true")
    parser.add_argument("--mock",         action="store_true")
    parser.add_argument("--ai-hub",       action="store_true")
    parser.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY",""))
    parser.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID",""))
    args = parser.parse_args()

    config = TaskConfig.load(args.config)
    trainer = CortexTrainer(args)
    trainer.run(config)


if __name__ == "__main__":
    main()

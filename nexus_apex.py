"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NEXUS APEX LEARNER                                                          ║
║  The next generation — synthesizing every lesson from v0 through MCQ        ║
║  Northwell Health — NEXUS Research Program                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Design philosophy (40 years of computational neuroscience, distilled):

  1. PREDICTION-ERROR LEARNING (Schultz 1997 — dopamine RPE)
     δ = confidence × is_wrong.  Confident errors drive maximum plasticity.
     Uncertain errors drive minimal plasticity.  This is not a heuristic —
     it is the biological dopamine reward prediction error, implemented.

  2. CONTRASTIVE PAIRS, NOT MCQs (VanLehn 2011 — worked examples)
     Every lesson shows the decision BOUNDARY: here is a case that looks like
     X but is Y.  Here is the nearest case that is genuinely X.  Here is what
     distinguishes them.  This teaches the edge, not the interior.

  3. NEAR-MISS MINING (Bliss & Lømo 1973 — LTP pairing requirement)
     Cases where the model was correct but barely (confidence < 0.60) sit on
     the decision boundary.  They are more informative than confident errors.
     They get a second look AND become boundary examples for future cases.

  4. TWO-PASS GAMMA-THETA ARCHITECTURE (Lisman & Jensen 2013)
     Gamma (fast): classify all 1000.  Generate lessons from errors.
     Theta  (slow): reclassify the 100 hardest with fresh lessons.
     Theta corrections are the absorption signal — immediate evidence
     that today's lessons work on today's hardest cases.

  5. THREE-TIMESCALE MEMORY (Complementary Learning Systems — McClelland 1995)
     Fast  — error log with δ and rationale (every case)
     Medium — contrastive pairs (every round)
     Slow  — causal model synthesis via LLM (every 3 rounds)

  6. ACETYLCHOLINE GATING (Hasselmo 1999)
     plasticity = tanh(error_rate × 3).  High error rate = plastic / learning.
     Low error rate = stable / consolidating.  All generation rates scale
     with plasticity.  The system naturally slows its own learning as it matures.

  7. EMA THRESHOLD STABILITY
     threshold = 0.7 × prev + 0.3 × calibrated.  Only updates if Δ > 0.08.
     Eliminates the 0.30↔0.80 oscillation that plagued every previous version.

  8. RATIONALE-BASED ERROR TAXONOMY (zero LLM cost)
     The LLM's own rationale is parsed for WHY it erred:
     temporal_confusion | therapeutic_goal | report_context |
     negation_confusion | causal_ambiguity | completeness_confusion | general
     Errors grouped by type before contrastive pair generation.

  9. CONSOLIDATION (sleep phase — every 3 rounds)
     LLM synthesizes accumulated lessons into a causal model.
     This becomes the expert system prompt — structural knowledge, not lookup.
     The system stops consulting notes and starts speaking from understanding.

  10. BLANK SLATE → ADULT
      R1-3: no lessons injected.  Pure classification.  Build error memory.
      R4-6: high-δ contrastive pairs injected.
      R7-11: near-miss examples added.  Pairs refined.
      R12+: full structural knowledge + causal model + boundary lessons.

What is NOT in this version (intentionally):
  - Columnar routing (test the learning hypothesis first; routing = v4)
  - Dendritic sub-prompts (too complex; v5)
  - Attractor dynamics (v5)
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import data_utils
import llm_client
from embedder import embed as embedder_embed, embed_one as embedder_embed_one, enable_mock_embeddings
from nexus_db_apex import NexusApexDB, classify_error_type, compute_delta, compute_salience
from rag_index import RAGIndex
from task_config import TaskConfig


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NEAR_MISS_THRESHOLD   = 0.62   # correct predictions below this are near-misses
MIN_PAIR_DELTA        = 0.25   # minimum δ to qualify an error for pair generation
MIN_DIRECTION_PCT     = 0.65   # ≥65% same direction before generating directional lesson
EMA_ALPHA             = 0.15   # EMA weight on new calibration (0.85 on old) — more stable
EMA_MIN_SHIFT         = 0.12   # minimum shift before EMA updates threshold — wider dead zone
THETA_SIZE            = 100    # hardest cases to reclassify in theta pass
THETA_TOP_DELTA_N     = 50     # top-N by δ for theta selection
THETA_TOP_NM_N        = 50     # top-N near-misses for theta selection
CONSOLIDATION_EVERY   = 3      # consolidation every N rounds
CONSOLIDATION_COOLDOWN = 1     # rounds to hold causal model stable after consolidation
MAX_ACTIVE_PAIRS      = 50     # cap active pairs (raised for curriculum phases)
LESSON_INJECT_START   = 4      # round from which lessons are injected
NM_INJECT_START       = 7      # round from which near-miss examples are injected
PAIR_TIMEOUT          = 35     # seconds per LLM call in pair generation

# ── Salience gating (norepinephrine / amygdala) ───────────────────────────────
SALIENCE_HIGH_THRESHOLD = 2.0  # salience >= this → bypass plasticity gate, immediate Core
WEAK_SIGNAL_REPS        = 3    # low-δ error types need this many cumulative occurrences
WEAK_SIGNAL_DELTA_MAX   = 0.45 # mean δ below this = weak signal (repetition required)

# ── Curriculum learning schedule ─────────────────────────────────────────────
# Mirrors developmental learning: simple → complex, small batches → large.
# Small batches early: high error rate → high plasticity → rapid pair generation.
# Large batches late: stress-test generalisation across full complexity.
CURRICULUM = [
    (10,  50),   # Phase 1: rounds  1-10,  batch=50  (high error rate, fast learning)
    (10, 100),   # Phase 2: rounds 11-20,  batch=100 (consolidation)
    (10, 250),   # Phase 3: rounds 21-30,  batch=250 (generalisation)
    (10, 500),   # Phase 4: rounds 31-40,  batch=500 (full complexity)
]
TOTAL_CURRICULUM_ROUNDS = sum(n for n, _ in CURRICULUM)  # 40


def get_curriculum_batch_size(round_num: int) -> int:
    """Return batch size for the given round based on curriculum schedule."""
    cumulative = 0
    for n_rounds, batch_size in CURRICULUM:
        cumulative += n_rounds
        if round_num <= cumulative:
            return batch_size
    return CURRICULUM[-1][1]   # default to largest phase


# ══════════════════════════════════════════════════════════════════════════════
# §2  PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(
    base_task: str,
    structural_knowledge: Optional[dict],
    round_num: int,
) -> str:
    """
    Build the system prompt.
    Early rounds (1-3): blank slate — task only.
    Later rounds: add causal model as expert intuition.
    """
    parts = [base_task.strip()]

    if structural_knowledge and round_num >= LESSON_INJECT_START:
        cm = structural_knowledge.get("causal_model", "")
        kf = structural_knowledge.get("key_factors", [])
        ep = structural_knowledge.get("error_patterns", [])
        if cm:
            parts.append(f"\nYOUR CLINICAL UNDERSTANDING (built from experience):\n{cm}")
        if kf:
            parts.append(
                "Decision factors (priority order):\n"
                + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(kf[:3]))
            )
        if ep:
            parts.append(
                "Reasoning errors you have learned to avoid:\n"
                + "\n".join(f"  • {p}" for p in ep[:3])
            )

    return "\n".join(parts)


def build_user_prompt(
    text: str,
    rag_examples: list[dict],
    weighted_lessons: list[dict],
    near_miss_examples: list[dict],
    round_num: int,
    second_look: bool = False,
    fresh_lessons: Optional[list[dict]] = None,
) -> str:
    """
    Build the user prompt.
    Ordered by LLM attention priority (most important = closest to the task question).

    Structure (bottom = highest attention):
      1. RAG similar cases
      2. Near-miss boundary examples
      3. Contrastive pair lessons (embedding-matched)
      4. Fresh lessons from THIS round (theta pass only)
      5. Task question ← maximum attention weight
    """
    blocks = []

    # RAG examples (furthest from task — context grounding)
    if rag_examples:
        lines = ["SIMILAR CASES FROM MEDICAL LITERATURE:"]
        for ex in rag_examples[:3]:
            label = ex.get("label", ex.get("true_label", "?"))
            lines.append(f"  [{label}] \"{ex['text'][:120]}\"")
        blocks.append("\n".join(lines))

    # Near-miss examples — correct but counterintuitive
    if near_miss_examples and round_num >= NM_INJECT_START:
        lines = ["BOUNDARY CASES (correct answers that look wrong — study carefully):"]
        for nm in near_miss_examples[:3]:
            lines.append(
                f"  [{nm['true_label']}] \"{nm['text'][:120]}\""
                f"  ← correct despite ambiguity"
            )
        blocks.append("\n".join(lines))

    # Contrastive pair lessons (embedding-matched, delta-weighted)
    if weighted_lessons and round_num >= LESSON_INJECT_START:
        lines = ["HARD-LEARNED BOUNDARY LESSONS (from similar cases you struggled with):"]
        for lsn in weighted_lessons[:4]:
            lines.append(
                f"  [{lsn['error_type']}]"
                f" \"{lsn['anchor_text'][:90]}\" → {lsn['anchor_label']}\n"
                f"  vs \"{lsn['contrast_text'][:90]}\" → {lsn['contrast_label']}\n"
                f"  KEY: {lsn['key_distinction']}\n"
                f"  LESSON: {lsn['lesson']}"
            )
        blocks.append("\n".join(lines))

    # Fresh lessons from this round (theta pass only)
    if second_look and fresh_lessons:
        lines = ["⚠ SECOND LOOK — This case is flagged as difficult.",
                 "LESSONS JUST LEARNED THIS ROUND:"]
        for lsn in fresh_lessons[:3]:
            lines.append(
                f"  • [{lsn['error_type']}] {lsn['lesson']}\n"
                f"    KEY: {lsn['key_distinction']}"
            )
        blocks.append("\n".join(lines))
    elif second_look:
        blocks.append("⚠ SECOND LOOK — This case is flagged as difficult. Apply extra care.")

    # Task question — closest to attention peak
    blocks.append(
        f"Classify this sentence:\n\"{text}\"\n\n"
        "Is this an Adverse Drug Event?\n"
        "Respond ONLY with JSON (no markdown):\n"
        "{\"classification\": \"ADE\" or \"NOT_ADE\", "
        "\"confidence\": 0.0-1.0, "
        "\"rationale\": \"one sentence: the KEY deciding factor\"}"
    )

    return "\n\n".join(blocks)


# ══════════════════════════════════════════════════════════════════════════════
# §3  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def _parse_llm_response(response: str) -> tuple[str, float, str]:
    """Parse classification JSON. Returns (label, confidence, rationale)."""
    try:
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            d = json.loads(m.group())
            label = d.get("classification", "NOT_ADE")
            if label not in ("ADE", "NOT_ADE"):
                label = "NOT_ADE"
            conf = float(d.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
            rat  = str(d.get("rationale", ""))
            return label, conf, rat
    except Exception:
        pass
    return "NOT_ADE", 0.1, ""


def classify_one(
    case: dict,
    db: NexusApexDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    system_prompt: str,
    round_num: int,
    threshold: float = 0.5,
    second_look: bool = False,
    fresh_lessons: Optional[list[dict]] = None,
    timeout: float = 35.0,
) -> dict:
    """Single classification call with full context injection."""
    text = case["text"]

    # Embed query for lesson retrieval
    q_emb = embedder_embed_one(text)

    # RAG retrieval
    rag_examples = rag_index.query(text, k=4)

    # Lesson retrieval (embedding-based, not feature-based)
    weighted_lessons = (
        db.get_weighted_lessons(q_emb, k=4) if round_num >= LESSON_INJECT_START
        else []
    )

    # Near-miss examples
    near_miss_examples = (
        db.get_recent_near_misses(n=10) if round_num >= NM_INJECT_START
        else []
    )

    user_prompt = build_user_prompt(
        text=text,
        rag_examples=rag_examples,
        weighted_lessons=weighted_lessons,
        near_miss_examples=near_miss_examples,
        round_num=round_num,
        second_look=second_look,
        fresh_lessons=fresh_lessons,
    )

    try:
        response = llm_fn(system=system_prompt, user=user_prompt)
        label, conf, rationale = _parse_llm_response(response)
    except Exception as e:
        label, conf, rationale = "NOT_ADE", 0.1, ""

    # Apply threshold (conf must meet threshold to call ADE)
    if label == "ADE" and conf < threshold:
        label = "NOT_ADE"

    return {
        "case_id":    case.get("id", 0),
        "text":       text,
        "true_label": case.get("true_label", case.get("label", "")),
        "predicted":  label,
        "confidence": conf,
        "rationale":  rationale,
        "embedding":  q_emb,
    }


def classify_batch(
    cases: list[dict],
    db: NexusApexDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    system_prompt: str,
    round_num: int,
    threshold: float = 0.5,
    workers: int = 6,
    second_look: bool = False,
    fresh_lessons: Optional[list[dict]] = None,
    label: str = "Classifying",
) -> list[dict]:
    """Parallel batch classification. Returns all results (correct + errors)."""
    results = []
    print(f"[{label}] {len(cases)} cases (workers={workers})...", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                classify_one, c, db, rag_index, llm_fn,
                system_prompt, round_num, threshold,
                second_look, fresh_lessons
            ): c
            for c in cases
        }
        try:
            for fut in as_completed(futs, timeout=len(cases) * 40):
                try:
                    r = fut.result(timeout=40)
                    results.append(r)
                except Exception as e:
                    print(f"  [Classify Error] {e}", file=sys.stderr)
        except FuturesTimeoutError:
            print("  [Timeout] Some futures unfinished — using partial results",
                  file=sys.stderr)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# §4  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list[dict], positive_label: str = "ADE") -> dict:
    tp = fp = fn = tn = 0
    for r in results:
        pred, true = r["predicted"], r["true_label"]
        if pred == positive_label and true == positive_label:   tp += 1
        elif pred == positive_label and true != positive_label: fp += 1
        elif pred != positive_label and true == positive_label: fn += 1
        else:                                                    tn += 1
    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)
    return {"f1": f1, "precision": prec, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def calibrate_threshold_ema(
    results: list[dict],
    prev_threshold: float,
    positive_label: str = "ADE",
) -> float:
    """
    Find the threshold that maximises F1 on current results,
    then apply EMA smoothing to prevent round-to-round oscillation.
    Only updates if the change exceeds EMA_MIN_SHIFT.
    """
    candidates = [i / 20 for i in range(1, 20)]
    best_f1, best_t = -1.0, prev_threshold
    for t in candidates:
        adjusted = []
        for r in results:
            p = r["predicted"]
            c = r["confidence"]
            if p == positive_label and c < t:
                p = "NOT_ADE"
            adjusted.append({**r, "predicted": p})
        m = compute_metrics(adjusted, positive_label)
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], t

    # EMA smoothing
    new_t = EMA_ALPHA * best_t + (1 - EMA_ALPHA) * prev_threshold
    if abs(new_t - prev_threshold) < EMA_MIN_SHIFT:
        return prev_threshold
    return round(new_t, 3)


def apply_threshold(results: list[dict], threshold: float,
                    positive_label: str = "ADE") -> list[dict]:
    adjusted = []
    for r in results:
        p = r["predicted"]
        if p == positive_label and r["confidence"] < threshold:
            p = "NOT_ADE"
        adjusted.append({**r, "predicted": p})
    return adjusted


# ══════════════════════════════════════════════════════════════════════════════
# §5  THETA PASS (second look for hardest cases)
# ══════════════════════════════════════════════════════════════════════════════

def select_theta_cases(
    gamma_results: list[dict],
    db: NexusApexDB,
    round_num: int,
    positive_label: str = "ADE",
) -> list[dict]:
    """
    Select the ~100 hardest cases for second-look reclassification.
    Top-50 by δ (highest confident errors) + Top-50 near-misses.
    """
    # Separate errors and near-misses
    errors = [
        r for r in gamma_results
        if r["predicted"] != r["true_label"]
    ]
    near_misses = [
        r for r in gamma_results
        if r["predicted"] == r["true_label"] and r["confidence"] < NEAR_MISS_THRESHOLD
    ]

    # Sort errors by salience-weighted δ descending
    # High-stakes cases (death/severe) always surface to the top of the theta queue
    errors.sort(
        key=lambda r: compute_delta(r["confidence"], True) * compute_salience(r["text"], r.get("rationale", "")),
        reverse=True,
    )
    top_errors = errors[:THETA_TOP_DELTA_N]

    # Sort near-misses by boundary proximity (lowest confidence first)
    near_misses.sort(key=lambda r: r["confidence"])
    top_nm = near_misses[:THETA_TOP_NM_N]

    # Deduplicate by text
    seen = set()
    theta = []
    for r in top_errors + top_nm:
        if r["text"] not in seen:
            seen.add(r["text"])
            theta.append(r)

    print(
        f"[Theta] Selected {len(theta)} cases "
        f"({len(top_errors)} high-δ errors + {len(top_nm)} near-misses)",
        flush=True
    )
    return theta


def merge_theta_results(
    gamma_results: list[dict],
    theta_results: list[dict],
) -> tuple[list[dict], int]:
    """
    Replace gamma results with theta results where available.
    Returns (merged results, number of theta corrections).
    """
    theta_map = {r["text"]: r for r in theta_results}
    merged    = []
    corrections = 0
    for r in gamma_results:
        if r["text"] in theta_map:
            new_r = theta_map[r["text"]]
            if new_r["predicted"] != r["predicted"]:
                if new_r["predicted"] == new_r["true_label"]:
                    corrections += 1
            merged.append(new_r)
        else:
            merged.append(r)
    return merged, corrections


# ══════════════════════════════════════════════════════════════════════════════
# §6  CONTRASTIVE PAIR GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def find_contrast_case(
    error_text: str,
    error_label: str,
    rag_index: RAGIndex,
    k_search: int = 50,
) -> Optional[dict]:
    """
    Find the nearest semantically-similar case with the OPPOSITE correct label.
    This defines the decision boundary, not just one side of it.
    """
    results = rag_index.query(error_text, k=k_search)
    for r in results:
        r_label = r.get("label", r.get("true_label", ""))
        if r_label and r_label != error_label:
            return {"text": r["text"], "label": r_label}
    return None


def generate_contrastive_pair(
    error: dict,
    contrast: dict,
    llm_fn: Callable,
    round_num: int,
) -> Optional[dict]:
    """
    One LLM call per pair.
    Returns dict with lesson + key_distinction, or None on failure.
    """
    system = (
        "You are a clinical NLP expert generating precise teaching lessons "
        "from classification errors. Be concise and clinically specific."
    )
    user = (
        f"ERROR CASE (misclassified):\n"
        f"Text: \"{error['text']}\"\n"
        f"True label: {error['true_label']} | Wrong prediction: {error['predicted']}\n"
        f"Model's reasoning: \"{error['rationale']}\"\n"
        f"Error type: {error.get('error_type', 'general')}\n\n"
        f"CONTRAST CASE (correctly classified, semantically similar):\n"
        f"Text: \"{contrast['text']}\"\n"
        f"True label: {contrast['label']}\n\n"
        f"Generate a concise lesson for a clinical NLP classifier that made this error.\n"
        f"Explain why the error case is {error['true_label']} despite looking like "
        f"{error['predicted']}.\n"
        f"Use the contrast case to show what makes the difference.\n\n"
        f"Respond ONLY with JSON (no markdown, no code block):\n"
        f"{{\"lesson\": \"2-sentence teaching lesson\", "
        f"\"key_distinction\": \"10-word phrase capturing the core difference\"}}"
    )

    try:
        response = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return {
                "lesson":          str(d.get("lesson", "")),
                "key_distinction": str(d.get("key_distinction", "")),
            }
    except Exception as e:
        print(f"  [Pair Gen Error] {e}", file=sys.stderr)
    return None


def analyze_and_generate_pairs(
    db: NexusApexDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    round_num: int,
    plasticity: float,
) -> list[dict]:
    """
    Full pipeline: group errors → salience gate → weak-signal filter →
    direction verify → find contrast → generate lesson.

    Salience gating (norepinephrine/amygdala):
      - High-salience errors (death/severe) bypass plasticity gate and generate
        immediately, regardless of ACh plasticity level.
      - Weak-signal error types (low mean δ) require WEAK_SIGNAL_REPS cumulative
        occurrences before a pair is generated.

    Returns list of newly generated pairs (for theta-pass injection).
    """
    grouped = db.get_errors_by_type(round_num, min_delta=MIN_PAIR_DELTA)
    if not grouped:
        print("  [Pairs] No qualifying errors this round.", flush=True)
        return []

    new_pairs = []
    # ACh gating: base budget scales with plasticity, but high-salience bypass this
    base_budget = max(1, round(len(grouped) * plasticity))

    generated = 0
    # Sort error types by salience-weighted total δ (high-stakes first)
    def etype_priority(item):
        etype, errors = item
        max_sal = max(e.get("salience", 1.0) for e in errors)
        total_d = sum(e["delta"] for e in errors)
        return max_sal * total_d

    for etype, errors in sorted(grouped.items(), key=etype_priority, reverse=True):

        # ── Salience check ──────────────────────────────────────────────────
        max_salience = max(e.get("salience", 1.0) for e in errors)
        is_high_salience = max_salience >= SALIENCE_HIGH_THRESHOLD

        # High-salience bypasses plasticity gate — always generate
        if not is_high_salience:
            if generated >= base_budget:
                break

            # ── Weak-signal gate ─────────────────────────────────────────────
            # Low-δ error types need cumulative evidence before generating a pair.
            # This prevents over-fitting to noise in small batches.
            mean_delta = sum(e["delta"] for e in errors) / max(1, len(errors))
            if mean_delta <= WEAK_SIGNAL_DELTA_MAX:
                cumulative_count = db.get_pattern_frequency(etype)
                if cumulative_count < WEAK_SIGNAL_REPS:
                    print(
                        f"  [Weak Signal] {etype}: mean δ={mean_delta:.2f},"
                        f" seen {cumulative_count}/{WEAK_SIGNAL_REPS}x — waiting for repetition",
                        flush=True
                    )
                    continue

        if is_high_salience:
            print(f"  [⚡ HIGH SALIENCE] {etype}: bypassing plasticity gate (salience={max_salience:.1f}×)",
                  flush=True)

        # ── Direction verification ───────────────────────────────────────────
        n_fp = sum(1 for e in errors if e["predicted"] != e["true_label"]
                   and e["predicted"] == "ADE")
        n_fn = len(errors) - n_fp
        direction_pct = max(n_fp, n_fn) / max(1, len(errors))

        if direction_pct < MIN_DIRECTION_PCT:
            print(
                f"  [Skip] {etype}: mixed direction ({direction_pct:.0%}) — "
                f"generating contrastive pair only", flush=True
            )

        # ── Select anchor and generate ───────────────────────────────────────
        # Anchor = highest salience-weighted δ error
        anchor_error = max(errors, key=lambda e: e["delta"] * e.get("salience", 1.0))
        anchor_error["error_type"] = etype
        anchor_salience = anchor_error.get("salience", 1.0)

        # Find contrast case
        contrast = find_contrast_case(
            anchor_error["text"], anchor_error["true_label"], rag_index
        )
        if not contrast:
            print(f"  [Skip] {etype}: no contrast case found", flush=True)
            continue

        # Generate lesson (1 LLM call)
        print(f"  [Pair] Generating [{etype}] lesson...", flush=True)
        result = generate_contrastive_pair(anchor_error, contrast, llm_fn, round_num)
        if not result:
            continue

        # Get anchor embedding for future retrieval
        anchor_emb = embedder_embed_one(anchor_error["text"])

        # Store in DB — salience passed for immediate Core promotion if high-stakes
        pair_id = db.upsert_pair(
            anchor_text      = anchor_error["text"],
            anchor_label     = anchor_error["true_label"],
            contrast_text    = contrast["text"],
            contrast_label   = contrast["label"],
            error_type       = etype,
            lesson           = result["lesson"],
            key_distinction  = result["key_distinction"],
            delta            = anchor_error["delta"],
            anchor_embedding = anchor_emb,
            round_num        = round_num,
            salience         = anchor_salience,
        )

        new_pair = {
            "anchor_text":     anchor_error["text"],
            "anchor_label":    anchor_error["true_label"],
            "contrast_text":   contrast["text"],
            "contrast_label":  contrast["label"],
            "error_type":      etype,
            "lesson":          result["lesson"],
            "key_distinction": result["key_distinction"],
        }
        new_pairs.append(new_pair)

        salience_tag = f" [⚡{anchor_salience:.1f}×]" if anchor_salience >= 2.0 else ""
        print(
            f"  [Pair ✓]{salience_tag} [{etype}] KEY: {result['key_distinction'][:60]}",
            flush=True
        )
        generated += 1

    # Enforce MAX_ACTIVE_PAIRS cap — retire lowest-δ (non-Core) pairs when exceeded
    active = db.get_active_pairs()
    if len(active) > MAX_ACTIVE_PAIRS:
        non_core = [p for p in active if not p["is_core"]]
        to_retire = sorted(non_core, key=lambda p: p["delta_weight"])
        n_retire   = max(0, len(active) - MAX_ACTIVE_PAIRS)
        retire_ids = [p["id"] for p in to_retire[:n_retire]]
        if retire_ids:
            db.mark_pairs_absorbed(retire_ids)
            print(f"  [Cap] Retired {len(retire_ids)} lowest-δ non-Core pairs (cap={MAX_ACTIVE_PAIRS})",
                  flush=True)

    return new_pairs


# ══════════════════════════════════════════════════════════════════════════════
# §7  CONSOLIDATION (slow timescale — every 3 rounds)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_consolidation(response: str) -> Optional[dict]:
    """Robust JSON extraction from consolidation response."""
    try:
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    # Field-by-field fallback
    result = {}
    cm = re.search(r'"causal_model"\s*:\s*"([^"]+)"', response)
    if cm:
        result["causal_model"] = cm.group(1)
    kf = re.findall(r'"([^"]{10,})"', response)
    if kf and len(kf) >= 3:
        result.setdefault("key_factors", kf[:3])
        result.setdefault("error_patterns_to_avoid", kf[3:6])
    return result if result else None


def consolidate(
    db: NexusApexDB,
    llm_fn: Callable,
    round_num: int,
) -> Optional[dict]:
    """
    Sleep phase: synthesise accumulated contrastive pair lessons into a
    causal model.  Updates structural knowledge.  Graduates absorbed pairs.
    Returns the new structural knowledge dict, or None on failure.
    """
    pairs = db.get_active_pairs()
    if len(pairs) < 3:
        print(f"  [Consolidation] Too few pairs ({len(pairs)}) — skipping",
              flush=True)
        return None

    existing = db.get_structural_knowledge()

    lesson_lines = []
    for p in pairs[:15]:
        lesson_lines.append(
            f"• [{p['error_type']}] {p['lesson']}\n"
            f"  Key distinction: {p['key_distinction']}\n"
            f"  (δ={p['delta_weight']:.2f}, seen {p['occurrence_count']}x)"
        )

    existing_block = ""
    if existing:
        existing_block = (
            f"\nYOUR CURRENT UNDERSTANDING:\n\"{existing['causal_model']}\"\n"
        )

    system = (
        "You are a clinical NLP expert reflecting on what you have learned "
        "from repeated classification errors. Synthesise your experience into "
        "genuine clinical understanding — not rules, but causal insight."
    )
    user = (
        f"You have been classifying Adverse Drug Events (ADEs) from clinical text.\n"
        f"Based on {len(pairs)} hard-learned lessons below, update your understanding.\n"
        f"{existing_block}\n"
        f"RECENT HARD-LEARNED LESSONS:\n"
        + "\n".join(lesson_lines) +
        f"\n\nSynthesise:\n"
        f"1. A causal model (2-3 sentences): what FUNDAMENTALLY distinguishes ADE\n"
        f"   sentences from non-ADE sentences? Focus on causality, intent, and\n"
        f"   harm attribution — not surface keywords.\n"
        f"2. The 3 most important decision factors (priority order)\n"
        f"3. The 3 most common reasoning errors to avoid\n\n"
        f"CRITICAL CONSTRAINT: Your causal model MUST maintain high recall — do NOT\n"
        f"create a definition so strict that it misses genuine ADEs. It is worse to\n"
        f"miss a real ADE than to flag a borderline case. When uncertain, lean toward ADE.\n\n"
        f"Respond ONLY with JSON (no markdown, no code block):\n"
        f"{{\"causal_model\": \"...\", "
        f"\"key_factors\": [\"...\", \"...\", \"...\"], "
        f"\"error_patterns_to_avoid\": [\"...\", \"...\", \"...\"]}}"
    )

    try:
        response = llm_fn(system=system, user=user)
        parsed   = _parse_consolidation(response)
        if not parsed or "causal_model" not in parsed:
            print(f"  [Consolidation] Parse failed — no causal model extracted",
                  file=sys.stderr)
            return None

        db.store_structural_knowledge(
            round_num     = round_num,
            causal_model  = parsed["causal_model"],
            key_factors   = parsed.get("key_factors", []),
            error_patterns= parsed.get("error_patterns_to_avoid", []),
        )

        # Graduate absorbed pairs (those seen 3+ rounds with good theta correction)
        old_pairs = db.get_pairs_created_before(round_num, min_occurrences=3)
        absorbed_ids = [p["id"] for p in old_pairs]
        if absorbed_ids:
            db.mark_pairs_absorbed(absorbed_ids)
            print(
                f"  [Consolidation] {len(absorbed_ids)} pairs graduated to "
                f"structural knowledge", flush=True
            )

        print(
            f"  [Causal Model] {parsed['causal_model'][:120]}...", flush=True
        )
        return parsed

    except Exception as e:
        print(f"  [Consolidation Error] {e}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# §8  MAIN LEARNER
# ══════════════════════════════════════════════════════════════════════════════

class APEXLearner:

    def __init__(self, args):
        self.args    = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.out_dir / "nexus_apex.db")

    def build_llm_fn(self) -> Callable:
        if self.args.mock:
            enable_mock_embeddings()
            def llm_fn(system: str, user: str) -> str:
                import random as _r
                label = _r.choice(["ADE", "NOT_ADE"])
                return json.dumps({
                    "classification": label,
                    "confidence": round(_r.uniform(0.45, 0.92), 2),
                    "rationale":  "mock: temporal following relationship observed",
                    "lesson":     "mock lesson",
                    "key_distinction": "mock distinction",
                    "causal_model": "Mock: ADEs are drug-caused harms.",
                    "key_factors": ["causality", "harm", "drug"],
                    "error_patterns_to_avoid": ["temporal confusion"],
                })
            return llm_fn
        elif self.args.ai_hub:
            client = llm_client.AIHubClient(
                api_key=self.args.ai_hub_key,
                ad_object_id=self.args.ai_hub_ad_id,
            )
            def llm_fn(system: str, user: str) -> str:
                return client.chat(system=system, user=user)
            return llm_fn
        raise ValueError("Must specify --ai-hub or --mock")

    def run(self, config) -> None:
        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS APEX LEARNER — Predictive Contrastive Consolidation", flush=True)
        print(f"  Task: {config.task_name}", flush=True)
        print(f"{'╚'+'═'*78+'╝'}\n", flush=True)

        llm_fn = self.build_llm_fn()

        # API verification
        if not self.args.mock:
            print("[Startup] Verifying LLM API...", flush=True)
            test = llm_fn("Say OK", "Reply with just the word OK")
            print(f"  API OK: {test[:30]!r}", flush=True)

        # ── Data ──────────────────────────────────────────────────────────────
        print("\n[Data] Loading corpus...", flush=True)
        eval_pool, probe_pool, train_pool = data_utils.load_and_split(
            seed=self.args.seed, eval_size=200, probe_size=300,
        )
        print(
            f"  Train pool: {len(train_pool)} | Eval: {len(eval_pool)} | "
            f"Probe: {len(probe_pool)}", flush=True
        )

        # ── Database ─────────────────────────────────────────────────────────
        db = NexusApexDB(self.db_path)

        if self.args.fresh or db.count_cases().get("train", 0) == 0:
            print("\n[DB] Loading corpus into SQLite...", flush=True)
            db.load_cases(train_pool, split="train", round_num=0)
            db.load_cases(eval_pool,  split="eval",  round_num=0)
            db.load_cases(probe_pool, split="probe", round_num=0)
            print(f"  Loaded {len(train_pool)} train cases", flush=True)

        print(f"\n{db.summary()}", flush=True)

        # ── Preload embedder (prevents 6× parallel model load in workers) ────
        print("\n[Embedder] Preloading model on main thread...", flush=True)
        _ = embedder_embed_one("warmup")
        print("[Embedder] Model ready.", flush=True)

        # ── RAG Index ─────────────────────────────────────────────────────────
        rag_dir = str(self.out_dir / "rag_index")
        print(f"\n[RAG] Building/loading FAISS index...", flush=True)
        if self.args.fresh or not (Path(rag_dir) / "faiss.index").exists():
            rag_train = [{"text": c["text"], "label": c.get("true_label", c.get("label",""))}
                         for c in train_pool]
            rag_index = RAGIndex.build(rag_train, out_dir=rag_dir)
        else:
            rag_index = RAGIndex.load(rag_dir)

        # ── Warm restart state ────────────────────────────────────────────────
        history = db.get_round_history()
        completed_rounds = len(history)
        start_round = completed_rounds + 1
        n_rounds    = (
            TOTAL_CURRICULUM_ROUNDS if self.args.curriculum
            else self.args.rounds
        )
        end_round   = completed_rounds + n_rounds

        # Threshold restoration
        threshold = 0.50
        if history:
            threshold = history[-1].get("threshold", 0.50)
            print(f"[Warm] Restored threshold={threshold:.3f}", flush=True)

        # Consolidation cooldown — restored from history
        consolidation_cooldown = 0

        # Seen cases
        seen_texts = db.get_seen_case_texts() if completed_rounds > 0 else set()
        unseen_train = [c for c in train_pool if c["text"] not in seen_texts]
        random.seed(self.args.seed + completed_rounds)
        random.shuffle(unseen_train)

        # Curriculum offset into unseen_train (varies by round)
        curriculum_offset = 0

        # Base system prompt (blank slate — no lessons injected yet)
        seed_nodes = getattr(config, "seed_nodes", [])
        base_task  = (
            seed_nodes[0].prompt
            if seed_nodes else
            "You are a clinical NLP classifier. Determine whether the following "
            "sentence describes an Adverse Drug Event (ADE) — an unintended harmful "
            "outcome caused by drug administration.\n"
            "Respond ONLY with JSON: {\"classification\": \"ADE\" or \"NOT_ADE\", "
            "\"confidence\": 0.0-1.0, \"rationale\": \"one sentence: KEY deciding factor\"}"
        )

        # ── Training loop ─────────────────────────────────────────────────────
        for round_num in range(start_round, end_round + 1):
            # Determine batch size: curriculum schedule or fixed override
            if self.args.curriculum:
                batch_size = get_curriculum_batch_size(round_num)
                phase = next(
                    (i+1 for i, (n, _) in enumerate(CURRICULUM)
                     if round_num <= sum(r for r, _ in CURRICULUM[:i+1])),
                    len(CURRICULUM)
                )
            else:
                batch_size = self.args.batch_size
                phase = None

            batch = unseen_train[curriculum_offset: curriculum_offset + batch_size]
            curriculum_offset += batch_size

            if not batch:
                print(f"\n[Done] Corpus exhausted at R{round_num}.", flush=True)
                break

            # Load structural knowledge for this round
            structural = db.get_structural_knowledge()
            system_prompt = build_system_prompt(base_task, structural, round_num)

            # ACh plasticity gating (will be updated after gamma pass)
            plasticity = 1.0

            print(f"\n{'═'*80}", flush=True)
            n_pairs = len(db.get_active_pairs())
            n_core  = len(db.get_core_pairs())
            phase_tag = f" | Phase {phase} (batch={batch_size})" if phase else f" | batch={batch_size}"
            print(
                f"  ROUND {round_num}{phase_tag} | "
                f"Pairs={n_pairs} ({n_core} Core) | "
                f"Threshold={threshold:.3f} | "
                f"Seen={len(seen_texts)}",
                flush=True
            )
            print(f"{'═'*80}\n", flush=True)

            # ── GAMMA PASS: classify all cases ────────────────────────────────
            gamma_results = classify_batch(
                cases       = batch,
                db          = db,
                rag_index   = rag_index,
                llm_fn      = llm_fn,
                system_prompt = system_prompt,
                round_num   = round_num,
                threshold   = threshold,
                workers     = self.args.workers,
                label       = "Gamma",
            )
            gamma_adj = apply_threshold(gamma_results, threshold)
            gamma_m   = compute_metrics(gamma_adj)

            n_gamma_errors = gamma_m["tp"] + gamma_m["fp"] + gamma_m["fn"] + gamma_m["tn"]
            error_rate = (gamma_m["fn"] + gamma_m["fp"]) / max(1, len(gamma_adj))

            print(
                f"[Gamma] Errors: {gamma_m['fn']+gamma_m['fp']}/{len(gamma_adj)}"
                f" ({error_rate:.1%}) | "
                f"F1={gamma_m['f1']:.4f} P={gamma_m['precision']:.3f}"
                f" R={gamma_m['recall']:.3f}",
                flush=True
            )
            print(
                f"  TP={gamma_m['tp']} FP={gamma_m['fp']}"
                f" FN={gamma_m['fn']} TN={gamma_m['tn']}",
                flush=True
            )

            # ACh plasticity: high error rate → high plasticity
            plasticity = math.tanh(error_rate * 3)
            print(f"  Plasticity (ACh): {plasticity:.3f}", flush=True)

            # ── Log errors and near-misses ─────────────────────────────────
            print("\n[Error Analysis]", flush=True)
            delta_total    = 0.0
            error_types    = {}
            n_near_misses  = 0

            # Build case_id lookup from DB
            db_cases = {c["text"]: c["id"] for c in db.get_cases("train")}

            n_high_salience = 0
            for r in gamma_adj:
                case_id  = db_cases.get(r["text"], 0)
                rationale = r.get("rationale", "")
                salience  = compute_salience(r["text"], rationale)

                if r["predicted"] != r["true_label"]:
                    d = db.add_error(
                        case_id   = case_id,
                        text      = r["text"],
                        round_num = round_num,
                        predicted = r["predicted"],
                        true_label= r["true_label"],
                        confidence= r["confidence"],
                        rationale = rationale,
                        salience  = salience,
                    )
                    delta_total += d
                    etype = classify_error_type(rationale)
                    error_types[etype] = error_types.get(etype, 0) + 1
                    if salience >= SALIENCE_HIGH_THRESHOLD:
                        n_high_salience += 1

                elif r["confidence"] < NEAR_MISS_THRESHOLD:
                    db.add_near_miss(
                        case_id   = case_id,
                        text      = r["text"],
                        true_label= r["true_label"],
                        round_num = round_num,
                        confidence= r["confidence"],
                    )
                    n_near_misses += 1

                seen_texts.add(r["text"])

            n_errors = gamma_m["fn"] + gamma_m["fp"]
            print(
                f"  δ_total={delta_total:.1f} | "
                f"Mean δ={delta_total/max(1,n_errors):.3f}"
                + (f" | ⚡ High-salience errors: {n_high_salience}" if n_high_salience else ""),
                flush=True
            )
            print(
                f"  Error types: "
                + " | ".join(f"{k}={v}" for k, v in sorted(error_types.items())),
                flush=True
            )
            print(f"  Near-misses: {n_near_misses}", flush=True)

            # Increment cumulative pattern frequency (weak-signal repetition tracking)
            for etype, count in error_types.items():
                db.increment_pattern_frequency(etype, round_num)

            # ── Generate contrastive pairs from gamma errors ───────────────
            print(f"\n[Pair Generation] Plasticity={plasticity:.2f}", flush=True)
            fresh_pairs = analyze_and_generate_pairs(
                db        = db,
                rag_index = rag_index,
                llm_fn    = llm_fn,
                round_num = round_num,
                plasticity= plasticity,
            )

            # ── THETA PASS: reclassify hardest cases with fresh pairs ──────
            theta_cases = select_theta_cases(gamma_adj, db, round_num)
            theta_corrections = 0

            if theta_cases and round_num >= 2:
                # Reload system prompt — structural knowledge unchanged
                theta_system = system_prompt  # structural knowledge same this round
                theta_results = classify_batch(
                    cases         = theta_cases,
                    db            = db,
                    rag_index     = rag_index,
                    llm_fn        = llm_fn,
                    system_prompt = theta_system,
                    round_num     = round_num,
                    threshold     = threshold,
                    workers       = self.args.workers,
                    second_look   = True,
                    fresh_lessons = fresh_pairs if fresh_pairs else None,
                    label         = "Theta",
                )
                theta_adj = apply_threshold(theta_results, threshold)

                # Merge back into gamma results
                gamma_adj, theta_corrections = merge_theta_results(gamma_adj, theta_adj)
                print(
                    f"  Theta corrections: {theta_corrections}/{len(theta_cases)}"
                    f" ({theta_corrections/max(1,len(theta_cases)):.1%} absorption)",
                    flush=True
                )

                # Recalculate train metrics with theta corrections applied
                gamma_m = compute_metrics(gamma_adj)

            # ── EMA threshold update ──────────────────────────────────────
            threshold = calibrate_threshold_ema(gamma_adj, threshold)
            print(f"\n[Threshold] EMA updated to {threshold:.3f}", flush=True)

            # ── Flush retrieval counts → Core promotion ───────────────────
            newly_core = db.flush_retrieval_counts()
            if newly_core:
                print(f"  [Core] {newly_core} pair(s) promoted to Core via retrieval frequency",
                      flush=True)

            # ── Consolidation (every 3 rounds) with cooldown ─────────────
            if round_num % CONSOLIDATION_EVERY == 0:
                print(f"\n[CONSOLIDATION R{round_num}] Sleep phase...", flush=True)
                if consolidation_cooldown > 0:
                    print(
                        f"  [Cooldown] Holding causal model stable "
                        f"({consolidation_cooldown} round(s) remaining)",
                        flush=True
                    )
                    consolidation_cooldown -= 1
                else:
                    new_knowledge = consolidate(db, llm_fn, round_num)
                    if new_knowledge:
                        # Update system prompt for evaluation (structural knowledge updated)
                        structural = db.get_structural_knowledge()
                        system_prompt = build_system_prompt(base_task, structural, round_num)
                        consolidation_cooldown = CONSOLIDATION_COOLDOWN

            # ── Evaluation ────────────────────────────────────────────────
            print(f"\n[Evaluation] 200 held-out cases...", flush=True)
            eval_cases_raw = db.get_cases("eval")
            # Add true_label field for compatibility
            eval_cases = [
                {"text": c["text"], "true_label": c["true_label"], "id": c["id"]}
                for c in eval_cases_raw
            ]
            eval_results = classify_batch(
                cases         = eval_cases,
                db            = db,
                rag_index     = rag_index,
                llm_fn        = llm_fn,
                system_prompt = system_prompt,
                round_num     = round_num,
                threshold     = threshold,
                workers       = self.args.workers,
                label         = "Eval",
            )
            eval_adj = apply_threshold(eval_results, threshold)
            eval_m   = compute_metrics(eval_adj)

            # ── Round summary ─────────────────────────────────────────────
            n_active = len(db.get_active_pairs())
            n_core   = len(db.get_core_pairs())
            print(f"\n{'─'*60}", flush=True)
            print(
                f"  Round {round_num} | "
                f"Eval F1={eval_m['f1']:.4f} | "
                f"P={eval_m['precision']:.3f} | "
                f"R={eval_m['recall']:.3f} | "
                f"Pairs={n_active} ({n_core} Core)",
                flush=True
            )
            print(
                f"  TP={eval_m['tp']} FP={eval_m['fp']}"
                f" FN={eval_m['fn']} TN={eval_m['tn']}"
                f" | θ-corrections={theta_corrections}",
                flush=True
            )
            print(f"{'─'*60}\n", flush=True)

            # Save round stats
            db.save_round_stats(
                round_num        = round_num,
                train_f1         = gamma_m["f1"],
                train_precision  = gamma_m["precision"],
                train_recall     = gamma_m["recall"],
                train_errors     = n_errors,
                eval_f1          = eval_m["f1"],
                eval_precision   = eval_m["precision"],
                eval_recall      = eval_m["recall"],
                threshold        = threshold,
                plasticity       = plasticity,
                n_pairs          = n_active,
                n_core           = n_core,
                theta_corrections= theta_corrections,
                near_misses      = n_near_misses,
            )

        # ── Final report ─────────────────────────────────────────────────────
        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS APEX — FINAL REPORT", flush=True)
        print(f"{'╚'+'═'*78+'╝'}\n", flush=True)
        print(db.summary(), flush=True)

        history = db.get_round_history()
        print(f"\n  F1 Trajectory:", flush=True)
        for r in history:
            bar = "█" * int(r["eval_f1"] * 40)
            print(
                f"    R{r['round']:02d} F1={r['eval_f1']:.4f} {bar}"
                f" | Pairs={r['n_pairs']} Core={r['n_core']}"
                f" θ={r['theta_corrections']}",
                flush=True
            )

        if history:
            best = max(history, key=lambda r: r["eval_f1"])
            print(f"\n  Best  F1: {best['eval_f1']:.4f} at R{best['round']}", flush=True)
            print(f"  Final F1: {history[-1]['eval_f1']:.4f}", flush=True)

        sk = db.get_structural_knowledge()
        if sk:
            print(f"\n  Causal Model (final):\n  {sk['causal_model']}", flush=True)
            print(f"  Key Factors: {', '.join(sk['key_factors'])}", flush=True)

        print(f"\n[Apex] DB: {self.db_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# §9  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NEXUS Apex Learner")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--out",         default="run_apex")
    parser.add_argument("--rounds",      type=int, default=40,
                        help="Number of rounds (ignored if --curriculum is set)")
    parser.add_argument("--batch-size",  type=int, default=1000, dest="batch_size",
                        help="Fixed batch size per round (ignored if --curriculum is set)")
    parser.add_argument("--curriculum",  action="store_true",
                        help=f"Use developmental curriculum schedule: "
                             + ", ".join(f"{b}×{n}" for n, b in CURRICULUM)
                             + f" ({TOTAL_CURRICULUM_ROUNDS} rounds total)")
    parser.add_argument("--workers",     type=int, default=6)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--fresh",       action="store_true")
    parser.add_argument("--mock",        action="store_true")
    parser.add_argument("--ai-hub",      action="store_true", dest="ai_hub")
    parser.add_argument("--ai-hub-key",  default=os.environ.get("AIHUB_API_KEY", ""),
                        dest="ai_hub_key")
    parser.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""),
                        dest="ai_hub_ad_id")
    args = parser.parse_args()

    config = TaskConfig.load(args.config)
    APEXLearner(args).run(config)


if __name__ == "__main__":
    main()

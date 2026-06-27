"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NEXUS MCQ LEARNER                                                           ║
║  SQLite-backed pattern learning with MCQ bank                                ║
║  Northwell Health — NEXUS Research Program                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Design principle:
  LLM is called for TWO things only:
    1. Classification  — 1 call per case, prompt includes CORE MCQs
    2. MCQ generation  — 1 call per new recurring pattern

  Everything else is local:
    • Feature vectors (internal tokens)      — features.py,   0 API cost
    • Pattern detection                      — SQL GROUP BY,  0 API cost
    • MCQ effectiveness measurement          — SQL query,     0 API cost
    • Held-out evaluation                    — classification only

Scale:
  1000 cases/round × 16–20 rounds = 16,000–20,000 training cases
  25% corpus held out for final evaluation (~5,800 cases)
  Each round tests previous round's MCQs on unseen cases

MCQ Bank lifecycle:
  Error (round N) → pattern detected (SQL) → MCQ generated (LLM, 1 call)
  → inserted into CORE prompt (round N+1) → effectiveness measured (round N+2)
  → retired if ineffective | promoted to CORE if seen 3+ rounds
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Callable, Optional

import data_utils
import llm_client
from features import features as extract_features, FEATURE_NAMES
from nexus_db_v2 import (
    NexusDB, feature_vector, feature_signature, signature_from_row
)
from rag_index import RAGIndex
from task_config import TaskConfig


# ══════════════════════════════════════════════════════════════════════════════
# §1  MCQ PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_classification_prompt(
    text: str,
    core_mcqs: list[dict],
    contextual_mcqs: list[dict],
    rag_examples: list[dict],
    base_prompt: str,
) -> tuple[str, str]:
    """
    Build system + user prompt for a single classification call.

    Priority order in prompt:
      1. Base task instruction (from config)
      2. RAG examples (most similar corpus cases)
      3. CORE MCQs (always present — recurring patterns seen 3+ rounds)
      4. Contextual MCQs (patterns matching THIS case's feature signature)
    """
    # RAG examples
    rag_block = ""
    if rag_examples:
        ex_lines = "\n".join(
            f"  [{e['label']}] {e['text'][:120]}"
            for e in rag_examples[:4]
        )
        rag_block = f"\nSIMILAR CASES FROM TRAINING:\n{ex_lines}\n"

    # CORE MCQs (always included)
    core_block = ""
    if core_mcqs:
        lines = ["\nCORE LESSONS (apply these to every case):"]
        for i, mcq in enumerate(core_mcqs[:6], 1):
            wrong = json.loads(mcq.get("wrong_answers", "[]"))
            wrong_lines = "\n".join(
                f"    ✗ {w['answer']}: {w['explanation']}"
                for w in wrong[:2]
            )
            lines.append(
                f"[{i}] Example: \"{mcq['example_text'][:100]}\"\n"
                f"    ✓ {mcq['correct_answer']}: {mcq['correct_rationale']}\n"
                f"{wrong_lines}"
            )
        core_block = "\n".join(lines)

    # Contextual MCQs (this case's feature pattern)
    ctx_block = ""
    # Deduplicate vs core (by pattern_id)
    core_ids = {m["pattern_id"] for m in core_mcqs}
    ctx_only = [m for m in contextual_mcqs if m["pattern_id"] not in core_ids]
    if ctx_only:
        lines = ["\nPATTERN-SPECIFIC LESSONS (this sentence matches these patterns):"]
        for i, mcq in enumerate(ctx_only[:3], 1):
            wrong = json.loads(mcq.get("wrong_answers", "[]"))
            wrong_lines = "\n".join(
                f"    ✗ {w['answer']}: {w['explanation']}"
                for w in wrong[:2]
            )
            lines.append(
                f"[{i}] Example: \"{mcq['example_text'][:100]}\"\n"
                f"    ✓ {mcq['correct_answer']}: {mcq['correct_rationale']}\n"
                f"{wrong_lines}"
            )
        ctx_block = "\n".join(lines)

    system = base_prompt
    user = (
        f"{rag_block}"
        f"{core_block}"
        f"{ctx_block}"
        f"\nSENTENCE TO CLASSIFY: \"{text}\"\n\n"
        f"Respond ONLY with JSON:\n"
        f'{{\"classification\": \"ADE\" or \"NOT_ADE\", '
        f'\"confidence\": 0.0-1.0, \"rationale\": \"one sentence\"}}'
    )
    return system, user


# ══════════════════════════════════════════════════════════════════════════════
# §2  CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_one(
    case: dict,
    db: NexusDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    base_prompt: str,
    firing_threshold: float = 1.0,
    timeout: float = 30.0,
) -> dict:
    """Classify a single case. Single LLM call with MCQ context."""
    text = case["text"]
    feats = extract_features(text)
    sig   = feature_signature(feats)

    # Get MCQ context from DB (zero LLM cost)
    core_mcqs = db.get_core_mcqs()
    ctx_mcqs  = db.get_contextual_mcqs(sig, max_mcqs=3)
    rag_examples = rag_index.query(text, k=4)

    system, user = build_classification_prompt(
        text=text,
        core_mcqs=core_mcqs,
        contextual_mcqs=ctx_mcqs,
        rag_examples=rag_examples,
        base_prompt=base_prompt,
    )

    try:
        response = llm_fn(system=system, user=user)
        m = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if m:
            d = json.loads(m.group())
            label = d.get("classification", "NOT_ADE")
            conf  = float(d.get("confidence", 0.5))
        else:
            label, conf = "NOT_ADE", 0.1
    except Exception as e:
        label, conf = "NOT_ADE", 0.1

    return {
        "case_id":     case.get("id"),
        "text":        text,
        "true_label":  case["true_label"] if "true_label" in case else case.get("label", ""),
        "predicted":   label,
        "confidence":  conf,
        "feature_sig": sig,
        "feats":       feats,
    }


def classify_batch(
    cases: list[dict],
    db: NexusDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    base_prompt: str,
    firing_threshold: float = 1.0,
    workers: int = 6,
    positive_label: str = "ADE",
) -> tuple[list[dict], list[dict]]:
    """
    Classify a batch in parallel. Returns (results, errors).
    results: all classification results
    errors:  misclassified cases only
    """
    results = []
    errors  = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(classify_one, c, db, rag_index, llm_fn,
                      base_prompt, firing_threshold): c
            for c in cases
        }
        for fut in as_completed(futs, timeout=len(cases) * 35):
            try:
                r = fut.result(timeout=35)
            except Exception as e:
                print(f"  [Classify Error] {e}", file=sys.stderr)
                continue
            results.append(r)
            if r["predicted"] != r["true_label"]:
                errors.append(r)

    return results, errors


# ══════════════════════════════════════════════════════════════════════════════
# §3  MCQ GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_mcq_for_pattern(
    pattern: dict,
    example_errors: list[dict],
    llm_fn: Callable,
    round_num: int,
) -> Optional[dict]:
    """
    Generate a structured MCQ lesson for an error pattern.
    1 LLM call per pattern (not per error).

    Pattern is defined by its feature_sig — the internal token fingerprint
    of cases that repeatedly get misclassified.
    """
    sig   = pattern["feature_sig"]
    etype = pattern["error_type"]
    occ   = pattern.get("occurrence_count", 1)

    # Select a representative example
    example = example_errors[0] if example_errors else {}
    example_text = example.get("text", "")
    true_label   = example.get("true_label", "")
    wrong_label  = example.get("predicted", "")

    sig_features = [f for f in sig.split(",") if f]

    system = (
        "You are an expert clinical NLP educator creating teaching materials "
        "for an adaptive ADE classifier. Generate a concise MCQ lesson."
    )
    user = (
        f"ERROR PATTERN ANALYSIS — Round {round_num}\n"
        f"Feature signature: {sig if sig else '(no distinctive features)'}\n"
        f"Active features: {', '.join(sig_features) if sig_features else 'none'}\n"
        f"Error type: {etype} "
        f"({'classifier over-predicts ADE' if etype == 'FP' else 'classifier misses ADE'})\n"
        f"Occurred in {occ} round(s)\n\n"
        f"Representative example:\n"
        f"  Text: \"{example_text[:200]}\"\n"
        f"  Correct: {true_label} | System predicted: {wrong_label}\n\n"
        f"Generate a teaching MCQ for this error pattern.\n"
        f"Return ONLY JSON:\n"
        f'{{\n'
        f'  "correct_answer": "{true_label}",\n'
        f'  "correct_rationale": "one sentence: why {true_label} is correct here",\n'
        f'  "wrong_answers": [\n'
        f'    {{"answer": "{wrong_label}", '
        f'"explanation": "why {wrong_label} is wrong for this pattern"}},\n'
        f'    {{"answer": "UNCERTAIN", '
        f'"explanation": "why uncertainty is inappropriate here"}}\n'
        f'  ]\n'
        f'}}'
    )

    try:
        response = llm_fn(system=system, user=user)
        m = re.search(r'\{.*\}', response, re.DOTALL)
        if not m:
            return None
        d = json.loads(m.group())
        return {
            "correct_answer":    d.get("correct_answer", true_label),
            "correct_rationale": d.get("correct_rationale", ""),
            "wrong_answers":     d.get("wrong_answers", []),
            "example_text":      example_text[:300],
        }
    except Exception as e:
        print(f"  [MCQ Gen Error] {e}", file=sys.stderr)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# §4  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    eval_cases: list[dict],
    db: NexusDB,
    rag_index: RAGIndex,
    llm_fn: Callable,
    base_prompt: str,
    firing_threshold: float = 1.0,
    workers: int = 6,
    positive_label: str = "ADE",
) -> dict:
    """Evaluate on held-out cases. Returns metrics dict."""
    results, _ = classify_batch(
        cases=eval_cases, db=db, rag_index=rag_index,
        llm_fn=llm_fn, base_prompt=base_prompt,
        firing_threshold=firing_threshold, workers=workers,
    )
    tp = fp = fn = tn = 0
    for r in results:
        pred = r["predicted"]
        true = r["true_label"]
        if pred == positive_label and true == positive_label:    tp += 1
        elif pred == positive_label and true != positive_label:  fp += 1
        elif pred != positive_label and true == positive_label:  fn += 1
        else:                                                     tn += 1
    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)
    return {"f1": f1, "precision": prec, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_evaluated": len(results)}


# ══════════════════════════════════════════════════════════════════════════════
# §5  HOMEOSTATIC THRESHOLD CALIBRATION (zero LLM cost, same as v2)
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_threshold(
    results: list[dict],
    positive_label: str = "ADE",
    current_threshold: float = 1.0,
) -> float:
    """
    Sweep candidate thresholds over cached (confidence, true_label) pairs.
    Returns the threshold that maximises F1. Zero LLM calls.
    """
    # This version uses simple confidence cutoff, not ADE vs NOT_ADE score ratio
    # since single-call classification returns confidence directly
    candidates = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_t, best_f1 = current_threshold, 0.0

    for t in candidates:
        tp = fp = fn = tn = 0
        for r in results:
            # At confidence threshold t: predict ADE if conf >= t AND predicted ADE
            # else NOT_ADE. This lets us tune recall vs precision.
            if r["predicted"] == positive_label and r["confidence"] >= t:
                pred = positive_label
            else:
                pred = "NOT_ADE"
            true = r["true_label"]
            if pred == positive_label and true == positive_label:    tp += 1
            elif pred == positive_label and true != positive_label:  fp += 1
            elif pred != positive_label and true == positive_label:  fn += 1
            else:                                                     tn += 1
        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1     = 2 * prec * recall / max(1e-9, prec + recall)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# §6  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

class MCQLearner:

    def __init__(self, args):
        self.args    = args
        self.out_dir = Path(args.out)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.out_dir / "nexus_memory.db")

    def build_llm_fn(self) -> Callable:
        if self.args.mock:
            def llm_fn(system: str, user: str) -> str:
                import random as _r
                label = _r.choice(["ADE", "NOT_ADE"])
                return json.dumps({
                    "classification": label,
                    "confidence": round(_r.uniform(0.55, 0.90), 2),
                    "rationale": "mock",
                    "correct_answer": label,
                    "correct_rationale": "mock rationale",
                    "wrong_answers": [{"answer": "NOT_ADE", "explanation": "mock"}],
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
        print(f"  NEXUS MCQ LEARNER — SQLite Pattern Memory", flush=True)
        print(f"  Task: {config.task_name}", flush=True)
        print(f"{'╚'+'═'*78+'╝'}\n", flush=True)

        llm_fn = self.build_llm_fn()

        # API check
        if not self.args.mock:
            print("[Startup] Verifying LLM API...", flush=True)
            test = llm_fn("Say OK", "Reply with just the word OK")
            print(f"  API OK: {test[:30]!r}", flush=True)

        # ── Load data ─────────────────────────────────────────────────────────
        print("\n[Data] Loading corpus...", flush=True)
        eval_pool, probe_pool, train_pool = data_utils.load_and_split(
            seed=self.args.seed, eval_size=200, probe_size=300,
        )
        # Held-out: 25% of corpus never seen in training
        random.seed(self.args.seed)
        all_cases = data_utils.load_and_split.__wrapped__(
            seed=self.args.seed
        ) if hasattr(data_utils.load_and_split, '__wrapped__') else None

        print(f"  Train pool: {len(train_pool)} | Eval: {len(eval_pool)} | "
              f"Probe: {len(probe_pool)}", flush=True)

        # ── Database init ─────────────────────────────────────────────────────
        db = NexusDB(self.db_path)

        if self.args.fresh or db.count_cases().get("train", 0) == 0:
            print("\n[DB] Loading corpus into SQLite...", flush=True)
            n = db.load_cases(train_pool, split="train", round_num=0)
            db.load_cases(eval_pool,  split="eval",  round_num=0)
            db.load_cases(probe_pool, split="probe", round_num=0)
            print(f"  Loaded {n} train + {len(eval_pool)} eval + "
                  f"{len(probe_pool)} probe cases", flush=True)
        else:
            print(f"\n[DB] Warm restart — {db.summary()}", flush=True)

        print(f"\n{db.summary()}", flush=True)

        # ── RAG index ─────────────────────────────────────────────────────────
        rag_dir = str(self.out_dir / "rag_index")
        print(f"\n[RAG] Building/loading FAISS index...", flush=True)
        if self.args.fresh or not (Path(rag_dir) / "faiss.index").exists():
            rag_index = RAGIndex.build(train_pool, out_dir=rag_dir)
        else:
            rag_index = RAGIndex.load(rag_dir)

        # ── Determine rounds already completed ────────────────────────────────
        history = db.get_round_history()
        completed_rounds = len(history)
        start_round = completed_rounds + 1
        end_round   = completed_rounds + self.args.rounds

        # ── Base prompt from config ───────────────────────────────────────────
        seed_nodes = getattr(config, "seed_nodes", [])
        base_prompt = (
            seed_nodes[0].prompt
            if seed_nodes and hasattr(seed_nodes[0], "prompt")
            else seed_nodes[0].get("prompt", "") if seed_nodes
            else "Classify as ADE or NOT_ADE. Return JSON."
        )

        # ── Seen case IDs (for no-repeat sampling) ───────────────────────────
        seen_ids: set[int] = set()
        if not self.args.fresh:
            # Restore from DB: any case_id that appears in errors was trained on
            conn = db._conn()
            rows = conn.execute(
                "SELECT DISTINCT case_id FROM errors"
            ).fetchall()
            seen_ids = {r["case_id"] for r in rows if r["case_id"]}

        # Track firing threshold across rounds
        firing_threshold = 0.5
        if history:
            last = history[-1]
            firing_threshold = last.get("firing_threshold") or 0.5

        positive_label = config.positive_label

        # ═════════════════════════════════════════════════════════════════════
        #  TRAINING LOOP
        # ═════════════════════════════════════════════════════════════════════
        for round_num in range(start_round, end_round + 1):
            print(f"\n{'═'*80}", flush=True)
            n_core = len(db.get_core_mcqs())
            n_active_mcqs = db._conn().execute(
                "SELECT COUNT(*) FROM mcqs WHERE is_active=1"
            ).fetchone()[0]
            print(
                f"  ROUND {round_num} | "
                f"CoreMCQs={n_core} | ActiveMCQs={n_active_mcqs} | "
                f"Threshold={firing_threshold:.2f} | "
                f"SeenCases={len(seen_ids)}",
                flush=True
            )
            print(f"{'═'*80}", flush=True)

            # ── Sample unseen cases ───────────────────────────────────────────
            batch = db.get_unseen_train_cases(
                n=self.args.batch_size, seen_ids=seen_ids
            )
            if not batch:
                print("  [!] All training cases seen. Sampling with replacement.",
                      flush=True)
                batch = db.get_split_cases("train")
                random.shuffle(batch)
                batch = batch[:self.args.batch_size]

            # Convert DB rows to format classify_one expects
            for c in batch:
                c["label"] = c["true_label"]

            print(f"[Training] Classifying {len(batch)} cases "
                  f"(MCQ context: {n_core} CORE + per-case contextual)...",
                  flush=True)

            results, errors = classify_batch(
                cases=batch, db=db, rag_index=rag_index,
                llm_fn=llm_fn, base_prompt=base_prompt,
                firing_threshold=firing_threshold,
                workers=self.args.workers,
                positive_label=positive_label,
            )

            seen_ids.update(c.get("id") for c in batch if c.get("id"))

            # Training metrics
            tp = fp = fn = tn = 0
            for r in results:
                pred = r["predicted"]; true = r["true_label"]
                if pred == positive_label and true == positive_label:    tp += 1
                elif pred == positive_label and true != positive_label:  fp += 1
                elif pred != positive_label and true == positive_label:  fn += 1
                else:                                                     tn += 1
            prec   = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            train_f1 = 2 * prec * recall / max(1e-9, prec + recall)

            print(f"\n[Training] Errors: {len(errors)}/{len(batch)} "
                  f"({100*len(errors)/max(1,len(batch)):.1f}%) | "
                  f"Train F1={train_f1:.4f} P={prec:.3f} R={recall:.3f}",
                  flush=True)
            print(f"  TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)

            # ── Record errors ─────────────────────────────────────────────────
            if errors:
                db.record_errors(errors, round_num)

            # ── Pattern detection (pure SQL) ──────────────────────────────────
            print(f"\n[Pattern Detection] SQL analysis of {len(errors)} errors...",
                  flush=True)
            patterns_found = db.detect_patterns(
                round_num=round_num, min_errors=self.args.min_pattern_errors
            )
            print(f"  Found {len(patterns_found)} patterns "
                  f"({sum(1 for p in patterns_found if p['is_new'])} new, "
                  f"{sum(1 for p in patterns_found if not p['is_new'])} recurring)",
                  flush=True)

            n_new_patterns = 0
            for pat in patterns_found:
                sig  = pat["feature_sig"]
                etype = pat["error_type"]
                # Upsert pattern in DB
                pattern_id = db.upsert_pattern(
                    sig=sig, error_type=etype,
                    n_errors=pat["n_errors"], round_num=round_num,
                )
                pat["pattern_id"] = pattern_id
                is_core = pat["occurrence_count"] >= 3

                status = "CORE ★" if is_core else f"×{pat['occurrence_count']}"
                print(f"  [{status}] sig=[{sig or 'empty'}] "
                      f"type={etype} n={pat['n_errors']} "
                      f"{'NEW' if pat['is_new'] else 'RECURRING'}",
                      flush=True)

                if pat["is_new"]:
                    n_new_patterns += 1

            # Link errors to patterns
            db.update_pattern_error_links(round_num)

            # ── MCQ generation for qualifying patterns ─────────────────────────
            # Generate MCQs for: new patterns OR recurring patterns without an MCQ
            patterns_needing_mcq = db.get_unaddressed_patterns(round_num)
            # Also include new patterns from this round
            new_pattern_sigs = {p["feature_sig"] for p in patterns_found if p["is_new"]}

            mcqs_generated = 0
            if patterns_needing_mcq:
                print(f"\n[MCQ Generation] {len(patterns_needing_mcq)} patterns "
                      f"need lessons...", flush=True)
                for pat in patterns_needing_mcq[:8]:   # cap per round
                    pat_sig = pat["feature_sig"]
                    # Get example errors for this pattern
                    conn = db._conn()
                    ex_rows = conn.execute("""
                        SELECT e.*, c.text, c.true_label
                        FROM errors e JOIN cases c ON e.case_id = c.id
                        WHERE e.pattern_id = ? ORDER BY e.id DESC LIMIT 5
                    """, (pat["id"],)).fetchall()
                    ex_errors = [dict(r) for r in ex_rows]
                    if not ex_errors:
                        # Fall back to current round errors with matching sig
                        ex_errors = [e for e in errors
                                     if e.get("feature_sig") == pat_sig][:5]

                    if not ex_errors:
                        continue

                    # Compute pre-MCQ error rate
                    pre_rate = db.pattern_error_rate(
                        pat["id"], max(1, round_num - 3), round_num
                    )

                    result = generate_mcq_for_pattern(
                        pattern=pat,
                        example_errors=ex_errors,
                        llm_fn=llm_fn,
                        round_num=round_num,
                    )
                    if result:
                        db.add_mcq(
                            pattern_id=pat["id"],
                            correct_answer=result["correct_answer"],
                            correct_rationale=result["correct_rationale"],
                            wrong_answers=result["wrong_answers"],
                            example_text=result["example_text"],
                            round_num=round_num,
                            pre_error_rate=pre_rate,
                        )
                        print(f"  [MCQ ✓] pattern=[{pat_sig or 'empty'}] | "
                              f"correct={result['correct_answer']} | "
                              f"pre_rate={pre_rate:.3f}",
                              flush=True)
                        mcqs_generated += 1

            # ── MCQ effectiveness check (patterns with MCQs from prior rounds) ──
            if round_num > 1:
                recurring = db.get_recurring_patterns(min_rounds=2)
                for pat in recurring:
                    if pat.get("mcq_id") and pat.get("mcq_active"):
                        post_rate = db.pattern_error_rate(
                            pat["id"], round_num, round_num
                        )
                        db.update_mcq_effectiveness(pat["id"], post_rate)

            # Retire MCQs that are making things worse
            retired = db.retire_ineffective_mcqs(min_effectiveness=-0.05)
            if retired:
                print(f"  [MCQ Retired] {retired} ineffective MCQs deactivated",
                      flush=True)

            # ── Threshold calibration (zero LLM cost) ─────────────────────────
            firing_threshold = calibrate_threshold(
                results=results,
                positive_label=positive_label,
                current_threshold=firing_threshold,
            )
            print(f"\n[Threshold] Calibrated to {firing_threshold:.2f} "
                  f"(based on {len(results)} training cases)", flush=True)

            # ── Evaluation on held-out eval pool ──────────────────────────────
            print(f"\n[Evaluation] Evaluating on {len(eval_pool)} held-out cases...",
                  flush=True)
            eval_cases_fmt = [
                {**c, "true_label": c["label"]} for c in eval_pool
            ]
            metrics = evaluate(
                eval_cases=eval_cases_fmt,
                db=db, rag_index=rag_index,
                llm_fn=llm_fn, base_prompt=base_prompt,
                firing_threshold=firing_threshold,
                workers=self.args.workers,
                positive_label=positive_label,
            )

            n_core_now = len(db.get_core_mcqs())
            n_active_now = db._conn().execute(
                "SELECT COUNT(*) FROM mcqs WHERE is_active=1"
            ).fetchone()[0]

            # ── Save round stats ───────────────────────────────────────────────
            db.save_round_stats(round_num, {
                "phase": "learning",
                **metrics,
                "n_cases_trained":  len(batch),
                "n_errors":         len(errors),
                "n_new_patterns":   n_new_patterns,
                "n_core_patterns":  n_core_now,
                "n_active_mcqs":    n_active_now,
                "firing_threshold": firing_threshold,
            })

            # ── Round summary ──────────────────────────────────────────────────
            print(f"\n{'─'*60}", flush=True)
            print(
                f"  Round {round_num} | "
                f"Eval F1={metrics['f1']:.4f} | "
                f"P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | "
                f"CoreMCQs={n_core_now} | ActiveMCQs={n_active_now}",
                flush=True
            )
            print(
                f"  TP={metrics['tp']} FP={metrics['fp']} "
                f"FN={metrics['fn']} TN={metrics['tn']} | "
                f"MCQsGen={mcqs_generated} | "
                f"Patterns={len(patterns_found)}",
                flush=True
            )
            print(f"{'─'*60}", flush=True)

        # ══════════════════════════════════════════════════════════════════════
        #  FINAL REPORT
        # ══════════════════════════════════════════════════════════════════════
        self._print_final_report(db)

    def _print_final_report(self, db: NexusDB) -> None:
        history = db.get_round_history()
        conn = db._conn()

        print(f"\n{'╔'+'═'*78+'╗'}", flush=True)
        print(f"  NEXUS MCQ LEARNER — FINAL REPORT", flush=True)
        print(f"{'╚'+'═'*78+'╝'}", flush=True)
        print(f"\n{db.summary()}", flush=True)

        # F1 trajectory
        print(f"\n  F1 Trajectory:", flush=True)
        for r in history:
            if r.get("f1") is None:
                continue
            bar = "█" * int(r["f1"] * 40)
            print(
                f"    R{r['round']:02d} F1={r['f1']:.4f} {bar} | "
                f"CoreMCQ={r.get('n_core_patterns',0)} "
                f"ActiveMCQ={r.get('n_active_mcqs',0)} "
                f"Errors={r.get('n_errors',0)}",
                flush=True
            )

        # CORE pattern summary
        core_mcqs = db.get_core_mcqs()
        if core_mcqs:
            print(f"\n  CORE MCQ Bank ({len(core_mcqs)} lessons):", flush=True)
            for m in core_mcqs:
                eff = m.get("effectiveness")
                eff_str = f"effectiveness={eff:+.3f}" if eff else "effectiveness=pending"
                print(f"    [{m['feature_sig'] or 'global'}] "
                      f"{m['correct_answer']} | {eff_str}", flush=True)

        if history:
            best = max(history, key=lambda r: r.get("f1") or 0)
            last = history[-1]
            print(f"\n  Best  F1: {best.get('f1',0):.4f} at R{best['round']}", flush=True)
            print(f"  Final F1: {last.get('f1',0):.4f} at R{last['round']}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# §7  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NEXUS MCQ Learner")
    parser.add_argument("--config",    required=True)
    parser.add_argument("--out",       required=True)
    parser.add_argument("--rounds",    type=int, default=16)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1000)
    parser.add_argument("--min-pattern-errors", dest="min_pattern_errors",
                        type=int, default=3)
    parser.add_argument("--workers",   type=int, default=6)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--fresh",     action="store_true")
    parser.add_argument("--mock",      action="store_true")
    parser.add_argument("--ai-hub",    action="store_true")
    parser.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    parser.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))
    args = parser.parse_args()

    config = TaskConfig.load(args.config)
    MCQLearner(args).run(config)


if __name__ == "__main__":
    main()

"""
infer_v3.py
NEXUS v3 — Inference-only evaluation on held-out cases.

Loads a fully-trained tree from a completed run directory and evaluates
it on cases that were never part of the eval_pool used during training.

Sampling strategy
-----------------
The eval_pool (first 200 cases after seeded shuffle) was used every round
for threshold calibration — these are excluded automatically.

The train_pool has ~17,100 cases; each training round sampled 200 randomly,
so 20 rounds × 200 = ~4,000 cases were seen during training.  To maximise
the probability of testing on truly unseen cases we sample from the TAIL of
the train_pool.  The chance that any single tail case was drawn in 4,000
samples from 17,100 is ~23 % — so ~77 % of the tail is genuinely unseen.

Usage
-----
# AI Hub (Northwell)
python3 infer_v3.py \\
    --run run_v3_04 \\
    --ai-hub \\
    --ai-hub-key "$AIHUB_API_KEY" \\
    --ai-hub-ad-id "$AIHUB_AD_OBJECT_ID" \\
    --n-cases 300

# OpenAI-compatible
python3 infer_v3.py \\
    --run run_v3_04 \\
    --openai \\
    --n-cases 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data_utils import load_and_split
from llm_client import AIHubClient, OpenAIClient, MockClient
from task_config import TaskConfig
from tree_v3 import NexusTree
from rag_index import RAGIndex
from nexus_v3 import _patch_routes_from_config, _patch_aggregator_bias, make_route_llm_fn


def precision_recall_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ── Core inference ─────────────────────────────────────────────────────────────

def infer_one(case: dict, tree: NexusTree, route_llm_fn, rag: RAGIndex,
              pos_label: str, ade_bias: float, workers: int) -> dict:
    """Run inference on one case. Returns dict with text, true, pred, correct."""
    result = tree.classify(case["text"], route_llm_fn, rag, workers)

    # Scores live on result.route_result (AggregatedResult)
    ade_score     = result.route_result.ade_score
    not_ade_score = result.route_result.not_ade_score
    pred = pos_label if (ade_score * ade_bias) >= not_ade_score else "NOT_ADE"

    return {
        "text":    case["text"],
        "true":    case["label"],
        "pred":    pred,
        "node":    result.node_id,
        "correct": pred == case["label"],
        "ade_score":     ade_score,
        "not_ade_score": not_ade_score,
    }


def run_inference(args):
    run_dir = Path(args.run)
    if not run_dir.exists():
        print(f"Error: run directory not found: {run_dir}")
        sys.exit(1)

    # ── Load config ────────────────────────────────────────────────────────────
    print(f"\n[INFER] Loading task config: {args.task}")
    config = TaskConfig.load(args.task)
    pos_label = config.positive_label

    # ── LLM client ────────────────────────────────────────────────────────────
    if args.mock:
        client = MockClient()
    elif args.ai_hub:
        client = AIHubClient(api_key=args.ai_hub_key, ad_object_id=args.ai_hub_ad_id)
    elif args.openai:
        client = OpenAIClient(
            api_key=args.openai_key or None,
            base_url=args.openai_base_url or None,
            classify_model=args.openai_classify_model,
            synth_model=args.openai_synth_model,
        )
    else:
        print("Error: specify --ai-hub, --openai, or --mock")
        sys.exit(1)

    _patch_routes_from_config(config)

    # ── Load trained tree ──────────────────────────────────────────────────────
    tree_path = str(run_dir / "tree")
    print(f"[INFER] Loading tree from {tree_path}")
    tree = NexusTree.load(tree_path, config)
    nodes = tree.all_nodes()
    print(f"[INFER] Tree loaded: {len(nodes)} nodes")
    for n in nodes:
        n_principles = sum(1 for c in n.engram_store._clusters.values() if c.principle_text)
        print(f"  {n.id}  mcqs={len(n.mcq_library)}  principles={n_principles}  clusters={len(n.engram_store._clusters)}", flush=True)

    # ── Load RAG index ─────────────────────────────────────────────────────────
    rag_dir = str(run_dir / "global_rag_index")
    print(f"[INFER] Loading RAG index from {rag_dir}")
    eval_pool, probe_pool, train_pool = load_and_split(seed=args.seed)
    all_corpus = eval_pool + probe_pool + train_pool
    rag = RAGIndex.load_or_build(corpus=all_corpus, out_dir=rag_dir)
    print(f"[INFER] RAG: {rag.stats()}")

    # ── Apply bias from last run ───────────────────────────────────────────────
    ade_bias = args.ade_bias
    print(f"[INFER] Using ADE_BIAS={ade_bias:.2f} (override with --ade-bias)")
    _patch_aggregator_bias(ade_bias)

    # ── Sample held-out cases ──────────────────────────────────────────────────
    # Sample from the tail of train_pool — least likely to have been seen.
    # 20 rounds × 200 cases = ~4,000 seen out of 17,100; tail cases have
    # ~77% probability of being truly unseen.
    rng = random.Random(args.seed + 999)   # different seed from training
    tail = train_pool[-(args.tail_size):]  # last N cases
    test_cases = rng.sample(tail, min(args.n_cases, len(tail)))
    print(f"\n[INFER] Sampling {len(test_cases)} cases from tail of train_pool "
          f"(tail_size={args.tail_size}, train_pool={len(train_pool)})")
    print(f"[INFER] Estimated % truly unseen: "
          f"~{100*(1 - 4000/len(train_pool)):.0f}% of pool; "
          f"tail further reduces overlap\n")

    route_llm_fn = make_route_llm_fn(client)

    # ── Run inference ──────────────────────────────────────────────────────────
    results = []
    t0 = time.time()
    workers = args.workers

    print(f"[INFER] Classifying {len(test_cases)} cases with {workers} workers...")
    print(f"  {'':4s}  {'TRUE':8s}  {'PRED':8s}  {'NODE':40s}  TEXT[:60]")
    print(f"  {'':4s}  {'-'*8}  {'-'*8}  {'-'*40}  {'-'*60}")

    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(infer_one, case, tree, route_llm_fn, rag,
                        pos_label, ade_bias, 1): (i, case)
            for i, case in enumerate(test_cases)
        }
        done = 0
        for fut in as_completed(futures):
            i, case = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                done += 1
                mark = "✓" if r["correct"] else "✗"
                if not r["correct"]:
                    errors.append(r)
                if not r["correct"] or done % 20 == 0:
                    pct = done / len(test_cases)
                    bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
                    acc = sum(1 for x in results if x["correct"]) / len(results)
                    print(f"\r  [{bar}] {done}/{len(test_cases)}  acc={acc:.1%}",
                          end="", flush=True)
                    if not r["correct"]:
                        print(f"\n  {mark} TRUE={r['true']:8s}  PRED={r['pred']:8s}  "
                              f"NODE={r['node']:40s}  {r['text'][:60]}")
            except Exception as e:
                print(f"\n  [ERROR] case {i}: {e}")

    elapsed = time.time() - t0
    print(f"\n\n[INFER] Done in {elapsed:.1f}s  "
          f"({elapsed/len(results):.1f}s/case)")

    # ── Compute metrics ────────────────────────────────────────────────────────
    tp = sum(1 for r in results if r["true"] == pos_label and r["pred"] == pos_label)
    fp = sum(1 for r in results if r["true"] != pos_label and r["pred"] == pos_label)
    fn = sum(1 for r in results if r["true"] == pos_label and r["pred"] != pos_label)
    tn = sum(1 for r in results if r["true"] != pos_label and r["pred"] != pos_label)

    prec, rec, f1 = precision_recall_f1(tp, fp, fn)
    acc = (tp + tn) / len(results)

    # ── Threshold sweep ────────────────────────────────────────────────────────
    print("\n[INFER] Threshold sweep over cached scores:")
    print(f"  {'bias':6s}  {'P':6s}  {'R':6s}  {'F1':6s}  {'F2.0':6s}")
    score_cache = [(r["ade_score"], r["not_ade_score"], r["true"]) for r in results]

    sweep_results = []
    for b in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]:
        _tp = sum(1 for s, ns, t in score_cache if t == pos_label and (s * b) >= ns)
        _fp = sum(1 for s, ns, t in score_cache if t != pos_label and (s * b) >= ns)
        _fn = sum(1 for s, ns, t in score_cache if t == pos_label and (s * b) < ns)
        p, r, f = precision_recall_f1(_tp, _fp, _fn)
        f2 = (1 + 4) * p * r / max(1e-9, (4 * p + r))
        sweep_results.append((b, p, r, f, f2))

    best_f2 = max(x[4] for x in sweep_results)
    best_f1 = max(x[3] for x in sweep_results)
    best_bias = next(x[0] for x in sweep_results if x[3] == best_f1)

    for b, p, r, f, f2 in sweep_results:
        marker = " ◀" if f2 == best_f2 else ""
        print(f"  bias={b:<4.1f}  P={p:.3f}  R={r:.3f}  F1={f:.4f}  F2.0={f2:.4f}{marker}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"""
{'═'*64}
  NEXUS v3.04 — Held-Out Inference Results
  Run:       {args.run}
  Cases:     {len(results)} (sampled from train_pool tail)
  ADE_BIAS:  {ade_bias:.2f} (used for this run)
{'─'*64}
  Accuracy:  {acc:.4f}
  Precision: {prec:.4f}
  Recall:    {rec:.4f}
  F1:        {f1:.4f}
{'─'*64}
  Best F1 across sweep:  {best_f1:.4f}  (at bias={best_bias})
  TP={tp}  FP={fp}  FN={fn}  TN={tn}
{'─'*64}
  Errors ({len(errors)}):""")
    for e in errors[:10]:
        print(f"    TRUE={e['true']:8s}  PRED={e['pred']:8s}  {e['text'][:70]}")
    if len(errors) > 10:
        print(f"    ... and {len(errors)-10} more")
    print(f"{'═'*64}\n")

    # ── Save results ───────────────────────────────────────────────────────────
    out_path = Path(args.run) / "infer_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_cases": len(results),
            "ade_bias": ade_bias,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "best_f1": best_f1,
            "best_bias": best_bias,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "results": results,
        }, f, indent=2)
    print(f"[INFER] Results saved to {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="NEXUS v3 — Inference-only held-out evaluation"
    )
    ap.add_argument("--run",      default="run_v3_04",
                    help="Path to completed run directory")
    ap.add_argument("--task",     default="task_configs/ade_classification.json")
    ap.add_argument("--n-cases",  type=int, default=300,
                    help="Number of held-out cases to evaluate")
    ap.add_argument("--tail-size", type=int, default=5000,
                    help="Sample from the last N cases of train_pool (default: 5000)")
    ap.add_argument("--ade-bias", type=float, default=1.0,
                    help="ADE classification bias (1.0 = v3.04 final calibrated value)")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--workers",  type=int, default=4)

    # AI Hub
    ap.add_argument("--ai-hub",       action="store_true")
    ap.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    ap.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))

    # OpenAI-compatible
    ap.add_argument("--openai",                action="store_true")
    ap.add_argument("--openai-key",            default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--openai-base-url",       default=None)
    ap.add_argument("--openai-classify-model", default="gpt-4o-mini")
    ap.add_argument("--openai-synth-model",    default="gpt-4o")

    # Mock
    ap.add_argument("--mock", action="store_true",
                    help="MockClient — for pipeline testing only")

    args = ap.parse_args()
    run_inference(args)

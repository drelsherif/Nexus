"""
run_rag.py
NEXUS RAG + Parallel Expert Routes + Semantic Engram Learning

Full pipeline:
  1. Build FAISS vector index over ADE corpus (once, cached)
  2. For each case: retrieve K similar labeled examples
  3. Fan out to 4 parallel expert routes (causation / negation / drug-effect / context)
  4. Aggregate votes with confidence weighting
  5. Update route weights based on ground truth (reinforcement)
  6. Errors → semantic engram store (cluster by embedding proximity)
  7. Clusters reach threshold → LLM consolidation → NEXUS principle
  8. Principles injected into node prompts (live, next case benefits)
  9. Eval every round

Install dependencies first:
    pip3 install sentence-transformers faiss-cpu --break-system-packages

Usage:
    python3 -u run_rag.py \\
        --ai-hub \\
        --ai-hub-key   $AIHUB_API_KEY \\
        --ai-hub-ad-id $AIHUB_AD_OBJECT_ID \\
        --out run_07_rag \\
        --rounds 20 --workers 4 --threshold 5 --k 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

from data_utils import load_and_split
from llm_client import AIHubClient, MockClient
from expert_routes import RouteAggregator
from rag_index import RAGIndex
from semantic_engram import SemanticEngramStore
from tree import seed_tree, save_tree, load_tree, classify_with_tree
from features import features as extract_features


# ─── LLM callable factories ───────────────────────────────────────────────────

def make_route_llm_fn(client) -> Callable[[str, str], str]:
    """
    For expert routes: returns JSON string from classify model (haiku).
    Routes parse the response as {"vote":..., "confidence":..., "reasoning":...}
    """
    def fn(system: str, user: str) -> str:
        result, _, _, _ = client.classify(user, system)
        return json.dumps(result)
    return fn


def make_freeform_llm_fn(client) -> Callable[[str, str], str]:
    """
    For principle consolidation: returns raw freeform text from synth model (sonnet).
    Used by SemanticEngramStore.consolidate() to write NEXUS principles.
    """
    if hasattr(client, "chat"):
        def fn(system: str, user: str) -> str:
            return client.chat(system=system, user=user)
        return fn
    # MockClient fallback
    def fn(system: str, user: str) -> str:
        result, _, _, _ = client.classify(user, system)
        return f"NEXUS PRINCIPLE: Mock principle — {result.get('rationale', 'no detail')}"
    return fn


# ─── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(
    cases: list[dict],
    rag_index: RAGIndex,
    aggregator: RouteAggregator,
    llm_fn: Callable,
    k: int = 5,
    workers: int = 4,
) -> dict:
    tp = fp = fn = tn = 0
    for c in cases:
        examples = rag_index.query(c["text"], k=k)
        result = aggregator.classify(c["text"], examples, llm_fn, workers=workers, principle_context="")
        pred = result.final_label
        true = c["label"]
        if pred == "ADE"     and true == "ADE":     tp += 1
        elif pred == "ADE"   and true == "NOT_ADE": fp += 1
        elif pred == "NOT_ADE" and true == "ADE":   fn += 1
        else:                                        tn += 1
    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)
    return {
        "f1": round(f1, 4), "precision": round(prec, 4),
        "recall": round(recall, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─── Principle injection into tree node ───────────────────────────────────────

def inject_principle_into_tree(
    tree: dict, node_ids: list[str], principle_id: str, text: str
) -> dict:
    targets = {tree["id"]: tree}
    for child in tree.get("children", []):
        targets[child["id"]] = child
    for nid in node_ids:
        if nid in targets:
            targets[nid]["prompt"] += f"\n\n[PRINCIPLE {principle_id}]\n{text}"
    return tree


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_rag_loop(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # LLM client
    if args.mock:
        client = MockClient()
    elif args.ai_hub:
        client = AIHubClient(api_key=args.ai_hub_key, ad_object_id=args.ai_hub_ad_id)
    else:
        print("Error: specify --ai-hub or --mock"); sys.exit(1)

    route_llm_fn    = make_route_llm_fn(client)     # JSON, haiku — for routes
    freeform_llm_fn = make_freeform_llm_fn(client)  # text, sonnet — for principles

    print(f"\n{'═'*64}")
    print(f"  NEXUS RAG + PARALLEL EXPERT ROUTES")
    print(f"  Out: {args.out}  Rounds: {args.rounds}")
    print(f"  Workers: {args.workers}  k={args.k}  Threshold: {args.threshold}")
    print(f"{'═'*64}\n")

    # Load corpus
    print("[RAG] Loading ADE Corpus v2...")
    eval_pool, probe_pool, train_pool = load_and_split(seed=args.seed)
    all_corpus = eval_pool + probe_pool + train_pool
    print(f"[RAG] Total: {len(all_corpus)} | Train: {len(train_pool)} | Eval: {len(eval_pool)}")

    # Build or load RAG index
    rag_dir = str(out / "rag_index")
    rag_index = RAGIndex.load_or_build(
        corpus=all_corpus,
        out_dir=rag_dir,
        model_name=args.embed_model or None,
    )
    print(f"[RAG] Index ready — {rag_index.stats()}")

    # Route aggregator
    agg_path = out / "aggregator.json"
    if agg_path.exists() and not args.fresh:
        aggregator = RouteAggregator.from_dict(json.loads(agg_path.read_text()))
        print(f"[RAG] Loaded aggregator weights from {agg_path}")
    else:
        aggregator = RouteAggregator()
        print("[RAG] Fresh route aggregator")

    # Semantic engram store
    engram_dir = str(out / "engrams")
    engram_store = (
        SemanticEngramStore.load(engram_dir)
        if (Path(engram_dir) / "semantic_engrams.json").exists() and not args.fresh
        else SemanticEngramStore(threshold=args.threshold, path=engram_dir)
    )

    # Tree (for principle injection storage)
    tree_path = args.tree
    if tree_path and Path(tree_path).exists():
        tree = load_tree(tree_path)
        print(f"[RAG] Loaded tree from {tree_path}")
    else:
        tree = seed_tree()
        print("[RAG] Using seed tree")

    tree_ref = [tree]

    # Principles log
    principles_path = out / "rag_principles.json"
    principles: dict[str, str] = (
        json.loads(principles_path.read_text())
        if principles_path.exists() and not args.fresh
        else {}
    )

    # ── Baseline eval ──────────────────────────────────────────────────────────
    print("\n[RAG] Baseline eval (100 cases, 4 routes × k=5 examples each)...")
    print("      This makes 400+ LLM calls — takes ~5-8 minutes.")
    t0 = time.time()
    baseline = evaluate(eval_pool[:100], rag_index, aggregator, route_llm_fn,
                        k=args.k, workers=args.workers)
    print(f"[RAG] Baseline F1={baseline['f1']:.4f}  "
          f"P={baseline['precision']:.3f}  R={baseline['recall']:.3f}  "
          f"elapsed={( time.time()-t0)/60:.1f}m")

    best_f1 = baseline["f1"]
    history: list[dict] = []
    rng = random.Random(args.seed)

    for rnd in range(1, args.rounds + 1):
        t_round = time.time()
        print(f"\n{'─'*64}")
        print(f"  Round {rnd}/{args.rounds}")
        print(f"{'─'*64}")

        batch = rng.sample(train_pool, min(args.batch_size, len(train_pool)))
        correct = 0
        errors = 0
        swr_events = 0
        split_cases = 0

        for i, case in enumerate(batch):
            text = case["text"]
            true_label = case["label"]

            # Retrieve similar examples
            examples = rag_index.query(text, k=args.k)

            # Retrieve relevant engram principles (live injection)
            relevant_principles = engram_store.retrieve_principles(text, top_k=2)

            # Build principle prefix for route system prompts (full text, not truncated)
            principle_context = ""
            if relevant_principles:
                blocks = "\n\n".join(
                    f"[ENGRAM {i+1}] {eng['principle']}"
                    for i, eng in enumerate(relevant_principles)
                )
                principle_context = (
                    f"\n\nThe following NEXUS principles were learned from similar cases "
                    f"and MUST be considered before voting:\n\n{blocks}"
                )

            # Parallel expert classification (principles injected via prefix)
            result = aggregator.classify(
                text, examples, route_llm_fn,
                workers=args.workers,
                principle_context=principle_context,
            )

            if result.split:
                split_cases += 1

            # Update route weights
            aggregator.update_weights(result, true_label)

            # Track accuracy
            if result.final_label == true_label:
                correct += 1
            else:
                errors += 1
                # Add to semantic engram store
                node_id = "ROOT"  # RAG mode doesn't use tree routing for now
                fired = engram_store.add_error(
                    text=text,
                    true_label=true_label,
                    predicted_label=result.final_label,
                    node_id=node_id,
                    round_num=rnd,
                )
                if fired:
                    cluster_id, cluster_errors = fired
                    swr_events += 1
                    print(f"\n  ★ SWR EVENT — cluster {cluster_id} "
                          f"({len(cluster_errors)} errors, label={cluster_errors[0].true_label})")
                    principle = engram_store.consolidate(cluster_id, freeform_llm_fn, rnd)
                    if principle:
                        pid = f"RAG_R{rnd}_{len(principles)+1}"
                        principles[pid] = principle
                        principles_path.write_text(json.dumps(principles, indent=2))
                        print(f"  ✓ {pid} consolidated ({len(principle)} chars)")
                    else:
                        print(f"  ✗ Consolidation failed for cluster {cluster_id}")

            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(batch):
                pct = (i + 1) / len(batch)
                bar = "█" * int(pct * 20) + " " * (20 - int(pct * 20))
                acc = correct / max(1, i + 1)
                print(f"\r  [{bar}] {i+1}/{len(batch)}  "
                      f"acc={acc:.1%}  errors={errors}  splits={split_cases}",
                      end="", flush=True)

        print()  # newline after progress

        # Save aggregator weights
        agg_path.write_text(json.dumps(aggregator.to_dict(), indent=2))

        # Print route weights
        print(aggregator.weight_report())
        engram_store.print_report()
        engram_store.save()

        # Eval
        print(f"\n[RAG] Round {rnd} eval...")
        metrics = evaluate(eval_pool[:100], rag_index, aggregator, route_llm_fn,
                           k=args.k, workers=args.workers)
        elapsed = time.time() - t0
        eta = (elapsed / rnd) * (args.rounds - rnd)
        print(f"  [R{rnd}/{args.rounds}] "
              f"EVAL F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"elapsed={elapsed/60:.1f}m  ETA ~{eta/60:.1f}m")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            delta = best_f1 - baseline["f1"]
            print(f"  ★ NEW BEST  F1={best_f1:.4f}  (+{delta:.4f} vs baseline)")

        history.append({
            "round": rnd,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "batch_accuracy": round(correct / max(1, len(batch)), 4),
            "errors": errors,
            "split_cases": split_cases,
            "swr_events": swr_events,
            "principles_total": len(principles),
        })

        (out / "rag_summary.json").write_text(json.dumps({
            "best_f1": best_f1,
            "baseline_f1": baseline["f1"],
            "rounds_run": rnd,
            "history": history,
        }, indent=2))

    print(f"\n{'═'*64}")
    print(f"  RAG LEARNING COMPLETE")
    print(f"  Best F1:    {best_f1:.4f}  "
          f"(baseline {baseline['f1']:.4f}, Δ+{best_f1 - baseline['f1']:.4f})")
    print(f"  Principles: {len(principles)}")
    eng_stats = engram_store.summary_stats()
    print(f"  Clusters:   {eng_stats['total_clusters']} "
          f"| Promoted: {eng_stats['promoted']}")
    print(f"  Output:     {args.out}/")
    print(f"{'═'*64}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NEXUS RAG + Expert Routes Learning")
    ap.add_argument("--tree", default=None,
                    help="Optional starting tree (omit for seed tree)")
    ap.add_argument("--out", default="run_07_rag")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=50,
                    help="Cases per round for learning (smaller = faster iteration)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel routes per classification (4 = all routes)")
    ap.add_argument("--k", type=int, default=5,
                    help="Retrieved examples per query")
    ap.add_argument("--threshold", type=int, default=5,
                    help="Cluster size to trigger SWR consolidation")
    ap.add_argument("--embed-model", default=None,
                    help="Override embedding model (default: auto-select biomedical)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing index/aggregator/engrams")
    # AI Hub
    ap.add_argument("--ai-hub", action="store_true")
    ap.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    ap.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))
    # Mock
    ap.add_argument("--mock", action="store_true",
                    help="MockClient — no API cost, for pipeline testing")

    args = ap.parse_args()
    run_rag_loop(args)


if __name__ == "__main__":
    main()

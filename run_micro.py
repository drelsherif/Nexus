"""
run_micro.py
NEXUS Micro-Learning Mode — case-level Hebbian learning.

Replaces the batch→probe→refine loop with:
  N parallel workers × 1 case each
  → micro-rule on each misclassification (no probe needed)
  → RuleDictionary accumulation (Hebbian LTP, zero LLM cost)
  → threshold crossing → SWR consolidation event (one LLM call)
  → principle injected into node prompt automatically
  → next cases benefit immediately

Usage:
    # Start from seed tree (no prior run needed):
    python3 run_micro.py \\
        --ai-hub \\
        --ai-hub-key   $AIHUB_API_KEY \\
        --ai-hub-ad-id $AIHUB_AD_OBJECT_ID \\
        --out    run_06_micro \\
        --rounds 20 --workers 10 --threshold 5

    # Continue from a saved tree:
    python3 run_micro.py \\
        --ai-hub ... \\
        --tree run_05_branch_C/nexus_best_tree.json \\
        --out  run_06_micro --rounds 20

    # Mock mode (no API cost, for testing):
    python3 run_micro.py --mock --out run_micro_test --rounds 3
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

# ─── NEXUS core imports ───────────────────────────────────────────────────────
from features import features as extract_features
from tree import seed_tree, classify_with_tree, save_tree, load_tree
from llm_client import AIHubClient, MockClient
from drug_registry import DrugRegistry
from data_utils import load_and_split
from micro_learner import run_micro_batch, build_consolidation_call
from rule_dictionary import RuleDictionary


# ─── Classification wrapper ───────────────────────────────────────────────────

def make_classify_fn(tree_ref: list, client) -> Callable[[str], tuple[str, str]]:
    """
    Returns classify_fn(text) → (node_id, label).
    tree_ref is a mutable 1-element list so principle injection is live.
    Uses features() + classify_with_tree() + client.classify() —
    the exact same pipeline as the main nexus_run.py loop.
    """
    def classify_fn(text: str) -> tuple[str, str]:
        tree = tree_ref[0]
        feats = extract_features(text)
        node = classify_with_tree(tree, feats)
        result, _, _, _ = client.classify(text, node["prompt"])
        label = result.get("classification", "NOT_ADE")
        return node["id"], label
    return classify_fn


def make_rule_gen_fn(client) -> Callable[[str, str], str]:
    """Returns a (system, user) → str callable for rule/principle generation."""
    def rule_gen_fn(system: str, user: str) -> str:
        # AIHubClient.classify expects (text, node_prompt); for freeform calls
        # we use the generic chat method if available, else wrap as classify.
        if hasattr(client, "chat"):
            return client.chat(system=system, user=user)
        # Fallback: re-use classify with system as prompt, user as text
        result, _, _, _ = client.classify(user, system)
        return json.dumps(result)
    return rule_gen_fn


# ─── Corpus ───────────────────────────────────────────────────────────────────

def load_corpus(seed: int = 42) -> tuple[list[dict], list[dict]]:
    """
    Load ADE Corpus v2 from HuggingFace via data_utils.load_and_split().
    Returns (train_pool, eval_pool) matching the same fixed split nexus_run uses.
    """
    eval_pool, probe_pool, train_pool = load_and_split(seed=seed)
    # eval_pool is the fixed 200-case set; train is the rest
    return train_pool, eval_pool


# ─── Eval ─────────────────────────────────────────────────────────────────────

def evaluate(classify_fn: Callable, cases: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    for c in cases:
        _, pred = classify_fn(c["text"])
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


# ─── Principle injection ──────────────────────────────────────────────────────

def inject_principle(tree: dict, node_ids: list[str], pid: str, text: str) -> dict:
    """
    Append a consolidated principle to the relevant nodes' prompts.
    Operates on root and one level of children (NEXUS tree is shallow).
    Returns the same tree dict (modified in place).
    """
    targets = {tree["id"]: tree}
    for child in tree.get("children", []):
        targets[child["id"]] = child

    for nid in node_ids:
        if nid in targets:
            existing = targets[nid].get("prompt", "")
            targets[nid]["prompt"] = existing + f"\n\n[PRINCIPLE {pid}]\n{text}"
    return tree


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_micro_loop(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build LLM client
    if args.mock:
        client = MockClient()
    elif args.ai_hub:
        client = AIHubClient(
            api_key=args.ai_hub_key,
            ad_object_id=args.ai_hub_ad_id,
        )
    else:
        print("Error: specify --ai-hub or --mock"); sys.exit(1)

    # Load tree (seed or from file)
    if args.tree and Path(args.tree).exists():
        tree = load_tree(args.tree)
        print(f"[micro] Loaded tree from {args.tree}")
    else:
        tree = seed_tree()
        print("[micro] Using seed tree (no --tree file specified or found)")

    # tree_ref: mutable container so classify_fn always sees current tree
    tree_ref = [tree]

    classify_fn = make_classify_fn(tree_ref, client)
    rule_gen_fn = make_rule_gen_fn(client)

    # Load corpus (ADE Corpus v2 via HuggingFace, same split as nexus_run)
    print("[micro] Loading ADE Corpus v2...")
    train_cases, eval_cases = load_corpus(seed=args.seed)
    print(f"[micro] Train: {len(train_cases)}  Eval: {len(eval_cases)}")

    # Load RuleDictionary
    rd_path = str(out / "rule_dict.json")
    rule_dict = (
        RuleDictionary.load(rd_path)
        if (out / "rule_dict.json").exists() and not args.fresh
        else RuleDictionary(threshold=args.threshold, path=rd_path)
    )

    # Load DrugRegistry
    dr_path = str(out / "drug_registry.json")
    drug_registry = (
        DrugRegistry.load(dr_path)
        if (out / "drug_registry.json").exists() and not args.fresh
        else DrugRegistry(path=dr_path)
    )

    # Principles store
    principles_path = out / "micro_principles.json"
    principles: dict[str, str] = (
        json.loads(principles_path.read_text())
        if principles_path.exists() and not args.fresh
        else {}
    )

    print(f"\n{'═'*62}")
    print(f"  NEXUS MICRO-LEARNING")
    print(f"  Out: {args.out}  Rounds: {args.rounds}  "
          f"Workers: {args.workers}  Threshold: {args.threshold}")
    print(f"{'═'*62}\n")

    # Baseline eval
    print("[micro] Baseline eval (100 cases)...")
    t0 = time.time()
    baseline = evaluate(classify_fn, eval_cases[:100])
    print(f"[micro] Baseline F1={baseline['f1']:.4f}  "
          f"P={baseline['precision']:.3f}  R={baseline['recall']:.3f}")

    best_f1 = baseline["f1"]
    history: list[dict] = []
    rng = random.Random(args.seed)

    for rnd in range(1, args.rounds + 1):
        t_round = time.time()
        print(f"\n{'─'*62}")
        print(f"  Round {rnd}/{args.rounds}")
        print(f"{'─'*62}")

        # Sample batch
        batch = rng.sample(train_cases, min(args.batch_size, len(train_cases)))

        # Hebbian drug observation
        for c in batch:
            drug_registry.observe(c["text"], c["label"])

        # Parallel micro-learning
        batch_result = run_micro_batch(
            cases=batch,
            classify_fn=classify_fn,
            rule_gen_fn=rule_gen_fn,
            rule_dict=rule_dict,
            round_num=rnd,
            workers=args.workers,
            verbose=True,
        )

        errors = batch_result.error_count
        swr_count = len(batch_result.fired_keys)
        print(f"  Accuracy: {batch_result.accuracy:.1%}  "
              f"({errors} errors → {swr_count} SWR event(s))")

        # LTD: decay non-firing rules
        pruned = rule_dict.apply_ltd(rnd)
        if pruned:
            print(f"  [LTD] Pruned {pruned} weak rules")

        # SWR → consolidation events
        for key in batch_result.fired_keys:
            entry = rule_dict.get_entry(key)
            if not entry:
                continue
            print(f"\n  ★ SWR — consolidating: \"{entry.pattern[:55]}\"")
            principle_text = build_consolidation_call(key, rule_dict, rule_gen_fn)
            if principle_text:
                pid = f"ML_R{rnd}_{len(principles)+1}"
                principles[pid] = principle_text
                principles_path.write_text(json.dumps(principles, indent=2))
                rule_dict.mark_promoted(key, pid, rnd)
                # Inject into nodes that triggered the errors
                tree_ref[0] = inject_principle(
                    tree_ref[0], entry.node_ids, pid, principle_text
                )
                print(f"  ✓ {pid} → injected into {entry.node_ids}")
            else:
                print(f"  ✗ Consolidation failed")

        # Drug engrams → nuggets
        for drug in drug_registry.engrams_ready():
            nugget_text = drug_registry.build_nugget_text(drug)
            if nugget_text:
                pid = f"DRUG_{drug.upper().replace('-','_')}_R{rnd}"
                # Inject into ROOT as a background pharmacology fact
                tree_ref[0]["prompt"] += f"\n\n[DRUG ENGRAM {pid}]\n{nugget_text}"
                print(f"  [Hebbian] engram → {pid}")

        # Eval
        metrics = evaluate(classify_fn, eval_cases[:200])
        elapsed = time.time() - t0
        eta = (elapsed / rnd) * (args.rounds - rnd)
        print(f"\n  [R{rnd}/{args.rounds}] "
              f"EVAL F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"elapsed={elapsed/60:.1f}m  ETA ~{eta/60:.1f}m")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            delta = best_f1 - baseline["f1"]
            print(f"  ★ NEW BEST  F1={best_f1:.4f}  (+{delta:.4f} vs baseline)")
            save_tree(tree_ref[0], str(out / "nexus_micro_best_tree.json"))

        history.append({
            "round": rnd,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "errors_in_batch": errors,
            "swr_events": swr_count,
            "principles_total": len(principles),
        })

        # Save state
        rule_dict.save()
        drug_registry.save(dr_path)
        save_tree(tree_ref[0], str(out / "nexus_micro_final_tree.json"))
        (out / "micro_summary.json").write_text(json.dumps({
            "best_f1": best_f1,
            "baseline_f1": baseline["f1"],
            "rounds_run": rnd,
            "history": history,
        }, indent=2))

        rule_dict.print_report()

    # Final save
    save_tree(tree_ref[0], str(out / "nexus_micro_final_tree.json"))
    print(f"\n{'═'*62}")
    print(f"  MICRO-LEARNING COMPLETE")
    print(f"  Best F1:    {best_f1:.4f}  "
          f"(baseline {baseline['f1']:.4f}, Δ+{best_f1 - baseline['f1']:.4f})")
    print(f"  Principles: {len(principles)}")
    print(f"  Rules:      {rule_dict.summary_stats()['total_rules']}")
    print(f"  Output:     {args.out}/")
    print(f"{'═'*62}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="NEXUS Micro-Learning Mode")
    ap.add_argument("--tree", default=None,
                    help="Path to saved nexus_*_tree.json (omit to start from seed tree)")
    ap.add_argument("--out", default="run_micro_01")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=100,
                    help="Cases per round (split across workers)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--threshold", type=int, default=5,
                    help="SWR threshold — rule promoted after this many observations")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fresh", action="store_true",
                    help="Ignore existing rule_dict and principles")
    ap.add_argument("--max-graft-route-share", type=float, default=0.35,
                    help="(unused in micro mode, kept for compatibility)")
    # AI Hub
    ap.add_argument("--ai-hub", action="store_true")
    ap.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    ap.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))
    # Mock
    ap.add_argument("--mock", action="store_true",
                    help="Use MockClient (no API cost, for pipeline testing)")

    args = ap.parse_args()
    run_micro_loop(args)


if __name__ == "__main__":
    main()

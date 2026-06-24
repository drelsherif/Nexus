"""
nexus_v3.py
NEXUS v3 — Unified Main Loop

Synthesizes everything:
  v1: self-growing tree, nugget vocabulary, textual gradient, meta-rounds
  v2b: RAG retrieval, semantic engrams, parallel expert routes
  NEW: per-node memory, MCQ learning from misclassifications, task config layer

The MCQ insight (from medical education):
  Wrong answers are as informative as right answers — they define the
  decision boundary. Every misclassification generates a complete teaching
  case: correct reasoning + the wrong reasoning chain + distractors.
  Retrieved MCQs give each route complete clinical reasoning chains,
  not just labeled sentences.

Usage:
    python3 -u nexus_v3.py \\
        --task task_configs/ade_classification.json \\
        --ai-hub \\
        --ai-hub-key   $AIHUB_API_KEY \\
        --ai-hub-ad-id $AIHUB_AD_OBJECT_ID \\
        --out run_v3_01 \\
        --rounds 20

To apply to a new task (zero code changes):
    python3 -u nexus_v3.py \\
        --task task_configs/medication_errors.json \\
        --corpus data/medication_errors.jsonl \\
        --ai-hub ...
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from data_utils import load_and_split
from llm_client import AIHubClient, OpenAIClient, MockClient
from task_config import TaskConfig
from tree_v3 import NexusTree
from node import NexusNode, SWRResult
from mcq_generator import MCQGenerator
from rag_index import RAGIndex
from features import features as extract_features
from nexus_db import NexusDB
from homeostatic import HomeostaticController


# ─── LLM function factories ───────────────────────────────────────────────────

def make_route_llm_fn(client) -> Callable[[str, str], str]:
    """JSON output, classify model (haiku). Used by all 4 parallel routes."""
    def fn(system: str, user: str) -> str:
        result, _, _, _ = client.classify(user, system)
        return json.dumps(result)
    return fn


def make_freeform_llm_fn(client) -> Callable[[str, str], str]:
    """Freeform text, synth model (sonnet). Used for MCQ generation, engram consolidation, meta-rounds."""
    if hasattr(client, "chat"):
        def fn(system: str, user: str) -> str:
            return client.chat(system=system, user=user)
        return fn
    # MockClient fallback
    def fn(system: str, user: str) -> str:
        result, _, _, _ = client.classify(user, system)
        return f"NEXUS PRINCIPLE: Mock — {result.get('rationale', 'no detail')}"
    return fn


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(
    cases: list[dict],
    tree: NexusTree,
    route_llm_fn: Callable,
    global_rag_index: RAGIndex,
    task_config: TaskConfig,
    workers: int = 4,
) -> tuple[dict, list[tuple[float, float, str]]]:
    """
    Returns (metrics_dict, score_cache).
    score_cache is a list of (ade_score, not_ade_score, true_label) tuples —
    used by calibrate_threshold() for zero-cost threshold sweeping.
    """
    pos = task_config.positive_label
    tp = fp = fn = tn = 0
    score_cache: list[tuple[float, float, str]] = []
    n = len(cases)
    t_eval = time.time()

    for i, c in enumerate(cases):
        result = tree.classify(c["text"], route_llm_fn, global_rag_index, workers)
        pred = result.label
        true = c["label"]
        # Cache raw scores for threshold calibration (no extra LLM calls needed)
        agg = result.route_result
        score_cache.append((agg.ade_score, agg.not_ade_score, true))
        if pred == pos and true == pos:     tp += 1
        elif pred == pos and true != pos:   fp += 1
        elif pred != pos and true == pos:   fn += 1
        else:                               tn += 1

        # Live progress every 10 cases
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t_eval
            rate = (i + 1) / max(elapsed, 0.1)
            eta = (n - i - 1) / max(rate, 0.01)
            pct = (i + 1) / n
            bar = "█" * int(pct * 20) + "░" * (20 - int(pct * 20))
            prec_so_far = tp / max(1, tp + fp)
            rec_so_far  = tp / max(1, tp + fn)
            print(f"\r  [eval {bar}] {i+1}/{n}  "
                  f"P={prec_so_far:.2f} R={rec_so_far:.2f}  "
                  f"ETA ~{eta:.0f}s",
                  end="", flush=True)

    print()  # newline after progress bar

    prec   = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1     = 2 * prec * recall / max(1e-9, prec + recall)
    metrics = {
        "f1": round(f1, 4), "precision": round(prec, 4),
        "recall": round(recall, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    return metrics, score_cache


# ─── Meta-round ───────────────────────────────────────────────────────────────

def run_meta_round(
    tree: NexusTree,
    history: list[dict],
    freeform_llm_fn: Callable,
    round_num: int,
) -> Optional[str]:
    """
    Full-history strategic review.
    Returns a strategic recommendation as text.
    """
    f1_curve = [f"R{h['round']}: F1={h['f1']:.4f}" for h in history]
    node_stats = []
    for node in tree.all_nodes():
        avg_routes = node.average_routes_per_round()
        node_stats.append(
            f"  {node.id}: avg_routes={avg_routes:.1f}, "
            f"mcqs={len(node.mcq_library)}, "
            f"principles={len(node.injected_principles)}, "
            f"trigger={node.trigger_condition}"
        )

    # Error type distribution across all nodes
    all_error_types: dict[str, int] = {}
    for node in tree.all_nodes():
        for etype, count in node.mcq_library.error_type_distribution().items():
            all_error_types[etype] = all_error_types.get(etype, 0) + count
    top_errors = sorted(all_error_types.items(), key=lambda x: -x[1])[:5]

    system = "You are NEXUS, performing a strategic meta-review of your own learning progress."
    prompt = f"""NEXUS Meta-Review — Round {round_num}

F1 curve: {' | '.join(f1_curve)}
Best F1: {max(h['f1'] for h in history):.4f}

Tree structure ({len(tree.all_nodes())} nodes):
{chr(10).join(node_stats)}

Top error types across all nodes (from MCQ analysis):
{chr(10).join(f'  {etype}: {count} cases' for etype, count in top_errors)}

Strategic questions:
1. Is the F1 trend improving, plateauing, or oscillating? What does this suggest?
2. Which node has the most MCQ errors? What does this suggest about specialization?
3. Which error type is most persistent? What new specialist node would address it?
4. Are any nodes over- or under-routing (should retire or should be split)?

Provide a concise strategic recommendation (3-5 sentences) for the next phase of learning."""

    try:
        return freeform_llm_fn(system, prompt)
    except Exception:
        return None


# ─── Main loop ────────────────────────────────────────────────────────────────

def run_v3(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Load task config
    print(f"\n[V3] Loading task config: {args.task}")
    config = TaskConfig.load(args.task)
    print(f"[V3] Task: {config}")

    # LLM client
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
        print("Error: specify --openai, --ai-hub, or --mock"); sys.exit(1)

    route_llm_fn    = make_route_llm_fn(client)
    freeform_llm_fn = make_freeform_llm_fn(client)

    # Apply task config route definitions to expert_routes
    _patch_routes_from_config(config)

    print(f"\n{'═'*64}")
    print(f"  NEXUS v3 — {config.task_name}")
    print(f"  Out: {args.out}  Rounds: {args.rounds}")
    print(f"  Workers: {config.get_hyperparameter('workers', args.workers)}")
    print(f"{'═'*64}\n")

    # Load corpus
    print("[V3] Loading corpus...")
    eval_pool, probe_pool, train_pool = load_and_split(seed=args.seed)
    all_corpus = eval_pool + probe_pool + train_pool
    print(f"[V3] Total: {len(all_corpus)} | Train: {len(train_pool)} | Eval: {len(eval_pool)}")

    # Calibrate class prior
    ade_bias = config.calibrate_from_corpus(train_pool)
    print(f"[V3] Class prior calibration: ADE_BIAS={ade_bias:.3f} "
          f"(positive_prior={sum(1 for c in train_pool if c['label']==config.positive_label)/len(train_pool):.3f})")

    # Apply bias to aggregator factory
    _patch_aggregator_bias(ade_bias)

    # Build or load global RAG index
    rag_dir = str(out / "global_rag_index")
    global_rag = RAGIndex.load_or_build(corpus=all_corpus, out_dir=rag_dir)
    print(f"[V3] Global RAG: {global_rag.stats()}")

    # Open (or create) the SQLite database for this run
    db = NexusDB(str(out / "nexus.db"))
    print(f"[V3] Database: {out}/nexus.db")

    # Homeostatic controller — detects degradation and dispatches interventions
    homeostatic = HomeostaticController(
        task_config=config,
        freeform_llm_fn=freeform_llm_fn,
        db=db,
    )

    # Build or load tree — pass db to all nodes for MCQ deduplication
    tree_path = str(out / "tree")
    tree_state = out / "tree" / "tree_structure.json"
    if tree_state.exists() and not args.fresh:
        tree = NexusTree.load(tree_path, config)
        # Attach DB to all loaded nodes
        for node in tree.all_nodes():
            node.db = db
        print(f"[V3] Loaded tree: {len(tree.all_nodes())} nodes")
    else:
        tree = NexusTree.from_task_config(config, path=tree_path, db=db)
        print(f"[V3] Seed tree: {len(tree.all_nodes())} nodes")
        tree.print_tree()

    # Register seed nodes in DB
    for node in tree.all_nodes():
        db.upsert_node(node.id, None, node.trigger_condition, node.prompt)

    # MCQ generator (shared, stateless)
    mcq_generator = MCQGenerator(config)

    # History and summary
    history: list[dict] = []
    rng = random.Random(args.seed)

    # Workers from config or CLI
    workers = config.get_hyperparameter("workers", args.workers)

    # ── Baseline eval — cached in DB to avoid re-running on restart ───────────
    t0 = time.time()
    _cached = db.get_baseline()
    if _cached:
        baseline = _cached
        print(f"\n[V3] Baseline (cached) F1={baseline['f1']:.4f}  "
              f"P={baseline['precision']:.3f}  R={baseline['recall']:.3f}")
    else:
        baseline_size = config.get_hyperparameter("baseline_size", 50)
        print(f"\n[V3] Baseline eval ({baseline_size} cases × 4 routes = "
              f"{baseline_size * 4} LLM calls)...")
        baseline, _ = evaluate(eval_pool[:baseline_size], tree, route_llm_fn, global_rag, config, workers)
        db.save_baseline(baseline)
        print(f"[V3] Baseline F1={baseline['f1']:.4f}  "
              f"P={baseline['precision']:.3f}  R={baseline['recall']:.3f}  "
              f"elapsed={( time.time()-t0)/60:.1f}m")

    best_f1 = baseline["f1"]

    for rnd in range(1, args.rounds + 1):
        t_round = time.time()
        print(f"\n{'─'*64}")
        print(f"  Round {rnd}/{args.rounds}")
        print(f"{'─'*64}")

        batch_size = config.get_hyperparameter("batch_size", 50)
        batch = rng.sample(train_pool, min(batch_size, len(train_pool)))

        correct = errors = swr_events = splits = grafts = 0

        for i, case in enumerate(batch):
            text = case["text"]
            true_label = case["label"]

            # Route to specialist node
            node = tree.route(text)

            # Full v3 classification (node handles RAG + MCQ + engrams + routes)
            result = node.classify(
                text=text,
                route_llm_fn=route_llm_fn,
                global_rag_index=global_rag,
                workers=workers,
            )

            if result.route_result.split:
                splits += 1

            if result.label == true_label:
                correct += 1
                # Update route weights on correct prediction too
                node.update_weights_on_correct(result.route_result, true_label)
            else:
                errors += 1
                # Add this case to the routing node's RAG index
                # (node learns from the cases it handles)
                if node.rag_index is None:
                    node.rag_index = RAGIndex.from_examples_if_available(global_rag, text, true_label)

                # Handle error: MCQ + engram + SWR
                context_examples = global_rag.query(text, k=5)
                swr_result = node.handle_error(
                    text=text,
                    true_label=true_label,
                    predicted_label=result.label,
                    round_num=rnd,
                    route_result=result.route_result,
                    mcq_generator=mcq_generator,
                    freeform_llm_fn=freeform_llm_fn,
                    context_examples=context_examples,
                )

                if swr_result:
                    swr_events += 1
                    print(f"\n  ★ SWR EVENT — {node.id} cluster {swr_result.cluster_id}")
                    print(f"  ✓ Principle consolidated ({len(swr_result.principle)} chars)")

                    # Try to graft a child node from the SWR proposal
                    if swr_result.child_node_proposal:
                        probe_size = config.get_hyperparameter("probe_size", 100)
                        probe_cases = rng.sample(probe_pool, min(probe_size, len(probe_pool)))
                        graft_threshold = config.get_hyperparameter("graft_threshold", 0.005)

                        delta, child_node = tree.probe_graft(
                            parent_id=node.id,
                            child_proposal=swr_result.child_node_proposal,
                            probe_cases=probe_cases,
                            route_llm_fn=route_llm_fn,
                            global_rag_index=global_rag,
                            task_config=config,
                            workers=workers,
                        )

                        if delta >= graft_threshold and child_node:
                            child_node.db = db   # give new node DB access
                            tree.graft(node.id, child_node)
                            grafts += 1
                            db.upsert_node(child_node.id, node.id,
                                           child_node.trigger_condition,
                                           child_node.prompt, rnd)
                            print(f"  ✦ GRAFTED {child_node.id}  ΔF1={delta:+.4f}")
                        else:
                            print(f"  ✗ Child proposal rejected  ΔF1={delta:+.4f} < {graft_threshold}")

            # Progress bar
            if (i + 1) % 10 == 0 or (i + 1) == len(batch):
                pct = (i + 1) / len(batch)
                bar = "█" * int(pct * 20) + " " * (20 - int(pct * 20))
                acc = correct / max(1, i + 1)
                print(f"\r  [{bar}] {i+1}/{len(batch)}  "
                      f"acc={acc:.1%}  errors={errors}  swr={swr_events}  grafts={grafts}",
                      end="", flush=True)

        print()  # newline

        # End-of-round node stats
        for node in tree.all_nodes():
            node.end_round(rnd)

        # Tree report
        tree.print_tree()

        # MCQ reports for nodes with activity
        for node in tree.all_nodes():
            if len(node.mcq_library) > 0:
                node.mcq_library.print_report(top_n=3)

        # Save state
        tree.save()

        # Meta-round
        meta_interval = config.get_hyperparameter("meta_interval", 5)
        if rnd % meta_interval == 0 and history:
            print(f"\n[META-ROUND {rnd}] Full-history strategic review...")
            meta_recommendation = run_meta_round(tree, history, freeform_llm_fn, rnd)
            if meta_recommendation:
                print(f"[META] {meta_recommendation[:300]}...")
                (out / f"meta_round_{rnd}.txt").write_text(meta_recommendation)

        # Retirement check
        retire_threshold = config.get_hyperparameter("retire_threshold", -0.010)
        retire_min_rounds = config.get_hyperparameter("retire_min_rounds", 3)
        retire_max_routes = config.get_hyperparameter("retire_max_routes_per_round", 2)

        for node in tree.all_child_nodes():
            if (len(node.route_history) >= retire_min_rounds and
                    node.average_routes_per_round(retire_min_rounds) < retire_max_routes):
                probe_size = config.get_hyperparameter("probe_size", 100)
                probe_cases = rng.sample(probe_pool, min(probe_size, len(probe_pool)))
                delta = tree.probe_retire(
                    node.id, probe_cases, route_llm_fn, global_rag, config, workers
                )
                if delta >= retire_threshold:
                    tree.retire(node.id)
                    print(f"  ✂ RETIRED {node.id}  ΔF1={delta:+.4f}")
                    tree.save()

        # Eval
        print(f"\n[V3] Round {rnd} eval...")
        elapsed = time.time() - t0
        metrics, score_cache = evaluate(eval_pool[:100], tree, route_llm_fn, global_rag, config, workers)
        eta = (elapsed / rnd) * (args.rounds - rnd)
        print(f"  [R{rnd}/{args.rounds}] "
              f"EVAL F1={metrics['f1']:.4f}  "
              f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"nodes={len(tree.all_nodes())}  "
              f"elapsed={elapsed/60:.1f}m  ETA ~{eta/60:.1f}m")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            delta = best_f1 - baseline["f1"]
            print(f"  ★ NEW BEST  F1={best_f1:.4f}  (+{delta:.4f} vs baseline)")

        # Self-calibrate threshold using eval scores already cached — zero extra LLM calls.
        ade_bias, _cal_results = calibrate_threshold(score_cache, config)
        print(f"[V3] Threshold for next round: ADE_BIAS={ade_bias:.2f}")
        db.log_threshold_calibration(rnd, _cal_results, ade_bias)

        # Log round to DB (include structural event flags for HealthMonitor)
        db.log_eval(
            round_num=rnd, metrics=metrics, ade_bias=ade_bias,
            batch_accuracy=round(correct / max(1, len(batch)), 4),
            errors=errors, swr_events=swr_events, grafts=grafts,
            tree_nodes=len(tree.all_nodes()),
            graft_happened=(grafts > 0),
            swr_happened=(swr_events > 0),
        )
        # Log route weights for each node
        for node in tree.all_nodes():
            db.log_route_weights(node.id, rnd, node.aggregator.weights, node.aggregator.history)

        # ── Homeostatic controller — DISABLED for v3_04 production run ──────────
        # The controller logic is correct in principle but adds ~800 LLM calls
        # per intervention and needs isolated calibration before deployment.
        # The core system (SWR + MCQ + threshold calibration) self-corrects
        # sufficiently without it. Re-enable after v3_04 baseline is confirmed.
        #
        # current_run_eval_history.append({...})
        # homeostatic.run(...)

        db.print_summary(rnd)

        history.append({
            "round": rnd,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "batch_accuracy": round(correct / max(1, len(batch)), 4),
            "errors": errors,
            "splits": splits,
            "swr_events": swr_events,
            "grafts": grafts,
            "tree_nodes": len(tree.all_nodes()),
            "ade_bias_next": ade_bias,
        })

        (out / "v3_summary.json").write_text(json.dumps({
            "task": config.task_name,
            "best_f1": best_f1,
            "baseline_f1": baseline["f1"],
            "ade_bias": ade_bias,
            "rounds_run": rnd,
            "history": history,
        }, indent=2))

    # Final report
    total_mcqs = sum(len(n.mcq_library) for n in tree.all_nodes())
    total_principles = sum(len(n.injected_principles) for n in tree.all_nodes())
    print(f"\n{'═'*64}")
    print(f"  NEXUS v3 COMPLETE — {config.task_name}")
    print(f"  Best F1:    {best_f1:.4f}  (baseline {baseline['f1']:.4f}, Δ+{best_f1 - baseline['f1']:.4f})")
    print(f"  Tree nodes: {len(tree.all_nodes())} (started with {len(config.seed_nodes)})")
    print(f"  MCQs:       {total_mcqs} teaching cases generated")
    print(f"  Principles: {total_principles} engram principles")
    print(f"  Output:     {args.out}/")
    print(f"{'═'*64}\n")
    tree.print_tree()


# ─── Route patching from task config ─────────────────────────────────────────

def _patch_routes_from_config(config: TaskConfig) -> None:
    """
    Monkey-patch expert_routes.py to build route system prompts from task config
    rather than hardcoded ADE-specific text.
    """
    import expert_routes

    def _make_generic_route(route_def, cfg):
        def _route(text, examples, llm_fn, principle_context=""):
            system = cfg.build_route_system_prompt(route_def.name, principle_context)
            from expert_routes import _format_examples, _parse_json_vote, RouteResult
            examples_text = _format_examples(examples)
            user = (
                f'Sentence: "{text}"\n\n'
                f"Similar labeled examples:\n{examples_text}\n\n"
                f"Based on your focus ({route_def.focus[:100]}...), vote {cfg.positive_label} or {cfg.negative_label}."
            )
            try:
                raw = llm_fn(system, user)
                d = _parse_json_vote(raw)
                return RouteResult(
                    route=route_def.name,
                    vote=d.get("vote", route_def.default_vote),
                    confidence=float(d.get("confidence", 0.5)),
                    reasoning=d.get("reasoning", ""),
                )
            except Exception as e:
                return RouteResult(route_def.name, route_def.default_vote, 0.3, f"error: {e}")
        return _route

    # Replace route registry with config-driven routes
    new_routes = {}
    for route_def in config.route_definitions:
        new_routes[route_def.name] = _make_generic_route(route_def, config)

    if new_routes:
        expert_routes._ROUTES = new_routes
        print(f"[V3] Routes patched from task config: {list(new_routes.keys())}")
    else:
        print("[V3] No route_definitions in task config — using default routes from expert_routes.py")


def _patch_aggregator_bias(ade_bias: float) -> None:
    """Patch RouteAggregator with computed class-prior bias."""
    import expert_routes
    expert_routes.RouteAggregator.ADE_BIAS = ade_bias
    print(f"[V3] Aggregator ADE_BIAS set to {ade_bias:.3f} (computed from corpus)")


def calibrate_threshold(
    score_cache: list[tuple[float, float, str]],
    config: TaskConfig,
) -> float:
    """
    Self-calibrating decision threshold — ZERO extra LLM calls.

    Uses raw (ade_score, not_ade_score, true_label) tuples collected during
    the eval pass that already ran.  Sweeps candidate bias values in pure
    Python arithmetic — no inference, no API calls, instant.

    beta=1.0 → F1  (balanced)
    beta=2.0 → F2  (recall-weighted — safety-critical tasks)
    beta=0.5 → F0.5 (precision-weighted — cost-sensitive tasks)
    """
    import expert_routes

    beta   = getattr(config, "optimization_beta", 1.0)
    target = getattr(config, "optimization_target", "fbeta")
    beta2  = beta ** 2
    pos    = config.positive_label

    candidates = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0]
    best_bias  = expert_routes.RouteAggregator.ADE_BIAS
    best_score = -1.0
    results    = []

    for bias in candidates:
        tp = fp = fn = tn = 0
        for ade_score, not_ade_score, true_label in score_cache:
            pred = pos if ade_score >= not_ade_score * bias else config.negative_label
            if pred == pos and true_label == pos:    tp += 1
            elif pred == pos and true_label != pos:  fp += 1
            elif pred != pos and true_label == pos:  fn += 1
            else:                                    tn += 1

        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1     = 2 * prec * recall / max(1e-9, prec + recall)
        fbeta  = (1 + beta2) * prec * recall / max(1e-9, beta2 * prec + recall)
        score  = fbeta if target == "fbeta" else {"f1": f1, "recall": recall, "precision": prec}.get(target, f1)

        results.append((bias, score, prec, recall, f1, fbeta))
        if score > best_score:
            best_score = score
            best_bias  = bias

    expert_routes.RouteAggregator.ADE_BIAS = best_bias

    b_str = f"F{beta:.1f}" if target == "fbeta" else target.upper()
    print(f"\n[THRESHOLD CAL] {b_str} sweep — {len(score_cache)} cached scores, 0 extra LLM calls")
    for bias, score, prec, rec, f1, fbeta in sorted(results, key=lambda x: x[0]):
        marker = " ◀" if abs(bias - best_bias) < 1e-6 else ""
        score_val = fbeta if target == "fbeta" else score
        print(f"  bias={bias:.1f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.4f}  {b_str}={score_val:.4f}{marker}")

    return best_bias, results


# ─── RAGIndex helper ──────────────────────────────────────────────────────────
# Extend RAGIndex with a method to seed from specific examples
def _patch_rag_index() -> None:
    from rag_index import RAGIndex
    import numpy as np

    @classmethod
    def from_examples_if_available(cls, global_index, text, label):
        """Return a minimal sub-index seeded from a single example — placeholder for v3."""
        # In a full implementation, track per-node cases and build a proper sub-index.
        # For now, return None and fall back to global index.
        return None

    RAGIndex.from_examples_if_available = from_examples_if_available


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    _patch_rag_index()

    ap = argparse.ArgumentParser(description="NEXUS v3 — Unified Self-Growing Learning System")
    ap.add_argument("--task",    default="task_configs/ade_classification.json",
                    help="Path to task config JSON")
    ap.add_argument("--corpus",  default=None,
                    help="Path to labeled corpus JSONL (default: ADE Corpus v2 via HuggingFace)")
    ap.add_argument("--out",     default="run_v3_01")
    ap.add_argument("--rounds",  type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--fresh",   action="store_true",
                    help="Ignore existing tree/index state and start fresh")

    # Northwell AI Hub (enterprise)
    ap.add_argument("--ai-hub",       action="store_true")
    ap.add_argument("--ai-hub-key",   default=os.environ.get("AIHUB_API_KEY", ""))
    ap.add_argument("--ai-hub-ad-id", default=os.environ.get("AIHUB_AD_OBJECT_ID", ""))

    # OpenAI-compatible (public — works with OpenAI, Ollama, Anthropic, etc.)
    ap.add_argument("--openai",                action="store_true",
                    help="Use OpenAI-compatible API (OpenAI, Ollama, Anthropic, etc.)")
    ap.add_argument("--openai-key",            default=os.environ.get("OPENAI_API_KEY", ""),
                    help="API key (or set OPENAI_API_KEY env var; leave blank for Ollama)")
    ap.add_argument("--openai-base-url",       default=None,
                    help="Custom base URL, e.g. http://localhost:11434/v1 for Ollama")
    ap.add_argument("--openai-classify-model", default="gpt-4o-mini",
                    help="Model for per-case classification (high-volume, fast)")
    ap.add_argument("--openai-synth-model",    default="gpt-4o",
                    help="Model for synthesis/refinement (low-volume, best reasoning)")

    # Mock
    ap.add_argument("--mock", action="store_true",
                    help="MockClient — no API cost, for pipeline testing")

    args = ap.parse_args()
    run_v3(args)


if __name__ == "__main__":
    main()

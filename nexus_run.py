"""
nexus_run.py
Main entry point — the NEXUS self-optimizing decision-tree loop.

Round structure (11 steps):
  1.  SAMPLE     — draw BATCH_SIZE cases from training pool
  2.  CLASSIFY   — route + classify each case; record routing + prediction
  3.  COLLECT    — group misclassified cases by node
  4.  SYNTHESIZE — ask LLM to propose a new branch from the errors
                   (synthesis prompt includes nugget catalogue)
  5.  VALIDATE   — check trigger_condition is safe Python with whitelisted names;
                   reject duplicates
  6.  PROBE      — compare current tree vs candidate on probe set
  7.  DECIDE     — accept graft only if delta_F1 > ACCEPT_THRESHOLD
  8.  REFINE     — for each node with enough errors, ask LLM to improve its
                   prompt; probe the change; accept if F1 improves
                   *** core self-optimization contribution ***
  9.  EXTRACT    — after any accepted change, ask LLM to pull new reusable
                   fragments into the nugget library (token compression)
  10. EVAL       — full eval on the fixed eval set; compute per-node metrics
  11. LOG        — write round stats (F1, tokens, actions) to history

Usage:
    export GEMINI_API_KEY=your_key_here
    python nexus_run.py --rounds 10 --batch-size 20

Dry run (no API key, no network):
    python nexus_run.py --mock --rounds 3 --batch-size 10 --eval-size 40 --probe-size 40

Flags:
    --no-refine     skip step 8 (baseline comparison for ablation study)
    --no-nuggets    skip nugget system (another ablation baseline)
    --no-extract    skip nugget extraction after each accepted change
    --debug-console also echo the debug log to stdout (very verbose)
    --min-refine-errors N   min errors per node to trigger refinement (default 2)
    --refine-probe-size N   cases to probe refinements with (default 30)
    --sleep S       seconds between API calls (free-tier rate limiting)
"""

import argparse
import json
import random
import time
from collections import defaultdict, deque
from copy import deepcopy

import pandas as pd

import re
from features import features, is_valid_condition, FEATURE_NAMES
from tree import seed_tree, classify_with_tree, insert_graft, node_summaries, save_tree, load_tree
from metrics import calc_metrics
from llm_client import GeminiClient, AIHubClient, MockClient
from token_tracker import TokenTracker
from nuggets import NuggetStore
from principles import PrinciplesStore
from drug_registry import DrugRegistry
from debug_logger import DebugLogger

ACCEPT_THRESHOLD = 0.005
REJECT_THRESHOLD = -0.005
REFINE_THRESHOLD = 0.002   # lower bar — prompt tweaks are smaller changes

_SAFE_TOKENS = set(FEATURE_NAMES) | {"and", "or", "not", "True", "False"}


def _bad_vars(condition: str) -> list[str]:
    """Return any identifier tokens in `condition` that aren't in the whitelist."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", condition)
    return [t for t in tokens if t not in _SAFE_TOKENS]


def _route_quality(tree: dict, node_id: str, cases: list[dict]) -> dict:
    """
    Cheap pre-probe routing check for a candidate graft.

    A proposed specialist should catch a focused slice. If it routes a large
    fraction of the probe pool, it is behaving like a second root prompt and is
    likely to regress globally.
    """
    routed = 0
    for case in cases:
        node = classify_with_tree(tree, features(case["text"]))
        if node["id"] == node_id:
            routed += 1
    total = max(1, len(cases))
    return {"routed": routed, "total": total, "share": routed / total}


# ---------------------------------------------------------------------------
# Per-case classification with full instrumentation
# ---------------------------------------------------------------------------

def classify_one(tree, client, text, nugget_store=None, logger=None,
                 label=None, sleep_s=0.0):
    """
    Route `text` through `tree`, call classify(), return (pred, node_id, confidence).
    Logs routing and prediction to `logger` when provided.
    `label` is the ground truth — needed for the prediction log line only.
    """
    feats = features(text)
    node = classify_with_tree(tree, feats)

    prompt = node["prompt"]
    if nugget_store:
        prompt = nugget_store.assemble(prompt)

    result, inp, out, lat = client.classify(text, prompt)

    pred = result.get("classification", "NOT_ADE")
    if pred not in ("ADE", "NOT_ADE"):
        pred = "NOT_ADE"
    confidence = result.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    if logger:
        logger.log_routing(text, feats, node["id"])
        if label is not None:
            logger.log_prediction(
                text, node["id"], pred, label, inp, out, lat,
                rationale=result.get("rationale", ""),
            )

    if sleep_s:
        time.sleep(sleep_s)

    return pred, node["id"], confidence


def run_eval(tree, client, cases, sleep_s=0.0, nugget_store=None, logger=None,
             phase_label: str = "", confidence_tracker: dict = None):
    """Classify every case in `cases`. Returns (preds, labels, node_ids).
    phase_label is shown in the inline progress counter when set.
    confidence_tracker: optional dict mutated with per-node confidence counts."""
    preds, labels, node_ids = [], [], []
    total = len(cases)
    t_phase = time.time()
    for i, c in enumerate(cases):
        pred, nid, conf = classify_one(
            tree, client, c["text"],
            nugget_store=nugget_store,
            logger=logger,
            label=c["label"],
            sleep_s=sleep_s,
        )
        preds.append(pred)
        labels.append(c["label"])
        node_ids.append(nid)
        if confidence_tracker is not None:
            if nid not in confidence_tracker:
                confidence_tracker[nid] = {"high": 0, "medium": 0, "low": 0}
            confidence_tracker[nid][conf] = confidence_tracker[nid].get(conf, 0) + 1
        # Inline progress — overwrite the same line
        if phase_label:
            done = i + 1
            pct  = done / total * 100
            bar  = ("█" * int(pct // 5)).ljust(20)
            elapsed = time.time() - t_phase
            eta_s   = (elapsed / done) * (total - done) if done else 0
            print(f"    {phase_label}  [{bar}] {done}/{total}  "
                  f"({pct:.0f}%)  ETA {_fmt_time(eta_s)}   ",
                  end="\r", flush=True)
    if phase_label:
        elapsed = time.time() - t_phase
        print(f"    {phase_label}  [{'█'*20}] {total}/{total}  "
              f"(100%)  done in {_fmt_time(elapsed)}      ")
    return preds, labels, node_ids


def _fmt_time(seconds: float) -> str:
    """Format seconds as m:ss or s."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60:02d}s"


def _print_estimate(args, is_mock: bool):
    """Print an upfront estimate of API calls and wall time before the run starts."""
    if is_mock:
        return
    per_call = args.sleep + 1.5   # sleep + ~1.5s avg API latency
    # Minimum calls per round (no probe accepted, just batch + eval)
    min_calls = args.batch_size + args.eval_size
    # Maximum calls per round (probe accepted + refine probe on 2 nodes)
    max_calls = (args.batch_size + args.probe_size * 2 +
                 args.eval_size + args.refine_probe_size * 2 * 2)
    total_min = (args.eval_size + 1 + args.rounds * min_calls) * per_call
    total_max = (args.eval_size + 1 + args.rounds * max_calls) * per_call
    print(f"\n{'─'*60}")
    print(f"  NEXUS run estimate ({args.rounds} rounds)")
    print(f"  Eval {args.eval_size} | Probe {args.probe_size} | "
          f"Batch {args.batch_size} | Sleep {args.sleep}s")
    print(f"  API calls/round: {min_calls}–{max_calls}  "
          f"(depends on probe/refine)")
    print(f"  Est. wall time : {_fmt_time(total_min)} – {_fmt_time(total_max)}")
    print(f"{'─'*60}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing_conditions(tree: dict) -> set:
    return {
        c["trigger_condition"]
        for c in tree.get("children", [])
        if c.get("trigger_condition")
    }


def _find_node(tree: dict, node_id: str) -> dict | None:
    """Find a node by ID (root or any direct child)."""
    if tree["id"] == node_id:
        return tree
    for c in tree.get("children", []):
        if c["id"] == node_id:
            return c
    return None


def _swap_node_prompt(tree: dict, node_id: str, new_prompt: str) -> dict:
    """Deep-copy `tree` with `node_id`'s prompt replaced by `new_prompt`."""
    candidate = deepcopy(tree)
    node = _find_node(candidate, node_id)
    if node:
        node["prompt"] = new_prompt
    return candidate


def compute_per_node_metrics(preds, labels, node_ids) -> dict:
    """Per-node precision/recall/F1 from eval results."""
    buckets: dict = defaultdict(lambda: {"preds": [], "labels": []})
    for p, l, n in zip(preds, labels, node_ids):
        buckets[n]["preds"].append(p)
        buckets[n]["labels"].append(l)
    result = {}
    for nid, d in buckets.items():
        m = calc_metrics(d["preds"], d["labels"])
        m["count"] = len(d["preds"])
        result[nid] = m
    return result


def _do_extract_nuggets(prompt_text: str, client, nugget_store, logger,
                        round_num: int, source: str):
    """
    Ask the LLM to pull reusable fragments from `prompt_text` into the
    nugget library. Returns list of newly added nugget ids.
    """
    candidates, _, _, _ = client.extract_nuggets(prompt_text, nugget_store)
    added, skipped = [], []
    for n in candidates:
        if not isinstance(n, dict) or "id" not in n or "text" not in n:
            skipped.append({"id": "?", "reason": "malformed entry"})
            continue
        ok = nugget_store.add_nugget(n["id"], n["text"],
                                     source=f"round_{round_num}")
        if ok:
            added.append(n)
        else:
            n["reason"] = "duplicate/too-short/substring"
            skipped.append(n)
    if logger:
        logger.log_nugget_extraction(source, candidates, added, skipped)
    return [n["id"] for n in added]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="NEXUS self-growing, self-optimizing decision tree runner"
    )
    ap.add_argument("--rounds",             type=int,   default=10)
    ap.add_argument("--batch-size",         type=int,   default=20)
    ap.add_argument("--probe-size",         type=int,   default=300)
    ap.add_argument("--eval-size",          type=int,   default=200)
    ap.add_argument("--refine-probe-size",  type=int,   default=30,
                    help="cases used to validate prompt refinements (kept small for cost)")
    ap.add_argument("--min-refine-errors",  type=int,   default=2,
                    help="min errors a node must have before refinement is attempted")
    ap.add_argument("--seed",               type=int,   default=42)
    ap.add_argument("--data-seed",          type=int,   default=None,
                    help="seed for eval/probe/train split; defaults to --seed")
    ap.add_argument("--sleep",              type=float, default=0.0,
                    help="seconds to sleep between API calls")
    ap.add_argument("--out-dir",            type=str,   default=".")
    ap.add_argument("--initial-tree",        type=str,   default=None,
                    help="Start from a saved nexus_best_tree.json instead of seed_tree().")
    ap.add_argument("--model",              type=str,
                    default="gemini-2.5-flash",
                    help="Gemini model (ignored when --ai-hub is set).")
    # --- Northwell AI Hub ---
    ap.add_argument("--ai-hub",             action="store_true",
                    help="Use Northwell AI Hub instead of Gemini API.")
    ap.add_argument("--ai-hub-key",         type=str,
                    default=None,
                    help="AI Hub API key (or set AIHUB_API_KEY env var).")
    ap.add_argument("--ai-hub-ad-id",       type=str,
                    default=None,
                    help="Your AD Object ID (or set AIHUB_AD_OBJECT_ID env var).")
    ap.add_argument("--classify-model",     type=str,
                    default="claude-haiku-4.5",
                    help="AI Hub model for classify calls (high-volume). "
                         "Default: claude-haiku-4.5 (fast + cheap).")
    ap.add_argument("--synth-model",        type=str,
                    default="claude-sonnet-4.5",
                    help="AI Hub model for synthesize/refine calls (low-volume). "
                         "Default: claude-sonnet-4.5. "
                         "Options: claude-opus-4.6, gemini-2.5-pro, gpt-5, o3.")
    ap.add_argument("--mock",               action="store_true",
                    help="offline MockClient + synthetic data (plumbing test only)")
    ap.add_argument("--mock-pool-size",     type=int,   default=1000)
    ap.add_argument("--no-refine",          action="store_true",
                    help="skip REFINE step (ablation baseline)")
    ap.add_argument("--no-nuggets",         action="store_true",
                    help="disable nugget system (ablation baseline)")
    ap.add_argument("--fresh-nuggets",      action="store_true",
                    help="ignore any existing nugget file and start from seed nuggets "
                         "only. Use this for every clean research run so prior "
                         "experiments do not contaminate the nugget library.")
    ap.add_argument("--no-extract",         action="store_true",
                    help="skip nugget extraction after accepted changes")
    ap.add_argument("--no-meta",            action="store_true",
                    help="skip strategic meta-rounds (ablation baseline)")
    ap.add_argument("--no-retire",          action="store_true",
                    help="skip node retirement check (ablation baseline)")
    ap.add_argument("--no-refine-root",     action="store_true",
                    help="skip ROOT prompt refinement (use when ROOT is at prompt ceiling)")
    ap.add_argument("--meta-interval",      type=int,   default=3,
                    help="run meta-round every N rounds (default 3)")
    ap.add_argument("--error-buffer-size",  type=int,   default=60,
                    help="rolling error buffer size across rounds (default 60)")
    ap.add_argument("--retire-min-rounds",  type=int,   default=3,
                    help="min rounds a node must exist before retirement check")
    ap.add_argument("--retire-max-routes",  type=int,   default=2,
                    help="avg routes/round below which a node is retirement-eligible")
    ap.add_argument("--min-graft-routes",   type=int,   default=3,
                    help="reject new grafts that route fewer than this many probe cases")
    ap.add_argument("--max-graft-route-share", type=float, default=0.20,
                    help="reject new grafts that route more than this fraction of probe cases")
    ap.add_argument("--debug-console",      action="store_true",
                    help="also print debug log to stdout (very verbose)")
    args = ap.parse_args()

    random.seed(args.seed)
    round_rng = random.Random(args.seed)
    data_seed = args.data_seed if args.data_seed is not None else args.seed

    import os as _os
    import subprocess as _sub
    import sys as _sys
    out = args.out_dir
    _os.makedirs(out, exist_ok=True)

    # ── Prevent macOS sleep while run is active ───────────────────────────────
    # caffeinate -i keeps the system awake; -w PID auto-releases when we exit.
    _caffeinate = None
    if _sys.platform == "darwin":
        try:
            _caffeinate = _sub.Popen(
                ["caffeinate", "-i", "-w", str(_os.getpid())],
                stdout=_sub.DEVNULL, stderr=_sub.DEVNULL,
            )
            print("[Sleep] caffeinate started — system will stay awake for this run.")
        except FileNotFoundError:
            print("[Sleep] caffeinate not found (non-macOS?). Sleep prevention skipped.")

    # ── Local sentinel file paths for manual control/status ──────────────────
    _KILL_FILE   = f"{out}/KILL_NEXUS"
    _PAUSE_FILE  = f"{out}/PAUSE_NEXUS"
    _STATUS_REQ  = f"{out}/STATUS_REQUEST"
    _STATUS_RESP = f"{out}/STATUS_RESPONSE"

    def _check_sentinel(current_round: int, current_f1: float,
                        best_f1_val: float, history_len: int) -> bool:
        """
        Check local sentinel files.
        Returns True if run should stop (KILL received).
        Handles PAUSE (blocks until RESUME), STATUS (writes response).
        """
        # STATUS request — write a reply file for local tooling to pick up
        if _os.path.exists(_STATUS_REQ):
            try:
                _os.remove(_STATUS_REQ)
                status_msg = (
                    f"Round {current_round}/{args.rounds} | "
                    f"F1={current_f1:.3f} (best={best_f1_val:.3f})"
                )
                with open(_STATUS_RESP, "w") as _f:
                    _f.write(status_msg)
                print(f"[Sentinel] STATUS reply written: {status_msg}")
            except Exception:
                pass

        # PAUSE — block here until PAUSE file is removed (RESUME command)
        if _os.path.exists(_PAUSE_FILE):
            print(f"\n[Sentinel] PAUSED by SMS. Text RESUME to {out} listener to continue.")
            while _os.path.exists(_PAUSE_FILE):
                time.sleep(10)
            print("[Sentinel] RESUMED.")

        # KILL — stop after this round
        if _os.path.exists(_KILL_FILE):
            try:
                _os.remove(_KILL_FILE)
            except Exception:
                pass
            print("\n[Sentinel] KILL received via SMS — stopping run gracefully.")
            return True

        return False

    # --- Infrastructure ---
    tracker = TokenTracker()
    logger  = DebugLogger(f"{out}/nexus_debug.log", console=args.debug_console)

    # --- Nugget store ---
    # Nuggets are saved to {out_dir}/nuggets/ so each --out-dir run is isolated.
    # --fresh-nuggets starts from seed only (required for clean research runs).
    if args.no_nuggets:
        nugget_store = None
        nugget_path  = None
        print("[Nuggets] disabled (--no-nuggets)")
    else:
        import os as _os
        nugget_dir  = f"{out}/nuggets"
        _os.makedirs(nugget_dir, exist_ok=True)
        nugget_path = f"{nugget_dir}/nexus_nuggets.json"

        if args.fresh_nuggets:
            nugget_store = NuggetStore(path=nugget_path)   # seed nuggets only
            nugget_store.save(nugget_path)
            print(f"[Nuggets] fresh start — {len(nugget_store.nuggets)} seed nuggets "
                  f"→ {nugget_path}")
        else:
            nugget_store = NuggetStore.load(nugget_path)
            print(f"[Nuggets] loaded {len(nugget_store.nuggets)} nuggets "
                  f"from {nugget_path}")

    # --- Principles store ---
    # Stores learned prompt engineering principles + agent identity.
    # Persisted in {out_dir}/principles/ so each run starts from its own state.
    # If --fresh-nuggets is set, also start fresh principles (clean research run).
    import os as _os
    principles_dir  = f"{out}/principles"
    _os.makedirs(principles_dir, exist_ok=True)
    principles_path = f"{principles_dir}/nexus_principles.json"
    if args.fresh_nuggets:
        principles_store = PrinciplesStore(path=principles_path)
        principles_store.save(principles_path)
        print(f"[Principles] fresh start — 0 principles → {principles_path}")
    else:
        principles_store = PrinciplesStore.load(principles_path)
        print(f"[Principles] loaded {len(principles_store.principles)} principles "
              f"from {principles_path}")

    # --- Drug Registry (Hebbian pharmacological memory) ---
    # Persisted in {out_dir}/drug_registry.json.  --fresh-nuggets also resets it.
    import os as _os
    registry_path = f"{out}/drug_registry.json"
    if args.fresh_nuggets:
        drug_registry = DrugRegistry(path=registry_path)
        drug_registry.save(registry_path)
        print(f"[DrugRegistry] fresh start — {len(drug_registry.drug_to_id)} seed drugs")
    else:
        drug_registry = DrugRegistry.load(registry_path)
        print(f"[DrugRegistry] loaded — {len(drug_registry.drug_to_id)} drugs, "
              f"{sum(1 for s in drug_registry.drug_stats.values() if s['engram_formed'])} engrams")

    # --- Cross-run chronicle (shared across all runs in this folder) ---
    # Loaded at startup so the LLM can reference prior run findings.
    chronicle_path = f"{out}/../nexus_chronicle.json"
    try:
        with open(chronicle_path) as f:
            chronicle = json.load(f)
        print(f"[Chronicle] loaded {len(chronicle)} prior runs from {chronicle_path}")
    except FileNotFoundError:
        chronicle = []
        print("[Chronicle] no prior runs — starting fresh chronicle")

    # --- Client + data ---
    if args.mock:
        eval_pool, probe_pool, train_pool = _make_synthetic_pools(
            args.eval_size, args.probe_size, args.mock_pool_size, seed=data_seed
        )
        client = MockClient(seed=args.seed, tracker=tracker)
        print("[MOCK MODE] synthetic data + deterministic classifier. "
              "Numbers are for plumbing verification only.")
    else:
        from data_utils import load_and_split
        eval_pool, probe_pool, train_pool = load_and_split(
            seed=data_seed, eval_size=args.eval_size, probe_size=args.probe_size
        )
        if args.ai_hub:
            import os
            api_key  = args.ai_hub_key  or os.environ.get("AIHUB_API_KEY")
            ad_id    = args.ai_hub_ad_id or os.environ.get("AIHUB_AD_OBJECT_ID")
            if not api_key:
                raise RuntimeError(
                    "AI Hub API key required. Pass --ai-hub-key or set AIHUB_API_KEY."
                )
            if not ad_id:
                raise RuntimeError(
                    "AD Object ID required. Pass --ai-hub-ad-id or set AIHUB_AD_OBJECT_ID."
                )
            client = AIHubClient(
                api_key=api_key, ad_object_id=ad_id,
                classify_model=args.classify_model,
                synth_model=args.synth_model,
                tracker=tracker,
            )
            print(f"[AI Hub] classify={args.classify_model}  synth={args.synth_model}")
        else:
            client = GeminiClient(model=args.model, tracker=tracker)
            print(f"[Model] {args.model}")

    print(f"Eval pool: {len(eval_pool)} | Probe pool: {len(probe_pool)} | "
          f"Train pool: {len(train_pool)}")

    _print_estimate(args, is_mock=args.mock)

    # Probe subset used for refinement validation (cheaper than full probe)
    refine_probe = probe_pool[:args.refine_probe_size]

    # --- Seed or inherited tree + compress prompts via nuggets ---
    if args.initial_tree:
        tree = load_tree(args.initial_tree)
        print(f"[Tree] loaded initial tree from {args.initial_tree}")
    else:
        tree = seed_tree()
        print("[Tree] using seed tree")
    if nugget_store:
        for child in tree.get("children", []):
            child["prompt"] = nugget_store.compress(child["prompt"])
        tree["prompt"] = nugget_store.compress(tree["prompt"])

    # --- Round 0: baseline eval ---
    logger.set_round(0)
    print("[Round 0] Baseline eval...")
    preds, labels, node_ids = run_eval(
        tree, client, eval_pool,
        sleep_s=args.sleep, nugget_store=nugget_store, logger=logger,
        phase_label="Round 0 | Baseline",
    )
    m   = calc_metrics(preds, labels)
    pnm = compute_per_node_metrics(preds, labels, node_ids)
    tok = tracker.round_delta()
    print(f"[Round 0] Baseline F1={m['f1']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f}  tokens={tok['round_tokens_total']:,}")
    logger.log_eval(m["f1"], m["precision"], m["recall"], m["accuracy"],
                    pnm, tok["round_tokens_total"], tracker.total_tokens())

    if nugget_store:
        for child in tree.get("children", []):
            nugget_store.record_usage(child["prompt"], m["f1"])
        nugget_store.record_usage(tree["prompt"], m["f1"])

    history = [{
        "round": 0,
        "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
        "accuracy": m["accuracy"],
        "action": "baseline",
        "tree_nodes": len(node_summaries(tree)),
        "per_node_f1": {nid: v["f1"] for nid, v in pnm.items()},
        **tok,
    }]
    best_tree = deepcopy(tree)
    best_f1   = m["f1"]

    # -----------------------------------------------------------------------
    # Cross-round state
    # -----------------------------------------------------------------------
    # Rolling error buffer — last N errors from any round (for richer LLM context)
    error_buffer: deque = deque(maxlen=args.error_buffer_size)

    # Per-node route history: node_id -> [routes_per_round, ...]
    node_route_history: dict = defaultdict(list)

    # Per-node F1 history: node_id -> [f1_per_round, ...]
    node_f1_history: dict = defaultdict(list)

    # Gradient-eligible nodes: set of node IDs that should be priority-refined
    # next round even if below min_refine_errors threshold
    gradient_eligible: set = set()

    # Accepted changes log for meta-round context
    accepted_changes_log: list = []

    # Rejected proposals log — prevents the LLM from repeating failed ideas
    rejected_proposals_log: list = []

    # Last meta-round analysis text (for identity evolution)
    last_meta_analysis: str = ""

    # -----------------------------------------------------------------------
    # Learning loop
    # -----------------------------------------------------------------------
    round_times = []   # track elapsed time per round for ETA

    for rnd in range(1, args.rounds + 1):
        rnd_t0 = time.time()
        logger.set_round(rnd)
        actions_this_round = []

        # ── Sentinel check (SMS commands: KILL / PAUSE / STATUS) ─────────────
        if _check_sentinel(rnd, history[-1]["f1"] if history else 0.0,
                           best_f1, len(history)):
            print(f"[Sentinel] Exiting run at round {rnd} by SMS request.")
            break   # exit cleanly — saves outputs below

        remaining = args.rounds - rnd
        eta_str   = (f"  ETA ~{_fmt_time(sum(round_times)/len(round_times)*remaining)}"
                     if round_times else "")
        print(f"\n{'─'*60}")
        print(f"  Round {rnd}/{args.rounds}{eta_str}")

        # 1. SAMPLE
        batch = round_rng.sample(train_pool, min(args.batch_size, len(train_pool)))

        # 2+3. CLASSIFY + COLLECT
        errors     = []
        node_errors: dict = defaultdict(list)   # node_id -> [error dicts]
        low_conf_counts: dict = defaultdict(int) # node_id -> low-confidence count
        for c in batch:
            pred, node_id, conf = classify_one(
                tree, client, c["text"],
                nugget_store=nugget_store, logger=logger,
                label=c["label"], sleep_s=args.sleep,
            )
            if conf == "low":
                low_conf_counts[node_id] += 1
            if pred != c["label"]:
                err = {"text": c["text"], "label": c["label"],
                       "pred": pred, "node": node_id}
                errors.append(err)
                node_errors[node_id].append(err)
                error_buffer.append(err)   # cross-round rolling buffer

            # HEBBIAN UPDATE — zero API cost, fires on every training case
            # "neurons that fire together, wire together"
            drug_registry.observe(c["text"], c["label"])

        logger.log_error_summary(errors)
        if error_buffer:
            print(f"  [Buffer] {len(error_buffer)} cumulative errors in rolling buffer")

        # ENGRAM FORMATION — check for newly matured drug associations
        # Biological analog: Repeated LTP → stable memory trace → engram
        if nugget_store and not args.no_nuggets:
            new_engrams = drug_registry.engrams_ready()
            for drug in new_engrams:
                nugget_id   = f"DRUG_{drug.upper().replace('-', '_')}"
                nugget_text = drug_registry.build_nugget_text(drug)
                if nugget_text:
                    ok = nugget_store.add_nugget(nugget_id, nugget_text,
                                                 source=f"hebbian_r{rnd}")
                    if ok:
                        print(f"  [Hebbian] engram formed → [{nugget_id}]: "
                              f"{nugget_text[:70]}...")

        # Confidence-driven gradient: nodes with ≥50% low-confidence classifications
        # are flagged for priority refinement (zero extra API calls — uses existing data)
        batch_size_actual = len(batch)
        for nid, lc_count in low_conf_counts.items():
            total_for_node = sum(1 for c in batch
                                 if classify_with_tree(tree, features(c["text"]))["id"] == nid)
            lc_rate = lc_count / max(1, total_for_node)
            if lc_rate >= 0.5 and nid not in gradient_eligible:
                gradient_eligible.add(nid)
                print(f"  [Confidence] {nid}: {lc_rate:.0%} low-confidence → "
                      f"adding to gradient queue")

        # ----------------------------------------------------------------
        # 4–7. SYNTHESIZE → VALIDATE → PROBE → DECIDE
        # ----------------------------------------------------------------
        graft_action = "no_errors"

        # ----------------------------------------------------------------
        # META-ROUND (every meta_interval rounds) — strategic review
        # ----------------------------------------------------------------
        if (not args.no_meta and rnd % args.meta_interval == 0
                and rnd > 1 and len(history) >= 2):
            print(f"\n  ★ META-ROUND {rnd} — strategic tree review...")
            f1_curve = [h["f1"] for h in history]
            last_pnm = history[-1].get("per_node_f1", {})
            pn_stats  = {
                nid: {
                    "f1":        f1,
                    "count":     sum(node_route_history.get(nid, [0])),
                    "avg_routes": round(
                        sum(node_route_history.get(nid, [0])) /
                        max(1, len(node_route_history.get(nid, [1]))), 1
                    ),
                }
                for nid, f1 in last_pnm.items()
            }
            try:
                meta_p, _, _, _ = client.meta_synthesize(
                    f1_history=f1_curve,
                    node_list=node_summaries(tree),
                    per_node_stats=pn_stats,
                    error_buffer=error_buffer,
                    accepted_changes=accepted_changes_log,
                    nugget_store=nugget_store,
                )
            except Exception as e:
                print(f"  [Meta] call failed: {e}")
                meta_p = None

            if meta_p and isinstance(meta_p, dict):
                action_type = meta_p.get("action_type", "")
                last_meta_analysis = str(meta_p.get("strategic_analysis", ""))
                print(f"  [Meta] action_type={action_type}  "
                      f"analysis={last_meta_analysis[:80]}")

                # Identity evolution: NEXUS updates its self-concept after each meta-round
                try:
                    last_pnm_for_id = history[-1].get("per_node_f1", {}) if history else {}
                    pn_stats_for_id = {nid: {"f1": f1} for nid, f1 in last_pnm_for_id.items()}
                    new_id, _, _, _ = client.evolve_identity(
                        current_identity=principles_store.identity,
                        f1_history=[h["f1"] for h in history],
                        per_node_stats=pn_stats_for_id,
                        accepted_count=len(accepted_changes_log),
                        principle_count=len(principles_store.principles),
                        last_meta_analysis=last_meta_analysis,
                    )
                    if new_id:
                        principles_store.update_identity(new_id)
                        print(f"  [Identity] evolved: {new_id[:100]}...")
                except Exception as e:
                    print(f"  [Identity] evolution failed: {e}")
                # If meta proposes a new_node, inject it as errors for normal synthesis
                if action_type == "new_node" and meta_p.get("new_node"):
                    # Treat meta proposal like a regular synthesis proposal
                    mn = meta_p["new_node"]
                    cond = mn.get("trigger_condition", "")
                    if (all(k in mn for k in ("id", "trigger_condition", "prompt"))
                            and is_valid_condition(cond)
                            and cond not in _existing_conditions(tree)):
                        candidate_meta = insert_graft(tree, mn)
                        cur_pm, cur_lm, _ = run_eval(
                            tree, client, probe_pool,
                            sleep_s=args.sleep, nugget_store=nugget_store,
                            phase_label=f"R{rnd} | Meta-Probe(cur) ",
                        )
                        cnd_pm, cnd_lm, _ = run_eval(
                            candidate_meta, client, probe_pool,
                            sleep_s=args.sleep, nugget_store=nugget_store,
                            phase_label=f"R{rnd} | Meta-Probe(cnd)",
                        )
                        delta_m = (calc_metrics(cnd_pm, cnd_lm)["f1"]
                                   - calc_metrics(cur_pm, cur_lm)["f1"])
                        if delta_m > ACCEPT_THRESHOLD:
                            tree = candidate_meta
                            accepted_changes_log.append(
                                f"meta_graft(delta={delta_m:+.4f}, node={mn['id']})")
                            actions_this_round.append(
                                f"meta_accept(delta={delta_m:+.4f}, node={mn['id']})")
                            print(f"  [Meta] graft accepted delta={delta_m:+.4f}")
                            # Gradient propagation: flag siblings and parent
                            gradient_eligible.update(
                                {c["id"] for c in tree.get("children", [])}
                                | {"ROOT"}
                            )
                            gradient_eligible.discard(mn["id"])
                        else:
                            print(f"  [Meta] graft rejected delta={delta_m:+.4f}")

        if errors:
            logger.log_synthesis_input(rnd, errors, node_summaries(tree),
                                       nugget_store is not None)
            try:
                proposal, s_inp, s_out, s_lat = client.synthesize(
                    rnd, errors, node_summaries(tree),
                    nugget_store=nugget_store, error_buffer=error_buffer,
                    principles_store=principles_store,
                    rejected_proposals=rejected_proposals_log,
                )
            except Exception as e:
                print(f"[Round {rnd}] synthesis call failed: {e}")
                proposal = None
                s_inp = s_out = 0

            candidate_tree = None

            if proposal and isinstance(proposal, dict) and proposal.get("new_node"):
                new_node = proposal["new_node"]
                cond     = new_node.get("trigger_condition", "")
                keys_ok  = all(k in new_node for k in
                               ("id", "trigger_condition", "prompt"))
                cond_ok  = is_valid_condition(cond)

                # --- Self-repair: if condition uses invented variable names,
                #     ask the model to fix it before giving up ---
                if keys_ok and not cond_ok:
                    bad = _bad_vars(cond)
                    if bad:
                        print(f"[Round {rnd}] invalid_condition — bad vars {bad} in {cond!r} "
                              f"— attempting self-repair...")
                        logger.log_proposal(proposal, valid=False,
                                            reason=f"invalid vars {bad} — repairing")
                        try:
                            repaired, _, _, _ = client.repair_condition(cond, bad)
                        except Exception as e:
                            repaired = None
                            print(f"[Round {rnd}] repair call failed: {e}")
                        if repaired and is_valid_condition(repaired):
                            print(f"[Round {rnd}] repair succeeded: {repaired!r}")
                            new_node["trigger_condition"] = repaired
                            cond   = repaired
                            cond_ok = True
                        else:
                            print(f"[Round {rnd}] repair failed — repaired={repaired!r}")

                if keys_ok and cond_ok:
                    if cond in _existing_conditions(tree):
                        graft_action = "duplicate_condition"
                        logger.log_proposal(proposal, valid=False,
                                            reason="duplicate_condition")
                    else:
                        candidate_tree = insert_graft(tree, new_node)
                        route_q = _route_quality(
                            candidate_tree, new_node["id"], probe_pool
                        )
                        if route_q["routed"] < args.min_graft_routes:
                            graft_action = (
                                "reject_route_too_narrow"
                                f"(routed={route_q['routed']}/{route_q['total']})"
                            )
                            rejected_proposals_log.append({
                                "round": rnd,
                                "node_id": new_node.get("id", "?"),
                                "condition": new_node.get("trigger_condition", ""),
                                "reason": graft_action,
                            })
                            logger.log_proposal(
                                proposal, valid=False, reason=graft_action
                            )
                            candidate_tree = None
                            print(f"[Round {rnd}] {graft_action}")
                        elif route_q["share"] > args.max_graft_route_share:
                            graft_action = (
                                "reject_route_too_broad"
                                f"(routed={route_q['routed']}/{route_q['total']}, "
                                f"share={route_q['share']:.1%})"
                            )
                            rejected_proposals_log.append({
                                "round": rnd,
                                "node_id": new_node.get("id", "?"),
                                "condition": new_node.get("trigger_condition", ""),
                                "reason": graft_action,
                            })
                            logger.log_proposal(
                                proposal, valid=False, reason=graft_action
                            )
                            candidate_tree = None
                            print(f"[Round {rnd}] {graft_action}")
                        else:
                            graft_action = (
                                "candidate_proposed"
                                f"(routes={route_q['routed']}/{route_q['total']}, "
                                f"share={route_q['share']:.1%})"
                            )
                            logger.log_proposal(
                                proposal, valid=True, reason=graft_action
                            )
                else:
                    graft_action = "invalid_condition"
                    logger.log_proposal(
                        proposal, valid=False,
                        reason=f"keys_ok={keys_ok} cond_ok={cond_ok} cond={cond!r}",
                    )
                    print(f"[Round {rnd}] invalid_condition (unrecoverable) — cond={cond!r}")
            else:
                graft_action = "synthesis_failed"
                logger.log_proposal(proposal, valid=False, reason="synthesis_failed")
                print(f"[Round {rnd}] synthesis_failed — proposal={proposal!r}")

            if candidate_tree is not None:
                # 6. PROBE
                cur_p,  cur_l,  _ = run_eval(
                    tree,           client, probe_pool,
                    sleep_s=args.sleep, nugget_store=nugget_store,
                    phase_label=f"R{rnd} | Probe(current) ",
                )
                cand_p, cand_l, _ = run_eval(
                    candidate_tree, client, probe_pool,
                    sleep_s=args.sleep, nugget_store=nugget_store,
                    phase_label=f"R{rnd} | Probe(candidate)",
                )
                cur_f1  = calc_metrics(cur_p,  cur_l)["f1"]
                cand_f1 = calc_metrics(cand_p, cand_l)["f1"]
                delta   = cand_f1 - cur_f1

                # 7. DECIDE
                if delta > ACCEPT_THRESHOLD:
                    tree         = candidate_tree
                    # Compress newly grafted node's prompt
                    if nugget_store:
                        tree["children"][0]["prompt"] = nugget_store.compress(
                            tree["children"][0]["prompt"]
                        )
                    graft_action = f"accept(delta={delta:+.4f}, node={new_node['id']})"
                    accepted_prompt = (
                        nugget_store.assemble(tree["children"][0]["prompt"])
                        if nugget_store
                        else tree["children"][0]["prompt"]
                    )
                    accepted_changes_log.append(graft_action)

                    # Nugget promotion: record accepted_count for nuggets in this prompt
                    if nugget_store:
                        promoted = nugget_store.record_accepted(
                            tree["children"][0]["prompt"]
                        )
                        if promoted:
                            print(f"  [Nuggets] promoted to CORE: {promoted}")

                    # Gradient propagation: flag parent + siblings for priority refine
                    gradient_eligible.update(
                        {c["id"] for c in tree.get("children", [])
                         if c["id"] != new_node["id"]}
                        | {tree["id"]}   # ROOT
                    )

                    # 9a. EXTRACT from accepted graft
                    if nugget_store and not args.no_extract:
                        _do_extract_nuggets(
                            accepted_prompt, client, nugget_store, logger,
                            rnd, source=f"graft:{new_node['id']}",
                        )

                    # PRINCIPLES: extract what prompt engineering rule made this work
                    try:
                        original_p = (
                            nugget_store.assemble(tree["children"][0]["prompt"])
                            if nugget_store and len(tree.get("children", [])) > 1
                            else new_node.get("prompt", "")
                        )
                        principle, _, _, _ = client.extract_principles(
                            original_p, accepted_prompt, delta_f1=delta,
                        )
                        if principle:
                            pid = principles_store.add_principle(
                                principle, source_round=rnd, delta_f1=delta,
                                source=f"graft:{new_node['id']}",
                            )
                            print(f"  [Principles] learned [{pid}]: {principle[:80]}")
                    except Exception as e:
                        print(f"  [Principles] extraction failed: {e}")

                elif delta < REJECT_THRESHOLD:
                    graft_action = f"reject_regression(delta={delta:+.4f})"
                    rejected_proposals_log.append({
                        "round": rnd, "node_id": new_node.get("id", "?"),
                        "condition": new_node.get("trigger_condition", ""),
                        "reason": graft_action,
                    })
                else:
                    graft_action = f"reject_neutral(delta={delta:+.4f})"
                    rejected_proposals_log.append({
                        "round": rnd, "node_id": new_node.get("id", "?"),
                        "condition": new_node.get("trigger_condition", ""),
                        "reason": graft_action,
                    })

                logger.log_probe("graft", cur_f1, cand_f1, delta, graft_action)
                print(f"[Round {rnd}] probe delta_F1={delta:+.4f} -> {graft_action}")

        actions_this_round.append(graft_action)

        # ----------------------------------------------------------------
        # 8. REFINE — self-optimization of existing node prompts
        # ----------------------------------------------------------------
        refine_actions = []
        # Build combined node_errors including gradient-eligible nodes (lower threshold)
        all_refine_candidates = dict(node_errors)
        for ge_node in gradient_eligible:
            if ge_node not in all_refine_candidates:
                all_refine_candidates[ge_node] = []  # eligible with 0 errors (gradient)
        if gradient_eligible:
            print(f"  [Gradient] priority-refining: {gradient_eligible}")

        if not args.no_refine and all_refine_candidates:
            for node_id, err_list in all_refine_candidates.items():
                # Gradient-eligible nodes bypass the min_refine_errors threshold
                is_gradient = node_id in gradient_eligible
                eligible = is_gradient or len(err_list) >= args.min_refine_errors
                reason = ("gradient_propagation" if is_gradient
                          else ("ok" if eligible
                                else f"<{args.min_refine_errors} errors"))
                logger.log_refine_candidate(node_id, len(err_list), eligible,
                                            reason=reason)
                if not eligible:
                    continue

                node = _find_node(tree, node_id)
                if node is None:
                    continue

                # Skip ROOT refinement if --no-refine-root is set
                if args.no_refine_root and node_id == tree["id"]:
                    logger.log_refine_candidate(node_id, len(err_list), False,
                                                reason="no_refine_root flag")
                    continue

                try:
                    ref_proposal, _, _, _ = client.refine_prompt(
                        node, err_list, nugget_store=nugget_store,
                        error_buffer=error_buffer, principles_store=principles_store,
                    )
                except Exception as e:
                    print(f"[Round {rnd}] refine call failed for {node_id}: {e}")
                    continue

                logger.log_refine_proposal(node_id, ref_proposal)

                if not (isinstance(ref_proposal, dict)
                        and ref_proposal.get("improved_prompt")):
                    refine_actions.append(f"refine_failed:{node_id}")
                    continue

                new_prompt = ref_proposal["improved_prompt"]
                if new_prompt == node["prompt"]:
                    refine_actions.append(f"refine_unchanged:{node_id}")
                    continue

                # Quick probe: swap prompt on refine_probe subset
                candidate_r = _swap_node_prompt(tree, node_id, new_prompt)
                cur_rp,  cur_rl,  _ = run_eval(
                    tree,       client, refine_probe,
                    sleep_s=args.sleep, nugget_store=nugget_store,
                    phase_label=f"R{rnd} | Refine(cur) {node_id[:12]}",
                )
                cand_rp, cand_rl, _ = run_eval(
                    candidate_r, client, refine_probe,
                    sleep_s=args.sleep, nugget_store=nugget_store,
                    phase_label=f"R{rnd} | Refine(new) {node_id[:12]}",
                )
                cur_rf1  = calc_metrics(cur_rp,  cur_rl)["f1"]
                cand_rf1 = calc_metrics(cand_rp, cand_rl)["f1"]
                delta_r  = cand_rf1 - cur_rf1

                accepted_r = delta_r > REFINE_THRESHOLD
                logger.log_refine_result(
                    node_id, node["prompt"], new_prompt, delta_r, accepted_r,
                )
                logger.log_probe(f"refine:{node_id}", cur_rf1, cand_rf1,
                                 delta_r, "accept" if accepted_r else "reject")

                if accepted_r:
                    tree = candidate_r
                    action_r = f"refine_accept(delta={delta_r:+.4f}, node={node_id})"
                    accepted_changes_log.append(action_r)
                    print(f"[Round {rnd}] REFINE accepted for {node_id} "
                          f"delta_F1={delta_r:+.4f}")

                    # Nugget promotion for refined prompt
                    if nugget_store:
                        promoted_r = nugget_store.record_accepted(new_prompt)
                        if promoted_r:
                            print(f"  [Nuggets] promoted to CORE: {promoted_r}")

                    # Gradient propagation: flag sibling/parent of refined node
                    gradient_eligible.update(
                        {c["id"] for c in tree.get("children", [])
                         if c["id"] != node_id}
                        | {tree["id"]}
                    )
                    gradient_eligible.discard(node_id)  # just refined — skip next round

                    # 9b. EXTRACT from refined prompt
                    assembled_refined = (
                        nugget_store.assemble(new_prompt) if nugget_store else new_prompt
                    )
                    if nugget_store and not args.no_extract:
                        _do_extract_nuggets(
                            assembled_refined, client, nugget_store, logger,
                            rnd, source=f"refine:{node_id}",
                        )

                    # PRINCIPLES: what rule made this refine work?
                    try:
                        old_assembled = (
                            nugget_store.assemble(node["prompt"])
                            if nugget_store else node["prompt"]
                        )
                        principle_r, _, _, _ = client.extract_principles(
                            old_assembled, assembled_refined, delta_f1=delta_r,
                        )
                        if principle_r:
                            pid_r = principles_store.add_principle(
                                principle_r, source_round=rnd, delta_f1=delta_r,
                                source=f"refine:{node_id}",
                            )
                            print(f"  [Principles] learned [{pid_r}]: {principle_r[:80]}")
                    except Exception as e:
                        print(f"  [Principles] extraction failed: {e}")

                else:
                    action_r = f"refine_reject(delta={delta_r:+.4f}, node={node_id})"

                refine_actions.append(action_r)

        # Clear gradient eligibility after refine pass — refreshed next round if needed
        gradient_eligible.clear()

        actions_this_round.extend(refine_actions)

        # ----------------------------------------------------------------
        # 10. EVAL — full eval set
        # ----------------------------------------------------------------
        preds, labels, node_ids = run_eval(
            tree, client, eval_pool,
            sleep_s=args.sleep, nugget_store=nugget_store,
            phase_label=f"R{rnd} | Eval      ",
        )
        m   = calc_metrics(preds, labels)
        pnm = compute_per_node_metrics(preds, labels, node_ids)

        if nugget_store:
            for child in tree.get("children", []):
                nugget_store.record_usage(child["prompt"], m["f1"])
            nugget_store.record_usage(tree["prompt"], m["f1"])
            if nugget_path:
                nugget_store.save(nugget_path)

        # Save drug registry each round (Hebbian memory persists across crashes)
        drug_registry.save(registry_path)

        # Save principles store each round (preserves progress on crash)
        if principles_store.path:
            principles_store.save(principles_store.path)
        # Save rejected proposals log
        with open(f"{out}/rejected_proposals.json", "w") as f:
            json.dump(rejected_proposals_log, f, indent=2)

        # Update per-node route + F1 history for retirement tracking
        for nid, stats in pnm.items():
            node_route_history[nid].append(stats.get("count", 0))
            node_f1_history[nid].append(stats.get("f1", 0.0))

        # ----------------------------------------------------------------
        # NODE RETIREMENT — prune chronically underperforming nodes
        # ----------------------------------------------------------------
        retire_actions = []
        if not args.no_retire:
            for child in list(tree.get("children", [])):
                nid = child["id"]
                route_hist = node_route_history.get(nid, [])
                if len(route_hist) < args.retire_min_rounds:
                    continue   # not enough history yet
                avg_routes = sum(route_hist) / len(route_hist)
                avg_f1     = (sum(node_f1_history.get(nid, [0]))
                              / max(1, len(node_f1_history.get(nid, [1]))))
                if avg_routes <= args.retire_max_routes:
                    print(f"  [Retire] {nid}: avg_routes={avg_routes:.1f} "
                          f"over {len(route_hist)} rounds — flagging for retirement")
                    try:
                        ret_p, _, _, _ = client.retire_node(
                            child, routes=int(sum(route_hist)),
                            rounds=len(route_hist), avg_f1=avg_f1,
                        )
                    except Exception as e:
                        print(f"  [Retire] call failed: {e}")
                        ret_p = None

                    if ret_p and isinstance(ret_p, dict):
                        # Remove child from tree
                        retired_tree = deepcopy(tree)
                        retired_tree["children"] = [
                            c for c in retired_tree.get("children", [])
                            if c["id"] != nid
                        ]
                        # Probe: does removing this node hurt?
                        cur_ret, cur_rl, _ = run_eval(
                            tree, client, refine_probe,
                            sleep_s=args.sleep, nugget_store=nugget_store,
                            phase_label=f"R{rnd} | Retire-probe(cur)",
                        )
                        cnd_ret, cnd_rl, _ = run_eval(
                            retired_tree, client, refine_probe,
                            sleep_s=args.sleep, nugget_store=nugget_store,
                            phase_label=f"R{rnd} | Retire-probe(ret)",
                        )
                        delta_ret = (calc_metrics(cnd_ret, cnd_rl)["f1"]
                                     - calc_metrics(cur_ret, cur_rl)["f1"])
                        if delta_ret >= -0.01:  # neutral or positive = safe to retire
                            tree = retired_tree
                            del node_route_history[nid]
                            del node_f1_history[nid]
                            action_ret = f"retire_accept(node={nid}, delta={delta_ret:+.4f})"
                            accepted_changes_log.append(action_ret)
                            print(f"  [Retire] {nid} retired, delta={delta_ret:+.4f}")
                        else:
                            action_ret = f"retire_reject(node={nid}, delta={delta_ret:+.4f})"
                            print(f"  [Retire] {nid} kept, delta={delta_ret:+.4f} too costly")
                        retire_actions.append(action_ret)

        actions_this_round.extend(retire_actions)

        tok = tracker.round_delta()
        combined_action = " | ".join(actions_this_round) if actions_this_round else "no_errors"

        # Per-round timing + ETA
        rnd_elapsed = time.time() - rnd_t0
        round_times.append(rnd_elapsed)
        avg_rnd = sum(round_times) / len(round_times)
        rounds_left = args.rounds - rnd
        eta_display = f"  ETA ~{_fmt_time(avg_rnd * rounds_left)}" if rounds_left else ""

        logger.log_eval(m["f1"], m["precision"], m["recall"], m["accuracy"],
                        pnm, tok["round_tokens_total"], tracker.total_tokens())
        print(f"[Round {rnd}/{args.rounds}] EVAL F1={m['f1']:.3f} P={m['precision']:.3f} "
              f"R={m['recall']:.3f}  tokens={tok['round_tokens_total']:,}  "
              f"elapsed={_fmt_time(rnd_elapsed)}{eta_display}")
        print(f"  action={combined_action}")

        if m["f1"] > best_f1:
            best_f1   = m["f1"]
            best_tree = deepcopy(tree)

        # 11. LOG
        history.append({
            "round": rnd,
            "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "accuracy": m["accuracy"],
            "action": combined_action,
            "tree_nodes": len(node_summaries(tree)),
            "per_node_f1": {nid: v["f1"] for nid, v in pnm.items()},
            **tok,
        })

    # -----------------------------------------------------------------------
    # End-of-run reporting
    # -----------------------------------------------------------------------
    tracker.print_summary()
    if nugget_store:
        nugget_store.print_report()
    principles_store.print_report()
    drug_registry.print_report()

    logger.log_run_complete(
        best_f1, args.rounds,
        tracker.summary_dict(),
        nugget_store.summary_dict() if nugget_store else {},
    )
    logger.close()

    # Append this run's summary to the cross-run chronicle
    run_entry = {
        "run_dir":         out,
        "rounds":          args.rounds,
        "best_f1":         best_f1,
        "final_f1":        history[-1]["f1"] if history else None,
        "accepted_changes": accepted_changes_log,
        "principles_learned": len(principles_store.principles),
        "final_identity":  principles_store.identity,
        "key_insights":    last_meta_analysis[:300] if last_meta_analysis else "",
        "f1_curve":        [h["f1"] for h in history],
        "drug_registry":   drug_registry.summary_stats(),
    }
    chronicle.append(run_entry)
    try:
        with open(chronicle_path, "w") as f:
            json.dump(chronicle, f, indent=2)
        print(f"[Chronicle] saved run summary → {chronicle_path}")
    except Exception as e:
        print(f"[Chronicle] save failed (non-fatal): {e}")

    _save_outputs(out, history, best_tree, tree, best_f1, tracker, nugget_store,
                  principles_store=principles_store)

    # ── Release caffeinate (allow sleep again) ────────────────────────────────
    if _caffeinate is not None:
        _caffeinate.terminate()
        print("[Sleep] caffeinate released — system can sleep again.")


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def _save_outputs(out_dir, history, best_tree, final_tree,
                  best_f1, tracker, nugget_store, principles_store=None):
    df = pd.DataFrame(history)
    # per_node_f1 is a dict column — drop for CSV, keep in JSON
    csv_df = df.drop(columns=["per_node_f1"], errors="ignore")
    csv_df.to_csv(f"{out_dir}/nexus_results.csv", index=False)
    save_tree(best_tree,  f"{out_dir}/nexus_best_tree.json")
    save_tree(final_tree, f"{out_dir}/nexus_final_tree.json")

    summary = {
        "best_f1":           best_f1,
        "rounds_run":        len(history) - 1,
        "final_tree_nodes":  len(node_summaries(final_tree)),
        "token_usage":       tracker.summary_dict(),
        "nugget_store":      nugget_store.summary_dict() if nugget_store else None,
        "principles_store":  principles_store.summary_dict() if principles_store else None,
        "history":           history,
    }
    with open(f"{out_dir}/nexus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot(out_dir, df)

    print(f"\nDone. Best F1 = {best_f1:.3f}")
    print(f"Outputs: {out_dir}/nexus_results.csv, nexus_best_tree.json, "
          "nexus_final_tree.json, nexus_summary.json, nexus_nuggets.json, "
          "nexus_f1_curve.png, nexus_debug.log")


def _plot(out_dir, df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(df["round"], df["f1"],       marker="o", label="F1")
        ax.plot(df["round"], df["precision"], marker="s", label="Precision", alpha=0.7)
        ax.plot(df["round"], df["recall"],    marker="^", label="Recall",    alpha=0.7)
        ax.set_xlabel("Round"); ax.set_ylabel("Score")
        ax.set_title("NEXUS: F1 / Precision / Recall by Round")
        ax.legend(); ax.grid(alpha=0.3)

        if "round_tokens_total" in df.columns:
            ax2 = axes[1]
            cumtok = df["round_tokens_total"].cumsum()
            ax2.bar(df["round"], df["round_tokens_total"],
                    alpha=0.6, color="steelblue", label="Per-round tokens")
            ax2.plot(df["round"], cumtok, color="red", marker="o",
                     label="Cumulative tokens")
            ax2.set_xlabel("Round"); ax2.set_ylabel("Tokens")
            ax2.set_title("Token Usage per Round")
            ax2.legend(); ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(f"{out_dir}/nexus_f1_curve.png", dpi=150)
        print(f"Saved plot to {out_dir}/nexus_f1_curve.png")
    except Exception as e:
        print(f"Plotting failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Synthetic data generator (--mock mode only)
# ---------------------------------------------------------------------------

def _make_synthetic_pools(eval_size, probe_size, train_size, seed):
    rng = random.Random(seed)
    templates_ade = [
        "Patient developed {drug}-induced hepatotoxicity after treatment.",
        "{drug} toxicity was reported following high-dose therapy.",
        "Severe hypersensitivity reaction associated with {drug} administration.",
        "Case report: {drug}-associated nephrotoxicity in an elderly patient.",
        "{drug} toxicity",
        "Severe {drug}-induced pancytopenia developed during therapy.",
        "Anaphylaxis following {drug} infusion required ICU admission.",
    ]
    templates_not = [
        "Patient tolerated {drug} well with no adverse effects.",
        "{drug} was not associated with any side effects in this cohort.",
        "No reaction was observed following {drug} administration.",
        "{drug} remains an effective first-line therapy for this condition.",
        "Routine follow-up after {drug} initiation showed stable labs.",
        "{drug} showed excellent tolerability in the treatment group.",
        "No significant adverse events were attributed to {drug}.",
    ]
    drugs = ["methotrexate", "vancomycin", "cisplatin", "lithium", "warfarin",
             "clozapine", "valproate", "phenytoin", "tacrolimus", "amiodarone",
             "rituximab", "cyclosporine", "doxorubicin", "carbamazepine"]

    def gen(n):
        out = []
        for _ in range(n):
            drug  = rng.choice(drugs)
            is_ade = rng.random() < 0.5
            tmpl  = rng.choice(templates_ade if is_ade else templates_not)
            out.append({"text": tmpl.format(drug=drug),
                        "label": "ADE" if is_ade else "NOT_ADE"})
        return out

    return gen(eval_size), gen(probe_size), gen(train_size)


if __name__ == "__main__":
    main()

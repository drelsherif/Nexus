"""
debug_logger.py
Comprehensive observability for NEXUS runs.

Writes a structured append-mode log to nexus_debug.log capturing every
decision the pipeline makes: routing, API calls, proposals, probe results,
refinements, and nugget extractions. Essential for the research paper —
this log is the evidence trail for the self-optimization claims.

Usage:
    logger = DebugLogger(path="./nexus_debug.log", console=False)
    logger.set_round(1)
    logger.log_routing(text, feats, node_id)
    ...
    logger.close()

Set console=True to also print everything to stdout (very verbose).
The default is console=False — the file always captures everything.
"""

import json
import time
from collections import defaultdict
from datetime import datetime


class DebugLogger:
    """
    Structured logger for NEXUS. Every method appends a labelled line
    (or block) to the log file. Caller controls whether it also echoes
    to the console via the `console` flag.
    """

    def __init__(self, path: str, console: bool = False):
        self.path = path
        self.console = console
        self._fh = open(path, "a", buffering=1)  # line-buffered
        self._round = 0
        self._round_t0 = time.time()

        # Accumulated per-round counters (reset by set_round)
        self._round_routes: dict = defaultdict(int)          # node_id -> count routed
        self._round_correct: dict = defaultdict(int)         # node_id -> count correct
        self._round_wrong: dict = defaultdict(int)           # node_id -> count wrong

        self._write_separator("=")
        self._write(f"RUN START  {datetime.now().isoformat()}  path={path}")
        self._write_separator("=")

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def set_round(self, rnd: int):
        self._round = rnd
        self._round_t0 = time.time()
        self._round_routes = defaultdict(int)
        self._round_correct = defaultdict(int)
        self._round_wrong = defaultdict(int)
        self._write_separator("-")
        self._write(f"ROUND {rnd}  started={datetime.now().isoformat()}")
        self._write_separator("-")

    def round_elapsed(self) -> float:
        return time.time() - self._round_t0

    # ------------------------------------------------------------------
    # Classification / routing
    # ------------------------------------------------------------------

    def log_routing(self, text: str, feats: dict, node_id: str):
        """Called for every case routed through the tree."""
        active = [k for k, v in feats.items() if v]
        self._round_routes[node_id] += 1
        self._write(
            f"  ROUTE  node={node_id}  feats={active}  "
            f"text={text[:80]!r}"
        )

    def log_prediction(self, text: str, node_id: str, pred: str, label: str,
                       inp_tok: int, out_tok: int, latency_ms: float,
                       rationale: str = ""):
        """Called after classify() returns, when ground truth is available."""
        correct = pred == label
        mark = "✓" if correct else "✗"
        if correct:
            self._round_correct[node_id] += 1
        else:
            self._round_wrong[node_id] += 1
        self._write(
            f"  PRED {mark} node={node_id} pred={pred} gt={label} "
            f"tok={inp_tok}+{out_tok} lat={latency_ms:.0f}ms "
            f"rationale={rationale[:60]!r}  text={text[:60]!r}"
        )

    # ------------------------------------------------------------------
    # Error analysis
    # ------------------------------------------------------------------

    def log_error_summary(self, errors: list):
        """Log all misclassified cases for the round batch."""
        self._write(f"\n  ERRORS this batch: {len(errors)}")
        for i, e in enumerate(errors, 1):
            self._write(
                f"    [{i}] node={e['node']} GT={e['label']} PRED={e['pred']} "
                f"text={e['text'][:80]!r}"
            )
        # Per-node routing summary
        self._write(f"\n  ROUTING SUMMARY (batch):")
        for nid in sorted(self._round_routes):
            r = self._round_routes[nid]
            c = self._round_correct[nid]
            w = self._round_wrong[nid]
            acc = c / r if r else 0.0
            self._write(f"    {nid}: routed={r} correct={c} wrong={w} acc={acc:.2f}")

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def log_synthesis_input(self, round_num: int, error_cases: list,
                            node_list: list, nuggets_used: bool):
        self._write(
            f"\n  SYNTHESIZE  round={round_num} errors={len(error_cases)} "
            f"nodes={len(node_list)} nuggets_in_prompt={nuggets_used}"
        )

    def log_proposal(self, proposal, valid: bool, reason: str):
        """Log the full synthesis proposal and why it was accepted/rejected."""
        self._write(f"\n  PROPOSAL  valid={valid}  reason={reason}")
        self._write(f"  PROPOSAL_JSON={json.dumps(proposal, indent=4)}")

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def log_probe(self, context: str, cur_f1: float, cand_f1: float,
                  delta: float, decision: str):
        """Log probe comparison result. context is e.g. 'graft' or 'refine:NODE_X'."""
        self._write(
            f"\n  PROBE [{context}]  cur_F1={cur_f1:.4f}  cand_F1={cand_f1:.4f}  "
            f"delta={delta:+.4f}  decision={decision}"
        )

    # ------------------------------------------------------------------
    # Self-optimization: REFINE
    # ------------------------------------------------------------------

    def log_refine_candidate(self, node_id: str, n_errors: int, eligible: bool,
                             reason: str = ""):
        self._write(
            f"\n  REFINE_CANDIDATE  node={node_id}  errors={n_errors}  "
            f"eligible={eligible}  reason={reason}"
        )

    def log_refine_proposal(self, node_id: str, proposal):
        self._write(f"\n  REFINE_PROPOSAL  node={node_id}")
        self._write(f"  REFINE_JSON={json.dumps(proposal, indent=4)}")

    def log_refine_result(self, node_id: str, old_prompt: str, new_prompt: str,
                          delta_f1: float, accepted: bool):
        self._write(
            f"\n  REFINE_RESULT  node={node_id}  delta_F1={delta_f1:+.4f}  "
            f"accepted={accepted}"
        )
        if accepted:
            self._write(f"    OLD_PROMPT: {old_prompt[:120]!r}")
            self._write(f"    NEW_PROMPT: {new_prompt[:120]!r}")
        else:
            self._write(f"    (prompt unchanged — improvement below threshold)")

    # ------------------------------------------------------------------
    # Nugget extraction
    # ------------------------------------------------------------------

    def log_nugget_extraction(self, source: str, candidates: list,
                              added: list, skipped: list):
        """Log which nuggets were proposed, added, and why some were skipped."""
        self._write(
            f"\n  NUGGET_EXTRACT [{source}]  "
            f"proposed={len(candidates)}  added={len(added)}  skipped={len(skipped)}"
        )
        for n in added:
            self._write(f"    +ADD  [{n['id']}]: {n['text'][:80]!r}")
        for n in skipped:
            self._write(f"    -SKIP [{n.get('id','?')}]: {n.get('reason','?')}")

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    def log_eval(self, f1: float, precision: float, recall: float,
                 accuracy: float, per_node: dict, tokens_this_round: int,
                 cumulative_tokens: int):
        elapsed = self.round_elapsed()
        self._write(
            f"\n  EVAL  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
            f"acc={accuracy:.4f}  elapsed={elapsed:.1f}s  "
            f"round_tokens={tokens_this_round:,}  total_tokens={cumulative_tokens:,}"
        )
        self._write("  PER_NODE_EVAL:")
        for nid, stats in sorted(per_node.items()):
            self._write(
                f"    {nid}: n={stats['count']}  "
                f"F1={stats['f1']:.3f}  P={stats['precision']:.3f}  "
                f"R={stats['recall']:.3f}  "
                f"tp={stats['tp']} fp={stats['fp']} fn={stats['fn']} tn={stats['tn']}"
            )

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------

    def log_run_complete(self, best_f1: float, rounds: int,
                         token_summary: dict, nugget_summary: dict):
        self._write_separator("=")
        self._write(f"RUN COMPLETE  {datetime.now().isoformat()}")
        self._write(f"  best_F1={best_f1:.4f}  rounds={rounds}")
        self._write(f"  TOKEN_SUMMARY={json.dumps(token_summary)}")
        self._write(f"  NUGGET_SUMMARY={json.dumps(nugget_summary)}")
        self._write_separator("=")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, msg: str):
        self._fh.write(msg + "\n")
        if self.console:
            print(msg)

    def _write_separator(self, char: str = "-", width: int = 72):
        self._write(char * width)

    def close(self):
        self._fh.close()

"""
homeostatic.py
NEXUS — Self-healing homeostatic controller.

Detects performance degradation after each eval round and dispatches the
minimum intervention required to recover. All interventions are:
  1. Probe-validated before committing (same pattern as graft validation)
  2. Logged to the DB with their delta_F1 outcome
  3. Never repeated if they already failed in this run

Intervention menu (ordered by invasiveness / cost):
  ─────────────────────────────────────────────────
  Tier 1 — Reversible, zero structural change
    • principle_rollback      Remove overcorrecting principle, re-probe
    • principle_refinement    LLM rewrites a flawed principle given error evidence
    • route_weight_reset      Reset Hebbian weights to equal, re-probe

  Tier 2 — Structural, reversible via save/load
    • trigger_narrowing       Retire newest node if it's net-negative on probe
    • node_retirement         Probe-retire any child node whose removal helps

  Tier 3 — Generative (LLM proposes new signals)
    • feature_flag_proposal   LLM proposes new routing feature flag from error taxonomy

Design principle: the controller is task-agnostic. Domain specifics live
in the task config and node prompts — not here.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from health_monitor import HealthMonitor, DegradationType, HealthReport

logger = logging.getLogger(__name__)

# ── Probe delta thresholds ────────────────────────────────────────────────────
COMMIT_THRESHOLD   = 0.0     # Probe must improve F1 by at least this to commit
ROLLBACK_MIN_DELTA = 0.001   # Rollback commits even on tiny improvement (conservative)


class HomeostaticController:
    """
    Monitors system health after each eval and dispatches the minimum
    intervention required to recover performance.

    Usage in nexus_v3.py main loop:
        controller = HomeostaticController(task_config, freeform_llm_fn, db)
        ...
        applied = controller.run(
            round_num, eval_metrics, eval_history,
            tree, probe_cases, global_rag_index, route_llm_fn, workers
        )
    """

    def __init__(
        self,
        task_config,
        freeform_llm_fn: Callable,   # sonnet-class LLM for generating refinements
        db=None,
    ):
        self.task_config      = task_config
        self.freeform_llm_fn  = freeform_llm_fn
        self.db               = db
        self.monitor          = HealthMonitor(task_config)
        self._failed: set[str] = set()  # intervention types that failed this run

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        round_num:        int,
        eval_metrics:     dict,
        eval_history:     list[dict],   # full history from DB
        tree,
        probe_cases:      list[dict],
        global_rag_index=None,
        route_llm_fn:     Callable = None,
        workers:          int = 4,
    ) -> list[str]:
        """
        Called after each eval. Returns list of intervention names committed.
        """
        report = self.monitor.assess(eval_history)

        if report.status == DegradationType.HEALTHY:
            return []

        print(f"\n  [HOMEOSTATIC] {report.status.value.upper()} — {report.message}")

        ctx = dict(
            round_num=round_num, tree=tree,
            probe_cases=probe_cases, route_llm_fn=route_llm_fn,
            global_rag_index=global_rag_index, workers=workers,
        )

        applied = []

        if report.status == DegradationType.PRINCIPLE_OVERCORRECT:
            applied += self._dispatch(["principle_rollback", "principle_refinement"], ctx, report)

        elif report.status == DegradationType.POST_GRAFT_DISRUPTION:
            applied += self._dispatch(["trigger_narrowing", "node_retirement"], ctx, report)

        elif report.status == DegradationType.RECALL_COLLAPSE:
            applied += self._dispatch(["route_weight_reset", "principle_rollback"], ctx, report)

        elif report.status == DegradationType.PRECISION_COLLAPSE:
            applied += self._dispatch(["principle_rollback", "trigger_narrowing"], ctx, report)

        elif report.status == DegradationType.PLATEAU:
            applied += self._dispatch(["feature_flag_proposal", "route_weight_reset"], ctx, report)

        elif report.status == DegradationType.GENERAL_DECLINE:
            applied += self._dispatch([
                "principle_rollback", "trigger_narrowing",
                "route_weight_reset", "node_retirement",
            ], ctx, report)

        if applied:
            print(f"  [HOMEOSTATIC] Committed: {', '.join(applied)}")
        else:
            print(f"  [HOMEOSTATIC] No intervention improved probe F1 — continuing.")

        return applied

    # ── Intervention dispatcher ───────────────────────────────────────────────

    def _dispatch(self, intervention_order: list[str], ctx: dict, report: HealthReport) -> list[str]:
        """
        Try interventions in order. Stop after first successful commit.
        Skip any that previously failed in this run.
        """
        for name in intervention_order:
            if name in self._failed:
                logger.debug(f"[HOMEOSTATIC] Skipping {name} — already failed this run.")
                continue

            fn = getattr(self, f"_intervention_{name}", None)
            if fn is None:
                continue

            print(f"  [HOMEOSTATIC] Trying {name}...")
            committed, delta = fn(**ctx)
            if committed:
                if self.db:
                    self.db.log_intervention(ctx["round_num"], name, committed=True, delta_f1=delta)
                return [name]
            else:
                self._failed.add(name)
                if self.db:
                    self.db.log_intervention(ctx["round_num"], name, committed=False, delta_f1=delta)

        return []

    # ── Intervention implementations ──────────────────────────────────────────

    def _intervention_principle_rollback(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Remove the most recently injected principle from any node and re-probe.
        Commits if probe F1 improves.
        """
        # Find node + principle added most recently
        best_node = None
        latest_round = -1
        best_p_idx   = -1

        for node in tree.all_nodes():
            for i, p in enumerate(node.injected_principles):
                r = p.get("round_added", 0) if isinstance(p, dict) else 0
                if r > latest_round:
                    latest_round = r
                    best_node    = node
                    best_p_idx   = i

        if best_node is None or best_p_idx < 0:
            print(f"    No rollback-able principle found.")
            return False, 0.0

        baseline = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)

        # Remove principle
        removed = best_node.injected_principles.pop(best_p_idx)
        best_node.rebuild_prompt()

        candidate = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
        delta = candidate["f1"] - baseline["f1"]

        if delta >= ROLLBACK_MIN_DELTA:
            print(f"    Rolled back principle from {best_node.id} — ΔF1={delta:+.4f}")
            return True, delta
        else:
            # Restore
            best_node.injected_principles.insert(best_p_idx, removed)
            best_node.rebuild_prompt()
            print(f"    Rollback did not help (ΔF1={delta:+.4f}) — restored.")
            return False, delta

    def _intervention_principle_refinement(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Ask the LLM to refine the most recently added principle given evidence
        of what it broke. Commits the refined version if it improves probe F1.
        """
        # Find most recent principle
        best_node = None
        latest_round = -1
        best_p_idx   = -1

        for node in tree.all_nodes():
            for i, p in enumerate(node.injected_principles):
                r = p.get("round_added", 0) if isinstance(p, dict) else 0
                if r > latest_round:
                    latest_round = r
                    best_node    = node
                    best_p_idx   = i

        if best_node is None or best_p_idx < 0:
            return False, 0.0

        original_p = best_node.injected_principles[best_p_idx]
        principle_text = original_p.get("principle", "") if isinstance(original_p, dict) else str(original_p)

        if not principle_text:
            return False, 0.0

        baseline = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)

        # Ask LLM to refine the principle
        system = (
            f"You are NEXUS, refining a classification principle for {self.task_config.task_name}. "
            f"The principle was added to improve performance but may have overcorrected."
        )
        prompt = f"""The following principle was added to node {best_node.id} but appears to have hurt performance:

ORIGINAL PRINCIPLE:
{principle_text[:600]}

The principle may be:
- Too broad (applies to cases it shouldn't)
- Missing an important exception
- Overcorrecting toward one class

Rewrite the principle to be more precise and balanced. Keep the core insight but add the necessary constraints.

Respond ONLY with the improved principle text (no JSON, no headers):"""

        try:
            refined_text = self.freeform_llm_fn(system, prompt).strip()
            if not refined_text or len(refined_text) < 50:
                return False, 0.0

            # Temporarily apply refined version
            if isinstance(original_p, dict):
                best_node.injected_principles[best_p_idx] = {
                    **original_p,
                    "principle": refined_text,
                    "refined_at_round": round_num,
                }
            else:
                best_node.injected_principles[best_p_idx] = refined_text
            best_node.rebuild_prompt()

            candidate = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
            delta = candidate["f1"] - baseline["f1"]

            if delta >= COMMIT_THRESHOLD:
                print(f"    Refined principle in {best_node.id} — ΔF1={delta:+.4f}")
                return True, delta
            else:
                # Restore original
                best_node.injected_principles[best_p_idx] = original_p
                best_node.rebuild_prompt()
                print(f"    Refinement did not help (ΔF1={delta:+.4f}) — restored.")
                return False, delta

        except Exception as e:
            logger.warning(f"[HOMEOSTATIC] principle_refinement failed: {e}")
            return False, 0.0

    def _intervention_route_weight_reset(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Reset all route weights to equal (1.0) and re-probe.
        Commits if it helps; restores if not.
        """
        baseline = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)

        # Save and reset — weights is a dict[str, float]
        saved = {}
        for node in tree.all_nodes():
            saved[node.id] = dict(node.aggregator.weights)  # deep copy
            node.aggregator.weights = {k: 1.0 for k in node.aggregator.weights}

        candidate = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
        delta = candidate["f1"] - baseline["f1"]

        if delta >= COMMIT_THRESHOLD:
            print(f"    Route weights reset — ΔF1={delta:+.4f}")
            return True, delta
        else:
            for node in tree.all_nodes():
                node.aggregator.weights = saved[node.id]
            print(f"    Weight reset did not help (ΔF1={delta:+.4f}) — restored.")
            return False, delta

    def _intervention_trigger_narrowing(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Identify the newest node (fewest rounds active) and probe whether
        removing it improves F1. Acts as trigger narrowing — effectively
        retiring an over-broad new node.
        """
        children = tree.all_child_nodes()
        if not children:
            return False, 0.0

        # Newest node = fewest non-zero route history entries
        def _age(n):
            return sum(1 for r in n.route_history if r > 0)

        candidates = sorted(children, key=_age)
        target = candidates[0]

        baseline = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)

        # Temporarily remove
        parent = self._find_parent(tree, target.id)
        if parent is None:
            return False, 0.0

        parent.children = [c for c in parent.children if c.id != target.id]
        tree._rebuild_index()

        candidate = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
        delta = candidate["f1"] - baseline["f1"]

        if delta >= COMMIT_THRESHOLD:
            print(f"    Retired {target.id} via trigger narrowing — ΔF1={delta:+.4f}")
            return True, delta
        else:
            # Restore
            parent.children.append(target)
            tree._rebuild_index()
            print(f"    Trigger narrowing of {target.id} did not help (ΔF1={delta:+.4f}) — restored.")
            return False, delta

    def _intervention_node_retirement(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Probe retiring each child node. Commit the retirement that
        maximally improves probe F1.
        """
        children = tree.all_child_nodes()
        if not children:
            return False, 0.0

        baseline = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
        best_delta = COMMIT_THRESHOLD
        best_target = None

        for node in children:
            parent = self._find_parent(tree, node.id)
            if parent is None:
                continue

            parent.children = [c for c in parent.children if c.id != node.id]
            tree._rebuild_index()

            candidate = self._probe(tree, probe_cases, route_llm_fn, global_rag_index, workers)
            delta = candidate["f1"] - baseline["f1"]

            if delta > best_delta:
                best_delta  = delta
                best_target = node

            # Restore this node for next iteration
            parent.children.append(node)
            tree._rebuild_index()

        if best_target:
            parent = self._find_parent(tree, best_target.id)
            parent.children = [c for c in parent.children if c.id != best_target.id]
            tree._rebuild_index()
            print(f"    Retired {best_target.id} — ΔF1={best_delta:+.4f}")
            return True, best_delta

        print(f"    No retirement improved probe F1.")
        return False, 0.0

    def _intervention_feature_flag_proposal(
        self, round_num, tree, probe_cases, route_llm_fn,
        global_rag_index, workers, **_
    ) -> tuple[bool, float]:
        """
        Ask the LLM to propose a new feature flag (regex-based routing signal)
        based on the current error taxonomy. Tests whether any probe case
        would be rerouted beneficially.

        Note: this proposes a new flag but does NOT modify the Python code directly.
        It logs the proposal to the DB for human review. Returns False always
        (informational only until code_modifier.py is implemented).
        """
        # Collect error taxonomy from all nodes
        error_types = []
        for node in tree.all_nodes():
            for err in list(node.error_buffer)[-10:]:
                error_types.append(f"  Node={node.id}  True={err.get('true_label','')}  "
                                   f"Pred={err.get('predicted_label','')}  "
                                   f"Text={err.get('text','')[:100]}")

        if not error_types:
            return False, 0.0

        existing_flags = list(self.task_config.feature_flags.keys())

        system = f"You are NEXUS, proposing a new routing signal for {self.task_config.task_name}."
        prompt = f"""The system has plateaued. Current feature flags: {existing_flags}

Recent misclassified cases:
{chr(10).join(error_types[:15])}

Propose ONE new boolean feature flag that would improve routing for this error pattern.
The flag should be detectable by a simple Python regex on the sentence text.

Respond ONLY with JSON:
{{
  "flag_name": "has_DESCRIPTIVE_NAME",
  "regex_pattern": "<python regex string>",
  "rationale": "<one sentence: what linguistic pattern does this detect?>",
  "proposed_node_trigger": "<suggested trigger expression using existing + new flag>"
}}"""

        try:
            import re, json as _json
            raw = self.freeform_llm_fn(system, prompt).strip()
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                proposal = _json.loads(m.group())
                print(f"  [HOMEOSTATIC] Feature flag proposal: {proposal.get('flag_name')} — "
                      f"{proposal.get('rationale','')}")
                if self.db:
                    self.db.log_intervention(
                        round_num, "feature_flag_proposal",
                        committed=False, delta_f1=0.0,
                        detail=str(proposal),
                    )
        except Exception as e:
            logger.warning(f"[HOMEOSTATIC] feature_flag_proposal failed: {e}")

        # Always returns False — proposals require human review before code change
        return False, 0.0

    # ── Probe helper ──────────────────────────────────────────────────────────

    def _probe(self, tree, probe_cases, route_llm_fn, global_rag_index, workers) -> dict:
        pos = self.task_config.positive_label
        tp = fp = fn = tn = 0
        for c in probe_cases:
            result = tree.classify(c["text"], route_llm_fn, global_rag_index, workers)
            pred, true = result.label, c["label"]
            if pred == pos and true == pos:    tp += 1
            elif pred == pos and true != pos:  fp += 1
            elif pred != pos and true == pos:  fn += 1
            else:                              tn += 1
        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1     = 2 * prec * recall / max(1e-9, prec + recall)
        return {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(recall, 4)}

    def _find_parent(self, tree, node_id: str):
        """Find the parent node of the given node_id."""
        for node in tree.all_nodes():
            for child in node.children:
                if child.id == node_id:
                    return node
        return None

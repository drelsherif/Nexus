"""
micro_learner.py
Single-case processor for NEXUS micro-learning mode.

Biological analogy:
  MicroLearner  =  CA3 pyramidal cell (rapid one-shot encoding)
  MicroRule     =  Synaptic trace from a single episode
  RuleDictionary = CA1 / entorhinal cortex (pattern accumulation)
  Consolidation =  Neocortical replay during SWR

Architecture vs. current node-refine loop:
  OLD: batch 50 → LLM proposes new node prompt → probe 300 → accept/reject
       Cost: ~350 LLM calls per accepted change, most rejected
  NEW: 10 workers × 1 case → rule if wrong → accumulate free
       Cost: 1 LLM call per misclassification (rule generation)
             1 LLM call per threshold event (consolidation, rare)

Usage:
    from micro_learner import run_micro_batch
    from rule_dictionary import RuleDictionary

    rule_dict = RuleDictionary(threshold=5, path=f"{out}/rule_dict.json")

    def classify_fn(text):
        feats = features(text)
        node  = classify_with_tree(tree, feats)
        result, _, _, _ = client.classify(text, node["prompt"])
        return node["id"], result.get("classification", "NOT_ADE")

    results = run_micro_batch(
        cases=batch,
        classify_fn=classify_fn,
        rule_gen_fn=rule_gen_fn,
        rule_dict=rule_dict,
        round_num=rnd,
        workers=10,
    )
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

from rule_dictionary import RuleDictionary, MicroRule


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    text: str
    true_label: str
    predicted_label: str
    correct: bool
    node_id: str
    rule: MicroRule | None = None
    fired: bool = False


@dataclass
class BatchResult:
    total: int
    correct: int
    fired_keys: list[str]
    case_results: list[CaseResult]

    @property
    def accuracy(self) -> float:
        return self.correct / max(1, self.total)

    @property
    def error_count(self) -> int:
        return self.total - self.correct


# ─── Rule generation prompt ───────────────────────────────────────────────────

_RULE_GEN_SYSTEM = """You are a clinical pharmacovigilance expert and a component of NEXUS,
a self-improving ADE (Adverse Drug Event) classifier.

You will be shown a sentence that NEXUS misclassified and the correct label.
Your task: identify the specific linguistic or clinical pattern that NEXUS missed,
and express it as a concise, generalizable rule.

Output JSON only — no prose, no markdown fences:
{
  "pattern": "<the generalizable pattern, 5-15 words>",
  "signal": "<ADE or NOT_ADE>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining why this pattern signals the label>"
}"""


def _rule_gen_user(text: str, true_label: str, predicted_label: str, node_id: str) -> str:
    return (
        f'Sentence: "{text}"\n\n'
        f"NEXUS predicted: {predicted_label}\n"
        f"Correct answer:  {true_label}\n"
        f"Node that classified it: {node_id}\n\n"
        f"What pattern in this sentence should have led NEXUS to predict {true_label}?\n"
        "Respond with JSON only."
    )


# ─── Single-case processor ────────────────────────────────────────────────────

class MicroLearner:
    """
    Processes one case at a time.
    On correct classification: nothing.
    On misclassification: one LLM call to generate a micro-rule.

    Args:
        classify_fn:  (text: str) → (node_id: str, label: str)
                      Wraps features() + classify_with_tree() + client.classify()
        rule_gen_fn:  (system: str, user: str) → str
                      LLM call for rule generation (can be same client)
        rule_dict:    Shared RuleDictionary (thread-safe)
        round_num:    Current training round (for provenance)
    """

    def __init__(
        self,
        classify_fn: Callable[[str], tuple[str, str]],
        rule_gen_fn: Callable[[str, str], str],
        rule_dict: RuleDictionary,
        round_num: int = 0,
    ):
        self.classify_fn = classify_fn
        self.rule_gen_fn = rule_gen_fn
        self.rule_dict = rule_dict
        self.round_num = round_num

    def process(self, case: dict) -> CaseResult:
        """Classify one case; generate micro-rule on misclassification."""
        text = case["text"]
        true_label = case["label"]

        try:
            node_id, predicted_label = self.classify_fn(text)
        except Exception:
            return CaseResult(
                text=text, true_label=true_label,
                predicted_label="ERROR", correct=False, node_id="UNKNOWN",
            )

        correct = (predicted_label == true_label)
        if correct:
            return CaseResult(
                text=text, true_label=true_label,
                predicted_label=predicted_label, correct=True, node_id=node_id,
            )

        # Misclassification — generate micro-rule
        rule = self._generate_rule(text, true_label, predicted_label, node_id)
        fired = False
        if rule:
            fired = self.rule_dict.add(rule)

        return CaseResult(
            text=text, true_label=true_label,
            predicted_label=predicted_label, correct=False,
            node_id=node_id, rule=rule, fired=fired,
        )

    def _generate_rule(
        self, text: str, true_label: str, predicted_label: str, node_id: str,
    ) -> MicroRule | None:
        user = _rule_gen_user(text, true_label, predicted_label, node_id)
        try:
            raw = self.rule_gen_fn(_RULE_GEN_SYSTEM, user)
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            data = json.loads(raw)
            return MicroRule(
                pattern=data["pattern"],
                signal=data["signal"],
                confidence=float(data.get("confidence", 0.7)),
                context=text,
                node_id=node_id,
                round_num=self.round_num,
            )
        except Exception:
            return None


# ─── Parallel batch runner ────────────────────────────────────────────────────

def run_micro_batch(
    cases: list[dict],
    classify_fn: Callable[[str], tuple[str, str]],
    rule_gen_fn: Callable[[str, str], str],
    rule_dict: RuleDictionary,
    round_num: int = 0,
    workers: int = 10,
    verbose: bool = True,
) -> BatchResult:
    """
    Run MicroLearner across a batch of cases using a thread pool.
    All workers share the same classify_fn, rule_gen_fn, and rule_dict.
    """
    learner = MicroLearner(
        classify_fn=classify_fn,
        rule_gen_fn=rule_gen_fn,
        rule_dict=rule_dict,
        round_num=round_num,
    )

    results: list[CaseResult | None] = [None] * len(cases)
    done = 0
    total = len(cases)
    lock = threading.Lock()

    if verbose:
        print(
            f"  [MicroLearn] {total} cases | {workers} workers | "
            f"threshold={rule_dict.threshold}",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(learner.process, case): i
            for i, case in enumerate(cases)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = CaseResult(
                    text=cases[idx]["text"], true_label=cases[idx]["label"],
                    predicted_label="ERROR", correct=False, node_id="UNKNOWN",
                )
            with lock:
                done += 1
                if verbose and done % max(1, total // 10) == 0:
                    pct = done / total
                    bar = "█" * int(pct * 20) + " " * (20 - int(pct * 20))
                    errors = sum(1 for r in results if r and not r.correct)
                    print(
                        f"\r  [MicroLearn] [{bar}] {done}/{total}  errors={errors}",
                        end="", flush=True,
                    )

    if verbose:
        print()

    fired_keys = rule_dict.pop_fired()
    correct = sum(1 for r in results if r and r.correct)
    return BatchResult(
        total=total,
        correct=correct,
        fired_keys=fired_keys,
        case_results=[r for r in results if r is not None],
    )


# ─── Consolidation ────────────────────────────────────────────────────────────

def build_consolidation_call(
    key: str,
    rule_dict: RuleDictionary,
    rule_gen_fn: Callable[[str, str], str],
) -> str | None:
    """SWR event: consolidate a threshold-crossing rule into a NEXUS principle."""
    prompt = rule_dict.build_consolidation_prompt(key)
    if not prompt:
        return None
    try:
        return rule_gen_fn(
            "You are NEXUS, a clinical pharmacovigilance classifier. "
            "Write principles precisely and concisely.",
            prompt,
        ).strip()
    except Exception:
        return None

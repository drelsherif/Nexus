"""
expert_routes.py
Parallel expert agents for NEXUS RAG-augmented classification.

Biological analogy:
  Expert routes  =  Parallel cortical processing streams
                    (ventral "what" + dorsal "where" in vision;
                     here: causation / negation / drug-effect / context)

  Each route receives the same input (query + retrieved examples)
  but focuses on a different clinical reasoning dimension.
  Routes run simultaneously via ThreadPoolExecutor.
  Their votes are aggregated by the RouteAggregator.

  This mirrors how the brain integrates parallel expert opinions
  before committing to a percept — no single route is authoritative.

Routes:
  A. CausationRoute  — Is there direct causal language linking drug→harm?
  B. NegationRoute   — Is the adverse outcome explicitly negated?
  C. DrugEffectRoute — Does the retrieved evidence show this drug-effect pair?
  D. ContextRoute    — Therapeutic intent vs. documented adverse outcome?

Each returns:
  {"route": str, "vote": "ADE"|"NOT_ADE", "confidence": float, "reasoning": str}
"""

from __future__ import annotations

import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable


# ─── Route result ─────────────────────────────────────────────────────────────

@dataclass
class RouteResult:
    route: str
    vote: str           # "ADE" or "NOT_ADE"
    confidence: float   # 0.0 – 1.0
    reasoning: str


@dataclass
class AggregatedResult:
    final_label: str
    confidence: float
    route_results: list[RouteResult]
    agreement: float        # fraction of routes that agree with final
    split: bool             # True if routes disagreed (uncertain case)
    ade_score: float = 0.0      # raw weighted ADE vote sum — used for threshold calibration
    not_ade_score: float = 0.0  # raw weighted NOT_ADE vote sum

    def to_dict(self) -> dict:
        return {
            "label":      self.final_label,
            "confidence": round(self.confidence, 3),
            "agreement":  round(self.agreement, 3),
            "split":      self.split,
            "routes": [
                {"route": r.route, "vote": r.vote,
                 "confidence": round(r.confidence, 3), "reasoning": r.reasoning}
                for r in self.route_results
            ],
        }


# ─── Prompt helpers ───────────────────────────────────────────────────────────

def _format_examples(examples: list[dict], max_k: int = 4) -> str:
    lines = []
    for i, ex in enumerate(examples[:max_k]):
        bar = "▲ADE" if ex["label"] == "ADE" else "▽NOT"
        sim = f"{ex['score']:.2f}" if "score" in ex else "?"
        lines.append(f"  [{i+1}] {bar} (sim={sim}) \"{ex['text'][:100]}\"")
    return "\n".join(lines)


def _safe_float(val, default: float = 0.5) -> float:
    """Parse confidence value — handles both numeric and string ('low'/'medium'/'high')."""
    try:
        return float(val)
    except (ValueError, TypeError):
        if isinstance(val, str):
            return {"very high": 0.95, "high": 0.85, "medium": 0.6,
                    "low": 0.3, "very low": 0.1}.get(val.lower().strip(), default)
        return default


def _parse_json_vote(raw: str) -> dict:
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: extract vote from text
        vote = "ADE" if '"ADE"' in raw or "'ADE'" in raw else "NOT_ADE"
        return {"vote": vote, "confidence": 0.5, "reasoning": "parse fallback"}


# ─── Individual route implementations ────────────────────────────────────────

_JSON_SCHEMA = (
    'Respond ONLY with JSON:\n'
    '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
)


def _causation_route(text: str, examples: list[dict], llm_fn: Callable, principle_context: str = "") -> RouteResult:
    """Route A: Is there direct causal language linking a specific drug to harm?"""
    system = (
        "You are a causation expert in pharmacovigilance. "
        "Your task: determine whether the sentence contains DIRECT causal language "
        "linking a specific drug to a harmful or unintended outcome.\n\n"
        "Causal signals: caused, induced, associated with, resulted in, led to, "
        "following [drug], due to [drug], [drug]-related.\n"
        "NOT causal: drug mentioned but no causal link, desired therapeutic effect, "
        "negated outcome, hypothetical.\n\n"
        + _JSON_SCHEMA
        + principle_context
    )
    examples_text = _format_examples(examples)
    user = (
        f'Sentence: "{text}"\n\n'
        f"Similar labeled examples from literature:\n{examples_text}\n\n"
        "Does this sentence show direct drug-to-harm causation? "
        "Vote ADE only if there is explicit causal linkage."
    )
    try:
        raw = llm_fn(system, user)
        d = _parse_json_vote(raw)
        return RouteResult(
            route="causation",
            vote=d.get("vote", "NOT_ADE"),
            confidence=_safe_float(d.get("confidence"), 0.5),
            reasoning=d.get("reasoning", ""),
        )
    except Exception as e:
        print(f"[ROUTE ERROR] causation: {type(e).__name__}: {e}", file=sys.stderr)
        return RouteResult("causation", "NOT_ADE", 0.3, f"error: {e}")


def _negation_route(text: str, examples: list[dict], llm_fn: Callable, principle_context: str = "") -> RouteResult:
    """Route B: Is the adverse outcome explicitly negated or hypothetical?"""
    system = (
        "You are a negation expert in clinical NLP. "
        "Your task: determine whether any adverse outcome in the sentence is "
        "NEGATED, denied, hypothetical, or qualified as absent.\n\n"
        "Negation signals: no, not, without, denied, failed to develop, "
        "did not experience, absence of, ruled out, tolerates well.\n"
        "Scope matters: 'no relief from pain' negates relief NOT an ADE. "
        "'did not develop nephrotoxicity' negates the ADE → NOT_ADE.\n\n"
        + _JSON_SCHEMA
        + principle_context
    )
    examples_text = _format_examples(examples)
    user = (
        f'Sentence: "{text}"\n\n'
        f"Similar labeled examples:\n{examples_text}\n\n"
        "Is the adverse outcome negated or hypothetical? "
        "Vote NOT_ADE if the adverse event is explicitly absent."
    )
    try:
        raw = llm_fn(system, user)
        d = _parse_json_vote(raw)
        return RouteResult(
            route="negation",
            vote=d.get("vote", "NOT_ADE"),
            confidence=_safe_float(d.get("confidence"), 0.5),
            reasoning=d.get("reasoning", ""),
        )
    except Exception as e:
        print(f"[ROUTE ERROR] negation: {type(e).__name__}: {e}", file=sys.stderr)
        return RouteResult("negation", "NOT_ADE", 0.3, f"error: {e}")


def _drug_effect_route(text: str, examples: list[dict], llm_fn: Callable, principle_context: str = "") -> RouteResult:
    """Route C: Do the retrieved examples confirm this drug-effect pair as an ADE?"""
    system = (
        "You are a pharmacology expert. "
        "You have access to retrieved literature examples showing how similar "
        "sentences were labeled in an ADE classification task.\n\n"
        "Your task: based on the retrieved examples, does this sentence describe "
        "a known drug-adverse effect relationship?\n"
        "If the retrieved ADE examples are highly similar (score > 0.85), "
        "weight them heavily. If retrieved examples are mixed or dissimilar, "
        "rely on clinical reasoning.\n\n"
        + _JSON_SCHEMA
        + principle_context
    )
    ade_examples = [e for e in examples if e["label"] == "ADE"]
    not_ade_examples = [e for e in examples if e["label"] == "NOT_ADE"]
    examples_text = (
        f"ADE examples:\n{_format_examples(ade_examples, 3)}\n\n"
        f"NOT_ADE examples:\n{_format_examples(not_ade_examples, 2)}"
    )
    user = (
        f'Sentence: "{text}"\n\n'
        f"{examples_text}\n\n"
        "Based on pharmacological knowledge and retrieved evidence, "
        "is this an ADE?"
    )
    try:
        raw = llm_fn(system, user)
        d = _parse_json_vote(raw)
        return RouteResult(
            route="drug_effect",
            vote=d.get("vote", "NOT_ADE"),
            confidence=_safe_float(d.get("confidence"), 0.5),
            reasoning=d.get("reasoning", ""),
        )
    except Exception as e:
        print(f"[ROUTE ERROR] drug_effect: {type(e).__name__}: {e}", file=sys.stderr)
        return RouteResult("drug_effect", "NOT_ADE", 0.3, f"error: {e}")


def _context_route(text: str, examples: list[dict], llm_fn: Callable, principle_context: str = "") -> RouteResult:
    """Route D: Therapeutic intent vs. documented adverse outcome?"""
    system = (
        "You are a clinical context expert in pharmacovigilance. "
        "Your task: determine whether the described outcome is a desired "
        "therapeutic effect or an unintended/harmful adverse event.\n\n"
        "Therapeutic (NOT_ADE): pain relief, infection cleared, BP controlled, "
        "tumor shrank, glucose normalised.\n"
        "Adverse (ADE): unexpected toxicity, organ damage, hypersensitivity, "
        "side effects not part of therapeutic goal.\n"
        "Also NOT_ADE: general mechanism descriptions, laboratory findings "
        "without documented patient harm, resistance mutations.\n\n"
        + _JSON_SCHEMA
        + principle_context
    )
    examples_text = _format_examples(examples)
    user = (
        f'Sentence: "{text}"\n\n'
        f"Similar labeled examples:\n{examples_text}\n\n"
        "Is this describing a therapeutic intent or an adverse outcome?"
    )
    try:
        raw = llm_fn(system, user)
        d = _parse_json_vote(raw)
        return RouteResult(
            route="context",
            vote=d.get("vote", "NOT_ADE"),
            confidence=_safe_float(d.get("confidence"), 0.5),
            reasoning=d.get("reasoning", ""),
        )
    except Exception as e:
        print(f"[ROUTE ERROR] context: {type(e).__name__}: {e}", file=sys.stderr)
        return RouteResult("context", "NOT_ADE", 0.3, f"error: {e}")


# ─── Route registry ───────────────────────────────────────────────────────────

_ROUTES = {
    "causation":  _causation_route,
    "negation":   _negation_route,
    "drug_effect": _drug_effect_route,
    "context":    _context_route,
}


# ─── Aggregator ───────────────────────────────────────────────────────────────

class RouteAggregator:
    """
    Weighted voting across expert routes.

    Route weights start equal and are updated by historical accuracy:
    - Correct vote → weight += learning_rate
    - Wrong vote   → weight -= learning_rate * penalty

    This mirrors dopaminergic reinforcement — routes that predict well
    gain influence; routes that predict poorly lose it.
    """

    def __init__(
        self,
        routes: list[str] | None = None,
        learning_rate: float = 0.05,
        penalty: float = 1.5,
    ):
        self.routes = routes or list(_ROUTES.keys())
        self.lr = learning_rate
        self.penalty = penalty
        # Weights: start equal, updated by feedback
        self.weights: dict[str, float] = {r: 1.0 for r in self.routes}
        self.history: dict[str, list[bool]] = {r: [] for r in self.routes}

    def classify(
        self,
        text: str,
        examples: list[dict],
        llm_fn: Callable,
        workers: int = 4,
        principle_context: str = "",
    ) -> AggregatedResult:
        """Run all routes in parallel, aggregate votes."""
        route_results: list[RouteResult] = [None] * len(self.routes)  # type: ignore

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_ROUTES[r], text, examples, llm_fn, principle_context): i
                for i, r in enumerate(self.routes)
                if r in _ROUTES
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    route_results[i] = future.result()
                except Exception as e:
                    route_results[i] = RouteResult(
                        self.routes[i], "NOT_ADE", 0.2, f"exception: {e}"
                    )

        results = [r for r in route_results if r is not None]
        return self._aggregate(results)

    # Class-prior bias: corpus is 29% ADE / 71% NOT_ADE.
    # ADE score must exceed NOT_ADE score by this factor to win.
    # 1.0 = symmetric (old behaviour, over-predicts ADE).
    # 1.3 = moderate correction (reduces false positives, slight recall risk).
    ADE_BIAS = 1.3

    def _aggregate(self, results: list[RouteResult]) -> AggregatedResult:
        ade_score = 0.0
        not_ade_score = 0.0
        for r in results:
            w = self.weights.get(r.route, 1.0)
            weighted_conf = r.confidence * w
            if r.vote == "ADE":
                ade_score += weighted_conf
            else:
                not_ade_score += weighted_conf

        total = ade_score + not_ade_score
        if total == 0:
            final = "NOT_ADE"
            conf = 0.5
        else:
            # ADE must beat NOT_ADE * ADE_BIAS to account for class imbalance
            final = "ADE" if ade_score >= not_ade_score * self.ADE_BIAS else "NOT_ADE"
            conf = max(ade_score, not_ade_score) / total

        agreement = sum(1 for r in results if r.vote == final) / max(1, len(results))
        split = agreement < 0.75  # less than 75% agreement = uncertain

        return AggregatedResult(
            final_label=final,
            confidence=conf,
            route_results=results,
            agreement=agreement,
            split=split,
            ade_score=ade_score,
            not_ade_score=not_ade_score,
        )

    def update_weights(self, result: AggregatedResult, true_label: str):
        """Reinforcement learning on route weights based on ground truth."""
        for r in result.route_results:
            correct = (r.vote == true_label)
            self.history[r.route].append(correct)
            if correct:
                self.weights[r.route] = min(3.0, self.weights[r.route] + self.lr)
            else:
                self.weights[r.route] = max(0.1, self.weights[r.route] - self.lr * self.penalty)

    def weight_report(self) -> str:
        lines = ["[Routes] Current weights:"]
        for r, w in sorted(self.weights.items(), key=lambda x: -x[1]):
            acc = ""
            if self.history[r]:
                acc_val = sum(self.history[r]) / len(self.history[r])
                acc = f"  acc={acc_val:.1%} over {len(self.history[r])} cases"
            bar = "█" * int(w * 5)
            lines.append(f"  {r:12s} w={w:.2f} {bar}{acc}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"weights": self.weights, "history_lengths": {r: len(v) for r, v in self.history.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "RouteAggregator":
        agg = cls()
        agg.weights = d.get("weights", agg.weights)
        return agg

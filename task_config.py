"""
task_config.py
NEXUS v3 — Task Configuration Layer

Makes NEXUS generalizable: all domain-specific elements (labels, routes,
features, seed nodes, class priors) live in a JSON config file.

A new classification task needs only:
  1. A labeled corpus (JSONL: {"text": "...", "label": "CLASS_A"})
  2. A task_config.json
  3. An embedding model selection

The tree, MCQ generator, engrams, RAG, and learning loop need no code changes.

Usage:
    config = TaskConfig.load("task_configs/ade_classification.json")
    config = TaskConfig.load("task_configs/medication_errors.json")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Route definition ─────────────────────────────────────────────────────────

@dataclass
class RouteDefinition:
    name: str
    focus: str          # Specialist reasoning focus — injected into route system prompt
    default_vote: str   # What to return on parse failure

    @classmethod
    def from_dict(cls, d: dict) -> "RouteDefinition":
        return cls(
            name=d["name"],
            focus=d["focus"],
            default_vote=d.get("default_vote", ""),
        )

    def to_dict(self) -> dict:
        return {"name": self.name, "focus": self.focus, "default_vote": self.default_vote}


# ─── Seed node ────────────────────────────────────────────────────────────────

@dataclass
class SeedNode:
    id: str
    trigger: Optional[str]   # Boolean expression over feature flags; None = ROOT
    prompt: str
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SeedNode":
        return cls(
            id=d["id"],
            trigger=d.get("trigger"),
            prompt=d["prompt"],
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trigger": self.trigger,
            "prompt": self.prompt,
            "description": self.description,
        }


# ─── Main config ──────────────────────────────────────────────────────────────

@dataclass
class TaskConfig:
    # Identity
    task_name: str
    description: str

    # Labels
    labels: list[str]
    positive_label: str     # The label NEXUS is trying to detect (e.g., "ADE")
    negative_label: str     # The other label

    # Class prior — used only for initial bias estimate; system self-calibrates each round
    class_prior: str        # "auto" or e.g. "0.29"
    ade_bias_softening: float = 0.5   # Initial softening — overridden after round 1

    # Optimization objective — what the system maximizes when self-calibrating threshold.
    # The threshold (ADE_BIAS) is learned automatically each round on the probe set;
    # this declares what metric to optimize, not what the threshold value should be.
    #
    # optimization_beta controls the F-score weighting (F_beta):
    #   beta = 1.0  →  F1  (balanced, default)
    #   beta = 2.0  →  F2  (recall-weighted — missing a positive costs twice as much)
    #   beta = 0.5  →  F0.5 (precision-weighted — false alarms cost twice as much)
    #
    # Use beta > 1 for safety-critical tasks (pharmacovigilance, cancer screening).
    # Use beta < 1 for cost-sensitive tasks (spam filtering, fraud detection).
    # The system finds the threshold that maximizes F_beta on its own probe set.
    optimization_target: str = "fbeta"
    optimization_beta: float = 1.0

    # Embeddings
    embed_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"

    # Routes (define specialist reasoning dimensions)
    route_definitions: list[RouteDefinition] = field(default_factory=list)

    # Feature flags (regex or special keywords)
    feature_flags: dict[str, str] = field(default_factory=dict)

    # Seed tree nodes
    seed_nodes: list[SeedNode] = field(default_factory=list)

    # Hyperparameters
    hyperparameters: dict = field(default_factory=dict)

    # ── Computed at runtime ────────────────────────────────────────────────────

    _computed_ade_bias: Optional[float] = field(default=None, repr=False, compare=False)

    # ── Loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "TaskConfig":
        data = json.loads(Path(path).read_text())
        return cls(
            task_name=data["task_name"],
            description=data.get("description", ""),
            labels=data["labels"],
            positive_label=data["positive_label"],
            negative_label=data["negative_label"],
            class_prior=str(data.get("class_prior", "auto")),
            ade_bias_softening=float(data.get("ade_bias_softening", 0.5)),
            optimization_target=data.get("optimization_target", "fbeta"),
            optimization_beta=float(data.get("optimization_beta", 1.0)),
            embed_model=data.get("embed_model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            route_definitions=[
                RouteDefinition.from_dict(r)
                for r in data.get("route_definitions", [])
            ],
            feature_flags=data.get("feature_flags", {}),
            seed_nodes=[SeedNode.from_dict(n) for n in data.get("seed_nodes", [])],
            hyperparameters=data.get("hyperparameters", {}),
        )

    def get_hyperparameter(self, key: str, default=None):
        return self.hyperparameters.get(key, default)

    # ── Class prior calibration ────────────────────────────────────────────────

    def calibrate_from_corpus(self, corpus: list[dict]) -> float:
        """
        Compute ADE_BIAS from training data class frequencies.

        The bias corrects for class imbalance in the aggregator:
          ade_score >= not_ade_score * ADE_BIAS  →  predict positive_label

        ADE_BIAS = 1.0  →  symmetric (balanced dataset)
        ADE_BIAS > 1.0  →  positive label needs stronger evidence to win (imbalanced)
        ADE_BIAS < 1.0  →  positive label wins more easily (rare, for recall-critical tasks)

        Softening parameter prevents over-correction:
          full_bias = not_ade_prior / ade_prior
          ADE_BIAS  = 1.0 + (full_bias - 1.0) * (1 - softening)
        """
        if self.class_prior != "auto":
            try:
                pos_prior = float(self.class_prior)
            except ValueError:
                pos_prior = 0.5
        else:
            n_pos = sum(1 for c in corpus if c["label"] == self.positive_label)
            n_total = len(corpus)
            pos_prior = n_pos / max(1, n_total)

        neg_prior = 1.0 - pos_prior
        full_bias = neg_prior / max(1e-9, pos_prior)  # e.g., 71/29 = 2.45
        softened_bias = 1.0 + (full_bias - 1.0) * (1.0 - self.ade_bias_softening)

        self._computed_ade_bias = max(1.0, softened_bias)
        return self._computed_ade_bias

    @property
    def ade_bias(self) -> float:
        if self._computed_ade_bias is None:
            raise RuntimeError(
                "ADE_BIAS not yet computed. Call config.calibrate_from_corpus(train_pool) first."
            )
        return self._computed_ade_bias

    # ── Feature flag compilation ───────────────────────────────────────────────

    def compile_feature_flags(self) -> dict[str, re.Pattern | str]:
        """
        Compile regex feature flags.
        Special values:
          __len_lt_15__   → sentence has fewer than 15 words
          __drug_registry__ → drug name detected by DrugRegistry
        """
        compiled = {}
        for name, pattern in self.feature_flags.items():
            if pattern.startswith("__"):
                compiled[name] = pattern  # special — handled in features.py
            else:
                try:
                    compiled[name] = re.compile(pattern, re.IGNORECASE)
                except re.error:
                    compiled[name] = pattern
        return compiled

    # ── Route prompt builder ───────────────────────────────────────────────────

    def build_route_system_prompt(self, route_name: str, principle_context: str = "") -> str:
        """
        Build a system prompt for a route using its RouteDefinition from the task config.
        This replaces hardcoded route system prompts in expert_routes.py.
        """
        route_def = next(
            (r for r in self.route_definitions if r.name == route_name), None
        )
        if not route_def:
            return f"You are a specialist. Vote {self.positive_label} or {self.negative_label}.\n\n" + _JSON_SCHEMA(self)

        return (
            f"You are a {route_name} specialist for {self.task_name}.\n"
            f"Task description: {self.description}\n\n"
            f"Your specific focus: {route_def.focus}\n\n"
            f"Labels: {self.positive_label} (positive) vs {self.negative_label} (negative)\n\n"
            + _JSON_SCHEMA(self)
            + (f"\n\n{principle_context}" if principle_context else "")
        )

    # ── MCQ error taxonomy ────────────────────────────────────────────────────

    def get_error_taxonomy(self) -> list[str]:
        """
        Returns domain-appropriate error type taxonomy.
        Used by MCQGenerator to categorize distractor types.
        Subclasses or config extensions can override.
        """
        return [
            "negation_confusion",         # missed or over-applied negation
            "therapeutic_goal_confusion", # confused therapeutic effect with ADE
            "causal_language_miss",       # missed causal attribution signal
            "drug_mention_overweight",    # drug mentioned but no harm
            "case_report_intro",          # intro sentence about ADE patterns, not a case
            "dosing_management",          # dose adjustment following prior ADE
            "mechanism_description",      # mechanism of action, not a patient event
            "epidemiology_statement",     # statistical association, not case report
            "other",
        ]

    # ── Seed tree builder ─────────────────────────────────────────────────────

    def build_seed_tree_dict(self) -> dict:
        """
        Returns the seed tree in the format expected by tree_v3.py / tree.py.
        ROOT node is always first; children are the rest.
        """
        if not self.seed_nodes:
            raise ValueError("TaskConfig has no seed_nodes defined.")
        root = next((n for n in self.seed_nodes if n.trigger is None), None)
        if not root:
            raise ValueError("TaskConfig seed_nodes must include a ROOT node with trigger=null.")
        children = [n for n in self.seed_nodes if n.trigger is not None]
        return {
            "id": root.id,
            "prompt": root.prompt,
            "trigger_condition": None,
            "children": [
                {
                    "id": c.id,
                    "prompt": c.prompt,
                    "trigger_condition": c.trigger,
                    "children": [],
                }
                for c in children
            ],
        }

    def __repr__(self) -> str:
        return (
            f"TaskConfig({self.task_name!r}, "
            f"labels={self.labels}, "
            f"routes={[r.name for r in self.route_definitions]}, "
            f"nodes={[n.id for n in self.seed_nodes]})"
        )


# ─── JSON schema helper ───────────────────────────────────────────────────────

def _JSON_SCHEMA(config: TaskConfig) -> str:
    return (
        f'Respond ONLY with JSON:\n'
        f'{{"vote": "{config.positive_label}" or "{config.negative_label}", '
        f'"confidence": 0.0-1.0, "reasoning": "<one sentence>"}}'
    )

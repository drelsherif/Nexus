"""
tree.py
The NEXUS decision tree: a nested dict of specialist prompt nodes.

ROOT has no trigger_condition (default fallback). Children are evaluated
in order; first match wins. classify_with_tree() returns the node that
actually fired so callers can log routing decisions.
"""

import json
from copy import deepcopy

from features import safe_eval_condition


def _new_stats():
    return {"n_correct": 0, "n_wrong": 0, "error_patterns": []}


def seed_tree() -> dict:
    """Round 0 tree: ROOT + 3 specialist children, per spec section 6."""
    root = {
        "id": "ROOT",
        "version": 1,
        "description": "General fallback for all unmatched cases",
        "trigger_condition": None,
        "prompt": (
            "You are a pharmacovigilance expert classifying clinical sentences for Adverse Drug Events (ADEs). "
            "An ADE requires DIRECT causal evidence: a specific drug caused a specific harmful or unintended outcome. "
            "CLASSIFY AS NOT_ADE when: (1) no drug is mentioned, (2) the effect is a desired therapeutic outcome, "
            "(3) the outcome is negated or hypothetical, (4) causation is absent or ambiguous, "
            "(5) the sentence is a general medical condition without drug linkage. "
            "CLASSIFY AS ADE only when a drug is explicitly linked to a harmful effect with clear causal language. "
            "When in doubt, classify NOT_ADE. "
            "Respond ONLY as JSON: "
            '{"classification": "ADE" or "NOT_ADE", "confidence": "high|medium|low", "rationale": "<one sentence>"}.'
        ),
        "stats": _new_stats(),
        "children": [
            {
                "id": "NODE_NEGATION",
                "version": 1,
                "description": "Explicit negation of an adverse outcome (e.g. 'no side effects', 'did not develop')",
                "trigger_condition": "has_negation and not has_induced",
                "prompt": (
                    "You are a pharmacovigilance expert. This sentence contains a negation word "
                    "(no/not/without/denied). Most such sentences explicitly state the ABSENCE of an "
                    "adverse event and should be classified NOT_ADE -- but watch for negations that "
                    "modify something other than the adverse outcome itself (e.g. 'did not stop the rash'). "
                    'Respond ONLY as JSON: {"classification": "ADE" or "NOT_ADE", "confidence": "high|medium|low", "rationale": "<one sentence>"}.'
                ),
                "stats": _new_stats(),
                "children": [],
            },
            {
                "id": "NODE_INDUCED",
                "version": 1,
                "description": "Drug-X-induced / drug-X-associated pattern -- almost always ADE",
                "trigger_condition": "has_induced or has_associated",
                "prompt": (
                    "You are a pharmacovigilance expert. This sentence uses a 'drug-induced' or "
                    "'drug-associated' construction (e.g. 'cisplatin-induced nephrotoxicity'), which "
                    "almost always denotes a confirmed Adverse Drug Event. Classify ADE unless the "
                    "sentence explicitly negates the outcome or describes a desired therapeutic effect. "
                    'Respond ONLY as JSON: {"classification": "ADE" or "NOT_ADE", "confidence": "high|medium|low", "rationale": "<one sentence>"}.'
                ),
                "stats": _new_stats(),
                "children": [],
            },
            {
                "id": "NODE_SHORT",
                "version": 1,
                "description": "Short titles/headers (e.g. 'Vancomycin toxicity')",
                "trigger_condition": "has_short and not has_induced",
                "prompt": (
                    "You are a pharmacovigilance expert. This is a short title or header-style fragment "
                    "rather than a full clinical sentence. Such fragments (e.g. 'Vancomycin toxicity', "
                    "'Lithium-induced tremor case report') typically name a drug + an adverse effect "
                    "directly. Classify ADE if a harmful effect is named in connection with a drug; "
                    "otherwise NOT_ADE. "
                    'Respond ONLY as JSON: {"classification": "ADE" or "NOT_ADE", "confidence": "high|medium|low", "rationale": "<one sentence>"}.'
                ),
                "stats": _new_stats(),
                "children": [],
            },
        ],
    }
    return root


def classify_with_tree(tree: dict, feats: dict):
    """
    Route feats through tree.children in order (first match wins),
    falling back to the root node (tree itself) if nothing matches.

    Returns the node dict that should handle this case (caller makes the
    actual LLM call with node['prompt']).
    """
    for child in tree.get("children", []):
        cond = child.get("trigger_condition")
        if cond and safe_eval_condition(cond, feats):
            return child
    return tree


def insert_graft(tree: dict, new_node: dict) -> dict:
    """
    Insert a new specialist node at the TOP of tree['children'] (spec
    section 7, step 7: "insert node at top of children list"). Returns a
    new tree (deep copy) so the caller can still discard the candidate if
    probing rejects it.
    """
    candidate = deepcopy(tree)
    node = {
        "id": new_node["id"],
        "version": 1,
        "description": new_node.get("description", ""),
        "trigger_condition": new_node["trigger_condition"],
        "prompt": new_node["prompt"],
        "stats": _new_stats(),
        "children": [],
    }
    candidate["children"].insert(0, node)
    return candidate


def all_node_ids(tree: dict) -> list:
    ids = [tree["id"]]
    for c in tree.get("children", []):
        ids.extend(all_node_ids(c))
    return ids


def node_summaries(tree: dict) -> list:
    """[{id, description}, ...] for every node -- used in the synthesis prompt."""
    out = [{"id": tree["id"], "description": tree.get("description", "")}]
    for c in tree.get("children", []):
        out.append({"id": c["id"], "description": c.get("description", "")})
    return out


def save_tree(tree: dict, path: str):
    with open(path, "w") as f:
        json.dump(tree, f, indent=2)


def load_tree(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

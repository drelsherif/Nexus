"""
tree_v3.py
NEXUS v3 — Self-Growing Decision Tree

The tree is the architecture. RAG, engrams, MCQs, and parallel routes
are capabilities of each tree node — not replacements for the tree.

Growth is memory-driven:
  SWR events (engram cluster fires) → principle + new child node proposal
  This means topology emerges from error patterns in the data, not design.

Routing is interpretable:
  Every classification has an auditable path through named specialist nodes.
  Feature flags provide cheap deterministic routing before any LLM call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from node import NexusNode, NodeResult, SWRResult
from rag_index import RAGIndex
from features import features as extract_features


# ─── Probe result ─────────────────────────────────────────────────────────────

class NexusTree:
    """
    NEXUS v3 decision tree.

    Maintains a tree of NexusNodes. Routes cases to the first matching
    child node (by trigger condition on feature flags); falls back to ROOT.

    Growth:
      graft(parent, child)  — add a new specialist child node
      retire(node_id)        — remove a chronically underperforming node

    Serialization:
      save(path) / load(path, task_config) — full tree state
    """

    def __init__(self, root: NexusNode, path: Optional[str] = None):
        self.root = root
        self.path = Path(path) if path else None
        self._node_index: dict[str, NexusNode] = {}
        self._rebuild_index()

    # ── Index ─────────────────────────────────────────────────────────────────

    def _rebuild_index(self):
        self._node_index = {}
        self._index_node(self.root)

    def _index_node(self, node: NexusNode):
        self._node_index[node.id] = node
        for child in node.children:
            self._index_node(child)

    def get_node(self, node_id: str) -> Optional[NexusNode]:
        return self._node_index.get(node_id)

    def all_nodes(self) -> list[NexusNode]:
        return list(self._node_index.values())

    def all_child_nodes(self) -> list[NexusNode]:
        return [n for n in self._node_index.values() if n.id != self.root.id]

    # ── Routing ───────────────────────────────────────────────────────────────

    def route(self, text: str) -> NexusNode:
        """
        Route a sentence to the appropriate specialist node.
        Returns the first child node whose trigger matches; else ROOT.
        """
        feats = extract_features(text)
        for child in self.root.children:
            if child.matches_trigger(feats):
                return child
        return self.root

    def classify(
        self,
        text: str,
        route_llm_fn: Callable,
        global_rag_index: Optional[RAGIndex] = None,
        workers: int = 4,
    ) -> NodeResult:
        """Route + classify. Full v3 pipeline."""
        node = self.route(text)
        return node.classify(
            text=text,
            route_llm_fn=route_llm_fn,
            global_rag_index=global_rag_index,
            workers=workers,
        )

    # ── Growth ────────────────────────────────────────────────────────────────

    def graft(self, parent_id: str, child_node: NexusNode) -> bool:
        """
        Graft a new child node onto a parent.
        Returns True if successful.
        """
        parent = self.get_node(parent_id)
        if not parent:
            return False
        # Dedup: don't add a node with an existing ID
        if child_node.id in self._node_index:
            # Rename with suffix
            child_node.id = f"{child_node.id}_v{len(self._node_index)}"
        parent.children.append(child_node)
        self._rebuild_index()
        return True

    def retire(self, node_id: str) -> bool:
        """
        Remove a chronically underperforming node.
        Cannot retire ROOT.
        """
        if node_id == self.root.id:
            return False

        def _remove(parent: NexusNode, target_id: str) -> bool:
            for child in parent.children:
                if child.id == target_id:
                    # Reconnect grandchildren to parent
                    parent.children.remove(child)
                    parent.children.extend(child.children)
                    return True
                if _remove(child, target_id):
                    return True
            return False

        result = _remove(self.root, node_id)
        if result:
            self._rebuild_index()
        return result

    # ── Probe ─────────────────────────────────────────────────────────────────

    def probe_graft(
        self,
        parent_id: str,
        child_proposal: dict,
        probe_cases: list[dict],
        route_llm_fn: Callable,
        global_rag_index: Optional[RAGIndex],
        task_config,
        workers: int = 4,
    ) -> tuple[float, Optional[NexusNode]]:
        """
        Test whether grafting a proposed child node improves F1 on probe set.
        Returns (delta_f1, child_node_if_accepted).

        Does NOT modify the tree — caller decides whether to graft.
        """
        # Baseline F1 without child
        baseline = self._eval_probe(probe_cases, route_llm_fn, global_rag_index, task_config, workers)

        # Build candidate node
        child = NexusNode(
            node_id=child_proposal.get("id", f"NODE_AUTO_{len(self._node_index)}"),
            prompt=child_proposal.get("prompt", ""),
            trigger_condition=child_proposal.get("trigger"),
            task_config=task_config,
            path=str(self.path) if self.path else None,
        )
        # Seed child's RAG from parent's engram cluster if available
        # (handled in nexus_v3.py where we have access to cluster errors)

        # Temporarily graft
        parent = self.get_node(parent_id)
        if not parent:
            return 0.0, None

        parent.children.insert(0, child)  # insert at front — try to route to child first
        self._rebuild_index()

        # Candidate F1
        candidate = self._eval_probe(probe_cases, route_llm_fn, global_rag_index, task_config, workers)

        # Undo temporary graft
        parent.children.remove(child)
        self._rebuild_index()

        delta = candidate["f1"] - baseline["f1"]
        return delta, child

    def probe_retire(
        self,
        node_id: str,
        probe_cases: list[dict],
        route_llm_fn: Callable,
        global_rag_index: Optional[RAGIndex],
        task_config,
        workers: int = 4,
    ) -> float:
        """
        Test whether retiring a node hurts F1 on probe set.
        Returns delta_f1 (negative = retirement hurts).
        """
        baseline = self._eval_probe(probe_cases, route_llm_fn, global_rag_index, task_config, workers)

        # Temporarily retire
        if not self.retire(node_id):
            return 0.0

        candidate = self._eval_probe(probe_cases, route_llm_fn, global_rag_index, task_config, workers)

        # Undo: can't easily undo retire without re-building, so re-load
        # For now, return delta and let caller handle the undo by not saving
        delta = candidate["f1"] - baseline["f1"]

        # Re-add the node (caller should use save/load pattern for safety)
        return delta

    def _eval_probe(
        self,
        probe_cases: list[dict],
        route_llm_fn: Callable,
        global_rag_index: Optional[RAGIndex],
        task_config,
        workers: int,
    ) -> dict:
        pos = task_config.positive_label
        tp = fp = fn = tn = 0
        for c in probe_cases:
            result = self.classify(c["text"], route_llm_fn, global_rag_index, workers)
            pred = result.label
            true = c["label"]
            if pred == pos and true == pos:     tp += 1
            elif pred == pos and true != pos:   fp += 1
            elif pred != pos and true == pos:   fn += 1
            else:                               tn += 1
        prec   = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1     = 2 * prec * recall / max(1e-9, prec + recall)
        return {"f1": round(f1, 4), "precision": round(prec, 4), "recall": round(recall, 4)}

    # ── Reporting ─────────────────────────────────────────────────────────────

    def print_tree(self, indent: int = 0) -> None:
        def _print(node: NexusNode, depth: int):
            prefix = "  " * depth + ("└─ " if depth > 0 else "")
            route_avg = node.average_routes_per_round()
            mcqs = len(node.mcq_library)
            principles = len(node.injected_principles)
            clusters = len(node.engram_store._clusters)
            print(f"{prefix}{node.id}  "
                  f"[trigger: {node.trigger_condition or 'ROOT'}]  "
                  f"routes/rnd={route_avg:.1f}  "
                  f"mcqs={mcqs}  principles={principles}  clusters={clusters}")
            for child in node.children:
                _print(child, depth + 1)
        _print(self.root, indent)

    def summary(self) -> dict:
        nodes = self.all_nodes()
        return {
            "total_nodes": len(nodes),
            "child_nodes": len(nodes) - 1,
            "nodes": [
                {
                    "id": n.id,
                    "trigger": n.trigger_condition,
                    "mcqs": len(n.mcq_library),
                    "principles": len(n.injected_principles),
                    "clusters": len(n.engram_store._clusters),
                    "routes_per_round": round(n.average_routes_per_round(), 1),
                }
                for n in nodes
            ],
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        save_path = Path(path) if path else self.path
        if not save_path:
            return
        save_path.mkdir(parents=True, exist_ok=True)

        # Save tree structure (node IDs + relationships)
        structure = self._serialize_structure(self.root)
        (save_path / "tree_structure.json").write_text(
            json.dumps(structure, indent=2)
        )

        # Save each node's state
        for node in self.all_nodes():
            node.path = save_path
            node.save()

    def _serialize_structure(self, node: NexusNode) -> dict:
        return {
            "id": node.id,
            "trigger_condition": node.trigger_condition,
            "children": [self._serialize_structure(c) for c in node.children],
        }

    @classmethod
    def load(cls, path: str, task_config) -> "NexusTree":
        p = Path(path)
        structure_file = p / "tree_structure.json"
        if not structure_file.exists():
            raise FileNotFoundError(f"Tree structure not found at {path}")

        structure = json.loads(structure_file.read_text())

        def _load_node(s: dict) -> NexusNode:
            node = NexusNode.load(s["id"], str(p), task_config)
            node.children = [_load_node(c) for c in s.get("children", [])]
            return node

        root = _load_node(structure)
        return cls(root=root, path=path)

    @classmethod
    def from_task_config(cls, task_config, path: Optional[str] = None, db=None) -> "NexusTree":
        """Build a fresh seed tree from the task config's seed_nodes."""
        seed_nodes = task_config.seed_nodes
        if not seed_nodes:
            raise ValueError("TaskConfig has no seed_nodes.")

        root_def = next((n for n in seed_nodes if n.trigger is None), None)
        if not root_def:
            raise ValueError("TaskConfig must have a ROOT node with trigger=null.")

        root = NexusNode(
            node_id=root_def.id,
            prompt=root_def.prompt,
            trigger_condition=None,
            task_config=task_config,
            path=path,
            db=db,
        )

        for sn in seed_nodes:
            if sn.trigger is None:
                continue
            child = NexusNode(
                node_id=sn.id,
                prompt=sn.prompt,
                trigger_condition=sn.trigger,
                task_config=task_config,
                path=path,
                db=db,
            )
            root.children.append(child)

        tree = cls(root=root, path=path)
        return tree

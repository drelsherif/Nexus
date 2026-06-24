"""
semantic_engram.py
Cluster-based engram formation — replaces string-matching RuleDictionary.

The core problem with RuleDictionary:
  Same error pattern described differently each time
  → never accumulates → SWR never fires → no learning

This module fixes it by working in embedding space:
  Two errors are "the same pattern" if their sentence embeddings are close,
  regardless of how the LLM described them.

Biological analogy:
  ErrorCluster     = Sparse population code in DG (Dentate Gyrus)
                     Similar inputs activate overlapping cell assemblies.
  Cluster threshold = LTP threshold — enough co-activation → engram
  LLM summarization = Cortical consolidation during SWR replay
                      The hippocampal episode is compressed into a
                      semantic principle for neocortical storage.
  Engram embedding  = Pattern completion cue — a new error near an
                      existing engram retrieves the principle automatically.

Usage:
    from semantic_engram import SemanticEngramStore
    store = SemanticEngramStore(threshold=5, path="run/engrams")

    fired = store.add_error(text="...", true_label="ADE", predicted="NOT_ADE")
    if fired:
        cluster_id, examples = fired
        principle = store.consolidate(cluster_id, llm_fn)
        # principle is now injected into the node that made the error
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from embedder import embed_one, embed


@dataclass
class ErrorRecord:
    text: str
    true_label: str
    predicted_label: str
    node_id: str
    round_num: int
    embedding: list[float] = field(default_factory=list)  # stored as list for JSON


@dataclass
class ErrorCluster:
    cluster_id: str
    centroid: list[float]           # running mean embedding
    errors: list[ErrorRecord] = field(default_factory=list)
    promoted: bool = False
    principle_text: str = ""
    principle_round: int = -1
    dominant_label: str = ""        # what the correct label should be

    @property
    def size(self) -> int:
        return len(self.errors)

    def update_centroid(self):
        """Recompute centroid from all member error embeddings."""
        if not self.errors:
            return
        vecs = np.array([e.embedding for e in self.errors], dtype=np.float32)
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        self.centroid = centroid.tolist()

    def dominant_true_label(self) -> str:
        labels = [e.true_label for e in self.errors]
        return "ADE" if labels.count("ADE") >= labels.count("NOT_ADE") else "NOT_ADE"


class SemanticEngramStore:
    """
    Maintains a set of error clusters in embedding space.
    New errors are assigned to the nearest cluster (if close enough)
    or start a new cluster.
    When a cluster reaches `threshold` size, it fires (SWR event).
    """

    def __init__(
        self,
        threshold: int = 5,
        similarity_threshold: float = 0.80,   # min cosine sim to join cluster
        path: str | None = None,
    ):
        self.threshold = threshold
        self.sim_threshold = similarity_threshold
        self.path = Path(path) if path else None
        self._clusters: dict[str, ErrorCluster] = {}
        self._cluster_counter = 0
        self._fired_this_round: list[str] = []
        self._engrams: list[dict] = []   # promoted principles with embeddings

    # ── Add error ─────────────────────────────────────────────────────────────

    def add_error(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        node_id: str,
        round_num: int,
    ) -> tuple[str, list[ErrorRecord]] | None:
        """
        Embed the error sentence and assign to nearest cluster.
        Returns (cluster_id, cluster.errors) if this addition crosses threshold,
        else None.
        """
        vec = embed_one(text)
        record = ErrorRecord(
            text=text,
            true_label=true_label,
            predicted_label=predicted_label,
            node_id=node_id,
            round_num=round_num,
            embedding=vec.tolist(),
        )

        # Find nearest unpromoted cluster
        best_cluster_id = None
        best_sim = -1.0
        vec_arr = np.array(vec, dtype=np.float32)

        for cid, cluster in self._clusters.items():
            if cluster.promoted:
                continue
            centroid = np.array(cluster.centroid, dtype=np.float32)
            sim = float(np.dot(vec_arr, centroid))
            if sim > best_sim:
                best_sim = sim
                best_cluster_id = cid

        # Assign to cluster or create new
        if best_cluster_id and best_sim >= self.sim_threshold:
            cluster = self._clusters[best_cluster_id]
        else:
            # New cluster
            self._cluster_counter += 1
            cid = f"C{self._cluster_counter:04d}"
            cluster = ErrorCluster(
                cluster_id=cid,
                centroid=vec.tolist(),
            )
            self._clusters[cid] = cluster
            best_cluster_id = cid

        cluster.errors.append(record)
        cluster.update_centroid()
        cluster.dominant_label = cluster.dominant_true_label()

        prev_size = cluster.size - 1
        if prev_size < self.threshold <= cluster.size:
            self._fired_this_round.append(best_cluster_id)
            return best_cluster_id, list(cluster.errors)

        return None

    # ── SWR consolidation ─────────────────────────────────────────────────────

    def pop_fired(self) -> list[str]:
        fired = list(self._fired_this_round)
        self._fired_this_round.clear()
        return fired

    def build_consolidation_prompt(self, cluster_id: str) -> str | None:
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return None

        examples = "\n".join(
            f"  [{i+1}] (True={e.true_label}, Predicted={e.predicted_label})\n"
            f"       \"{e.text[:200]}\""
            for i, e in enumerate(cluster.errors[:6])
        )
        return f"""You are NEXUS, a clinical pharmacovigilance classifier.

The following {cluster.size} sentences were ALL misclassified by the same node.
They were semantically clustered — they share a common underlying pattern.
The correct label for most of them is: {cluster.dominant_label}

Misclassified sentences:
{examples}

Task: Identify the SINGLE shared pattern that caused these misclassifications
and write a concise, actionable NEXUS principle (3-5 sentences) that:
1. States the pattern precisely
2. Explains why it signals {cluster.dominant_label}
3. Distinguishes it from superficially similar NOT-{cluster.dominant_label} cases
4. Provides a decision rule for future cases

Begin with "NEXUS PRINCIPLE:" and write in present tense."""

    def consolidate(
        self,
        cluster_id: str,
        llm_fn: Callable[[str, str], str],
        round_num: int = -1,
    ) -> str | None:
        """LLM consolidation: cluster → principle text."""
        prompt = self.build_consolidation_prompt(cluster_id)
        if not prompt:
            return None
        try:
            principle = llm_fn(
                "You are NEXUS, a self-improving clinical ADE classifier. "
                "Write principles that are precise, clinical, and actionable.",
                prompt,
            ).strip()
            cluster = self._clusters[cluster_id]
            cluster.promoted = True
            cluster.principle_text = principle
            cluster.principle_round = round_num

            # Store engram with embedding for future RAG retrieval
            centroid_vec = np.array(cluster.centroid, dtype=np.float32)
            self._engrams.append({
                "cluster_id": cluster_id,
                "principle": principle,
                "dominant_label": cluster.dominant_label,
                "cluster_size": cluster.size,
                "embedding": cluster.centroid,
                "round": round_num,
            })

            return principle
        except Exception as e:
            return None

    def retrieve_principles(self, text: str, top_k: int = 3) -> list[dict]:
        """
        Given a new sentence, retrieve the most relevant engram principles.
        Used to inject relevant principles into node prompts at query time.
        """
        if not self._engrams:
            return []
        vec = embed_one(text)
        scores = []
        for eng in self._engrams:
            centroid = np.array(eng["embedding"], dtype=np.float32)
            sim = float(np.dot(vec, centroid))
            scores.append((sim, eng))
        scores.sort(key=lambda x: -x[0])
        return [eng for _, eng in scores[:top_k] if _ > 0.6]

    # ── Reporting ─────────────────────────────────────────────────────────────

    def summary_stats(self) -> dict:
        total = len(self._clusters)
        promoted = sum(1 for c in self._clusters.values() if c.promoted)
        near = sum(
            1 for c in self._clusters.values()
            if not c.promoted and c.size >= self.threshold - 1
        )
        top = sorted(
            [c for c in self._clusters.values() if not c.promoted],
            key=lambda c: -c.size
        )[:5]
        return {
            "total_clusters": total,
            "promoted": promoted,
            "near_threshold": near,
            "threshold": self.threshold,
            "engrams": len(self._engrams),
            "top_clusters": [
                {"id": c.cluster_id, "size": c.size, "label": c.dominant_label,
                 "sample": c.errors[0].text[:60] if c.errors else ""}
                for c in top
            ],
        }

    def print_report(self):
        stats = self.summary_stats()
        print(
            f"\n[Engrams] {stats['total_clusters']} clusters "
            f"| {stats['promoted']} promoted "
            f"| {stats['near_threshold']} near threshold ({self.threshold})"
            f"| {stats['engrams']} principles"
        )
        if stats["top_clusters"]:
            print("[Engrams] Largest clusters:")
            for c in stats["top_clusters"]:
                bar = "█" * c["size"] + "░" * max(0, self.threshold - c["size"])
                print(f"  [{bar}] x{c['size']} {c['label']:7s} \"{c['sample']}\"")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self):
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        data = {
            "threshold": self.threshold,
            "sim_threshold": self.sim_threshold,
            "cluster_counter": self._cluster_counter,
            "clusters": {
                cid: {
                    "cluster_id": c.cluster_id,
                    "centroid": c.centroid,
                    "promoted": c.promoted,
                    "principle_text": c.principle_text,
                    "principle_round": c.principle_round,
                    "dominant_label": c.dominant_label,
                    "errors": [
                        {
                            "text": e.text, "true_label": e.true_label,
                            "predicted_label": e.predicted_label,
                            "node_id": e.node_id, "round_num": e.round_num,
                            "embedding": e.embedding,
                        }
                        for e in c.errors
                    ],
                }
                for cid, c in self._clusters.items()
            },
            "engrams": self._engrams,
        }
        (self.path / "semantic_engrams.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "SemanticEngramStore":
        p = Path(path)
        f = p / "semantic_engrams.json"
        if not f.exists():
            return cls(path=path)
        data = json.loads(f.read_text())
        store = cls(
            threshold=data.get("threshold", 5),
            similarity_threshold=data.get("sim_threshold", 0.80),
            path=path,
        )
        store._cluster_counter = data.get("cluster_counter", 0)
        store._engrams = data.get("engrams", [])
        for cid, cd in data.get("clusters", {}).items():
            cluster = ErrorCluster(
                cluster_id=cd["cluster_id"],
                centroid=cd["centroid"],
                promoted=cd["promoted"],
                principle_text=cd.get("principle_text", ""),
                principle_round=cd.get("principle_round", -1),
                dominant_label=cd.get("dominant_label", ""),
            )
            for ed in cd.get("errors", []):
                cluster.errors.append(ErrorRecord(**ed))
            store._clusters[cid] = cluster
        return store

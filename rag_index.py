"""
rag_index.py
FAISS vector index over the ADE Corpus for NEXUS retrieval-augmented classification.

Biological analogy:
  RAGIndex  =  Hippocampal CA3 pattern completion
               A partial or noisy input (new clinical sentence) retrieves
               the closest stored memories (labeled training examples).
               This is how the brain finishes a partial pattern from
               fragmentary cues — without needing exact string match.

Build once (~2-3 min for 23K sentences), query in milliseconds.

Install:
    pip3 install faiss-cpu sentence-transformers --break-system-packages

Usage:
    from rag_index import RAGIndex
    idx = RAGIndex.build(corpus, out_dir="run_07/rag")   # one-time
    idx = RAGIndex.load("run_07/rag")                    # subsequent runs

    results = idx.query("Patient developed rash after amoxicillin.", k=5)
    # [{"text": ..., "label": "ADE", "score": 0.92}, ...]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from embedder import embed, embed_one, model_dim


class RAGIndex:
    """
    FAISS flat inner-product index (= cosine similarity on unit vectors).
    Stores all training sentences + labels. Supports instant k-NN retrieval.
    """

    def __init__(self):
        self._index = None          # faiss.IndexFlatIP
        self._texts: list[str] = []
        self._labels: list[str] = []
        self._dim: int = 0

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        corpus: list[dict],
        out_dir: str | None = None,
        batch_size: int = 128,
        model_name: str | None = None,
    ) -> "RAGIndex":
        """
        Embed all corpus sentences and build the FAISS index.
        corpus: list of {"text": str, "label": "ADE"|"NOT_ADE"}
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed.\n"
                "Run: pip3 install faiss-cpu --break-system-packages"
            )

        idx = cls()
        texts  = [c["text"]  for c in corpus]
        labels = [c["label"] for c in corpus]

        print(f"[RAGIndex] Embedding {len(texts)} sentences...", flush=True)
        t0 = time.time()
        vecs = embed(texts, batch_size=batch_size, model_name=model_name)
        elapsed = time.time() - t0
        print(f"[RAGIndex] Embedded in {elapsed:.1f}s  dim={vecs.shape[1]}", flush=True)

        idx._dim = vecs.shape[1]
        idx._texts = texts
        idx._labels = labels

        # Inner-product on unit vectors = cosine similarity
        idx._index = faiss.IndexFlatIP(idx._dim)
        idx._index.add(vecs)
        print(f"[RAGIndex] FAISS index built — {idx._index.ntotal} vectors", flush=True)

        if out_dir:
            idx.save(out_dir)

        return idx

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, out_dir: str):
        import faiss
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(p / "faiss.index"))
        (p / "corpus.json").write_text(
            json.dumps({"texts": self._texts, "labels": self._labels})
        )
        (p / "meta.json").write_text(json.dumps({"dim": self._dim}))
        print(f"[RAGIndex] Saved to {out_dir}/", flush=True)

    @classmethod
    def load(cls, out_dir: str) -> "RAGIndex":
        import faiss
        p = Path(out_dir)
        idx = cls()
        idx._index = faiss.read_index(str(p / "faiss.index"))
        corpus_data = json.loads((p / "corpus.json").read_text())
        idx._texts  = corpus_data["texts"]
        idx._labels = corpus_data["labels"]
        idx._dim    = json.loads((p / "meta.json").read_text())["dim"]
        print(f"[RAGIndex] Loaded — {idx._index.ntotal} vectors ({idx._dim}-dim)", flush=True)
        return idx

    @classmethod
    def load_or_build(
        cls,
        corpus: list[dict],
        out_dir: str,
        model_name: str | None = None,
    ) -> "RAGIndex":
        """Load from disk if exists, otherwise build and save."""
        p = Path(out_dir)
        if (p / "faiss.index").exists() and (p / "corpus.json").exists():
            print(f"[RAGIndex] Found existing index at {out_dir}, loading...", flush=True)
            return cls.load(out_dir)
        print(f"[RAGIndex] Building new index at {out_dir}...", flush=True)
        return cls.build(corpus, out_dir=out_dir, model_name=model_name)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, text: str, k: int = 5) -> list[dict]:
        """
        Retrieve k most similar labeled examples.
        Returns list of {"text": str, "label": str, "score": float}.
        Thread-safe (FAISS flat index is read-only at query time).
        """
        vec = embed_one(text).reshape(1, -1)
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text":  self._texts[idx],
                "label": self._labels[idx],
                "score": float(score),
            })
        return results

    def query_batch(self, texts: list[str], k: int = 5) -> list[list[dict]]:
        """Batch query for multiple texts at once."""
        vecs = embed(texts).reshape(len(texts), -1)
        scores_batch, indices_batch = self._index.search(vecs, k)
        out = []
        for scores, indices in zip(scores_batch, indices_batch):
            results = []
            for score, idx in zip(scores, indices):
                if idx < 0:
                    continue
                results.append({
                    "text":  self._texts[idx],
                    "label": self._labels[idx],
                    "score": float(score),
                })
            out.append(results)
        return out

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        ade_count = sum(1 for l in self._labels if l == "ADE")
        return {
            "total":     len(self._texts),
            "ade":       ade_count,
            "not_ade":   len(self._texts) - ade_count,
            "dim":       self._dim,
            "ade_ratio": round(ade_count / max(1, len(self._texts)), 3),
        }

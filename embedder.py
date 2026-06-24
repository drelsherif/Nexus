"""
embedder.py
Biomedical sentence encoder for NEXUS RAG system.

Biological analogy:
  Embedder  =  Entorhinal Cortex (EC)
               Converts raw sensory input (text) into a compressed
               high-dimensional pattern before hippocampal indexing.

  Embedding space  =  the "grid cell" coordinate system — similar
                       clinical meanings land near each other regardless
                       of surface wording.

Model preference order (best biomedical → fastest general):
  1. pritamdeka/S-PubMedBert-MS-MARCO  (biomedical, 768-dim)
  2. NLP4Science/pubmedbert-base-embeddings (lighter pubmed)
  3. all-MiniLM-L6-v2                  (fast general, 384-dim)

Install:
    pip3 install sentence-transformers --break-system-packages
"""

from __future__ import annotations

import numpy as np

_MODEL_PREFERENCE = [
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "NLP4Science/pubmedbert-base-embeddings",
    "all-MiniLM-L6-v2",
]

_model = None       # lazy singleton
_model_name = None


def _load_model(preferred: str | None = None):
    global _model, _model_name
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed.\n"
            "Run: pip3 install sentence-transformers --break-system-packages"
        )

    candidates = ([preferred] if preferred else []) + _MODEL_PREFERENCE
    for name in candidates:
        try:
            print(f"[Embedder] Loading model: {name} ...", flush=True)
            _model = SentenceTransformer(name)
            _model_name = name
            dim = _model.get_sentence_embedding_dimension()
            print(f"[Embedder] Ready — {name} ({dim}-dim)", flush=True)
            return _model
        except Exception as e:
            print(f"[Embedder] {name} unavailable ({e}), trying next...", flush=True)

    raise RuntimeError("No embedding model could be loaded.")


def embed(texts: list[str], batch_size: int = 64, model_name: str | None = None) -> np.ndarray:
    """
    Embed a list of strings.
    Returns float32 ndarray of shape (N, dim), L2-normalised.
    """
    model = _load_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 200,
        normalize_embeddings=True,   # unit vectors → dot product = cosine sim
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_one(text: str, model_name: str | None = None) -> np.ndarray:
    """Embed a single string. Returns shape (dim,)."""
    return embed([text], model_name=model_name)[0]


def model_dim(model_name: str | None = None) -> int:
    """Return embedding dimension without encoding anything."""
    model = _load_model(model_name)
    return model.get_sentence_embedding_dimension()


def current_model_name() -> str | None:
    return _model_name

"""
data_utils.py
Loads ADE Corpus v2 (Gurulingappa et al.) and splits into eval/probe/train
pools per spec section 12.

Local cache: data/ade_corpus.jsonl (created by save_dataset_locally.py).
Falls back to HuggingFace download if local file is absent.
Run `python3 save_dataset_locally.py` once to avoid network dependency.
"""

import json
import random
from pathlib import Path

LABEL_MAP = {1: "ADE", 0: "NOT_ADE"}
_LOCAL_DATA = Path(__file__).parent / "data" / "ade_corpus.jsonl"


def _load_items() -> list[dict]:
    """Load corpus rows from local JSONL if present, else HuggingFace."""
    if _LOCAL_DATA.exists():
        items = []
        with _LOCAL_DATA.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        print(f"[V3] Loaded {len(items):,} cases from local cache ({_LOCAL_DATA.name})")
        return items

    print("[V3] Local dataset not found — downloading from HuggingFace...")
    print("[V3] Tip: run `python3 save_dataset_locally.py` once to avoid this.")
    from datasets import load_dataset
    ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification")
    split = ds["train"]
    items = [{"text": row["text"], "label": LABEL_MAP[row["label"]]} for row in split]
    print(f"[V3] Loaded {len(items):,} cases from HuggingFace")
    return items


def load_and_split(seed: int = 42, eval_size: int = 200, probe_size: int = 300):
    """
    Returns (eval_pool, probe_pool, train_pool), each a list of
    {"text": str, "label": "ADE"|"NOT_ADE"} dicts.

    Per spec section 12:
      - First 500 cases (after seeded shuffle) = locked evaluation pool
          - first `eval_size` (200) -> fixed eval set
          - remaining (300) -> probe set for graft validation
      - Remaining ~17,100 -> training pool
    """
    items = _load_items()

    rng = random.Random(seed)
    rng.shuffle(items)

    # Assign stable corpus index (position after seeded shuffle) to each case.
    # This lets us track exactly which cases were used in training vs held out.
    for i, item in enumerate(items):
        item["_corpus_idx"] = i

    locked = items[:eval_size + probe_size]
    eval_pool = locked[:eval_size]
    probe_pool = locked[eval_size:eval_size + probe_size]
    train_pool = items[eval_size + probe_size:]

    return eval_pool, probe_pool, train_pool


def sample_batch(train_pool, batch_size, rng):
    """Sample `batch_size` cases from train_pool using the given random.Random."""
    return rng.sample(train_pool, min(batch_size, len(train_pool)))

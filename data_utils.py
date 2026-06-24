"""
data_utils.py
Loads ADE Corpus v2 (Gurulingappa et al.) and splits into eval/probe/train
pools per spec section 12.

NOTE: requires network access to huggingface.co. If that's blocked in your
environment, set HF_ENDPOINT or download the dataset manually first --
see README.md.
"""

import random

LABEL_MAP = {1: "ADE", 0: "NOT_ADE"}


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
    from datasets import load_dataset

    ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification")
    split = ds["train"]  # this config only ships a single split

    items = [{"text": row["text"], "label": LABEL_MAP[row["label"]]} for row in split]

    rng = random.Random(seed)
    rng.shuffle(items)

    locked = items[:eval_size + probe_size]
    eval_pool = locked[:eval_size]
    probe_pool = locked[eval_size:eval_size + probe_size]
    train_pool = items[eval_size + probe_size:]

    return eval_pool, probe_pool, train_pool


def sample_batch(train_pool, batch_size, rng):
    """Sample `batch_size` cases from train_pool using the given random.Random."""
    return rng.sample(train_pool, min(batch_size, len(train_pool)))

#!/usr/bin/env python3
"""
save_dataset_locally.py — One-time ADE corpus export

Downloads (or uses cached) ade-benchmark-corpus/ade_corpus_v2 from HuggingFace
and saves it as data/ade_corpus.jsonl in this folder.

After running this once, data_utils.py will load from the local file
(no network needed, ~2x faster startup).

Usage:
    python3 save_dataset_locally.py
"""

import json
import os
from pathlib import Path

LOCAL_PATH = Path(__file__).parent / "data" / "ade_corpus.jsonl"
LABEL_MAP = {1: "ADE", 0: "NOT_ADE"}


def main():
    if LOCAL_PATH.exists():
        n = sum(1 for _ in LOCAL_PATH.open())
        print(f"✓ Local dataset already exists: {LOCAL_PATH} ({n:,} rows)")
        print("  Delete it and re-run to refresh from HuggingFace.")
        return

    print("Downloading ade-benchmark-corpus/ade_corpus_v2 from HuggingFace...")
    print("(This may take 30-60 seconds on first run; subsequent runs use the cache.)")
    from datasets import load_dataset

    ds = load_dataset("ade-benchmark-corpus/ade_corpus_v2", "Ade_corpus_v2_classification")
    split = ds["train"]

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with LOCAL_PATH.open("w") as f:
        for row in split:
            record = {"text": row["text"], "label": LABEL_MAP[row["label"]]}
            f.write(json.dumps(record) + "\n")

    n = sum(1 for _ in LOCAL_PATH.open())
    print(f"✓ Saved {n:,} rows to {LOCAL_PATH}")
    print("  data_utils.py will now load from this local file (no network needed).")


if __name__ == "__main__":
    main()

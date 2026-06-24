# NEXUS — Self-Growing Decision Tree for Clinical NLP

**NEXUS** is a continuously learning, interpretable AI classification framework for medical text. It starts from a small set of seed nodes and grows its own decision tree through experience — no GPU required, no retraining, full auditability.

Designed and built by **Yasir El-Sherif, MD** — Northwell Health.

---

## How it works

NEXUS routes each sentence through a self-growing decision tree. Each node is a specialist classifier with three types of memory:

- **RAG (case examples):** retrieves similar past sentences at inference time
- **MCQs (hard case flashcards):** remembers mistakes and reviews them when similar cases recur
- **Principles:** consolidates patterns from repeated errors into durable written rules

When a cluster of similar errors grows large enough, NEXUS fires a **Sharp-Wave Ripple (SWR) event** — a consolidation step borrowed from neuroscience — and either writes a new principle or grafts a new specialist branch onto the tree. Dead branches are retired automatically.

The result: a living classifier that improves with every round of cases, explains its reasoning, and adapts without retraining.

See [`NEXUS_plain_language.md`](NEXUS_plain_language.md) for a non-technical explanation, and [`NEXUS_v3_spec.md`](NEXUS_v3_spec.md) for the full architecture specification.

---

## Performance (ADE classification, v3.04)

Trained on ADE-Corpus-V2 (Gurulingappa et al.), 200-case batches, 20 rounds:

| Metric | Value |
|---|---|
| F1 (pharmacovigilance threshold) | **0.9455** |
| Precision | 0.897 |
| Recall | 1.000 |
| LLM call reduction via deduplication | **94%** |
| Final tree nodes | 9 (grown from 4 seed nodes) |

---

## Setup

```bash
git clone https://github.com/drelsherif/Nexus.git
cd Nexus
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

---

## Run

### OpenAI
```bash
export OPENAI_API_KEY=sk-...
./run_v3_public.sh
```

### Ollama (free, runs locally — no API key needed)
```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2
ollama pull llama3.1:70b

./run_v3_public.sh \
    --base-url http://localhost:11434/v1 \
    --classify llama3.2 \
    --synth llama3.1:70b
```

### Anthropic
```bash
export OPENAI_API_KEY=sk-ant-...
./run_v3_public.sh \
    --base-url https://api.anthropic.com/v1 \
    --classify claude-haiku-4-5 \
    --synth claude-sonnet-4-5
```

### Any OpenAI-compatible endpoint
```bash
python3 nexus_v3.py \
    --task task_configs/ade_classification.json \
    --openai \
    --openai-base-url https://your-endpoint/v1 \
    --openai-classify-model your-fast-model \
    --openai-synth-model your-best-model \
    --out my_run --rounds 20 --fresh
```

### Mock mode (offline, no API cost — for testing the pipeline)
```bash
python3 nexus_v3.py --task task_configs/ade_classification.json --mock --out test_run
```

---

## Custom tasks

NEXUS is not ADE-specific. To apply it to a new classification task:

1. Copy `task_configs/ade_classification.json`
2. Edit `labels`, `positive_label`, `description`, `route_definitions`, and `feature_flags`
3. Provide your labeled corpus as a JSONL file: `{"text": "...", "label": "POSITIVE|NEGATIVE"}`
4. Run with `--task your_config.json --corpus your_corpus.jsonl`

The learning engine, tree, MCQ system, and SWR mechanism are fully domain-agnostic.

---

## Project structure

| File | Purpose |
|---|---|
| `nexus_v3.py` | Main entry point — round loop, evaluation, calibration |
| `tree_v3.py` | Self-growing decision tree with SWR growth |
| `node.py` | NexusNode — per-node RAG, MCQ, principle memory |
| `expert_routes.py` | Parallel expert routes (causation, negation, drug_effect, context) |
| `semantic_engram.py` | Cluster-based engram formation (SWR trigger) |
| `mcq_generator.py` | MCQ generation from misclassified cases |
| `rag_index.py` | FAISS vector index over corpus |
| `embedder.py` | Biomedical sentence encoder (PubMedBERT-based) |
| `llm_client.py` | `OpenAIClient`, `AIHubClient`, `GeminiClient`, `MockClient` |
| `task_config.py` | Task configuration loader |
| `task_configs/` | Task config JSONs (ADE classification included) |
| `data_utils.py` | Corpus loader and train/eval/probe split |
| `features.py` | Regex feature flag extractor |
| `health_monitor.py` | Degradation detection |
| `homeostatic.py` | Intervention controller (experimental) |
| `nexus_db.py` | SQLite persistence layer |
| `run_v3_public.sh` | Public run script (OpenAI-compatible) |

---

## Requirements

```
datasets
faiss-cpu
numpy
openai
requests
sentence-transformers
torch
```

See `requirements.txt` for pinned versions.

---

## Citation

If you use NEXUS in your research, please cite:

```
El-Sherif Y. NEXUS: A Self-Growing Decision Tree Framework for Continuous
Clinical NLP Classification. [Preprint] 2025.
```

---

## License

MIT License.

---

*For Northwell enterprise deployment using the internal AI Hub, contact the author.*

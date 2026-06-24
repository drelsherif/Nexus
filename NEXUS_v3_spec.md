# NEXUS v3 — Architecture Specification

**Status:** Design spec — pre-implementation  
**Version:** 3.0  
**Supersedes:** nexus_run.py (v1), run_rag.py (v2b)

---

## Overview

NEXUS v3 unifies the self-growing decision tree (v1) with retrieval-augmented generation, semantic engram memory, and parallel expert consensus (v2b). Every component developed in v1 and v2b is retained; nothing is discarded. v3 is their synthesis.

**The one-sentence design rule:** The tree is the architecture — RAG, engrams, and parallel routes are capabilities of each tree node, not replacements for the tree.

---

## Key Design Decisions

### 1. Per-node memory (not global)
Each tree node has its own RAG index and engram store. This means:
- NODE_NEGATION learns from negation errors specifically, not from all errors
- A new node spawned for "chemotherapy toxicity documentation" starts with RAG context seeded from its parent's relevant errors
- Memory is local and specialist, not a global soup

### 2. SWR events grow the tree
When a node's engram cluster fires (threshold reached), two things happen:
1. Principle injected into node's system prompt (immediate fix)
2. New child node proposed targeting the error pattern (structural growth)

This means the tree topology emerges from the error distribution in the data. It is not designed — it is learned.

### 3. Class-prior calibration at runtime
`ADE_BIAS` is never hardcoded. The system computes:
```python
not_ade_prior = sum(1 for c in train_pool if c["label"] == "NOT_ADE") / len(train_pool)
ade_prior = 1 - not_ade_prior
ADE_BIAS = not_ade_prior / ade_prior  # ~2.45 for ADE corpus (71/29)
# Softened: ADE_BIAS = 1.0 + (not_ade_prior / ade_prior - 1.0) * 0.3
```
For a balanced dataset (50/50), ADE_BIAS = 1.0 (symmetric). For ADE corpus (71/29), softened ADE_BIAS ≈ 1.45.

### 4. Generalizability via task config
All domain-specific elements live in `task_config.json`. A new task needs:
- A labeled corpus (JSONL: `{"text": "...", "label": "CLASS_A"}`)
- A task config JSON (labels, routes, features, seed nodes)
- An embedding model selection

The learning loop, tree, engrams, and RAG require no code changes.

---

## File Structure

```
nexus_v3/
├── nexus_v3.py           # Main entry point (replaces nexus_run.py + run_rag.py)
├── node.py               # NexusNode class — prompt, RAG index, engram store, routes
├── tree_v3.py            # Tree: routing, grafting, retiring, serialization
├── task_config.py        # TaskConfig loader and validator
├── task_configs/
│   ├── ade_classification.json
│   └── template.json     # Copy this for new tasks
│
# Retained from v1:
├── features.py           # Feature flag extraction (unchanged)
├── nuggets.py            # Nugget vocabulary (unchanged)
├── llm_client.py         # AIHubClient with .classify() and .chat() (unchanged)
│
# Retained from v2b:
├── embedder.py           # PubMedBERT embedder (unchanged)
├── rag_index.py          # FAISS index (unchanged)
├── expert_routes.py      # 4 parallel routes (signature updated for principle_context)
├── semantic_engram.py    # SemanticEngramStore (unchanged)
│
# Data utilities (unchanged):
├── data_utils.py
├── drug_registry.py
```

---

## NexusNode Class (new)

```python
class NexusNode:
    id: str
    prompt: str                    # Specialist LLM prompt (with nugget placeholders)
    trigger_condition: str | None  # Boolean expression over feature flags
    children: list[NexusNode]
    
    # Memory (new in v3 — per-node)
    rag_index: RAGIndex            # Seeded from global index, grows with handled cases
    engram_store: SemanticEngramStore
    aggregator: RouteAggregator    # Per-node route weights
    
    # Inherited from v1
    nugget_refs: list[str]
    error_buffer: deque            # Last 60 errors at this node
    route_history: list[int]       # Cases routed here per round
    f1_history: list[float]        # This node's F1 per round
    
    def classify(self, text, features, route_llm_fn, freeform_llm_fn) -> NodeResult:
        """Full v3 classification: RAG → engram principles → parallel routes → aggregate."""
        
    def handle_error(self, text, true_label, predicted_label, round_num):
        """Add to engram store. Returns SWR event if cluster fires."""
        
    def apply_swr(self, cluster_id, freeform_llm_fn, round_num) -> SWRResult:
        """Consolidate cluster → principle → inject into prompt → propose child node."""
```

---

## Round Loop v3

```
for each round:
    1. SAMPLE       — draw B cases from train pool
    2. ROUTE        — feature extract → tree routing → specialist node
    3. RETRIEVE     — node.rag_index.query(text, k=5) per case
    4. CLASSIFY     — parallel routes × 4, aggregated with class-prior correction
    5. LEARN        — update node aggregator weights from ground truth
    6. ENGRAM       — on error: node.handle_error() → SWR if cluster fires
    7. SWR_GROW     — SWR event: inject principle + propose + probe + graft child
    8. REFINE       — nodes with ≥2 errors: LLM refines prompt, probe, accept if ΔF1>0.002
    9. EXTRACT      — accepted changes: extract nugget fragments
    10. META         — every meta_interval rounds: full-history tree-level review
    11. RETIRE       — nodes with <2 routes/round for 3+ rounds: probe removal
    12. EVAL         — full eval on held-out set
    13. LOG          — save all state, print report
```

Steps 1-6 are unchanged from their v1/v2b implementations.  
Steps 7-11 are enhanced versions of v1's SYNTHESIZE/REFINE/EXTRACT/META/RETIRE.  
The key change: step 7 (SWR_GROW) replaces pure SYNTHESIZE — growth is now memory-driven, not LLM-proposed-from-scratch.

---

## SWR Growth Algorithm

```python
def swr_grow(node, cluster_id, freeform_llm_fn, round_num, tree):
    # Step 1: Consolidate cluster into principle
    principle = node.engram_store.consolidate(cluster_id, freeform_llm_fn, round_num)
    if not principle:
        return None
    
    # Step 2: Inject principle into node's system prompt
    node.prompt += f"\n\n[ENGRAM R{round_num}]\n{principle}"
    
    # Step 3: Propose a new specialist child node for this error pattern
    cluster = node.engram_store.get_cluster(cluster_id)
    child_proposal = freeform_llm_fn(
        system="You are NEXUS, proposing a new specialist tree node.",
        user=build_child_proposal_prompt(node, cluster, principle)
    )
    child_node = parse_child_proposal(child_proposal)
    if not child_node:
        return principle  # Principle injected but no child proposed
    
    # Step 4: Seed child node's RAG from parent's cluster errors
    child_node.rag_index = RAGIndex.from_examples(cluster.errors)
    
    # Step 5: Probe — does the child improve F1?
    delta = probe_graft(tree, node, child_node, probe_pool, route_llm_fn)
    if delta >= GRAFT_THRESHOLD:
        tree.graft(node, child_node)
        return SWRResult(principle=principle, child_grafted=child_node.id, delta=delta)
    
    return SWRResult(principle=principle, child_grafted=None, delta=delta)
```

---

## Task Config Format

```json
{
  "task_name": "ADE Classification",
  "description": "Classify medical sentences as reporting an Adverse Drug Event (ADE) or not.",
  "labels": ["ADE", "NOT_ADE"],
  "positive_label": "ADE",
  "negative_label": "NOT_ADE",
  "class_prior": "auto",
  "ade_bias_softening": 0.3,
  
  "embed_model": "pritamdeka/S-PubMedBert-MS-MARCO",
  
  "route_definitions": [
    {
      "name": "causation",
      "focus": "Does the sentence contain DIRECT causal language linking a specific drug to a harmful outcome? Look for: caused, induced, associated with, resulted in, led to, following [drug], due to [drug], [drug]-related.",
      "default_vote": "NOT_ADE"
    },
    {
      "name": "negation",
      "focus": "Is any adverse outcome NEGATED, denied, hypothetical, or qualified as absent? Signals: no, not, without, denied, failed to develop, did not experience, absence of, ruled out.",
      "default_vote": "NOT_ADE"
    },
    {
      "name": "drug_effect",
      "focus": "Do the retrieved examples confirm this drug-adverse effect pair as a known ADE? Weight high-similarity retrieved examples heavily.",
      "default_vote": "NOT_ADE"
    },
    {
      "name": "context",
      "focus": "Is this outcome a desired therapeutic effect or an unintended adverse event? Therapeutic = NOT_ADE; unexpected toxicity, organ damage, hypersensitivity = ADE.",
      "default_vote": "NOT_ADE"
    }
  ],
  
  "feature_flags": {
    "has_induced":    "\\binduced\\b",
    "has_associated": "\\bassociated with\\b",
    "has_toxicity":   "\\btoxicit|\\btoxic\\b",
    "has_adverse":    "\\badverse\\b",
    "has_developed":  "\\bdeveloped\\b",
    "has_following":  "\\bfollowing\\b",
    "has_reaction":   "\\breaction\\b",
    "has_report":     "\\breport(ed|ing)?\\b",
    "has_negation":   "\\b(no|not|without|denied|absence)\\b",
    "has_short":      "__len_lt_15__",
    "has_drug_name":  "__drug_registry__"
  },
  
  "seed_nodes": [
    {
      "id": "ROOT",
      "trigger": null,
      "prompt": "You are an expert pharmacovigilance classifier. Classify the following sentence as ADE or NOT_ADE based on whether it reports an Adverse Drug Event..."
    },
    {
      "id": "NODE_NEGATION",
      "trigger": "has_negation",
      "prompt": "You are a negation specialist. This sentence contains negation language. Your task is to determine whether the adverse outcome is being DENIED..."
    },
    {
      "id": "NODE_INDUCED",
      "trigger": "has_induced or has_associated",
      "prompt": "You are a causation specialist. This sentence contains direct causal language. Determine whether this represents a true drug-induced adverse event..."
    }
  ],
  
  "hyperparameters": {
    "rounds": 20,
    "batch_size": 50,
    "k": 5,
    "engram_threshold": 5,
    "engram_similarity": 0.80,
    "principle_retrieval_sim": 0.60,
    "graft_threshold": 0.005,
    "refine_threshold": 0.002,
    "retire_threshold": -0.010,
    "error_buffer_size": 60,
    "meta_interval": 5,
    "workers": 4
  }
}
```

---

## What Stays the Same vs What Changes

### Unchanged from v1:
- `features.py` — feature flag extraction
- `nuggets.py` — nugget vocabulary system  
- `data_utils.py` — corpus loading and splitting
- `llm_client.py` — AIHubClient (both `.classify()` and `.chat()`)
- Probe-and-accept mechanism for grafts and refinements
- Rolling error buffer, meta-rounds, node retirement logic

### Unchanged from v2b:
- `embedder.py` — PubMedBERT singleton
- `rag_index.py` — FAISS build/load/query
- `expert_routes.py` — 4 parallel routes + RouteAggregator
- `semantic_engram.py` — SemanticEngramStore

### Changed:
- `RouteAggregator._aggregate()` — class-prior calibrated threshold (not hardcoded ADE_BIAS)
- Principle injection — full text into system prompt (not truncated fake examples)
- `run_rag.py` — replaced by `nexus_v3.py` which re-introduces tree routing

### New:
- `node.py` — NexusNode with per-node RAG + engram + aggregator
- `tree_v3.py` — updated tree with v3 growth and serialization
- `task_config.py` — task config loader
- `task_configs/ade_classification.json` — ADE task config
- `nexus_v3.py` — main entry point

---

## Generalizability Test Plan

After ADE classification, apply NEXUS v3 to:

1. **Medication error detection** — different positive label, different routes, same corpus structure
2. **Clinical trial eligibility** — multi-class (Eligible / Ineligible / Ambiguous), demonstrates label generalization  
3. **Radiology report classification** — different domain, different embedding model (general BERT)

Each test needs only a new `task_config.json` and a labeled corpus. No code changes.

---

## Open Questions

1. **Per-node RAG seeding**: should child nodes inherit only the parent's error cluster, or the full parent corpus? Start with error cluster only; add global fallback if recall drops.

2. **Route definitions from task config**: current routes are hardcoded in `expert_routes.py`. v3 should build route prompts dynamically from `route_definitions` in task config. This requires refactoring `_causation_route` etc. into a single `_generic_route(name, focus, text, examples, llm_fn, principle_context)`.

3. **Negation route**: consistently underperforms (70-73% accuracy vs 90%+ for others). Consider whether negation is better handled as a feature flag (pre-classification filter) rather than a voting route.

4. **Probe cost**: each SWR graft event requires a 300-case probe. With per-node RAG, this may need to be scoped to the node's routing share (e.g., 50-case node probe + 100-case global eval).

---

## Build Order

1. `task_config.py` + `task_configs/ade_classification.json` — config layer first
2. `node.py` — NexusNode with per-node memory
3. `tree_v3.py` — routing + graft + retire using NexusNode
4. Refactor `expert_routes.py` — generic route from task config definitions
5. `nexus_v3.py` — main loop wiring everything together
6. Verify on ADE corpus: baseline should match v2b (~0.788), rounds should improve past 0.839
7. Port task config to second domain (medication error or radiology)

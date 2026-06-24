"""
node.py
NEXUS v3 — NexusNode with Per-Node Memory

Each tree node is a specialist with its own:
  - RAG index (cases it has handled)
  - MCQ library (teaching cases from its own errors)
  - Semantic engram store (error clusters → principles)
  - Route aggregator (specialist route weights)
  - Nugget references and error buffer

Biological analogy:
  Each NexusNode is a cortical column — a specialist area that develops
  its own internal representations through experience. Memory is local
  and specialist, not global. The hippocampus (engrams + RAG) feeds each
  column specifically, not uniformly.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from expert_routes import RouteAggregator, AggregatedResult
from mcq_generator import MCQGenerator, MCQLibrary, NEXUSQuestion
from semantic_engram import SemanticEngramStore
from rag_index import RAGIndex
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nexus_db import NexusDB


# ─── Node result ──────────────────────────────────────────────────────────────

@dataclass
class NodeResult:
    label: str
    confidence: float
    node_id: str
    route_result: AggregatedResult
    mcq_context_used: bool
    principle_context_used: bool
    rag_examples_used: int

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "node_id": self.node_id,
            "agreement": round(self.route_result.agreement, 3),
            "split": self.route_result.split,
            "mcq_used": self.mcq_context_used,
            "principle_used": self.principle_context_used,
            "rag_examples": self.rag_examples_used,
        }


# ─── SWR result ───────────────────────────────────────────────────────────────

@dataclass
class SWRResult:
    cluster_id: str
    principle: str
    child_node_proposal: Optional[dict]   # parsed from LLM, None if not proposed/accepted


# ─── NexusNode ────────────────────────────────────────────────────────────────

class NexusNode:
    """
    A specialist node in the NEXUS v3 decision tree.

    Memory is per-node:
      - rag_index:     cases this node has classified (grows over rounds)
      - mcq_library:   MCQ teaching cases from this node's errors
      - engram_store:  semantic error clusters → principles
      - aggregator:    route weights learned by this node specifically

    Growth:
      When engram_store fires a SWR event, the node:
        1. Updates its system prompt with the consolidated principle
        2. Proposes a new specialist child node for the error pattern
    """

    def __init__(
        self,
        node_id: str,
        prompt: str,
        trigger_condition: Optional[str],
        task_config,
        rag_index: Optional[RAGIndex] = None,
        mcq_library: Optional[MCQLibrary] = None,
        engram_store: Optional[SemanticEngramStore] = None,
        aggregator: Optional[RouteAggregator] = None,
        path: Optional[str] = None,
        db: Optional["NexusDB"] = None,
    ):
        self.id = node_id
        self.prompt = prompt
        self.trigger_condition = trigger_condition
        self.task_config = task_config
        self.children: list["NexusNode"] = []

        # Memory (per-node)
        self.path = Path(path) if path else None
        node_path = str(self.path / self.id) if self.path else None

        self.rag_index = rag_index          # Seeded from global; grows with handled cases
        self.mcq_library = mcq_library or MCQLibrary(
            path=f"{node_path}/mcq" if node_path else None
        )
        self.engram_store = engram_store or SemanticEngramStore(
            threshold=task_config.get_hyperparameter("engram_threshold", 5),
            similarity_threshold=task_config.get_hyperparameter("engram_similarity", 0.80),
            path=f"{node_path}/engrams" if node_path else None,
        )
        self.aggregator = aggregator or RouteAggregator(
            learning_rate=task_config.get_hyperparameter("learning_rate", 0.05),
            penalty=task_config.get_hyperparameter("route_penalty", 1.5),
        )

        # Inherited from v1
        self.error_buffer: deque = deque(
            maxlen=task_config.get_hyperparameter("error_buffer_size", 60)
        )
        self.route_history: list[int] = []   # cases routed here per round
        self.f1_history: list[float] = []    # this node's F1 per round
        self.nugget_refs: list[str] = []

        # Principles injected into this node's prompt (from SWR events)
        # Each entry is a dict: {principle, round_added, cluster_id}
        # (legacy str entries are supported for backwards compat)
        self.injected_principles: list = []

        # Seed prompt — stored separately so rebuild_prompt() can reconstruct
        self._seed_prompt: str = prompt

        # SQLite DB (optional — enables MCQ deduplication + LLM call tracking)
        self.db: Optional["NexusDB"] = db

        # Round counter
        self._round_routes = 0   # cases routed this round (reset each round)

    # ── Classification ────────────────────────────────────────────────────────

    def classify(
        self,
        text: str,
        route_llm_fn: Callable,
        global_rag_index: Optional[RAGIndex] = None,
        workers: int = 4,
    ) -> NodeResult:
        """
        Full v3 classification:
          1. Retrieve from node's RAG index (+ global fallback)
          2. Retrieve relevant MCQ teaching cases
          3. Retrieve relevant engram principles
          4. Run parallel routes with full context
          5. Aggregate with class-prior correction
        """
        self._round_routes += 1

        # Step 1: RAG retrieval
        k = self.task_config.get_hyperparameter("k", 5)
        examples = []
        if self.rag_index and self.rag_index._index is not None:
            examples = self.rag_index.query(text, k=k)
        # Fallback to global index for any gaps
        if len(examples) < k and global_rag_index:
            needed = k - len(examples)
            global_examples = global_rag_index.query(text, k=needed + 2)
            # Deduplicate
            existing_texts = {e["text"] for e in examples}
            for ex in global_examples:
                if ex["text"] not in existing_texts and len(examples) < k:
                    examples.append(ex)

        # Step 2: MCQ teaching cases
        mcq_k = self.task_config.get_hyperparameter("mcq_k", 3)
        mcq_sim = self.task_config.get_hyperparameter("mcq_retrieval_sim", 0.70)
        mcq_context = self.mcq_library.format_for_context(text, k=mcq_k, min_sim=mcq_sim)
        mcq_used = bool(mcq_context)

        # Step 3: Engram principles
        top_k_principles = self.task_config.get_hyperparameter("top_k_principles", 2)
        principle_sim = self.task_config.get_hyperparameter("principle_retrieval_sim", 0.60)
        relevant_principles = self.engram_store.retrieve_principles(text, top_k=top_k_principles)
        principle_context = ""
        principle_used = False
        if relevant_principles:
            blocks = "\n\n".join(
                f"[ENGRAM {i+1}] {eng['principle']}"
                for i, eng in enumerate(relevant_principles)
            )
            principle_context = (
                f"\n\nThe following NEXUS principles were learned from similar past errors "
                f"and MUST inform your vote:\n\n{blocks}"
            )
            principle_used = True

        # Combine: principles + MCQ teaching cases
        full_principle_context = principle_context
        if mcq_context:
            full_principle_context += f"\n\n{mcq_context}"

        # Step 4: Parallel routes
        route_result = self.aggregator.classify(
            text=text,
            examples=examples,
            llm_fn=route_llm_fn,
            workers=workers,
            principle_context=full_principle_context,
        )

        return NodeResult(
            label=route_result.final_label,
            confidence=route_result.confidence,
            node_id=self.id,
            route_result=route_result,
            mcq_context_used=mcq_used,
            principle_context_used=principle_used,
            rag_examples_used=len(examples),
        )

    # ── Error handling ────────────────────────────────────────────────────────

    def handle_error(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        round_num: int,
        route_result: AggregatedResult,
        mcq_generator: MCQGenerator,
        freeform_llm_fn: Callable,
        context_examples: Optional[list[dict]] = None,
    ) -> Optional[SWRResult]:
        """
        Process a misclassification:
          1. Update route weights
          2. Add to error buffer
          3. Generate MCQ teaching case
          4. Add to engram store — may fire SWR
          5. If SWR fires: consolidate → principle → propose child node
        """
        # 1. Route weight update
        self.aggregator.update_weights(route_result, true_label)

        # 2. Error buffer (for meta-round context)
        self.error_buffer.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "round": round_num,
        })

        # 3. Generate MCQ — with DB deduplication to save LLM calls.
        #    Flow: embed text (free, local) → check DB → skip LLM if duplicate exists.
        q = None
        _mcq_skipped = False
        if self.db is not None:
            # Pre-check: embed and search before calling the LLM
            try:
                from embedder import embed as _embed
                _emb = _embed([text])[0].tolist()
                existing_id = self.db.find_similar_mcq(_emb, self.id, min_sim=0.85)
                if existing_id is not None:
                    # Similar teaching case already exists — skip LLM call
                    self.db.log_llm_call("mcq_gen", round_num, self.id, skipped=True)
                    _mcq_skipped = True
                else:
                    # No match → generate new MCQ
                    self.db.log_llm_call("mcq_gen", round_num, self.id, skipped=False)
                    q = mcq_generator.generate(
                        text=text,
                        true_label=true_label,
                        predicted_label=predicted_label,
                        context_examples=context_examples or [],
                        llm_fn=freeform_llm_fn,
                        node_id=self.id,
                        round_num=round_num,
                    )
                    if q:
                        self.mcq_library.add(q)
                        # Persist to DB with embedding for future dedup
                        self.db.insert_mcq(
                            node_id=self.id,
                            round_num=round_num,
                            text=q.text,
                            true_label=q.correct_label,
                            predicted_label=q.predicted_label,
                            correct_reasoning=q.correct_reasoning,
                            error_type=q.distractors[0].error_type if q.distractors else "other",
                            difficulty=q.difficulty,
                            embedding=q.embedding,
                            distractors=[
                                {"label": d.label, "reasoning": d.reasoning,
                                 "correction": d.correction, "error_type": d.error_type}
                                for d in q.distractors
                            ],
                        )
            except Exception as _e:
                # DB dedup failed — fall back to always generating
                q = mcq_generator.generate(
                    text=text,
                    true_label=true_label,
                    predicted_label=predicted_label,
                    context_examples=context_examples or [],
                    llm_fn=freeform_llm_fn,
                    node_id=self.id,
                    round_num=round_num,
                )
                if q:
                    self.mcq_library.add(q)
        else:
            # No DB — original behaviour
            q = mcq_generator.generate(
                text=text,
                true_label=true_label,
                predicted_label=predicted_label,
                context_examples=context_examples or [],
                llm_fn=freeform_llm_fn,
                node_id=self.id,
                round_num=round_num,
            )
            if q:
                self.mcq_library.add(q)

        # 4. Semantic engram
        fired = self.engram_store.add_error(
            text=text,
            true_label=true_label,
            predicted_label=predicted_label,
            node_id=self.id,
            round_num=round_num,
        )

        # 5. SWR consolidation
        if fired:
            cluster_id, cluster_errors = fired
            principle = self.engram_store.consolidate(
                cluster_id, freeform_llm_fn, round_num
            )
            if principle:
                # Inject principle into this node's prompt (stored with metadata)
                self.injected_principles.append({
                    "principle": principle,
                    "round_added": round_num,
                    "cluster_id": cluster_id,
                })
                self.rebuild_prompt()

                # Propose a new specialist child node
                child_proposal = self._propose_child_node(
                    cluster_errors, principle, freeform_llm_fn, round_num
                )
                return SWRResult(
                    cluster_id=cluster_id,
                    principle=principle,
                    child_node_proposal=child_proposal,
                )

        return None

    def update_weights_on_correct(self, route_result: AggregatedResult, true_label: str):
        """Update route weights for a correct classification."""
        self.aggregator.update_weights(route_result, true_label)

    # ── SWR child node proposal ───────────────────────────────────────────────

    def _propose_child_node(
        self,
        cluster_errors: list,
        principle: str,
        freeform_llm_fn: Callable,
        round_num: int,
    ) -> Optional[dict]:
        """
        Ask the LLM to propose a new specialist child node for the error pattern.
        Returns a dict with id, trigger, prompt, description — or None if proposal fails.
        """
        error_samples = "\n".join(
            f"  [{i+1}] (True={e.true_label}) \"{e.text[:150]}\""
            for i, e in enumerate(cluster_errors[:5])
        )

        system = (
            f"You are NEXUS, designing a new specialist tree node for {self.task_config.task_name}. "
            f"Available feature flags: {list(self.task_config.feature_flags.keys())}"
        )

        prompt = f"""Node {self.id} repeatedly misclassified these similar sentences:
{error_samples}

The following principle was consolidated from these errors:
{principle[:500]}

Design a NEW SPECIALIST CHILD NODE that would correctly handle this error pattern.

Available feature flags for trigger conditions: {list(self.task_config.feature_flags.keys())}

Respond ONLY with JSON:
{{
  "id": "NODE_[DESCRIPTIVE_NAME_IN_CAPS]",
  "trigger": "<boolean expression using available feature flags, or null if should catch all>",
  "description": "<one sentence: what pattern does this node specialize in?>",
  "prompt": "<specialist classification prompt, 100-200 words, incorporating the lesson learned>"
}}"""

        try:
            raw = freeform_llm_fn(system, prompt)
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            # Extract JSON
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return None
            proposal = json.loads(m.group())
            # Validate required fields
            if not all(k in proposal for k in ("id", "prompt")):
                return None
            return proposal
        except Exception:
            return None

    # ── Round management ──────────────────────────────────────────────────────

    def end_round(self, round_num: int, f1: Optional[float] = None):
        """Called at end of each round to record stats and reset counters."""
        self.route_history.append(self._round_routes)
        if f1 is not None:
            self.f1_history.append(f1)
        self._round_routes = 0

    def rebuild_prompt(self) -> None:
        """
        Reconstruct self.prompt from the seed prompt + all currently
        injected principles. Called after any principle is added or removed
        so the node's active prompt always reflects current principle state.
        """
        self.prompt = self._seed_prompt
        for p in self.injected_principles:
            if isinstance(p, dict):
                r   = p.get("round_added", "?")
                txt = p.get("principle", "")
                self.prompt += f"\n\n[ENGRAM R{r}]\n{txt}"
            else:
                # Legacy string format
                self.prompt += f"\n\n[ENGRAM]\n{p}"

    def average_routes_per_round(self, last_n: int = 3) -> float:
        if not self.route_history:
            return 0.0
        recent = self.route_history[-last_n:]
        return sum(recent) / len(recent)

    # ── Trigger evaluation ────────────────────────────────────────────────────

    def matches_trigger(self, feature_flags: dict[str, bool]) -> bool:
        """
        Evaluate trigger_condition against extracted feature flags.
        Returns True if this node should handle the case.
        Returns True for ROOT (trigger is None).
        """
        if self.trigger_condition is None:
            return True  # ROOT always matches

        expr = self.trigger_condition
        # Replace feature flag names with their boolean values
        for flag, val in feature_flags.items():
            expr = re.sub(r'\b' + re.escape(flag) + r'\b', str(val), expr)

        # Replace 'and'/'or'/'not' (already Python keywords)
        try:
            return bool(eval(expr))  # Safe: only booleans + and/or/not
        except Exception:
            return False

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if not self.path:
            return
        node_path = self.path / self.id
        node_path.mkdir(parents=True, exist_ok=True)

        # Node state
        state = {
            "id": self.id,
            "seed_prompt": self._seed_prompt,
            "prompt": self.prompt,
            "trigger_condition": self.trigger_condition,
            "injected_principles": self.injected_principles,
            "nugget_refs": self.nugget_refs,
            "route_history": self.route_history,
            "f1_history": self.f1_history,
            "error_buffer": list(self.error_buffer),
            "aggregator": self.aggregator.to_dict(),
        }
        (node_path / "node_state.json").write_text(json.dumps(state, indent=2))

        # Sub-stores
        self.mcq_library.save()
        self.engram_store.save()

    @classmethod
    def load(cls, node_id: str, path: str, task_config) -> "NexusNode":
        node_path = Path(path) / node_id
        f = node_path / "node_state.json"
        if not f.exists():
            raise FileNotFoundError(f"Node state not found: {f}")

        state = json.loads(f.read_text())

        # Load sub-stores
        mcq_path = str(node_path / "mcq")
        engram_path = str(node_path / "engrams")
        mcq_library = MCQLibrary.load(mcq_path) if (node_path / "mcq" / "mcq_library.json").exists() else MCQLibrary(path=mcq_path)
        engram_store = SemanticEngramStore.load(engram_path) if (node_path / "engrams" / "semantic_engrams.json").exists() else SemanticEngramStore(path=engram_path)

        aggregator = RouteAggregator()
        if "aggregator" in state:
            aggregator = RouteAggregator.from_dict(state["aggregator"])

        node = cls(
            node_id=state["id"],
            prompt=state["prompt"],
            trigger_condition=state.get("trigger_condition"),
            task_config=task_config,
            mcq_library=mcq_library,
            engram_store=engram_store,
            aggregator=aggregator,
            path=path,
        )
        node._seed_prompt = state.get("seed_prompt", state.get("prompt", ""))
        node.injected_principles = state.get("injected_principles", [])
        node.rebuild_prompt()  # Reconstruct prompt from seed + principles
        node.nugget_refs = state.get("nugget_refs", [])
        node.route_history = state.get("route_history", [])
        node.f1_history = state.get("f1_history", [])
        for err in state.get("error_buffer", []):
            node.error_buffer.append(err)

        return node

    def __repr__(self) -> str:
        return (
            f"NexusNode(id={self.id!r}, "
            f"trigger={self.trigger_condition!r}, "
            f"mcqs={len(self.mcq_library)}, "
            f"engrams={len(self.engram_store._clusters)}, "
            f"principles={len(self.injected_principles)}, "
            f"children={[c.id for c in self.children]})"
        )

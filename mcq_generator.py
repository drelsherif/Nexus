"""
mcq_generator.py
NEXUS v3 — MCQ-Based Learning from Misclassifications

Biological inspiration:
  Medical students learn best by generating MCQ questions from their own
  mistakes: correct answer with clinical reasoning, the wrong choice with
  explanation of exactly what pattern led to the error, and distractors
  that sharpen adjacent decision boundaries.

  NEXUS applies this principle: every misclassification generates a full
  teaching case. Wrong answers are as informative as right answers — they
  define the decision boundary, not just one side of it.

Architecture:
  NEXUSQuestion  — complete MCQ: correct + wrong + distractors + difficulty
  Distractor     — a wrong answer with reasoning chain and error taxonomy
  MCQGenerator   — generates MCQs from misclassifications using LLM
  MCQLibrary     — stores, embeds, retrieves MCQs by semantic similarity

At inference time, retrieved MCQs inject complete clinical reasoning chains
into each route's context — including the specific wrong reasoning to avoid.

Example retrieved MCQ injection:
  [TEACHING CASE 1 — ADE (sim=0.89)]
  Sentence: "The patient developed nephrotoxicity following gentamicin."
  ✓ ADE because: Nephrotoxicity is an unintended harmful outcome. "Developed
    following" establishes causal attribution. Aminoglycoside nephrotoxicity
    is well-documented.
  ✗ NOT_ADE would be wrong because: Nephrotoxicity is NOT a therapeutic goal
    of gentamicin. The word "following" signals temporal causal proximity.
    Error type: therapeutic_goal_confusion
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from embedder import embed_one, embed


# ─── Datatypes ────────────────────────────────────────────────────────────────

@dataclass
class Distractor:
    """A wrong answer choice in a NEXUS MCQ."""
    label: str              # The wrong label (e.g., "NOT_ADE")
    reasoning: str          # Why this choice is plausible (what the model over-weighted)
    correction: str         # Why it's wrong; what signal proves the correct answer
    error_type: str         # Taxonomy: negation_confusion, therapeutic_goal_confusion, etc.


@dataclass
class NEXUSQuestion:
    """
    A complete clinical MCQ generated from a NEXUS misclassification.

    Stored in MCQLibrary and retrieved at inference time to give routes
    both the correct reasoning AND the wrong reasoning to avoid.
    """
    # Case
    text: str
    node_id: str
    round_num: int

    # Correct answer
    correct_label: str
    correct_reasoning: str  # Full clinical reasoning for the correct answer

    # Wrong answers
    distractors: list[Distractor]  # [0] = model's actual wrong choice; [1+] = additional

    # Metadata
    difficulty: str         # "easy" | "medium" | "hard"
    predicted_label: str    # What the model actually predicted (wrong)

    # Embedding for retrieval (set by MCQLibrary after generation)
    embedding: list[float] = field(default_factory=list)

    @property
    def primary_distractor(self) -> Optional[Distractor]:
        """The distractor matching the model's actual wrong prediction."""
        return self.distractors[0] if self.distractors else None

    def format_for_injection(self, sim_score: float = 0.0) -> str:
        """
        Format this MCQ as a teaching case for route context injection.
        This is the key: routes see complete reasoning chains, not just labels.
        """
        lines = [
            f"[TEACHING CASE — {self.correct_label} (sim={sim_score:.2f}, {self.difficulty})]",
            f'Sentence: "{self.text}"',
            f"✓ {self.correct_label} because: {self.correct_reasoning}",
        ]
        for d in self.distractors[:2]:  # max 2 distractors in context
            lines.append(
                f"✗ {d.label} would be wrong because: {d.correction} "
                f"[Error type: {d.error_type}]"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "node_id": self.node_id,
            "round_num": self.round_num,
            "correct_label": self.correct_label,
            "correct_reasoning": self.correct_reasoning,
            "distractors": [
                {
                    "label": d.label,
                    "reasoning": d.reasoning,
                    "correction": d.correction,
                    "error_type": d.error_type,
                }
                for d in self.distractors
            ],
            "difficulty": self.difficulty,
            "predicted_label": self.predicted_label,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NEXUSQuestion":
        return cls(
            text=d["text"],
            node_id=d["node_id"],
            round_num=d["round_num"],
            correct_label=d["correct_label"],
            correct_reasoning=d["correct_reasoning"],
            distractors=[
                Distractor(
                    label=dist["label"],
                    reasoning=dist["reasoning"],
                    correction=dist["correction"],
                    error_type=dist["error_type"],
                )
                for dist in d.get("distractors", [])
            ],
            difficulty=d.get("difficulty", "medium"),
            predicted_label=d.get("predicted_label", ""),
            embedding=d.get("embedding", []),
        )


# ─── MCQ Generator ────────────────────────────────────────────────────────────

class MCQGenerator:
    """
    Generates MCQs from misclassifications using freeform LLM calls.

    On every error, the LLM generates:
      1. Complete reasoning for the correct answer
      2. Explanation of the model's wrong choice (what pattern was over-weighted)
      3. One additional distractor to sharpen adjacent decision boundaries
      4. Difficulty rating

    This produces a teaching case that encodes both what's right AND
    what's wrong — the full decision boundary, not just one side.
    """

    def __init__(self, task_config):
        self.config = task_config

    def generate(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        context_examples: list[dict],
        llm_fn: Callable[[str, str], str],
        node_id: str,
        round_num: int,
    ) -> Optional[NEXUSQuestion]:
        """
        Generate a complete MCQ from a misclassification.
        Returns NEXUSQuestion or None if generation fails.
        """
        prompt = self._build_prompt(text, true_label, predicted_label, context_examples)
        system = self._build_system()

        try:
            raw = llm_fn(system, prompt)
            return self._parse_mcq(raw, text, true_label, predicted_label, node_id, round_num)
        except Exception as e:
            # Fallback: minimal MCQ from the error alone
            return self._minimal_mcq(text, true_label, predicted_label, node_id, round_num, str(e))

    def _build_system(self) -> str:
        return (
            f"You are NEXUS, a self-improving {self.config.task_name} classifier. "
            f"You just made a classification error. Your task is to generate a clinical "
            f"MCQ teaching case that will help you avoid this mistake in the future. "
            f"Be precise, clinical, and focus on the specific signal that distinguishes "
            f"the correct answer from the wrong one."
        )

    def _build_prompt(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        context_examples: list[dict],
    ) -> str:
        error_taxonomy = self.config.get_error_taxonomy()
        taxonomy_str = ", ".join(f'"{e}"' for e in error_taxonomy)

        examples_str = ""
        if context_examples:
            lines = []
            for i, ex in enumerate(context_examples[:4]):
                bar = f"▲{ex['label']}" if ex["label"] == self.config.positive_label else f"▽{ex['label']}"
                sim = f"{ex.get('score', 0):.2f}"
                lines.append(f"  [{i+1}] {bar} (sim={sim}) \"{ex['text'][:120]}\"")
            examples_str = f"\n\nSimilar cases from literature:\n" + "\n".join(lines)

        return f"""You incorrectly classified the following sentence as {predicted_label}.
The correct answer is {true_label}.

Sentence: "{text}"{examples_str}

Generate a complete MCQ teaching case. Respond ONLY with JSON in this exact format:
{{
  "correct_reasoning": "<2-3 sentences: what specific clinical signals confirm {true_label}? What is the key decision rule?>",
  "distractors": [
    {{
      "label": "{predicted_label}",
      "reasoning": "<1-2 sentences: what made {predicted_label} plausible? What signal did you over-weight or miss?>",
      "correction": "<1-2 sentences: why is {predicted_label} definitively wrong here? What specific signal rules it out?>",
      "error_type": "<one of: {taxonomy_str}>"
    }},
    {{
      "label": "{predicted_label}",
      "reasoning": "<a different wrong reasoning path — what else might lead someone astray on this sentence?>",
      "correction": "<why that reasoning is also wrong>",
      "error_type": "<error type>"
    }}
  ],
  "difficulty": "<easy|medium|hard — based on how subtle the distinction is>"
}}"""

    def _parse_mcq(
        self,
        raw: str,
        text: str,
        true_label: str,
        predicted_label: str,
        node_id: str,
        round_num: int,
    ) -> Optional[NEXUSQuestion]:
        # Strip markdown code fences
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from surrounding text
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return self._minimal_mcq(text, true_label, predicted_label, node_id, round_num, "json_parse_fail")
            try:
                d = json.loads(m.group())
            except Exception:
                return self._minimal_mcq(text, true_label, predicted_label, node_id, round_num, "json_extract_fail")

        distractors = [
            Distractor(
                label=dist.get("label", predicted_label),
                reasoning=dist.get("reasoning", ""),
                correction=dist.get("correction", ""),
                error_type=dist.get("error_type", "other"),
            )
            for dist in d.get("distractors", [])
        ]

        return NEXUSQuestion(
            text=text,
            node_id=node_id,
            round_num=round_num,
            correct_label=true_label,
            correct_reasoning=d.get("correct_reasoning", ""),
            distractors=distractors,
            difficulty=d.get("difficulty", "medium"),
            predicted_label=predicted_label,
        )

    def _minimal_mcq(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        node_id: str,
        round_num: int,
        reason: str,
    ) -> NEXUSQuestion:
        """Fallback MCQ when LLM generation fails — minimal but still useful."""
        return NEXUSQuestion(
            text=text,
            node_id=node_id,
            round_num=round_num,
            correct_label=true_label,
            correct_reasoning=f"Correct label is {true_label}. (Reasoning generation failed: {reason})",
            distractors=[
                Distractor(
                    label=predicted_label,
                    reasoning=f"Model predicted {predicted_label}.",
                    correction=f"Correct answer is {true_label}.",
                    error_type="other",
                )
            ],
            difficulty="medium",
            predicted_label=predicted_label,
        )


# ─── MCQ Library ──────────────────────────────────────────────────────────────

class MCQLibrary:
    """
    Stores MCQs with embeddings, supports semantic retrieval.

    At inference time, given a new sentence, retrieves the K most similar
    MCQs — complete teaching cases with correct + wrong reasoning chains.
    These replace or augment the plain labeled examples from RAG retrieval.

    Biological analogy:
      Each MCQ is a cortical engram with both the memory trace and the
      error signal that formed it. Retrieval completes the pattern:
      "This new case is similar to a case I got wrong; here's what I
       missed and what the correct reasoning was."
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._questions: list[NEXUSQuestion] = []
        self._embeddings: Optional[np.ndarray] = None  # (N, dim) matrix for fast search

    def add(self, question: NEXUSQuestion) -> None:
        """Add a new MCQ, compute its embedding."""
        vec = embed_one(question.text)
        question.embedding = vec.tolist()
        self._questions.append(question)
        # Invalidate embedding cache
        self._embeddings = None

    def _ensure_matrix(self) -> bool:
        """Build embedding matrix if needed. Returns False if empty."""
        if not self._questions:
            return False
        if self._embeddings is None:
            self._embeddings = np.array(
                [q.embedding for q in self._questions], dtype=np.float32
            )
        return True

    def retrieve(self, text: str, k: int = 3, min_sim: float = 0.70) -> list[tuple[float, NEXUSQuestion]]:
        """
        Retrieve K most similar MCQs for a given text.
        Returns [(similarity_score, NEXUSQuestion), ...] sorted by similarity desc.
        """
        if not self._ensure_matrix():
            return []

        vec = embed_one(text)
        sims = self._embeddings @ vec  # cosine similarity (unit vectors)
        top_k = min(k, len(self._questions))
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim >= min_sim:
                results.append((sim, self._questions[idx]))

        return results

    def format_for_context(self, text: str, k: int = 3, min_sim: float = 0.70) -> str:
        """
        Retrieve and format MCQs as a teaching context block for route injection.

        Returns a string like:
          === NEXUS TEACHING CASES (from similar past errors) ===
          [TEACHING CASE 1 — ADE (sim=0.89, medium)]
          Sentence: "..."
          ✓ ADE because: ...
          ✗ NOT_ADE would be wrong because: ...
        """
        retrieved = self.retrieve(text, k=k, min_sim=min_sim)
        if not retrieved:
            return ""

        blocks = ["\n=== NEXUS TEACHING CASES (learned from similar past errors) ==="]
        for sim, q in retrieved:
            blocks.append(q.format_for_injection(sim_score=sim))
        return "\n\n".join(blocks)

    def error_type_distribution(self) -> dict[str, int]:
        """Count error types across all MCQs — useful for diagnostics."""
        counts: dict[str, int] = {}
        for q in self._questions:
            for d in q.distractors:
                counts[d.error_type] = counts.get(d.error_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def difficulty_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self._questions:
            counts[q.difficulty] = counts.get(q.difficulty, 0) + 1
        return counts

    def print_report(self, top_n: int = 5) -> None:
        if not self._questions:
            print("[MCQ] No questions in library yet.")
            return
        print(f"\n[MCQ] Library: {len(self._questions)} questions")
        error_dist = self.error_type_distribution()
        diff_dist = self.difficulty_distribution()
        print(f"[MCQ] Difficulty: {diff_dist}")
        print(f"[MCQ] Top error types: {dict(list(error_dist.items())[:top_n])}")
        if self._questions:
            recent = self._questions[-3:]
            print("[MCQ] Recent questions:")
            for q in recent:
                print(f"  R{q.round_num} {q.node_id}: '{q.text[:60]}' "
                      f"({q.predicted_label}→{q.correct_label}, {q.difficulty})")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        data = [q.to_dict() for q in self._questions]
        (self.path / "mcq_library.json").write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "MCQLibrary":
        p = Path(path)
        lib = cls(path=path)
        f = p / "mcq_library.json"
        if not f.exists():
            return lib
        data = json.loads(f.read_text())
        for d in data:
            q = NEXUSQuestion.from_dict(d)
            lib._questions.append(q)
        # Rebuild embedding matrix
        if lib._questions:
            lib._embeddings = np.array(
                [q.embedding for q in lib._questions], dtype=np.float32
            )
        return lib

    def __len__(self) -> int:
        return len(self._questions)

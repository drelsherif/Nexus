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

  v3.05 additions:
    - Near-miss MCQs: generated from close-call correct predictions where
      the model almost got it wrong. These sharpen the exact decision boundary.
    - Positive anchor MCQs: generated from high-confidence correct predictions.
      These reinforce the clearest signals on each side of the boundary.
    - Difficulty-weighted retrieval: hard MCQs rank higher in similarity search.
    - Cascading min_sim fallback: 0.70 → 0.55 if nothing found above threshold.
    - Cross-node retrieval: thin libraries can supplement from a global pool.

Architecture:
  NEXUSQuestion  — complete MCQ: correct + wrong + distractors + difficulty
  Distractor     — a wrong answer with reasoning chain and error taxonomy
  MCQGenerator   — generates MCQs from misclassifications using LLM
  MCQLibrary     — stores, embeds, retrieves MCQs by semantic similarity

At inference time, retrieved MCQs inject complete clinical reasoning chains
into each route's context — including the specific wrong reasoning to avoid.

Example retrieved MCQ injection:
  [TEACHING CASE — ADE (sim=0.89, hard)]
  Sentence: "The patient developed nephrotoxicity following gentamicin."
  ✓ ADE because: Nephrotoxicity is an unintended harmful outcome. "Developed
    following" establishes causal attribution. Aminoglycoside nephrotoxicity
    is well-documented.
  ✗ NOT_ADE trap 1: Nephrotoxicity is NOT a therapeutic goal of gentamicin.
    The word "following" signals temporal causal proximity. [negation_confusion]
  ✗ NOT_ADE trap 2: Surface reading — "following" might be read as temporal
    sequence only, not causation. But the organ damage noun ("nephrotoxicity")
    is ADE-specific vocabulary. [temporal_confusion]
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
    A complete clinical MCQ generated from a NEXUS classification event.

    Three MCQ types:
      "error"      — from a misclassification (model was wrong)
      "near_miss"  — from a close-call correct (margin < 20%, model almost wrong)
      "positive"   — from a high-confidence correct (anchors the clear boundary)

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
    distractors: list[Distractor]  # [0] = primary trap; [1] = surface-reading trap

    # Metadata
    difficulty: str         # "easy" | "medium" | "hard"
    predicted_label: str    # What the model actually predicted (wrong); "" for positive MCQs
    mcq_type: str           # "error" | "near_miss" | "positive"

    # Usage tracking
    review_count: int = 0           # how many times this MCQ was retrieved
    last_matched_round: int = -1    # round when last retrieved

    # Embedding for retrieval (set by MCQLibrary after generation)
    embedding: list[float] = field(default_factory=list)

    @property
    def primary_distractor(self) -> Optional[Distractor]:
        """The distractor matching the model's actual wrong prediction."""
        return self.distractors[0] if self.distractors else None

    def format_for_injection(self, sim_score: float = 0.0) -> str:
        """
        Format this MCQ as a teaching case for route context injection.
        Different headers for each MCQ type make the context clearer.
        """
        if self.mcq_type == "positive":
            header = f"[ANCHOR CASE — {self.correct_label} (sim={sim_score:.2f}, {self.difficulty})]"
        elif self.mcq_type == "near_miss":
            header = f"[NEAR-MISS CASE — {self.correct_label} (sim={sim_score:.2f}, {self.difficulty})]"
        else:
            header = f"[TEACHING CASE — {self.correct_label} (sim={sim_score:.2f}, {self.difficulty})]"

        lines = [
            header,
            f'Sentence: "{self.text}"',
            f"✓ {self.correct_label} because: {self.correct_reasoning}",
        ]
        for i, d in enumerate(self.distractors[:2]):
            lines.append(
                f"✗ {d.label} trap {i+1}: {d.correction} "
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
            "mcq_type": self.mcq_type,
            "review_count": self.review_count,
            "last_matched_round": self.last_matched_round,
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
            mcq_type=d.get("mcq_type", "error"),
            review_count=d.get("review_count", 0),
            last_matched_round=d.get("last_matched_round", -1),
            embedding=d.get("embedding", []),
        )


# ─── MCQ Generator ────────────────────────────────────────────────────────────

class MCQGenerator:
    """
    Generates MCQs from three types of classification events:

      1. Error MCQs (generate): model was wrong — encodes both correct reasoning
         AND two distinct wrong-reasoning traps (cognitive traps, not just the
         model's actual error).

      2. Near-miss MCQs (generate_near_miss): model was right but barely —
         score margin < 20%. Encodes why the correct label wins despite ambiguity.
         These are the most valuable boundary cases.

      3. Positive anchor MCQs (generate_positive): high-confidence correct —
         encodes the clear signals that make this an unambiguous case. Prevents
         the model from drifting toward over-generalizing its error corrections.

    Biological analogy:
      Error MCQs       → learning from failure (error-driven LTP)
      Near-miss MCQs   → learning at the decision boundary (predictive coding)
      Positive MCQs    → reinforcement of clear patterns (Hebbian consolidation)
    """

    def __init__(self, task_config):
        self.config = task_config

    # ── Error MCQ ─────────────────────────────────────────────────────────────

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
        Generate a complete error MCQ from a misclassification.
        Two distractors: the model's actual error trap + a different surface trap.
        """
        prompt = self._build_error_prompt(text, true_label, predicted_label, context_examples)
        system = self._build_system("error")

        try:
            raw = llm_fn(system, prompt)
            q = self._parse_mcq(raw, text, true_label, predicted_label, node_id, round_num, "error")
            return q
        except Exception as e:
            return self._minimal_mcq(text, true_label, predicted_label, node_id, round_num, str(e), "error")

    # ── Near-miss MCQ ─────────────────────────────────────────────────────────

    def generate_near_miss(
        self,
        text: str,
        correct_label: str,
        wrong_label: str,
        score_margin: float,
        context_examples: list[dict],
        llm_fn: Callable[[str, str], str],
        node_id: str,
        round_num: int,
    ) -> Optional[NEXUSQuestion]:
        """
        Generate a near-miss MCQ when the model was correct but close (margin < 20%).
        Focuses on: why does the correct label win despite the surface ambiguity?
        """
        prompt = self._build_near_miss_prompt(text, correct_label, wrong_label,
                                              score_margin, context_examples)
        system = self._build_system("near_miss")

        try:
            raw = llm_fn(system, prompt)
            q = self._parse_mcq(raw, text, correct_label, wrong_label, node_id, round_num, "near_miss")
            return q
        except Exception as e:
            return self._minimal_mcq(text, correct_label, wrong_label, node_id, round_num, str(e), "near_miss")

    # ── Positive anchor MCQ ───────────────────────────────────────────────────

    def generate_positive(
        self,
        text: str,
        correct_label: str,
        wrong_label: str,
        confidence: float,
        llm_fn: Callable[[str, str], str],
        node_id: str,
        round_num: int,
    ) -> Optional[NEXUSQuestion]:
        """
        Generate a positive anchor MCQ from a high-confidence correct prediction.
        Encodes the clear signals that make this an unambiguous case.
        Prevents over-correction from error MCQs.
        """
        prompt = self._build_positive_prompt(text, correct_label, wrong_label, confidence)
        system = self._build_system("positive")

        try:
            raw = llm_fn(system, prompt)
            q = self._parse_mcq(raw, text, correct_label, "", node_id, round_num, "positive")
            return q
        except Exception as e:
            return self._minimal_mcq(text, correct_label, "", node_id, round_num, str(e), "positive")

    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_system(self, mcq_type: str) -> str:
        if mcq_type == "near_miss":
            return (
                f"You are NEXUS, a self-improving {self.config.task_name} classifier. "
                f"You just made a borderline prediction — you were correct, but only barely. "
                f"Your task is to generate a teaching case that explains WHY the correct answer "
                f"wins despite the surface ambiguity. Focus on the decisive signal that tips the "
                f"balance, and document the cognitive traps that make this case difficult."
            )
        elif mcq_type == "positive":
            return (
                f"You are NEXUS, a self-improving {self.config.task_name} classifier. "
                f"You just made a high-confidence correct prediction. "
                f"Your task is to generate an anchor teaching case that encodes the clear signals "
                f"that make this an unambiguous example. This prevents over-generalizing error "
                f"corrections. Be precise about WHAT makes this case clear-cut."
            )
        else:
            return (
                f"You are NEXUS, a self-improving {self.config.task_name} classifier. "
                f"You just made a classification error. Your task is to generate a clinical "
                f"MCQ teaching case with TWO distinct wrong-reasoning traps: "
                f"(1) the specific reasoning that led to YOUR error, and "
                f"(2) a different surface-reading trap that could also mislead. "
                f"Both traps must be genuinely different cognitive failure modes."
            )

    def _build_error_prompt(
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

Generate a complete MCQ teaching case with TWO distinct wrong-reasoning traps.
Trap 1 = what specifically caused YOUR error on this sentence.
Trap 2 = a DIFFERENT surface-reading cognitive trap (e.g., a misleading word, structural pattern,
         or clinical term that might lead someone astray via a completely different reasoning path).
Both traps must have the label "{predicted_label}" but represent genuinely different failure modes.

Respond ONLY with JSON in this exact format:
{{
  "correct_reasoning": "<2-3 sentences: what specific clinical signals confirm {true_label}? What is the decisive rule?>",
  "distractors": [
    {{
      "label": "{predicted_label}",
      "reasoning": "<1-2 sentences: what made {predicted_label} plausible in YOUR case? What signal did you over-weight?>",
      "correction": "<1-2 sentences: why is {predicted_label} definitively wrong? What specific signal overrides it?>",
      "error_type": "<one of: {taxonomy_str}>"
    }},
    {{
      "label": "{predicted_label}",
      "reasoning": "<DIFFERENT trap: what surface feature of this sentence (word, phrase, structure) could lead someone astray via a DIFFERENT reasoning path than trap 1?>",
      "correction": "<why that surface reading is wrong — what deeper signal it misses>",
      "error_type": "<error type — should differ from trap 1>"
    }}
  ],
  "difficulty": "<easy|medium|hard — how subtle is the distinction?>"
}}"""

    def _build_near_miss_prompt(
        self,
        text: str,
        correct_label: str,
        wrong_label: str,
        score_margin: float,
        context_examples: list[dict],
    ) -> str:
        error_taxonomy = self.config.get_error_taxonomy()
        taxonomy_str = ", ".join(f'"{e}"' for e in error_taxonomy)

        examples_str = ""
        if context_examples:
            lines = []
            for i, ex in enumerate(context_examples[:3]):
                bar = f"▲{ex['label']}" if ex["label"] == self.config.positive_label else f"▽{ex['label']}"
                sim = f"{ex.get('score', 0):.2f}"
                lines.append(f"  [{i+1}] {bar} (sim={sim}) \"{ex['text'][:100]}\"")
            examples_str = f"\n\nSimilar cases:\n" + "\n".join(lines)

        margin_pct = int(score_margin * 100)
        return f"""You correctly classified the following sentence as {correct_label}, but the score
margin was only {margin_pct}% — this was a borderline case. Generate a near-miss teaching case
explaining why {correct_label} wins despite the ambiguity.

Sentence: "{text}"{examples_str}

This case is DIFFICULT. The sentence has features that pull toward BOTH {correct_label} AND {wrong_label}.
Your task: document the deciding signal AND the two main cognitive traps that make this ambiguous.

Respond ONLY with JSON:
{{
  "correct_reasoning": "<2-3 sentences: what is the DECISIVE signal that tips this to {correct_label}? Why does it outweigh the ambiguous features?>",
  "distractors": [
    {{
      "label": "{wrong_label}",
      "reasoning": "<what feature of this sentence pulls toward {wrong_label}? This is a legitimate source of confusion>",
      "correction": "<why {correct_label} still wins — what overrides this ambiguous feature>",
      "error_type": "<one of: {taxonomy_str}>"
    }},
    {{
      "label": "{wrong_label}",
      "reasoning": "<a SECOND different feature that pulls toward {wrong_label}>",
      "correction": "<why it's still {correct_label} — the deeper signal that prevails>",
      "error_type": "<different error type>"
    }}
  ],
  "difficulty": "hard"
}}"""

    def _build_positive_prompt(
        self,
        text: str,
        correct_label: str,
        wrong_label: str,
        confidence: float,
    ) -> str:
        error_taxonomy = self.config.get_error_taxonomy()
        taxonomy_str = ", ".join(f'"{e}"' for e in error_taxonomy)

        conf_pct = int(confidence * 100)
        return f"""You correctly classified the following sentence as {correct_label} with {conf_pct}%
confidence — this is a clear, unambiguous case. Generate an anchor teaching case that encodes
exactly what makes this an obvious {correct_label} example.

Sentence: "{text}"

Your task: document the strong clear signals, AND the cognitive traps someone might try to use
to misclassify it (even though they don't work here — explaining WHY they fail is instructive).

Respond ONLY with JSON:
{{
  "correct_reasoning": "<2-3 sentences: what are the CLEAR, STRONG signals that make this unambiguously {correct_label}? Name the specific linguistic or clinical features>",
  "distractors": [
    {{
      "label": "{wrong_label}",
      "reasoning": "<could someone argue this is {wrong_label}? What weak feature might they point to?>",
      "correction": "<why that argument fails — why the strong {correct_label} signals dominate>",
      "error_type": "<one of: {taxonomy_str}>"
    }},
    {{
      "label": "{wrong_label}",
      "reasoning": "<a second weak argument for {wrong_label}>",
      "correction": "<why it also fails>",
      "error_type": "<error type>"
    }}
  ],
  "difficulty": "easy"
}}"""

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_mcq(
        self,
        raw: str,
        text: str,
        true_label: str,
        predicted_label: str,
        node_id: str,
        round_num: int,
        mcq_type: str,
    ) -> Optional[NEXUSQuestion]:
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if not m:
                return self._minimal_mcq(text, true_label, predicted_label,
                                         node_id, round_num, "json_parse_fail", mcq_type)
            try:
                d = json.loads(m.group())
            except Exception:
                return self._minimal_mcq(text, true_label, predicted_label,
                                         node_id, round_num, "json_extract_fail", mcq_type)

        distractors = [
            Distractor(
                label=dist.get("label", predicted_label or "OTHER"),
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
            mcq_type=mcq_type,
        )

    def _minimal_mcq(
        self,
        text: str,
        true_label: str,
        predicted_label: str,
        node_id: str,
        round_num: int,
        reason: str,
        mcq_type: str,
    ) -> NEXUSQuestion:
        """Fallback MCQ when LLM generation fails — minimal but still useful."""
        return NEXUSQuestion(
            text=text,
            node_id=node_id,
            round_num=round_num,
            correct_label=true_label,
            correct_reasoning=f"Correct label is {true_label}. (Generation failed: {reason})",
            distractors=[
                Distractor(
                    label=predicted_label or "OTHER",
                    reasoning=f"Model predicted {predicted_label}.",
                    correction=f"Correct answer is {true_label}.",
                    error_type="other",
                )
            ],
            difficulty="medium",
            predicted_label=predicted_label,
            mcq_type=mcq_type,
        )


# ─── MCQ Library ──────────────────────────────────────────────────────────────

# Difficulty multipliers for retrieval scoring
_DIFFICULTY_WEIGHT = {"hard": 1.5, "medium": 1.0, "easy": 0.7}
# MCQ type multipliers — error/near_miss are more informative than positive
_TYPE_WEIGHT = {"error": 1.2, "near_miss": 1.3, "positive": 0.8}


class MCQLibrary:
    """
    Stores MCQs with embeddings, supports semantic retrieval.

    v3.05 improvements:
      - Difficulty-weighted retrieval: hard MCQs score 1.5×, easy 0.7×
      - Type-weighted retrieval: near_miss 1.3×, error 1.2×, positive 0.8×
      - Cascading min_sim: tries 0.70, falls back to 0.55 if nothing found
      - Cross-node retrieval: supplement thin libraries from a global pool
      - Usage tracking: review_count and last_matched_round per MCQ

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
        self._embeddings = None  # Invalidate cache

    def _ensure_matrix(self) -> bool:
        """Build embedding matrix if needed. Returns False if empty."""
        if not self._questions:
            return False
        if self._embeddings is None:
            self._embeddings = np.array(
                [q.embedding for q in self._questions], dtype=np.float32
            )
        return True

    def retrieve(
        self,
        text: str,
        k: int = 3,
        min_sim: float = 0.70,
        round_num: int = -1,
        error_types_only: bool = False,
    ) -> list[tuple[float, NEXUSQuestion]]:
        """
        Retrieve K most similar MCQs, weighted by difficulty and type.

        Scoring: raw_cosine × difficulty_weight × type_weight
        No cascading fallback — min_sim is a hard gate. Biomedical text is
        dense enough that 0.55 cosine matches nearly everything, making a
        cascade useless and poisonous.

        error_types_only: if True, only return error/near_miss MCQs (not
        positive anchors). Used for cross-node retrieval to prevent NOT_ADE
        anchors from biasing classification of true ADE cases.

        Returns [(weighted_score, NEXUSQuestion), ...] sorted descending.
        Updates review_count and last_matched_round on matched MCQs.
        """
        if not self._ensure_matrix():
            return []

        vec = embed_one(text)
        raw_sims = self._embeddings @ vec  # cosine similarity (unit vectors)

        # Weighted scores
        weighted = []
        for idx, sim in enumerate(raw_sims):
            q = self._questions[idx]
            if error_types_only and q.mcq_type == "positive":
                continue   # skip positive anchors for cross-node use
            dw = _DIFFICULTY_WEIGHT.get(q.difficulty, 1.0)
            tw = _TYPE_WEIGHT.get(q.mcq_type, 1.0)
            weighted.append((float(sim) * dw * tw, idx, float(sim)))

        weighted.sort(key=lambda x: -x[0])

        # Hard min_sim gate — no cascade. Cascade at 0.55 retrieves everything
        # in the same biomedical domain and poisons the context.
        results = []
        for wscore, idx, raw_sim in weighted:
            if raw_sim < min_sim:
                break   # sorted descending, so once below threshold we're done
            results.append((wscore, self._questions[idx]))
            if len(results) >= k:
                break

        # Update usage tracking
        for score, q in results:
            q.review_count += 1
            if round_num >= 0:
                q.last_matched_round = round_num

        return results

    def format_for_context(
        self,
        text: str,
        k: int = 3,
        min_sim: float = 0.70,
        round_num: int = -1,
        global_pool: Optional["MCQLibrary"] = None,
    ) -> str:
        """
        Retrieve and format MCQs as a teaching context block for route injection.

        If local retrieval returns fewer than 2 results AND a global_pool is
        provided, supplement with cross-node MCQs (filtered to avoid duplicates).

        Returns a formatted string injected into route prompts.
        """
        # Always filter positive anchors from classification context.
        # Positive anchors encode "prototype of class X" — retrieved for a sentence
        # of the OPPOSITE class they inject wrong-class context and kill recall.
        # Error/near-miss MCQs are safe (they teach what to watch out for).
        retrieved = self.retrieve(text, k=k, min_sim=min_sim, round_num=round_num,
                                  error_types_only=True)

        # Cross-node fallback: supplement if local library is thin.
        if global_pool is not None and len(retrieved) < 2:
            needed = k - len(retrieved)
            seen_texts = {q.text for _, q in retrieved}
            global_results = global_pool.retrieve(
                text, k=needed + 3, min_sim=min_sim,
                round_num=round_num, error_types_only=True,
            )
            for score, q in global_results:
                if q.text not in seen_texts and len(retrieved) < k:
                    retrieved.append((score, q))
                    seen_texts.add(q.text)

        if not retrieved:
            return ""

        # Sort by score descending (global pool entries may be interleaved)
        retrieved.sort(key=lambda x: -x[0])

        # Build context block with type-aware headers
        type_counts = {"error": 0, "near_miss": 0, "positive": 0}
        for _, q in retrieved:
            type_counts[q.mcq_type] = type_counts.get(q.mcq_type, 0) + 1

        header = "=== NEXUS TEACHING CASES (learned from similar past cases) ==="
        blocks = [f"\n{header}"]
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

    def type_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for q in self._questions:
            counts[q.mcq_type] = counts.get(q.mcq_type, 0) + 1
        return counts

    def print_report(self, top_n: int = 5) -> None:
        if not self._questions:
            print("[MCQ] No questions in library yet.")
            return
        print(f"\n[MCQ] Library: {len(self._questions)} questions")
        error_dist = self.error_type_distribution()
        diff_dist = self.difficulty_distribution()
        type_dist = self.type_distribution()
        print(f"[MCQ] Types: {type_dist}")
        print(f"[MCQ] Difficulty: {diff_dist}")
        print(f"[MCQ] Top error types: {dict(list(error_dist.items())[:top_n])}")
        # Most-reviewed MCQs
        by_review = sorted(self._questions, key=lambda q: -q.review_count)
        if by_review and by_review[0].review_count > 0:
            print(f"[MCQ] Most reviewed: '{by_review[0].text[:60]}' (×{by_review[0].review_count})")
        if self._questions:
            recent = self._questions[-3:]
            print("[MCQ] Recent questions:")
            for q in recent:
                print(f"  R{q.round_num} {q.node_id} [{q.mcq_type}]: '{q.text[:60]}' "
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
        if lib._questions:
            lib._embeddings = np.array(
                [q.embedding for q in lib._questions], dtype=np.float32
            )
        return lib

    def __len__(self) -> int:
        return len(self._questions)

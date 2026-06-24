"""
principles.py
Prompt Engineering Principles Library for NEXUS.

After each accepted graft or refine, the LLM is asked to articulate WHAT
prompt engineering principle made the improvement work. These principles
accumulate across rounds and runs, and are injected into future synthesis
and refine prompts — so NEXUS becomes a better prompt engineer over time.

Unlike nuggets (reusable text fragments), principles are meta-knowledge:
generalizable rules about HOW to write effective prompts for this task.

Examples:
  P001: "Lead with explicit NOT_ADE criteria before ADE criteria to improve
         precision without sacrificing recall." (delta=+0.023, R3)
  P002: "Listing 2 concrete negative examples outperforms abstract negation
         rules for handling boundary cases." (delta=+0.011, R5)

The identity field is the agent's evolving self-concept — updated after
each meta-round by asking the LLM to reflect on what it has learned.
"""

import json
from copy import deepcopy


class PrinciplesStore:
    """
    Library of learned prompt engineering principles + agent identity.

    Principles are added after each accepted change (one LLM call per
    acceptance). The identity is updated after each meta-round (one call
    per meta-round). Both are injected into synthesis/refine prompts so
    the agent accumulates domain expertise over time.
    """

    _DEFAULT_IDENTITY = (
        "I am NEXUS, a clinical pharmacovigilance classifier in early training. "
        "I am learning to distinguish Adverse Drug Events (ADE) from non-ADE "
        "clinical text. My current focus is on improving precision — I tend to "
        "over-classify borderline cases as ADE when causal evidence is weak."
    )

    def __init__(self, principles: list = None, path: str = None,
                 identity: str = None):
        self.principles: list = deepcopy(principles or [])
        self.path = path
        self.identity: str = identity or self._DEFAULT_IDENTITY

    # ------------------------------------------------------------------
    # Principle management
    # ------------------------------------------------------------------

    def add_principle(self, text: str, source_round: int, delta_f1: float,
                      source: str = "") -> str:
        """
        Add a new principle after an accepted change.
        Returns the auto-generated principle ID.
        """
        pid = f"P{len(self.principles) + 1:03d}"
        self.principles.append({
            "id": pid,
            "text": text.strip(),
            "source_round": source_round,
            "source": source,
            "delta_f1": round(delta_f1, 4),
        })
        if self.path:
            self.save(self.path)
        return pid

    def update_identity(self, new_identity: str):
        """Replace the agent's self-concept with an evolved version."""
        self.identity = new_identity.strip()
        if self.path:
            self.save(self.path)

    # ------------------------------------------------------------------
    # Synthesis prompt helper
    # ------------------------------------------------------------------

    def catalogue_for_synthesis(self, top_n: int = 5) -> str:
        """
        Return the top_n principles (highest delta_F1) as a formatted block
        for injection into synthesis and refine prompts.
        """
        if not self.principles:
            return "  (none yet — principles accumulate after accepted changes)"
        sorted_p = sorted(self.principles, key=lambda p: -p.get("delta_f1", 0))
        lines = []
        for p in sorted_p[:top_n]:
            lines.append(
                f"  [{p['id']}] (Δ={p['delta_f1']:+.4f}, R{p['source_round']}): "
                f"{p['text']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "version": 1,
                "identity": self.identity,
                "principles": self.principles,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PrinciplesStore":
        """Load from file; returns a fresh store if the file is missing."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(
                principles=data.get("principles", []),
                identity=data.get("identity"),
                path=path,
            )
        except FileNotFoundError:
            store = cls(path=path)
            store.save(path)
            return store

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict:
        return {
            "identity": self.identity,
            "total_principles": len(self.principles),
            "principles": self.principles,
        }

    def print_report(self):
        print("\n=== Principles Library ===")
        print(f"  Agent identity: {self.identity[:120]}...")
        if not self.principles:
            print("  (no principles learned yet)")
            return
        sorted_p = sorted(self.principles, key=lambda p: -p.get("delta_f1", 0))
        for p in sorted_p:
            print(f"  [{p['id']}] Δ={p['delta_f1']:+.4f}  "
                  f"R{p['source_round']}  {p['text'][:100]}")

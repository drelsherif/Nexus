"""
nuggets.py
Nugget store for NEXUS prompt optimization.

A "nugget" is a small, named, reusable prompt fragment stored in a JSON file.
Tree node prompts can reference nuggets via [NUGGET_ID] placeholders:

    "[EXPERT_ROLE] This sentence uses a 'drug-induced' pattern... [JSON_SCHEMA]"

At inference time, assemble() expands placeholders to full text before
the prompt is sent to the LLM. The full assembled text is what reaches
the API, so classification quality is unaffected.

The research value comes from:
  1. Compressed storage — tree JSON and synthesis prompts are shorter.
  2. Controlled vocabulary — the synthesis LLM reuses proven fragments
     rather than regenerating them, reducing output token cost.
  3. Attribution — we can track which nuggets appear in accepted vs
     rejected grafts and measure their F1 correlation over time.
  4. Prompt evolution — compress() can automatically compress any existing
     full-text prompt by substituting known nuggets, making historical
     nodes cheaper to represent in future synthesis context.

Nugget lifecycle:
  - Seed nuggets are created from the initial tree prompts at startup.
  - The synthesis prompt includes a nugget catalogue and asks the LLM to
    reference nuggets by ID when writing new node prompts.
  - When a graft is accepted, record_usage() attributes the round's F1
    to every nugget referenced in that node's prompt template.
  - Periodically (or on demand), compress() can rewrite full-text prompts
    to use placeholders, shrinking the synthesis context each round.
"""

import json
import re
from copy import deepcopy


# Regex to find [NUGGET_ID] placeholders in a prompt template
PLACEHOLDER_RE = re.compile(r'\[([A-Z0-9_]+)\]')

# --- Seed nuggets extracted from the initial tree ---
# These cover the recurring phrases across all four seed node prompts.
# Stored verbatim so assemble() reproduces the original prompt exactly.
_SEED_NUGGETS: dict = {
    "EXPERT_ROLE": {
        "text": "You are a pharmacovigilance expert.",
        "usage_count": 0,
        "accepted_count": 0,
        "core": False,
        "f1_history": [],
        "source": "seed",
    },
    "ADE_DEFINITION": {
        "text": (
            "An Adverse Drug Event (ADE) is a harmful or unintended effect "
            "plausibly caused by a drug."
        ),
        "usage_count": 0,
        "accepted_count": 0,
        "core": False,
        "f1_history": [],
        "source": "seed",
    },
    "JSON_SCHEMA": {
        "text": (
            'Respond ONLY as JSON: {"classification": "ADE" or "NOT_ADE", '
            '"confidence": "high|medium|low", "rationale": "<one sentence>"}.'
        ),
        "usage_count": 0,
        "accepted_count": 0,
        "core": False,
        "f1_history": [],
        "source": "seed",
    },
    "CLASSIFY_ADE_UNLESS": {
        "text": (
            "Classify ADE unless the sentence explicitly negates the outcome "
            "or describes a desired therapeutic effect."
        ),
        "usage_count": 0,
        "accepted_count": 0,
        "core": False,
        "f1_history": [],
        "source": "seed",
    },
}

# Nuggets appearing in this many accepted changes are promoted to CORE status
CORE_THRESHOLD = 3


def _rough_tokens(text: str) -> int:
    """Rough token count: 1 token ≈ 4 characters (GPT/Gemini rule of thumb)."""
    return max(1, len(text) // 4)


class NuggetStore:
    """
    Manages a library of reusable prompt fragments.

    Parameters
    ----------
    nuggets : dict, optional
        Initial nugget dict (id -> nugget dict). Defaults to _SEED_NUGGETS.
    path : str, optional
        JSON file path. If given, save() writes here automatically.
    """

    def __init__(self, nuggets: dict = None, path: str = None):
        self.nuggets: dict = deepcopy(nuggets or _SEED_NUGGETS)
        self.path = path

    # ------------------------------------------------------------------
    # Core prompt assembly / compression
    # ------------------------------------------------------------------

    def assemble(self, template: str) -> str:
        """Replace every [NUGGET_ID] placeholder with its full text."""
        def _replace(m):
            nid = m.group(1)
            return self.nuggets[nid]["text"] if nid in self.nuggets else m.group(0)
        return PLACEHOLDER_RE.sub(_replace, template)

    def compress(self, prompt: str) -> str:
        """
        Replace known nugget text in `prompt` with [NUGGET_ID] placeholders.
        Longer nuggets are substituted first to avoid partial replacements.
        Returns the compressed template (may still contain un-nuggetised text).
        """
        result = prompt
        for nid, n in sorted(self.nuggets.items(), key=lambda x: -len(x[1]["text"])):
            result = result.replace(n["text"], f"[{nid}]")
        return result

    def tokens_saved(self, template: str) -> int:
        """
        Estimate tokens saved versus sending the fully assembled prompt.
        Positive means the template is cheaper to store/transmit than the
        expanded version.
        """
        assembled = self.assemble(template)
        return _rough_tokens(assembled) - _rough_tokens(template)

    # ------------------------------------------------------------------
    # Nugget management
    # ------------------------------------------------------------------

    def add_nugget(self, nid: str, text: str, source: str = "learned") -> bool:
        """
        Add a new nugget. Returns True if added, False if skipped.
        Skipped when: already exists, text too short, or text is a substring
        of an existing nugget (wouldn't save any tokens).
        """
        if nid in self.nuggets:
            return False
        text = text.strip()
        if len(text) < 20:
            return False
        for n in self.nuggets.values():
            if text in n["text"]:
                return False
        self.nuggets[nid] = {
            "text": text,
            "usage_count": 0,
            "accepted_count": 0,
            "core": False,
            "f1_history": [],
            "source": source,
        }
        if self.path:
            self.save(self.path)
        return True

    def record_accepted(self, template: str):
        """
        Call after any accepted graft or refine. Increments accepted_count for
        every nugget referenced in the template, and promotes to CORE when
        accepted_count >= CORE_THRESHOLD.
        Returns list of newly promoted nugget IDs.
        """
        promoted = []
        for nid in PLACEHOLDER_RE.findall(template):
            if nid not in self.nuggets:
                continue
            n = self.nuggets[nid]
            n.setdefault("accepted_count", 0)
            n["accepted_count"] += 1
            if not n.get("core") and n["accepted_count"] >= CORE_THRESHOLD:
                n["core"] = True
                promoted.append(nid)
        if promoted and self.path:
            self.save(self.path)
        return promoted

    def record_usage(self, template: str, f1: float):
        """
        After a round's eval, attribute the achieved F1 to every nugget
        referenced in `template`. Call with each active node's prompt.
        """
        for nid in PLACEHOLDER_RE.findall(template):
            if nid in self.nuggets:
                self.nuggets[nid]["usage_count"] += 1
                self.nuggets[nid]["f1_history"].append(round(f1, 4))

    # ------------------------------------------------------------------
    # Synthesis prompt helpers
    # ------------------------------------------------------------------

    def catalogue_for_synthesis(self) -> str:
        """
        A compact, token-efficient catalogue of available nuggets for
        inclusion in the synthesis prompt. CORE nuggets are marked with ★
        and listed first — the synthesis LLM should prefer them.
        """
        lines = []
        # CORE nuggets first
        for nid, n in sorted(self.nuggets.items(),
                              key=lambda x: (not x[1].get("core"), x[0])):
            preview = n["text"][:70].replace("\n", " ")
            if len(n["text"]) > 70:
                preview += "..."
            tok   = _rough_tokens(n["text"])
            badge = " ★CORE" if n.get("core") else ""
            lines.append(
                f'  [{nid}]{badge} (~{tok} tok, used {n["usage_count"]}×, '
                f'accepted {n.get("accepted_count",0)}×): {preview}'
            )
        return "\n".join(lines)

    @property
    def core_nuggets(self) -> list[str]:
        """IDs of all nuggets that have reached CORE status."""
        return [nid for nid, n in self.nuggets.items() if n.get("core")]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"version": 1, "nuggets": self.nuggets}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NuggetStore":
        """Load from file; falls back to seed nuggets if the file is missing."""
        try:
            with open(path) as f:
                data = json.load(f)
            return cls(nuggets=data["nuggets"], path=path)
        except FileNotFoundError:
            store = cls(path=path)
            store.save(path)  # create the file immediately
            return store

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_dict(self) -> dict:
        return {
            nid: {
                "usage_count": n["usage_count"],
                "accepted_count": n.get("accepted_count", 0),
                "core": n.get("core", False),
                "avg_f1": (
                    round(sum(n["f1_history"]) / len(n["f1_history"]), 4)
                    if n["f1_history"] else None
                ),
                "tokens_in_full": _rough_tokens(n["text"]),
                "source": n["source"],
            }
            for nid, n in self.nuggets.items()
        }

    def print_report(self):
        print("\n=== Nugget Store Report ===")
        core_ids = self.core_nuggets
        if core_ids:
            print(f"  CORE nuggets ({len(core_ids)}): {', '.join(core_ids)}")
        for nid, n in self.nuggets.items():
            avg_f1 = (
                f"{sum(n['f1_history']) / len(n['f1_history']):.3f}"
                if n["f1_history"] else "n/a"
            )
            badge = " ★" if n.get("core") else ""
            print(f"  [{nid}]{badge} used={n['usage_count']}  "
                  f"accepted={n.get('accepted_count',0)}  avg_f1={avg_f1}  "
                  f"src={n['source']}  tok≈{_rough_tokens(n['text'])}")

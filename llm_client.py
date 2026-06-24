"""
llm_client.py
Thin wrapper around the Gemini API for all four call types NEXUS needs:

  1. classify()       — run a node's specialist prompt against one sentence
  2. synthesize()     — propose a new tree branch from a batch of errors
  3. refine_prompt()  — improve an existing node's prompt given its errors
                        (self-optimization step — core research contribution)
  4. extract_nuggets()— identify reusable fragments from an accepted prompt
                        to grow the nugget library (token-compression step)

GeminiClient (real API) and MockClient (deterministic, offline) share the
same interface so nexus_run.py doesn't need to care which one it's using.

Key features vs baseline:
  - Retry with exponential backoff on 429 / 500 / 503
  - Token usage reported via TokenTracker on every call
  - Latency measured and returned so DebugLogger can record it
  - synthesize() and refine_prompt() inject the nugget catalogue so the
    LLM can reference existing fragments by [NUGGET_ID] — reducing output
    tokens over time as the nugget library grows
  - thinking_config disabled on all calls (flash-lite default, explicit
    here for reproducibility and cost certainty)
"""

import json
import os
import random
import time


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT_TEMPLATE = """\
You are a clinical NLP expert improving a decision tree classifier.

MISCLASSIFIED CASES (round {round_num}):
{cases_block}

CURRENT TREE NODES: {node_list}
{nugget_section}
Propose ONE new specialist branch node.

TRIGGER_CONDITION RULES — read carefully:
  ALLOWED variables (these are the ONLY ones that exist — do not invent others):
    has_induced, has_associated, has_toxicity, has_adverse, has_developed,
    has_following, has_reaction, has_report, has_negation, has_short, has_drug_name
  ALLOWED operators: and, or, not
  NO other identifiers, functions, or attributes are permitted.
  Example valid conditions:
    has_toxicity and not has_negation
    has_report and has_drug_name and not has_induced
    (has_reaction or has_adverse) and has_drug_name

In the "prompt" field, reuse nuggets via [NUGGET_ID] placeholders where they fit.
Write custom text only for the parts unique to this new node. Shorter is better.

Respond with ONLY valid JSON (no markdown fences):
{{
  "error_pattern": "dominant pattern in 1 sentence",
  "new_node": {{
    "id": "NODE_<DESCRIPTIVE_NAME>",
    "description": "what this node handles",
    "trigger_condition": "expression using ONLY the allowed variables above",
    "prompt": "specialist prompt using [NUGGET_ID] placeholders + any custom text"
  }}
}}"""

REPAIR_CONDITION_TEMPLATE = """\
A trigger_condition you proposed was rejected because it uses a variable that does not exist.

REJECTED CONDITION: {bad_condition}
INVALID VARIABLE(S): {bad_vars}

THE ONLY ALLOWED VARIABLES ARE:
  has_induced, has_associated, has_toxicity, has_adverse, has_developed,
  has_following, has_reaction, has_report, has_negation, has_short, has_drug_name

Rewrite the trigger_condition to express the same intent using ONLY allowed variables.
Operators allowed: and, or, not, parentheses.

Respond with ONLY valid JSON (no markdown fences):
{{"trigger_condition": "corrected expression using only the allowed variables"}}"""

_NUGGET_SECTION_TEMPLATE = """
AVAILABLE NUGGETS — reference by ID to save tokens:
{catalogue}
"""

REFINE_PROMPT_TEMPLATE = """\
You are a clinical NLP expert optimizing a single classification node.

NODE ID:    {node_id}
DESCRIPTION: {description}

CURRENT PROMPT (full assembled text):
{assembled_prompt}

CASES THIS NODE GOT WRONG:
{error_cases}
{nugget_section}
Task:
1. Identify what pattern the current prompt fails on (1-2 sentences).
2. Write an improved prompt that fixes the error pattern.
3. Keep it concise — shorter prompts that maintain accuracy save tokens.
4. Use [NUGGET_ID] placeholders for any text matching available nuggets.
5. Do NOT change the node's trigger_condition — only the prompt.

Respond with ONLY valid JSON (no markdown fences):
{{
  "analysis": "what the current prompt misses or gets wrong",
  "improved_prompt": "new prompt using [NUGGET_ID] placeholders + custom text"
}}"""

EXTRACT_NUGGETS_TEMPLATE = """\
You are building a reusable prompt-fragment library for a clinical NLP classifier.

NEWLY ACCEPTED PROMPT:
{prompt}

EXISTING NUGGETS (do NOT re-extract these — they are already captured):
{catalogue}

Identify ALL fragments in the accepted prompt that:
  - Are at least 25 characters long
  - Would be genuinely reusable across DIFFERENT specialist nodes
  - Are NOT already covered (even partially) by an existing nugget
  - Carry meaningful domain content, not just punctuation or filler

There is NO limit on how many nuggets you propose — extract every useful fragment.
Each nugget should be a self-contained, standalone phrase or sentence.
Assign each a concise ALL_CAPS_UNDERSCORE ID that describes its clinical role.
If nothing new is worth extracting, return an empty list.

Respond with ONLY valid JSON (no markdown fences):
{{"new_nuggets": [
  {{"id": "NUGGET_ID_1", "text": "exact fragment text"}},
  {{"id": "NUGGET_ID_2", "text": "another fragment"}},
  ...
]}}"""

_ERROR_BUFFER_SECTION = """
RECURRING ERRORS (accumulated across recent rounds — these are SYSTEMATIC failures):
{buffer_block}
"""

_PRINCIPLES_SECTION_TEMPLATE = """
LEARNED PROMPT ENGINEERING PRINCIPLES (apply these — they come from accepted improvements):
{principles}
"""

_IDENTITY_SECTION_TEMPLATE = """
YOUR CURRENT EXPERTISE LEVEL (self-concept built from this run's experience):
{identity}
"""

_REJECTED_PROPOSALS_SECTION = """
RECENTLY REJECTED PROPOSALS (do NOT repeat these — try a different approach):
{rejected_block}
"""

EXTRACT_PRINCIPLES_TEMPLATE = """\
A prompt improvement was accepted — it raised F1 by {delta_f1:+.4f}.

ORIGINAL PROMPT:
{original_prompt}

IMPROVED PROMPT:
{improved_prompt}

What generalizable prompt engineering principle made this improvement effective?
State it as a single concrete rule that could guide future prompt improvements.
The principle should be specific enough to be actionable but general enough to
apply across different specialist node types in clinical NLP classification.

Examples of good principles:
  "Leading with explicit NOT_ADE exclusion criteria before ADE criteria improves precision."
  "Short-sentence fragments need drug+harm co-occurrence check before ADE classification."
  "Listing 2 concrete counter-examples outperforms abstract negation rules."

Respond with ONLY valid JSON (no markdown fences):
{{"principle": "Your generalizable rule in 1-2 sentences"}}"""

EVOLVE_IDENTITY_TEMPLATE = """\
You are NEXUS, a self-optimizing clinical NLP classifier. Based on your performance
history and what you have learned, update your self-concept.

CURRENT IDENTITY:
{current_identity}

PERFORMANCE HISTORY (F1 per round): {f1_history}
STRONGEST NODE: {strongest_node} (F1={best_node_f1:.3f})
WEAKEST AREA: {weakest_node} (F1={worst_node_f1:.3f})
TOTAL ACCEPTED IMPROVEMENTS: {accepted_count}
PRINCIPLES LEARNED: {principle_count}
LATEST STRATEGIC INSIGHT: {last_meta_analysis}

Rewrite your identity statement to reflect what you have genuinely learned.
Keep it under 3 sentences. Address: (1) what you now understand about this
classification task that you did not at the start, (2) your current strongest
capability, (3) the specific gap you are still working to close.

Respond with ONLY valid JSON (no markdown fences):
{{"identity": "Updated self-concept in 2-3 sentences"}}"""

META_SYNTHESIZE_TEMPLATE = """\
You are a clinical NLP expert conducting a STRATEGIC REVIEW of a self-growing decision tree.

This is a META-ROUND. Instead of reacting to a single batch of errors, you are looking at
the full run history to identify structural improvements to the tree architecture.

RUN HISTORY (F1 per round):
{f1_history}

CURRENT TREE NODES:
{node_list}

PER-NODE PERFORMANCE (last round):
{per_node_stats}

RECURRING ERROR PATTERNS (accumulated across all rounds):
{error_buffer_block}

ACCEPTED CHANGES SO FAR:
{accepted_changes}

NUGGETS IN LIBRARY:
{nugget_section}

Based on this full picture, propose ONE high-value structural improvement. This could be:
  a) A new specialist node targeting a persistent error pattern
  b) A refinement of an underperforming node's prompt
  c) A suggestion to retire a dead node (one that routes 0 cases consistently)

TRIGGER_CONDITION RULES (same as always):
  ALLOWED variables ONLY: has_induced, has_associated, has_toxicity, has_adverse,
  has_developed, has_following, has_reaction, has_report, has_negation, has_short, has_drug_name
  ALLOWED operators: and, or, not

Respond with ONLY valid JSON (no markdown fences):
{{
  "strategic_analysis": "what pattern or structural issue this addresses",
  "action_type": "new_node" | "refine_node" | "retire_node",
  "new_node": {{
    "id": "NODE_<NAME>",
    "description": "what this handles",
    "trigger_condition": "valid expression",
    "prompt": "specialist prompt using [NUGGET_ID] placeholders"
  }}
}}
(For refine_node or retire_node, "new_node" can be null — include a "target_node_id" field instead.)"""

RETIRE_NODE_TEMPLATE = """\
A node in the NEXUS decision tree has been flagged for retirement because it has
routed very few cases over multiple rounds and shows poor performance.

NODE TO RETIRE: {node_id}
DESCRIPTION: {description}
CURRENT PROMPT: {prompt}
PERFORMANCE: routed {routes} cases over last {rounds} rounds, avg F1={avg_f1:.3f}

Propose how to merge this node's intent back into the ROOT node's prompt.

Respond with ONLY valid JSON (no markdown fences):
{{
  "rationale": "why retiring this node is safe",
  "root_prompt_addition": "text to append to the ROOT prompt to cover cases this node handled"
}}"""


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE = {429, 500, 503}
_MAX_RETRIES = 6
_BACKOFF_BASE = 5.0  # seconds — longer base helps on free-tier 503 spikes
_JITTER = 2.0        # random jitter added to each wait to avoid thundering-herd


def _with_retry(fn, *args, **kwargs):
    """Exponential backoff with jitter on retryable HTTP errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status is None:
                msg = str(e)
                for code in _RETRYABLE:
                    if str(code) in msg:
                        status = code
                        break
            if status in _RETRYABLE and attempt < _MAX_RETRIES - 1:
                wait = _BACKOFF_BASE * (2 ** attempt) + random.uniform(0, _JITTER)
                print(f"  [retry {attempt + 1}/{_MAX_RETRIES - 1}] "
                      f"HTTP {status} — waiting {wait:.0f}s before retrying...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


def _extract_tokens(resp) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a Gemini response."""
    try:
        um = resp.usage_metadata
        return (
            getattr(um, "prompt_token_count", 0) or 0,
            getattr(um, "candidates_token_count", 0) or 0,
        )
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# GeminiClient — real API
# ---------------------------------------------------------------------------

class GeminiClient:
    """Real Gemini 2.5 Flash Lite client."""

    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash",
                 tracker=None):
        from google import genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.tracker = tracker

    def _gen(self, contents: str, system: str = None,
             temperature: float = 0.0, max_tokens: int = 150) -> tuple:
        """
        Raw generate call. Returns (parsed_dict_or_None, input_tokens, output_tokens, latency_ms).
        """
        from google.genai import types
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        if system:
            cfg.system_instruction = system

        t0 = time.time()
        resp = _with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=contents,
            config=cfg,
        )
        latency_ms = (time.time() - t0) * 1000
        inp, out = _extract_tokens(resp)
        return _safe_json(resp.text, fallback=None), inp, out, latency_ms

    def classify(self, text: str, node_prompt: str) -> tuple[dict, int, int, float]:
        """
        Classify `text` with `node_prompt` as system instruction.
        Returns (result_dict, input_tokens, output_tokens, latency_ms).
        """
        result, inp, out, lat = self._gen(
            contents=text, system=node_prompt,
            temperature=0.0, max_tokens=120,
        )
        if self.tracker:
            self.tracker.record_classify(inp, out)
        if result is None:
            result = {"classification": "NOT_ADE", "confidence": "low",
                      "rationale": "parse_error"}
        return result, inp, out, lat

    def synthesize(self, round_num: int, error_cases: list, node_list: list,
                   nugget_store=None, error_buffer=None,
                   principles_store=None, rejected_proposals=None) -> tuple[dict, int, int, float]:
        """
        Propose a new branch node from misclassified cases.
        error_buffer: iterable of prior-round errors for cross-round context.
        principles_store: PrinciplesStore with learned prompt engineering rules.
        rejected_proposals: list of recently rejected proposals to avoid repeating.
        Returns (proposal_dict_or_None, inp, out, latency_ms).
        """
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} NODE={c['node']} "
            f"TEXT={c['text']}"
            for i, c in enumerate(error_cases[:8])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(
                catalogue=nugget_store.catalogue_for_synthesis()
            ) if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            buf_list = list(error_buffer)[-20:]
            if buf_list:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} "
                    f"TEXT={c['text'][:80]}"
                    for c in buf_list
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = _PRINCIPLES_SECTION_TEMPLATE.format(
                principles=principles_store.catalogue_for_synthesis()
            ) + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
        rejected_section = ""
        if rejected_proposals:
            recent = rejected_proposals[-5:]
            rej_block = "\n".join(
                f"  R{r['round']}: node={r['node_id']!r} "
                f"cond={r['condition']!r} → {r['reason']}"
                for r in recent
            )
            rejected_section = _REJECTED_PROPOSALS_SECTION.format(
                rejected_block=rej_block)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            round_num=round_num,
            cases_block=cases_block,
            node_list=json.dumps(node_list),
            nugget_section=(nugget_section + buffer_section
                            + principles_section + rejected_section),
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.3, max_tokens=700,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def refine_prompt(self, node: dict, error_cases: list,
                      nugget_store=None, error_buffer=None,
                      principles_store=None) -> tuple[dict, int, int, float]:
        """
        Ask the LLM to improve `node`'s prompt given cases it got wrong.
        error_buffer: iterable of prior-round errors for cross-round context.
        principles_store: PrinciplesStore with learned prompt engineering rules.
        Returns (proposal_dict_or_None, inp, out, latency_ms).
        """
        assembled = (
            nugget_store.assemble(node["prompt"]) if nugget_store
            else node["prompt"]
        )
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} TEXT={c['text']}"
            for i, c in enumerate(error_cases[:6])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(
                catalogue=nugget_store.catalogue_for_synthesis()
            ) if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            node_buf = [e for e in list(error_buffer)[-30:]
                        if e.get("node") == node["id"]][-8:]
            if node_buf:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} TEXT={c['text'][:80]}"
                    for c in node_buf
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = _PRINCIPLES_SECTION_TEMPLATE.format(
                principles=principles_store.catalogue_for_synthesis()
            ) + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
        prompt = REFINE_PROMPT_TEMPLATE.format(
            node_id=node["id"],
            description=node.get("description", ""),
            assembled_prompt=assembled,
            error_cases=cases_block,
            nugget_section=nugget_section + buffer_section + principles_section,
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.3, max_tokens=500,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def meta_synthesize(self, f1_history: list, node_list: list,
                        per_node_stats: dict, error_buffer,
                        accepted_changes: list,
                        nugget_store=None) -> tuple[dict, int, int, float]:
        """Strategic meta-round: full-history tree restructuring proposal."""
        f1_str = "  " + " → ".join(f"R{i}: {f:.3f}" for i, f in enumerate(f1_history))
        pn_str = "\n".join(
            f"  {nid}: F1={s.get('f1',0):.3f}  routed={s.get('count',0)}"
            for nid, s in per_node_stats.items()
        )
        buf_list = list(error_buffer)[-30:]
        buf_block = "\n".join(
            f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} TEXT={c['text'][:80]}"
            for c in buf_list
        ) or "  (none)"
        changes_str = "\n".join(f"  R{i+1}: {a}" for i, a in enumerate(accepted_changes))
        nugget_section = (nugget_store.catalogue_for_synthesis()
                          if nugget_store else "(none)")
        prompt = META_SYNTHESIZE_TEMPLATE.format(
            f1_history=f1_str,
            node_list=json.dumps(node_list),
            per_node_stats=pn_str,
            error_buffer_block=buf_block,
            accepted_changes=changes_str or "  (none yet)",
            nugget_section=nugget_section,
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.4, max_tokens=800,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def retire_node(self, node: dict, routes: int, rounds: int,
                    avg_f1: float) -> tuple[dict, int, int, float]:
        """Propose how to retire an underperforming node."""
        prompt = RETIRE_NODE_TEMPLATE.format(
            node_id=node["id"],
            description=node.get("description", ""),
            prompt=node.get("prompt", "")[:300],
            routes=routes, rounds=rounds, avg_f1=avg_f1,
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.2, max_tokens=300,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def extract_principles(self, original_prompt: str, improved_prompt: str,
                           delta_f1: float) -> tuple[str | None, int, int, float]:
        """
        After an accepted change, ask the LLM what prompt engineering principle
        made the improvement effective.
        Returns (principle_text_or_None, inp, out, latency_ms).
        """
        prompt = EXTRACT_PRINCIPLES_TEMPLATE.format(
            delta_f1=delta_f1,
            original_prompt=original_prompt[:400],
            improved_prompt=improved_prompt[:400],
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.2, max_tokens=150,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        principle = None
        if isinstance(result, dict):
            principle = result.get("principle")
        return principle, inp, out, lat

    def evolve_identity(self, current_identity: str, f1_history: list,
                        per_node_stats: dict, accepted_count: int,
                        principle_count: int,
                        last_meta_analysis: str = "") -> tuple[str | None, int, int, float]:
        """
        After a meta-round, ask the LLM to update NEXUS's self-concept
        based on what it has learned about this classification task.
        Returns (new_identity_or_None, inp, out, latency_ms).
        """
        f1_str = " → ".join(f"R{i}:{f:.3f}" for i, f in enumerate(f1_history))
        if per_node_stats:
            best_nid  = max(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            worst_nid = min(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            best_f1   = per_node_stats[best_nid].get("f1", 0.0)
            worst_f1  = per_node_stats[worst_nid].get("f1", 0.0)
        else:
            best_nid = worst_nid = "N/A"
            best_f1 = worst_f1 = 0.0
        prompt = EVOLVE_IDENTITY_TEMPLATE.format(
            current_identity=current_identity,
            f1_history=f1_str,
            strongest_node=best_nid,
            best_node_f1=best_f1,
            weakest_node=worst_nid,
            worst_node_f1=worst_f1,
            accepted_count=accepted_count,
            principle_count=principle_count,
            last_meta_analysis=last_meta_analysis[:200] or "(none yet)",
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.3, max_tokens=200,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        identity = None
        if isinstance(result, dict):
            identity = result.get("identity")
        return identity, inp, out, lat

    def extract_nuggets(self, prompt_text: str,
                        nugget_store=None) -> tuple[list, int, int, float]:
        """
        Ask the LLM to identify reusable fragments in `prompt_text` worth
        adding to the nugget library.
        Returns (list_of_nugget_dicts, inp, out, latency_ms).
        """
        catalogue = nugget_store.catalogue_for_synthesis() if nugget_store else "(none)"
        prompt = EXTRACT_NUGGETS_TEMPLATE.format(
            prompt=prompt_text,
            catalogue=catalogue,
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.2, max_tokens=400,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        nuggets = []
        if isinstance(result, dict):
            nuggets = result.get("new_nuggets", []) or []
        return nuggets, inp, out, lat

    def repair_condition(self, bad_condition: str,
                         bad_vars: list) -> tuple[str | None, int, int, float]:
        """
        When synthesize() returns a trigger_condition containing invalid
        variable names, call this to ask the model to fix it in-place.
        Returns (repaired_condition_or_None, inp, out, latency_ms).
        """
        prompt = REPAIR_CONDITION_TEMPLATE.format(
            bad_condition=bad_condition,
            bad_vars=", ".join(bad_vars),
        )
        result, inp, out, lat = self._gen(
            contents=prompt, temperature=0.1, max_tokens=100,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        if isinstance(result, dict) and result.get("trigger_condition"):
            return result["trigger_condition"], inp, out, lat
        return None, inp, out, lat


# ---------------------------------------------------------------------------
# AIHubClient — Northwell Health enterprise AI Hub
# ---------------------------------------------------------------------------

class AIHubClient:
    """
    Client for Northwell's internal AI Hub API (https://api.ai.northwell.edu).

    Supports all models available on the platform:
      Gemini:  gemini-2.5-flash-lite, gemini-2.5-flash, gemini-2.5-pro
      Claude:  claude-haiku-4.5, claude-sonnet-4.5, claude-opus-4.5, claude-opus-4.6
      OpenAI:  gpt-5-nano, gpt-5-mini, gpt-5, gpt-5.1, gpt-5.2, o3, o4-mini

    Authentication: X-API-Key header.
    Prompts and context are base64-encoded before sending.

    Recommended model split for NEXUS:
      classify → claude-haiku-4.5  (fast, cheap, high-volume)
      synthesize/refine → claude-sonnet-4.5 or claude-opus-4.6  (best reasoning)

    Token counts are estimated from text length since the API does not return
    usage metadata — mark as estimated in the tracker summary.
    """

    BASE_URL = "https://api.ai.northwell.edu"

    def __init__(self, api_key: str, ad_object_id: str,
                 classify_model: str = "claude-haiku-4.5",
                 synth_model: str = "claude-sonnet-4.5",
                 tracker=None):
        """
        Parameters
        ----------
        api_key : str
            AI Hub API key (X-API-Key header).
        ad_object_id : str
            Your Active Directory Object ID (UUID format).
        classify_model : str
            Model used for per-case classification (high-volume calls).
        synth_model : str
            Model used for synthesis, refinement, extraction (low-volume, complex).
        tracker : TokenTracker, optional
            Token usage tracker (counts estimated from text length).
        """
        import requests as _req
        self._requests = _req
        self.api_key      = api_key
        self.ad_object_id = ad_object_id
        self.classify_model = classify_model
        self.synth_model    = synth_model
        self.tracker = tracker
        self._session = _req.Session()
        self._session.headers.update({
            "X-API-Key":    api_key,
            "Content-Type": "application/json",
        })

    def _b64(self, text: str) -> str:
        import base64
        return base64.b64encode(text.encode("utf-8")).decode("ascii")

    def _call(self, prompt: str, context: str = None, model: str = None,
              temperature: float = 0.0, max_tokens: int = 150) -> tuple:
        """
        POST to /generative. Returns (parsed_result, inp_tokens, out_tokens, latency_ms).
        Token counts are estimated (~1 token per 4 chars).
        """
        payload = {
            "ad_object_id": self.ad_object_id,
            "models":        [model or self.classify_model],
            "prompt":        self._b64(prompt),
            "advanced": {
                "temperature": temperature,
                "max_tokens":  max_tokens,
            },
        }
        if context:
            payload["context"] = self._b64(context)

        def _do_request():
            r = self._session.post(
                f"{self.BASE_URL}/generative",
                json=payload,
                timeout=90,
            )
            # Map HTTP errors to exceptions _with_retry can inspect
            if r.status_code in _RETRYABLE:
                err = self._requests.exceptions.HTTPError(
                    f"{r.status_code} from AI Hub"
                )
                err.status_code = r.status_code
                raise err
            r.raise_for_status()
            return r

        t0 = time.time()
        r   = _with_retry(_do_request)
        latency_ms = (time.time() - t0) * 1000

        data = r.json()
        if data.get("has_error"):
            raise RuntimeError(f"AI Hub error: {data.get('error')}")

        responses = data.get("data", {}).get("generative_responses", [])
        text = responses[0].get("response", "") if responses else ""

        # Estimate tokens (API does not return usage metadata)
        inp_tok = max(1, (len(prompt) + len(context or "")) // 4)
        out_tok = max(1, len(text) // 4)

        return _safe_json(text, fallback=None), inp_tok, out_tok, latency_ms

    # ------------------------------------------------------------------
    # Public interface — same signatures as GeminiClient
    # ------------------------------------------------------------------

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        """
        Freeform text generation (no JSON parsing).
        Used for principle consolidation, rule generation, and any call
        that expects a natural language response rather than structured JSON.
        """
        t0 = time.time()
        payload = {
            "ad_object_id": self.ad_object_id,
            "models":        [self.synth_model],
            "prompt":        self._b64(user),
            "advanced": {
                "temperature": 0.3,
                "max_tokens":  max_tokens,
            },
        }
        if system:
            payload["context"] = self._b64(system)

        def _do_request():
            r = self._session.post(
                f"{self.BASE_URL}/generative",
                json=payload,
                timeout=90,
            )
            if r.status_code in _RETRYABLE:
                err = self._requests.exceptions.HTTPError(
                    f"{r.status_code} from AI Hub"
                )
                err.status_code = r.status_code
                raise err
            r.raise_for_status()
            return r

        r = _with_retry(_do_request)
        data = r.json()
        if data.get("has_error"):
            raise RuntimeError(f"AI Hub error: {data.get('error')}")
        responses = data.get("data", {}).get("generative_responses", [])
        return responses[0].get("response", "") if responses else ""

    def classify(self, text: str, node_prompt: str) -> tuple:
        result, inp, out, lat = self._call(
            prompt=text,
            context=node_prompt,
            model=self.classify_model,
            temperature=0.0,
            max_tokens=120,
        )
        if self.tracker:
            self.tracker.record_classify(inp, out)
        if result is None:
            result = {"classification": "NOT_ADE", "confidence": "low",
                      "rationale": "parse_error"}
        return result, inp, out, lat

    def synthesize(self, round_num: int, error_cases: list, node_list: list,
                   nugget_store=None, error_buffer=None,
                   principles_store=None, rejected_proposals=None) -> tuple:
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} NODE={c['node']} "
            f"TEXT={c['text']}"
            for i, c in enumerate(error_cases[:8])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(
                catalogue=nugget_store.catalogue_for_synthesis()
            ) if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            buf_list = list(error_buffer)[-20:]
            if buf_list:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} "
                    f"TEXT={c['text'][:80]}"
                    for c in buf_list
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = _PRINCIPLES_SECTION_TEMPLATE.format(
                principles=principles_store.catalogue_for_synthesis()
            ) + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
        rejected_section = ""
        if rejected_proposals:
            recent = rejected_proposals[-5:]
            rej_block = "\n".join(
                f"  R{r['round']}: node={r['node_id']!r} "
                f"cond={r['condition']!r} → {r['reason']}"
                for r in recent
            )
            rejected_section = _REJECTED_PROPOSALS_SECTION.format(
                rejected_block=rej_block)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            round_num=round_num,
            cases_block=cases_block,
            node_list=json.dumps(node_list),
            nugget_section=(nugget_section + buffer_section
                            + principles_section + rejected_section),
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=700,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def refine_prompt(self, node: dict, error_cases: list,
                      nugget_store=None, error_buffer=None,
                      principles_store=None) -> tuple:
        assembled = (
            nugget_store.assemble(node["prompt"]) if nugget_store
            else node["prompt"]
        )
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} TEXT={c['text']}"
            for i, c in enumerate(error_cases[:6])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(
                catalogue=nugget_store.catalogue_for_synthesis()
            ) if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            node_buf = [e for e in list(error_buffer)[-30:]
                        if e.get("node") == node["id"]][-8:]
            if node_buf:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} TEXT={c['text'][:80]}"
                    for c in node_buf
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = _PRINCIPLES_SECTION_TEMPLATE.format(
                principles=principles_store.catalogue_for_synthesis()
            ) + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
        prompt = REFINE_PROMPT_TEMPLATE.format(
            node_id=node["id"],
            description=node.get("description", ""),
            assembled_prompt=assembled,
            error_cases=cases_block,
            nugget_section=nugget_section + buffer_section + principles_section,
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=500,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def meta_synthesize(self, f1_history: list, node_list: list,
                        per_node_stats: dict, error_buffer,
                        accepted_changes: list, nugget_store=None) -> tuple:
        """Strategic meta-round: full-history tree restructuring proposal."""
        f1_str = "  " + " → ".join(f"R{i}: {f:.3f}" for i, f in enumerate(f1_history))
        pn_str = "\n".join(
            f"  {nid}: F1={s.get('f1',0):.3f}  routed={s.get('count',0)}"
            for nid, s in per_node_stats.items()
        )
        buf_list = list(error_buffer)[-30:]
        buf_block = "\n".join(
            f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} TEXT={c['text'][:80]}"
            for c in buf_list
        ) or "  (none)"
        changes_str = "\n".join(f"  R{i+1}: {a}" for i, a in enumerate(accepted_changes))
        nugget_section = (nugget_store.catalogue_for_synthesis()
                          if nugget_store else "(none)")
        prompt = META_SYNTHESIZE_TEMPLATE.format(
            f1_history=f1_str,
            node_list=json.dumps(node_list),
            per_node_stats=pn_str,
            error_buffer_block=buf_block,
            accepted_changes=changes_str or "  (none yet)",
            nugget_section=nugget_section,
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.4, max_tokens=800,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def retire_node(self, node: dict, routes: int, rounds: int,
                    avg_f1: float) -> tuple:
        prompt = RETIRE_NODE_TEMPLATE.format(
            node_id=node["id"],
            description=node.get("description", ""),
            prompt=node.get("prompt", "")[:300],
            routes=routes, rounds=rounds, avg_f1=avg_f1,
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=300,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def extract_principles(self, original_prompt: str, improved_prompt: str,
                           delta_f1: float) -> tuple:
        """After an accepted change, ask what prompt engineering principle worked."""
        prompt = EXTRACT_PRINCIPLES_TEMPLATE.format(
            delta_f1=delta_f1,
            original_prompt=original_prompt[:400],
            improved_prompt=improved_prompt[:400],
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=150,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        principle = None
        if isinstance(result, dict):
            principle = result.get("principle")
        return principle, inp, out, lat

    def evolve_identity(self, current_identity: str, f1_history: list,
                        per_node_stats: dict, accepted_count: int,
                        principle_count: int,
                        last_meta_analysis: str = "") -> tuple:
        """After a meta-round, update NEXUS's self-concept."""
        f1_str = " → ".join(f"R{i}:{f:.3f}" for i, f in enumerate(f1_history))
        if per_node_stats:
            best_nid  = max(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            worst_nid = min(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            best_f1   = per_node_stats[best_nid].get("f1", 0.0)
            worst_f1  = per_node_stats[worst_nid].get("f1", 0.0)
        else:
            best_nid = worst_nid = "N/A"
            best_f1 = worst_f1 = 0.0
        prompt = EVOLVE_IDENTITY_TEMPLATE.format(
            current_identity=current_identity,
            f1_history=f1_str,
            strongest_node=best_nid,
            best_node_f1=best_f1,
            weakest_node=worst_nid,
            worst_node_f1=worst_f1,
            accepted_count=accepted_count,
            principle_count=principle_count,
            last_meta_analysis=last_meta_analysis[:200] or "(none yet)",
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=200,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        identity = None
        if isinstance(result, dict):
            identity = result.get("identity")
        return identity, inp, out, lat

    def extract_nuggets(self, prompt_text: str,
                        nugget_store=None) -> tuple:
        catalogue = nugget_store.catalogue_for_synthesis() if nugget_store else "(none)"
        prompt = EXTRACT_NUGGETS_TEMPLATE.format(
            prompt=prompt_text, catalogue=catalogue,
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=400,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        nuggets = []
        if isinstance(result, dict):
            nuggets = result.get("new_nuggets", []) or []
        return nuggets, inp, out, lat

    def repair_condition(self, bad_condition: str,
                         bad_vars: list) -> tuple:
        prompt = REPAIR_CONDITION_TEMPLATE.format(
            bad_condition=bad_condition,
            bad_vars=", ".join(bad_vars),
        )
        result, inp, out, lat = self._call(
            prompt=prompt, model=self.synth_model,
            temperature=0.1, max_tokens=100,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        if isinstance(result, dict) and result.get("trigger_condition"):
            return result["trigger_condition"], inp, out, lat
        return None, inp, out, lat


# ---------------------------------------------------------------------------
# OpenAIClient — generic OpenAI-compatible client (OpenAI, Ollama, Anthropic, etc.)
# ---------------------------------------------------------------------------

class OpenAIClient:
    """
    Generic OpenAI-compatible client for NEXUS.

    Works with any OpenAI-compatible API endpoint:
      - OpenAI:    base_url=None (default), set OPENAI_API_KEY
      - Ollama:    base_url="http://localhost:11434/v1", no key needed
      - Anthropic: base_url="https://api.anthropic.com/v1", set ANTHROPIC_API_KEY
      - Together:  base_url="https://api.together.xyz/v1", set TOGETHER_API_KEY
      - Any other OpenAI-compatible provider

    Recommended model splits:
      OpenAI:    classify_model="gpt-4o-mini"   synth_model="gpt-4o"
      Ollama:    classify_model="llama3.2"       synth_model="llama3.1:70b"
      Anthropic: classify_model="claude-haiku-4-5" synth_model="claude-sonnet-4-5"

    Requires: pip install openai
    """

    def __init__(self, api_key: str = None, base_url: str = None,
                 classify_model: str = "gpt-4o-mini",
                 synth_model: str = "gpt-4o",
                 tracker=None):
        from openai import OpenAI
        key = api_key or os.environ.get("OPENAI_API_KEY") or "ollama"
        self._client = OpenAI(api_key=key, base_url=base_url)
        self.classify_model = classify_model
        self.synth_model    = synth_model
        self.tracker        = tracker

    def _gen(self, prompt: str, system: str = None, model: str = None,
             temperature: float = 0.0, max_tokens: int = 150,
             json_mode: bool = True) -> tuple:
        """
        Core generation call.
        Returns (parsed_dict_or_str, input_tokens, output_tokens, latency_ms).
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = dict(
            model=model or self.classify_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.time()
        resp = _with_retry(self._client.chat.completions.create, **kwargs)
        latency_ms = (time.time() - t0) * 1000

        usage = resp.usage
        inp = getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "completion_tokens", 0) or 0
        text = resp.choices[0].message.content or ""

        if json_mode:
            return _safe_json(text, fallback=None), inp, out, latency_ms
        return text, inp, out, latency_ms

    def chat(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Freeform text generation — used by principle consolidation and meta-rounds."""
        text, _, _, _ = self._gen(
            prompt=user, system=system, model=self.synth_model,
            temperature=0.3, max_tokens=max_tokens, json_mode=False,
        )
        return text

    def classify(self, text: str, node_prompt: str) -> tuple:
        result, inp, out, lat = self._gen(
            prompt=text, system=node_prompt, model=self.classify_model,
            temperature=0.0, max_tokens=120, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_classify(inp, out)
        if result is None:
            result = {"classification": "NOT_ADE", "confidence": "low",
                      "rationale": "parse_error"}
        return result, inp, out, lat

    def synthesize(self, round_num: int, error_cases: list, node_list: list,
                   nugget_store=None, error_buffer=None,
                   principles_store=None, rejected_proposals=None) -> tuple:
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} NODE={c['node']} TEXT={c['text']}"
            for i, c in enumerate(error_cases[:8])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(catalogue=nugget_store.catalogue_for_synthesis())
            if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            buf_list = list(error_buffer)[-20:]
            if buf_list:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} TEXT={c['text'][:80]}"
                    for c in buf_list
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = (
                _PRINCIPLES_SECTION_TEMPLATE.format(principles=principles_store.catalogue_for_synthesis())
                + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
            )
        rejected_section = ""
        if rejected_proposals:
            rej_block = "\n".join(
                f"  R{r['round']}: node={r['node_id']!r} cond={r['condition']!r} → {r['reason']}"
                for r in rejected_proposals[-5:]
            )
            rejected_section = _REJECTED_PROPOSALS_SECTION.format(rejected_block=rej_block)
        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            round_num=round_num,
            cases_block=cases_block,
            node_list=json.dumps(node_list),
            nugget_section=(nugget_section + buffer_section + principles_section + rejected_section),
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=700, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def refine_prompt(self, node: dict, error_cases: list,
                      nugget_store=None, error_buffer=None,
                      principles_store=None) -> tuple:
        assembled = (nugget_store.assemble(node["prompt"]) if nugget_store else node["prompt"])
        cases_block = "\n".join(
            f"  [{i+1}] GT={c['label']} PRED={c['pred']} TEXT={c['text']}"
            for i, c in enumerate(error_cases[:6])
        )
        nugget_section = (
            _NUGGET_SECTION_TEMPLATE.format(catalogue=nugget_store.catalogue_for_synthesis())
            if nugget_store else ""
        )
        buffer_section = ""
        if error_buffer:
            node_buf = [e for e in list(error_buffer)[-30:] if e.get("node") == node["id"]][-8:]
            if node_buf:
                buf_block = "\n".join(
                    f"  GT={c['label']} PRED={c['pred']} TEXT={c['text'][:80]}" for c in node_buf
                )
                buffer_section = _ERROR_BUFFER_SECTION.format(buffer_block=buf_block)
        principles_section = ""
        if principles_store:
            principles_section = (
                _PRINCIPLES_SECTION_TEMPLATE.format(principles=principles_store.catalogue_for_synthesis())
                + _IDENTITY_SECTION_TEMPLATE.format(identity=principles_store.identity)
            )
        prompt = REFINE_PROMPT_TEMPLATE.format(
            node_id=node["id"],
            description=node.get("description", ""),
            assembled_prompt=assembled,
            error_cases=cases_block,
            nugget_section=nugget_section + buffer_section + principles_section,
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=500, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def meta_synthesize(self, f1_history: list, node_list: list,
                        per_node_stats: dict, error_buffer,
                        accepted_changes: list, nugget_store=None) -> tuple:
        f1_str = "  " + " → ".join(f"R{i}: {f:.3f}" for i, f in enumerate(f1_history))
        pn_str = "\n".join(
            f"  {nid}: F1={s.get('f1',0):.3f}  routed={s.get('count',0)}"
            for nid, s in per_node_stats.items()
        )
        buf_list = list(error_buffer)[-30:]
        buf_block = "\n".join(
            f"  GT={c['label']} PRED={c['pred']} NODE={c['node']} TEXT={c['text'][:80]}"
            for c in buf_list
        ) or "  (none)"
        changes_str = "\n".join(f"  R{i+1}: {a}" for i, a in enumerate(accepted_changes))
        nugget_section = nugget_store.catalogue_for_synthesis() if nugget_store else "(none)"
        prompt = META_SYNTHESIZE_TEMPLATE.format(
            f1_history=f1_str,
            node_list=json.dumps(node_list),
            per_node_stats=pn_str,
            error_buffer_block=buf_block,
            accepted_changes=changes_str or "  (none yet)",
            nugget_section=nugget_section,
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.4, max_tokens=800, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def retire_node(self, node: dict, routes: int, rounds: int, avg_f1: float) -> tuple:
        prompt = RETIRE_NODE_TEMPLATE.format(
            node_id=node["id"], description=node.get("description", ""),
            prompt=node.get("prompt", "")[:300],
            routes=routes, rounds=rounds, avg_f1=avg_f1,
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=300, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, lat

    def extract_principles(self, original_prompt: str, improved_prompt: str,
                           delta_f1: float) -> tuple:
        prompt = EXTRACT_PRINCIPLES_TEMPLATE.format(
            delta_f1=delta_f1,
            original_prompt=original_prompt[:400],
            improved_prompt=improved_prompt[:400],
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=150, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        principle = result.get("principle") if isinstance(result, dict) else None
        return principle, inp, out, lat

    def evolve_identity(self, current_identity: str, f1_history: list,
                        per_node_stats: dict, accepted_count: int,
                        principle_count: int, last_meta_analysis: str = "") -> tuple:
        f1_str = " → ".join(f"R{i}:{f:.3f}" for i, f in enumerate(f1_history))
        if per_node_stats:
            best_nid  = max(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            worst_nid = min(per_node_stats, key=lambda n: per_node_stats[n].get("f1", 0))
            best_f1   = per_node_stats[best_nid].get("f1", 0.0)
            worst_f1  = per_node_stats[worst_nid].get("f1", 0.0)
        else:
            best_nid = worst_nid = "N/A"
            best_f1 = worst_f1 = 0.0
        prompt = EVOLVE_IDENTITY_TEMPLATE.format(
            current_identity=current_identity, f1_history=f1_str,
            strongest_node=best_nid, best_node_f1=best_f1,
            weakest_node=worst_nid, worst_node_f1=worst_f1,
            accepted_count=accepted_count, principle_count=principle_count,
            last_meta_analysis=last_meta_analysis[:200] or "(none yet)",
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.3, max_tokens=200, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        identity = result.get("identity") if isinstance(result, dict) else None
        return identity, inp, out, lat

    def extract_nuggets(self, prompt_text: str, nugget_store=None) -> tuple:
        catalogue = nugget_store.catalogue_for_synthesis() if nugget_store else "(none)"
        prompt = EXTRACT_NUGGETS_TEMPLATE.format(prompt=prompt_text, catalogue=catalogue)
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.2, max_tokens=400, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        nuggets = result.get("new_nuggets", []) if isinstance(result, dict) else []
        return nuggets or [], inp, out, lat

    def repair_condition(self, bad_condition: str, bad_vars: list) -> tuple:
        prompt = REPAIR_CONDITION_TEMPLATE.format(
            bad_condition=bad_condition, bad_vars=", ".join(bad_vars),
        )
        result, inp, out, lat = self._gen(
            prompt=prompt, model=self.synth_model,
            temperature=0.1, max_tokens=100, json_mode=True,
        )
        if self.tracker:
            self.tracker.record_synth(inp, out)
        if isinstance(result, dict) and result.get("trigger_condition"):
            return result["trigger_condition"], inp, out, lat
        return None, inp, out, lat


# ---------------------------------------------------------------------------
# MockClient — deterministic offline stand-in
# ---------------------------------------------------------------------------

class MockClient:
    """
    Offline stand-in for GeminiClient. NOT a real classifier — results
    are for pipeline-plumbing verification only. Token counts are
    simulated from prompt length so TokenTracker has non-zero numbers.
    """

    def __init__(self, seed: int = 0, tracker=None):
        self.rng = random.Random(seed)
        self.tracker = tracker
        self._call_count = 0

    def classify(self, text: str, node_prompt: str) -> tuple[dict, int, int, float]:
        t = text.lower()
        looks_ade = any(k in t for k in ["induced", "toxicity", "adverse",
                                          "reaction", "associated"])
        if self.rng.random() < 0.15:
            looks_ade = not looks_ade
        self._call_count += 1
        inp = max(1, len(node_prompt) // 4)
        out = 15
        if self.tracker:
            self.tracker.record_classify(inp, out)
        return (
            {"classification": "ADE" if looks_ade else "NOT_ADE",
             "confidence": "medium", "rationale": "mock heuristic"},
            inp, out, float(self.rng.randint(50, 200)),
        )

    def synthesize(self, round_num: int, error_cases: list, node_list: list,
                   nugget_store=None, error_buffer=None,
                   principles_store=None, rejected_proposals=None) -> tuple[dict, int, int, float]:
        existing_ids = {n["id"] for n in node_list}
        new_id = f"NODE_MOCK_{round_num}"
        while new_id in existing_ids:
            new_id += "_b"

        conditions = [
            "has_report and not has_induced",
            "has_toxicity and not has_negation",
            "has_adverse and has_drug_name",
            "has_reaction and not has_negation",
            "has_developed and has_drug_name",
        ]
        cond = conditions[round_num % len(conditions)]

        if nugget_store and "EXPERT_ROLE" in nugget_store.nuggets:
            prompt = (
                f"[EXPERT_ROLE] Specialist for pattern detected in round {round_num}. "
                f"[JSON_SCHEMA]"
            )
        else:
            prompt = f"Mock specialist prompt for pipeline testing (round {round_num})."

        inp, out = 200, 80
        if self.tracker:
            self.tracker.record_synth(inp, out)

        proposal = {
            "error_pattern": f"mock pattern for testing (round {round_num})",
            "new_node": {
                "id": new_id,
                "description": f"Mock graft for pipeline testing (round {round_num})",
                "trigger_condition": cond,
                "prompt": prompt,
            },
        }
        return proposal, inp, out, float(self.rng.randint(300, 800))

    def refine_prompt(self, node: dict, error_cases: list,
                      nugget_store=None, error_buffer=None,
                      principles_store=None) -> tuple[dict, int, int, float]:
        """Mock refinement: append a targeted clarification sentence."""
        base = node.get("prompt", "")
        clarification = " Pay special attention to negation patterns."
        improved = base + clarification if clarification not in base else base

        inp, out = 150, 60
        if self.tracker:
            self.tracker.record_synth(inp, out)

        result = {
            "analysis": f"mock analysis: node {node['id']} misses negation nuance",
            "improved_prompt": improved,
        }
        return result, inp, out, float(self.rng.randint(200, 600))

    def extract_principles(self, original_prompt: str, improved_prompt: str,
                           delta_f1: float) -> tuple:
        """Mock: return a placeholder principle."""
        principle = (
            f"Mock principle: prompt specificity correlated with Δ={delta_f1:+.4f} improvement."
        )
        inp, out = 80, 30
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return principle, inp, out, float(self.rng.randint(50, 150))

    def evolve_identity(self, current_identity: str, f1_history: list,
                        per_node_stats: dict, accepted_count: int,
                        principle_count: int,
                        last_meta_analysis: str = "") -> tuple:
        """Mock: return a placeholder evolved identity."""
        identity = (
            f"I am NEXUS (mock), trained for {len(f1_history)} rounds. "
            f"I have accepted {accepted_count} improvements and learned "
            f"{principle_count} prompt engineering principles."
        )
        inp, out = 100, 40
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return identity, inp, out, float(self.rng.randint(50, 200))

    def extract_nuggets(self, prompt_text: str,
                        nugget_store=None) -> tuple[list, int, int, float]:
        """Mock nugget extraction: propose one generic nugget per call."""
        count = getattr(self, "_nugget_count", 0) + 1
        self._nugget_count = count

        new_nuggets = [{
            "id": f"MOCK_NUGGET_{count}",
            "text": f"This is a mock reusable fragment number {count} for testing.",
        }]
        inp, out = 100, 40
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return new_nuggets, inp, out, float(self.rng.randint(100, 300))

    def meta_synthesize(self, f1_history, node_list, per_node_stats, error_buffer,
                        accepted_changes, nugget_store=None) -> tuple[dict, int, int, float]:
        """Mock meta-synthesis: propose a basic new node."""
        conditions = ["has_report and has_drug_name", "has_adverse and not has_negation"]
        cond = conditions[len(f1_history) % len(conditions)]
        result = {
            "strategic_analysis": "mock meta-analysis",
            "action_type": "new_node",
            "new_node": {
                "id": f"NODE_META_{len(f1_history)}",
                "description": "Meta-round proposed specialist",
                "trigger_condition": cond,
                "prompt": "[EXPERT_ROLE] Meta-round specialist. [JSON_SCHEMA]",
            },
        }
        inp, out = 300, 100
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, float(self.rng.randint(200, 600))

    def retire_node(self, node, routes, rounds, avg_f1) -> tuple[dict, int, int, float]:
        result = {
            "rationale": f"mock retire: node routed only {routes} cases",
            "root_prompt_addition": "",
        }
        inp, out = 100, 30
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return result, inp, out, float(self.rng.randint(50, 150))

    def repair_condition(self, bad_condition: str,
                         bad_vars: list) -> tuple[str | None, int, int, float]:
        """Mock repair: substitute the first bad var with has_report."""
        repaired = bad_condition
        for bad in bad_vars:
            repaired = repaired.replace(bad, "has_report")
        inp, out = 60, 20
        if self.tracker:
            self.tracker.record_synth(inp, out)
        return repaired, inp, out, float(self.rng.randint(50, 150))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _safe_json(text: str, fallback):
    """Parse JSON, stripping markdown fences if the model added them."""
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        cleaned = text.strip().strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return fallback

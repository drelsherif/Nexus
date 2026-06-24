"""
token_tracker.py
Tracks input/output token usage across all Gemini API calls in a NEXUS run.

Two call types are tracked separately so you can see where tokens are going:
  - classify: the per-case routing calls (high volume, short prompts)
  - synth:    the per-round synthesis calls (low volume, longer prompts)

At the end of a run, print_summary() gives a breakdown and a rough cost
estimate based on Gemini 2.5 Flash Lite pricing.
"""

# Approximate Gemini 2.5 Flash Lite pricing (USD per million tokens).
# Check https://ai.google.dev/pricing for current rates before citing in a paper.
_PRICE_INPUT_PER_M = 0.10
_PRICE_OUTPUT_PER_M = 0.40


class TokenTracker:
    """Accumulate token usage across all API calls."""

    def __init__(self):
        self.classify_input: int = 0
        self.classify_output: int = 0
        self.classify_calls: int = 0
        self.synth_input: int = 0
        self.synth_output: int = 0
        self.synth_calls: int = 0
        # Running per-round snapshots so nexus_run can compute deltas
        self._last_snapshot: dict = self._snapshot()

    def record_classify(self, input_tokens: int, output_tokens: int):
        self.classify_input += input_tokens
        self.classify_output += output_tokens
        self.classify_calls += 1

    def record_synth(self, input_tokens: int, output_tokens: int):
        self.synth_input += input_tokens
        self.synth_output += output_tokens
        self.synth_calls += 1

    def _snapshot(self) -> dict:
        return {
            "classify_input": self.classify_input,
            "classify_output": self.classify_output,
            "classify_calls": self.classify_calls,
            "synth_input": self.synth_input,
            "synth_output": self.synth_output,
            "synth_calls": self.synth_calls,
        }

    def round_delta(self) -> dict:
        """
        Return token counts consumed since the last call to round_delta().
        Call once per round at the END of the round to capture per-round usage.
        """
        now = self._snapshot()
        prev = self._last_snapshot
        delta = {
            "round_tokens_classify_in":  now["classify_input"]  - prev["classify_input"],
            "round_tokens_classify_out": now["classify_output"] - prev["classify_output"],
            "round_tokens_synth_in":     now["synth_input"]     - prev["synth_input"],
            "round_tokens_synth_out":    now["synth_output"]     - prev["synth_output"],
            "round_api_calls":           (now["classify_calls"] + now["synth_calls"])
                                       - (prev["classify_calls"] + prev["synth_calls"]),
        }
        delta["round_tokens_total"] = (
            delta["round_tokens_classify_in"]  + delta["round_tokens_classify_out"] +
            delta["round_tokens_synth_in"]     + delta["round_tokens_synth_out"]
        )
        self._last_snapshot = now
        return delta

    def total_tokens(self) -> int:
        return (self.classify_input + self.classify_output +
                self.synth_input    + self.synth_output)

    def total_calls(self) -> int:
        return self.classify_calls + self.synth_calls

    def cost_usd(self) -> float:
        total_in  = self.classify_input  + self.synth_input
        total_out = self.classify_output + self.synth_output
        return (total_in  * _PRICE_INPUT_PER_M  / 1_000_000 +
                total_out * _PRICE_OUTPUT_PER_M / 1_000_000)

    def summary_dict(self) -> dict:
        return {
            "classify_calls":    self.classify_calls,
            "classify_input_tokens":  self.classify_input,
            "classify_output_tokens": self.classify_output,
            "synth_calls":       self.synth_calls,
            "synth_input_tokens":     self.synth_input,
            "synth_output_tokens":    self.synth_output,
            "total_tokens":      self.total_tokens(),
            "total_api_calls":   self.total_calls(),
            "est_cost_usd":      round(self.cost_usd(), 6),
        }

    def print_summary(self):
        print("\n=== Token Usage Summary ===")
        print(f"  Classify  {self.classify_calls:>5} calls | "
              f"in {self.classify_input:>8,} | out {self.classify_output:>7,}")
        print(f"  Synthesize {self.synth_calls:>4} calls | "
              f"in {self.synth_input:>8,} | out {self.synth_output:>7,}")
        print(f"  TOTAL tokens : {self.total_tokens():>10,}")
        print(f"  Est. cost    : ${self.cost_usd():.4f} USD")
        print(f"  (Pricing: ~${_PRICE_INPUT_PER_M}/M input, "
              f"~${_PRICE_OUTPUT_PER_M}/M output — verify at ai.google.dev/pricing)")

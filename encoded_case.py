"""
encoded_case.py
Numerical encoding of training cases for NEXUS.

All internal routing logic operates on integers (feature_bits, drug_ids,
effect_ids). Text is reconstructed only when building an LLM prompt —
the lazy-decoding / 'recreate from code' principle.

Biological mapping:
  EncodedCase.feature_bits  → dentate gyrus sparse code (11-bit activity pattern)
  EncodedCase.drug_ids      → presynaptic neuron identity (which drug fired)
  EncodedCase.effect_ids    → postsynaptic neuron identity (which effect fired)
  from_raw()                → entorhinal cortex → DG encoding pass
  unpack()                  → hippocampal → cortical reconstruction (decode)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from drug_registry import DrugRegistry

from features import features as _extract_features, FEATURE_NAMES


@dataclass
class EncodedCase:
    """
    Numerical representation of a single classified sentence.

    feature_bits: 11-bit packed integer (one bit per boolean feature).
      bit 0 = has_induced     bit 6 = has_reaction
      bit 1 = has_associated  bit 7 = has_report
      bit 2 = has_toxicity    bit 8 = has_negation
      bit 3 = has_adverse     bit 9 = has_short
      bit 4 = has_developed   bit 10 = has_drug_name
      bit 5 = has_following

    Bit operations on feature_bits replace per-feature lookups in routing:
      has_negation:  (feature_bits >> 8) & 1
      ADE-likely:    feature_bits & 0b00000001111  (bits 0-3 all set)
    """

    text:         str
    label:        int           # 0 = NOT_ADE,  1 = ADE
    drug_ids:     list = field(default_factory=list)    # int IDs from DrugRegistry
    effect_ids:   list = field(default_factory=list)    # int IDs from DrugRegistry
    feature_bits: int  = 0                              # 11-bit packed boolean vector

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_raw(cls, item: dict, drug_registry=None) -> "EncodedCase":
        """
        Encode a raw {text, label} dict into a numerical EncodedCase.
        drug_registry is optional — if None, drug_ids and effect_ids are empty.
        This is the 'entorhinal cortex → DG encoding' step.
        """
        text  = item["text"]
        label = 1 if item["label"] == "ADE" else 0

        # Feature extraction → pack into 11-bit integer
        feats = _extract_features(text)
        bits  = _pack(feats)

        # Drug + effect encoding via registry
        drug_ids   = []
        effect_ids = []
        if drug_registry is not None:
            drugs    = drug_registry.detect_drugs(text)
            drug_ids = [
                drug_registry.drug_to_id[d]
                for d in drugs
                if d in drug_registry.drug_to_id
            ]
            effects    = drug_registry.detect_effects(text)
            effect_ids = [
                drug_registry.effect_to_id[e]
                for e in effects
                if e in drug_registry.effect_to_id
            ]

        return cls(
            text=text, label=label,
            drug_ids=drug_ids, effect_ids=effect_ids,
            feature_bits=bits,
        )

    # ── Decoding (numbers → text, lazy) ───────────────────────────────────────

    def unpack(self) -> dict:
        """
        Decode feature_bits back to named boolean dict.
        Used for logging, display, and LLM prompt assembly.
        Biological analog: hippocampal → cortical reconstruction.
        """
        return {
            name: bool(self.feature_bits & (1 << i))
            for i, name in enumerate(FEATURE_NAMES)
        }

    def label_str(self) -> str:
        return "ADE" if self.label == 1 else "NOT_ADE"

    # ── Bit-level routing helpers ─────────────────────────────────────────────

    def has_bit(self, feature_name: str) -> bool:
        """Check a single feature bit by name (O(1))."""
        try:
            idx = FEATURE_NAMES.index(feature_name)
            return bool(self.feature_bits & (1 << idx))
        except ValueError:
            return False

    def ade_signal_strength(self) -> int:
        """
        Count set bits in ADE-positive features (bits 0–7 excluding negation).
        Higher = stronger ADE signal in the encoding.
        """
        ade_mask = 0b0_00_11111111   # bits 0–7
        neg_mask = 0b0_01_00000000   # bit 8 (has_negation)
        positive = self.feature_bits & ade_mask & ~neg_mask
        return bin(positive).count("1")

    # ── Representation ────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EncodedCase(label={self.label_str()}, "
            f"bits={self.feature_bits:011b}, "
            f"drugs={self.drug_ids}, "
            f"effects={self.effect_ids})"
        )


# ── Module-level packing helper ───────────────────────────────────────────────

def _pack(feats: dict) -> int:
    """Pack a feature dict into a single 11-bit integer."""
    bits = 0
    for i, name in enumerate(FEATURE_NAMES):
        if feats.get(name):
            bits |= (1 << i)
    return bits


def encode_pool(items: list, drug_registry=None) -> list:
    """
    Encode a list of {text, label} dicts into EncodedCase objects.
    Optionally uses drug_registry for drug/effect ID extraction.
    """
    return [EncodedCase.from_raw(item, drug_registry=drug_registry)
            for item in items]

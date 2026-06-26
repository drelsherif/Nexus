#!/usr/bin/env python3
"""
diagnose_routes.py — NEXUS v3 Route API Diagnostic

Tests route LLM calls directly against 5 known ADE and 3 known NOT_ADE sentences.
Run this BEFORE a full training run to verify the API is working and the model
is voting correctly.

Usage:
    python3 diagnose_routes.py

Requires $AIHUB_API_KEY and $AIHUB_AD_OBJECT_ID to be set.
"""

import json
import os
import sys
import time

# ─── Minimal inline setup (no imports from NEXUS internals) ──────────────────

KNOWN_ADE = [
    "Phenytoin-induced hypersensitivity reactions.",
    "Vancomycin-induced nephrotoxicity was observed in 3 of 12 patients.",
    "The patient developed severe hepatotoxicity following methotrexate therapy.",
    "Amiodarone-induced pulmonary toxicity resulted in respiratory failure.",
    "Agranulocytosis secondary to clozapine treatment was diagnosed.",
]

KNOWN_NOT_ADE = [
    "The patient tolerated the drug well with no adverse effects.",
    "Ibuprofen reduces pain and inflammation by inhibiting COX enzymes.",
    "No nephrotoxicity was observed during the treatment period.",
]

ROUTE_SYSTEM_PROMPTS = {
    "causation": (
        "You are a causation expert in pharmacovigilance. "
        "Determine whether the sentence contains DIRECT causal language "
        "linking a specific drug to a harmful or unintended outcome in a patient.\n\n"
        "Causal signals: caused, induced, associated with, resulted in, led to, "
        "following [drug], due to [drug], [drug]-related, developed after.\n"
        "NOT causal: desired therapeutic effect, negated outcome, hypothetical.\n\n"
        'Respond ONLY with JSON:\n'
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
    ),
    "negation": (
        "You are a negation expert in clinical NLP. "
        "Determine whether any adverse outcome in the sentence is NEGATED, denied, "
        "hypothetical, or qualified as absent.\n\n"
        "Negation signals: no, not, without, denied, failed to develop, "
        "did not experience, absence of, tolerates well.\n"
        "Vote NOT_ADE if the adverse event is explicitly absent. "
        "Vote ADE if the adverse event is affirmed even in presence of other negations.\n\n"
        'Respond ONLY with JSON:\n'
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
    ),
    "drug_effect": (
        "You are a pharmacology expert classifying ADE reports. "
        "Determine whether the sentence describes a known or plausible drug-adverse "
        "effect relationship based on clinical pharmacology.\n\n"
        "Vote ADE if: the drug caused unexpected toxicity, organ damage, "
        "hypersensitivity, or side effects beyond therapeutic intent.\n"
        "Vote NOT_ADE if: therapeutic effect, negated outcome, or no clear drug-harm link.\n\n"
        'Respond ONLY with JSON:\n'
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
    ),
    "context": (
        "You are a clinical context expert in pharmacovigilance. "
        "Determine whether the described outcome is a desired therapeutic effect "
        "or an unintended adverse event.\n\n"
        "NOT_ADE: pain relief, infection cleared, BP controlled, tumor regression.\n"
        "ADE: unexpected toxicity, organ damage, hypersensitivity, unwanted side effects.\n"
        "Also NOT_ADE: mechanism descriptions, lab findings without patient harm, "
        "resistance mutations, epidemiological statements.\n\n"
        'Respond ONLY with JSON:\n'
        '{"vote": "ADE" or "NOT_ADE", "confidence": 0.0-1.0, "reasoning": "<one sentence>"}'
    ),
}


def check_env():
    api_key = os.environ.get("AIHUB_API_KEY", "")
    ad_id = os.environ.get("AIHUB_AD_OBJECT_ID", "")
    if not api_key:
        print("❌ AIHUB_API_KEY is not set.")
        print("   Run: export AIHUB_API_KEY='your-key-here'")
        sys.exit(1)
    if not ad_id:
        print("❌ AIHUB_AD_OBJECT_ID is not set.")
        print("   Run: export AIHUB_AD_OBJECT_ID='your-uuid-here'")
        sys.exit(1)
    print(f"✓ AIHUB_API_KEY set ({len(api_key)} chars)")
    print(f"✓ AIHUB_AD_OBJECT_ID set ({ad_id[:8]}...)")
    return api_key, ad_id


def call_route(session, ad_id, model, system_prompt, sentence):
    """Make a single route LLM call and return (vote, confidence, reasoning, raw_response)."""
    import base64

    def b64(s):
        return base64.b64encode(s.encode("utf-8")).decode("ascii")

    payload = {
        "ad_object_id": ad_id,
        "models": [model],
        "prompt": b64(f'Sentence: "{sentence}"'),
        "context": b64(system_prompt),
        "advanced": {
            "temperature": 0.0,
            "max_tokens": 120,
        },
    }

    t0 = time.time()
    try:
        r = session.post(
            "https://api.ai.northwell.edu/generative",
            json=payload,
            timeout=30,
        )
        latency = (time.time() - t0) * 1000
        r.raise_for_status()
        data = r.json()

        if data.get("has_error"):
            return None, None, f"API error: {data.get('error')}", None, latency

        responses = data.get("data", {}).get("generative_responses", [])
        raw_text = responses[0].get("response", "") if responses else ""

        # Parse JSON
        try:
            import re
            cleaned = re.sub(r"```[a-z]*\n?", "", raw_text).strip()
            d = json.loads(cleaned)
            vote = d.get("vote", "MISSING_VOTE_KEY")
            conf = d.get("confidence", "?")
            reason = d.get("reasoning", d.get("rationale", "?"))
            return vote, conf, reason, raw_text, latency
        except Exception as pe:
            return "PARSE_ERROR", 0.0, f"JSON parse failed: {pe}", raw_text, latency

    except Exception as e:
        latency = (time.time() - t0) * 1000
        return "EXCEPTION", 0.0, str(e), None, latency


def run_diagnostics(api_key, ad_id, model="claude-haiku-4.5"):
    import requests

    session = requests.Session()
    session.headers.update({
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    })

    print(f"\n{'═'*70}")
    print(f"  NEXUS Route Diagnostic — Model: {model}")
    print(f"{'═'*70}")

    results = {"ade": [], "not_ade": []}

    for label, cases in [("ADE", KNOWN_ADE), ("NOT_ADE", KNOWN_NOT_ADE)]:
        print(f"\n{'─'*70}")
        print(f"  Testing {label} sentences ({len(cases)} cases × 4 routes)")
        print(f"{'─'*70}")

        for sentence in cases:
            print(f"\n  Sentence: \"{sentence[:80]}\"")
            print(f"  True label: {label}")

            votes = {}
            for route_name, system_prompt in ROUTE_SYSTEM_PROMPTS.items():
                vote, conf, reason, raw, latency = call_route(
                    session, ad_id, model, system_prompt, sentence
                )
                votes[route_name] = vote

                # Flag issues
                issue = ""
                if vote == "MISSING_VOTE_KEY":
                    issue = " ← ❌ WRONG KEY (model returned 'classification' not 'vote'?)"
                    if raw:
                        # Show what keys are present
                        try:
                            d = json.loads(raw)
                            issue += f" Keys: {list(d.keys())}"
                        except:
                            pass
                elif vote == "PARSE_ERROR":
                    issue = f" ← ❌ NOT JSON: {raw[:80]}"
                elif vote == "EXCEPTION":
                    issue = f" ← ❌ API EXCEPTION: {reason[:80]}"
                elif vote != label and label == "ADE":
                    issue = " ← ⚠ WRONG (should be ADE)"
                elif vote != label and label == "NOT_ADE":
                    issue = " ← ⚠ WRONG (should be NOT_ADE)"

                conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else str(conf)
                print(f"    [{route_name:12}] {vote:8} conf={conf_str:4}  {reason[:50]}{issue}")

                if latency > 5000:
                    print(f"    ⚠ Slow response: {latency:.0f}ms")

            # Aggregate
            ade_votes = sum(1 for v in votes.values() if v == "ADE")
            not_ade_votes = sum(1 for v in votes.values() if v == "NOT_ADE")
            agg = "ADE" if ade_votes >= 2 else "NOT_ADE"
            correct = "✓" if agg == label else "✗"
            print(f"  → Aggregate: {agg} ({ade_votes} ADE / {not_ade_votes} NOT_ADE) {correct}")

            results[label.lower() if label == "ADE" else "not_ade"].append({
                "sentence": sentence[:60],
                "votes": votes,
                "aggregate": agg,
                "correct": agg == label,
            })

    # Summary
    print(f"\n{'═'*70}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'═'*70}")

    ade_correct = sum(1 for r in results["ade"] if r["correct"])
    not_ade_correct = sum(1 for r in results["not_ade"] if r["correct"])

    print(f"  ADE recall:    {ade_correct}/{len(KNOWN_ADE)} ({100*ade_correct/len(KNOWN_ADE):.0f}%)")
    print(f"  NOT_ADE spec:  {not_ade_correct}/{len(KNOWN_NOT_ADE)} ({100*not_ade_correct/len(KNOWN_NOT_ADE):.0f}%)")

    # Check for systematic issues
    all_votes_flat = []
    for r in results["ade"] + results["not_ade"]:
        all_votes_flat.extend(r["votes"].values())

    exception_count = sum(1 for v in all_votes_flat if v in ("EXCEPTION", "PARSE_ERROR", "MISSING_VOTE_KEY"))
    not_ade_count = sum(1 for v in all_votes_flat if v == "NOT_ADE")
    ade_count = sum(1 for v in all_votes_flat if v == "ADE")

    print(f"\n  All route votes:  ADE={ade_count}  NOT_ADE={not_ade_count}  Errors={exception_count}")

    if exception_count > 0:
        print(f"\n  ❌ API ISSUES DETECTED: {exception_count} calls failed.")
        print("     Check API key, network connectivity, and model availability.")
    elif ade_count == 0:
        print(f"\n  ❌ MODEL NEVER VOTES ADE — something is wrong with the prompts or model.")
        print("     Try running with --model flag to test a different model.")
    elif ade_correct < len(KNOWN_ADE) // 2:
        print(f"\n  ⚠ ADE recall is low. Model may need prompt adjustment or ADE_BIAS tuning.")
    else:
        print(f"\n  ✓ Routes are working. Proceed with training run.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Diagnose NEXUS route API calls")
    parser.add_argument("--model", default="claude-haiku-4.5",
                        help="Model to test (default: claude-haiku-4.5)")
    parser.add_argument("--api-key", help="Override AIHUB_API_KEY env var")
    parser.add_argument("--ad-id", help="Override AIHUB_AD_OBJECT_ID env var")
    args = parser.parse_args()

    if args.api_key:
        os.environ["AIHUB_API_KEY"] = args.api_key
    if args.ad_id:
        os.environ["AIHUB_AD_OBJECT_ID"] = args.ad_id

    api_key, ad_id = check_env()
    run_diagnostics(api_key, ad_id, model=args.model)

"""
compare_runs.py
Compare multiple NEXUS run directories and report the winner.

Usage:
    python3 compare_runs.py run_05_branch_A run_05_branch_B run_05_branch_C
    python3 compare_runs.py run_05_branch_* --short    # one-line SMS summary
"""

import argparse
import json
import sys
from pathlib import Path


def load_summary(run_dir: str) -> dict | None:
    path = Path(run_dir) / "nexus_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_history(run_dir: str) -> list:
    s = load_summary(run_dir)
    return s.get("history", []) if s else []


def branch_report(run_dir: str) -> dict:
    s = load_summary(run_dir)
    if not s:
        return {"run_dir": run_dir, "status": "incomplete", "best_f1": 0}

    history = s.get("history", [])
    f1s     = [h["f1"] for h in history]
    best_f1 = s.get("best_f1", 0)
    final_f1 = history[-1]["f1"] if history else 0
    final_p  = history[-1].get("precision", 0) if history else 0
    final_r  = history[-1].get("recall", 0) if history else 0
    rounds   = s.get("rounds_run", len(history) - 1)

    # Count accepted changes
    accepted = sum(
        1 for h in history
        if h.get("action") and "accept" in h.get("action", "")
        and "reject" not in h.get("action", "")
    )

    principles = s.get("principles_store", {})
    n_principles = principles.get("total_principles", 0) if principles else 0

    drug_stats = s.get("drug_registry", {}) or {}
    engrams = drug_stats.get("engrams_formed", 0)

    return {
        "run_dir":      run_dir,
        "status":       "complete",
        "best_f1":      best_f1,
        "final_f1":     final_f1,
        "final_p":      final_p,
        "final_r":      final_r,
        "rounds":       rounds,
        "accepted":     accepted,
        "principles":   n_principles,
        "engrams":      engrams,
        "f1_curve":     f1s,
        "peak_round":   f1s.index(max(f1s)) if f1s else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--short", action="store_true",
                    help="One-line output for SMS")
    args = ap.parse_args()

    reports = [branch_report(d) for d in args.run_dirs]
    reports.sort(key=lambda r: r["best_f1"], reverse=True)

    if args.short:
        winner = reports[0]
        others = [f"{r['run_dir'].split('_')[-1]}={r['best_f1']:.3f}" for r in reports[1:]]
        print(
            f"Winner: {winner['run_dir']} F1={winner['best_f1']:.3f} "
            f"P={winner['final_p']:.3f} R={winner['final_r']:.3f} | "
            + " ".join(others)
        )
        return

    print("\n" + "═" * 70)
    print("  NEXUS PARALLEL RUN COMPARISON")
    print("═" * 70)

    for rank, r in enumerate(reports):
        medal = ["🥇", "🥈", "🥉", "  ", "  "][min(rank, 4)]
        status_tag = "" if r["status"] == "complete" else " [INCOMPLETE]"
        print(f"\n{medal} {r['run_dir']}{status_tag}")

        if r["status"] == "incomplete":
            print(f"   (no summary.json — still running or crashed)")
            continue

        print(f"   Best F1    : {r['best_f1']:.4f}  (peak at round {r['peak_round']})")
        print(f"   Final      : F1={r['final_f1']:.3f}  P={r['final_p']:.3f}  R={r['final_r']:.3f}")
        print(f"   Accepted   : {r['accepted']} changes over {r['rounds']} rounds")
        print(f"   Principles : {r['principles']}  |  Engrams: {r['engrams']}")

        # Mini F1 curve
        curve = r["f1_curve"]
        if curve:
            bar = "  F1 curve  : "
            for i, f in enumerate(curve):
                bar += f"R{i}={f:.3f} "
            print(bar)

    print("\n" + "═" * 70)
    if reports:
        w = reports[0]
        print(f"  WINNER: {w['run_dir']}  (Best F1 = {w['best_f1']:.4f})")
        print(f"  To use winner's tree:  cp {w['run_dir']}/nexus_best_tree.json nexus_best_tree.json")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()

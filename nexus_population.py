"""
nexus_population.py
Population runner for NEXUS.

Runs many small NEXUS branches, selects the best completed branches, then
expands the next generation with a larger batch size and inherited memory.

Example:
    export AIHUB_API_KEY=...
    export AIHUB_AD_OBJECT_ID=...
    python3 nexus_population.py --ai-hub --branches 10 --max-parallel 3
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _seed_branch_memory(parent_dir: Path, child_dir: Path):
    """Copy parent memory into a child branch before it starts."""
    _copy_if_exists(parent_dir / "nuggets", child_dir / "nuggets")
    _copy_if_exists(parent_dir / "principles", child_dir / "principles")
    _copy_if_exists(parent_dir / "drug_registry.json", child_dir / "drug_registry.json")


def _load_summary(branch_dir: Path) -> dict | None:
    path = branch_dir / "nexus_summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _branch_result(branch: dict) -> dict:
    summary = _load_summary(branch["out_dir"])
    result = {
        "branch_id": branch["branch_id"],
        "generation": branch["generation"],
        "batch_size": branch["batch_size"],
        "seed": branch["seed"],
        "parent_branch": branch.get("parent_branch"),
        "out_dir": str(branch["out_dir"]),
        "returncode": branch.get("returncode"),
        "log": str(branch["log_path"]),
        "complete": summary is not None and branch.get("returncode") == 0,
    }
    if summary:
        history = summary.get("history", [])
        baseline = history[0]["f1"] if history else None
        final_f1 = history[-1]["f1"] if history else None
        result.update({
            "baseline_f1": baseline,
            "best_f1": summary.get("best_f1"),
            "final_f1": final_f1,
            "rounds_run": summary.get("rounds_run"),
            "final_tree_nodes": summary.get("final_tree_nodes"),
        })
    return result


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def _merge_shared_memory(pop_dir: Path, winners: list[dict]):
    shared = pop_dir / "shared_memory"
    elite_dir = shared / "elite_trees"
    elite_dir.mkdir(parents=True, exist_ok=True)

    merged_nuggets = {"version": 1, "nuggets": {}}
    merged_principles = {"version": 1, "identity": "", "principles": []}
    seen_principles = set()

    for rank, winner in enumerate(winners, start=1):
        out_dir = Path(winner["out_dir"])
        tree_path = out_dir / "nexus_best_tree.json"
        if tree_path.exists():
            shutil.copy2(tree_path, elite_dir / f"rank{rank}_{winner['branch_id']}.json")

        nuggets_path = out_dir / "nuggets" / "nexus_nuggets.json"
        if nuggets_path.exists():
            with nuggets_path.open() as f:
                data = json.load(f)
            for nid, nugget in data.get("nuggets", {}).items():
                current = merged_nuggets["nuggets"].get(nid)
                if current is None or nugget.get("accepted_count", 0) > current.get("accepted_count", 0):
                    merged_nuggets["nuggets"][nid] = nugget

        principles_path = out_dir / "principles" / "nexus_principles.json"
        if principles_path.exists():
            with principles_path.open() as f:
                data = json.load(f)
            if not merged_principles["identity"]:
                merged_principles["identity"] = data.get("identity", "")
            for principle in data.get("principles", []):
                text = principle.get("text", "")
                if text and text not in seen_principles:
                    seen_principles.add(text)
                    merged_principles["principles"].append(principle)

    _write_json(shared / "merged_nuggets.json", merged_nuggets)
    _write_json(shared / "merged_principles.json", merged_principles)
    _write_json(shared / "elite_manifest.json", winners)


def _build_command(args, branch: dict, initial_tree: Path | None, fresh: bool) -> list[str]:
    cmd = [
        sys.executable,
        "nexus_run.py",
        "--rounds", str(args.rounds),
        "--batch-size", str(branch["batch_size"]),
        "--probe-size", str(args.probe_size),
        "--eval-size", str(args.eval_size),
        "--refine-probe-size", str(args.refine_probe_size),
        "--min-refine-errors", str(args.min_refine_errors),
        "--seed", str(branch["seed"]),
        "--data-seed", str(args.data_seed),
        "--out-dir", str(branch["out_dir"]),
        "--max-graft-route-share", str(args.max_graft_route_share),
        "--min-graft-routes", str(args.min_graft_routes),
    ]
    if initial_tree:
        cmd.extend(["--initial-tree", str(initial_tree)])
    if fresh:
        cmd.append("--fresh-nuggets")
    if args.ai_hub:
        cmd.append("--ai-hub")
        cmd.extend(["--classify-model", args.classify_model])
        cmd.extend(["--synth-model", args.synth_model])
    if args.mock:
        cmd.append("--mock")
        cmd.extend(["--mock-pool-size", str(args.mock_pool_size)])
    if args.no_meta:
        cmd.append("--no-meta")
    if args.no_retire:
        cmd.append("--no-retire")
    if args.no_refine:
        cmd.append("--no-refine")
    return cmd


def _run_generation(args, generation: int, batch_size: int,
                    parents: list[dict] | None) -> list[dict]:
    pop_dir = Path(args.out_dir)
    gen_dir = pop_dir / f"gen_{generation:02d}_batch_{batch_size}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    branches = []
    for idx in range(args.branches):
        parent = parents[idx % len(parents)] if parents else None
        branch_id = f"g{generation:02d}_b{idx + 1:02d}"
        out_dir = gen_dir / branch_id
        out_dir.mkdir(parents=True, exist_ok=True)
        if parent:
            _seed_branch_memory(Path(parent["out_dir"]), out_dir)
        branches.append({
            "branch_id": branch_id,
            "generation": generation,
            "batch_size": batch_size,
            "seed": args.seed + generation * 1000 + idx,
            "parent_branch": parent["branch_id"] if parent else None,
            "parent_best_f1": parent.get("best_f1") if parent else None,
            "out_dir": out_dir,
            "log_path": out_dir / "branch_stdout.log",
            "initial_tree": (
                Path(parent["out_dir"]) / "nexus_best_tree.json"
                if parent else None
            ),
        })

    running = []
    finished = []
    pending = branches[:]

    while pending or running:
        while pending and len(running) < args.max_parallel:
            branch = pending.pop(0)
            fresh = branch["initial_tree"] is None
            cmd = _build_command(args, branch, branch["initial_tree"], fresh=fresh)
            with branch["log_path"].open("w") as log:
                log.write("$ " + " ".join(cmd) + "\n\n")
                log.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=Path(__file__).resolve().parent,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            branch["process"] = proc
            branch["started_at"] = time.time()
            running.append(branch)
            print(f"[Population] started {branch['branch_id']} "
                  f"batch={batch_size} pid={proc.pid}")

        still_running = []
        for branch in running:
            proc = branch["process"]
            rc = proc.poll()
            if rc is None:
                still_running.append(branch)
                continue
            branch["returncode"] = rc
            branch["ended_at"] = time.time()
            branch.pop("process", None)
            result = _branch_result(branch)
            finished.append(result)
            print(f"[Population] finished {branch['branch_id']} rc={rc} "
                  f"best_f1={result.get('best_f1')}")
        running = still_running
        if running or pending:
            time.sleep(args.poll_seconds)

    _write_json(gen_dir / "generation_results.json", finished)
    return finished


def _select_winners(args, results: list[dict], parents: list[dict] | None) -> list[dict]:
    complete = [r for r in results if r.get("complete") and r.get("best_f1") is not None]
    if not complete:
        return []

    parent_best_by_id = {
        p["branch_id"]: p.get("best_f1")
        for p in parents
    } if parents else {}

    successful = []
    for result in complete:
        reference = parent_best_by_id.get(result.get("parent_branch"))
        if reference is None:
            reference = result.get("baseline_f1")
        if reference is None:
            reference = 0.0
        result["reference_f1"] = reference
        result["improvement"] = result["best_f1"] - reference
        if result["improvement"] >= args.min_improvement:
            successful.append(result)

    successful.sort(
        key=lambda r: (
            r.get("improvement", 0.0),
            r.get("best_f1", 0.0),
            r.get("final_f1", 0.0),
        ),
        reverse=True,
    )
    complete.sort(
        key=lambda r: (r.get("best_f1", 0.0), r.get("final_f1", 0.0)),
        reverse=True,
    )

    winners = []
    seen = set()
    for result in successful + complete:
        if result["branch_id"] in seen:
            continue
        winners.append(result)
        seen.add(result["branch_id"])
        if len(winners) >= args.parents_kept:
            break
    return winners


def main():
    ap = argparse.ArgumentParser(description="Run population-based NEXUS branches")
    ap.add_argument("--out-dir", default="population_runs")
    ap.add_argument("--branches", type=int, default=10)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--batch-start", type=int, default=5)
    ap.add_argument("--batch-end", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--probe-size", type=int, default=300)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--refine-probe-size", type=int, default=30)
    ap.add_argument("--min-refine-errors", type=int, default=2)
    ap.add_argument("--min-graft-routes", type=int, default=3)
    ap.add_argument("--max-graft-route-share", type=float, default=0.20)
    ap.add_argument("--parents-kept", type=int, default=3)
    ap.add_argument("--min-improvement", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=4200)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--poll-seconds", type=int, default=10)
    ap.add_argument("--ai-hub", action="store_true")
    ap.add_argument("--classify-model", default="claude-haiku-4.5")
    ap.add_argument("--synth-model", default="claude-sonnet-4.5")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--mock-pool-size", type=int, default=1000)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--no-retire", action="store_true")
    ap.add_argument("--no-refine", action="store_true")
    args = ap.parse_args()

    if args.max_parallel > args.branches:
        args.max_parallel = args.branches

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    all_generations = []
    parents = None

    for generation, batch_size in enumerate(
            range(args.batch_start, args.batch_end + 1), start=1):
        print(f"\n[Population] generation={generation} batch_size={batch_size}")
        results = _run_generation(args, generation, batch_size, parents)
        winners = _select_winners(args, results, parents)
        _merge_shared_memory(Path(args.out_dir), winners)
        all_generations.append({
            "generation": generation,
            "batch_size": batch_size,
            "results": results,
            "winners": winners,
        })
        _write_json(Path(args.out_dir) / "population_summary.json", all_generations)
        if not winners:
            print("[Population] no completed branches; stopping.")
            break
        parents = winners
        print("[Population] winners: " + ", ".join(
            f"{w['branch_id']} best_f1={w.get('best_f1'):.4f}"
            for w in winners
        ))

    print(f"\n[Population] done. Summary: {Path(args.out_dir) / 'population_summary.json'}")


if __name__ == "__main__":
    main()

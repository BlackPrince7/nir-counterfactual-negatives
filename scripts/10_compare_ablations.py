#!/usr/bin/env python3
"""Paired comparison on identical evaluation inputs; one training seed per table.

Bootstrap intervals measure query sampling uncertainty, not training-seed
uncertainty. No arbitrary effect-size threshold is called statistical success.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.baseline_data import load_jsonl


def compare(results, baseline_tag, metric="ndcg@10", samples=2000, seed=42):
    if baseline_tag not in results or samples < 1:
        raise ValueError("baseline must exist and bootstrap samples must be positive")
    signatures = set()
    training_signatures = set()
    query_sets = []
    for tag, result in results.items():
        summary, rows = result["summary"], result["per_query"]
        if "training_identity" in summary:
            training_signatures.add(json.dumps(summary["training_identity"], sort_keys=True))
        fingerprint = summary.get("evaluation_hash")
        if not fingerprint:
            raise ValueError("legacy results lack evaluation identity; rerun evaluator")
        cfg = summary["config"]
        signatures.add((fingerprint, summary["split"], cfg["top_k"], cfg["query_prefix"],
                        cfg["document_prefix"], cfg["max_seq_length"], cfg["gain"], cfg["empty_qrels"]))
        ids = {r["qid"] for r in rows}
        if len(ids) != len(rows) or not rows:
            raise ValueError("duplicate or empty per-query results")
        query_sets.append(ids)
        if any(metric not in r["metrics"] for r in rows):
            raise ValueError(f"{tag}: missing {metric}; increase evaluation top-k")
    if len(signatures) != 1 or any(ids != query_sets[0] for ids in query_sets):
        raise ValueError("cannot compare different evaluation sets/settings")
    if len(training_signatures) > 1:
        raise ValueError("different training cohorts, hyperparameters or seeds; compare matched runs")
    base = {r["qid"]: r["metrics"][metric] for r in results[baseline_tag]["per_query"]}
    output = {}
    for tag, result in results.items():
        values = {r["qid"]: r["metrics"][metric] for r in result["per_query"]}
        differences = np.array([values[q] - base[q] for q in sorted(base)])
        if not np.isfinite(differences).all(): raise ValueError("non-finite query metrics")
        rng = np.random.default_rng(seed)
        means = [rng.choice(differences, len(differences), replace=True).mean() for _ in range(samples)]
        output[tag] = dict(mean=float(np.mean(list(values.values()))),
                           delta_points=float(100 * differences.mean()),
                           query_bootstrap_95ci_points=[float(x) for x in 100 * np.quantile(means, [.025, .975])],
                           improved=int((differences > 0).sum()), regressed=int((differences < 0).sum()))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-base", default=str(PROJECT_ROOT / "outputs/eval_v2/dev"))
    ap.add_argument("--tags", nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--baseline-tag")
    ap.add_argument("--metric", default="ndcg@10")
    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    args = ap.parse_args()
    base = Path(args.eval_base)
    tags = args.tags or sorted(p.name for p in base.iterdir() if (p / "summary.json").exists()
                               and (p.name.endswith(f"_seed{args.seed}") or p.name == "BASE"))
    if not tags: ap.error("no matching evaluation results")
    results = {tag: dict(summary=json.loads((base / tag / "summary.json").read_text(encoding="utf-8")),
                         per_query=load_jsonl(base / tag / "per_query.jsonl")) for tag in tags}
    baseline = args.baseline_tag or f"A_seed{args.seed}"
    comparisons = compare(results, baseline, args.metric, args.bootstrap_samples, args.seed)
    output = dict(baseline=baseline, metric=args.metric, bootstrap_samples=args.bootstrap_samples,
                  uncertainty="paired query bootstrap; does not estimate training-seed variability",
                  results=comparisons)
    (base / f"comparison_seed{args.seed}.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = [f"# {args.metric}: paired comparison", "", output["uncertainty"], "",
             "| Run | Mean | Delta (points) | Query bootstrap 95% CI |",
             "|---|---:|---:|---|"]
    for tag, r in comparisons.items():
        lo, hi = r["query_bootstrap_95ci_points"]
        lines.append(f"| {tag} | {r['mean']:.4f} | {r['delta_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    report = "\n".join(lines) + "\n"
    (base / f"comparison_seed{args.seed}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Freeze train/dev/test before generation, then build a matched ablation cohort.

First use --prepare-only. Generate candidates using train_pool.jsonl.
Then run again with --split-file <output-dir>/split.json to assemble A/B/C.
Official policy never trains on source test questions. RuBQ dev is subdivided
into training and validation; whole duplicate groups may exceed requested sizes.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.baseline_data import (
    CONDITION_FILES, build_conditions, digest, load_jsonl, make_split,
    normalize, save_jsonl, validate_split,
)
from src.data_io import load_query_docs


def evaluation_data(examples, selected):
    # Text-derived stable IDs, independent of which query first mentions a doc.
    docs = {normalize(e["d_plus"]): e["d_plus"] for e in examples}
    ids = {key: "doc_" + digest(key) for key in docs}
    corpus = [{"qid": ids[key], "text": docs[key]} for key in sorted(docs)]
    relevant = {}
    for e in examples:
        relevant.setdefault(normalize(e["query"]), set()).add(ids[normalize(e["d_plus"])])
    queries = []
    for e in selected:
        gold = ids[normalize(e["d_plus"])]
        queries.append(dict(qid=e["qid"], query=e["query"], gold_qid=gold,
                            qrels={doc: 1 for doc in sorted(relevant[normalize(e["query"])])}))
    return corpus, queries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rubq-full", default=str(PROJECT_ROOT / "data/rubq/rubq_full.jsonl"))
    ap.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/baseline_v2"))
    ap.add_argument("--split-file")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--split-policy", choices=["official", "custom-grouped"], default="official")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cap-negatives", type=int, default=1,
                    help="Exact negatives/query for every explicit-negative condition")
    ap.add_argument("--conditions", nargs="+", choices=list(CONDITION_FILES), default=["A", "B", "C"])
    for source, folder in [("cn", "cn_pipeline"), ("bm25", "bm25_negatives"), ("gp", "gp_pipeline_iter")]:
        ap.add_argument("--" + source + "-input", default=str(PROJECT_ROOT / "outputs" / folder / "prod/candidates_passed.jsonl"))
    args = ap.parse_args()
    examples = [r.model_dump() for r in load_query_docs(args.rubq_full)]
    out = Path(args.output_dir)
    if args.prepare_only:
        if args.split_file:
            ap.error("--prepare-only creates a split; omit --split-file")
        if out.exists() and any(out.iterdir()):
            ap.error("prepare into a new/empty output directory to preserve frozen artifacts")
        split = make_split(examples, args.n_train, args.n_dev, args.n_test, args.seed, args.split_policy)
    else:
        if not args.split_file:
            ap.error("freeze splits first with --prepare-only, then supply --split-file")
        split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        if (out / "manifest.json").exists():
            ap.error("manifest already exists; use a new output directory for a new build")
        if (out / "split.json").exists():
            frozen = json.loads((out / "split.json").read_text(encoding="utf-8"))
            if frozen != split:
                ap.error("output contains a different frozen split; use a new output directory")
    validate_split(examples, split)
    by_id = {e["qid"]: e for e in examples}
    pools = {name: [by_id[qid] for qid in split[name + "_qids"]] for name in ("train", "dev", "test")}
    output, coverage, candidate_hashes = {}, {}, {}
    if not args.prepare_only:
        required = set()
        if "B" in args.conditions: required.add("bm25")
        if set(args.conditions) & {"C", "E"}: required.add("cn")
        if set(args.conditions) & {"D", "E"}: required.add("gp")
        candidates = {s: load_jsonl(getattr(args, s + "_input")) for s in required}
        candidate_hashes = {s: digest(rows) for s, rows in candidates.items()}
        output, coverage = build_conditions(pools["train"], candidates, args.conditions, args.cap_negatives)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    save_jsonl(out / "train_pool.jsonl", pools["train"])
    for name in ("dev", "test"):
        corpus, queries = evaluation_data(examples, pools[name])
        save_jsonl(out / (name + ".jsonl"), queries)
    save_jsonl(out / "corpus.jsonl", corpus)
    if args.prepare_only:
        print(f"Frozen split in {out}; generate ONLY from {out / 'train_pool.jsonl'}")
        print({name: len(rows) for name, rows in pools.items()})
        return
    for condition, rows in output.items():
        save_jsonl(out / CONDITION_FILES[condition], rows)
    manifest = dict(schema_version=2, split_hash=digest(split), source_hash=split["source_hash"],
                    conditions=args.conditions, negatives_per_query=args.cap_negatives,
                    coverage=coverage, candidate_hashes=candidate_hashes,
                    condition_hashes={c: digest(rows) for c, rows in output.items()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = ("# Matched baseline build\n\n"
              + f"Conditions: {', '.join(args.conditions)}\n\n"
              + f"Matched queries: {coverage['n_matched']} / {coverage['n_train_pool']}\n\n"
              + f"Exact explicit negatives per query: {args.cap_negatives}\n\n"
              + "Coverage selection limits conclusions to the matched cohort. "
              + "Inspect manifest.json for exclusions and source coverage.\n")
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

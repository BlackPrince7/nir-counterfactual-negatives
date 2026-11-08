#!/usr/bin/env python3
"""Train matched conditions using query-level sampling and a fixed step budget.

Dry runs validate data hashes and plan batches without importing Torch or loading
models. Outputs are immutable: choose a new directory for a different experiment.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.baseline_data import CONDITION_FILES, batch_plan, digest, save_jsonl
from src.baseline_training import TrainingConfig, load_condition, train_steps


def train_one_condition(*, condition, train_dir, output_dir, cfg, dry_run):
    rows, manifest = load_condition(train_dir, condition)
    plan = batch_plan(rows, condition, cfg.batch_size, cfg.max_steps, cfg.seed)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output exists: {output_dir}; choose a new --output-base")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dict(condition=condition, seed=cfg.seed, base_model=cfg.base_model,
                   hyperparams=cfg.model_dump(), manifest_hash=digest(manifest),
                   data_hash=digest(rows), n_queries=len(manifest["coverage"]["matched_qids"]),
                   planned_steps=cfg.max_steps, training_steps=0, plan_hash=digest(plan), status="dry_run",
                   model_path=None)
    # Store sampled IDs/variants so sampling and GP exposure are inspectable.
    save_jsonl(output_dir / "batch_plan.jsonl",
               [{"step": i + 1, "samples": [{"qid": r["qid"], "variant": r["source_variant"]}
                                           for r in batch]} for i, batch in enumerate(plan)])
    try:
        if not dry_run:
            import numpy as np
            import torch
            from sentence_transformers import SentenceTransformer
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
            model = SentenceTransformer(cfg.base_model)
            start = time.time()
            history = train_steps(model, plan, cfg, output_dir)
            save_jsonl(output_dir / "loss.jsonl", history)
            summary.update(status="trained", training_seconds=time.time() - start,
                           training_steps=len(history),
                           model_path=str(output_dir / "model"))
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{condition}: {summary['status']}, {summary['n_queries']} queries, {cfg.max_steps} steps")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default=str(PROJECT_ROOT / "data/baseline_v2"))
    ap.add_argument("--output-base", default=str(PROJECT_ROOT / "outputs/training_v2"))
    which = ap.add_mutually_exclusive_group()
    which.add_argument("--condition", choices=list(CONDITION_FILES))
    which.add_argument("--conditions", nargs="+", choices=list(CONDITION_FILES))
    ap.add_argument("--base-model", default="ai-forever/ru-en-RoSBERTa")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-fraction", type=float, default=0.1)
    ap.add_argument("--max-seq-length", type=int, default=256)
    ap.add_argument("--query-prefix", default="search_query: ")
    ap.add_argument("--document-prefix", default="search_document: ")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = TrainingConfig(**{key: getattr(args, key) for key in TrainingConfig.model_fields})
    for condition in args.conditions or [args.condition or "A"]:
        train_one_condition(condition=condition, train_dir=Path(args.train_dir),
                            output_dir=Path(args.output_base) / f"{condition}_seed{args.seed}",
                            cfg=cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

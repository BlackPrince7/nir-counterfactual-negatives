#!/usr/bin/env python3
"""Evaluate dense retrieval with graded qrels and saved training encoding settings.

Defaults to dev; select --split test only for the frozen final evaluation.
Legacy single-gold query files are accepted but remain incomplete judgments.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.baseline_data import CONDITION_FILES, digest, load_jsonl, save_jsonl
from src.baseline_training import TrainingConfig
from src.retrieval_eval import evaluate_rankings, retrieve_top_k, validate_evaluation


def eval_one_model(*, model_path, test_queries, corpus, output_dir, top_k,
                   batch_size, tag, split="dev", cfg=None, model=None):
    validate_evaluation(test_queries, corpus)
    if top_k < 1 or batch_size < 1:
        raise ValueError("top-k and batch-size must be positive")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output exists: {output_dir}; choose another --output-base")
    saved_config = Path(model_path) / "baseline_config.json"
    if saved_config.exists():
        saved = TrainingConfig.model_validate_json(saved_config.read_text(encoding="utf-8"))
        if cfg is not None and any(getattr(cfg, k) != getattr(saved, k) for k in
                                   ("query_prefix", "document_prefix", "max_seq_length")):
            raise ValueError("evaluation encoding differs from saved training configuration")
        cfg = saved
    elif cfg is None:
        raise ValueError("model lacks baseline_config.json; provide explicit encoding settings")
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_path)
    model.max_seq_length = cfg.max_seq_length
    # Empty prompt disables automatic model prompts; explicit prefixes are added once.
    kwargs = dict(batch_size=batch_size, normalize_embeddings=True,
                  convert_to_numpy=True, show_progress_bar=True, prompt="")
    docs = model.encode([cfg.document_prefix + c["text"] for c in corpus], **kwargs)
    queries = model.encode([cfg.query_prefix + q["query"] for q in test_queries], **kwargs)
    rankings = retrieve_top_k(queries, docs, [c["qid"] for c in corpus], top_k=top_k)
    per_query, metrics = evaluate_rankings(test_queries, rankings, top_k)
    summary = dict(tag=tag, model_path=model_path, split=split,
                   evaluation_hash=digest({"queries": test_queries, "corpus": corpus}),
                   config=dict(top_k=top_k, batch_size=batch_size,
                               query_prefix=cfg.query_prefix, document_prefix=cfg.document_prefix,
                               max_seq_length=cfg.max_seq_length, gain="linear", empty_qrels="zero"),
                   data=dict(n_test_queries=len(test_queries), n_corpus=len(corpus)),
                   metrics=metrics)
    training_summary = Path(model_path).parent / "summary.json"
    if saved_config.exists() and training_summary.exists():
        training = json.loads(training_summary.read_text(encoding="utf-8"))
        if training.get("status") != "trained":
            raise ValueError("model belongs to an unfinished/failed training run")
        summary["training_identity"] = {
            "manifest_hash": training["manifest_hash"],
            "hyperparams": training["hyperparams"],
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(output_dir / "per_query.jsonl", per_query)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        f"# Retrieval: {tag} ({split})\n\n" + "\n".join(f"- {k}: {v}" for k, v in metrics.items()),
        encoding="utf-8")
    print(f"{tag} ({split}): {metrics}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", default=str(PROJECT_ROOT / "data/baseline_v2"))
    ap.add_argument("--training-base", default=str(PROJECT_ROOT / "outputs/training_v2"))
    ap.add_argument("--output-base", default=str(PROJECT_ROOT / "outputs/eval_v2"))
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--condition", choices=list(CONDITION_FILES))
    target.add_argument("--conditions", nargs="+", choices=list(CONDITION_FILES))
    target.add_argument("--baseline", action="store_true")
    target.add_argument("--model-path")
    ap.add_argument("--base-model", default="ai-forever/ru-en-RoSBERTa")
    ap.add_argument("--tag", default="custom")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--document-prefix", default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    args = ap.parse_args()
    cfg = None
    explicit = any(getattr(args, k) is not None for k in ("query_prefix", "document_prefix", "max_seq_length"))
    if args.baseline or explicit:
        defaults = TrainingConfig()
        cfg = TrainingConfig(**{k: getattr(args, k) if getattr(args, k) is not None else getattr(defaults, k)
                                for k in ("query_prefix", "document_prefix", "max_seq_length")})
    targets = [(str(Path(args.training_base) / f"{c}_seed{args.seed}" / "model"), f"{c}_seed{args.seed}")
               for c in (args.conditions or ([args.condition] if args.condition else []))]
    if args.baseline: targets = [(args.base_model, "BASE")]
    if args.model_path: targets = [(args.model_path, args.tag)]
    queries = load_jsonl(Path(args.train_dir) / (args.split + ".jsonl"))
    corpus = load_jsonl(Path(args.train_dir) / "corpus.jsonl")
    for path, tag in targets:
        eval_one_model(model_path=path, test_queries=queries, corpus=corpus,
                       output_dir=Path(args.output_base) / args.split / tag,
                       top_k=args.top_k, batch_size=args.batch_size, tag=tag,
                       split=args.split, cfg=cfg)


if __name__ == "__main__":
    main()

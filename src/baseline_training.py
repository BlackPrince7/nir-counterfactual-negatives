"""Fixed-step MNRL training; planning and dry runs require no Torch."""
import json
import random
from pathlib import Path

from pydantic import Field

from .config import StrictModel
from .baseline_data import CONDITION_FILES, digest, load_jsonl, validate_records


class TrainingConfig(StrictModel):
    base_model: str = "ai-forever/ru-en-RoSBERTa"
    seed: int = Field(default=42, ge=0)
    max_steps: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, ge=2)
    lr: float = Field(default=2e-5, gt=0)
    warmup_fraction: float = Field(default=0.1, ge=0, lt=1)
    max_seq_length: int = Field(default=256, gt=0)
    query_prefix: str = "search_query: "
    document_prefix: str = "search_document: "


def load_condition(directory, condition):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or condition not in manifest["conditions"]:
        raise ValueError("rebuild data with the v2 matched builder")
    expected = set(manifest["coverage"]["matched_qids"])
    selected = None
    for c in manifest["conditions"]:
        rows = load_jsonl(directory / CONDITION_FILES[c])
        if digest(rows) != manifest["condition_hashes"][c]:
            raise ValueError(f"{c}: data changed since build")
        grouped = validate_records(rows, c)
        if set(grouped) != expected:
            raise ValueError("conditions have different query cohorts")
        k = manifest["negatives_per_query"] if c in {"B", "C", "E"} else 0
        if any(len(r["negatives"]) != k for r in rows):
            raise ValueError("negative count differs from manifest")
        if c == condition: selected = rows
    return selected, manifest


def train_steps(model, plan, cfg, output_dir):
    """Plain Torch loop avoids legacy fit conversion of variable-width examples."""
    import numpy as np
    import torch
    try:
        from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    except ImportError:  # Sentence Transformers 3.x/4.x layout
        from sentence_transformers.losses import MultipleNegativesRankingLoss

    if len(plan) != cfg.max_steps:
        raise ValueError("batch plan length differs from optimizer step budget")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg.seed)
    model.max_seq_length = cfg.max_seq_length
    model.train()
    objective = MultipleNegativesRankingLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    warmup = int(cfg.max_steps * cfg.warmup_fraction)
    history = []
    for step, batch in enumerate(plan):
        # Full, constant-size batches and fixed arity have already been validated.
        columns = [[cfg.query_prefix + r["query"] for r in batch],
                   [cfg.document_prefix + r["positive"] for r in batch]]
        for i in range(len(batch[0]["negatives"])):
            columns.append([cfg.document_prefix + r["negatives"][i] for r in batch])
        features = [{k: v.to(model.device) for k, v in model.tokenize(texts).items()}
                    for texts in columns]
        factor = (step + 1) / warmup if step < warmup else (cfg.max_steps - step) / max(1, cfg.max_steps - warmup)
        for group in optimizer.param_groups: group["lr"] = cfg.lr * factor
        optimizer.zero_grad(set_to_none=True)
        loss = objective(features, None)
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach().cpu()), "lr": cfg.lr * factor})
    model_path = Path(output_dir) / "model"
    model.save(str(model_path))
    (model_path / "baseline_config.json").write_text(cfg.model_dump_json(indent=2), encoding="utf-8")
    return history

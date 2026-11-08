import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.baseline_data import (batch_plan, build_conditions, digest, make_split,
                               normalize, save_jsonl, validate_split)
from src.baseline_training import TrainingConfig, load_condition
from src.retrieval_eval import (evaluate_rankings, query_metrics, retrieve_top_k,
                                validate_evaluation)

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def examples():
    return [dict(qid=f"q{i}", query=f"query {i}", d_plus=f"doc {i}",
                 meta={"source_split": "dev" if i < 20 else "test"}) for i in range(30)]


def candidates(train):
    return {s: [dict(qid=e["qid"], query=e["query"], d_plus=e["d_plus"],
                     d_minus=f"{s} negative {e['qid']}", passed_filters=True)
                for e in train] for s in ("cn", "bm25")}


def local_model_modules():
    try:
        from sentence_transformers.sentence_transformer import modules
    except ImportError:  # Older supported Sentence Transformers layout
        from sentence_transformers import models as modules
    return modules


def test_official_split_group_leakage_and_reuse():
    rows = examples()
    rows[0]["d_plus"] = rows[21]["d_plus"]  # dev/test shared source: exclude dev
    rows[1]["d_plus"] = rows[2]["d_plus"]
    rows[2]["query"] = rows[3]["query"]  # transitive group
    split = make_split(rows, 10, 3, 5, 42)
    validate_split(rows, split)
    assert "q0" in split["excluded_source_overlap"]
    assert not set(split["train_qids"]) & {f"q{i}" for i in range(20, 30)}
    assert split == make_split(list(reversed(rows)), 10, 3, 5, 42)
    rows[5]["d_plus"] = "changed corpus"
    with pytest.raises(ValueError, match="mismatch"):
        validate_split(rows, split)


def test_split_exact_counts_without_duplicate_groups():
    split = make_split(examples(), 11, 3, 7, 42)
    assert [len(split[s + "_qids"]) for s in ("train", "dev", "test")] == [11, 3, 7]


def test_reused_split_cannot_move_official_test_into_training():
    rows = examples()
    split = make_split(rows, 10, 3, 5, 42)
    split["train_qids"][0], split["test_qids"][0] = split["test_qids"][0], split["train_qids"][0]
    with pytest.raises(ValueError, match="official source split"):
        validate_split(rows, split)


def test_split_insufficient_data_and_missing_official_labels():
    with pytest.raises(ValueError): make_split(examples(), 100, 3, 5, 42)
    rows = examples()
    rows[0]["meta"] = {}
    with pytest.raises(ValueError, match="source_split"): make_split(rows, 10, 3, 5, 42)


def test_matched_cohort_and_known_positive_exclusion():
    rows = examples()[:4]
    raw = candidates(rows)
    raw["cn"] = raw["cn"][:3]
    raw["bm25"][0]["d_minus"] = " DOC   0 "
    conditions, coverage = build_conditions(rows, raw, ["A", "B", "C"], 1)
    assert coverage["matched_qids"] == ["q1", "q2"]
    assert all({r["qid"] for r in records} == {"q1", "q2"} for records in conditions.values())
    assert all(len(r["negatives"]) == 1 for c in ["B", "C"] for r in conditions[c])
    assert all(not r["negatives"] for r in conditions["A"])


def test_gp_conditions_share_cohort_and_reject_alternative_positive():
    rows = examples()[:4]
    raw = candidates(rows)
    raw["gp"] = [dict(qid=e["qid"], text=f"alternative {e['qid']}", passed_filters=True)
                 for e in rows[:3]]
    raw["cn"][0]["d_minus"] = " ALTERNATIVE   q0 "
    conditions, coverage = build_conditions(rows, raw, ["A", "B", "C", "D", "E"], 1)
    assert coverage["matched_qids"] == ["q1", "q2"]
    for condition, records in conditions.items():
        assert {r["qid"] for r in records} == {"q1", "q2"}
        assert len(records) == (4 if condition in {"D", "E"} else 2)
        plan = batch_plan(records, condition, 2, 3, 42)
        assert all(len({r["qid"] for r in batch}) == 2 for batch in plan)


def test_batch_plan_has_no_duplicate_queries_or_competing_variants():
    rows = [dict(qid=f"q{i}", query=f"query {i}", positive=f"positive {i} v{j}",
                 negatives=[], source_variant=str(j)) for i in range(8) for j in range(3)]
    plan = batch_plan(rows, "D", 4, 25, 42)
    assert len(plan) == 25
    assert plan == batch_plan(rows, "D", 4, 25, 42)
    for batch in plan:
        assert len(batch) == len({r["qid"] for r in batch}) == 4
    assert len({r["source_variant"] for b in plan for r in b}) > 1


def test_variable_arity_and_collision_fail_before_training():
    rows, _ = build_conditions(examples()[:4], candidates(examples()[:4]), ["C"], 1)
    rows["C"][0]["negatives"] = []
    with pytest.raises(ValueError, match="fixed negative"):
        batch_plan(rows["C"], "C", 2, 3, 42)
    same = [dict(qid=f"q{i}", query=f"q{i}", positive="same", negatives=[]) for i in range(3)]
    with pytest.raises(ValueError, match="collision-free"):
        batch_plan(same, "A", 2, 3, 42)


@pytest.mark.parametrize("k", [1, 2, 10])
def test_top_k_boundary_and_stable_ties(k):
    hits = retrieve_top_k(np.array([[1., 0.]]), np.array([[1., 0.], [1., 0.]]), ["b", "a"], k)
    assert [doc for doc, _ in hits[0]] == ["a", "b"][:k]
    assert hits == retrieve_top_k(np.array([[1., 0.]]), np.array([[1., 0.], [1., 0.]]), ["a", "b"], k)


def test_multiple_relevant_and_graded_ndcg():
    metrics = query_metrics({"a": 3, "b": 1}, ["b", "x"], 2)
    assert metrics["recall@2"] == 0.5
    assert metrics["hits@2"] == 1
    assert metrics["map@2"] == 0.5
    assert metrics["ndcg@2"] == pytest.approx(1 / (3 + 1 / math.log2(3)))
    _, metrics = evaluate_rankings([{"qid": "q", "query": "q", "gold_qid": "b"}],
                                   [[("a", .9), ("b", .8)]], 2)
    assert metrics["hits@2"] == 1  # old implementation incorrectly used recall@1
    assert metrics["mrr@2"] == 0.5


def test_invalid_and_empty_qrels_policy():
    assert query_metrics({}, ["a"], 10)["ndcg@10"] == 0
    with pytest.raises(ValueError, match="missing"):
        validate_evaluation([dict(qid="q", query="q", qrels={"absent": 1})], [dict(qid="a", text="a")])
    with pytest.raises(ValueError, match="duplicate"):
        query_metrics({"a": 1}, ["a", "a"], 2)


def test_eval_uses_training_prefixes_and_length(tmp_path):
    runner = script("09_eval_retriever")
    cfg = TrainingConfig(max_seq_length=19, query_prefix="Q:", document_prefix="D:")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "baseline_config.json").write_text(cfg.model_dump_json())
    class Encoder:
        calls = []
        def encode(self, texts, **kwargs):
            self.calls.append((texts, kwargs))
            return np.ones((len(texts), 2))
    model = Encoder()
    summary = runner.eval_one_model(model_path=str(model_dir), test_queries=[dict(qid="q", query="hello", qrels={"d": 1})],
        corpus=[dict(qid="d", text="world")], output_dir=tmp_path / "eval", top_k=10, batch_size=2,
        tag="test", model=model)
    assert model.calls[0][0] == ["D:world"] and model.calls[1][0] == ["Q:hello"]
    assert model.calls[0][1]["prompt"] == ""
    assert model.max_seq_length == 19 and summary["metrics"]["recall@10"] == 1


def test_bm25_excludes_positive_with_other_id():
    module = script("06_mine_bm25_negatives")
    index = module.BM25Index.__new__(module.BM25Index)
    index.docs = [dict(qid="other", d_plus=" Same  text "), dict(qid="n", d_plus="different")]
    index.tokenizer = str.split
    class Scores:
        def get_scores(self, _): return [10, 5]
    index.bm25 = Scores()
    assert index.top_k("q", 1, exclude_qid="q", exclude_text="same text")[0]["qid"] == "n"


def test_prepare_build_dry_run_cli_and_hash_guard(tmp_path):
    source, output = tmp_path / "source.jsonl", tmp_path / "build"
    save_jsonl(source, examples())
    def run(name, *args):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    run("07_build_train_data.py", "--rubq-full", source, "--output-dir", output,
        "--prepare-only", "--n-train", 10, "--n-dev", 3, "--n-test", 5)
    frozen_path = output / "split.json"
    original_split = frozen_path.read_text()
    different = json.loads(original_split)
    different["seed"] = 43
    alternative = tmp_path / "alternative_split.json"
    alternative.write_text(json.dumps(different))
    rejected = subprocess.run([sys.executable, str(ROOT / "scripts/07_build_train_data.py"),
        "--rubq-full", str(source), "--output-dir", str(output), "--split-file", str(alternative)],
        capture_output=True, text=True)
    assert rejected.returncode != 0 and "different frozen split" in rejected.stderr
    assert frozen_path.read_text() == original_split
    raw = candidates(examples())
    for name, rows in raw.items(): save_jsonl(tmp_path / (name + ".jsonl"), rows)
    run("07_build_train_data.py", "--rubq-full", source, "--output-dir", output,
        "--split-file", output / "split.json", "--cn-input", tmp_path / "cn.jsonl",
        "--bm25-input", tmp_path / "bm25.jsonl")
    run("08_train_retriever.py", "--train-dir", output, "--output-base", tmp_path / "dry",
        "--conditions", "A", "B", "C", "--dry-run", "--batch-size", 2, "--max-steps", 3)
    for c in "ABC":
        result = json.loads((tmp_path / "dry" / f"{c}_seed42" / "summary.json").read_text())
        assert result["n_queries"] == 10 and result["planned_steps"] == 3
        assert result["training_steps"] == 0
    with (output / "A_baseline.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(qid="extra")) + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_condition(output, "C")
    failure = subprocess.run([sys.executable, str(ROOT / "scripts/08_train_retriever.py"),
                              "--train-dir", str(output), "--dry-run"], capture_output=True, text=True)
    assert failure.returncode != 0


def test_comparison_rejects_incompatible_evaluation():
    module = script("10_compare_ablations")
    cfg = dict(top_k=10, query_prefix="Q", document_prefix="D", max_seq_length=256,
               gain="linear", empty_qrels="zero")
    results = {tag: dict(summary=dict(evaluation_hash="same", split="dev", config=cfg),
                         per_query=[dict(qid="q", metrics={"ndcg@10": value})])
               for tag, value in [("A", 0.2), ("C", 0.4)]}
    diff = module.compare(results, "A", samples=20)
    assert diff["C"]["delta_points"] == pytest.approx(20)
    assert diff["C"]["query_bootstrap_95ci_points"] == pytest.approx([20, 20])
    results["C"]["summary"]["evaluation_hash"] = "different"
    with pytest.raises(ValueError, match="different evaluation"):
        module.compare(results, "A")
    results["C"]["summary"]["evaluation_hash"] = "same"
    for tag in results:
        results[tag]["summary"]["training_identity"] = {"manifest_hash": tag}
    with pytest.raises(ValueError, match="different training"):
        module.compare(results, "A")


def test_real_mnrl_training_and_saved_model(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("sentence_transformers")
    models = local_model_modules()
    from src.baseline_training import train_steps
    # Bag-of-words plus a trainable projection: real ST and Torch, no downloads.
    torch.manual_seed(42)
    model = st.SentenceTransformer(modules=[models.BoW(["alpha", "beta", "gamma", "delta"]),
        models.Dense(in_features=4, out_features=3)])
    rows = [dict(qid="a", query="alpha", positive="alpha beta", negatives=["gamma delta"], source_variant="original"),
            dict(qid="b", query="beta", positive="beta delta", negatives=["alpha gamma"], source_variant="original")]
    cfg = TrainingConfig(batch_size=2, max_steps=3, query_prefix="", document_prefix="", lr=0.01)
    before = [p.detach().clone() for p in model.parameters()]
    history = train_steps(model, batch_plan(rows, "C", 2, 3, 42), cfg, tmp_path)
    assert len(history) == 3 and all(math.isfinite(r["loss"]) for r in history)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    restored = st.SentenceTransformer(str(tmp_path / "model"))
    assert restored.encode(["alpha"]).shape == (1, 3)
    evaluator = script("09_eval_retriever")
    result = evaluator.eval_one_model(model_path=str(tmp_path / "model"),
        test_queries=[dict(qid="q", query="alpha", qrels={"a": 2, "b": 1})],
        corpus=[dict(qid="a", text="alpha beta"), dict(qid="b", text="beta delta")],
        output_dir=tmp_path / "eval", top_k=10, batch_size=2, tag="tiny")
    assert result["metrics"]["recall@10"] == 1.0


def test_real_cli_end_to_end(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("sentence_transformers")
    models = local_model_modules()

    torch.manual_seed(42)
    vocabulary = ["query", "doc", "cn", "bm25", "negative"] + [str(i) for i in range(30)]
    model = st.SentenceTransformer(modules=[models.BoW(vocabulary),
        models.Dense(in_features=len(vocabulary), out_features=8)])
    base_model = tmp_path / "base_model"
    model.save(str(base_model))
    source, build = tmp_path / "source.jsonl", tmp_path / "build"
    training, evaluation = tmp_path / "training", tmp_path / "evaluation"
    save_jsonl(source, examples())

    def run(name, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
        assert result.returncode == 0, result.stdout + result.stderr

    run("07_build_train_data.py", "--rubq-full", source, "--output-dir", build,
        "--prepare-only", "--n-train", 10, "--n-dev", 3, "--n-test", 5)
    for name, rows in candidates(examples()).items():
        save_jsonl(tmp_path / (name + ".jsonl"), rows)
    run("07_build_train_data.py", "--rubq-full", source, "--output-dir", build,
        "--split-file", build / "split.json", "--cn-input", tmp_path / "cn.jsonl",
        "--bm25-input", tmp_path / "bm25.jsonl")
    run("08_train_retriever.py", "--conditions", "A", "B", "C", "--train-dir", build,
        "--output-base", training, "--base-model", base_model, "--batch-size", 2,
        "--max-steps", 2, "--query-prefix", "", "--document-prefix", "")
    for condition in "ABC":
        directory = training / f"{condition}_seed42"
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["status"] == "trained" and summary["training_steps"] == 2
        assert (directory / "model" / "baseline_config.json").exists()
    run("09_eval_retriever.py", "--baseline", "--base-model", base_model,
        "--train-dir", build, "--output-base", evaluation,
        "--query-prefix", "", "--document-prefix", "")
    run("09_eval_retriever.py", "--conditions", "A", "B", "C", "--training-base", training,
        "--train-dir", build, "--output-base", evaluation)
    run("10_compare_ablations.py", "--eval-base", evaluation / "dev", "--bootstrap-samples", 20)
    result = json.loads((evaluation / "dev" / "comparison_seed42.json").read_text())
    assert set(result["results"]) == {"BASE", "A_seed42", "B_seed42", "C_seed42"}
    assert all(math.isfinite(r["mean"]) for r in result["results"].values())
    assert (evaluation / "dev" / "comparison_seed42.md").exists()

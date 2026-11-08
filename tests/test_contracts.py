import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.config import EncodersConfig, ExperimentConfig, LLMConfig, load_config
from src.data_io import load_query_docs
from src.encoders import EncoderPool
from src.llm_client import LLMConfig as LegacyLLMConfig
from src.schemas import Fact, QueryDoc, PositiveRecord

ROOT = Path(__file__).resolve().parents[1]


def config_dict():
    return load_config(ROOT / "configs/default.yaml").model_dump()


def test_shipped_config_and_round_trip():
    cfg = load_config(ROOT / "configs/default.yaml")
    assert cfg.llm.backend == "llama_cpp"
    assert len(cfg.encoders.ensemble) == 2
    assert ExperimentConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert LegacyLLMConfig is LLMConfig


@pytest.mark.parametrize("field,value", [
    ("backend", "olama"), ("temperature", -1.0), ("top_p", 1.1),
    ("max_new_tokens", "512"), ("max_new_tokens", True),
    ("load_in_4bit", "false"), ("temprature", 0.3),
    ("temperature", float("nan")), ("ollama_base_url", "localhost:11434"),
])
def test_bad_llm_config_rejected(field, value):
    with pytest.raises(ValidationError):
        LLMConfig.from_dict({field: value})


def test_assignment_is_validated():
    cfg = LLMConfig()
    with pytest.raises(ValidationError):
        cfg.temperature = -1


@pytest.mark.parametrize("ensemble", [[], [{"name": "x", "weight": 0.0}],
    [{"name": "x"}, {"name": "x"}]])
def test_encoder_validation_precedes_loading(ensemble):
    with patch("src.encoders.SingleEncoder") as encoder:
        with pytest.raises(ValidationError):
            EncoderPool.from_config({"ensemble": ensemble})
        encoder.assert_not_called()


def test_encoder_factory_accepts_typed_config():
    cfg = EncodersConfig(ensemble=[{"name": "x"}], device="cpu", batch_size=2)
    with patch("src.encoders.SingleEncoder") as encoder:
        encoder.return_value.spec = cfg.ensemble[0]
        pool = EncoderPool.from_config(cfg)
        encoder.assert_called_once_with(cfg.ensemble[0], device="cpu", max_length=512)
        assert pool.weights == [1.0]
        assert pool.batch_size == 2


def test_cross_field_validation():
    data = config_dict()
    data["negatives"]["delta_buckets"][1]["min"] = 0.4
    with pytest.raises(ValidationError, match="overlap"):
        ExperimentConfig.model_validate(data)
    data = config_dict()
    data["cross_encoder"]["irrelevance_threshold"] = 0.9
    with pytest.raises(ValidationError, match="irrelevance_threshold"):
        ExperimentConfig.model_validate(data)


@pytest.mark.parametrize("content", ["", "[]", "llm: [", "unknown: 1"])
def test_invalid_yaml_has_path(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match="bad.yaml"):
        load_config(path)


def test_fact_validation_does_not_coerce_llm_output():
    assert Fact.from_dict({"text": "факт", "type": "entity", "criticality": 3}).text == "факт"
    for criticality in [0, 6, "3", True]:
        with pytest.raises(ValidationError):
            Fact.from_dict({"text": "факт", "type": "entity", "criticality": criticality})


def test_query_doc_preserves_source_offsets_and_meta():
    record = QueryDoc(qid="q", query="вопрос", d_plus="  Текст.  ", meta={"tags": ["a"]})
    assert record.d_plus == "  Текст.  "
    assert QueryDoc.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValidationError):
        QueryDoc(qid="q", query="  ", d_plus="Текст")


def test_jsonl_errors_include_line_and_duplicates(tmp_path):
    path = tmp_path / "input.jsonl"
    record = {"qid": "q", "query": "вопрос", "d_plus": "текст"}
    path.write_text(json.dumps(record) + "\n\n{}", encoding="utf-8")
    with pytest.raises(ValueError, match="input.jsonl:3"):
        load_query_docs(path)
    path.write_text((json.dumps(record) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate qid"):
        load_query_docs(path)


def test_legacy_record_serialization():
    record = PositiveRecord(
        qid="q", query="q", d_plus="d", d_plus_i="variant", method="gradient",
        target_level="near", similarity_to_d_plus=0.9, sim_per_encoder={"x": 0.9},
        ce_score=-1.0, llm_judge="skipped", passed_filters=True,
    )
    assert record.to_dict()["ce_score"] == -1.0
    assert "schema_version" not in record.to_dict()  # preserve artifact shape


def test_production_fact_boundary():
    spec = importlib.util.spec_from_file_location("cn_runner", ROOT / "scripts/05_run_cn_pipeline.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    class FakeLLM:
        def generate(self, _):
            return [json.dumps([
                {"text": "факт", "type": "entity", "criticality": 3},
                {"text": "факт", "type": "entity", "criticality": "3"},
                {"text": "факт", "type": "entity", "criticality": 8},
            ])]
    facts, _ = runner.extract_facts(FakeLLM(), {"query": "q", "d_plus": "факт"},
                                  min_criticality=2, max_facts=5)
    assert len(facts) == 1

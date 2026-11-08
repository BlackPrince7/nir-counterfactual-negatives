"""Validated configuration shared by YAML runners and model factories.

No model downloads or GPU imports are needed to validate configuration.
"""
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmpty = Annotated[str, Field(min_length=1, pattern=r"\S")]
PositiveInt = Annotated[int, Field(gt=0)]
UnitFloat = Annotated[float, Field(ge=0, le=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, validate_assignment=True,
        validate_default=True, allow_inf_nan=False,
    )


class LLMConfig(StrictModel):
    model_name: NonEmpty = "Qwen/Qwen2.5-7B-Instruct"
    backend: Literal["llama_cpp", "ollama", "vllm", "transformers"] = "llama_cpp"
    max_new_tokens: PositiveInt = 1024
    temperature: Annotated[float, Field(ge=0)] = 0.7
    top_p: Annotated[float, Field(gt=0, le=1)] = 0.9
    judge_mode: Literal["cot", "fast"] = "cot"
    judge_temperature: Annotated[float, Field(ge=0)] = 0.0
    judge_max_new_tokens: PositiveInt = 8
    judge_cot_max_new_tokens: PositiveInt = 320
    gguf_path: NonEmpty = "W:/models/qwen2.5-7b-q4.gguf"
    n_gpu_layers: Annotated[int, Field(ge=-1)] = -1
    n_ctx: PositiveInt = 4096
    ollama_model: NonEmpty = "qwen2.5:7b"
    ollama_base_url: Annotated[str, Field(pattern=r"^https?://[^/\s]+.*$")] = "http://localhost:11434"
    load_in_4bit: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """Compatibility entry point; unknown keys now raise validation errors."""
        return cls.model_validate(data)


class EncoderSpec(StrictModel):
    name: NonEmpty
    weight: Annotated[float, Field(gt=0)] = 1.0
    query_prefix: str = ""
    passage_prefix: str = ""


class EncodersConfig(StrictModel):
    ensemble: Annotated[list[EncoderSpec], Field(min_length=1)]
    device: NonEmpty = "cuda"
    batch_size: PositiveInt = 32
    max_length: PositiveInt = 512

    @model_validator(mode="after")
    def unique_names(self) -> "EncodersConfig":
        names = [e.name for e in self.ensemble]
        if len(names) != len(set(names)):
            raise ValueError("encoder names must be unique (used as diagnostic keys)")
        return self


class CrossEncoderConfig(StrictModel):
    model_name: NonEmpty = "BAAI/bge-reranker-v2-m3"
    device: NonEmpty = "cuda"
    batch_size: PositiveInt = 16
    relevance_threshold: float = 0.5
    irrelevance_threshold: float = 0.2

    @model_validator(mode="after")
    def ordered_thresholds(self) -> "CrossEncoderConfig":
        if self.irrelevance_threshold > self.relevance_threshold:
            raise ValueError("irrelevance_threshold must be <= relevance_threshold")
        return self


class BM25Config(StrictModel):
    similarity_max: UnitFloat = 0.6


class DeltaBucket(StrictModel):
    name: NonEmpty
    min: float
    max: float
    target_per_query: PositiveInt = 1

    @model_validator(mode="after")
    def ordered_bounds(self) -> "DeltaBucket":
        if self.min >= self.max:
            raise ValueError("bucket min must be < max")
        return self


class NegConfig(StrictModel):
    max_facts_per_doc: PositiveInt = 5
    max_mutations_per_fact: PositiveInt = 1
    min_criticality: Annotated[int, Field(ge=1, le=5)] = 3


class NegativesConfig(NegConfig):
    delta_buckets: list[DeltaBucket] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_buckets(self) -> "NegativesConfig":
        buckets = sorted(self.delta_buckets, key=lambda b: b.min)
        if len({b.name for b in buckets}) != len(buckets):
            raise ValueError("bucket names must be unique")
        if any(a.max > b.min for a, b in zip(buckets, buckets[1:])):
            raise ValueError("delta buckets must not overlap")
        return self


class PosTarget(StrictModel):
    name: NonEmpty
    target: Annotated[float, Field(ge=-1, le=1)]
    tolerance: Annotated[float, Field(gt=0, le=2)]


class PosConfig(StrictModel):
    branches_per_target: PositiveInt = 3
    targets: Annotated[list[PosTarget], Field(min_length=1)]
    must_remain_positive: bool = True


class PositivesConfig(StrictModel):
    branches_per_target: PositiveInt = 3
    similarity_targets: Annotated[list[PosTarget], Field(min_length=1)]
    must_remain_positive: bool = True


class RetrieverConfig(StrictModel):
    model_name: NonEmpty = "ai-forever/ru-en-RoSBERTa"
    reason: str = ""


class IOConfig(StrictModel):
    # Strings preserve Windows paths when config is inspected on Linux.
    input_data: NonEmpty = "data/sample_rubq.jsonl"
    output_dir: NonEmpty = "outputs/"
    negatives_file: NonEmpty = "negatives.jsonl"
    positives_file: NonEmpty = "positives.jsonl"
    log_file: NonEmpty = "run.log"


class RunConfig(StrictModel):
    seed: Annotated[int, Field(ge=0)] = 42
    limit: PositiveInt | None = None
    verbose: bool = True


class ExperimentConfig(StrictModel):
    schema_version: Literal[1] = 1
    llm: LLMConfig
    encoders: EncodersConfig
    retriever_to_train: RetrieverConfig = Field(default_factory=RetrieverConfig)
    cross_encoder: CrossEncoderConfig = Field(default_factory=CrossEncoderConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    negatives: NegativesConfig = Field(default_factory=NegativesConfig)
    positives: PositivesConfig
    io: IOConfig = Field(default_factory=IOConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load explicitly selected YAML; never silently read local overrides."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return ExperimentConfig.model_validate(data)
    except (yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"Invalid configuration {path}:\n{exc}") from exc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate config without loading models")
    parser.add_argument("path", nargs="?", default="configs/default.yaml")
    parser.add_argument("--schema", action="store_true", help="Print JSON Schema instead")
    args = parser.parse_args()
    if args.schema:
        import json
        print(json.dumps(ExperimentConfig.model_json_schema(), ensure_ascii=False, indent=2))
    else:
        print(load_config(args.path).model_dump_json(indent=2))

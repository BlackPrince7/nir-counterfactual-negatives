"""Validated pipeline records; existing JSON field names are preserved."""
from __future__ import annotations

from pydantic import Field
from .config import StrictModel, NonEmpty
from typing import Optional


class QueryDoc(StrictModel):
    """Входная пара: запрос + положительный документ."""
    qid: NonEmpty
    query: NonEmpty
    d_plus: NonEmpty
    meta: dict = Field(default_factory=dict)


class Fact(StrictModel):
    """Извлечённый из d⁺ атомарный факт."""
    text: NonEmpty
    type: NonEmpty
    criticality: int = Field(ge=1, le=5)

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls.model_validate(d)


class NegativeRecord(StrictModel):
    """Сгенерированный негатив со всеми диагностиками."""
    qid: NonEmpty
    query: NonEmpty
    d_plus: NonEmpty
    d_minus: NonEmpty
    method: str                       # "counterfactual"
    fact_text: str                    # какой факт мутировали
    fact_type: str
    fact_criticality: int = Field(ge=1, le=5)
    delta: float                      # ансамблевое значение
    delta_per_encoder: dict[str, float]
    bm25_overlap: float
    ce_score: float                   # cross-encoder relevance(q, d⁻)
    llm_judge: str                    # "yes" | "no"
    difficulty_bucket: Optional[str]  # "easy" | "medium" | "hard"
    passed_filters: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class PositiveRecord(StrictModel):
    """Сгенерированный позитивный вариант d⁺ᵢ."""
    qid: NonEmpty
    query: NonEmpty
    d_plus: NonEmpty
    d_plus_i: NonEmpty
    method: str                       # "gradient"
    target_level: str                 # "near" | "mid" | "far"
    similarity_to_d_plus: float       # ансамблевое значение
    sim_per_encoder: dict[str, float]
    ce_score: float                   # CE(q, dᵢ) — должен оставаться высоким
    llm_judge: str
    passed_filters: bool
    rejection_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")

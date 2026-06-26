#!/usr/bin/env python3
"""
Phase 0: Sanity check Qwen2.5-7B на рукотворных 15 примерах.

Что измеряем:
  1. Fact extraction:
     - доля валидного JSON
     - hallucination rate (факт не verbatim в d⁺)
     - recall vs expected_mutable_facts
     - распределение criticality
  2. Counterfactual mutation:
     - self-similarity rate (LLM вернула оригинал)
     - length preservation ratio
  3. LLM judge:
     - precision на (q, d⁺)         → ждём "Да"
     - recall  на (q, unrelated_d)  → ждём "Нет"
     - на (q, d⁻)                    → ждём "Нет"

Выход:
  outputs/sanity/
    ├── facts.jsonl         — все извлечённые факты
    ├── counterfactuals.jsonl
    ├── judge.jsonl
    ├── metrics.json
    └── report.md           — для человеческого просмотра

Запуск:
  python scripts/00_sanity_qwen.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import yaml

# чтобы импортировать src/ из родительского каталога
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm_client import LLMConfig, make_client, extract_json, parse_yes_no  # noqa: E402
from src.prompts import (  # noqa: E402
    build_fact_extraction_messages,
    build_counterfactual_messages,
    build_judge_messages,
)

log = logging.getLogger("sanity")


# ──────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fuzzy_in(needle: str, haystack: str) -> bool:
    """Грубая проверка: extracted_fact ↔ expected_fact матчатся, если один
    содержится в другом (без учёта регистра)."""
    n, h = needle.strip().lower(), haystack.strip().lower()
    return n in h or h in n


# ──────────────────────────────────────────────────────────────────────
# Этап 1: Fact extraction
# ──────────────────────────────────────────────────────────────────────

def run_fact_extraction(llm, examples: list[dict]) -> list[dict]:
    """Возвращает по записи на каждый пример с полной диагностикой."""
    log.info("=== Этап 1: Fact extraction ===")
    # Готовим батч из всех примеров — vLLM прожуёт за один проход
    messages_batch = [
        build_fact_extraction_messages(ex["query"], ex["d_plus"])
        for ex in examples
    ]
    raw_outputs = llm.generate(messages_batch)

    results = []
    for ex, raw in zip(examples, raw_outputs):
        parsed = extract_json(raw)
        valid_json = isinstance(parsed, list)

        all_facts, valid_facts, hallucinated = [], [], []
        if valid_json:
            for item in parsed:
                if not isinstance(item, dict) or "text" not in item:
                    continue
                fact = {
                    "text": str(item.get("text", "")).strip(),
                    "type": str(item.get("type", "")).strip(),
                    "criticality": int(item.get("criticality", 0)),
                }
                all_facts.append(fact)
                if fact["text"] in ex["d_plus"]:
                    valid_facts.append(fact)
                else:
                    hallucinated.append(fact)

        # Recall: сколько expected_facts покрыты валидными extracted_facts
        expected = ex["expected_mutable_facts"]
        covered = [
            e for e in expected
            if any(fuzzy_in(e, f["text"]) for f in valid_facts)
        ]
        recall = len(covered) / len(expected) if expected else 0.0

        results.append({
            "qid": ex["qid"],
            "query": ex["query"],
            "d_plus": ex["d_plus"],
            "expected_facts": expected,
            "raw_output": raw,
            "valid_json": valid_json,
            "all_facts": all_facts,
            "valid_facts": valid_facts,
            "hallucinated_facts": hallucinated,
            "covered_expected": covered,
            "recall_vs_expected": recall,
        })
        log.info(
            "  %s: json=%s facts=%d valid=%d hallu=%d recall=%.2f",
            ex["qid"], valid_json, len(all_facts),
            len(valid_facts), len(hallucinated), recall,
        )
    return results


# ──────────────────────────────────────────────────────────────────────
# Этап 2: Counterfactual mutation для top-K фактов
# ──────────────────────────────────────────────────────────────────────

def run_counterfactuals(
    llm, fact_results: list[dict], k_facts: int = 3
) -> list[dict]:
    log.info("=== Этап 2: Counterfactual mutation (top-%d по criticality) ===", k_facts)
    # Соберём пары (fact_result, fact_dict) — по top-K на пример
    pairs, batch_msgs = [], []
    for fr in fact_results:
        top = sorted(fr["valid_facts"], key=lambda f: -f["criticality"])[:k_facts]
        for fact in top:
            pairs.append((fr, fact))
            batch_msgs.append(build_counterfactual_messages(
                query=fr["query"], d_plus=fr["d_plus"],
                fact_text=fact["text"], fact_type=fact["type"],
            ))

    if not batch_msgs:
        log.warning("Нет фактов для мутации — пропускаем этап 2.")
        return []

    outputs = llm.generate(batch_msgs)

    cf_results = []
    for (fr, fact), raw in zip(pairs, outputs):
        d_plus = fr["d_plus"]
        d_minus = raw.strip()
        # Очистка от кавычек-обёртки, если LLM их добавила
        if d_minus.startswith('"') and d_minus.endswith('"') and len(d_minus) > 2:
            d_minus = d_minus[1:-1].strip()

        is_self = d_minus == d_plus.strip()
        length_ratio = len(d_minus) / len(d_plus) if d_plus else 0.0
        fact_still_present = fact["text"] in d_minus  # должно быть False!

        cf_results.append({
            "qid": fr["qid"],
            "query": fr["query"],
            "d_plus": d_plus,
            "fact_text": fact["text"],
            "fact_type": fact["type"],
            "fact_criticality": fact["criticality"],
            "d_minus": d_minus,
            "raw_output": raw,
            "is_self_copy": is_self,
            "length_ratio": length_ratio,
            "fact_still_present": fact_still_present,
        })
        log.info(
            "  %s [%s]: self=%s len_ratio=%.2f fact_still_in_d-=%s",
            fr["qid"], fact["text"][:30], is_self,
            length_ratio, fact_still_present,
        )
    return cf_results


# ──────────────────────────────────────────────────────────────────────
# Этап 3: LLM judge
# ──────────────────────────────────────────────────────────────────────

def run_judge(
    llm, examples: list[dict], cf_results: list[dict], llm_cfg: LLMConfig
) -> dict:
    log.info("=== Этап 3: LLM judge ===")

    by_qid = {e["qid"]: e for e in examples}

    # Готовим все пары для одного батч-прогона
    pairs, kinds, qids = [], [], []
    for ex in examples:
        pairs.append((ex["query"], ex["d_plus"]))
        kinds.append("positive_control")
        qids.append(ex["qid"])

        pairs.append((ex["query"], ex["unrelated_d"]))
        kinds.append("negative_control")
        qids.append(ex["qid"])

    for cf in cf_results:
        if cf["is_self_copy"]:
            continue  # бессмысленно судить копию d⁺
        pairs.append((cf["query"], cf["d_minus"]))
        kinds.append("counterfactual")
        qids.append(cf["qid"])

    batch_msgs = [build_judge_messages(q, d) for q, d in pairs]
    outputs = llm.generate(
        batch_msgs,
        temperature=llm_cfg.judge_temperature,
        max_new_tokens=llm_cfg.judge_max_new_tokens,
    )

    verdicts = []
    for (q, d), kind, qid, raw in zip(pairs, kinds, qids, outputs):
        v = parse_yes_no(raw)
        # Ожидание: positive_control → yes; остальные → no
        expected = "yes" if kind == "positive_control" else "no"
        verdicts.append({
            "qid": qid,
            "kind": kind,
            "query": q,
            "doc": d,
            "raw_output": raw,
            "verdict": v,
            "expected": expected,
            "correct": v == expected,
        })
    return {"verdicts": verdicts}


# ──────────────────────────────────────────────────────────────────────
# Метрики
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(facts, cfs, judge):
    n = len(facts)
    # Fact extraction
    json_ok_rate = sum(f["valid_json"] for f in facts) / n if n else 0.0
    facts_per_doc = [len(f["valid_facts"]) for f in facts]
    hallu_rate = (
        sum(len(f["hallucinated_facts"]) for f in facts)
        / max(1, sum(len(f["all_facts"]) for f in facts))
    )
    recalls = [f["recall_vs_expected"] for f in facts]
    crits = [it["criticality"] for f in facts for it in f["valid_facts"]]

    # Counterfactual
    cf_self = (
        sum(c["is_self_copy"] for c in cfs) / len(cfs) if cfs else 0.0
    )
    cf_len_ratios = [c["length_ratio"] for c in cfs]
    fact_leak = (
        sum(c["fact_still_present"] for c in cfs) / len(cfs) if cfs else 0.0
    )

    # Judge — точность по типам
    by_kind = {"positive_control": [], "negative_control": [], "counterfactual": []}
    for v in judge["verdicts"]:
        by_kind[v["kind"]].append(v["correct"])
    j_pos = mean(by_kind["positive_control"]) if by_kind["positive_control"] else None
    j_neg = mean(by_kind["negative_control"]) if by_kind["negative_control"] else None
    j_cf  = mean(by_kind["counterfactual"])    if by_kind["counterfactual"]    else None

    return {
        "n_examples": n,
        "fact_extraction": {
            "json_parse_rate": json_ok_rate,
            "facts_per_doc_mean": mean(facts_per_doc) if facts_per_doc else 0,
            "facts_per_doc_stdev": stdev(facts_per_doc) if len(facts_per_doc) > 1 else 0,
            "hallucination_rate": hallu_rate,
            "recall_vs_expected_mean": mean(recalls) if recalls else 0,
            "criticality_mean": mean(crits) if crits else 0,
            "criticality_dist": dict(Counter(crits)),
        },
        "counterfactual": {
            "n_generated": len(cfs),
            "self_copy_rate": cf_self,
            "length_ratio_mean": mean(cf_len_ratios) if cf_len_ratios else 0,
            "length_ratio_stdev": stdev(cf_len_ratios) if len(cf_len_ratios) > 1 else 0,
            "fact_still_present_rate": fact_leak,  # хотим близко к 0
        },
        "judge": {
            "accuracy_on_positive_q_dplus": j_pos,
            "accuracy_on_negative_q_unrelated": j_neg,
            "accuracy_on_counterfactual_q_dminus": j_cf,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Markdown отчёт
# ──────────────────────────────────────────────────────────────────────

def write_report(facts, cfs, judge, metrics, output_dir: Path):
    lines = ["# Phase 0 — Sanity Report", ""]

    # Сводка
    lines.append("## Сводные метрики\n")
    fe = metrics["fact_extraction"]
    cm = metrics["counterfactual"]
    jm = metrics["judge"]
    lines.append("### Fact extraction")
    lines.append(f"- JSON parse rate: **{fe['json_parse_rate']:.1%}**")
    lines.append(f"- Фактов на документ: **{fe['facts_per_doc_mean']:.1f} ± {fe['facts_per_doc_stdev']:.1f}**")
    lines.append(f"- Hallucination rate (факт не в d⁺): **{fe['hallucination_rate']:.1%}**")
    lines.append(f"- Recall vs expected_mutable_facts: **{fe['recall_vs_expected_mean']:.1%}**")
    lines.append(f"- Средняя criticality: **{fe['criticality_mean']:.2f}**, распределение: `{fe['criticality_dist']}`")
    lines.append("")
    lines.append("### Counterfactual")
    lines.append(f"- Сгенерировано: **{cm['n_generated']}**")
    lines.append(f"- Self-copy rate (вернули оригинал): **{cm['self_copy_rate']:.1%}** *(хотим 0)*")
    lines.append(f"- Length ratio (len(d⁻)/len(d⁺)): **{cm['length_ratio_mean']:.2f} ± {cm['length_ratio_stdev']:.2f}** *(хотим ~1.0)*")
    lines.append(f"- Старый факт ещё в d⁻: **{cm['fact_still_present_rate']:.1%}** *(хотим 0)*")
    lines.append("")
    lines.append("### Judge")
    lines.append(f"- Acc на (q, d⁺) → ожидание «Да»: **{jm['accuracy_on_positive_q_dplus']:.1%}** *(хотим 1.0)*")
    lines.append(f"- Acc на (q, unrelated) → ожидание «Нет»: **{jm['accuracy_on_negative_q_unrelated']:.1%}** *(хотим 1.0)*")
    lines.append(f"- Acc на (q, d⁻) → ожидание «Нет»: **{jm['accuracy_on_counterfactual_q_dminus']:.1%}**")
    lines.append("")

    # Per-example
    lines.append("## Подробно по примерам")
    lines.append("")
    cf_by_qid = {}
    for c in cfs:
        cf_by_qid.setdefault(c["qid"], []).append(c)
    judge_by_qid = {}
    for v in judge["verdicts"]:
        judge_by_qid.setdefault(v["qid"], []).append(v)

    for f in facts:
        qid = f["qid"]
        lines.append(f"### `{qid}`: {f['query']}")
        lines.append("")
        lines.append(f"**d⁺**: {f['d_plus']}")
        lines.append("")
        lines.append(f"**Ожидаемые факты**: {f['expected_facts']}")
        lines.append(f"**Покрыто**: {f['covered_expected']} (recall={f['recall_vs_expected']:.0%})")
        lines.append("")
        if not f["valid_json"]:
            lines.append("⚠️  **JSON не распарсился**. Raw:")
            lines.append("```")
            lines.append(f["raw_output"][:500])
            lines.append("```")

        if f["valid_facts"]:
            lines.append("**Извлечённые факты:**")
            lines.append("")
            lines.append("| text | type | crit |")
            lines.append("|---|---|---|")
            for ff in f["valid_facts"]:
                tx = ff["text"].replace("|", "\\|")
                ty = ff["type"].replace("|", "\\|")
                lines.append(f"| {tx} | {ty} | {ff['criticality']} |")
            lines.append("")
        if f["hallucinated_facts"]:
            lines.append("⚠️  **Hallucinated (нет в d⁺):**")
            for ff in f["hallucinated_facts"]:
                lines.append(f"- `{ff['text']}` (type={ff['type']}, crit={ff['criticality']})")
            lines.append("")

        # Контрфакты
        for c in cf_by_qid.get(qid, []):
            lines.append(f"**Контрфакт по факту**: `{c['fact_text']}` ({c['fact_type']}, crit={c['fact_criticality']})")
            lines.append("")
            lines.append(f"_d⁻_: {c['d_minus']}")
            lines.append("")
            flags = []
            if c["is_self_copy"]: flags.append("❌ self-copy")
            if c["fact_still_present"]: flags.append("⚠️ старый факт ещё внутри")
            if not (0.7 <= c["length_ratio"] <= 1.4):
                flags.append(f"⚠️ длина {c['length_ratio']:.2f}x")
            lines.append(f"_len ratio={c['length_ratio']:.2f}_  {' | '.join(flags) if flags else '✓'}")
            lines.append("")

        # Verdicts
        verdicts = judge_by_qid.get(qid, [])
        if verdicts:
            lines.append("**Judge verdicts:**")
            lines.append("")
            lines.append("| тип | вердикт | ожидание | OK |")
            lines.append("|---|---|---|---|")
            for v in verdicts:
                ok = "✓" if v["correct"] else "❌"
                lines.append(f"| {v['kind']} | {v['verdict']} | {v['expected']} | {ok} |")
            lines.append("")
        lines.append("---")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Отчёт записан в %s", report_path)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--examples", default="data/sanity_examples.jsonl")
    ap.add_argument("--output-dir", default="outputs/sanity")
    ap.add_argument("--limit", type=int, default=None,
                    help="ограничить число примеров (для отладки)")
    ap.add_argument("--k-facts", type=int, default=3,
                    help="сколько top-критичных фактов мутировать на пример")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    examples = load_jsonl(Path(args.examples))
    if args.limit:
        examples = examples[: args.limit]
    log.info("Загружено %d примеров", len(examples))

    llm_cfg = LLMConfig.from_dict(cfg["llm"])
    log.info("LLM: %s (backend=%s)", llm_cfg.model_name, llm_cfg.backend)
    llm = make_client(llm_cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    facts = run_fact_extraction(llm, examples)
    save_jsonl(out_dir / "facts.jsonl", facts)

    cfs = run_counterfactuals(llm, facts, k_facts=args.k_facts)
    save_jsonl(out_dir / "counterfactuals.jsonl", cfs)

    judge = run_judge(llm, examples, cfs, llm_cfg)
    save_jsonl(out_dir / "judge.jsonl", judge["verdicts"])

    metrics = compute_metrics(facts, cfs, judge)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(facts, cfs, judge, metrics, out_dir)

    # Краткая сводка в консоль
    print("\n" + "=" * 60)
    print("PHASE 0 SUMMARY")
    print("=" * 60)
    fe = metrics["fact_extraction"]
    cm = metrics["counterfactual"]
    jm = metrics["judge"]
    print(f"Fact extraction:")
    print(f"  JSON parse:      {fe['json_parse_rate']:.1%}")
    print(f"  Facts/doc:       {fe['facts_per_doc_mean']:.1f} ± {fe['facts_per_doc_stdev']:.1f}")
    print(f"  Hallucination:   {fe['hallucination_rate']:.1%}")
    print(f"  Recall expected: {fe['recall_vs_expected_mean']:.1%}")
    print(f"Counterfactual:")
    print(f"  Self-copy:       {cm['self_copy_rate']:.1%}  (want 0)")
    print(f"  Length ratio:    {cm['length_ratio_mean']:.2f}±{cm['length_ratio_stdev']:.2f}  (want ~1.0)")
    print(f"  Fact still in:   {cm['fact_still_present_rate']:.1%}  (want 0)")
    print(f"Judge:")
    print(f"  (q,d⁺)→Да:       {jm['accuracy_on_positive_q_dplus']:.1%}  (want 1.0)")
    print(f"  (q,unrel)→Нет:   {jm['accuracy_on_negative_q_unrelated']:.1%}  (want 1.0)")
    print(f"  (q,d⁻)→Нет:      {jm['accuracy_on_counterfactual_q_dminus']:.1%}")
    print(f"\nПодробный отчёт: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()

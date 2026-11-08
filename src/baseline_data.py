"""CPU-only split, cohort and batch contracts for controlled ablations."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

CONDITION_FILES = dict(A="A_baseline.jsonl", B="B_bm25_hn.jsonl", C="C_cn.jsonl",
                       D="D_gp.jsonl", E="E_cn_gp.jsonl")


def normalize(text):
    return " ".join(text.split()).casefold()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def groups(examples):
    """Connected components: shared query OR source document stays in one split."""
    parent = list(range(len(examples)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, ex in enumerate(examples):
        for field in ("query", "d_plus"):
            key = (field, normalize(ex[field]))
            if key in seen:
                parent[root(i)] = root(seen[key])
            seen[key] = i
    result = defaultdict(list)
    for i, ex in enumerate(examples):
        result[root(i)].append(ex)
    return list(result.values())


def make_split(examples, n_train, n_dev, n_test, seed, policy="official"):
    if min(n_train, n_dev, n_test) <= 0:
        raise ValueError("train/dev/test sizes must be positive")
    examples = sorted(examples, key=lambda ex: ex["qid"])
    if len({ex["qid"] for ex in examples}) != len(examples):
        raise ValueError("duplicate input qid")
    components = groups(examples)
    rng = random.Random(seed)
    rng.shuffle(components)
    excluded = []
    if policy == "official":
        train_groups, test_groups = [], []
        for group in components:
            labels = {e.get("meta", {}).get("source_split") for e in group}
            if not labels <= {"train", "dev", "test"}:
                raise ValueError("official policy requires meta.source_split on every example")
            if "test" in labels:
                test_groups.append([e for e in group if e["meta"]["source_split"] == "test"])
                excluded.extend(e["qid"] for e in group if e["meta"]["source_split"] != "test")
            else:
                train_groups.append(group)
    elif policy == "custom-grouped":
        train_groups, test_groups = components, None
    else:
        raise ValueError("unknown split policy")
    def take(pool, count):
        chosen, remainder = [], []
        for group in pool:
            if len(chosen) < count:
                chosen.extend(group)
            else:
                remainder.append(group)
        if len(chosen) < count:
            raise ValueError(f"requested {count} examples, only {len(chosen)} available")
        return chosen, remainder
    # Whole groups may exceed requested counts; record actual sizes, never split groups.
    if test_groups is None:
        test, train_groups = take(train_groups, n_test)
    else:
        test, _ = take(test_groups, n_test)
    dev, train_groups = take(train_groups, n_dev)
    train, _ = take(train_groups, n_train)
    return {"schema_version": 2, "policy": policy, "seed": seed,
            "source_hash": digest(examples), "excluded_source_overlap": sorted(excluded),
            "requested_sizes": dict(train=n_train, dev=n_dev, test=n_test),
            **{f"{name}_qids": sorted(e["qid"] for e in rows)
               for name, rows in (("train", train), ("dev", dev), ("test", test))}}


def validate_split(examples, split):
    if split.get("schema_version") != 2 or split.get("source_hash") != digest(sorted(examples, key=lambda e: e["qid"])):
        raise ValueError("split/source mismatch: prepare a new versioned split")
    if split.get("policy") not in {"official", "custom-grouped"}:
        raise ValueError("unknown split policy")
    by_id = {e["qid"]: e for e in examples}
    seen_ids, seen_queries, seen_docs = set(), set(), set()
    for name in ("train", "dev", "test"):
        ids = split[f"{name}_qids"]
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= by_id.keys():
            raise ValueError(f"invalid {name} IDs")
        rows = [by_id[qid] for qid in ids]
        if split["policy"] == "official":
            allowed = {"test"} if name == "test" else {"train", "dev"}
            if any(e.get("meta", {}).get("source_split") not in allowed for e in rows):
                raise ValueError("official source split violation")
        queries = {normalize(e["query"]) for e in rows}
        docs = {normalize(e["d_plus"]) for e in rows}
        if seen_ids.intersection(ids) or seen_queries & queries or seen_docs & docs:
            raise ValueError("query or document leakage across splits")
        seen_ids.update(ids)
        seen_queries.update(queries)
        seen_docs.update(docs)


def build_conditions(train, candidates, conditions, k):
    if k < 1:
        raise ValueError("negative count must be positive")
    by_id = {e["qid"]: e for e in train}
    required = set()
    if set(conditions) & {"B"}: required.add("bm25")
    if set(conditions) & {"C", "E"}: required.add("cn")
    if set(conditions) & {"D", "E"}: required.add("gp")
    found = {source: defaultdict(list) for source in required}
    ignored = defaultdict(int)
    for source in required:
        for row in candidates.get(source, []):
            qid = row.get("qid")
            if qid not in by_id:
                ignored[source] += 1
                continue
            ex = by_id[qid]
            if row.get("query", ex["query"]) != ex["query"] or row.get("d_plus", ex["d_plus"]) != ex["d_plus"]:
                raise ValueError(f"stale {source} candidate for {qid}")
            if source == "gp":
                passed = row.get("passed_filters", row.get("llm_verdict") == "yes" and not row.get("stop_reason"))
                text = row.get("text") or row.get("d_plus_i")
            else:
                passed = row.get("passed_filters", True)
                text = row.get("d_minus")
            if not passed or not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()
            if normalize(text) == normalize(ex["d_plus"]):
                continue
            if normalize(text) not in {normalize(t) for t in found[source][qid]}:
                found[source][qid].append(text)
    # Reject known alternative positives for the SAME query, including GP variants.
    positives_by_query = defaultdict(set)
    for e in train:
        positives_by_query[normalize(e["query"])].add(normalize(e["d_plus"]))
        positives_by_query[normalize(e["query"])].update(normalize(t) for t in found.get("gp", {}).get(e["qid"], []))
    for source in required - {"gp"}:
        for qid, texts in found[source].items():
            known = positives_by_query[normalize(by_id[qid]["query"])]
            found[source][qid] = [t for t in texts if normalize(t) not in known]
    eligible = {s: {qid for qid, texts in found[s].items() if len(texts) >= (1 if s == "gp" else k)} for s in required}
    cohort = set(by_id)
    for ids in eligible.values(): cohort &= ids
    if not cohort:
        raise ValueError("empty matched cohort; inspect candidate coverage or lower negative count")
    result = {c: [] for c in conditions}
    for qid in sorted(cohort):
        ex = by_id[qid]
        for condition in conditions:
            neg_source = "bm25" if condition == "B" else "cn" if condition in {"C", "E"} else None
            negs = found[neg_source][qid][:k] if neg_source else []
            variants = [ex["d_plus"]] + (found["gp"][qid] if condition in {"D", "E"} else [])
            for i, positive in enumerate(variants):
                result[condition].append(dict(qid=qid, query=ex["query"], positive=positive,
                                              negatives=negs, source_variant="original" if i == 0 else f"gp_{i}"))
    return result, {"n_train_pool": len(train), "n_matched": len(cohort),
                    "matched_qids": sorted(cohort),
                    "excluded_qids": sorted(set(by_id) - cohort),
                    "source_eligible_counts": {s: len(ids) for s, ids in eligible.items()},
                    "ignored_nontrain_candidates": dict(ignored)}


def validate_records(records, condition):
    if not records:
        raise ValueError("empty training data")
    by_qid = defaultdict(list)
    arities = set()
    for row in records:
        if any(not isinstance(row.get(f), str) or not row[f].strip() for f in ("qid", "query", "positive")):
            raise ValueError("invalid training record")
        negs = row.get("negatives")
        if not isinstance(negs, list) or any(not isinstance(t, str) or not t.strip() for t in negs):
            raise ValueError("invalid negatives")
        arities.add(len(negs))
        if len({normalize(t) for t in negs}) != len(negs):
            raise ValueError("duplicate negatives within a record")
        by_qid[row["qid"]].append(row)
    if len(arities) != 1 or ((next(iter(arities)) > 0) != (condition in {"B", "C", "E"})):
        raise ValueError("fixed negative count required; rebuild the matched conditions")
    for rows in by_qid.values():
        if len({r["query"] for r in rows}) != 1:
            raise ValueError("qid has inconsistent queries")
        positives = {normalize(r["positive"]) for r in rows}
        if any(normalize(n) in positives for r in rows for n in r["negatives"]):
            raise ValueError("negative is a known positive")
    return by_qid


def batch_plan(records, condition, batch_size, steps, seed):
    """One variant/query per batch; query-uniform sampling, fixed optimizer budget.

    Exclude exact normalized collisions among all texts and all known positive
    variants. This cannot detect unannotated semantic false negatives.
    """
    by_qid = validate_records(records, condition)
    if batch_size < 2 or steps < 1 or len(by_qid) < batch_size:
        raise ValueError("need >= batch_size distinct queries, batch_size >= 2, steps >= 1")
    rng = random.Random(seed)
    ids = sorted(by_qid)
    plan = []
    for _ in range(steps):
        rng.shuffle(ids)
        batch, occupied = [], set()
        for qid in ids:
            rows = by_qid[qid]
            row = rng.choice(rows)
            keys = {normalize(r["positive"]) for r in rows}
            keys.update(normalize(t) for t in [row["query"], *row["negatives"]])
            if keys & occupied:
                continue
            occupied |= keys
            batch.append(row)
            if len(batch) == batch_size: break
        if len(batch) != batch_size:
            raise ValueError("cannot form a collision-free full batch; reduce batch size")
        plan.append(batch)
    return plan

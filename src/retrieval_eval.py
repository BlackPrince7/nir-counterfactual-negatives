"""Exact chunked retrieval and graded, multi-relevant query metrics."""
import math
import numpy as np


def retrieve_top_k(query_emb, corpus_emb, corpus_qids, top_k=10, query_batch_size=64):
    if top_k < 1 or query_batch_size < 1:
        raise ValueError("top_k and query_batch_size must be positive")
    query_emb, corpus_emb = np.asarray(query_emb), np.asarray(corpus_emb)
    if query_emb.ndim != 2 or corpus_emb.ndim != 2 or query_emb.shape[1] != corpus_emb.shape[1]:
        raise ValueError("incompatible embeddings")
    if not len(corpus_emb) or len(corpus_emb) != len(corpus_qids) or len(set(corpus_qids)) != len(corpus_qids):
        raise ValueError("empty corpus or invalid corpus IDs")
    if not np.isfinite(query_emb).all() or not np.isfinite(corpus_emb).all():
        raise ValueError("non-finite embeddings")
    # Sort by ID before stable score sorting: ties are independent of input order.
    order = sorted(range(len(corpus_qids)), key=lambda i: corpus_qids[i])
    corpus_emb = corpus_emb[order]
    ids = [corpus_qids[i] for i in order]
    k = min(top_k, len(ids))
    results = []
    for start in range(0, len(query_emb), query_batch_size):
        scores = query_emb[start:start + query_batch_size] @ corpus_emb.T
        for row in scores:
            ranked = np.argsort(-row, kind="stable")[:k]
            results.append([(ids[i], float(row[i])) for i in ranked])
    return results


def relevance(query):
    qrels = query.get("qrels")
    if qrels is None:
        if "gold_qid" not in query:
            raise ValueError("query needs qrels or legacy gold_qid")
        qrels = {query["gold_qid"]: 1}
    if not isinstance(qrels, dict):
        raise ValueError("qrels must map document IDs to nonnegative relevance grades")
    if any(not isinstance(doc, str) or isinstance(g, bool) or not isinstance(g, (float, int))
           or not math.isfinite(g) or g < 0 for doc, g in qrels.items()):
        raise ValueError("invalid relevance judgments")
    return qrels


def validate_evaluation(queries, corpus):
    if not queries or not corpus:
        raise ValueError("evaluation requires nonempty queries and corpus")
    docs = {c["qid"] for c in corpus}
    if len(docs) != len(corpus) or len({q["qid"] for q in queries}) != len(queries):
        raise ValueError("duplicate query/document IDs")
    for q in queries:
        if not set(relevance(q)) <= docs:
            raise ValueError(f"{q['qid']}: qrels reference missing corpus documents")


def query_metrics(qrels, predictions, k):
    if k < 1: raise ValueError("k must be positive")
    if len(set(predictions)) != len(predictions):
        raise ValueError("duplicate retrieved document IDs")
    relevant = {doc for doc, grade in qrels.items() if grade > 0}
    ranked = predictions[:k]
    hit_ranks = [i for i, doc in enumerate(ranked, 1) if doc in relevant]
    dcg = sum(qrels.get(doc, 0) / math.log2(i + 1) for i, doc in enumerate(ranked, 1))
    ideal = sum(g / math.log2(i + 1) for i, g in enumerate(sorted(qrels.values(), reverse=True)[:k], 1))
    # Linear gain and AP denominator total relevant documents (trec_eval convention).
    return {f"recall@{k}": len(hit_ranks) / len(relevant) if relevant else 0.0,
            f"hits@{k}": float(bool(hit_ranks)),
            f"mrr@{k}": 1 / hit_ranks[0] if hit_ranks else 0.0,
            f"ndcg@{k}": dcg / ideal if ideal else 0.0,
            f"map@{k}": sum(i / rank for i, rank in enumerate(hit_ranks, 1)) / len(relevant) if relevant else 0.0}


def evaluate_rankings(queries, rankings, top_k):
    if top_k < 1 or len(queries) != len(rankings) or not queries:
        raise ValueError("invalid query/ranking count or top_k")
    cutoffs = sorted({k for k in (1, 3, 5, 10, 100, top_k) if k <= top_k})
    records = []
    for query, hits in zip(queries, rankings):
        qrels = relevance(query)
        predicted = [doc for doc, _ in hits]
        metrics = {}
        for k in cutoffs: metrics.update(query_metrics(qrels, predicted, k))
        first = next((i for i, doc in enumerate(predicted[:top_k], 1) if qrels.get(doc, 0) > 0), None)
        records.append(dict(qid=query["qid"], query=query["query"], qrels=qrels,
                            gold_qid=query.get("gold_qid"), gold_rank=first,
                            gold_in_top_k=first is not None, metrics=metrics,
                            top_k_predictions=[dict(qid=doc, score=score) for doc, score in hits]))
    aggregate = {key: sum(r["metrics"][key] for r in records) / len(records) for key in records[0]["metrics"]}
    aggregate.update(n_queries=len(records), top_k=top_k,
                     n_queries_without_relevant=sum(not any(g > 0 for g in r["qrels"].values()) for r in records))
    return records, aggregate

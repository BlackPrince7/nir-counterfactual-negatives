# Baseline correctness repair

The v2 pipeline replaces scripts 07–10. Old artifact directories are preserved;
v2 defaults use `data/baseline_v2` and `outputs/*_v2`. Rebuild training data:
legacy files lack the cohort manifest and cannot be used by the new trainer.

## Problems and changes

| Previous behavior | Why it invalidates or weakens the comparison | New behavior |
|---|---|---|
| Trainer drops B/C/E rows without negatives | Different query populations across conditions | Builder selects a common eligible cohort; trainer never drops rows |
| Cap of three negatives allows 1/2/3 per row | Variable batch arity and different negative budgets | Exact K (default 1) for explicit-negative conditions |
| GP duplicates rows, training uses equal epochs | More GP variants mean more optimizer updates and unequal query weighting | Fixed `--max-steps`; select queries, then one positive variant/query |
| Ordinary shuffled batches | Same-query positives and duplicate documents compete as negatives | Full batches exclude duplicate queries and normalized text collisions, including known positive variants |
| Split after generation; combined RuBQ official dev/test reshuffled | Test labels may enter training/tuning; source documents repeat across splits | Prepare frozen grouped train/dev/test first; official policy reserves source test |
| Per-tag rounding silently produces fewer rows | Requested 100/200 can become 99/199 | Whole connected groups selected to meet requested minimums; actual IDs recorded |
| BM25 excludes only matching qid | Identical positive under another ID can be mined as negative | Exclude normalized positive text too; builder excludes known same-query positives |
| No retrieval prefixes; train/eval max lengths can differ | RoSBERTa is used inconsistently with its retrieval instructions | Explicit query/document prefixes and length saved with model and reused at eval |
| Single relevant document assumed | Recall equals Hits only in this special case; cannot handle graded medical qrels | Graded multi-document qrels; Recall, Hits, MRR, nDCG, AP at each cutoff |
| `argpartition(..., top_k)` | Fails when K reaches corpus size | Clamp result count and use stable score sorting with document-ID tie-breaks |
| Hits@K reads the largest fixed recall cutoff | At K=2 it reports Hits@1 | Every metric computed at its actual K |
| Full query-by-corpus score matrix | Memory grows with all queries | Score queries in chunks |
| Train/eval catch failures and continue successfully | A failed run can look completed to automation | Exceptions produce nonzero exit status; training records failed status |
| First C run / arbitrary +1.5 point threshold defines success | Hides seed choice and uncertainty | Per-seed comparisons, evaluation fingerprint guard and paired query bootstrap CI |

## Commands

Use the existing project environment. Lightweight tests:

```bash
.venv/bin/python -m pytest -q
```

Full training requires Torch and Sentence Transformers. The optional integration
tests build a tiny local bag-of-words/projection model, train the real MNRL loss,
and verify changed weights plus save/reload behavior. A separate offline CLI test
runs preparation, matched A/B/C assembly, actual training, baseline/finetuned
evaluation, and comparison report generation. These tests skip if those
dependencies are absent; they do not establish retrieval quality on real data.
For a CPU development environment (not GPU training), install them with:

```bash
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python 'sentence-transformers>=3,<6'
```

Freeze split BEFORE generation (requires the downloaded RuBQ pairs):

```bash
.venv/bin/python scripts/07_build_train_data.py --prepare-only \
  --n-train 300 --n-dev 50 --n-test 200
```

Official mode is intended for the existing RuBQ data: source dev supplies the
training/validation pool and source test supplies test. Exact normalized shared
queries or documents are connected groups. Dev records connected to source test
are excluded from training/validation. Counts may exceed requested minimums to
keep groups intact. If there is insufficient data, preparation fails clearly.
`--split-policy custom-grouped` is an explicitly custom experiment and must not
be reported as the official benchmark. Near duplicates and shared article IDs
are not detected by text equality; audit these separately for a thesis dataset.

Generate ONLY on the frozen training pool. Choose a working model config; the
committed default still contains a Windows GGUF path.

```bash
.venv/bin/python scripts/05_run_cn_pipeline.py --config configs/local.yaml \
  --input data/baseline_v2/train_pool.jsonl --n-examples 1000000 \
  --output-dir outputs/cn_baseline_v2
.venv/bin/python scripts/06_mine_bm25_negatives.py \
  --input data/baseline_v2/train_pool.jsonl \
  --corpus data/rubq/rubq_full.jsonl --n-examples 1000000 --k 5 \
  --output-dir outputs/bm25_baseline_v2
.venv/bin/python scripts/07_build_train_data.py \
  --split-file data/baseline_v2/split.json --conditions A B C --cap-negatives 1 \
  --cn-input outputs/cn_baseline_v2/candidates_passed.jsonl \
  --bm25-input outputs/bm25_baseline_v2/candidates_passed.jsonl
```

The corpus is shared retrieval content, so indexing test documents is allowed.
Test query labels must not guide generation, training or tuning. Assembly ignores
non-training candidate IDs and reports their count. It cannot retrospectively
prove that prompts/thresholds were not tuned on test data: regenerate candidates
for the new protocol if that history is unknown.

Validate the planned batches before loading a model:

```bash
.venv/bin/python scripts/08_train_retriever.py --conditions A B C --dry-run \
  --batch-size 8 --max-steps 100 --output-base outputs/dry_baseline_v2
.venv/bin/python scripts/08_train_retriever.py --conditions A B C \
  --batch-size 8 --max-steps 100 --seed 42
.venv/bin/python scripts/09_eval_retriever.py --baseline --split dev
.venv/bin/python scripts/09_eval_retriever.py --conditions A B C --seed 42 --split dev
.venv/bin/python scripts/10_compare_ablations.py --seed 42
```

Tune on dev. Repeat the training/evaluation for seeds 43 and 44 with identical
hyperparameters and build manifest. After freezing choices, evaluate with
`--split test`; compare with `--eval-base outputs/eval_v2/test`. A training seed
changes initialization/sampling, not the dataset split. Existing nonempty output
directories are rejected to avoid overwriting runs; select a new output base.
The legacy `--epochs` argument is deliberately removed in favor of `--max-steps`.
Assembly refuses to replace an existing frozen split with a different one;
reused official splits are checked against source labels again.

## Interpretation and remaining limits

The common cohort is a selection-conditioned experiment. Report excluded queries,
per-method coverage and generation costs alongside retrieval metrics. The trainer
checks all condition hashes and cohorts against `manifest.json`, including when
training one condition at a time. GP variants are sampled within query; they do
not multiply query exposure. Collision avoidance may change actual exposure;
inspect `batch_plan.jsonl` and reduce batch size if a full batch cannot be formed.

The pair loader retains only one positive paragraph per source question. The
builder can combine retained positives for identical query texts; it cannot
recover judgments discarded upstream. For a medical benchmark, supply complete
`qrels` in dev/test query rows: `{"qid":"q", "query":"...", "qrels":{"doc1":2,
"doc2":1,"doc3":0}}`. Corpus rows use `{"qid":"doc1", "text":"..."}`.
Legacy `gold_qid` is accepted as a single relevant judgment.

Metrics use linear relevance gain for nDCG and total relevant documents as the
AP@K denominator. Queries without any positive judgment contribute zero and are
counted explicitly. Unknown judgments score as zero; this does not prove those
documents are irrelevant. Unjudged semantic false negatives still need auditing.
Bootstrap intervals quantify query sampling uncertainty for one trained model
pair, not variability across training seeds. This code does not establish clinical
correctness or research gains; real data/model runs are still required.

RoSBERTa prefix reference: https://huggingface.co/ai-forever/ru-en-RoSBERTa
MNRL reference: https://www.sbert.net/docs/package_reference/sentence_transformer/losses.html

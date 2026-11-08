# Git workflow for this research project

Commit after a coherent, reviewable change works and its relevant checks pass.
A useful checkpoint is something you can describe in one sentence and revert
independently: a schema migration, a bug fix, a dataset adapter, or one experiment
configuration. This often means several commits per focused working day; use
completion of an idea rather than a timer. Commit before expensive experiments
so their code version is unambiguous.

Use short-lived branches for changes. `main` should remain runnable. A practical
sequence for the current migration is:

```bash
git switch -c feat/typed-config
.venv/bin/python -m pytest -q
git diff --check
git status --short
git diff
git add src/config.py src/data_io.py src/schemas.py src/llm_client.py src/encoders.py src/pipelines.py
git add scripts/00_sanity_qwen.py scripts/02_ce_judge_experiment.py scripts/03_delta_distribution.py scripts/04_run_gp_pipeline.py scripts/05_run_cn_pipeline.py
git add tests/test_contracts.py requirements.txt requirements-test.txt docs/configuration.md docs/version-control.md README.md
git diff --cached
git commit -m "feat: validate pipeline records and YAML configuration with Pydantic"
git push -u origin feat/typed-config
```

These are instructions, not actions performed by the migration. Staging selects
the exact snapshot for the commit. `git diff` shows unstaged edits;
`git diff --cached` shows what will be committed. Before `git add`, an empty
`git diff --cached` is expected. The correct flag is `--cached`, not `--cache`.
A commit is local; pushing
backs it up to the remote and makes it available to collaborators. Push useful
checkpoints and at the end of a work session. For published history, prefer
`git revert <commit>` to rewriting history when undoing a change.

Track source, prompts, shared configs, tests, small hand-reviewed fixtures,
dependency specifications/lockfiles and dataset manifests in Git. Keep model
weights, downloaded corpora, generated candidate files, secrets and machine
paths outside Git. A dataset manifest should identify its source revision,
license, split policy and content hashes; a path alone is insufficient.

When MLflow is added, log the Git commit, dirty-worktree flag, resolved config,
dataset/split/graph hashes and model revisions with every run. MLflow will store
metrics and artifacts; Git records the implementation. Record CLI arguments too:
they currently contain production runner settings absent from the YAML.

The repository currently tracks some `outputs/sanity/*` and small `data/train/*`
reports despite ignore patterns. `.gitignore` only prevents new untracked files
from being added; it does not untrack existing files. Keep useful reference
artifacts deliberately, and review their diffs rather than assuming they are
ignored. No tracked artifacts were removed during this migration.

Use milestone tags after validating baselines (for example `v0.1-baseline`),
and brief experiment notes explaining the hypothesis, configuration and result.
An unsuccessful experiment is still useful research history; record it rather
than deleting the evidence to make the history look successful.

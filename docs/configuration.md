# Typed configuration and data contracts

`src.config.load_config(path)` returns an `ExperimentConfig`. All five runners
that read YAML use this loader before loading models. Access settings as
`cfg.llm.temperature` and `cfg.encoders.batch_size`.

The OS `python3` interpreter does not have project dependencies installed.
Create and use the project virtual environment. These commands validate the
configuration without downloading models or installing Torch:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m src.config configs/default.yaml
.venv/bin/python -m src.config --schema
```

The first command prints the resolved configuration as JSON; the second prints
JSON Schema. `schema_version` defaults to 1 for existing YAML files.
Unknown fields, numeric strings, invalid backends, non-finite numbers, invalid
ranges, duplicate encoder names, empty ensembles and overlapping delta buckets
are rejected. Errors from YAML loading include the source path and field path.
Configuration objects validate attribute assignments too; in-place changes to
nested lists/dictionaries are not validated. Revalidate a complete dump after
programmatically editing a configuration.

Use actual YAML numbers and booleans (`512`, `false`), not quoted strings.
Paths remain strings so a Windows config can be inspected on Linux. Validation
does not check that model files exist or that a backend is available.

```python
from src.config import load_config
from src.llm_client import make_client
from src.encoders import EncoderPool

cfg = load_config("configs/default.yaml")
client = make_client(cfg.llm)
pool = EncoderPool.from_config(cfg.encoders)
```

`LLMConfig` remains importable from `src.llm_client`; `from_dict` now validates
unknown keys instead of silently discarding them. `EncoderPool.from_config`
accepts either its typed section or a dictionary, validating before construction.
Legacy pipeline config classes remain importable from `src.pipelines`.
`PosConfig.targets` must now be a nonempty list instead of defaulting to `None`.

`QueryDoc`, `Fact`, `NegativeRecord`, and `PositiveRecord` are Pydantic models.
Construct them with keyword arguments, use `model_validate` at external
boundaries and `model_dump(mode="json")` to serialize. Existing record field
names and record `to_dict()` methods are preserved. Document text is not stripped:
its offsets matter to mutation verification. Custom input metadata belongs in
`meta`; unexpected top-level fields are rejected.

The production CN and GP runners validate input JSONL through `load_query_docs`,
which reports bad records with file and line number and rejects duplicate qids.
CN also validates extracted facts (including criticality 1–5); malformed LLM
facts are skipped. Validation checks structure, not medical correctness.

Scope: existing CLI-specific settings still come from argparse, and production
candidate diagnostic dictionaries retain their current format. This migration
does not change filtering thresholds or consolidate the legacy and production
pipelines. In particular, the production CN runner uses its CLI filters, not the
legacy YAML delta buckets; GP uses its CLI iteration controls, not YAML targets.
Local configuration is never merged implicitly: copy the full YAML to the ignored
`configs/local.yaml`, edit it, and pass `--config configs/local.yaml` explicitly.

Run the lightweight contract suite without installing Torch or any LLM backend:

```bash
.venv/bin/python -m pytest -q
```

The test environment is not the complete training environment. Full generation
still requires the dependencies and backend described in the setup documentation.

# AGENTS.md

## Architecture
Follows MOABB's pipeline pattern: **Dataset -> Paradigm -> Evaluation -> Pipeline -> Results**.

- **fnirsatlas/datasets/** — MOABB `BaseDataset` subclasses that download or locate fNIRS data and load it into MNE Raw objects.
- **fnirsatlas/paradigms/** — `BaseFNIRS` and task-specific classification paradigms with fNIRS preprocessing.
- **fnirsatlas/pipelines/** — sklearn-compatible transformers and default pipeline factories.
- **fnirsatlas/config.py** — YAML configuration with packaged defaults and an optional project-local `config/config_local.yaml` override.
- **Snakefile** — Benchmark workflow, one job per paradigm, evaluation, dataset, pipeline and seed.

## Testing
- Test desired behaviour, not implementation details.
- Prefer real cached-data integration tests. Do not test hard-coded manifests or mocked helper outputs.

## Code style
- Imports at the top if possible.
- Use types.
- Use the terms in `./GLOSSARY.md` in code and docs.

## Environment
- use `uv`

## Format
- Run `uv run ruff check` for linting.
- Run `uv run ruff format` for formatting.

## LSP check
- Run `uv run ty check` for lsp check.

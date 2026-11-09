# Contributing

Contributions of datasets, tasks, models, fixes, documentation, or other improvements are welcome.

## Issues

Open an [issue](https://github.com/kalomak/fNIRSAtlas/issues/new/choose) before larger changes:

- **Dataset request**: a public fNIRS dataset to add.
- **Dataset or task correction**: an error in how a dataset is loaded, labelled, epoched, or documented.
- **Blank issue**: bugs, feature requests, and questions.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Set up the environment with [uv](https://docs.astral.sh/uv/):

   ```bash
   git clone https://github.com/<your-username>/fNIRSAtlas
   cd fNIRSAtlas
   uv sync --all-extras --group dev
   ```

3. Make the change. Guides:
   - [Adding a dataset](docs/adding_datasets.md)
   - [Paradigms](docs/paradigms.md)
   - [Adding a model](docs/custom_models.md)
4. Run the checks described below.
5. Open a pull request against `main`. Reference the related issue and describe how the change was checked.

## Tests

```bash
uv run pytest
```

This runs the offline tests. Tests marked `data` download real datasets, and tests marked `manual` are slow checks run on demand. Both are excluded by default. Tests for optional extras, such as `dl` and `aeon`, are skipped when the extra is not installed.

For a new or changed dataset, also run its data tests:

```bash
uv run pytest -m "data or manual" tests/test_<dataset>.py
```

## Linting, formatting, and type checking

```bash
uv run ruff check
uv run ruff format
uv run ty check
```

## Documentation

Build the documentation without executing the examples:

```bash
make docs
```

The output is `docs/build/html/index.html`. Execute all gallery examples with `make docs-gallery`, and clean generated files with `make docs-clean`.

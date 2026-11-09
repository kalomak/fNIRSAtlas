# Development

## Environment

The project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

Add `--extra dl` or `--extra braindecode` when working on those integrations.

## Tests

```bash
uv run pytest
```

Tests marked `data` download real datasets. Tests marked `manual` are slow checks run on demand. Both are excluded by default.

```bash
uv run pytest -m data
uv run pytest -m manual
```

## Type checking

```bash
uv run ty check
```

## Linting and formatting

```bash
uv run ruff check
uv run ruff format
```

## Documentation

Build documentation without executing examples:

```bash
make docs
```

The output is `docs/build/html/index.html`.

Execute all gallery examples with:

```bash
make docs-gallery
```

Clean generated files with:

```bash
make docs-clean
```

# Statistical analysis

## Compare more than two pipelines on multiple datasets

Critical-difference diagram:

```bash
uv run python scripts/cd_diagram.py --evaluation CrossSubject \
    --pipelines Windowed_LDA eegnet fnirsnet23 --seeds 0 1 2 3 4
```

Friedman test with Nemenyi post-hoc:

```bash
uv run python scripts/friedman_test.py --evaluation CrossSubject \
    --pipelines Windowed_LDA eegnet fnirsnet23 --seeds 0 1 2 3 4
```

## Compare two pipelines on multiple datasets

Wilcoxon signed-rank test across datasets:

```bash
uv run python scripts/wilcoxon_test.py --evaluation CrossSubject \
    --pipelines Windowed_LDA eegnet --seeds 0 1 2 3 4
```

MOABB meta-analysis:

```bash
uv run python scripts/meta_analysis.py --evaluation CrossSubject \
    --pipelines Windowed_LDA eegnet --seeds 0 1 2 3 4
```

## Compare two pipelines on a single dataset

- Nadeau-Bengio corrected t-test: `moabb.analysis.meta_analysis.compute_pvals_corrected_ttest`.

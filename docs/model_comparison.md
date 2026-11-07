# Model Comparison

## Statistical Methods

### Model Comparison

- **Corrected t-test** (Nadeau & Bengio, 2003) for cross-validation dependency
- **FDR correction** (Benjamini-Hochberg) for multiple comparisons
- **Significance** based on FDR-corrected p-values

### Wilcoxon Signed-Rank Test

- **Wilcoxon signed-rank test** for non-parametric comparison of paired samples
- **Rank-based analysis** showing signed ranks for each task
- **Task filtering** to avoid overlapping data between tasks
- **Multiple comparison correction** with significance levels

### Friedman Test

- **Friedman test** for omnibus comparison across 3 or more paired models
- **Mean-rank summary** across non-overlapping dataset-task blocks
- **Post-hoc Wilcoxon tests** with Holm or FDR correction

### Critical Difference Diagram

- **Mean ranks** of 3 or more models across non-overlapping dataset-task blocks
- **Bars** connect models whose difference is not significant, by the Nemenyi test (default) or Holm-corrected Wilcoxon tests
- Drawn with [aeon](https://www.aeon-toolkit.org/) `plot_critical_difference` (`fnirsatlas[aeon]` extra)

## Workflow

### Model Comparison (t-test)

1. **Run baseline model**: `uv run python scripts/classify_dataset.py --suite ma -m lda -c cross --seed 42`
2. **Run dummy model**: `uv run python scripts/dummy_model_benchmark.py`
3. **Use compare_models script**: `uv run python scripts/compare_models.py --models lda dummy --suite ma --metric cross_acc --seeds 0 1 2 3 4`
4. **Automatic comparison**: Script performs corrected t-test and FDR correction
5. **View results**: Shows mean differences and statistical significance

### Wilcoxon Signed-Rank Test

1. **Run models**: Generate results for models you want to compare
2. **Use wilcoxon_test script**: `uv run python scripts/wilcoxon_test.py --models lda dummy`
3. **Automatic filtering**: Script removes overlapping tasks automatically
4. **View results**: Shows signed ranks and statistical significance

### Friedman Test

1. **Run 3+ models**: Generate results for all models you want to compare
2. **Use friedman_test script**: `uv run python scripts/friedman_test.py --models lda eegnet dcn fnirst --metric cross_acc --seeds 0 1 2 3 4`
3. **Automatic filtering**: Script keeps one non-overlapping task per dataset
4. **View results**: Shows omnibus Friedman result, mean ranks, and corrected post-hoc Wilcoxon tests

### Critical Difference Diagram

1. **Run 3+ models**: Generate results for seeds 0-4. Cross-subject results must use `--search-strategy nested`, within-subject results `--search-strategy block`
2. **Use cd_diagram script**: `uv run python scripts/paper/cd_diagram.py --models lda eegnet dcn fnirst --metric cross_acc --seeds 0 1 2 3 4`
3. **View results**: Saves the diagram to `paths.results` (default `./results`) as `paper/cd_diagram_<metric>.png`. Use `--test wilcoxon` for Wilcoxon cliques

## Output Format

```
Custom SVM RBF vs LDA Baseline results comparison (α = 0.05)
Diff = Custom SVM RBF cross_acc - LDA Baseline cross_acc
=====================================================================================
Dataset              CLF Task            Diff   P-Orig    P-FDR Sig
-------------------------------------------------------------------------------------
Shin2016 (ma)        BL|MA             -0.170    0.000    0.000 ***
Shin2018b            BL|MA             -0.130    0.000    0.000 ***
Bnci                 BL|MA             -0.287    0.000    0.000 ***
Shin2019 (ma)        BL|MA             -0.164    0.000    0.000 ***
```

- **Diff**: Mean difference (your_model - baseline)
- **P-Orig**: Original p-value
- **P-FDR**: FDR-corrected p-value
- **Sig**: Significance (\* p<0.05, ** p<0.01, \*** p<0.001)

```
Wilcoxon test on cross_acc task means (α = 0.05)
====================================================================================================

Custom SVM RBF vs LDA Baseline
====================================================================================
Dataset              Task                                         Diff    Signed Rank
-------------------------------------------------------------------------------------
Shin2016 (ma)        BL|MA                                     -17.0               -4
Shin2018b            BL|MA                                     -13.0               -3
Bnci                 BL|MA                                     -28.7               -5
Shin2019 (ma)        BL|MA                                     -16.4               -2

Custom SVM RBF vs LDA Baseline: W=0.0, p=0.068 ***
```

- **Diff**: Mean difference (your_model - baseline)
- **Signed Rank**: Rank of absolute difference with sign indicating direction
- **W**: Wilcoxon test statistic
- **p**: p-value with significance indicators (\* p<0.05, ** p<0.01, \*** p<0.001)

The script automatically filters out tasks that may have overlapping data to ensure independent samples for the statistical test.

# Classification Tasks Configuration

Classification tasks define how experimental conditions are grouped for classification analyses. This is managed via a central YAML file: `classification_tasks.yaml` in the project root.

## How It Works

- If a dataset has a `classification_tasks` entry in `classification_tasks.yaml`, those tasks are used.
- If not, the parser's falls back to classifying all available conditions.

## Defining Tasks

Each dataset can have one or more classification tasks. Each task has a `description` and a `mapping` (from condition label to class index).

### Example: Explicit Mapping

```yaml
shin2019:
  classification_tasks:
    - description: "MA vs MI vs Rest"
      mapping:
        Idle State: 0
        Mental Arithmetic: 1
        Motor Imagery: 2
    - description: "MA vs Rest"
      mapping:
        Idle State: 0
        Mental Arithmetic: 1
```

### Example: Explicit Multi-Condition Mapping

```yaml
cockx2023:
  classification_tasks:
    - description: "Finger vs Foot"
      mapping:
        fingerauto_a: 0
        fingerauto_b: 0
        fingerautodual_a: 0
        fingerautodual_b: 0
        fingernonauto_a: 0
        fingernonauto_b: 0
        fingernonautodual_a: 0
        fingernonautodual_b: 0
        footauto_a: 1
        footauto_b: 1
        footautodual_a: 1
        footautodual_b: 1
        footnonauto_a: 1
        footnonauto_b: 1
        footnonautodual_a: 1
        footnonautodual_b: 1
```

### Example: Per-Task Context

For datasets with multiple sub-tasks (e.g., NEMO with `empe` and `afim`):

```yaml
nemo:
  classification_tasks:
    empe:
      - description: "4-class"
        mapping:
          HANV: 0
          HAPV: 1
          LANV: 2
          LAPV: 3
    afim:
      - description: "4-class"
        mapping:
          afim_HANV: 0
          afim_HAPV: 1
          afim_LANV: 2
          afim_LAPV: 3
```

## Adding or Editing Tasks

1. Open (or create) `classification_tasks.yaml` in the project root.
2. Add or edit the `classification_tasks` section for your dataset.
3. Use explicit condition-to-class mappings.
4. Save and rerun your pipeline—no code changes required!

# WISP LogP benchmark

This benchmark evaluates MATCHA's LIME explanations on the experimental LogP task with the matched-molecular-pair scores introduced by WISP.

The workflow has two phases. Counterfactual neighborhoods are generated once for the complete test split, then every MATCHA architecture predicts the complete neighborhoods and runs LIME.

## Environment

Install the benchmark dependencies:

```bash
uv sync --extra wisp-exp
```

## Prepare shared inputs

```bash
uv run python datasets/wisp/download_data.py
uv run --extra wisp-exp python explainability/prepare.py
```

This creates the fixed train/test split for the `exp` endpoint, test-only matched molecular pairs, and reusable LIME counterfactuals under `explainability/prepared/`. Interrupted counterfactual generation resumes from `neighborhood_status.csv`.

## Run models

List the model/task mapping:

```bash
uv run --extra wisp-exp python explainability/run_benchmark.py --list-models
```

Run one model directly:

```bash
uv run --extra wisp-exp python explainability/run_benchmark.py --model mlp
```


## Outputs

Generated plotting inputs are written under `explainability/wisp/`:

- `mmp_scores/<model>.csv.gz`: prediction and whole/variable attribution deltas for each MMP.
- `descriptor_scores/<model>.csv.gz`: per-compound descriptor coefficients, bootstrap standard deviations, and ranks.
- `summary.csv`: model names and test-set R² values.

Transient per-model summary shards used for resume and validation are written under `explainability/logs/wisp_summary_shards/`.

MATCHA owns the LIME implementation: 25 deterministic row-bootstrap fits, standardized descriptor features, raw ECFP8192 radius-3 features, and occurrence-aware coefficient-conserving atom weights. The benchmark only computes MMP region sums and serializes the plotting inputs.

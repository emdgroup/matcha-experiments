# Experiments with MATCHA

[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-preprint-b31b1b)](https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006766/v1)

This repo contains the code, configs and scripts to reproduce the results discussed in the following paper: MATCHA: a toolkit for streamlining molecular property prediction for drug discovery applications, from Davide Boldini, Lukas Friedrich, Jakub Gunera, Christina Schindler and Daniel Kuhn.

> [!NOTE]
> We refactored the full repository into a shorter version to make internal auditing easier. Scripts are included to rebuild the original layout:
> - `configs/generate_autotune.sh` and `configs/generate_eval.sh` regenerate the `configs/autotune/` and `configs/eval/` YAML files
> - `python eval/compress_performance.py expand` regenerates the per-run `eval/*/*/performance.json` files from `eval/performance.csv`
>
> The raw per-compound predictions and labels (`*_preds.csv`) are not included. They are available upon request.
>
> Training data is not stored in the repo. Download it with the Python scripts in `datasets/` (see below). For the ADME-pretraining foundation model, follow `datasets/adme-pretraining-public/manifest.yaml` (sources and processing steps) and `datasets/adme-pretraining-public/preprocessing.ipynb` to rebuild the pretraining dataset.

## 🧩 How to install

Run these commands to set up the performance-benchmark environment via UV:

```
git clone ...
cd ...
uv sync --extra performance
```

The optional environments are mutually exclusive. Use `--extra uq-exp` for the uncertainty benchmark or `--extra wisp-exp` for the WISP explainability benchmark.

Download then the data used in the paper:

```
uv run python datasets/openadmet/prepare_data.py
uv run python datasets/wisp/download_data.py
```

## 🔍 Repo structure

### Folders

- `configs`:                MATCHA CLI configs used to generate the OpenADMET-ExpansionRx results, from hparam tuning to evaluations
- `datasets`:               datasets used for benchmarking on OpenADMET-ExpansionRx and for pretraining on publicly available ADME data. Make sure to download/preprocess them before running any other commands
- `eval`:                   evaluation results from the MATCHA CLI. The per-run `performance.json` files are compressed into `eval/performance.csv`; run `python eval/compress_performance.py expand` to regenerate them
- `optima`:                 optimal hyperparameters from the tuning routine in MATCHA CLI
- `pretrain`:               path where the pretrained model will be stored. The config for re-training it is available at `configs/pretrain/gatedgcn.yaml`
- `summarize`:              statistical comparisons between models for a given dataset subset (e.g. caco2), ran with the MATCHA CLI
- `explainability`:         saved results for the explainability analysis and jupyter notebook to reproduce them
- `uncertainty`:            saved results for the uncertainty analysis and jupyter notebook to reproduce them
- `challenge_leaderboard`:  OpenADMET-ExpansionRx leaderboard results used in `eval_openadmet.py`
- `LLM`:                    Example conversation with Sonnet 4.6 to generate the configs used for the OpenADMET-ExpansionRx (redacted for confidentiality reasons)

### Scripts

- `eval_openadmet.py`:  reproduces Figure 1 of the paper from the saved evaluation results in this repo
- `uncertainty.py`:     reproduces Figure 2 of the paper from the saved results in this repo
- `explainability.py`:  reproduces figure 3 of the paper from the saved results in this repo
- `explainability_SI.py`: reproduces figure S1 of the paper from the saved results in this repo

## 🛠️ Reproducing

### OpenADMET-ExpansionRx results

For the OpenADMET-ExpansionRx results, run the related MATCHA CLI commands for each config. For example, to re-run hyperparameter optimization for a given model:

```
matcha autotune --config configs/autotune/caco2/attentivefp.yaml
```

To re-run metric calculations:

```
matcha evaluate --config configs/eval/caco2/attentivefp.yaml
```

To re-run foundation model pretraining:

```
matcha pretrain --config configs/pretrain/gatedgcn.yaml
```

To re-run the statistical analysis:

```
matcha summarize --config configs/summarize/caco2.yaml
```

### Paper figures

Just run e.g. `python explainability.py`, or any other figure-generating script described above.

## 📖 Citation

```bibtex
@article{matcha_2026,
  author = "Boldini, Davide and Friedrich, Lukas and
            Gunera, Jakub and Schindler, Christina and Kuhn, Daniel",
  title  = "{MATCHA}: a toolkit for streamlining molecular property prediction
            for drug discovery applications",
  journal = "ChemRxiv",
  year = "2026",
  doi = "10.26434/chemrxiv.15006766/v1",
  url = "https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15006766/v1",
}
```
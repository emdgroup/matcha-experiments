"""Benchmark uncertainty-quantification approaches across all endpoints and models.

For each (model, endpoint) combination we fit:
    * a single MC-dropout model (skipped for models that don't expose
      inference-time dropout, e.g. chemprop)
    * a single MVE model (skipped for models that don't support it, e.g. SNN)
    * a deep ensemble (MVE is not enabled inside the ensemble)
    * an error-model calibrator on top of the ensemble
and log Spearman rank correlation plus MAE for the top-/bottom-10% confidence
buckets.  Rows are appended to ``uncertainty/logging_df.csv`` so downstream
analyses (e.g. per-endpoint boxplots) can be produced without re-running the
training loop.  ``uncertainty.py`` at the repo root consumes this CSV for the
summary figure.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr

from matcha.calibration import EMRegressionCalibration
from matcha.sklearn import (
    AttentiveFPRegressor,
    ChempropRegressor,
    CNNRegressor,
    E3GNNRegressor,
    Ensemble,
    GatedGCNRegressor,
    GINRegressor,
    GPS3DRegressor,
    GPSRegressor,
    GT3DRegressor,
    GTRegressor,
    MLPRegressor,
    RNNRegressor,
    RoFormerRegressor,
    SNNRegressor,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
DATA_DIR = REPO_ROOT / "datasets" / "openadmet"
LOG_PATH = HERE / "logging_df.csv"
SHARD_DIR = HERE / "logging_shards"

DATASETS: dict[str, list[str]] = {
    "caco2": ["CACO2_Papp_AB_endpoint", "CACO2_ER_endpoint"],
    "clint": ["HLM_CLint_endpoint", "MLM_CLint_endpoint"],
    "physchem": ["KSOL_endpoint", "LogD_endpoint"],
    "unbound": ["MBPB_endpoint", "MGMB_endpoint", "MPPB_endpoint"],
}

MODELS: dict[str, type] = {
    "mlp": MLPRegressor,
    "snn": SNNRegressor,
    "cnn": CNNRegressor,
    "rnn": RNNRegressor,
    "roformer": RoFormerRegressor,
    "attentivefp": AttentiveFPRegressor,
    "chemprop": ChempropRegressor,
    "gatedgcn": GatedGCNRegressor,
    "gin": GINRegressor,
    "gps": GPSRegressor,
    "gt": GTRegressor,
    "e3gnn": E3GNNRegressor,
    "gps3d": GPS3DRegressor,
    "gt3d": GT3DRegressor,
}

# Models that don't support the MVE head — see matcha's SNNModel MVE check.
MVE_UNSUPPORTED_MODELS: set[str] = {"snn"}
# Models that don't support MC-dropout uncertainty (chemprop's D-MPNN encoder
# does not expose dropout at inference time).
MC_DROPOUT_UNSUPPORTED_MODELS: set[str] = {"chemprop"}
# Loss-fn kwarg required to pair with uncertainty="mve".  Chemprop uses its
# own MveFFN + MVELoss (alias "mve"); every other family uses matcha's
# β-NLL head.
MVE_LOSS_FN: dict[str, str] = {"chemprop": "mve"}
MVE_LOSS_FN_DEFAULT = "beta-nll"

N_ENSEMBLE_MODELS = 10
MC_DROPOUT_ITERATIONS = 100

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _to_1d_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def _normalised_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)
    denom = y_true + np.abs(np.min(y_true)) + 1
    return np.abs(y_true - y_pred) / denom


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(spearmanr(_to_1d_array(a), _to_1d_array(b)).correlation)


def _bucket_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uq: np.ndarray,
    fraction: float = 0.1,
) -> tuple[float, float]:
    """Return (top-confident MAE, bottom-confident MAE) — lowest uq is most confident."""
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)
    uq = _to_1d_array(uq)
    abs_err = np.abs(y_true - y_pred)
    order = np.argsort(uq)
    k = max(1, int(np.ceil(fraction * len(order))))
    return float(np.mean(abs_err[order[:k]])), float(np.mean(abs_err[order[-k:]]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(
    dataset: str, endpoint: str
) -> tuple[list, np.ndarray, list, np.ndarray, list, np.ndarray]:
    """Load train, val and test molecules/labels for one (dataset, endpoint)."""

    def _load(path: Path) -> tuple[list, np.ndarray]:
        df = pd.read_csv(path, usecols=["SMILES", endpoint]).dropna()
        mols = [Chem.MolFromSmiles(s) for s in df["SMILES"]]
        keep = np.array([m is not None for m in mols])
        df = df.loc[keep].reset_index(drop=True)
        mols = [m for m, k in zip(mols, keep) if k]
        return mols, df[endpoint].to_numpy(dtype=float)

    train_mols, train_y = _load(DATA_DIR / f"train_{dataset}.csv")
    test_mols, test_y = _load(DATA_DIR / f"test_{dataset}.csv")

    half = len(test_mols) // 2
    val_mols, val_y = test_mols[:half], test_y[:half]
    test_mols, test_y = test_mols[half:], test_y[half:]
    return train_mols, train_y, val_mols, val_y, test_mols, test_y


# ---------------------------------------------------------------------------
# Per-combo benchmark
# ---------------------------------------------------------------------------


def _metric_row(
    dataset: str,
    endpoint: str,
    model_name: str,
    approach: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uq: np.ndarray,
) -> dict:
    top10_mae, bottom10_mae = _bucket_mae(y_true, y_pred, uq)
    residuals = _normalised_residuals(y_true, y_pred)
    return {
        "dataset": dataset,
        "endpoint": endpoint,
        "model": model_name,
        "approach": approach,
        "n_test": int(len(y_true)),
        "spearman": _spearman(residuals, uq),
        "top10_mae": top10_mae,
        "bottom10_mae": bottom10_mae,
    }


def benchmark_combo(
    dataset: str,
    endpoint: str,
    model_name: str,
    model_cls: type,
) -> list[dict]:
    train_mols, train_y, val_mols, val_y, test_mols, test_y = load_split(dataset, endpoint)
    rows: list[dict] = []

    # --- MC dropout on a single model (skip families with no inference-time dropout)
    if model_name not in MC_DROPOUT_UNSUPPORTED_MODELS:
        single = model_cls()
        single.fit(train_mols, train_y)
        single_preds = single.predict(test_mols)
        single_uq = single.compute_uncertainty(test_mols, num_iterations=MC_DROPOUT_ITERATIONS)
        rows.append(_metric_row(dataset, endpoint, model_name, "mc_dropout", test_y, single_preds, single_uq))

    # --- MVE on a single model (skip families with no MVE head) -----------
    if model_name not in MVE_UNSUPPORTED_MODELS:
        loss_fn = MVE_LOSS_FN.get(model_name, MVE_LOSS_FN_DEFAULT)
        mve = model_cls(uncertainty="mve", loss_fn=loss_fn)
        mve.fit(train_mols, train_y)
        mve_preds = mve.predict(test_mols)
        mve_uq = mve.compute_uncertainty(test_mols)
        rows.append(_metric_row(dataset, endpoint, model_name, "mve", test_y, mve_preds, mve_uq))

    # --- Deep ensemble (plain MC-dropout base — no MVE by default) --------
    ensemble = Ensemble(model_cls(), n_models=N_ENSEMBLE_MODELS)
    ensemble.fit(train_mols, train_y)
    ensemble_preds, ensemble_uq = ensemble.predict(test_mols)
    ensemble_val_preds, ensemble_val_uq = ensemble.predict(val_mols)
    rows.append(_metric_row(dataset, endpoint, model_name, "ensemble", test_y, ensemble_preds, ensemble_uq))

    # --- Error-model calibrator on top of the ensemble --------------------
    calibrator = EMRegressionCalibration(
        log10=True,
        algorithm="GradientBoostingClassifier",
        algorithm_params={"n_estimators": 500, "subsample": 0.9, "learning_rate": 0.01},
    )
    calibrator.fit(val_mols, val_y, ensemble_val_preds, ensemble_val_uq)
    calibrator_uq = calibrator.compute_uncertainty(test_mols, ensemble_preds, ensemble_uq)
    # Calibrator returns a confidence-like score — invert so lower = more confident.
    calibrator_uq = 1 - _to_1d_array(calibrator_uq)
    rows.append(_metric_row(dataset, endpoint, model_name, "error_model", test_y, ensemble_preds, calibrator_uq))

    return rows


def _append_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _expected_approaches(model_name: str) -> set[str]:
    approaches = {"ensemble", "error_model"}
    if model_name not in MC_DROPOUT_UNSUPPORTED_MODELS:
        approaches.add("mc_dropout")
    if model_name not in MVE_UNSUPPORTED_MODELS:
        approaches.add("mve")
    return approaches


def _load_logged(path: Path) -> dict[tuple[str, str, str], set[str]]:
    logged: dict[tuple[str, str, str], set[str]] = {}
    if path.exists():
        prior = pd.read_csv(path)
        for r in prior.itertuples():
            logged.setdefault((r.dataset, r.endpoint, r.model), set()).add(r.approach)
    return logged


def run_benchmark(resume: bool = True) -> pd.DataFrame:
    """Run every (dataset, endpoint, model) combination into ``logging_df.csv``."""
    logged = _load_logged(LOG_PATH) if resume else {}
    if not resume and LOG_PATH.exists():
        LOG_PATH.unlink()
    if logged:
        print(f"Resuming — {len(logged)} (dataset, endpoint, model) combos already touched.")

    for dataset, endpoints in DATASETS.items():
        for endpoint in endpoints:
            for model_name, model_cls in MODELS.items():
                key = (dataset, endpoint, model_name)
                if _expected_approaches(model_name).issubset(logged.get(key, set())):
                    continue
                print(f"[running] dataset={dataset} endpoint={endpoint} model={model_name}")
                try:
                    rows = benchmark_combo(dataset, endpoint, model_name, model_cls)
                    _append_rows(rows, LOG_PATH)
                    logged[key] = logged.get(key, set()) | {r["approach"] for r in rows}
                except Exception:  # noqa: BLE001 — log and keep going
                    print(f"[failed] {key}")
                    traceback.print_exc()

    return pd.read_csv(LOG_PATH)


def merge_shards() -> Path:
    """Concatenate all per-task shards into ``logging_df.csv``."""
    shards = sorted(SHARD_DIR.glob("task_*.csv"))
    if not shards:
        raise FileNotFoundError(f"No shards found under {SHARD_DIR}")
    merged = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    merged.to_csv(LOG_PATH, index=False)
    print(f"Merged {len(shards)} shards → {LOG_PATH} ({len(merged)} rows)")
    return LOG_PATH


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge per-task shards into logging_df.csv and exit.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing logging_df.csv and start from scratch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.merge:
        merge_shards()
        return

    run_benchmark(resume=not args.no_resume)


if __name__ == "__main__":
    main()

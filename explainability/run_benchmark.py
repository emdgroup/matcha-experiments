"""Run MATCHA architectures against prepared WISP LogP neighborhoods."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matcha.explainability import MatchaExplainer
from matcha.sklearn import (
    AttentiveFPRegressor,
    ChempropRegressor,
    CNNRegressor,
    E3GNNRegressor,
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
from rdkit import Chem, rdBase
from scoring import parse_indices, sum_at_indices
from sklearn.metrics import r2_score

HERE = Path(__file__).resolve().parent
PREPARED_DIR = HERE / "prepared"
WISP_DIR = HERE / "wisp"
MMP_SCORE_DIR = WISP_DIR / "mmp_scores"
DESCRIPTOR_SCORE_DIR = WISP_DIR / "descriptor_scores"
SUMMARY_PATH = WISP_DIR / "summary.csv"
SUMMARY_SHARD_DIR = HERE / "logs" / "wisp_summary_shards"
LIME_BOOTSTRAP_NUM = 25
SCORING_SCHEMA_VERSION = 4

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepared_fingerprint() -> str:
    paths = [
        PREPARED_DIR / "manifest.json",
        PREPARED_DIR / "split.csv",
        PREPARED_DIR / "mmp_pairs.csv",
        PREPARED_DIR / "counterfactuals.csv.gz",
        PREPARED_DIR / "neighborhood_status.csv",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prepared WISP artifacts: {missing}")
    manifest = json.loads(paths[0].read_text())
    runtime = {
        "matcha_version": importlib.metadata.version("emd-matcha"),
        "rdkit_version": rdBase.rdkitVersion,
    }
    mismatches = {
        key: {"prepared": manifest.get(key), "runtime": value}
        for key, value in runtime.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Prepared artifacts are incompatible with runtime: {mismatches}"
        )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(_sha256(path).encode())
    return digest.hexdigest()


def _to_1d(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim > 1:
        array = array[:, 0]
    return array.reshape(-1)


def _molecules(smiles: list[str]) -> list:
    mols = [Chem.MolFromSmiles(value) for value in smiles]
    if any(mol is None for mol in mols):
        raise ValueError("Prepared artifacts contain an invalid SMILES value.")
    return mols


def _predict_batches(model, smiles: list[str], batch_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(smiles), batch_size):
        batch = _molecules(smiles[start : start + batch_size])
        predictions.append(_to_1d(model.predict(batch)))
    return np.concatenate(predictions) if predictions else np.array([], dtype=float)


def _write_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(temp, index=False, **kwargs)
    temp.replace(path)


def _descriptor_rows(df: pd.DataFrame, compound_id: str) -> pd.DataFrame:
    details = df.loc[df["Descriptor"] != "Local fit R2"].copy()
    details["absolute_coefficient"] = details["Coefficient"].abs()
    details["rank"] = details["absolute_coefficient"].rank(
        method="min", ascending=False
    ).astype(int)
    details.insert(0, "compound_id", compound_id)
    return details.rename(
        columns={
            "Descriptor": "descriptor",
            "Coefficient": "coefficient",
            "Standard deviation": "bootstrap_std",
        }
    )[
        [
            "compound_id",
            "descriptor",
            "coefficient",
            "absolute_coefficient",
            "bootstrap_std",
            "rank",
        ]
    ]


def _compound_explanations(
    neighborhoods: pd.DataFrame,
    statuses: pd.DataFrame,
    prediction_by_smiles: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid = statuses.loc[statuses["status"] != "valid", ["compound_id", "status"]]
    if len(invalid):
        raise RuntimeError(
            "Prepared neighborhoods contain invalid entries:\n"
            + invalid.to_string(index=False)
        )
    neighborhood_ids = set(neighborhoods["compound_id"])
    if set(statuses["compound_id"]) != neighborhood_ids:
        raise RuntimeError("Neighborhood and status compound cohorts differ.")

    explainer = MatchaExplainer(random_seed=0)
    compound_rows = []
    descriptor_frames = []
    for compound_id, group in neighborhoods.groupby("compound_id", sort=False):
        group = group.sort_values("neighbor_index")
        query_smiles = str(group.iloc[0]["query_smiles"])
        smiles = group["neighbor_smiles"].astype(str).tolist()
        predictions = np.asarray(
            [prediction_by_smiles[value] for value in smiles], dtype=float
        )
        try:
            explanation = explainer.explain(
                mols=_molecules(smiles),
                predictions=predictions,
                bootstrap_num=LIME_BOOTSTRAP_NUM,
            )
        except Exception as error:
            raise RuntimeError(f"LIME explanation failed for {compound_id}.") from error
        compound_rows.append(
            {
                "compound_id": compound_id,
                "prediction": prediction_by_smiles[query_smiles],
                "atom_scores": explanation.atom_weights.tolist(),
            }
        )
        descriptor_frames.append(_descriptor_rows(explanation.df_desc, compound_id))

    return (
        pd.DataFrame(compound_rows),
        pd.concat(descriptor_frames, ignore_index=True),
    )


def _score_mmp_pairs(pairs: pd.DataFrame, compounds: pd.DataFrame) -> pd.DataFrame:
    explanations = compounds.set_index("compound_id").to_dict("index")
    rows = []
    for pair in pairs.itertuples(index=False):
        left = explanations[pair.compound_id_1]
        right = explanations[pair.compound_id_2]
        left_scores = np.asarray(left["atom_scores"], dtype=float)
        right_scores = np.asarray(right["atom_scores"], dtype=float)
        left_variable = parse_indices(pair.variable_atom_indices_1)
        right_variable = parse_indices(pair.variable_atom_indices_2)
        rows.append(
            {
                "pair_id": pair.pair_id,
                "prediction_delta": float(left["prediction"] - right["prediction"]),
                "whole_attribution_delta": float(
                    left_scores.sum() - right_scores.sum()
                ),
                "variable_attribution_delta": sum_at_indices(left_scores, left_variable)
                - sum_at_indices(right_scores, right_variable),
            }
        )
    return pd.DataFrame(rows)


def run_model(model_name: str, batch_size: int, resume: bool = True) -> Path:
    prepared_fingerprint = _prepared_fingerprint()
    mmp_path = MMP_SCORE_DIR / f"{model_name}.csv.gz"
    descriptor_path = DESCRIPTOR_SCORE_DIR / f"{model_name}.csv.gz"
    summary_shard_path = SUMMARY_SHARD_DIR / f"{model_name}.csv"
    expected_outputs = [mmp_path, descriptor_path, summary_shard_path]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if resume and existing_outputs:
        if len(existing_outputs) != len(expected_outputs):
            raise RuntimeError(
                f"Existing {model_name} outputs are incomplete; rerun with --no-resume."
            )
        prior = pd.read_csv(summary_shard_path)
        required_columns = {
            "model",
            "prepared_fingerprint",
            "scoring_schema_version",
            "mmp_sha256",
            "descriptor_sha256",
        }
        compatible = (
            len(prior) == 1
            and required_columns.issubset(prior.columns)
            and prior.iloc[0]["model"] == model_name
            and prior.iloc[0]["prepared_fingerprint"] == prepared_fingerprint
            and int(prior.iloc[0]["scoring_schema_version"])
            == SCORING_SCHEMA_VERSION
            and prior.iloc[0]["mmp_sha256"] == _sha256(mmp_path)
            and prior.iloc[0]["descriptor_sha256"] == _sha256(descriptor_path)
        )
        if compatible:
            print(f"[skipped] {model_name}: compatible outputs already exist")
            return summary_shard_path
        raise RuntimeError(
            f"Existing {model_name} outputs are stale; rerun with --no-resume."
        )
    if not resume and summary_shard_path.exists():
        summary_shard_path.unlink()

    split = pd.read_csv(PREPARED_DIR / "split.csv", dtype={"compound_id": str})
    neighborhoods = pd.read_csv(
        PREPARED_DIR / "counterfactuals.csv.gz", dtype={"compound_id": str}
    )
    statuses = pd.read_csv(
        PREPARED_DIR / "neighborhood_status.csv", dtype={"compound_id": str}
    ).fillna({"message": ""})
    pairs = pd.read_csv(
        PREPARED_DIR / "mmp_pairs.csv",
        dtype={"compound_id_1": str, "compound_id_2": str},
    )

    train = split.loc[split["split"] == "train"]
    test = split.loc[split["split"] == "test"]
    model = MODELS[model_name]()
    model.fit(
        _molecules(train["smiles"].tolist()), train["target"].to_numpy(dtype=float)
    )

    test_predictions = _to_1d(model.predict(_molecules(test["smiles"].tolist())))
    unique_smiles = list(dict.fromkeys(neighborhoods["neighbor_smiles"].astype(str)))
    neighborhood_predictions = _predict_batches(model, unique_smiles, batch_size)
    prediction_by_smiles = dict(zip(unique_smiles, neighborhood_predictions))

    compounds, descriptors = _compound_explanations(
        neighborhoods, statuses, prediction_by_smiles
    )
    scored_pairs = _score_mmp_pairs(pairs, compounds)
    test_r2 = float(
        r2_score(test["target"].to_numpy(dtype=float), test_predictions)
    )
    _write_csv(scored_pairs, mmp_path, compression="gzip")
    _write_csv(descriptors, descriptor_path, compression="gzip")
    summary_shard = pd.DataFrame(
        [
            {
                "model": model_name,
                "test_r2": test_r2,
                "prepared_fingerprint": prepared_fingerprint,
                "scoring_schema_version": SCORING_SCHEMA_VERSION,
                "mmp_sha256": _sha256(mmp_path),
                "descriptor_sha256": _sha256(descriptor_path),
            }
        ]
    )
    _write_csv(summary_shard, summary_shard_path)
    print(summary_shard.iloc[0].to_string())
    return summary_shard_path


def merge_summaries() -> Path:
    paths = sorted(SUMMARY_SHARD_DIR.glob("*.csv"))
    expected_models = set(MODELS)
    observed_files = {path.stem for path in paths}
    if observed_files != expected_models:
        raise RuntimeError(
            "Cannot merge incomplete summaries; "
            f"missing={sorted(expected_models - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_models)}."
        )

    shards = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    if len(shards) != len(MODELS) or set(shards["model"]) != expected_models:
        raise RuntimeError("Summary shards contain duplicate or incorrect model rows.")
    if (
        set(shards["prepared_fingerprint"]) != {_prepared_fingerprint()}
        or set(shards["scoring_schema_version"]) != {SCORING_SCHEMA_VERSION}
    ):
        raise RuntimeError("Summary shards were produced from incompatible inputs.")
    for row in shards.itertuples(index=False):
        mmp_path = MMP_SCORE_DIR / f"{row.model}.csv.gz"
        descriptor_path = DESCRIPTOR_SCORE_DIR / f"{row.model}.csv.gz"
        if not mmp_path.exists() or not descriptor_path.exists():
            raise RuntimeError(f"Missing plotting outputs for {row.model}.")
        if _sha256(mmp_path) != row.mmp_sha256:
            raise RuntimeError(f"MMP output hash mismatch for {row.model}.")
        if _sha256(descriptor_path) != row.descriptor_sha256:
            raise RuntimeError(f"Descriptor output hash mismatch for {row.model}.")

    model_order = {model_name: index for index, model_name in enumerate(MODELS)}
    summary = shards[["model", "test_r2"]].copy()
    summary["model_order"] = summary["model"].map(model_order)
    summary = summary.sort_values("model_order").drop(columns="model_order")
    _write_csv(summary, SUMMARY_PATH)
    print(f"Merged {len(paths)} model summaries into {SUMMARY_PATH}.")
    return SUMMARY_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = list(MODELS)
    if args.list_models:
        for model_name in model_names:
            print(model_name)
        return
    if args.merge:
        merge_summaries()
        return

    selected = [args.model] if args.model is not None else model_names

    for model_name in selected:
        print(f"[running] {model_name}")
        run_model(model_name, args.batch_size, resume=not args.no_resume)


if __name__ == "__main__":
    main()

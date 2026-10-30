"""Prepare the shared WISP split, MMP pairs, and MATCHA LIME neighborhoods."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from matcha.explainability import MatchaExplainer
from rdkit import Chem, rdBase
from scoring import (
    complementary_indices,
    indices_with_neighbors,
    unmatched_atom_indices,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_DATASET = REPO_ROOT / "datasets" / "wisp" / "logP.csv"
DEFAULT_OUTPUT = HERE / "prepared"
SPLIT_SEED = 6
TEST_FRACTION = 0.2
PREPARATION_SCHEMA_VERSION = 2
GENERATOR_PARAMS = {
    "positional_analogue_scanning_params": {},
    "nitrogen_walk_params": {},
    "reverse_positional_analogue_scanning": True,
    "generation_timeout": 60.0,
    "num_sample": 100,
    "random_seed": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    df.to_csv(temp, index=False, **kwargs)
    temp.replace(path)


def _canonicalize(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def prepare_split(dataset_path: Path, output_dir: Path) -> pd.DataFrame:
    data = pd.read_csv(
        dataset_path,
        usecols=["CMPD_CHEMBLID", "smiles", "exp"],
        dtype={"CMPD_CHEMBLID": str},
    ).dropna()
    data["smiles"] = data["smiles"].map(_canonicalize)
    data = data.dropna(subset=["smiles"])
    data = data.drop_duplicates(subset=["smiles"], keep="first").reset_index(drop=True)
    data = data.rename(columns={"CMPD_CHEMBLID": "compound_id", "exp": "target"})

    test_count = round(len(data) * TEST_FRACTION)
    test_indices = set(data.sample(n=test_count, random_state=SPLIT_SEED).index)
    data["split"] = [
        "test" if index in test_indices else "train" for index in data.index
    ]
    data = data[["compound_id", "smiles", "target", "split"]]
    _atomic_csv(data, output_dir / "split.csv")
    return data


def _constant_atom_count(smarts: str) -> int:
    mol = Chem.MolFromSmarts(smarts)
    if mol is None:
        raise ValueError(f"Invalid MMP constant SMARTS: {smarts!r}")
    return mol.GetNumAtoms()


def prepare_mmp_pairs(split: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    executable = shutil.which("mmpdb")
    if executable is None:
        raise RuntimeError(
            "mmpdb is required; install the wisp-exp optional dependency."
        )

    columns = [
        "mmp_smiles_1",
        "mmp_smiles_2",
        "compound_id_1",
        "compound_id_2",
        "transformation",
        "constant",
    ]
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary_dir:
        build_dir = Path(temporary_dir)
        smiles_path = build_dir / "data.smi"
        fragment_path = build_dir / "fragments.fragdb"
        index_path = build_dir / "index.csv"
        split[["smiles", "compound_id", "target"]].to_csv(
            smiles_path, sep="\t", index=False, header=False
        )
        subprocess.run(
            [
                executable,
                "fragment",
                "--num-cuts",
                "1",
                str(smiles_path),
                "-o",
                str(fragment_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                executable,
                "index",
                "--max-variable-ratio",
                "0.2",
                str(fragment_path),
                "-o",
                str(index_path),
                "--out",
                "csv",
            ],
            check=True,
        )
        pairs = pd.read_csv(index_path, sep="\t", header=None, names=columns, dtype=str)
    pairs["constant_atom_count"] = pairs["constant"].map(_constant_atom_count)
    pairs["compound_pair"] = [
        json.dumps(sorted((left, right)))
        for left, right in zip(pairs["compound_id_1"], pairs["compound_id_2"])
    ]
    pairs = (
        pairs.sort_values(
            ["compound_pair", "constant_atom_count"],
            ascending=[True, False],
        )
        .drop_duplicates("compound_pair", keep="first")
        .drop(columns="compound_pair")
    )

    indexed = split.set_index("compound_id")
    for side in (1, 2):
        ids = pairs[f"compound_id_{side}"]
        pairs[f"smiles_{side}"] = ids.map(indexed["smiles"])
        pairs[f"target_{side}"] = ids.map(indexed["target"])
        pairs[f"split_{side}"] = ids.map(indexed["split"])
    pairs = pairs.loc[
        (pairs["split_1"] == "test") & (pairs["split_2"] == "test")
    ].copy()
    pairs = pairs.dropna(subset=["smiles_1", "smiles_2"]).reset_index(drop=True)
    pairs.insert(0, "pair_id", [f"mmp_{index:06d}" for index in range(len(pairs))])

    for side in (1, 2):
        smiles_column = f"smiles_{side}"
        variable_column = f"variable_atom_indices_{side}"
        neighbor_column = f"variable_neighbor_indices_{side}"
        constant_column = f"constant_atom_indices_{side}"
        pairs[variable_column] = [
            unmatched_atom_indices(smiles, constant)
            for smiles, constant in zip(pairs[smiles_column], pairs["constant"])
        ]
        pairs[neighbor_column] = [
            indices_with_neighbors(smiles, indices)
            for smiles, indices in zip(pairs[smiles_column], pairs[variable_column])
        ]
        pairs[constant_column] = [
            complementary_indices(smiles, indices)
            for smiles, indices in zip(pairs[smiles_column], pairs[variable_column])
        ]
        for column in (variable_column, neighbor_column, constant_column):
            pairs[column] = pairs[column].map(json.dumps)

    output_columns = [
        "pair_id",
        "compound_id_1",
        "compound_id_2",
        "smiles_1",
        "smiles_2",
        "target_1",
        "target_2",
        "transformation",
        "constant",
        "constant_atom_count",
        "variable_atom_indices_1",
        "variable_atom_indices_2",
        "variable_neighbor_indices_1",
        "variable_neighbor_indices_2",
        "constant_atom_indices_1",
        "constant_atom_indices_2",
    ]
    pairs = pairs[output_columns]
    _atomic_csv(pairs, output_dir / "mmp_pairs.csv")
    return pairs


def _save_neighborhoods(
    neighborhoods: list[dict], statuses: list[dict], output_dir: Path
) -> None:
    _atomic_csv(
        pd.DataFrame(neighborhoods),
        output_dir / "counterfactuals.csv.gz",
        compression="gzip",
    )
    _atomic_csv(pd.DataFrame(statuses), output_dir / "neighborhood_status.csv")


def prepare_neighborhoods(split: pd.DataFrame, output_dir: Path) -> None:
    neighborhood_path = output_dir / "counterfactuals.csv.gz"
    status_path = output_dir / "neighborhood_status.csv"
    neighborhoods = (
        pd.read_csv(neighborhood_path, dtype={"compound_id": str}).to_dict("records")
        if neighborhood_path.exists()
        else []
    )
    statuses = (
        pd.read_csv(status_path, dtype={"compound_id": str}).to_dict("records")
        if status_path.exists()
        else []
    )
    statuses = [row for row in statuses if row["status"] == "valid"]
    completed = {row["compound_id"] for row in statuses}
    neighborhoods = [row for row in neighborhoods if row["compound_id"] in completed]
    explainer = MatchaExplainer(**GENERATOR_PARAMS)

    test = split.loc[split["split"] == "test"]
    for count, row in enumerate(test.itertuples(index=False), start=1):
        if row.compound_id in completed:
            continue

        query = Chem.MolFromSmiles(row.smiles)
        query_smiles = Chem.MolToSmiles(query)
        molecule_smiles = [query_smiles]
        status = "valid"
        message = ""
        try:
            molecule_smiles.extend(
                Chem.MolToSmiles(mol) for mol in explainer.generate_analogues(query)
            )
            molecule_smiles = list(dict.fromkeys(molecule_smiles))
            if len(molecule_smiles) < 3:
                status = "undersized"
        except Exception as error:  # noqa: BLE001
            status = "error"
            message = f"{type(error).__name__}: {error}".replace("\n", " ")

        neighborhoods.extend(
            {
                "compound_id": row.compound_id,
                "query_smiles": query_smiles,
                "neighbor_index": index,
                "neighbor_smiles": smiles,
                "is_query": index == 0,
            }
            for index, smiles in enumerate(molecule_smiles)
        )
        statuses.append(
            {
                "compound_id": row.compound_id,
                "query_smiles": query_smiles,
                "status": status,
                "message": message,
                "n_analogues": len(molecule_smiles) - 1,
            }
        )
        if status == "valid":
            completed.add(row.compound_id)
        if count % 10 == 0:
            _save_neighborhoods(neighborhoods, statuses, output_dir)

    _save_neighborhoods(neighborhoods, statuses, output_dir)


def _matcha_version() -> str:
    try:
        return importlib.metadata.version("emd-matcha")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def expected_manifest(dataset_path: Path) -> dict:
    return {
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "dataset": dataset_path.name,
        "dataset_sha256": _sha256(dataset_path),
        "target": "experimental_logP",
        "split_seed": SPLIT_SEED,
        "test_fraction": TEST_FRACTION,
        "mmpdb": {
            "version": importlib.metadata.version("mmpdb"),
            "num_cuts": 1,
            "max_variable_ratio": 0.2,
        },
        "matcha_version": _matcha_version(),
        "rdkit_version": rdBase.rdkitVersion,
        "generator_params": GENERATOR_PARAMS,
    }


def write_manifest(manifest: dict, output_dir: Path) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = expected_manifest(args.dataset)
    prepared_paths = [
        args.output_dir / "split.csv",
        args.output_dir / "mmp_pairs.csv",
        args.output_dir / "counterfactuals.csv.gz",
        args.output_dir / "neighborhood_status.csv",
    ]
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != manifest:
            raise RuntimeError(
                "Prepared artifacts are incompatible with the current dataset, "
                "packages, or benchmark parameters."
            )
    elif any(path.exists() for path in prepared_paths):
        raise RuntimeError("Prepared artifacts exist without a provenance manifest.")
    else:
        write_manifest(manifest, args.output_dir)

    split_path = args.output_dir / "split.csv"
    split = (
        pd.read_csv(split_path, dtype={"compound_id": str})
        if split_path.exists()
        else prepare_split(args.dataset, args.output_dir)
    )
    if not (args.output_dir / "mmp_pairs.csv").exists():
        prepare_mmp_pairs(split, args.output_dir)
    prepare_neighborhoods(split, args.output_dir)

    statuses = pd.read_csv(args.output_dir / "neighborhood_status.csv")
    pairs = pd.read_csv(args.output_dir / "mmp_pairs.csv")
    print(f"Prepared {len(split.loc[split['split'] == 'test'])} test compounds.")
    print(f"Prepared {len(pairs)} test-set MMPs.")
    print(statuses["status"].value_counts(dropna=False).to_string())
    invalid = statuses.loc[statuses["status"] != "valid"]
    if len(invalid):
        raise RuntimeError(
            f"Preparation left {len(invalid)} non-valid neighborhoods; rerun to retry."
        )


if __name__ == "__main__":
    main()

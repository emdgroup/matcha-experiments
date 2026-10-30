"""Train independent XGBoost baselines for every OpenADMET endpoint."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matcha.cli.utils import aggregate_scores, store_split_scores
from matcha.utils.metrics import process_regression
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from xgboost import XGBRegressor

SMILES_COLUMN = "SMILES"
ENDPOINT_SUFFIX = "_endpoint"
N_DESCRIPTORS = 217
MODEL_NAME = "xgb_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost baselines on the OpenADMET train/test splits."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/openadmet"),
        help="Directory containing the OpenADMET train_*.csv and test_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval"),
        help="Root evaluation directory.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples used for metrics.",
    )
    return parser.parse_args()


def dataset_files(data_dir: Path) -> list[tuple[str, Path, Path]]:
    datasets = []

    for train_file in data_dir.glob("train_*.csv"):
        dataset = train_file.stem.removeprefix("train_")
        test_file = train_file.with_name(f"test_{dataset}.csv")
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test split for dataset '{dataset}'")
        datasets.append((dataset, train_file, test_file))

    if not datasets:
        raise ValueError(f"No train/test dataset pairs found in {data_dir}")
    return sorted(datasets)


def endpoint_names(train_file: Path, test_file: Path) -> list[str]:
    train_columns = pd.read_csv(train_file, nrows=0).columns
    test_columns = set(pd.read_csv(test_file, nrows=0).columns)
    return [
        column
        for column in train_columns
        if column.endswith(ENDPOINT_SUFFIX) and column in test_columns
    ]


def load_task(
    csv_file: Path, endpoint: str
) -> tuple[pd.DataFrame, list[Chem.Mol], np.ndarray, np.ndarray]:
    dataset = pd.read_csv(csv_file, usecols=[SMILES_COLUMN, endpoint]).dropna(
        subset=[SMILES_COLUMN]
    )

    molecules = [Chem.MolFromSmiles(smiles) for smiles in dataset[SMILES_COLUMN]]
    valid_molecules = np.array([molecule is not None for molecule in molecules])
    dataset = dataset.loc[valid_molecules].reset_index(drop=True)
    molecules = [molecule for molecule in molecules if molecule is not None]

    valid_targets = dataset[endpoint].notna().to_numpy()
    task = dataset.loc[valid_targets].reset_index(drop=True)
    task_molecules = [
        molecule for molecule, valid in zip(molecules, valid_targets) if valid
    ]
    return (
        task,
        task_molecules,
        dataset[endpoint].to_numpy(dtype=float),
        np.flatnonzero(valid_targets),
    )


def descriptor_matrix(
    smiles: pd.Series,
    molecules: list[Chem.Mol],
    calculator: MolecularDescriptorCalculator,
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    for smiles_value, molecule in zip(smiles, molecules):
        if smiles_value not in cache:
            descriptors = np.asarray(calculator.CalcDescriptors(molecule), dtype=float)
            cache[smiles_value] = np.nan_to_num(
                descriptors, nan=0.0, posinf=10000.0, neginf=-10000.0
            )
    return np.vstack([cache[smiles_value] for smiles_value in smiles])


def write_json(path: Path, data: dict) -> None:
    with path.open("w") as output_file:
        json.dump(data, output_file, indent=4)


def run_dataset(
    dataset: str,
    train_file: Path,
    test_file: Path,
    output_root: Path,
    n_bootstrap: int,
    calculator: MolecularDescriptorCalculator,
    descriptor_cache: dict[str, np.ndarray],
) -> None:
    output_dir = output_root / dataset / MODEL_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    performance: dict = {}

    for endpoint in endpoint_names(train_file, test_file):
        train, train_molecules, _, _ = load_task(train_file, endpoint)
        test, test_molecules, test_targets, test_positions = load_task(
            test_file, endpoint
        )
        train_x = descriptor_matrix(
            train[SMILES_COLUMN], train_molecules, calculator, descriptor_cache
        )
        test_x = descriptor_matrix(
            test[SMILES_COLUMN], test_molecules, calculator, descriptor_cache
        )
        train_y = train[endpoint].to_numpy(dtype=float)
        test_y = test[endpoint].to_numpy(dtype=float)

        model = XGBRegressor()
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        bootstrap_predictions = np.full(len(test_targets), np.nan)
        bootstrap_predictions[test_positions] = predictions

        with np.errstate(divide="ignore", invalid="ignore"):
            store_split_scores(
                performance,
                0,
                endpoint,
                process_regression,
                n_bootstrap,
                1.0,
                test_targets,
                bootstrap_predictions,
                log10=False,
            )

        prediction_data = pd.DataFrame(
            {
                "true": test_y,
                "pred": predictions,
            }
        )
        prediction_data.to_csv(output_dir / f"{endpoint}_0_preds.csv", index=False)
        print(
            f"{dataset}/{endpoint}: trained on {len(train):,}, "
            f"tested on {len(test):,}"
        )

    performance["mean"] = aggregate_scores(performance, "mean")
    performance["std"] = aggregate_scores(performance, "std")
    write_json(output_dir / "performance.json", performance)


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1")

    descriptor_names = [name for name, _ in Descriptors._descList]
    if len(descriptor_names) != N_DESCRIPTORS:
        raise RuntimeError(
            f"Expected {N_DESCRIPTORS} RDKit descriptors, found {len(descriptor_names)}"
        )

    calculator = MolecularDescriptorCalculator(descriptor_names)
    descriptor_cache: dict[str, np.ndarray] = {}
    for dataset, train_file, test_file in dataset_files(args.data_dir):
        run_dataset(
            dataset,
            train_file,
            test_file,
            args.output_dir,
            args.n_bootstrap,
            calculator,
            descriptor_cache,
        )


if __name__ == "__main__":
    main()

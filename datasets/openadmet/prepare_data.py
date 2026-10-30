#!/usr/bin/env python3
"""
Download and preprocess OpenADMET expansion challenge raw data.

Steps
-----
1. Download expansion_data_raw.csv from HuggingFace.
2. Derive the train/test split by cross-referencing molecule names in
   expansion_data_train.csv and expansion_data_test.csv.
3. Drop the RLM CLint column.
4. Parse censored cells (e.g. "> 30") into a float value and an operator.
5. Scale every endpoint except logD as ``log10(y + 1)``.
6. Replace Molecule Name with an integer ``mol_id`` (e.g. "E-0001123" → 1123).
7. For each of 4 task groups, write train_<group>.csv / test_<group>.csv,
   keeping only rows with at least one non-NaN endpoint for that group.
   OPERATOR columns are numbered locally (OPERATOR_1, OPERATOR_2, …).

Task groups
-----------
physchem : LogD, KSOL
caco2    : CACO2_Papp_AB, CACO2_ER
clint    : HLM_CLint, MLM_CLint
unbound  : MPPB, MBPB, MGMB
"""

import io
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HF_BASE = (
    "https://huggingface.co/datasets/openadmet"
    "/openadmet-expansionrx-challenge-data/resolve/6b898ccc43d10d25b230fb09e22a6e30c30022b5"
)
HF_RAW_URL   = f"{_HF_BASE}/expansion_data_raw.csv"
HF_TRAIN_URL = f"{_HF_BASE}/expansion_data_train.csv"
HF_TEST_URL  = f"{_HF_BASE}/expansion_data_test.csv"

OUT_DIR = Path(__file__).resolve().parent

CENSOR_RE = re.compile(r"^\s*(>=|<=|>|<)\s*([\d.eE+\-]+)\s*$")

_CLINT_RAT_RE = re.compile(r"clint.*rat|rat.*clint|rlm.*clint|clint.*rlm", re.IGNORECASE)

# Task group definitions: ordered list of (canonical_name, original_col, apply_log_scale)
TASK_GROUPS: dict[str, list[tuple[str, str, bool]]] = {
    "physchem": [
        ("LogD_endpoint",           "LogD",                            False),
        ("KSOL_endpoint",           "KSOL",                            True),
    ],
    "caco2": [
        ("CACO2_Papp_AB_endpoint",  "Caco-2 Permeability Papp A>B",   True),
        ("CACO2_ER_endpoint",       "Caco-2 Permeability Efflux",      True),
    ],
    "clint": [
        ("HLM_CLint_endpoint",      "HLM CLint",                       True),
        ("MLM_CLint_endpoint",      "MLM CLint",                       True),
    ],
    "unbound": [
        ("MPPB_endpoint",           "MPPB",                            True),
        ("MBPB_endpoint",           "MBPB",                            True),
        ("MGMB_endpoint",           "MGMB",                            True),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_bytes(url: str) -> io.BytesIO:
    print(f"  Downloading {url} ...")
    with urllib.request.urlopen(url) as resp:
        return io.BytesIO(resp.read())


def _load_raw() -> pd.DataFrame:
    try:
        import huggingface_hub  # noqa: F401
        hf_url = (
            "hf://datasets/openadmet/openadmet-expansionrx-challenge-data"
            "/expansion_data_raw.csv"
        )
        df = pd.read_csv(hf_url, dtype=str)
        print(f"  Loaded via hf:// ({len(df)} rows)")
        return df
    except Exception:
        pass
    return pd.read_csv(_fetch_bytes(HF_RAW_URL), dtype=str)


def _mol_names(url: str) -> set[str]:
    df = pd.read_csv(_fetch_bytes(url), usecols=["Molecule Name"], dtype=str)
    return set(df["Molecule Name"].str.strip())


def _parse_censored(series: pd.Series) -> tuple[np.ndarray, list]:
    floats: list[float] = []
    operators: list = []
    for val in series:
        if pd.isna(val) or str(val).strip() == "":
            floats.append(np.nan)
            operators.append(None)
            continue
        s = str(val).strip()
        m = CENSOR_RE.match(s)
        if m:
            floats.append(float(m.group(2)))
            operators.append(m.group(1))
        else:
            try:
                floats.append(float(s))
            except ValueError:
                floats.append(np.nan)
            operators.append(None)
    return np.array(floats, dtype=np.float64), operators


def _build_group_df(
    base: pd.DataFrame,
    endpoints: list[tuple[str, str, bool]],
) -> pd.DataFrame:
    """
    Build a per-group DataFrame with local OPERATOR_N columns.

    Parameters
    ----------
    base:
        Full DataFrame with SMILES, mol_id, split, and raw string endpoint cols
        named by their original column names.
    endpoints:
        List of (canonical_name, original_col, apply_log_scale) tuples.

    Returns
    -------
    DataFrame with columns: SMILES, mol_id, split, <canonical_names...>,
    OPERATOR_N (only for endpoints that have at least one censor value).
    Rows where all canonical endpoint columns are NaN are dropped.
    """
    out = base[["SMILES", "mol_id", "split"]].copy()

    operator_cols: dict[str, pd.Series] = {}

    for local_idx, (canonical, original, log_scale) in enumerate(endpoints, start=1):
        floats, operators = _parse_censored(base[original])
        values = floats.astype(np.float64)

        if log_scale:
            values = np.log10(values + 1)

        out[canonical] = values

        if any(op is not None for op in operators):
            op_name = f"OPERATOR_{local_idx}"
            operator_cols[op_name] = pd.array(
                [op if op is not None else pd.NA for op in operators],
                dtype="string",
            )
            n_censored = sum(op is not None for op in operators)
            print(f"    {canonical}: {n_censored} censored rows → {op_name}")

    for op_name, op_series in operator_cols.items():
        out[op_name] = op_series

    # Drop rows where every endpoint is NaN
    ep_names = [canonical for canonical, _, _ in endpoints]
    out = out[out[ep_names].notna().any(axis=1)].reset_index(drop=True)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching raw data ...")
    df = _load_raw()
    print(f"Raw data: {len(df)} rows × {len(df.columns)} cols")
    print(f"Columns: {list(df.columns)}")

    # Detect SMILES and Molecule Name columns
    mol_name_col = next(
        (c for c in df.columns if re.search(r"molecule.?name|compound.?name", c, re.IGNORECASE)),
        None,
    )
    smiles_col = next(
        (c for c in df.columns if re.match(r"^smiles$", c, re.IGNORECASE)),
        None,
    )
    if not smiles_col or not mol_name_col:
        sys.exit(f"Could not detect SMILES / molecule-name columns in: {list(df.columns)}")

    # Assign split
    print("\nFetching split annotations ...")
    train_names = _mol_names(HF_TRAIN_URL)
    test_names  = _mol_names(HF_TEST_URL)
    print(f"  Train molecules: {len(train_names)}, Test molecules: {len(test_names)}")

    def _assign_split(name: str) -> str | None:
        n = str(name).strip()
        if n in train_names:
            return "train"
        if n in test_names:
            return "test"
        return None

    df["split"] = df[mol_name_col].map(_assign_split)
    n_unknown = df["split"].isna().sum()
    if n_unknown:
        print(f"  WARNING: {n_unknown} rows not found in either split file — will be dropped")
    df = df[df["split"].notna()].copy()

    # Molecule Name → mol_id
    df["mol_id"] = (
        df[mol_name_col]
        .str.extract(r"(\d+)$")[0]
        .astype(float)
        .astype("Int64")
    )

    # Rename SMILES column to a canonical name
    df = df.rename(columns={smiles_col: "SMILES"})

    # Validate all expected original columns are present
    all_original = {orig for group in TASK_GROUPS.values() for _, orig, _ in group}
    missing = all_original - set(df.columns)
    if missing:
        sys.exit(f"Missing expected endpoint columns: {missing}")

    # Build per-group train/test files
    print()
    for group_name, endpoints in TASK_GROUPS.items():
        print(f"Processing group '{group_name}' ...")
        group_df = _build_group_df(df, endpoints)

        train_df = group_df[group_df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        test_df  = group_df[group_df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

        train_path = OUT_DIR / f"train_{group_name}.csv"
        test_path  = OUT_DIR / f"test_{group_name}.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        ep_names = [c for c, _, _ in endpoints]
        print(f"  Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        print(f"  Columns: {list(train_df.columns)}")
        print()

    print("Done.")


if __name__ == "__main__":
    main()

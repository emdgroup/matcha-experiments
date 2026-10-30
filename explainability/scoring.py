"""WISP matched-molecular-pair scoring helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

import numpy as np
from rdkit import Chem


def unmatched_atom_indices(smiles: str, constant_smarts: str) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    matched: set[int] = set()
    match_mol = Chem.Mol(mol)
    Chem.RemoveStereochemistry(match_mol)
    cleaned = re.sub(r"\[\*:\d+\]", "", constant_smarts)
    cleaned = re.sub(
        r"\[([A-Za-z][a-z]?)@@?H?([^\]]*)\]",
        r"[\1\2]",
        cleaned,
    )
    for fragment in cleaned.split("."):
        if not fragment:
            continue
        query = Chem.MolFromSmarts(fragment)
        if query is None:
            raise ValueError(f"Invalid constant SMARTS fragment: {fragment!r}")
        Chem.RemoveStereochemistry(query)
        matches = match_mol.GetSubstructMatches(query)
        if not matches:
            raise ValueError(
                f"Constant SMARTS fragment {fragment!r} does not match {smiles!r}."
            )

        selected = matches[0]
        for match in matches:
            match_set = set(match)
            external_bonds = sum(
                neighbor.GetIdx() not in match_set
                for atom_index in match
                for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
            )
            if external_bonds == 1:
                selected = match
                break
        matched.update(selected)

    return sorted(set(range(mol.GetNumAtoms())) - matched)


def indices_with_neighbors(smiles: str, indices: Iterable[int]) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    selected = set(indices)
    expanded = set(selected)
    for atom_index in selected:
        expanded.update(
            neighbor.GetIdx()
            for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
        )
    return sorted(expanded)


def complementary_indices(smiles: str, indices: Iterable[int]) -> list[int]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    selected = set(indices)
    return [index for index in range(mol.GetNumAtoms()) if index not in selected]


def parse_indices(value: str | list[int]) -> list[int]:
    if isinstance(value, str):
        return [int(index) for index in json.loads(value)]
    return [int(index) for index in value]


def sum_at_indices(scores: np.ndarray, indices: Iterable[int]) -> float:
    return float(sum(scores[index] for index in indices))

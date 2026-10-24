"""Cross-validation splits over solubility points.

  random       points shuffled into k folds; same solute in train and test, so
               this is the optimistic upper bound
  scaffold     whole solutes grouped by Bemis-Murcko scaffold; closest to a new API
  solvent      leave-one-solvent-out for the most common solvents
  temperature  per pair, the top third of temperatures is test; train at low T,
               predict high T

Each function returns a list of Fold(name, train, test) with positional indices.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .vanthoff import PAIR_KEYS


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray


@lru_cache(maxsize=None)
def scaffold(smiles: str) -> str:
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles))
    # acyclic solutes have no scaffold; give each its own group instead of lumping them
    return scaf or f"acyclic:{smiles}"


def _group_kfold(groups: pd.Series, k: int, seed: int, prefix: str) -> list[Fold]:
    """Assign whole groups to k folds, largest group first into the smallest fold."""
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    # shuffle before the stable size sort so ties don't always land the same way
    sizes = sizes.iloc[rng.permutation(len(sizes))].sort_values(ascending=False, kind="stable")
    load, assign = np.zeros(k), {}
    for g, n in sizes.items():
        f = int(np.argmin(load))
        assign[g] = f
        load[f] += n
    fold_of = groups.map(assign).to_numpy()
    idx = np.arange(len(groups))
    return [Fold(f"{prefix}{f}", idx[fold_of != f], idx[fold_of == f]) for f in range(k)]


def random_folds(points: pd.DataFrame, k: int = 5, seed: int = 0) -> list[Fold]:
    perm = np.random.default_rng(seed).permutation(len(points))
    return [
        Fold(f"random{f}", np.sort(np.setdiff1d(perm, part)), np.sort(part))
        for f, part in enumerate(np.array_split(perm, k))
    ]


def scaffold_folds(points: pd.DataFrame, k: int = 5, seed: int = 0) -> list[Fold]:
    groups = points["solute_smiles"].map(scaffold).reset_index(drop=True)
    return _group_kfold(groups, k, seed, "scaffold")


def solvent_folds(points: pd.DataFrame, n_solvents: int = 10) -> list[Fold]:
    idx = np.arange(len(points))
    solv = points["solvent_key"].to_numpy()
    top = points["solvent_key"].value_counts().index[:n_solvents]
    names = (
        points.groupby("solvent_key")["solvent_name"].first().to_dict()
        if "solvent_name" in points.columns else {}
    )
    return [Fold(f"solvent:{names.get(s) or s}", idx[solv != s], idx[solv == s]) for s in top]


def temperature_folds(points: pd.DataFrame, frac: float = 1 / 3, min_temps: int = 4) -> list[Fold]:
    d = points.reset_index(drop=True)
    rank = d.groupby(PAIR_KEYS)["T_K"].rank(method="dense", ascending=False)
    n_t = d.groupby(PAIR_KEYS)["T_K"].transform("nunique")
    n_test = np.maximum(1, np.floor(n_t * frac))
    is_test = ((n_t >= min_temps) & (rank <= n_test)).to_numpy()
    idx = np.arange(len(d))
    return [Fold("temperature", idx[~is_test], idx[is_test])]


SPLITS = {
    "random": random_folds,
    "scaffold": scaffold_folds,
    "solvent": solvent_folds,
    "temperature": temperature_folds,
}

"""Molecular features for solutes and solvents.

Every model sees the same featurization: a fixed list of RDKit 2D
descriptors for solute (prefix u_) and solvent (prefix v_), plus folded
Morgan counts for the solute. Point-level models add 1000/T.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

DESCRIPTORS = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA", "NumHDonors", "NumHAcceptors",
    "NHOHCount", "NOCount", "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "HeavyAtomCount", "NumValenceElectrons", "BertzCT",
    "MaxAbsPartialCharge", "MinAbsPartialCharge",
]
_DESC_FUNCS = dict(Descriptors.descList)
N_FP_BITS = 256
_MORGAN_COUNT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_FP_BITS)
_MORGAN_AD = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    return mol


@lru_cache(maxsize=None)
def descriptors(smiles: str) -> tuple[float, ...]:
    mol = _mol(smiles)
    return tuple(float(_DESC_FUNCS[d](mol)) for d in DESCRIPTORS)


@lru_cache(maxsize=None)
def morgan_counts(smiles: str) -> tuple[int, ...]:
    return tuple(_MORGAN_COUNT.GetCountFingerprintAsNumPy(_mol(smiles)).tolist())


@lru_cache(maxsize=None)
def ad_fingerprint(smiles: str):
    """2048-bit Morgan fingerprint for applicability-domain similarity."""
    return _MORGAN_AD.GetFingerprint(_mol(smiles))


def _block(smiles: pd.Series, fn, columns: list[str]) -> pd.DataFrame:
    uniq = pd.unique(smiles)
    table = pd.DataFrame([fn(s) for s in uniq], index=uniq, columns=columns)
    return table.loc[smiles.to_numpy()].reset_index(drop=True)


def pair_features(solute_smiles: pd.Series, solvent_smiles: pd.Series, fp: bool = True) -> pd.DataFrame:
    parts = [
        _block(solute_smiles, descriptors, [f"u_{d}" for d in DESCRIPTORS]),
        _block(solvent_smiles, descriptors, [f"v_{d}" for d in DESCRIPTORS]),
    ]
    if fp:
        parts.append(_block(solute_smiles, morgan_counts, [f"u_fp{i}" for i in range(N_FP_BITS)]))
    return pd.concat(parts, axis=1)


def point_features(points: pd.DataFrame, fp: bool = True) -> pd.DataFrame:
    X = pair_features(points["solute_smiles"], points["solvent_smiles"], fp)
    X["inv_T_1000"] = 1000.0 / points["T_K"].to_numpy()
    return X


def max_similarity(query: list[str], reference: list[str], exclude_self: bool = False) -> np.ndarray:
    """Highest Tanimoto similarity of each query solute to the reference set."""
    ref_fps = [ad_fingerprint(s) for s in reference]
    out = np.zeros(len(query))
    for i, s in enumerate(query):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(ad_fingerprint(s), ref_fps))
        if exclude_self:
            sims = sims[np.array(reference) != s]
        out[i] = sims.max() if len(sims) else 0.0
    return out

"""Case-study APIs held out of every training and tuning step.

Matching uses the InChIKey connectivity block, so racemates, single
enantiomers and isotopologues of the same API are all held out together.
"""
from __future__ import annotations

import pandas as pd

from .standardize import connectivity_key, standardize_solute

CASE_STUDY_APIS = {
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "carbamazepine": "NC(=O)N1c2ccccc2C=Cc2ccccc21",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
}


def holdout_keys(apis: dict[str, str] = CASE_STUDY_APIS) -> dict[str, str]:
    """Map connectivity key -> API name."""
    out = {}
    for name, smi in apis.items():
        res = standardize_solute(smi)
        if not res.ok:
            raise ValueError(f"case-study SMILES for {name} failed: {res.error}")
        out[connectivity_key(res.inchikey)] = name
    return out


def split_holdout(
    df: pd.DataFrame, apis: dict[str, str] = CASE_STUDY_APIS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (development rows, held-out rows with a `case_study` column)."""
    keys = holdout_keys(apis)
    conn = df["solute_inchikey"].map(connectivity_key)
    is_held = conn.isin(keys)
    held = df[is_held].assign(case_study=conn[is_held].map(keys))
    return df[~is_held].copy(), held

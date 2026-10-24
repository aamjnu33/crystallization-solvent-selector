"""Solvent table: curated properties + RDKit descriptors, keyed by InChIKey.

`data/solvents_curated.csv` holds the hand-entered columns:
  ich_q3c_class  ICH Q3C(R8) residual-solvent class (1, 2, 3, or "none")
  bp_C           normal boiling point
  density_g_mL   liquid density near 25 C
  chem21         CHEM21 ranking (Prat et al., Green Chem. 2016, 18, 288);
                 blank where not yet checked against the paper

Every value here must be checked against its source before the repo is public;
see the data card's "Solvent table" section.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from .standardize import canonical_solvent

CURATED_PATH = Path(__file__).resolve().parents[2] / "data" / "solvents_curated.csv"


def _descriptors(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "n_heavy": mol.GetNumHeavyAtoms(),
        "frac_sp3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def load_curated(path: Path = CURATED_PATH) -> pd.DataFrame:
    cur = pd.read_csv(path, dtype={"ich_q3c_class": str})
    std = cur["smiles"].map(canonical_solvent)
    bad = [n for n, s in zip(cur["name"], std) if not s.ok]
    if bad:
        raise ValueError(f"curated solvents with invalid SMILES: {bad}")
    cur["solvent_key"] = [s.inchikey for s in std]
    if cur["solvent_key"].duplicated().any():
        dups = cur.loc[cur["solvent_key"].duplicated(keep=False), "name"].tolist()
        raise ValueError(f"duplicate solvents in curated table: {dups}")
    return cur


def build_solvent_table(used: pd.DataFrame, curated: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per solvent seen in the data.

    `used` needs columns solvent_key, solvent_smiles, solvent_name, n_points.
    Curated columns are blank for solvents not in the curated file.
    """
    curated = load_curated() if curated is None else curated
    desc = pd.DataFrame([_descriptors(s) for s in used["solvent_smiles"]], index=used.index)
    table = pd.concat([used, desc], axis=1).merge(
        curated[["solvent_key", "ich_q3c_class", "bp_C", "density_g_mL", "chem21"]],
        on="solvent_key", how="left",
    )
    table["curated"] = table["ich_q3c_class"].notna()
    return table.sort_values("n_points", ascending=False).reset_index(drop=True)

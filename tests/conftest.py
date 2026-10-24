"""Synthetic fixture: real structures, made-up solubilities from known
van't Hoff parameters, plus one deliberately broken row per cleaning rule."""
import numpy as np
import pandas as pd
import pytest

TEMPS = [283.15, 293.15, 303.15, 313.15, 323.15]

# (solute SMILES, solvent SMILES, solvent name, a, b)
PAIRS = [
    ("O=C(O)c1ccccc1", "CCO", "ethanol", 4.0, -2000.0),             # benzoic acid
    ("O=C(O)c1ccccc1", "CC(C)=O", "acetone", 3.5, -1900.0),
    ("O=C(O)c1ccccc1O", "CCOC(C)=O", "ethyl acetate", 5.0, -2500.0),  # salicylic acid
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", "O", "water", 6.0, -3500.0),      # caffeine
    ("CC(=O)Nc1ccc(O)cc1", "CCO", "ethanol", 3.0, -1800.0),          # paracetamol (case study)
    ("COc1cc(C=O)ccc1O", "CC#N", "acetonitrile", -5.0, 500.0),  # vanillin, wrong-sign slope
]


def lnx(a, b, t):
    return a + b / t


def make_raw() -> pd.DataFrame:
    rows = []
    for solute, solvent, name, a, b in PAIRS:
        for t in TEMPS:
            rows.append(dict(SMILES_Solute=solute, SMILES_Solvent=solvent, Solvent=name,
                             Temperature_K=t, **{"Solubility(mole_fraction)": float(np.exp(lnx(a, b, t)))},
                             Source="synthetic-A"))
    benzoic = "O=C(O)c1ccccc1"
    x293 = float(np.exp(lnx(4.0, -2000.0, 293.15)))
    rows += [
        # same point from a second source, within 2x -> merged, no disagreement
        dict(SMILES_Solute=benzoic, SMILES_Solvent="CCO", Solvent="ethanol", Temperature_K=293.15,
             **{"Solubility(mole_fraction)": x293 * 1.2}, Source="synthetic-B"),
        # sodium salt of benzoic acid in acetone: standardizes to the parent acid
        dict(SMILES_Solute="O=C([O-])c1ccccc1.[Na+]", SMILES_Solvent="CC(C)=O", Solvent="acetone",
             Temperature_K=333.15, **{"Solubility(mole_fraction)": float(np.exp(lnx(3.5, -1900.0, 333.15)))},
             Source="synthetic-B"),
        dict(SMILES_Solute="not_a_smiles", SMILES_Solvent="CCO", Solvent="ethanol", Temperature_K=298.15,
             **{"Solubility(mole_fraction)": 0.01}, Source="bad"),
        dict(SMILES_Solute=benzoic, SMILES_Solvent="CCO.O", Solvent="ethanol+water", Temperature_K=298.15,
             **{"Solubility(mole_fraction)": 0.01}, Source="mixed"),
        dict(SMILES_Solute=benzoic, SMILES_Solvent="CCO", Solvent="ethanol", Temperature_K=298.15,
             **{"Solubility(mole_fraction)": 1.7}, Source="bad"),
        dict(SMILES_Solute=benzoic, SMILES_Solvent="CCO", Solvent="ethanol", Temperature_K=25.0,
             **{"Solubility(mole_fraction)": 0.01}, Source="bad"),
        # (S)-naproxen: held out even though only the racemate is listed as a case study
        dict(SMILES_Solute="COc1ccc2cc(ccc2c1)[C@H](C)C(=O)O", SMILES_Solvent="CO", Solvent="methanol",
             Temperature_K=298.15, **{"Solubility(mole_fraction)": 0.02}, Source="synthetic-A"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def raw_csv(tmp_path):
    p = tmp_path / "raw" / "sample.csv"
    p.parent.mkdir()
    make_raw().to_csv(p, index=False)
    return p

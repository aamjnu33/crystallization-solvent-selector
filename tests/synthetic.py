"""Synthetic processed dataset for model tests.

Real structures; ln x generated from a hidden smooth function of RDKit
descriptors, exactly van't Hoff in T, plus 0.1 ln-unit noise. Models that
learn anything should beat the mean, and the main model's curves should
recover the van't Hoff form.
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, rdMolDescriptors

from solvsel.standardize import canonical_solvent, standardize_solute

SOLUTES = [
    "O=C(O)c1ccccc1", "O=C(O)c1ccccc1O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "COc1cc(C=O)ccc1O",
    "CC(=O)Nc1ccccc1", "CCOc1ccc(NC(C)=O)cc1", "CC(=O)Oc1ccccc1C(=O)O", "NC(=O)c1cccnc1",
    "NC(=O)c1ccncc1", "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "NC(=O)c1ccccc1", "O=C(O)c1ccc(O)cc1",
    "Nc1ccc(C(=O)O)cc1", "c1ccc2cc3ccccc3cc2c1", "c1ccc2ccccc2c1", "c1ccc(-c2ccccc2)cc1",
    "c1ccc2c(c1)Cc1ccccc1-2", "c1ccc2c(c1)Nc1ccccc1S2", "Nc1ccc(S(N)(=O)=O)cc1",
    "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
    "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1", "CC(C(=O)O)c1ccc(-c2ccccc2)c(F)c1",
    "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1", "Cc1cccc(Nc2ccccc2C(=O)O)c1C",
    "CN1C(C(=O)Nc2ccccn2)=C(O)c2ccccc2S1(=O)=O", "O=C1NC(=O)C(c2ccccc2)(c2ccccc2)N1",
    "CCN(CC)CC(=O)Nc1c(C)cccc1C", "Oc1ccc(O)cc1", "Oc1cccc(O)c1", "Oc1ccccc1O",
    "O=[N+]([O-])c1ccc(O)cc1", "O=C(C(=O)c1ccccc1)c1ccccc1", "O=C(c1ccccc1)c1ccccc1",
    "O=C(O)CCCCC(=O)O", "O=C(O)CCC(=O)O", "O=C(O)/C=C/c1ccccc1", "O=c1ccc2ccccc2o1",
]
SOLVENTS = {
    "water": "O", "methanol": "CO", "ethanol": "CCO", "2-propanol": "CC(C)O", "acetone": "CC(C)=O",
    "ethyl acetate": "CCOC(C)=O", "acetonitrile": "CC#N", "toluene": "Cc1ccccc1",
    "n-heptane": "CCCCCCC", "tetrahydrofuran": "C1CCOC1",
}
TEMPS = np.array([283.15, 293.15, 303.15, 313.15, 323.15])


def truth(solute: str, solvent: str) -> tuple[float, float]:
    """Hidden (ln x at 298.15 K, b in K)."""
    u, v = Chem.MolFromSmiles(solute), Chem.MolFromSmiles(solvent)
    lp_u, lp_v = Crippen.MolLogP(u), Crippen.MolLogP(v)
    tpsa = rdMolDescriptors.CalcTPSA(u)
    hbd_v = rdMolDescriptors.CalcNumHBD(v)
    ln_x_ref = -3.0 - 0.35 * (lp_u - lp_v) ** 2 - 0.01 * tpsa + 0.3 * hbd_v * (tpsa > 60)
    b = -2000.0 - 8.0 * tpsa - 250.0 * abs(lp_u - lp_v)
    return ln_x_ref, b


def make_points(seed: int = 0, noise: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for su in SOLUTES:
        s = standardize_solute(su)
        for name, sv in SOLVENTS.items():
            v = canonical_solvent(sv)
            ln_ref, b = truth(s.smiles, v.smiles)
            ln_x = ln_ref + b * (1 / TEMPS - 1 / 298.15) + rng.normal(0, noise, len(TEMPS))
            for t, y in zip(TEMPS, np.minimum(ln_x, -0.05)):
                rows.append(dict(solute_key=s.inchikey, solvent_key=v.inchikey, T_K=t, ln_x=y,
                                 solute_smiles=s.smiles, solvent_smiles=v.smiles, solvent_name=name,
                                 qc_flags=""))
    return pd.DataFrame(rows)


def make_solvents() -> pd.DataFrame:
    from solvsel.solvents import build_solvent_table
    used = pd.DataFrame([
        {"solvent_key": canonical_solvent(s).inchikey, "solvent_smiles": canonical_solvent(s).smiles,
         "solvent_name": n, "n_points": 1}
        for n, s in SOLVENTS.items()
    ])
    return build_solvent_table(used)

"""Physics baseline: ideal solubility with a Hansen-distance activity correction.

    ln x = -(dH_fus / R) (1/T - 1/T_m)  -  k * MW * Ra^2 / (R T)
    Ra^2 = 4 (dD1 - dD2)^2 + (dP1 - dP2)^2 + (dH1 - dH2)^2        [MPa = J/cm3]

The first term is the ideal (Schroeder-van Laar) solubility; it needs the
solute's melting point and heat of fusion. The second is a regular-solution
activity coefficient using Hansen distance, with the solute molar volume
approximated as MW / density folded into the one fitted scalar k (cm3/g).
It needs solute HSP and solvent HSP; without them the prediction is ideal.

Solutes with no melting data get NaN, and the benchmark reports coverage.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..solvents import HSP_COLS
from ..standardize import standardize_solute
from ..vanthoff import R_GAS

# Hansen's rule of thumb: ln(gamma) ~ V Ra^2 / (4 R T); V ~ MW / 1.3 g/cm3
DEFAULT_K = 1.0 / (4 * 1.3)
SOLUTE_HSP = ["solute_dD", "solute_dP", "solute_dH"]


def load_melting_props(path: Path) -> pd.DataFrame:
    """CSV with solute_smiles, Tm_K, dHfus_kJ_mol and optional solute_dD/dP/dH."""
    props = pd.read_csv(path)
    std = props["solute_smiles"].map(standardize_solute)
    props["solute_key"] = [s.inchikey for s in std]
    bad = props.loc[props["solute_key"].isna(), "solute_smiles"].tolist()
    if bad:
        raise ValueError(f"unparseable SMILES in {path}: {bad}")
    for c in SOLUTE_HSP:
        if c not in props:
            props[c] = np.nan
    return props.drop_duplicates("solute_key")[["solute_key", "Tm_K", "dHfus_kJ_mol", *SOLUTE_HSP]]


class PhysicsBaseline:
    name = "physics"

    def __init__(self, melting_props: pd.DataFrame | None = None, min_fit_points: int = 10):
        self.props = melting_props if melting_props is not None else pd.DataFrame(
            columns=["solute_key", "Tm_K", "dHfus_kJ_mol", *SOLUTE_HSP])
        self.min_fit_points = min_fit_points
        self.k = DEFAULT_K

    def _terms(self, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (ln x_ideal, MW * Ra^2 / (R T)); NaN where inputs are missing."""
        d = points[["solute_key", "solvent_key", "T_K", "solute_smiles"]].merge(
            self.props, on="solute_key", how="left"
        ).merge(self.solvents[["solvent_key", *HSP_COLS]], on="solvent_key", how="left")
        T = d["T_K"].to_numpy(float)
        tm = pd.to_numeric(d["Tm_K"], errors="coerce").to_numpy(float)
        dh = pd.to_numeric(d["dHfus_kJ_mol"], errors="coerce").to_numpy(float) * 1000.0
        ideal = -(dh / R_GAS) * (1.0 / T - 1.0 / tm)
        u = d[SOLUTE_HSP].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        v = d[HSP_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        ra2 = 4 * (u[:, 0] - v[:, 0]) ** 2 + (u[:, 1] - v[:, 1]) ** 2 + (u[:, 2] - v[:, 2]) ** 2
        mw = d["solute_smiles"].map(lambda s: standardize_solute(s).mw).to_numpy(float)
        return ideal, mw * ra2 / (R_GAS * T)

    def fit(self, train: pd.DataFrame, solvents: pd.DataFrame) -> "PhysicsBaseline":
        self.solvents = solvents
        ideal, h = self._terms(train)
        ok = np.isfinite(ideal) & np.isfinite(h) & (h > 0)
        if ok.sum() >= self.min_fit_points:
            # least squares for ln x - ideal = -k h, with k >= 0
            resid = train["ln_x"].to_numpy(float)[ok] - ideal[ok]
            self.k = max(0.0, float(-(resid @ h[ok]) / (h[ok] @ h[ok])))
        return self

    def predict(self, points: pd.DataFrame) -> pd.DataFrame:
        ideal, h = self._terms(points)
        ln_x = ideal - self.k * np.nan_to_num(h, nan=0.0)
        return pd.DataFrame({"ln_x_pred": np.minimum(ln_x, 0.0)})

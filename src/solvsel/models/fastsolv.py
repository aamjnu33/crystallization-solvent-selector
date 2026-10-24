"""FastSolv reference model (Attia et al., 2025), wrapped for the benchmark.

FastSolv predicts log10 S in mol/L. It is converted to mole fraction with the
same per-litre conversion as the data pipeline, so it needs the solvent's
density; solvents without one in the curated table get NaN.

FastSolv was trained on BigSolDB 1.0, so on any split of BigSolDB data its
"test" set overlaps its training set. Its scores here are an optimistic
reference, not a like-for-like comparison.

Install separately (pulls in torch and downloads weights from Zenodo):
    pip install fastsolv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..standardize import standardize_solute
from ..units import to_mole_fraction


class FastSolvBaseline:
    name = "fastsolv"

    def fit(self, train: pd.DataFrame, solvents: pd.DataFrame) -> "FastSolvBaseline":
        self.solvents = solvents  # FastSolv is pre-trained; only the density lookup is needed
        return self

    def predict(self, points: pd.DataFrame) -> pd.DataFrame:
        try:
            from fastsolv import fastsolv
        except ImportError as e:
            raise ImportError("FastSolv is not installed: pip install fastsolv") from e
        query = pd.DataFrame({
            "solute_smiles": points["solute_smiles"].to_numpy(),
            "solvent_smiles": points["solvent_smiles"].to_numpy(),
            "temperature": points["T_K"].to_numpy(),
        })
        log_s = fastsolv(query)["predicted_logS"].to_numpy()
        solv = points[["solvent_key"]].merge(
            self.solvents[["solvent_key", "mw", "density_g_mL"]], on="solvent_key", how="left"
        )
        mw_u = points["solute_smiles"].map(lambda s: standardize_solute(s).mw)
        x, _ = to_mole_fraction(
            pd.Series(10.0 ** log_s), pd.Series(["mol/L"] * len(points)),
            mw_u.reset_index(drop=True), solv["mw"], solv["density_g_mL"],
        )
        return pd.DataFrame({"ln_x_pred": np.log(x.to_numpy())})

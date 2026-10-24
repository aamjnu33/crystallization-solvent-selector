"""Main model: predict van't Hoff parameters per solute-solvent pair.

Each pair's curve is ln x(T) = ln x_ref + b (1/T - 1/T_ref), i.e. ln x = a + b/T
with a = ln x_ref - b/T_ref. Predicting (ln x_ref, b) instead of (a, b)
removes most of the a-b correlation, so each target is easier to learn.

Training pairs come from van't Hoff fits on the training points only: pairs
with at least 3 temperatures and no wrong-sign or poor-fit flag.

Uncertainty:
  ensemble      n_models LightGBM pairs, each on a bootstrap sample of
                solutes; sigma = spread of ln x(T) across members
  conformal     a fraction of training solutes is held back for calibration;
                intervals are mean +- q (sigma + eps), with q the conformal
                quantile of |error| / (sigma + eps), so coverage holds for
                new solutes like those
  domain        max Tanimoto similarity of the solute to training solutes;
                below the training set's own 5th-percentile nearest-neighbour
                similarity the result is flagged out of domain

Averaging members keeps the mean curve exactly van't Hoff, so curves stay
smooth and the yield calculation stays stable.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..features import max_similarity, pair_features
from ..vanthoff import PAIR_KEYS, fit_pairs

T_REF = 298.15
EXCLUDE_FLAGS = ("wrong_sign_slope", "poor_fit")

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.8,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


def training_pairs(points: pd.DataFrame, min_temps: int = 3) -> pd.DataFrame:
    pairs = fit_pairs(points, min_temps=min_temps)
    keep = pairs["n_temps"] >= min_temps
    for f in EXCLUDE_FLAGS:
        keep &= ~pairs["qc_flags"].str.contains(f)
    pairs = pairs[keep].copy()
    smiles = points.groupby(PAIR_KEYS)[["solute_smiles", "solvent_smiles"]].first().reset_index()
    pairs = pairs.merge(smiles, on=PAIR_KEYS)
    pairs["ln_x_ref"] = pairs["a"] + pairs["b"] / T_REF
    pairs["b_kK"] = pairs["b"] / 1000.0  # slope in kK keeps both targets on similar scales
    return pairs.reset_index(drop=True)


class VantHoffEnsemble:
    name = "vanthoff"

    def __init__(
        self,
        n_models: int = 8,
        cal_frac: float = 0.15,
        alpha: float = 0.1,
        eps: float = 0.1,
        ad_percentile: float = 5.0,
        params: dict | None = None,
        num_boost_round: int = 400,
        seed: int = 0,
    ):
        self.n_models, self.cal_frac, self.alpha, self.eps = n_models, cal_frac, alpha, eps
        self.ad_percentile = ad_percentile
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round, self.seed = num_boost_round, seed

    # -- training ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, solvents: pd.DataFrame | None = None) -> "VantHoffEnsemble":
        rng = np.random.default_rng(self.seed)
        solutes = train["solute_key"].unique()
        n_cal = int(round(self.cal_frac * len(solutes))) if len(solutes) >= 10 else 0
        cal_solutes = set(rng.choice(solutes, n_cal, replace=False)) if n_cal else set()
        is_cal = train["solute_key"].isin(cal_solutes)
        fit_points, cal_points = train[~is_cal], train[is_cal]

        pairs = training_pairs(fit_points)
        if pairs.empty:
            raise ValueError("no pair has 3+ temperatures without QC flags; cannot train")
        X = pair_features(pairs["solute_smiles"], pairs["solvent_smiles"])
        fit_solutes = pairs["solute_key"].unique()
        self.members = []
        for m in range(self.n_models):
            # bootstrap whole solutes so members disagree most on unfamiliar chemistry
            draw = pd.Series(rng.choice(fit_solutes, len(fit_solutes), replace=True)).value_counts()
            w = pairs["solute_key"].map(draw).fillna(0).to_numpy()
            params = {**self.params, "seed": self.seed + m}
            heads = {}
            for target in ("ln_x_ref", "b_kK"):
                heads[target] = lgb.train(
                    params, lgb.Dataset(X, pairs[target].to_numpy(), weight=w), self.num_boost_round
                )
            self.members.append(heads)

        self.train_solute_smiles = sorted(set(fit_points["solute_smiles"]))
        nn = max_similarity(self.train_solute_smiles, self.train_solute_smiles, exclude_self=True)
        self.ad_cutoff = float(np.percentile(nn, self.ad_percentile)) if len(nn) > 1 else 0.0

        self.q = np.nan
        if len(cal_points):
            mu, sigma = self._ln_x(cal_points)
            scores = np.abs(cal_points["ln_x"].to_numpy() - mu) / (sigma + self.eps)
            n = len(scores)
            level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
            self.q = float(np.quantile(scores, level, method="higher"))
        self.n_train_pairs, self.n_cal_points = len(pairs), len(cal_points)
        return self

    # -- prediction -------------------------------------------------------
    def member_params(self, solute_smiles: pd.Series, solvent_smiles: pd.Series) -> np.ndarray:
        """Array (n_models, n, 2) of (ln x_ref, b in K) per member."""
        X = pair_features(solute_smiles.reset_index(drop=True), solvent_smiles.reset_index(drop=True))
        return np.stack([
            np.column_stack([h["ln_x_ref"].predict(X), h["b_kK"].predict(X) * 1000.0])
            for h in self.members
        ])

    def _ln_x(self, points: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.member_params(points["solute_smiles"], points["solvent_smiles"])
        dT = 1.0 / points["T_K"].to_numpy() - 1.0 / T_REF
        ln_x = p[:, :, 0] + p[:, :, 1] * dT  # (n_models, n)
        return ln_x.mean(axis=0), ln_x.std(axis=0)

    def predict(self, points: pd.DataFrame) -> pd.DataFrame:
        mu, sigma = self._ln_x(points)
        half = self.q * (sigma + self.eps)
        sim = self.domain_similarity(points["solute_smiles"])
        return pd.DataFrame({
            "ln_x_pred": mu, "sigma_ln": sigma,
            "ln_x_lo": mu - half, "ln_x_hi": mu + half,
            "ad_similarity": sim, "in_domain": sim >= self.ad_cutoff,
        })

    def predict_parameters(self, solute_smiles: str, solvent_smiles: list[str]) -> pd.DataFrame:
        """Mean a, b (ln x = a + b/T) and spread for one solute in many solvents."""
        n = len(solvent_smiles)
        p = self.member_params(pd.Series([solute_smiles] * n), pd.Series(solvent_smiles))
        ln_x_ref, b = p[:, :, 0].mean(axis=0), p[:, :, 1].mean(axis=0)
        return pd.DataFrame({
            "solvent_smiles": solvent_smiles,
            "a": ln_x_ref - b / T_REF, "b": b,
            "ln_x_ref_sd": p[:, :, 0].std(axis=0), "b_sd": p[:, :, 1].std(axis=0),
        })

    def domain_similarity(self, solute_smiles: pd.Series) -> np.ndarray:
        uniq = pd.unique(solute_smiles)
        sims = dict(zip(uniq, max_similarity(list(uniq), self.train_solute_smiles)))
        return solute_smiles.map(sims).to_numpy(float)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "VantHoffEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)

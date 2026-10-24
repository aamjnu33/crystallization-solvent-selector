"""ML baseline: LightGBM on solute descriptors + solvent descriptors + 1000/T.

Predicts ln x point by point, so nothing forces its curves to be smooth or
monotonic in T; that's what the main model adds.
"""
from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from ..features import point_features

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": 0,
}


class LGBMBaseline:
    name = "lgbm"

    def __init__(self, params: dict | None = None, num_boost_round: int = 600):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round

    def fit(self, train: pd.DataFrame, solvents: pd.DataFrame | None = None) -> "LGBMBaseline":
        X = point_features(train)
        self.booster = lgb.train(
            self.params, lgb.Dataset(X, train["ln_x"].to_numpy()), self.num_boost_round
        )
        return self

    def predict(self, points: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ln_x_pred": self.booster.predict(point_features(points))})

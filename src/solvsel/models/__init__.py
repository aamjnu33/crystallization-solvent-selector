"""Models share one interface:

    model.fit(train_points, solvents) -> model
    model.predict(points) -> DataFrame with ln_x_pred (+ ln_x_lo, ln_x_hi, ...)
"""
from .fastsolv import FastSolvBaseline
from .lgbm import LGBMBaseline
from .physics import PhysicsBaseline
from .vanthoff_model import VantHoffEnsemble

MODELS = {
    "physics": PhysicsBaseline,
    "lgbm": LGBMBaseline,
    "fastsolv": FastSolvBaseline,
    "vanthoff": VantHoffEnsemble,
}

__all__ = ["MODELS", "FastSolvBaseline", "LGBMBaseline", "PhysicsBaseline", "VantHoffEnsemble"]

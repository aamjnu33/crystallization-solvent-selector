"""Duplicate merging and per-pair van't Hoff checks.

    ln x = a + b / T

For an endothermic dissolution (almost every organic solid) solubility rises
with temperature, so b < 0. A positive b, a poor linear fit, or a single
point far off the line are flagged for review, not dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

R_GAS = 8.314462618  # J/(mol K)

PAIR_KEYS = ["solute_key", "solvent_key"]


def merge_duplicates(
    df: pd.DataFrame, t_tol: float = 0.1, max_spread_ln: float = np.log(2.0)
) -> pd.DataFrame:
    """Collapse repeated solute-solvent-T points to their median ln x.

    Within a pair, temperatures closer than `t_tol` K to the previous one are
    the same point (298.15 and 298.16 K merge); T is their mean.
    `ln_x_spread` is max - min over the merged points; above `max_spread_ln`
    (default: a factor of 2 in x) the point gets `disagreement=True`.
    """
    d = df.sort_values(PAIR_KEYS + ["T_K"]).reset_index(drop=True)
    new_pair = (d[PAIR_KEYS] != d[PAIR_KEYS].shift()).any(axis=1)
    d["t_group"] = (new_pair | (d["T_K"].diff() > t_tol)).cumsum()
    agg = d.groupby(PAIR_KEYS + ["t_group"], as_index=False, sort=False).agg(
        T_K=("T_K", "mean"),
        ln_x=("ln_x", "median"),
        ln_x_spread=("ln_x", lambda s: s.max() - s.min()),
        n_points=("ln_x", "size"),
        n_sources=("source", "nunique"),
        approx_conversion=("approx_conversion", "any"),
    ).drop(columns="t_group")
    agg["disagreement"] = agg["ln_x_spread"] > max_spread_ln
    return agg


def fit_pairs(
    points: pd.DataFrame,
    min_temps: int = 3,
    min_r2: float = 0.9,
    max_resid_ln: float = 0.5,
) -> pd.DataFrame:
    """Fit ln x vs 1/T for every solute-solvent pair."""
    rows = []
    for (solute, solvent), g in points.groupby(PAIR_KEYS, sort=False):
        n = len(g)
        row = {
            "solute_key": solute, "solvent_key": solvent, "n_temps": n,
            "T_min": g["T_K"].min(), "T_max": g["T_K"].max(),
            "a": np.nan, "b": np.nan, "r2": np.nan, "rmse_ln": np.nan,
            "max_abs_resid_ln": np.nan, "dH_app_kJ_mol": np.nan,
        }
        flags = []
        if n < min_temps:
            flags.append("too_few_temperatures")
        else:
            inv_t = 1.0 / g["T_K"].to_numpy()
            y = g["ln_x"].to_numpy()
            b, a = np.polyfit(inv_t, y, 1)
            resid = y - (a + b * inv_t)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - np.sum(resid**2) / ss_tot if ss_tot > 0 else np.nan
            row.update(
                a=a, b=b, r2=r2,
                rmse_ln=float(np.sqrt(np.mean(resid**2))),
                max_abs_resid_ln=float(np.max(np.abs(resid))),
                dH_app_kJ_mol=-b * R_GAS / 1000.0,
            )
            if b > 0:
                flags.append("wrong_sign_slope")
            # R2 is meaningless with 3 points on a near-flat line; need 4+
            if n >= 4 and not (r2 >= min_r2):
                flags.append("poor_fit")
            if row["max_abs_resid_ln"] > max_resid_ln:
                flags.append("outlier_point")
        row["qc_flags"] = ";".join(flags)
        rows.append(row)
    return pd.DataFrame(rows)

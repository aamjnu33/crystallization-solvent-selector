"""Benchmark metrics and the plain-CSV experiment log.

Errors are reported in log10 x so they read on the same scale as the logS
numbers in the solubility literature.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

LN10 = np.log(10.0)


def solvent_ranking_spearman(df: pd.DataFrame, t_bin: float = 5.0, min_solvents: int = 3) -> float:
    """Mean Spearman correlation of predicted vs observed solvent order.

    Computed per solute at each temperature (binned to `t_bin` K) with at
    least `min_solvents` solvents, then averaged.
    """
    d = df.assign(T_bin=(df["T_K"] / t_bin).round())
    rhos = []
    for _, g in d.groupby(["solute_key", "T_bin"]):
        constant = g["ln_x"].nunique() < 2 or g["ln_x_pred"].nunique() < 2
        if g["solvent_key"].nunique() >= min_solvents and not constant:
            rho = spearmanr(g["ln_x"], g["ln_x_pred"]).statistic
            if np.isfinite(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def evaluate(test: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """`pred` is aligned with `test`; NaN predictions count against coverage."""
    d = test.reset_index(drop=True).join(pred.reset_index(drop=True))
    ok = d["ln_x_pred"].notna()
    out = {"n_test": len(d), "coverage": float(ok.mean()) if len(d) else np.nan}
    d = d[ok]
    err = (d["ln_x_pred"] - d["ln_x"]) / LN10
    out["rmse_log10x"] = float(np.sqrt(np.mean(err**2))) if len(d) else np.nan
    out["mae_log10x"] = float(np.mean(np.abs(err))) if len(d) else np.nan
    out["spearman_solvent_rank"] = solvent_ranking_spearman(d) if len(d) else np.nan
    if {"ln_x_lo", "ln_x_hi"} <= set(d.columns):
        d = d[d["ln_x_lo"].notna() & d["ln_x_hi"].notna()]  # no intervals without calibration data
        if len(d):
            inside = (d["ln_x"] >= d["ln_x_lo"]) & (d["ln_x"] <= d["ln_x_hi"])
            out["interval_coverage"] = float(inside.mean())
            out["interval_width_log10x"] = float(((d["ln_x_hi"] - d["ln_x_lo"]) / LN10).mean())
    return out


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
    except Exception:
        return "unknown"


def data_fingerprint(data_dir: Path) -> str:
    """Short hash of the processed-data manifest, so results name their data."""
    m = Path(data_dir) / "manifest.json"
    if not m.exists():
        return "no-manifest"
    outputs = json.loads(m.read_text()).get("outputs", {})
    return hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest()[:12]


def log_experiment(path: Path, row: dict) -> None:
    """Append one row to a CSV log, widening the header if new keys appear."""
    path = Path(path)
    row = {"timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **row}
    if path.exists():
        old = pd.read_csv(path)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)

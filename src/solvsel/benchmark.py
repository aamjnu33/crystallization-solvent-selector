"""Benchmark every model on every split and log the results.

    python -m solvsel.benchmark --data data/processed
    python -m solvsel.benchmark --models lgbm,vanthoff --splits scaffold --max-solutes 300

Writes one row per model x fold to results/experiments.csv (with the git
commit and a hash of the processed data) and a summary table to
results/benchmark.md. FastSolv runs only if it is installed.
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import data_fingerprint, evaluate, git_commit, log_experiment
from .models import MODELS
from .models.physics import load_melting_props
from .splits import SPLITS

SUMMARY_COLS = ["rmse_log10x", "mae_log10x", "spearman_solvent_rank", "coverage",
                "interval_coverage", "interval_width_log10x"]


def load_processed(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    points = pd.read_csv(data_dir / "solubility_points.csv.gz")
    holdout = pd.read_csv(data_dir / "holdout_points.csv.gz")
    solvents = pd.read_csv(data_dir / "solvents.csv", dtype={"ich_q3c_class": str})
    return points, holdout, solvents


def subsample_solutes(points: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    if not n or points["solute_key"].nunique() <= n:
        return points
    keep = np.random.default_rng(seed).choice(points["solute_key"].unique(), n, replace=False)
    return points[points["solute_key"].isin(keep)].reset_index(drop=True)


def make_model(name: str, melting_props: pd.DataFrame | None, fast: bool = False):
    if name == "physics":
        return MODELS[name](melting_props)
    if fast and name == "lgbm":
        return MODELS[name](num_boost_round=100)
    if fast and name == "vanthoff":
        return MODELS[name](n_models=3, num_boost_round=100)
    return MODELS[name]()


def run(
    points: pd.DataFrame,
    solvents: pd.DataFrame,
    models: list[str],
    splits: list[str],
    log_path: Path | None = None,
    melting_props: pd.DataFrame | None = None,
    holdout: pd.DataFrame | None = None,
    context: dict | None = None,
    fast: bool = False,
) -> pd.DataFrame:
    rows = []
    jobs = [(s, f) for s in splits for f in SPLITS[s](points)]
    if holdout is not None and len(holdout):
        jobs.append(("case_studies", None))
    for split, fold in jobs:
        train = points if fold is None else points.iloc[fold.train].reset_index(drop=True)
        test = holdout if fold is None else points.iloc[fold.test].reset_index(drop=True)
        if test.empty or train.empty:
            continue
        for name in models:
            t0 = time.time()
            model = make_model(name, melting_props, fast).fit(train, solvents)
            res = evaluate(test, model.predict(test))
            row = {
                **(context or {}), "model": name, "split": split,
                "fold": "all" if fold is None else fold.name,
                "n_train": len(train), **res, "seconds": round(time.time() - t0, 1),
            }
            rows.append(row)
            if log_path:
                log_experiment(log_path, row)
            print(f"{split:12s} {row['fold']:24s} {name:9s} rmse={res['rmse_log10x']:.3f} "
                  f"rho={res['spearman_solvent_rank']:.2f} cov={res['coverage']:.2f}")
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLS if c in results]
    return results.groupby(["split", "model"], sort=False)[cols].mean().reset_index()


def to_markdown(summary: pd.DataFrame, context: dict) -> str:
    head = ["Split", "Model", "RMSE log10 x", "MAE log10 x", "Solvent-rank Spearman",
            "Coverage", "90% interval coverage", "Interval width log10 x"]
    fmt = lambda v: "" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    lines = [
        "# Benchmark", "",
        f"Commit `{context['commit']}`, data `{context['data']}`. Means over folds; "
        "`case_studies` trains on all development data and tests on the held-out APIs.",
        "Coverage is the share of test points the model could predict at all.", "",
        "| " + " | ".join(head) + " |", "|" + " --- |" * 2 + " ---: |" * (len(head) - 2),
    ]
    for _, r in summary.iterrows():
        vals = [r.get(c, np.nan) for c in SUMMARY_COLS]
        lines.append(f"| {r.split} | {r.model} | " + " | ".join(fmt(v) for v in vals) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--models", default="physics,lgbm,vanthoff")
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--melting-props", type=Path, help="CSV: solute_smiles, Tm_K, dHfus_kJ_mol[, solute_dD/dP/dH]")
    p.add_argument("--max-solutes", type=int, help="subsample solutes for a quick run")
    p.add_argument("--fast", action="store_true", help="fewer trees and ensemble members")
    p.add_argument("--results", type=Path, default=Path("results"))
    a = p.parse_args(argv)

    points, holdout, solvents = load_processed(a.data)
    points = subsample_solutes(points, a.max_solutes)
    props = load_melting_props(a.melting_props) if a.melting_props else None
    models = a.models.split(",")
    if "fastsolv" in models:
        if importlib.util.find_spec("fastsolv") is None:
            print("fastsolv not installed; skipping it (pip install fastsolv)")
            models.remove("fastsolv")
    context = {"commit": git_commit(), "data": data_fingerprint(a.data),
               "max_solutes": a.max_solutes or "all", "fast": a.fast}
    res = run(points, solvents, models, a.splits.split(","), a.results / "experiments.csv",
              props, holdout, context, a.fast)
    md = to_markdown(summarize(res), context)
    (a.results / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

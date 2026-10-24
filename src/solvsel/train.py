"""Train the main model on all development data and save it.

    python -m solvsel.train --data data/processed --out models/vanthoff.pkl
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import load_processed
from .models import VantHoffEnsemble


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("models/vanthoff.pkl"))
    p.add_argument("--n-models", type=int, default=8)
    a = p.parse_args(argv)
    points, _, solvents = load_processed(a.data)
    model = VantHoffEnsemble(n_models=a.n_models).fit(points, solvents)
    model.save(a.out)
    print(f"trained on {model.n_train_pairs:,} pairs, calibrated on {model.n_cal_points:,} points; "
          f"conformal q={model.q:.2f}, domain cutoff={model.ad_cutoff:.2f}; saved {a.out}")


if __name__ == "__main__":
    main()

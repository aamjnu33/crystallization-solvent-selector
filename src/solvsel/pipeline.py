"""Raw solubility table -> clean, versioned dataset + data card.

    python -m solvsel.pipeline data/raw/<record>/<file>.csv --out data/processed

Steps, each logging the rows it removes to removed_rows.csv.gz:
  1. standardize solutes (strip salts, neutralize, canonical SMILES)
  2. canonicalize solvents; mixed solvents are set aside for the antisolvent work
  3. convert to mole fraction, then ln x
  4. drop temperatures outside 200-500 K
  5. lock away the case-study APIs (holdout_points.csv.gz)
  6. merge duplicate solute-solvent-T points, flag disagreements
  7. van't Hoff fit per pair, flag wrong-sign slopes, poor fits, outliers
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .fetch import sha256
from .holdout import CASE_STUDY_APIS, split_holdout
from .io import load_raw
from .solvents import build_solvent_table, load_curated
from .standardize import canonical_solvent, standardize_solute
from .units import APPROXIMATE_UNITS, normalize_unit, to_mole_fraction
from .vanthoff import PAIR_KEYS, fit_pairs, merge_duplicates

T_MIN_K, T_MAX_K = 200.0, 500.0


class Log:
    """Collects removed rows and per-step counts for the data card."""

    def __init__(self, n_raw: int):
        self.steps = [("raw rows", n_raw, 0)]
        self.removed: list[pd.DataFrame] = []

    def drop(self, df: pd.DataFrame, mask: pd.Series, step: str, reason) -> pd.DataFrame:
        gone = df[mask]
        if len(gone):
            self.removed.append(
                pd.DataFrame({
                    "raw_row": gone["raw_row"], "step": step,
                    "reason": reason[mask] if isinstance(reason, pd.Series) else reason,
                    "solute_smiles": gone["solute_smiles"], "solvent_smiles": gone["solvent_smiles"],
                })
            )
        kept = df[~mask]
        self.steps.append((step, len(kept), int(mask.sum())))
        return kept

    def removed_frame(self) -> pd.DataFrame:
        cols = ["raw_row", "step", "reason", "solute_smiles", "solvent_smiles"]
        return pd.concat(self.removed, ignore_index=True) if self.removed else pd.DataFrame(columns=cols)


def clean(raw: pd.DataFrame, curated: pd.DataFrame | None = None) -> dict:
    curated = load_curated() if curated is None else curated
    log = Log(len(raw))
    df = raw.copy()

    sol = df["solute_smiles"].map(standardize_solute)
    df["solute_key"] = [s.inchikey for s in sol]
    df["solute_inchikey"] = df["solute_key"]
    df["solute_smiles_std"] = [s.smiles for s in sol]
    df["mw_solute"] = [s.mw for s in sol]
    err = pd.Series([s.error for s in sol], index=df.index)
    df = log.drop(df, err.notna(), "standardize solute", err)

    sv = df["solvent_smiles"].map(canonical_solvent)
    df["solvent_key"] = [s.inchikey for s in sv]
    df["solvent_smiles_std"] = [s.smiles for s in sv]
    df["mw_solvent"] = [s.mw for s in sv]
    err = pd.Series([s.error for s in sv], index=df.index)
    mixed = df[err == "mixed_solvent"].copy()
    df = log.drop(df, err.notna(), "canonicalize solvent", err)

    df = df.merge(curated[["solvent_key", "density_g_mL"]], on="solvent_key", how="left")
    x, reason = to_mole_fraction(df["value"], df["unit"], df["mw_solute"], df["mw_solvent"], df["density_g_mL"])
    df["x"] = x
    df["approx_conversion"] = df["unit"].map(normalize_unit).isin(APPROXIMATE_UNITS)
    df = log.drop(df, reason.notna(), "convert units", reason)

    bad_t = ~df["T_K"].between(T_MIN_K, T_MAX_K)
    df = log.drop(df, bad_t, "temperature range", "temperature_out_of_range")

    df["ln_x"] = np.log(df["x"])
    df["log10_x"] = np.log10(df["x"])

    dev, held = split_holdout(df)
    log.steps.append(("hold out case-study APIs", len(dev), len(held)))

    names = (
        df.groupby("solute_key")
        .agg(solute_smiles=("solute_smiles_std", "first"), solute_name=("solute_name", "first"))
        .reset_index()
    )

    def merged(part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        pts = merge_duplicates(part)
        pairs = fit_pairs(pts)
        pts = pts.merge(pairs[PAIR_KEYS + ["qc_flags"]], on=PAIR_KEYS)
        return pts.merge(names, on="solute_key", how="left"), pairs.merge(names, on="solute_key", how="left")

    points, pairs = merged(dev)
    held_points, held_pairs = merged(held)
    if len(held):
        case = held.groupby("solute_key")["case_study"].first()
        held_points["case_study"] = held_points["solute_key"].map(case)
        held_pairs["case_study"] = held_pairs["solute_key"].map(case)
    log.steps.append(("merge duplicates (dev)", len(points), len(dev) - len(points)))

    used = (
        df.groupby("solvent_key")
        .agg(
            solvent_smiles=("solvent_smiles_std", "first"),
            solvent_name=("solvent_name", "first"),
            n_points=("solvent_key", "size"),
        )
        .reset_index()
    )
    solvents = build_solvent_table(used, curated)

    return {
        "points": points, "pairs": pairs,
        "holdout_points": held_points, "holdout_pairs": held_pairs,
        "mixed_solvent_rows": mixed, "solvents": solvents,
        "removed": log.removed_frame(), "steps": log.steps,
    }


def _flag_counts(pairs: pd.DataFrame) -> dict[str, int]:
    flags = pairs["qc_flags"].str.split(";").explode()
    return flags[flags != ""].value_counts().to_dict()


def data_card(res: dict, manifest: dict) -> str:
    pts, pairs, solv = res["points"], res["pairs"], res["solvents"]
    removed = res["removed"]
    lines = [
        "# Data card: solubility dataset",
        "",
        f"Built {manifest['built_utc']} by solvsel {manifest['solvsel_version']} from "
        f"`{manifest['raw_file']['name']}` (sha256 `{manifest['raw_file']['sha256'][:12]}…`).",
    ]
    src = manifest.get("source")
    if src:
        lines.append(
            f"Source: {src.get('title')} v{src.get('version')}, DOI {src.get('doi')}, "
            f"license `{src.get('license')}`, published {src.get('publication_date')}."
        )
    else:
        lines.append("Source: **no MANIFEST.json next to the raw file; version and license not recorded.**")
    lines += [
        "",
        "## Result",
        "",
        f"- {len(pts):,} solubility points (unique solute–solvent–T) for development",
        f"- {pts['solute_key'].nunique():,} solutes × {pts['solvent_key'].nunique():,} solvents, "
        f"{len(pairs):,} solute–solvent pairs",
        f"- {pts['disagreement'].sum():,} merged points where sources disagree by more than 2× in x",
        f"- {len(res['holdout_points']):,} points for the case-study APIs, stored separately",
        f"- {len(res['mixed_solvent_rows']):,} mixed-solvent rows set aside",
        "",
        "## Cleaning steps",
        "",
        "| Step | Rows after | Rows removed |",
        "| --- | ---: | ---: |",
    ]
    lines += [f"| {s} | {n:,} | {r:,} |" for s, n, r in res["steps"]]
    lines += ["", "## Rows removed, by reason", "", "| Step | Reason | Rows |", "| --- | --- | ---: |"]
    if len(removed):
        for (step, reason), n in removed.groupby(["step", "reason"]).size().sort_values(ascending=False).items():
            lines.append(f"| {step} | {reason} | {n:,} |")
    lines += [
        "",
        "Every removed row is listed with its raw row number in `removed_rows.csv.gz`.",
        "",
        "## Van't Hoff checks (development pairs)",
        "",
        "ln x = a + b/T fitted per pair. Flagged pairs are kept; `qc_flags` on each point says why.",
        "",
        "| Flag | Pairs |",
        "| --- | ---: |",
    ]
    lines += [f"| {f} | {n:,} |" for f, n in _flag_counts(pairs).items()] or ["| none | 0 |"]
    clean_pairs = (pairs["qc_flags"] == "").sum()
    lines += [
        "",
        f"{clean_pairs:,} of {len(pairs):,} pairs have no flag.",
        "",
        "## Held-out case studies",
        "",
        "Never used for training, tuning or calibration: "
        + ", ".join(sorted(CASE_STUDY_APIS)) + ". Matched by InChIKey skeleton, so salts, "
        "enantiomers and racemates of these APIs are all held out.",
        "",
    ]
    if len(res["holdout_points"]):
        hp = res["holdout_points"].groupby("case_study").agg(
            points=("ln_x", "size"), solvents=("solvent_key", "nunique"))
        lines += ["| API | Points | Solvents |", "| --- | ---: | ---: |"]
        lines += [f"| {k} | {r.points:,} | {r.solvents:,} |" for k, r in hp.iterrows()]
    else:
        lines.append("None of the case-study APIs appear in this raw file.")
    covered = solv.loc[solv["curated"], "n_points"].sum()
    lines += [
        "",
        "## Solvent table",
        "",
        f"{len(solv):,} solvents in the data; {int(solv['curated'].sum())} have curated ICH class, "
        f"boiling point and density, covering {covered / max(solv['n_points'].sum(), 1):.0%} of points.",
        "Curated values (`data/solvents_curated.csv`) were entered by hand and must be checked against "
        "ICH Q3C(R8) and CHEM21 (Prat et al. 2016) before release.",
        "",
        "## Known limits",
        "",
        "- Per-litre units are converted using pure-solvent density (`approx_conversion=True`).",
        "- Solid form is not recorded; points may mix polymorphs, hydrates and solvates.",
        "- Solutes keep stereochemistry in their key; enantiomer and racemate are separate solutes.",
        "",
    ]
    return "\n".join(lines)


def run(raw_path: Path, out_dir: Path, temp_unit: str = "K", overrides: dict | None = None) -> dict:
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    raw = load_raw(raw_path, overrides, temp_unit)
    res = clean(raw)

    src_manifest = raw_path.parent / "MANIFEST.json"
    manifest = {
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "solvsel_version": __version__,
        "raw_file": {"name": raw_path.name, "sha256": sha256(raw_path)},
        "source": json.loads(src_manifest.read_text()) if src_manifest.exists() else None,
        "temp_unit_in": temp_unit,
        "case_study_apis": sorted(CASE_STUDY_APIS),
    }
    outputs = {
        "solubility_points.csv.gz": res["points"],
        "vanthoff_pairs.csv.gz": res["pairs"],
        "holdout_points.csv.gz": res["holdout_points"],
        "holdout_pairs.csv.gz": res["holdout_pairs"],
        "mixed_solvent_rows.csv.gz": res["mixed_solvent_rows"],
        "removed_rows.csv.gz": res["removed"],
        "solvents.csv": res["solvents"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest["outputs"] = {}
    for name, frame in outputs.items():
        path = out_dir / name
        # mtime=0 keeps gzip output byte-identical across runs, so hashes are stable
        comp = {"method": "gzip", "mtime": 0} if name.endswith(".gz") else None
        frame.to_csv(path, index=False, compression=comp)
        manifest["outputs"][name] = {"rows": len(frame), "sha256": sha256(path)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "DATA_CARD.md").write_text(data_card(res, manifest))
    return res


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, help="raw CSV/TSV/XLSX")
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--temp-unit", choices=["K", "C"], default="K")
    p.add_argument("--col", action="append", default=[], metavar="FIELD=COLUMN",
                   help="override a column mapping, e.g. --col T=Temp_K")
    a = p.parse_args(argv)
    overrides = dict(c.split("=", 1) for c in a.col)
    res = run(a.raw, a.out, a.temp_unit, overrides)
    for step, n, r in res["steps"]:
        print(f"{step:32s} {n:>9,}  (-{r:,})")
    print(f"wrote {a.out}/DATA_CARD.md")


if __name__ == "__main__":
    main()

"""Load a raw solubility table and map its columns to the pipeline schema.

BigSolDB releases have renamed columns between versions, so each schema
field lists the names it accepts. Pass `overrides` (field -> column) when a
file uses something else. A missing required field fails loudly with the
columns that were found.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# field -> accepted column names, first match wins (case-insensitive)
COLUMN_CANDIDATES = {
    "solute_smiles": ["SMILES_Solute", "SMILES", "solute_smiles"],
    "solute_name": ["Compound_Name", "Solute", "solute_name"],
    "solvent_smiles": ["SMILES_Solvent", "Solvent_SMILES", "solvent_smiles"],
    "solvent_name": ["Solvent", "solvent_name"],
    "T": ["Temperature_K", "T,K", "T_K", "Temperature"],
    "x": ["Solubility(mole_fraction)", "Solubility (mole fraction)", "mole_fraction", "x"],
    "value": ["Solubility", "value"],
    "unit": ["Unit", "Units", "unit"],
    "source": ["Source", "DOI", "Reference", "source"],
}
REQUIRED = ["solute_smiles", "solvent_smiles", "T"]


def resolve_columns(columns, overrides: dict[str, str] | None = None) -> dict[str, str]:
    lower = {c.lower(): c for c in columns}
    mapping = {}
    for field, names in COLUMN_CANDIDATES.items():
        for n in names:
            if n.lower() in lower:
                mapping[field] = lower[n.lower()]
                break
    mapping.update(overrides or {})
    missing = [f for f in REQUIRED if f not in mapping]
    if "x" not in mapping and not {"value", "unit"} <= mapping.keys():
        missing.append("x (or value + unit)")
    if missing:
        raise KeyError(f"cannot find columns for {missing}; file has {list(columns)}")
    return mapping


def load_raw(
    path: Path, overrides: dict[str, str] | None = None, temp_unit: str = "K"
) -> pd.DataFrame:
    """Read CSV/TSV/XLSX and return a frame with the schema field names."""
    if temp_unit not in {"K", "C"}:
        raise ValueError("temp_unit must be 'K' or 'C'")
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    mapping = resolve_columns(raw.columns, overrides)
    df = pd.DataFrame({field: raw[col] for field, col in mapping.items()})
    df["raw_row"] = raw.index
    if "x" in df:
        df["value"], df["unit"] = df["x"], "mole_fraction"
    if "source" not in df:
        df["source"] = "unknown"
    for opt in ("solute_name", "solvent_name"):
        if opt not in df:
            df[opt] = None
    t = pd.to_numeric(df["T"], errors="coerce")
    df["T_K"] = t + 273.15 if temp_unit == "C" else t
    return df.drop(columns=["T", "x"], errors="ignore")

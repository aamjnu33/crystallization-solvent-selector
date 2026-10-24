"""Convert reported solubilities to mole fraction x (then ln x / log10 x).

Supported units (case-insensitive aliases in UNIT_ALIASES):
    mole_fraction  x
    mass_fraction  w (g solute per g solution)
    molality       mol solute per kg solvent
    g_per_100g     g solute per 100 g solvent
    mol_per_L      mol solute per L solution   (needs solvent density; approximate)
    g_per_L        g solute per L solution     (needs solvent density; approximate)

Per-litre units assume the solution density equals the pure-solvent density,
which is fine for dilute solutions and increasingly wrong as x grows. Rows that
use them carry `approx_conversion=True`.

Anything else, or a conversion that comes out outside (0, 1], returns NaN with
a reason so the row can be logged and dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNIT_ALIASES = {
    "mole_fraction": "mole_fraction", "mole fraction": "mole_fraction", "x": "mole_fraction",
    "solubility(mole_fraction)": "mole_fraction",
    "mass_fraction": "mass_fraction", "mass fraction": "mass_fraction", "w": "mass_fraction",
    "mol/kg": "molality", "molality": "molality",
    "g/100g": "g_per_100g", "g/100 g": "g_per_100g", "g/100g solvent": "g_per_100g",
    "mol/l": "mol_per_L", "m": "mol_per_L", "mol/dm3": "mol_per_L",
    "g/l": "g_per_L", "mg/ml": "g_per_L", "kg/m3": "g_per_L",
}
APPROXIMATE_UNITS = {"mol_per_L", "g_per_L"}


def normalize_unit(unit: object) -> str | None:
    if not isinstance(unit, str):
        return None
    return UNIT_ALIASES.get(unit.strip().lower())


def to_mole_fraction(
    value: pd.Series,
    unit: pd.Series,
    mw_solute: pd.Series,
    mw_solvent: pd.Series,
    density_solvent: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return (x, reason). reason is None where x is valid."""
    v = pd.to_numeric(value, errors="coerce").to_numpy(float)
    mu = pd.to_numeric(mw_solute, errors="coerce").to_numpy(float)
    ms = pd.to_numeric(mw_solvent, errors="coerce").to_numpy(float)
    rho = (
        pd.to_numeric(density_solvent, errors="coerce").to_numpy(float)
        if density_solvent is not None
        else np.full(len(v), np.nan)
    )
    u = unit.map(normalize_unit).to_numpy(object)
    x = np.full(len(v), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        m = u == "mole_fraction"
        x[m] = v[m]

        m = u == "mass_fraction"
        n_u, n_s = v[m] / mu[m], (1 - v[m]) / ms[m]
        x[m] = n_u / (n_u + n_s)

        m = u == "molality"
        x[m] = v[m] / (v[m] + 1000.0 / ms[m])

        m = u == "g_per_100g"
        n_u = v[m] / mu[m]
        x[m] = n_u / (n_u + 100.0 / ms[m])

        # per litre of solution: solvent mass ~ 1000*rho - solute mass
        m = u == "mol_per_L"
        solvent_g = 1000.0 * rho[m] - v[m] * mu[m]
        x[m] = np.where(solvent_g > 0, v[m] / (v[m] + solvent_g / ms[m]), np.nan)

        m = u == "g_per_L"
        n_u = v[m] / mu[m]
        solvent_g = 1000.0 * rho[m] - v[m]
        x[m] = np.where(solvent_g > 0, n_u / (n_u + solvent_g / ms[m]), np.nan)

    reason = np.full(len(v), None, dtype=object)
    reason[np.isnan(v)] = "non_numeric_value"
    reason[(u == None) & (reason == None)] = "unknown_unit"  # noqa: E711
    needs_rho = np.isin(u, list(APPROXIMATE_UNITS)) & np.isnan(rho)
    reason[needs_rho & (reason == None)] = "no_solvent_density"  # noqa: E711
    bad = ~((x > 0) & (x <= 1))
    reason[bad & (reason == None)] = "x_out_of_range"  # noqa: E711
    x[reason != None] = np.nan  # noqa: E711
    return pd.Series(x, index=value.index), pd.Series(reason, index=value.index)

import numpy as np
import pandas as pd
import pytest

from solvsel.units import to_mole_fraction

WATER_MW = 18.015


def conv(values, units, mu, ms, rho=None):
    n = len(values)
    s = lambda v: pd.Series(v if isinstance(v, list) else [v] * n)  # noqa: E731
    return to_mole_fraction(s(values), s(units), s(mu), s(ms), None if rho is None else s(rho))


def test_mole_fraction_passthrough():
    x, why = conv([0.1, 1.0], "mole_fraction", 100.0, 50.0)
    assert x.tolist() == [0.1, 1.0] and why.isna().all()


def test_mass_fraction_equal_mw_equals_mole_fraction():
    x, _ = conv([0.25], "mass_fraction", 80.0, 80.0)
    assert x[0] == pytest.approx(0.25)


def test_molality_in_water():
    x, _ = conv([1.0], "mol/kg", 151.16, WATER_MW)
    assert x[0] == pytest.approx(1.0 / (1.0 + 1000.0 / WATER_MW))


def test_g_per_100g_solvent():
    # 15.116 g paracetamol (0.1 mol) in 100 g ethanol (2.1707 mol)
    x, _ = conv([15.116], "g/100g", 151.16, 46.069)
    assert x[0] == pytest.approx(0.1 / (0.1 + 100.0 / 46.069))


def test_mol_per_litre_uses_solvent_density():
    x, why = conv([0.5], "mol/L", 122.12, 46.069, rho=0.789)
    solvent_mol = (789.0 - 0.5 * 122.12) / 46.069
    assert why.isna().all()
    assert x[0] == pytest.approx(0.5 / (0.5 + solvent_mol))


def test_mg_per_ml_equals_g_per_l():
    a, _ = conv([20.0], "mg/mL", 122.12, 46.069, rho=0.789)
    b, _ = conv([20.0], "g/L", 122.12, 46.069, rho=0.789)
    assert a[0] == pytest.approx(b[0])


@pytest.mark.parametrize("value,unit,rho,reason", [
    (0.5, "mol/L", None, "no_solvent_density"),
    (0.5, "furlongs", None, "unknown_unit"),
    ("n/a", "mole_fraction", None, "non_numeric_value"),
    (1.5, "mole_fraction", None, "x_out_of_range"),
    (0.0, "mole_fraction", None, "x_out_of_range"),
    (5000.0, "g/L", 0.789, "x_out_of_range"),  # more solute than the litre can hold
])
def test_unresolvable_rows_get_a_reason(value, unit, rho, reason):
    x, why = conv([value], unit, 122.12, 46.069, rho=rho)
    assert np.isnan(x[0]) and why[0] == reason

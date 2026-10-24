import numpy as np
import pandas as pd
import pytest

from solvsel.vanthoff import fit_pairs, merge_duplicates

T = np.array([283.15, 293.15, 303.15, 313.15, 323.15])


def pts(a, b, temps=T, solute="S", solvent="V"):
    return pd.DataFrame({"solute_key": solute, "solvent_key": solvent, "T_K": temps, "ln_x": a + b / temps})


def test_recovers_parameters_and_enthalpy():
    r = fit_pairs(pts(4.0, -2000.0)).iloc[0]
    assert r.a == pytest.approx(4.0) and r.b == pytest.approx(-2000.0)
    assert r.dH_app_kJ_mol == pytest.approx(2000 * 8.314462618 / 1000)
    assert r.qc_flags == ""


def test_wrong_sign_slope_flagged():
    assert "wrong_sign_slope" in fit_pairs(pts(1.0, 800.0)).iloc[0].qc_flags


def test_outlier_and_poor_fit_flagged():
    d = pts(4.0, -2000.0)
    d.loc[2, "ln_x"] += 2.0
    flags = fit_pairs(d).iloc[0].qc_flags
    assert "outlier_point" in flags and "poor_fit" in flags


def test_too_few_temperatures():
    r = fit_pairs(pts(4.0, -2000.0, temps=T[:2])).iloc[0]
    assert r.qc_flags == "too_few_temperatures" and np.isnan(r.b)


def test_merge_duplicates_takes_median_and_flags_spread():
    d = pd.DataFrame({
        "solute_key": "S", "solvent_key": "V",
        "T_K": [298.15, 298.15, 298.16, 310.0],
        "ln_x": [-3.0, -3.2, -1.0, -2.5],
        "source": ["a", "b", "c", "a"],
        "approx_conversion": False,
    })
    m = merge_duplicates(d)
    first = m[m.T_K < 300].iloc[0]
    assert first.n_points == 3 and first.n_sources == 3
    assert first.ln_x == pytest.approx(-3.0)
    assert first.disagreement  # spread of 2.2 in ln x is ~9x
    assert not m[m.T_K == 310.0].iloc[0].disagreement

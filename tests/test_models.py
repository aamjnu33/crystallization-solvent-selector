import numpy as np
import pandas as pd
import pytest

from solvsel.benchmark import run, summarize, to_markdown
from solvsel.metrics import evaluate, solvent_ranking_spearman
from solvsel.models import LGBMBaseline, PhysicsBaseline, VantHoffEnsemble
from solvsel.splits import SPLITS, scaffold
from solvsel.standardize import standardize_solute
from synthetic import SOLUTES, make_points, make_solvents

FAST_LGBM = dict(num_boost_round=150)
FAST_VH = dict(n_models=3, num_boost_round=150)


@pytest.fixture(scope="module")
def data():
    return make_points(), make_solvents()


@pytest.fixture(scope="module")
def scaffold_fold(data):
    points, _ = data
    fold = SPLITS["scaffold"](points)[0]
    return points.iloc[fold.train].reset_index(drop=True), points.iloc[fold.test].reset_index(drop=True)


@pytest.fixture(scope="module")
def vh(data):
    points, solvents = data
    return VantHoffEnsemble(**FAST_VH).fit(points, solvents)


# --- splits -----------------------------------------------------------------
def test_scaffold_folds_never_share_a_scaffold(data):
    points, _ = data
    folds = SPLITS["scaffold"](points)
    assert sorted(np.concatenate([f.test for f in folds]).tolist()) == list(range(len(points)))
    for f in folds:
        tr = set(points.iloc[f.train]["solute_smiles"].map(scaffold))
        te = set(points.iloc[f.test]["solute_smiles"].map(scaffold))
        assert not tr & te


def test_solvent_folds_hold_out_one_solvent(data):
    points, _ = data
    for f in SPLITS["solvent"](points):
        assert points.iloc[f.test]["solvent_key"].nunique() == 1
        assert not set(points.iloc[f.test]["solvent_key"]) & set(points.iloc[f.train]["solvent_key"])


def test_temperature_fold_tests_the_hottest_points(data):
    points, _ = data
    f = SPLITS["temperature"](points)[0]
    assert set(points.iloc[f.test]["T_K"]) == {323.15}
    assert points.iloc[f.train]["T_K"].max() < 323.15


# --- metrics ----------------------------------------------------------------
def test_spearman_and_coverage():
    d = pd.DataFrame({"solute_key": "s", "solvent_key": list("abcd"), "T_K": 298.15,
                      "ln_x": [-1.0, -2.0, -3.0, -4.0]})
    assert solvent_ranking_spearman(d.assign(ln_x_pred=d.ln_x * 2)) == pytest.approx(1.0)
    res = evaluate(d, pd.DataFrame({"ln_x_pred": [-1.0, np.nan, -3.0, -4.0]}))
    assert res["coverage"] == 0.75 and res["rmse_log10x"] == 0.0


# --- baselines --------------------------------------------------------------
def test_lgbm_beats_the_mean(scaffold_fold, data):
    train, test = scaffold_fold
    pred = LGBMBaseline(**FAST_LGBM).fit(train, data[1]).predict(test)
    mean_rmse = evaluate(test, pd.DataFrame({"ln_x_pred": np.full(len(test), train.ln_x.mean())}))["rmse_log10x"]
    assert evaluate(test, pred)["rmse_log10x"] < 0.7 * mean_rmse


def test_physics_ideal_solubility_matches_hand_calculation(data):
    points, solvents = data
    benz = standardize_solute(SOLUTES[0])
    props = pd.DataFrame({"solute_key": [benz.inchikey], "Tm_K": [395.0], "dHfus_kJ_mol": [18.0],
                          "solute_dD": [np.nan], "solute_dP": [np.nan], "solute_dH": [np.nan]})
    pred = PhysicsBaseline(props).fit(points, solvents).predict(points)
    is_benz = (points.solute_key == benz.inchikey).to_numpy()
    T = points.loc[is_benz, "T_K"].to_numpy()
    expected = -(18000 / 8.314462618) * (1 / T - 1 / 395.0)
    np.testing.assert_allclose(pred.ln_x_pred[is_benz], expected)
    assert pred.ln_x_pred[~is_benz].isna().all()


def test_physics_hansen_term_lowers_solubility_in_distant_solvents(data):
    points, solvents = data
    benz = standardize_solute(SOLUTES[0])
    props = pd.DataFrame({"solute_key": [benz.inchikey], "Tm_K": [395.0], "dHfus_kJ_mol": [18.0],
                          "solute_dD": [18.2], "solute_dP": [6.9], "solute_dH": [9.8]})
    pts = points[points.solute_key == benz.inchikey].reset_index(drop=True)
    pred = PhysicsBaseline(props, min_fit_points=10**9).fit(pts, solvents).predict(pts)
    at = pts.assign(p=pred.ln_x_pred)[pts.T_K == 293.15].set_index("solvent_name")["p"]
    assert at["water"] < at["ethanol"]  # water is far from benzoic acid in Hansen space


# --- main model -------------------------------------------------------------
def test_vanthoff_beats_the_mean_with_calibrated_intervals(scaffold_fold, data):
    train, test = scaffold_fold
    pred = VantHoffEnsemble(**FAST_VH).fit(train, data[1]).predict(test)
    res = evaluate(test, pred)
    mean_rmse = evaluate(test, pd.DataFrame({"ln_x_pred": np.full(len(test), train.ln_x.mean())}))["rmse_log10x"]
    assert res["rmse_log10x"] < 0.7 * mean_rmse
    assert 0.5 <= res["interval_coverage"] <= 1.0
    assert (pred.ln_x_hi > pred.ln_x_lo).all()


def test_vanthoff_curves_are_linear_in_inverse_T(vh):
    temps = np.linspace(270, 350, 9)
    q = pd.DataFrame({"solute_smiles": SOLUTES[0], "solvent_smiles": "CCO", "T_K": temps})
    ln_x = vh.predict(q).ln_x_pred.to_numpy()
    slope, icpt = np.polyfit(1 / temps, ln_x, 1)
    np.testing.assert_allclose(ln_x, icpt + slope / temps, atol=1e-9)
    params = vh.predict_parameters(SOLUTES[0], ["CCO"]).iloc[0]
    assert params.a == pytest.approx(icpt) and params.b == pytest.approx(slope)


def test_applicability_domain(vh):
    seen = vh.train_solute_smiles[0]
    perfluoro = "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)F"
    sims = vh.domain_similarity(pd.Series([seen, perfluoro]))
    assert sims[0] == pytest.approx(1.0) and sims[0] >= vh.ad_cutoff
    assert sims[1] < vh.ad_cutoff
    pred = vh.predict(pd.DataFrame({"solute_smiles": [perfluoro], "solvent_smiles": ["CCO"], "T_K": [298.15]}))
    assert not pred.in_domain[0]


def test_save_load_roundtrip(vh, tmp_path, data):
    vh.save(tmp_path / "m.pkl")
    again = VantHoffEnsemble.load(tmp_path / "m.pkl")
    q = data[0].head(20)
    pd.testing.assert_frame_equal(vh.predict(q), again.predict(q))


# --- benchmark --------------------------------------------------------------
def test_benchmark_runs_and_logs(data, tmp_path):
    points, solvents = data
    caffeine = standardize_solute(SOLUTES[2]).smiles
    holdout = points[points.solute_smiles == caffeine]
    dev = points[points.solute_smiles != caffeine].reset_index(drop=True)
    assert len(holdout)
    log = tmp_path / "experiments.csv"
    res = run(dev, solvents, ["lgbm", "vanthoff"], ["temperature"], log,
              holdout=holdout, context={"commit": "test", "data": "synthetic"}, fast=True)
    assert set(res.split) == {"temperature", "case_studies"}
    assert len(pd.read_csv(log)) == 4
    md = to_markdown(summarize(res), {"commit": "test", "data": "synthetic"})
    assert "| temperature | vanthoff |" in md

from solvsel.holdout import split_holdout
from solvsel.standardize import canonical_solvent, connectivity_key, standardize_solute

import pandas as pd


def test_hydrochloride_salt_is_stripped():
    assert standardize_solute("CC(=O)Nc1ccc(O)cc1.Cl").smiles == standardize_solute("CC(=O)Nc1ccc(O)cc1").smiles


def test_sodium_carboxylate_is_neutralized():
    salt = standardize_solute("CC(C)Cc1ccc(cc1)C(C)C(=O)[O-].[Na+]")
    acid = standardize_solute("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    assert salt.ok and salt.inchikey == acid.inchikey


def test_bad_inputs_report_a_reason():
    assert standardize_solute("C1CC").error == "unparseable_smiles"
    assert standardize_solute("").error == "missing_smiles"


def test_stereo_kept_but_skeleton_matches():
    s = standardize_solute("COc1ccc2cc(ccc2c1)[C@H](C)C(=O)O")
    rac = standardize_solute("COc1ccc2cc(ccc2c1)C(C)C(=O)O")
    assert s.inchikey != rac.inchikey
    assert connectivity_key(s.inchikey) == connectivity_key(rac.inchikey)


def test_solvent_mixture_detected():
    assert canonical_solvent("CCO.O").error == "mixed_solvent"
    assert canonical_solvent("OCC").smiles == "CCO"


def test_holdout_catches_enantiomer_and_keeps_others():
    keys = [standardize_solute(s).inchikey for s in
            ["COc1ccc2cc(ccc2c1)[C@H](C)C(=O)O", "O=C(O)c1ccccc1"]]
    dev, held = split_holdout(pd.DataFrame({"solute_inchikey": keys}))
    assert held["case_study"].tolist() == ["naproxen"]
    assert len(dev) == 1

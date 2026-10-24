import json

from solvsel.pipeline import run


def test_end_to_end(raw_csv, tmp_path):
    out = tmp_path / "out"
    res = run(raw_csv, out)

    reasons = res["removed"].groupby("reason").size().to_dict()
    assert reasons == {
        "unparseable_smiles": 1, "mixed_solvent": 1,
        "x_out_of_range": 1, "temperature_out_of_range": 1,
    }
    assert len(res["mixed_solvent_rows"]) == 1

    # paracetamol and (S)-naproxen are held out; nothing of theirs leaks into dev
    assert set(res["holdout_points"]["case_study"]) == {"paracetamol", "naproxen"}
    assert not res["points"]["solute_smiles"].str.contains("ccc2cc").any()

    assert len(res["pairs"]) == 5  # benzoic/EtOH, benzoic/acetone, salicylic, caffeine, vanillin
    wrong = res["pairs"][res["pairs"].b > 0]
    assert wrong["qc_flags"].str.contains("wrong_sign_slope").all() and len(wrong) == 1

    # duplicate at 293.15 K merged; the salt row joined the benzoic acid/acetone curve
    benz = res["points"][res["points"].solute_smiles == "O=C(O)c1ccccc1"]
    assert benz.n_points.max() == 2
    assert benz.groupby("solvent_key").size().sort_values().tolist() == [5, 6]

    for name in ["solubility_points.csv.gz", "vanthoff_pairs.csv.gz", "removed_rows.csv.gz",
                 "solvents.csv", "DATA_CARD.md", "manifest.json"]:
        assert (out / name).exists()
    card = (out / "DATA_CARD.md").read_text()
    assert "no MANIFEST.json" in card and "x_out_of_range" in card


def test_outputs_are_reproducible(raw_csv, tmp_path):
    run(raw_csv, tmp_path / "a")
    run(raw_csv, tmp_path / "b")
    a = json.loads((tmp_path / "a" / "manifest.json").read_text())["outputs"]
    b = json.loads((tmp_path / "b" / "manifest.json").read_text())["outputs"]
    assert a == b


def test_source_manifest_is_carried_through(raw_csv, tmp_path):
    (raw_csv.parent / "MANIFEST.json").write_text(json.dumps(
        {"title": "BigSolDB", "version": "2.0", "doi": "10.5281/zenodo.0", "license": "cc-by-4.0",
         "publication_date": "2025-01-01"}))
    run(raw_csv, tmp_path / "out")
    card = (tmp_path / "out" / "DATA_CARD.md").read_text()
    assert "license `cc-by-4.0`" in card


def test_solvent_table_marks_curated(raw_csv, tmp_path):
    solv = run(raw_csv, tmp_path / "out")["solvents"]
    assert solv["curated"].all()
    water = solv[solv.solvent_smiles == "O"].iloc[0]
    assert water.ich_q3c_class == "none" and water.bp_C == 100.0

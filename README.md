# Crystallization Solvent Selector

Ranks crystallization solvents for an API from its SMILES: van't Hoff
solubility curves with uncertainty, cooling yield, and ICH Q3C class.

**Status: weeks 2–5 modelling code done; not yet run on BigSolDB.** See
[docs/problem_statement.md](docs/problem_statement.md) for the user, decision
and success metric.

## Data pipeline

```bash
pip install -e ".[dev]"

# 1. Find and download a BigSolDB release; records DOI, version, license, SHA-256
python -m solvsel.fetch --search
python -m solvsel.fetch --record-id <zenodo-record-id>

# 2. Clean it
python -m solvsel.pipeline data/raw/<record-id>/<file>.csv --out data/processed
```

The pipeline:

1. standardizes solutes with RDKit (largest organic fragment, neutralized,
   canonical SMILES, InChIKey)
2. canonicalizes solvents and sets mixed solvents aside
3. converts mole fraction, mass fraction, molality, g/100 g and per-litre
   units to mole fraction, then ln x
4. drops temperatures outside 200–500 K
5. locks away the case-study APIs (paracetamol, carbamazepine, ibuprofen,
   naproxen, matched by InChIKey skeleton)
6. merges duplicate solute–solvent–temperature points and flags sources that
   disagree by more than 2×
7. fits ln x = a + b/T per pair and flags wrong-sign slopes, poor fits and
   outlier points

Every removed row is logged with its reason. Outputs in `data/processed/`:

| File | Contents |
| --- | --- |
| `solubility_points.csv.gz` | development points, one per solute–solvent–T |
| `vanthoff_pairs.csv.gz` | a, b, R², apparent ΔH and QC flags per pair |
| `holdout_points.csv.gz`, `holdout_pairs.csv.gz` | case-study APIs only |
| `mixed_solvent_rows.csv.gz` | rows kept for the antisolvent work |
| `removed_rows.csv.gz` | every dropped row, with step and reason |
| `solvents.csv` | solvents seen, with RDKit descriptors and curated properties |
| `manifest.json` | source release, raw-file hash, output hashes |
| `DATA_CARD.md` | sources, cleaning counts, flags, known limits |

Output files are byte-reproducible: the same raw file gives the same hashes.

## Models and benchmark

```bash
python -m solvsel.benchmark --data data/processed            # all models, all splits
python -m solvsel.benchmark --max-solutes 300 --fast         # quick check
python -m solvsel.train --data data/processed --out models/vanthoff.pkl
```

| Model | What it is |
| --- | --- |
| `physics` | ideal solubility from melting point and heat of fusion, with a Hansen-distance activity correction (one fitted scale factor). Needs `--melting-props`; see `data/melting_props_template.csv` |
| `lgbm` | LightGBM on solute + solvent RDKit descriptors, solute Morgan counts and 1000/T |
| `fastsolv` | FastSolv, converted from log S (mol/L) to mole fraction; `pip install -e ".[fastsolv]"` |
| `vanthoff` | main model: predicts ln x at 298.15 K and the slope b per pair, so every curve is ln x = a + b/T |

The main model is an ensemble of 8 LightGBM pairs trained on bootstrap samples of
solutes. Its 90% intervals are split-conformal, calibrated on 15% of training
solutes held back from fitting. Each prediction also reports the solute's
highest Tanimoto similarity to the training solutes and `in_domain=False` below
the training set's own 5th-percentile nearest-neighbour similarity.

Splits: `random` (optimistic), `scaffold` (whole Bemis–Murcko scaffolds held
out), `solvent` (leave one of the 10 most common solvents out) and
`temperature` (the top third of each pair's temperatures held out). A
`case_studies` row trains on all development data and tests on the held-out
APIs.

Every model × fold is appended to `results/experiments.csv` with the git
commit and a hash of the processed data; `results/benchmark.md` has the means.

Caveats:

- FastSolv was trained on BigSolDB 1.0, so its BigSolDB test points overlap
  its training data. Treat its scores as an optimistic reference.
- Conformal coverage is calibrated for new solutes. On the leave-one-solvent-out
  split, intervals come out narrower than they should be.

## Solvent table

`data/solvents_curated.csv` holds ICH Q3C class, boiling point, density and
CHEM21 rank for 40 common solvents, and Hansen parameters for 25 of them. These were entered by hand. Check them
against ICH Q3C(R8) and Prat et al., *Green Chem.* 2016, 18, 288 before
publishing results. CHEM21 is left blank where it hasn't been checked.

## Tests

```bash
pytest
```

The tests use synthetic data: real structures with solubilities generated from
known van't Hoff parameters (plus one broken row per cleaning rule for the
pipeline). They check the code, not model accuracy on real data.

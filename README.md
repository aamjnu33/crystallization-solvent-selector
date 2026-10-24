# Crystallization Solvent Selector

Ranks crystallization solvents for an API from its SMILES: van't Hoff
solubility curves with uncertainty, cooling yield, and ICH Q3C class.

**Status: weeks 1–2, building the dataset.** See
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

## Solvent table

`data/solvents_curated.csv` holds ICH Q3C class, boiling point, density and
CHEM21 rank for 40 common solvents. These were entered by hand. Check them
against ICH Q3C(R8) and Prat et al., *Green Chem.* 2016, 18, 288 before
publishing results. CHEM21 is left blank where it hasn't been checked.

## Tests

```bash
pytest
```

The tests use a synthetic fixture: real structures with solubilities generated
from known van't Hoff parameters, plus one broken row for each cleaning rule.

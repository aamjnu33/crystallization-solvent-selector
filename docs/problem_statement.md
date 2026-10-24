# Problem statement

## User

A process chemist picking crystallization solvents for a new API, early in
development, with a few hundred milligrams to screen with.

## Decision supported

Which 3–5 solvents (or solvent–antisolvent pairs) to put into the first
solubility and crystallization screen. Today that list comes from experience
and a generic solvent-selection guide, and a poor first screen costs material
and a week of lab time.

## What the tool does

Takes the API's SMILES (plus melting point and heat of fusion if known) and
returns, for each candidate solvent:

- predicted solubility vs. temperature, with an uncertainty band
- cooling-crystallization yield between two temperatures
- ICH Q3C class, boiling point and CHEM21 greenness rank
- a warning when the API is unlike anything in the training data

## Success metric

For held-out APIs (paracetamol, carbamazepine, ibuprofen, naproxen), the top 3
suggested solvents include a solvent actually used to crystallize that API in
the literature. Secondary metrics: RMSE in log x and per-solute Spearman
correlation of the solvent ranking, each reported on random, scaffold,
leave-one-solvent-out and temperature-extrapolation splits against FastSolv.

## Out of scope for v1

Polymorph prediction, solvate/hydrate formation, impurity rejection, and
kinetic effects (nucleation, growth, oiling out). Every result says so.

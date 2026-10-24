"""Structure standardization with RDKit.

Solutes: cleanup -> keep the largest organic fragment (strips salts and
counter-ions) -> neutralize -> canonical isomeric SMILES + InChIKey.
Solvents: cleanup + canonical SMILES only; a "." means a mixed solvent.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

_UNCHARGER = rdMolStandardize.Uncharger()
_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


@dataclass(frozen=True)
class StdResult:
    smiles: str | None
    inchikey: str | None
    mw: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@lru_cache(maxsize=None)
def standardize_solute(smiles: str) -> StdResult:
    if not isinstance(smiles, str) or not smiles.strip():
        return StdResult(None, None, None, "missing_smiles")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return StdResult(None, None, None, "unparseable_smiles")
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _CHOOSER.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:  # RDKit raises several exception types here
        return StdResult(None, None, None, "standardization_failed")
    if mol.GetNumHeavyAtoms() < 2:
        return StdResult(None, None, None, "too_small")
    if Chem.GetFormalCharge(mol) != 0:
        return StdResult(None, None, None, "charged_after_neutralize")
    return StdResult(
        Chem.MolToSmiles(mol),
        Chem.MolToInchiKey(mol),
        Descriptors.MolWt(mol),
    )


@lru_cache(maxsize=None)
def canonical_solvent(smiles: str) -> StdResult:
    if not isinstance(smiles, str) or not smiles.strip():
        return StdResult(None, None, None, "missing_solvent_smiles")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return StdResult(None, None, None, "unparseable_solvent_smiles")
    can = Chem.MolToSmiles(mol)
    if "." in can:
        return StdResult(can, None, None, "mixed_solvent")
    return StdResult(can, Chem.MolToInchiKey(mol), Descriptors.MolWt(mol))


def connectivity_key(inchikey: str | None) -> str | None:
    """First InChIKey block: same skeleton, ignoring stereo and isotopes."""
    return inchikey.split("-")[0] if inchikey else None

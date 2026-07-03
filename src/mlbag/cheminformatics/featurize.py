"""SMILES-based molecular featurization. Requires the `chem` extra (RDKit)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def canonicalize_smiles(smiles: list[str], *, allow_invalid: bool = False) -> list[str | None]:
    """Map each SMILES to RDKit's canonical form. Invalid SMILES raise unless `allow_invalid`."""
    from rdkit import Chem

    canonical: list[str | None] = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            if allow_invalid:
                canonical.append(None)
                continue
            raise ValueError(f"Invalid SMILES: {smi!r}")
        canonical.append(Chem.MolToSmiles(mol, canonical=True))
    return canonical


def morgan_fingerprints(smiles: list[str], *, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan (ECFP-style) fingerprints as a dense `(len(smiles), n_bits)` uint8 array."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fingerprints = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi!r}")
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(bit_vector, fingerprints[i])
    return fingerprints


def rdkit_descriptors(smiles: list[str], descriptors: list[str] | None = None) -> pd.DataFrame:
    """Compute RDKit physicochemical descriptors (default: all of them) as a DataFrame.

    Output has a `smiles` column plus one column per descriptor name.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    available = dict(Descriptors._descList)
    names = descriptors if descriptors is not None else list(available)
    unknown = set(names) - set(available)
    if unknown:
        raise ValueError(f"Unknown RDKit descriptor(s): {sorted(unknown)}")

    rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES at index {i}: {smi!r}")
        rows.append({"smiles": smi, **{name: available[name](mol) for name in names}})
    return pd.DataFrame(rows)

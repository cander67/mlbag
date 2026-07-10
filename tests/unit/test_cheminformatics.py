import numpy as np
import pytest

pytest.importorskip("rdkit")

from mlbag.cheminformatics import canonicalize_smiles, morgan_fingerprints, rdkit_descriptors


def test_canonicalize_smiles_normalizes_equivalent_forms():
    # Both SMILES describe ethanol; RDKit should canonicalize them to the same string.
    canonical = canonicalize_smiles(["CCO", "OCC"])
    assert canonical[0] == canonical[1]


def test_canonicalize_smiles_raises_on_invalid_by_default():
    with pytest.raises(ValueError):
        canonicalize_smiles(["not_a_smiles("])


def test_canonicalize_smiles_allow_invalid_returns_none():
    result = canonicalize_smiles(["CCO", "not_a_smiles("], allow_invalid=True)
    assert result[0] is not None
    assert result[1] is None


def test_morgan_fingerprints_shape_and_dtype():
    fps = morgan_fingerprints(["CCO", "CCC"], radius=2, n_bits=128)
    assert fps.shape == (2, 128)
    assert fps.dtype == np.uint8


def test_morgan_fingerprints_identical_smiles_give_identical_fingerprints():
    fps = morgan_fingerprints(["CCO", "CCO"], n_bits=256)
    assert (fps[0] == fps[1]).all()


def test_morgan_fingerprints_raises_on_invalid_smiles():
    with pytest.raises(ValueError):
        morgan_fingerprints(["CCO", "not_a_smiles("])


def test_rdkit_descriptors_default_includes_molwt_column():
    df = rdkit_descriptors(["CCO"])
    assert "smiles" in df.columns
    assert "MolWt" in df.columns
    assert len(df) == 1


def test_rdkit_descriptors_subset_returns_only_requested_columns():
    df = rdkit_descriptors(["CCO", "CCC"], descriptors=["MolWt", "TPSA"])
    assert set(df.columns) == {"smiles", "MolWt", "TPSA"}
    assert len(df) == 2


def test_rdkit_descriptors_unknown_descriptor_raises():
    with pytest.raises(ValueError):
        rdkit_descriptors(["CCO"], descriptors=["NotARealDescriptor"])


def test_rdkit_descriptors_raises_on_invalid_smiles():
    with pytest.raises(ValueError):
        rdkit_descriptors(["not_a_smiles("])

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from torch_geometric.data import Data

ATOM_FEATURE_NAMES = (
    "atomic_num",
    "degree",
    "total_num_hs",
    "formal_charge",
    "is_aromatic",
)

BOND_FEATURE_NAMES = (
    "bond_type",
    "stereo",
    "is_conjugated",
    "in_ring",
)

DESCRIPTOR_FUNCTIONS = {
    "mol_weight": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "num_h_donors": Descriptors.NumHDonors,
    "num_h_acceptors": Descriptors.NumHAcceptors,
    "num_rotatable_bonds": Descriptors.NumRotatableBonds,
    "ring_count": Descriptors.RingCount,
}

DESCRIPTOR_NAMES = tuple(DESCRIPTOR_FUNCTIONS.keys())


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetTotalNumHs()),
        float(atom.GetFormalCharge()),
        float(int(atom.GetIsAromatic())),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [
        float(int(bond.GetBondType())),
        float(int(bond.GetStereo())),
        float(int(bond.GetIsConjugated())),
        float(int(bond.IsInRing())),
    ]


def compute_descriptor_vector(mol: Chem.Mol) -> np.ndarray:
    values = [float(descriptor(mol)) for descriptor in DESCRIPTOR_FUNCTIONS.values()]
    return np.asarray(values, dtype=np.float32)


def ecfp4_fingerprint(mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_data(smiles: str, label: Optional[int] = None) -> Optional[Data]:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = [atom_features(atom) for atom in mol.GetAtoms()]
    if not node_features:
        return None

    directed_edges: list[list[int]] = []
    edge_attributes: list[list[float]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        directed_edges.extend([[i, j], [j, i]])
        edge_attributes.extend([feats, feats])

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attributes, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(BOND_FEATURE_NAMES)), dtype=torch.float32)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        desc=torch.tensor(compute_descriptor_vector(mol), dtype=torch.float32).unsqueeze(0),
        smiles=smiles,
    )
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)

    return data


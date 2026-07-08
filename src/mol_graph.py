"""
mol_graph.py
Convert a drug SMILES string → PyTorch Geometric Data object.

Graph structure:
  - Nodes  = atoms  (features: atomic num, degree, charge, etc.)
  - Edges  = bonds  (features: bond type, ring membership, etc.)
"""

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from torch_geometric.data import Data


# ── Atom-level feature helpers ─────────────────────────────────────────────

ATOM_TYPES = [
    "C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "Other"
]

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def one_hot(value, choices: list) -> list:
    """Return a one-hot list; last slot = 'Other'."""
    vec = [0] * len(choices)
    if value in choices:
        vec[choices.index(value)] = 1
    else:
        vec[-1] = 1
    return vec

def atom_features(atom) -> list:
    """
    Build a 39-dimensional feature vector for one atom.
    
    Features:
      [0:11]  atom type one-hot (C, N, O, S, F, P, Cl, Br, I, H, Other)
      [11:17] degree one-hot (0-5)
      [17:22] num Hs one-hot (0-4)
      [22:27] hybridisation one-hot (SP, SP2, SP3, SP3D, SP3D2)
      [27]    is in ring
      [28]    is aromatic
      [29]    formal charge (normalised ÷ 5)
      [30]    num radical electrons
      [31:39] unused / padding (zeros — space for future features)
    """
    symbol = atom.GetSymbol()

    feats = (
        one_hot(symbol, ATOM_TYPES)                          # 11
        + one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])     # 6
        + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])    # 5
        + one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES)  # 5
        + [int(atom.IsInRing())]                             # 1
        + [int(atom.GetIsAromatic())]                        # 1
        + [atom.GetFormalCharge() / 5.0]                     # 1
        + [atom.GetNumRadicalElectrons()]                    # 1
    )
    # Total so far = 31. Pad to 39.
    feats += [0] * (39 - len(feats))
    return feats


# ── Bond-level feature helpers ─────────────────────────────────────────────

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

def bond_features(bond) -> list:
    """
    Build a 6-dimensional feature vector for one bond.
    
    Features:
      [0:4]  bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC)
      [4]    is in ring
      [5]    is conjugated
    """
    return (
        one_hot(bond.GetBondType(), BOND_TYPES)  # 4
        + [int(bond.IsInRing())]                 # 1
        + [int(bond.GetIsConjugated())]          # 1
    )


# ── Main conversion function ───────────────────────────────────────────────

def smiles_to_graph(smiles: str, label: int = -1) -> Data | None:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    Args:
        smiles : SMILES string of the molecule e.g. "CC(=O)Oc1ccccc1C(=O)O"
        label  : integer severity label (-1 = unknown)

    Returns:
        torch_geometric.data.Data with:
            x         — node feature matrix  [num_atoms, 39]
            edge_index — COO edge index       [2, num_edges*2]
            edge_attr  — edge feature matrix  [num_edges*2, 6]
            y          — label tensor         [1]
            smiles     — original SMILES string
        or None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  ⚠️  Invalid SMILES: {smiles}")
        return None

    # ── Node features ──────────────────────────────────────────────────────
    atom_feats = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # ── Edge index + edge features (undirected → add both directions) ──────
    edge_indices = []
    edge_attrs   = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)

        # Add both directions (undirected graph)
        edge_indices += [[i, j], [j, i]]
        edge_attrs   += [bf, bf]

    if len(edge_indices) == 0:
        # Single-atom molecule (e.g. [K+]) — no bonds
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 6),  dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attrs,   dtype=torch.float)

    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=y, smiles=smiles)


def describe_graph(g: Data, drug_name: str = "") -> None:
    """Pretty-print a graph summary."""
    title = f"  {drug_name}" if drug_name else "  Graph"
    print(f"{title}")
    print(f"    Atoms (nodes)  : {g.x.shape[0]}")
    print(f"    Node features  : {g.x.shape[1]}")
    print(f"    Bonds (edges)  : {g.edge_index.shape[1] // 2}")
    print(f"    Edge features  : {g.edge_attr.shape[1] if g.edge_attr.shape[0] > 0 else 0}")
    print(f"    Label          : {g.y.item()}")


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Molecular Graph Builder — Smoke Test")
    print("=" * 55)

    test_drugs = [
        ("Aspirin",    "CC(=O)Oc1ccccc1C(=O)O",                  2),
        ("Warfarin",   "CC(=O)Cc1ccc(OC)c(c1)C(=O)O",            0),
        ("Metformin",  "CN(C)C(=N)NC(=N)N",                       1),
        ("Potassium",  "[K+]",                                     0),  # edge case: 1 atom
        ("Ibuprofen",  "CC(C)Cc1ccc(cc1)C(C)C(=O)O",             2),
    ]

    all_ok = True
    for name, smi, lbl in test_drugs:
        g = smiles_to_graph(smi, label=lbl)
        if g is None:
            print(f"\n  ❌ FAILED: {name}")
            all_ok = False
        else:
            print()
            describe_graph(g, name)

    print()
    if all_ok:
        print("✅  All molecules converted successfully!")
        print("    mol_graph.py is ready.\n")
    else:
        print("❌  Some molecules failed — check SMILES strings.\n")
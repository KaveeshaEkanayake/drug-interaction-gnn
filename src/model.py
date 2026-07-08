"""
model.py
Graph Neural Network for drug interaction prediction.

Architecture:
    1. GNN Encoder   — 3 message passing layers per molecule
    2. Readout       — global mean pooling → one vector per molecule
    3. Classifier    — combines both vectors → predicts severity class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class MoleculeEncoder(nn.Module):
    """
    Encodes a single molecule graph into a fixed-size vector.

    Input  : PyG graph  (x=[num_atoms, 39], edge_index, edge_attr)
    Output : tensor     [batch_size, hidden_dim]
    """

    def __init__(self, node_features: int = 39, hidden_dim: int = 128):
        super().__init__()

        # 3 GCN (Graph Convolutional Network) layers
        # Each layer = one round of message passing
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim,    hidden_dim)
        self.conv3 = GCNConv(hidden_dim,    hidden_dim)

        # Batch normalisation — stabilises training
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Dropout — prevents overfitting
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index, batch):
        """
        x          : [num_atoms_in_batch, 39]   node features
        edge_index : [2, num_edges_in_batch]     bond connections
        batch      : [num_atoms_in_batch]         which graph each atom belongs to
        """

        # Layer 1 — atoms learn about immediate neighbours
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2 — atoms learn about 2-hop neighbours
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3 — atoms learn about 3-hop neighbours
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Readout — average all atom vectors → one vector per molecule
        # [num_atoms, hidden_dim] → [batch_size, hidden_dim]
        x = global_mean_pool(x, batch)

        return x  # shape: [batch_size, hidden_dim]


class DrugInteractionGNN(nn.Module):
    """
    Full model for drug interaction prediction.

    Takes two molecule graphs (Drug A + Drug B),
    encodes each independently with shared weights,
    combines their fingerprints,
    classifies the interaction severity.

    Output classes:
        0 = dangerous
        1 = moderate
        2 = mild
        3 = safe
    """

    def __init__(
        self,
        node_features : int = 39,
        hidden_dim    : int = 128,
        num_classes   : int = 4,
        dropout       : float = 0.3,
    ):
        super().__init__()

        # Shared encoder — same weights used for both Drug A and Drug B
        # This makes sense because both drugs are molecules (same domain)
        self.encoder = MoleculeEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
        )

        # Classifier — takes concatenated fingerprints of both drugs
        # Input size = hidden_dim * 2 (drug A + drug B concatenated)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, batch_a, batch_b):
        """
        batch_a : PyG Batch object for Drug A molecules
        batch_b : PyG Batch object for Drug B molecules
        """

        # Encode each drug independently using shared encoder
        fp_a = self.encoder(batch_a.x, batch_a.edge_index, batch_a.batch)
        fp_b = self.encoder(batch_b.x, batch_b.edge_index, batch_b.batch)

        # Concatenate both fingerprints
        # [batch_size, 128] + [batch_size, 128] → [batch_size, 256]
        combined = torch.cat([fp_a, fp_b], dim=1)

        # Classify
        logits = self.classifier(combined)  # [batch_size, 4]

        return logits

    def predict(self, batch_a, batch_b):
        """
        Returns predicted class index and confidence scores.
        Use this at inference time (not during training).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch_a, batch_b)
            probs  = F.softmax(logits, dim=1)
            pred   = torch.argmax(probs, dim=1)
        return pred, probs


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from mol_graph import smiles_to_graph
    from torch_geometric.data import Batch

    print("=" * 55)
    print("  Drug Interaction GNN — Model Smoke Test")
    print("=" * 55)

    # Build model
    model = DrugInteractionGNN(
        node_features=39,
        hidden_dim=128,
        num_classes=4,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model built successfully!")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Architecture     :")
    print(f"    MoleculeEncoder × 1 (shared for both drugs)")
    print(f"    → 3 GCN layers (message passing)")
    print(f"    → global mean pooling")
    print(f"    Classifier")
    print(f"    → Linear(256→128) → Linear(128→64) → Linear(64→4)")

    # Create two test molecules
    graph_a = smiles_to_graph("CC(=O)Cc1ccc(OC)c(c1)C(=O)O", label=0)  # Warfarin
    graph_b = smiles_to_graph("CC(=O)Oc1ccccc1C(=O)O",        label=0)  # Aspirin

    # Batch them (needed even for single pairs)
    batch_a = Batch.from_data_list([graph_a])
    batch_b = Batch.from_data_list([graph_b])

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(batch_a, batch_b)
        probs  = F.softmax(logits, dim=1)

    print(f"\n  Test pair : Warfarin + Aspirin")
    print(f"  Raw logits: {logits.numpy().round(3)}")
    print(f"\n  Predictions (untrained — random):")
    labels = ["dangerous", "moderate", "mild", "safe"]
    for label, prob in zip(labels, probs[0].numpy()):
        bar = "█" * int(prob * 20)
        print(f"    {label:<12} {bar:<20} {prob:.1%}")

    print(f"\n✅  model.py is ready. Predictions are random until trained.\n")
"""
dataset.py
Loads the drug interaction CSV and builds a PyTorch Geometric dataset.

Each item in the dataset = one drug pair:
    graph_a  → Drug 1 molecular graph
    graph_b  → Drug 2 molecular graph
    label    → 0=dangerous, 1=moderate, 2=mild, 3=safe
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from mol_graph import smiles_to_graph

# ── Severity → integer label ───────────────────────────────────────────────
SEVERITY_MAP = {
    "dangerous" : 0,
    "moderate"  : 1,
    "mild"      : 2,
    "safe"      : 3,
}

LABEL_NAMES = {v: k for k, v in SEVERITY_MAP.items()}  # reverse lookup


class DrugInteractionDataset(Dataset):
    """
    PyTorch Dataset for drug pair interaction prediction.

    Each __getitem__ returns a dict:
        {
            "graph_a"    : PyG Data object for drug 1,
            "graph_b"    : PyG Data object for drug 2,
            "label"      : integer (0-3),
            "drug1_name" : str,
            "drug2_name" : str,
            "interaction_type" : str,
        }
    """

    def __init__(self, csv_path: str, verbose: bool = True):
        super().__init__()
        self.samples = []
        self.skipped = 0

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if verbose:
            print(f"  Loading {len(df)} rows from {os.path.basename(csv_path)} …")

        for _, row in df.iterrows():
            label_str = str(row["severity"]).strip().lower()
            label     = SEVERITY_MAP.get(label_str, -1)

            if label == -1:
                if verbose:
                    print(f"  ⚠️  Unknown severity '{label_str}' — skipping row.")
                self.skipped += 1
                continue

            graph_a = smiles_to_graph(str(row["smiles1"]), label=label)
            graph_b = smiles_to_graph(str(row["smiles2"]), label=label)

            if graph_a is None or graph_b is None:
                if verbose:
                    print(f"  ⚠️  Invalid SMILES for {row['drug1_name']} or {row['drug2_name']} — skipping.")
                self.skipped += 1
                continue

            self.samples.append({
                "graph_a"          : graph_a,
                "graph_b"          : graph_b,
                "label"            : label,
                "drug1_name"       : str(row["drug1_name"]),
                "drug2_name"       : str(row["drug2_name"]),
                "interaction_type" : str(row["interaction_type"]),
            })

        if verbose:
            self._print_summary()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _print_summary(self):
        """Print dataset statistics."""
        total = len(self.samples)
        print(f"\n  ✅ Dataset loaded: {total} valid pairs  |  {self.skipped} skipped")

        # Count per severity
        counts = {name: 0 for name in SEVERITY_MAP}
        for s in self.samples:
            counts[LABEL_NAMES[s["label"]]] += 1

        print("\n  Severity breakdown:")
        for severity, count in counts.items():
            bar = "█" * count
            print(f"    {severity:<12} {bar} ({count})")


# ── Quick test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 55)
    print("  Drug Interaction Dataset — Smoke Test")
    print("=" * 55)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(BASE_DIR, "data", "raw", "synthetic_interactions.csv")

    dataset = DrugInteractionDataset(csv_path, verbose=True)

    print("\n  First 3 samples:")
    print(f"  {'Drug 1':<15} {'Drug 2':<15} {'Severity':<12} {'Interaction'}")
    print(f"  {'─'*15} {'─'*15} {'─'*12} {'─'*20}")

    for i in range(min(3, len(dataset))):
        s = dataset[i]
        print(
            f"  {s['drug1_name']:<15} "
            f"{s['drug2_name']:<15} "
            f"{LABEL_NAMES[s['label']]:<12} "
            f"{s['interaction_type']}"
        )

    print(f"\n  Sample graph_a shape : {dataset[0]['graph_a'].x.shape}")
    print(f"  Sample graph_b shape : {dataset[0]['graph_b'].x.shape}")
    print(f"\n✅  dataset.py is ready.\n")
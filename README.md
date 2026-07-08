# 💊 Drug Interaction Checker — Graph Neural Network

A machine learning system that analyses the **molecular structure** of two drugs
and predicts how they interact — dangerous, moderate, or mild.

🔗 **Live Demo:** [coming soon]

---

## What Makes This Different

Most drug interaction tools look up a table of known interactions.
This system uses a **Graph Neural Network (GNN)** to analyse the actual
molecular structure of each drug — the same approach used by Pfizer and
Google DeepMind for drug discovery.
User types two drug names
↓
PubChem API fetches molecular structures (SMILES)
↓
RDKit converts SMILES → molecular graph (atoms=nodes, bonds=edges)
↓
GNN runs message passing across both molecular graphs
↓
Classifier predicts: DANGEROUS / MODERATE / MILD
↓
Gemini 2.5 Flash explains the interaction in plain English

---

## Why Graph Neural Networks?

Molecules are naturally graph-shaped data — atoms are nodes, bonds are edges.
Standard neural networks expect flat arrays and cannot capture this structure.
GNNs are designed specifically for graph-shaped data, preserving the full
molecular geometry during learning.

---

## Architecture
MoleculeEncoder (shared weights for both drugs)
→ GCNConv layer 1  (message passing — 1-hop neighbours)
→ GCNConv layer 2  (message passing — 2-hop neighbours)
→ GCNConv layer 3  (message passing — 3-hop neighbours)
→ Global mean pooling → molecular fingerprint [128-dim]
DrugInteractionGNN
→ Encode Drug A → fingerprint A [128-dim]
→ Encode Drug B → fingerprint B [128-dim]
→ Concatenate   → combined      [256-dim]
→ Linear(256→128) → ReLU → Dropout
→ Linear(128→64)  → ReLU → Dropout
→ Linear(64→4)    → severity prediction

| Component | Technology |
|---|---|
| GNN Framework | PyTorch Geometric |
| Molecular Processing | RDKit |
| Molecular Data | PubChem API (free, no key) |
| Interaction Labels | Kaggle DDI Dataset (191K pairs) |
| Explanations | Gemini 2.5 Flash |
| UI | Streamlit + 3Dmol.js (3D molecules) |

---

## Training Data

- **Source:** Kaggle Drug-Drug Interactions dataset (191,541 pairs)
- **Processing:** Severity labels extracted from interaction descriptions
- **Final dataset:** 1,488 balanced pairs (496 per class)
- **Train/Val split:** 80/20
- **Validation accuracy:** 64.4% (vs 33% random baseline)

---

## Project Structure
drug-interaction-gnn/
│
├── src/
│   ├── download_data.py      # synthetic dataset creation
│   ├── build_dataset.py      # real DDI dataset + PubChem pipeline
│   ├── mol_graph.py          # SMILES → PyG graph conversion
│   ├── dataset.py            # PyTorch dataset loader
│   ├── model.py              # GNN architecture
│   ├── train.py              # training pipeline
│   └── app.py                # Streamlit web app
│
├── data/
│   ├── raw/                  # original downloaded datasets
│   └── processed/            # training-ready CSVs + SMILES cache
│
├── models/
│   └── best_model.pt         # saved trained model
│
├── requirements.txt
└── README.md

---

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/KaveeshaEkanayake/drug-interaction-gnn
cd drug-interaction-gnn

# 2. Install dependencies (use conda for RDKit)
conda install -c conda-forge rdkit -y
pip install -r requirements.txt

# 3. Add your Gemini API key
echo GEMINI_API_KEY=your_key_here > .env

# 4. Train the model
python src/train.py

# 5. Run the app
streamlit run src/app.py
```

---

## Results

| Metric | Score |
|---|---|
| Validation Accuracy | 64.4% |
| Random Baseline | 33.3% |
| Improvement | +31.1% |
| Mild F1 | 0.76 |
| Moderate F1 | 0.65 |
| Dangerous F1 | 0.49 |

---

## Limitations

- Trained on 1,488 pairs — production systems use 50,000+
- Severity labels extracted via keyword rules, not manual annotation
- No "safe" class in current training data
- For educational purposes only — not for clinical use

---

## Author

**Kaveesha Ekanayake**
Computing Student | ML Engineering

[![GitHub](https://img.shields.io/badge/GitHub-KaveeshaEkanayake-black)](https://github.com/KaveeshaEkanayake)
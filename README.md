# 💊 Drug Interaction Checker

A machine learning web app that predicts how two drugs interact by analysing their molecular structures using Graph Neural Networks.

🔗 **Live Demo:** [drug-interaction-gnn.streamlit.app](https://drug-interaction-gnn.streamlit.app)

---

## Overview

Most drug interaction tools work by looking up a fixed table of known combinations. This project takes a different approach. It uses a Graph Neural Network to read the actual molecular structure of each drug and reason about how they might interact, the same technique used in pharmaceutical research for drug discovery.

Type in two drug names and the app will predict whether the combination is dangerous, moderate, or mild, show you both molecules spinning in 3D, and explain the interaction in plain English using Gemini 2.5 Flash.

---

## How It Works

Drugs are molecules. Molecules are graphs where atoms are nodes and bonds are edges. A standard neural network cannot process graph shaped data without flattening it and losing the structure. Graph Neural Networks are built specifically to work with graphs, preserving all the spatial and chemical relationships between atoms.
User enters two drug names
↓
PubChem API fetches molecular structures
↓
RDKit converts each molecule into a graph
↓
GNN runs message passing across both graphs
↓
Classifier predicts: Dangerous / Moderate / Mild
↓
Gemini 2.5 Flash explains the result in plain English

---

## Tech Stack

| Component | Technology |
|---|---|
| GNN Framework | PyTorch Geometric |
| Molecular Processing | RDKit |
| Molecular Data | PubChem API |
| Interaction Labels | Kaggle DDI Dataset (191K pairs) |
| Explanations | Gemini 2.5 Flash |
| UI | Streamlit with 3Dmol.js |

---

## Model

The GNN encoder runs three rounds of message passing on each molecule, where every atom gathers information from its neighbours and updates its own representation. After three rounds each atom carries a summary of its broader chemical environment. A global pooling step then collapses all atom vectors into a single molecular fingerprint. Both fingerprints are concatenated and passed through a classifier.

**Validation accuracy: 64.4%** across three classes on 298 held out drug pairs, compared to a 33.3% random baseline.

---
## Project Structure

```
drug-interaction-gnn/
├── src/
│   ├── mol_graph.py          # SMILES to PyG graph conversion
│   ├── dataset.py            # PyTorch dataset loader
│   ├── model.py              # GNN architecture
│   ├── train.py              # training pipeline
│   ├── build_dataset.py      # data processing and PubChem pipeline
│   └── app.py                # Streamlit web app
├── data/
│   ├── raw/                  # source datasets
│   └── processed/            # training ready CSVs
├── models/
│   └── best_model.pt         # trained model weights
└── requirements.txt
```
---

## Run Locally

```bash
git clone https://github.com/KaveeshaEkanayake/drug-interaction-gnn
cd drug-interaction-gnn

conda install -c conda-forge rdkit -y
pip install -r requirements.txt

echo GEMINI_API_KEY=your_key_here > .env

python src/train.py
streamlit run src/app.py
```

---

## Data Sources

The interaction dataset used for training is the [Drug-Drug Interactions dataset](https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions) by MGhobashy on Kaggle, which contains 191,541 drug interaction pairs sourced from DrugBank. Molecular structures were fetched at runtime using the [PubChem REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest).

---
## Limitations

This project is for educational purposes only and should not be used to make real medical decisions. The model was trained on 1,488 drug pairs with labels extracted using keyword rules rather than clinical annotation. A production system would require significantly more data and rigorous validation.

---


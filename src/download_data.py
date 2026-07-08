"""
Download and prepare drug interaction data.
Uses a synthetic dataset for immediate development,
plus attempts to fetch TWOSIDES for real training data.
"""

import os
import csv
import requests
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── Synthetic dataset (always created — works offline) ─────────────────────
SYNTHETIC_ROWS = [
    ("Warfarin",      "Aspirin",         "CC(=O)Cc1ccc(OC)c(c1)C(=O)O",                        "CC(=O)Oc1ccccc1C(=O)O",                          "bleeding_risk",        "dangerous"),
    ("Metformin",     "Ibuprofen",       "CN(C)C(=N)NC(=N)N",                                   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                     "renal_impairment",     "moderate"),
    ("Lisinopril",    "Potassium",       "OC(=O)C1CCN(CC1)C(=O)C(N)CCc1ccccc1",                "[K+]",                                            "hyperkalemia",         "dangerous"),
    ("Atorvastatin",  "Clarithromycin",  "CC(C)c1c(C(=O)Nc2ccccc2)c(c(c(n1)c3ccc(F)cc3)O)CC", "CC1OC(=O)c2cc(C)ccc2O1",                         "myopathy_risk",        "moderate"),
    ("Omeprazole",    "Clopidogrel",     "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",        "OC(=O)c1ccccc1Cl",                               "reduced_efficacy",     "mild"),
    ("Amoxicillin",   "Methotrexate",    "CC1(C)SC2C(NC(=O)Cc3ccc(N)cc3)C(=O)N2C1C(=O)O",    "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O", "methotrexate_toxicity", "dangerous"),
    ("Simvastatin",   "Amlodipine",      "CCC(C)(C)OC(=O)C1CC(O)(C(=O)OC)C(C)(C)O1",         "CCOC(=O)c1c(COCCN)nc(C)c(c1)C(=O)OC",           "increased_exposure",   "moderate"),
    ("Digoxin",       "Amiodarone",      "OC1CC2CC(C1)C3C2CC4(O)CCC(OC5OC(CO)C(O)C(O)C5O)C4C3=O", "CCCC(=O)Oc1ccc(cc1)c2cc3ccccc3o2",         "bradycardia",          "dangerous"),
    ("Paracetamol",   "Ibuprofen",       "CC(=O)Nc1ccc(O)cc1",                                 "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                     "renal_stress",         "mild"),
    ("Fluoxetine",    "Tramadol",        "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",                "OC1(CC2CC1CC2N(C)C)c1ccccc1",                    "serotonin_syndrome",   "dangerous"),
    ("Ciprofloxacin", "Antacid",         "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",       "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",    "reduced_absorption",   "mild"),
    ("Lithium",       "Ibuprofen",       "[Li+]",                                               "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                     "lithium_toxicity",     "dangerous"),
    ("Sildenafil",    "Nitrates",        "CCCC1=NN(C)C(=C1C(=O)NCC2=CC=CC=N2)c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC", "O=N[O-]",               "hypotension",          "dangerous"),
    ("Warfarin",      "Vitamin K",       "CC(=O)Cc1ccc(OC)c(c1)C(=O)O",                        "CC(=CCC1C(=O)c2ccccc2C1=O)C",                    "reduced_anticoag",     "moderate"),
    ("Metformin",     "Alcohol",         "CN(C)C(=N)NC(=N)N",                                   "CCO",                                             "lactic_acidosis",      "moderate"),
]

SEVERITY_LABEL = {"dangerous": 0, "moderate": 1, "mild": 2, "safe": 3}


def write_synthetic(path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "drug1_name", "drug2_name",
            "smiles1", "smiles2",
            "interaction_type", "severity"
        ])
        writer.writerows(SYNTHETIC_ROWS)
    print(f"  ✅ Synthetic dataset saved → {path}  ({len(SYNTHETIC_ROWS)} pairs)")


def download_file(url: str, dest: str, desc: str = "Downloading") -> bool:
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(desc=desc, total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ⚠️  Download failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  Drug Interaction GNN — Data Setup")
    print("=" * 55)

    # 1. Synthetic dataset (always)
    print("\n[1/2] Synthetic dev dataset")
    synthetic_path = os.path.join(RAW_DIR, "synthetic_interactions.csv")
    if os.path.exists(synthetic_path):
        print(f"  Already exists → {synthetic_path}")
    else:
        write_synthetic(synthetic_path)

    # 2. TWOSIDES (optional, large — skip if already exists)
    print("\n[2/2] TWOSIDES real-world dataset (optional)")
    twosides_path = os.path.join(RAW_DIR, "twosides.csv")
    if os.path.exists(twosides_path):
        print(f"  Already exists → {twosides_path}")
    else:
        print("  Attempting download (this may take a while) …")
        url = "https://static-content.springer.com/esm/srep00196/MediaObjects/srep00196-s1.csv"
        ok  = download_file(url, twosides_path, "TWOSIDES")
        if not ok:
            print("  ℹ️  We'll use the synthetic dataset for now — that's fine for development.")

    print("\n✅  Data setup complete! Ready to build molecular graphs.\n")


if __name__ == "__main__":
    main()
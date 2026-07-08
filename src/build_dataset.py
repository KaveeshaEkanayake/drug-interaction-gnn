"""
build_dataset.py

Processes the Kaggle DDI dataset (191K rows) into a training-ready CSV.

Steps:
    1. Extract severity label from interaction description text
    2. Sample a balanced subset (we don't need all 191K)
    3. Call PubChem API to get SMILES for each unique drug
    4. Save final training CSV with smiles1, smiles2, severity
"""

import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR     = os.path.join(BASE_DIR, "data", "raw")
PROCESSED   = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

INPUT_CSV   = os.path.join(RAW_DIR,   "db_drug_interactions.csv")
OUTPUT_CSV  = os.path.join(PROCESSED, "real_interactions.csv")
SMILES_CACHE= os.path.join(PROCESSED, "smiles_cache.csv")

# How many pairs to keep per severity class (balanced dataset)
SAMPLES_PER_CLASS = 500  # 500 × 4 classes = 2000 total pairs max


# ── Step 1 — Extract severity from description text ────────────────────────

def extract_severity(description: str) -> str:
    """
    Map interaction description text to severity label.

    Rules derived from patterns in the dataset:
        dangerous → risk/severity increased, toxicity increased
        moderate  → metabolism changed, absorption changed
        mild      → activity increased/decreased (non-toxic)
        safe      → activity decreased (protective effect)
    """
    d = str(description).lower()

    # DANGEROUS patterns
    if any(p in d for p in [
        "risk or severity of adverse effects can be increased",
        "risk or severity of adverse effects may be increased",
        "toxic activities",
        "toxicity",
        "cardiotoxic" and "increase",
        "nephrotoxic",
        "hepatotoxic",
        "bleeding",
        "hemorrhag",
        "seizure",
        "serotonin",
        "hypotension",
        "bradycardia",
        "qt-prolonging",
    ]):
        return "dangerous"

    # MODERATE patterns
    if any(p in d for p in [
        "metabolism of",
        "metabolic",
        "serum concentration",
        "plasma concentration",
        "absorption",
        "excretion",
        "clearance",
        "bioavailability",
        "auc",
        "half-life",
    ]):
        return "moderate"

    # MILD patterns
    if any(p in d for p in [
        "activities of",
        "activity of",
        "photosensitizing",
        "hypoglycemic",
        "anticoagulant",
        "sedative",
        "diuretic",
        "antihypertensive",
    ]):
        return "mild"

    # Default → mild (unknown but some interaction exists)
    return "mild"


# ── Step 2 — PubChem SMILES lookup with caching ────────────────────────────

def load_smiles_cache() -> dict:
    """Load previously fetched SMILES from cache file."""
    if os.path.exists(SMILES_CACHE):
        df = pd.read_csv(SMILES_CACHE)
        return dict(zip(df["drug_name"], df["smiles"]))
    return {}


def save_smiles_cache(cache: dict) -> None:
    """Save SMILES cache to disk."""
    rows = [{"drug_name": k, "smiles": v} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(SMILES_CACHE, index=False)


def get_smiles_pubchem(drug_name: str, cache: dict) -> str | None:
    """
    Fetch SMILES from PubChem API with caching.
    Tries IsomericSMILES first, falls back to ConnectivitySMILES.
    """
    key = drug_name.strip().lower()

    # Check cache first
    if key in cache:
        val = cache[key]
        return None if val == "NOT_FOUND" else val

    # Try both SMILES property types
    for prop in ["IsomericSMILES", "CanonicalSMILES", "ConnectivitySMILES"]:
        url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{requests.utils.quote(drug_name)}/property/{prop}/JSON"
        )
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                continue
            data       = r.json()
            properties = data["PropertyTable"]["Properties"][0]
            # Get whatever SMILES field is present
            smiles = (
                properties.get("IsomericSMILES")
                or properties.get("CanonicalSMILES")
                or properties.get("ConnectivitySMILES")
            )
            if smiles:
                cache[key] = smiles
                time.sleep(0.2)
                return smiles
        except Exception:
            continue

    cache[key] = "NOT_FOUND"
    time.sleep(0.1)
    return None
    """
    Fetch SMILES from PubChem API with caching.
    Caches both hits (SMILES string) and misses ('NOT_FOUND').
    """
    key = drug_name.strip().lower()

    # Check cache first
    if key in cache:
        val = cache[key]
        return None if val == "NOT_FOUND" else val

    # Call PubChem
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(drug_name)}/property/CanonicalSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        smiles = r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        cache[key] = smiles
        time.sleep(0.2)  # Be polite to PubChem API
        return smiles
    except Exception:
        cache[key] = "NOT_FOUND"
        time.sleep(0.1)
        return None


# ── Step 3 — Main pipeline ─────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Drug Interaction Dataset Builder")
    print("=" * 60)

    # ── Load raw data ──────────────────────────────────────────────────────
    print(f"\n[1/5] Loading raw data …")
    df = pd.read_csv(INPUT_CSV)
    df.columns = ["drug1_name", "drug2_name", "description"]
    print(f"      Total rows: {len(df):,}")

    # ── Extract severity labels ────────────────────────────────────────────
    print(f"\n[2/5] Extracting severity labels from descriptions …")
    df["severity"] = df["description"].apply(extract_severity)

    counts = df["severity"].value_counts()
    print(f"      Label distribution (before sampling):")
    for sev, count in counts.items():
        print(f"        {sev:<12} {count:>7,}")

    # ── Sample balanced dataset ────────────────────────────────────────────
    print(f"\n[3/5] Sampling {SAMPLES_PER_CLASS} pairs per class …")
    sampled_parts = []
    for severity in ["dangerous", "moderate", "mild", "safe"]:
        subset = df[df["severity"] == severity]
        if len(subset) == 0:
            print(f"      ⚠️  No '{severity}' samples found — skipping.")
            continue
        n = min(SAMPLES_PER_CLASS, len(subset))
        sampled_parts.append(subset.sample(n=n, random_state=42))
        print(f"      {severity:<12} → sampled {n}")

    sampled = pd.concat(sampled_parts).reset_index(drop=True)
    sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n      Total sampled: {len(sampled)} pairs")

    # ── Fetch SMILES from PubChem ──────────────────────────────────────────
    print(f"\n[4/5] Fetching SMILES from PubChem (this takes a few minutes) …")
    print(f"      Results are cached — re-runs are instant.\n")

    cache = load_smiles_cache()

    # Get all unique drug names
    all_drugs = pd.unique(
        sampled[["drug1_name", "drug2_name"]].values.ravel()
    )
    unique_drugs = [d for d in all_drugs if str(d).strip().lower() not in cache]
    print(f"      Unique drugs to fetch: {len(unique_drugs)} "
          f"({len(all_drugs) - len(unique_drugs)} cached)")

    # Fetch with progress bar
    failed = 0
    for drug in tqdm(unique_drugs, desc="      PubChem"):
        result = get_smiles_pubchem(drug, cache)
        if result is None:
            failed += 1

    save_smiles_cache(cache)
    print(f"\n      Fetched: {len(unique_drugs) - failed}  |  "
          f"Not found: {failed}  |  "
          f"Cache size: {len(cache)}")

    # ── Build final dataset ────────────────────────────────────────────────
    print(f"\n[5/5] Building final training dataset …")

    rows = []
    skipped = 0

    for _, row in sampled.iterrows():
        smiles1 = get_smiles_pubchem(row["drug1_name"], cache)
        smiles2 = get_smiles_pubchem(row["drug2_name"], cache)

        if smiles1 is None or smiles2 is None:
            skipped += 1
            continue

        rows.append({
            "drug1_name"  : row["drug1_name"],
            "drug2_name"  : row["drug2_name"],
            "smiles1"     : smiles1,
            "smiles2"     : smiles2,
            "interaction_type" : row["description"][:80],
            "severity"    : row["severity"],
        })

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n      Valid pairs : {len(rows)}")
    print(f"      Skipped     : {skipped} (SMILES not found on PubChem)")
    print(f"\n      Severity breakdown:")
    for sev, count in final_df["severity"].value_counts().items():
        bar = "█" * (count // 10)
        print(f"        {sev:<12} {bar} ({count})")

    print(f"\n      Saved → {OUTPUT_CSV}")
    print(f"\n✅  Done! Run train.py next to retrain the GNN.\n")


if __name__ == "__main__":
    main()
"""
app.py
Streamlit web app for Drug Interaction Checker.

Flow:
    User enters two drug names
    → fetch SMILES from PubChem (free API, no key needed)
    → fallback to local dictionary if PubChem fails
    → convert to molecular graphs
    → GNN predicts interaction severity
    → Gemini explains the interaction in plain English
"""

import os
import sys
import torch
import requests
import streamlit as st
import torch.nn.functional as F
from torch_geometric.data import Batch
from dotenv import load_dotenv
import google.genai as genai

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mol_graph import smiles_to_graph
from model     import DrugInteractionGNN

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pt")

# ── Constants ──────────────────────────────────────────────────────────────
SEVERITY_LABELS = {
    0: "🔴 DANGEROUS",
    1: "🟠 MODERATE",
    2: "🟡 MILD",
    3: "🟢 SAFE",
}

SEVERITY_COLORS = {
    0: "#ff4444",
    1: "#ff8c00",
    2: "#ffd700",
    3: "#44bb44",
}

LABEL_NAMES = {
    0: "dangerous",
    1: "moderate",
    2: "mild",
    3: "safe",
}

# ── Fallback SMILES dictionary (used when PubChem is unreachable) ──────────
KNOWN_SMILES = {
    "aspirin"        : "CC(=O)Oc1ccccc1C(=O)O",
    "warfarin"       : "CC(=O)Cc1ccc(OC)c(c1)C(=O)O",
    "metformin"      : "CN(C)C(=N)NC(=N)N",
    "ibuprofen"      : "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "lisinopril"     : "OC(=O)C1CCN(CC1)C(=O)C(N)CCc1ccccc1",
    "fluoxetine"     : "CNCCC(Oc1ccc(cc1)C(F)(F)F)c1ccccc1",
    "tramadol"       : "OC1(CC2CC1CC2N(C)C)c1ccccc1",
    "amoxicillin"    : "CC1(C)SC2C(NC(=O)Cc3ccc(N)cc3)C(=O)N2C1C(=O)O",
    "omeprazole"     : "COc1ccc2nc(S(=O)Cc3ncc(C)c(OC)c3C)[nH]c2c1",
    "clopidogrel"    : "OC(=O)c1ccccc1Cl",
    "atorvastatin"   : "CC(C)c1c(C(=O)Nc2ccccc2)c(c(c(n1)c3ccc(F)cc3)O)CC",
    "simvastatin"    : "CCC(C)(C)OC(=O)C1CC(O)(C(=O)OC)C(C)(C)O1",
    "amlodipine"     : "CCOC(=O)c1c(COCCN)nc(C)c(c1)C(=O)OC",
    "digoxin"        : "OC1CC2CC(C1)C3C2CC4(O)CCC(OC5OC(CO)C(O)C(O)C5O)C4C3=O",
    "amiodarone"     : "CCCC(=O)Oc1ccc(cc1)c2cc3ccccc3o2",
    "paracetamol"    : "CC(=O)Nc1ccc(O)cc1",
    "acetaminophen"  : "CC(=O)Nc1ccc(O)cc1",
    "ciprofloxacin"  : "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "lithium"        : "[Li+]",
    "sildenafil"     : "CCCC1=NN(C)C(=C1C(=O)NCC2=CC=CC=N2)c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC",
    "methotrexate"   : "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)NC(CCC(=O)O)C(=O)O",
    "clarithromycin" : "CC1OC(=O)c2cc(C)ccc2O1",
    "alcohol"        : "CCO",
    "ethanol"        : "CCO",
    "potassium"      : "[K+]",
    "digoxin"        : "OC1CC2CC(C1)C3C2CC4(O)CCC(OC5OC(CO)C(O)C(O)C5O)C4C3=O",
    "tramadol"       : "OC1(CC2CC1CC2N(C)C)c1ccccc1",
    "naproxen"       : "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "diazepam"       : "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
    "codeine"        : "COc1ccc2CC3N(C)CCC4(c1c2O)C3=C4O",
    "morphine"       : "OC1CC2N(C)CCC3(c4ccc(O)cc4O1)C2=C3",
    "penicillin"     : "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "doxycycline"    : "OC1=C(O)C(=O)C2C(N(C)C)C3CC(O)c4c(O)cccc4C3C2C1=O",
    "insulin"        : "CC(O)C(=O)O",
    "prednisone"     : "CC1CC2C3CCC4=CC(=O)CCC4(C)C3C(=O)CC2(C)C1(O)C(=O)CO",
    "levothyroxine"  : "NC(Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O",
    "gabapentin"     : "NCC1(CC(=O)O)CCCCC1",
    "sertraline"     : "CNC1CCC(c2ccccc21)c1ccc(Cl)c(Cl)c1",
    "alprazolam"     : "Cc1nnc2n1-c1ccc(Cl)cc1C(=NCC2)c1ccccc1",
}


# ── Model loader ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = DrugInteractionGNN(
        node_features=39,
        hidden_dim=checkpoint.get("hidden_dim", 128),
        num_classes=checkpoint.get("num_classes", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


# ── PubChem SMILES lookup with local fallback ──────────────────────────────
@st.cache_data(ttl=3600)
def get_smiles(drug_name: str) -> tuple[str | None, str]:
    """
    Fetch SMILES for a drug name.
    Returns (smiles, source) where source is 'pubchem' or 'local'.

    1. Try PubChem API first
    2. Fall back to local dictionary if API fails
    """
    name_lower = drug_name.strip().lower()

    # Try PubChem first
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{requests.utils.quote(drug_name)}/property/CanonicalSMILES/JSON"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        smiles = r.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        if smiles:
            return smiles, "pubchem"
    except Exception:
        pass

    # Fallback — local dictionary
    if name_lower in KNOWN_SMILES:
        return KNOWN_SMILES[name_lower], "local"

    return None, "not_found"


# ── 3D molecule viewer via py3Dmol HTML ────────────────────────────────────
def molecule_3d_viewer(smiles: str, drug_name: str, color: str = "#4fc3f7") -> str:
    """
    Generate an HTML string with a 3D rotating molecule viewer.
    Uses 3Dmol.js loaded from CDN — no extra Python packages needed.
    The molecule is built from SMILES using RDKit → SDF → 3Dmol.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        sdf_block = Chem.MolToMolBlock(mol)
        sdf_escaped = sdf_block.replace("`", "\\`").replace("\\", "\\\\").replace("\n", "\\n")
        viewer_ok = True
    except Exception:
        sdf_escaped = ""
        viewer_ok = False

    if not viewer_ok:
        return f"""
        <div style="
            width:100%; height:200px;
            display:flex; align-items:center; justify-content:center;
            background:#1a1a2e; border-radius:12px;
            color:#aaa; font-size:14px;">
            3D viewer unavailable for this molecule
        </div>
        """

    html = f"""
    <div style="position:relative; width:100%;">
        <div style="
            text-align:center;
            font-size:13px;
            color:{color};
            font-weight:600;
            margin-bottom:6px;
            letter-spacing:1px;
            text-transform:uppercase;">
            {drug_name}
        </div>
        <div id="viewer_{drug_name.replace(' ','_')}"
             style="width:100%; height:220px; border-radius:12px;
                    background:linear-gradient(135deg,#0d0d1a,#1a1a2e);
                    border:1px solid {color}33; overflow:hidden;">
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
        (function() {{
            var sdf = "{sdf_escaped}";
            sdf = sdf.replace(/\\\\n/g, '\\n');

            var viewer = $3Dmol.createViewer(
                document.getElementById("viewer_{drug_name.replace(' ','_')}"),
                {{ backgroundColor: "transparent" }}
            );

            viewer.addModel(sdf, "sdf");
            viewer.setStyle({{}}, {{
                stick: {{ radius: 0.15, colorscheme: "Jmol" }},
                sphere: {{ scale: 0.25, colorscheme: "Jmol" }}
            }});
            viewer.zoomTo();
            viewer.render();

            // Auto-rotate
            viewer.spin("y", 1);
        }})();
    </script>
    """
    return html


# ── Gemini explanation ─────────────────────────────────────────────────────
def get_gemini_explanation(
    drug1: str,
    drug2: str,
    severity: str,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ GEMINI_API_KEY not found in .env file."
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"""
You are a clinical pharmacist explaining a drug interaction to a patient.

Drug 1: {drug1}
Drug 2: {drug2}
Predicted interaction severity: {severity}

Please explain in simple, clear English (no jargon):
1. WHY these two drugs interact at the molecular level
2. WHAT symptoms or risks can occur
3. WHAT the patient should do (consult doctor, avoid combination, monitor symptoms)
4. A safer alternative if one exists

Keep it under 180 words. Be empathetic and clear.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"⚠️ Gemini explanation unavailable: {e}"


# ── Custom CSS ─────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
        /* Dark background */
        .stApp { background: #0a0a14; }

        /* Hide default streamlit header */
        header[data-testid="stHeader"] { background: transparent; }

        /* Main title */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea, #764ba2, #f64f59);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0;
        }

        .subtitle {
            text-align: center;
            color: #888;
            font-size: 0.95rem;
            margin-top: 4px;
            margin-bottom: 32px;
        }

        /* Result card */
        .result-card {
            padding: 24px;
            border-radius: 16px;
            margin: 20px 0;
            text-align: center;
        }

        .result-severity {
            font-size: 2rem;
            font-weight: 800;
            margin: 0;
        }

        .result-pair {
            font-size: 1.1rem;
            color: #ccc;
            margin-top: 8px;
        }

        /* Molecule section */
        .molecule-section {
            background: #111122;
            border-radius: 16px;
            padding: 16px;
            margin: 16px 0;
        }

        /* Confidence bar labels */
        .conf-label {
            font-size: 13px;
            color: #aaa;
            margin-bottom: 4px;
        }

        /* Info badge */
        .source-badge {
            font-size: 11px;
            color: #666;
            text-align: right;
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Drug Interaction Checker",
        page_icon="💊",
        layout="wide",
    )

    inject_css()

    # ── Header ─────────────────────────────────────────────────────────────
    st.markdown('<h1 class="main-title">💊 Drug Interaction Checker</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Powered by <b>Graph Neural Networks</b> — '
        'analyses molecular structures to predict how two drugs interact</p>',
        unsafe_allow_html=True,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    model = load_model()
    if model is None:
        st.error("⚠️ Trained model not found. Please run `python src/train.py` first.")
        return

    # ── Input section ──────────────────────────────────────────────────────
    col1, spacer, col2 = st.columns([5, 1, 5])
    with col1:
        drug1 = st.text_input("💊 Drug 1", placeholder="e.g. Warfarin")
    with spacer:
        st.markdown("<div style='text-align:center; padding-top:32px; font-size:1.4rem; color:#555;'>VS</div>", unsafe_allow_html=True)
    with col2:
        drug2 = st.text_input("💊 Drug 2", placeholder="e.g. Aspirin")

    # Quick examples
    st.markdown("**Quick examples:**")
    ex1, ex2, ex3, ex4 = st.columns(4)
    with ex1:
        if st.button("Warfarin + Aspirin"):
            drug1, drug2 = "Warfarin", "Aspirin"
    with ex2:
        if st.button("Fluoxetine + Tramadol"):
            drug1, drug2 = "Fluoxetine", "Tramadol"
    with ex3:
        if st.button("Metformin + Ibuprofen"):
            drug1, drug2 = "Metformin", "Ibuprofen"
    with ex4:
        if st.button("Digoxin + Amiodarone"):
            drug1, drug2 = "Digoxin", "Amiodarone"

    st.divider()

    # ── Check button ───────────────────────────────────────────────────────
    check = st.button("🔬 Check Interaction", type="primary", use_container_width=True)

    if check:
        if not drug1 or not drug2:
            st.warning("Please enter both drug names.")
            return
        if drug1.strip().lower() == drug2.strip().lower():
            st.warning("Please enter two different drugs.")
            return

        # Step 1 — Fetch SMILES
        with st.spinner(f"🔍 Looking up molecular structures…"):
            smiles1, source1 = get_smiles(drug1)
            smiles2, source2 = get_smiles(drug2)

        if smiles1 is None:
            st.error(
                f"❌ Could not find **{drug1}**. "
                f"Try: Aspirin, Warfarin, Metformin, Ibuprofen, Fluoxetine, Tramadol, "
                f"Omeprazole, Amoxicillin, Digoxin, Amiodarone"
            )
            return
        if smiles2 is None:
            st.error(
                f"❌ Could not find **{drug2}**. "
                f"Try: Aspirin, Warfarin, Metformin, Ibuprofen, Fluoxetine, Tramadol, "
                f"Omeprazole, Amoxicillin, Digoxin, Amiodarone"
            )
            return

        # Step 2 — Convert to graphs
        with st.spinner("🧬 Building molecular graphs…"):
            graph_a = smiles_to_graph(smiles1, label=0)
            graph_b = smiles_to_graph(smiles2, label=0)

        if graph_a is None or graph_b is None:
            st.error("❌ Could not convert molecules to graphs.")
            return

        # Step 3 — GNN prediction
        with st.spinner("🧠 GNN analysing interaction…"):
            batch_a = Batch.from_data_list([graph_a])
            batch_b = Batch.from_data_list([graph_b])
            with torch.no_grad():
                logits = model(batch_a, batch_b)
                probs  = F.softmax(logits, dim=1)[0]
                pred   = torch.argmax(probs).item()

        severity     = SEVERITY_LABELS[pred]
        color        = SEVERITY_COLORS[pred]
        severity_str = LABEL_NAMES[pred]

        # ── Results layout ─────────────────────────────────────────────────
        # Result card
        st.markdown(f"""
        <div class="result-card" style="
            background: {color}15;
            border: 2px solid {color}55;">
            <p class="result-severity" style="color:{color};">{severity}</p>
            <p class="result-pair">{drug1} &nbsp;+&nbsp; {drug2}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── 3D molecule viewers ────────────────────────────────────────────
        st.markdown("### 🔬 3D Molecular Structures")
        st.caption("Rotate with mouse • Scroll to zoom • Both molecules spinning automatically")

        mol_col1, mol_col2 = st.columns(2)

        with mol_col1:
            html1 = molecule_3d_viewer(smiles1, drug1, color="#4fc3f7")
            st.components.v1.html(html1, height=270)
            st.markdown(
                f"<div class='source-badge'>Atoms: {graph_a.x.shape[0]} | "
                f"Bonds: {graph_a.edge_index.shape[1]//2} | "
                f"Source: {source1}</div>",
                unsafe_allow_html=True
            )

        with mol_col2:
            html2 = molecule_3d_viewer(smiles2, drug2, color="#f48fb1")
            st.components.v1.html(html2, height=270)
            st.markdown(
                f"<div class='source-badge'>Atoms: {graph_b.x.shape[0]} | "
                f"Bonds: {graph_b.edge_index.shape[1]//2} | "
                f"Source: {source2}</div>",
                unsafe_allow_html=True
            )

        # ── Confidence scores ──────────────────────────────────────────────
        st.markdown("### 📊 Confidence Scores")
        conf_cols = st.columns(4)
        conf_labels  = ["Dangerous", "Moderate", "Mild", "Safe"]
        conf_colors  = ["#ff4444", "#ff8c00", "#ffd700", "#44bb44"]

        for i, (col, label, clr) in enumerate(zip(conf_cols, conf_labels, conf_colors)):
            prob = probs[i].item()
            with col:
                st.markdown(f"""
                <div style="
                    background:#111122;
                    border:1px solid {clr}44;
                    border-radius:12px;
                    padding:16px;
                    text-align:center;">
                    <div style="color:{clr}; font-size:1.6rem; font-weight:800;">
                        {prob:.0%}
                    </div>
                    <div style="color:#aaa; font-size:0.85rem; margin-top:4px;">
                        {label}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── SMILES expander ────────────────────────────────────────────────
        with st.expander("🧪 View SMILES Strings"):
            s1, s2 = st.columns(2)
            with s1:
                st.markdown(f"**{drug1}**")
                st.code(smiles1, language=None)
            with s2:
                st.markdown(f"**{drug2}**")
                st.code(smiles2, language=None)

        # ── Gemini explanation ─────────────────────────────────────────────
        st.divider()
        st.markdown("### 📋 What This Means For You")
        with st.spinner("✍️ Generating explanation…"):
            explanation = get_gemini_explanation(drug1, drug2, severity_str)
        st.markdown(explanation)

        # ── Disclaimer ─────────────────────────────────────────────────────
        st.divider()
        st.caption(
            "⚠️ This tool is for educational purposes only. "
            "Always consult a qualified healthcare professional "
            "before making any medication decisions."
        )


if __name__ == "__main__":
    main()
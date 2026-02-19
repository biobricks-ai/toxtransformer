#!/usr/bin/env python3
"""
ToxTransformer Streamlit UI

Interactive web interface for toxicity predictions using the ToxTransformer model.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
import io
import os
from typing import List, Dict, Optional

# Configuration
API_URL = os.environ.get("TOXTRANSFORMER_API_URL", "http://localhost:6515")
st.set_page_config(
    page_title="ToxTransformer - AI Toxicity Prediction",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    max-width: 100%;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.prediction-high {
    background-color: #ffebee;
    color: #c62828;
}
.prediction-low {
    background-color: #e8f5e9;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
def smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert SMILES to InChI."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchi(mol)
    except Exception:
        return None

def draw_molecule(smiles: str, size=(400, 300)):
    """Draw molecule from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        return img
    except Exception:
        return None

def get_molecular_properties(smiles: str) -> Dict:
    """Calculate basic molecular properties."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        return {
            "Molecular Weight": f"{Descriptors.MolWt(mol):.2f}",
            "LogP": f"{Descriptors.MolLogP(mol):.2f}",
            "H-Bond Donors": Descriptors.NumHDonors(mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "TPSA": f"{Descriptors.TPSA(mol):.2f}",
        }
    except Exception:
        return {}

def predict_toxicity(inchi: str) -> Optional[List[Dict]]:
    """Call ToxTransformer API for predictions."""
    try:
        response = requests.get(
            f"{API_URL}/predict_all",
            params={"inchi": inchi},
            timeout=120
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def categorize_predictions(predictions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group predictions by category."""
    categories = {}
    for pred in predictions:
        cat = pred.get("category", "Other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(pred)
    return categories

# Main app
def main():
    st.title("⚗️ ToxTransformer")
    st.markdown("### AI-Powered Toxicity Prediction")
    st.markdown("Predict 6,647 toxicity properties from molecular structure using a decoder-only transformer model.")

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        **ToxTransformer** predicts toxicity across:
        - 🧬 Genotoxicity & Mutagenicity
        - ❤️ Cardiotoxicity
        - 🧠 Neurotoxicity
        - 🫀 Hepatotoxicity
        - 🔬 ADMET Properties
        - 🐟 Environmental Toxicity
        - And many more...

        **Model**: Decoder-only transformer trained on 6,647 binary toxicity properties

        **Performance**: Mean zero-shot AUC 0.799 on external benchmarks
        """)

        st.divider()
        st.markdown("**Example SMILES:**")
        example_smiles = {
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            "Benzene": "c1ccccc1",
        }
        for name, smi in example_smiles.items():
            if st.button(name, key=f"ex_{name}"):
                st.session_state.input_smiles = smi

    # Main input
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input")
        input_type = st.radio("Input Format:", ["SMILES", "InChI"], horizontal=True)

        if input_type == "SMILES":
            smiles_input = st.text_input(
                "Enter SMILES:",
                value=st.session_state.get("input_smiles", ""),
                placeholder="e.g., CC(=O)Oc1ccccc1C(=O)O (Aspirin)"
            )
            if smiles_input:
                inchi_input = smiles_to_inchi(smiles_input)
                if inchi_input:
                    st.success(f"✓ Valid SMILES. InChI: `{inchi_input[:60]}...`")
                else:
                    st.error("Invalid SMILES string")
                    smiles_input = None
        else:
            inchi_input = st.text_input(
                "Enter InChI:",
                placeholder="e.g., InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
            )
            smiles_input = None  # We'll derive it if needed

    with col2:
        if input_type == "SMILES" and smiles_input:
            st.subheader("Structure")
            img = draw_molecule(smiles_input)
            if img:
                st.image(img, use_container_width=True)

            # Molecular properties
            props = get_molecular_properties(smiles_input)
            if props:
                st.markdown("**Properties:**")
                for k, v in props.items():
                    st.metric(k, v)

    # Predict button
    if st.button("🔬 Predict Toxicity", type="primary", use_container_width=True):
        if (input_type == "SMILES" and not smiles_input) or (input_type == "InChI" and not inchi_input):
            st.error("Please enter a valid molecule")
            return

        with st.spinner("Running predictions on 6,647 toxicity endpoints..."):
            predictions = predict_toxicity(inchi_input)

        if predictions:
            st.success(f"✓ Predicted {len(predictions)} properties")

            # Store in session state
            st.session_state.predictions = predictions
            st.session_state.last_smiles = smiles_input
            st.session_state.last_inchi = inchi_input

    # Display results
    if "predictions" in st.session_state:
        st.divider()
        st.subheader("Results")

        predictions = st.session_state.predictions

        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        positive_preds = [p for p in predictions if p.get("value", 0) > 0.5]
        high_conf = [p for p in predictions if p.get("value", 0) > 0.8]

        with col1:
            st.metric("Total Properties", len(predictions))
        with col2:
            st.metric("Positive Predictions", len(positive_preds))
        with col3:
            st.metric("High Confidence (>0.8)", len(high_conf))
        with col4:
            avg_score = sum(p.get("value", 0) for p in predictions) / len(predictions)
            st.metric("Average Score", f"{avg_score:.3f}")

        # Category breakdown
        st.subheader("Category Breakdown")
        categories = categorize_predictions(predictions)

        cat_data = []
        for cat, preds in categories.items():
            pos = sum(1 for p in preds if p.get("value", 0) > 0.5)
            cat_data.append({
                "Category": cat,
                "Total": len(preds),
                "Positive": pos,
                "Positive Rate": pos / len(preds) if preds else 0
            })

        cat_df = pd.DataFrame(cat_data).sort_values("Positive Rate", ascending=False)

        fig = px.bar(
            cat_df,
            x="Category",
            y="Positive",
            title="Positive Predictions by Category",
            color="Positive Rate",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top predictions
        st.subheader("Top Toxic Predictions")
        top_n = st.slider("Number of top predictions to show:", 10, 100, 20)

        sorted_preds = sorted(predictions, key=lambda x: x.get("value", 0), reverse=True)
        top_preds = sorted_preds[:top_n]

        df = pd.DataFrame([{
            "Property": p.get("title", "Unknown")[:80],
            "Probability": f"{p.get('value', 0):.3f}",
            "Category": p.get("category", "Other"),
            "Source": p.get("source", "Unknown"),
        } for p in top_preds])

        st.dataframe(df, use_container_width=True, height=400)

        # Download results
        st.subheader("Export Results")
        col1, col2 = st.columns(2)

        with col1:
            csv = pd.DataFrame(predictions).to_csv(index=False)
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "toxtransformer_predictions.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            import json
            json_str = json.dumps(predictions, indent=2)
            st.download_button(
                "📥 Download Full Results (JSON)",
                json_str,
                "toxtransformer_predictions.json",
                "application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()

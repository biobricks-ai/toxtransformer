import streamlit as st
import pandas as pd

st.set_page_config(page_title="ToxTransformer API", page_icon="🧬", layout="wide")

st.title("🧬 ToxTransformer API")
st.markdown("**Decoder-only transformer predicting 6,647 toxicity and ADMET properties**")

st.markdown("---")

# Navigation links
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔬 [Advanced Pathway Analysis →](/pdaa)")
    st.markdown("Explore adverse outcome pathways (AOPs) and molecular initiating events (MIEs) with AI-powered semantic search")
with col2:
    st.markdown("### 📊 [Benchmark Results →](#benchmarks)")
    st.markdown("View internal and external validation metrics")

st.markdown("---")

# API Documentation
st.header("API Documentation")

st.subheader("Endpoints")
st.code("""
GET /health
GET /predict_all?inchi=<InChI>
""", language="bash")

st.subheader("Example Usage")

tab1, tab2 = st.tabs(["Python", "cURL"])

with tab1:
    st.code("""
import requests

# Predict toxicity for aspirin
inchi = "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
response = requests.get(
    "https://toxtransformer.toxindex.com/predict_all",
    params={"inchi": inchi},
    timeout=120
)
predictions = response.json()

# predictions is a list of dicts with keys: inchi, property_token, value
for pred in predictions[:5]:
    print(f"Property {pred['property_token']}: {pred['value']:.4f}")
""", language="python")

with tab2:
    st.code("""
curl "https://toxtransformer.toxindex.com/predict_all?inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
""", language="bash")

st.subheader("Example Molecules")

examples = [
    ("Aspirin", "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"),
    ("Caffeine", "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"),
    ("Ibuprofen", "InChI=1S/C13H18O2/c1-9(2)8-11-4-6-12(7-5-11)10(3)13(14)15/h4-7,9-10H,8H2,1-3H3,(H,14,15)"),
    ("Benzene", "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"),
]

for name, inchi in examples:
    st.code(f"{name}: {inchi}", language="text")

st.markdown("---")

# Benchmarks section
st.header("📊 Benchmarks", anchor="benchmarks")

st.markdown("### 🏆 State-of-the-Art Results")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Tox21", "0.957", "+11.0 vs SOTA")
with col2:
    st.metric("SIDER", "0.762", "+7.2 vs SOTA")
with col3:
    st.metric("BBBP", "0.866", "+14.2 vs SOTA")
with col4:
    st.metric("Properties", "6,647", "1 model")

st.markdown("---")

st.subheader("1. Internal Holdout Validation")
st.markdown("""
**6,378 out-of-sample properties** | **Median AUC: 0.847** | All test compounds unseen during training
""")

internal_data = {
    "Source": ["Tox21", "ICE", "ToxCast", "BindingDB", "ChEMBL", "PubChem", "SIDER"],
    "Properties": ["118", "543", "605", "1,782", "560", "2,664", "26"],
    "Median AUC": ["0.965", "0.934", "0.916", "0.861", "0.840", "0.781", "0.762"],
    "≥ 0.8": ["98%", "87%", "83%", "78%", "66%", "45%", "23%"],
    "≥ 0.7": ["99%", "95%", "95%", "96%", "87%", "72%", "85%"],
}
df_internal = pd.DataFrame(internal_data)
st.dataframe(df_internal, hide_index=True)

st.markdown("**Overall**: 85% of properties exceed AUC 0.70 | 64% exceed AUC 0.80")

st.markdown("---")

st.subheader("2. Published Benchmark Comparison")
st.markdown("**vs MoLFormer-XL (2022)** — Multi-property benchmarks")

published_data = {
    "Dataset": ["Tox21", "SIDER", "BBBP", "BACE", "ClinTox"],
    "Tasks": ["8", "26", "1", "1", "2"],
    "ToxTransformer": ["**0.957**", "**0.762**", "**0.866**", "0.827", "0.786"],
    "Best Published": ["0.847", "0.690", "0.724", "**0.882**", "**0.948**"],
    "Δ": ["**+11.0**", "**+7.2**", "**+14.2**", "-5.5", "-16.2"],
}
df_published = pd.DataFrame(published_data)
st.dataframe(df_published, hide_index=True)

st.markdown("**Wins 3/5** — Largest margins on multi-property benchmarks where multitask architecture excels")

st.markdown("---")

st.subheader("3. External Transfer (Zero-Shot)")
st.markdown("""
**30 curated toxicity endpoints** | **Median AUC: 0.976** | Zero training labels via semantic token matching
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("TT Linked-Token Adapter", "0.976", "median AUC")
    st.metric("Head-to-Head Wins", "15 / 17", "vs Morgan FP")
with col2:
    st.metric("Morgan FP (2048 features)", "0.688", "median AUC")
    st.metric("Δ TT vs FP", "+28.8", "AUC points")

st.markdown("**Perfect Scores (AUC 1.000)**:")
transfer_perfect = {
    "Endpoint": [
        "Zebrafish mortality",
        "Ototoxicity (confident)",
        "ToxCast VDR element",
        "ToxCast AhR activation",
        "Tox21 AR antagonist v2",
        "ToxCast HSE activation",
    ],
    "N": ["20", "20", "10", "18", "12", "10"],
    "FP": ["0.450", "0.800", "—", "—", "—", "—"],
}
df_perfect = pd.DataFrame(transfer_perfect)
st.dataframe(df_perfect, hide_index=True)

st.markdown("**Selected Highlights**:")
transfer_highlights = {
    "Endpoint": [
        "Zebrafish pericardial edema",
        "Zebrafish snout",
        "DILI (drug-induced liver injury)",
        "Ames mutagenicity",
    ],
    "N": ["38", "42", "30", "500"],
    "TT": ["0.988", "0.958", "0.867", "0.803"],
    "FP": ["0.308", "0.360", "0.689", "0.797"],
    "Δ": ["+0.679", "+0.598", "+0.178", "+0.006"],
}
df_highlights = pd.DataFrame(transfer_highlights)
st.dataframe(df_highlights, hide_index=True)

st.markdown("---")

st.subheader("4. ADME Performance (Full-Feature Adapter)")
st.markdown("""
**6,647 TT predictions as features** | Logistic regression with nested CV
""")

adme_data = {
    "Endpoint": ["CYP1A2", "CYP2C19", "CYP2D6", "CYP3A4", "BBB", "CYP2C9"],
    "N": ["212", "170", "288", "500", "500", "182"],
    "Full TT": ["**0.881**", "**0.871**", "0.916", "0.887", "0.901", "0.827"],
    "Morgan FP": ["0.864", "0.813", "**0.919**", "**0.898**", "**0.915**", "**0.899**"],
    "TT+FP": ["0.891", "0.871", "**0.927**", "0.897", "**0.923**", "0.897"],
    "Winner": ["**TT**", "**TT**", "TT+FP", "FP", "TT+FP", "FP"],
}
df_adme = pd.DataFrame(adme_data)
st.dataframe(df_adme, hide_index=True)

st.markdown("**TT beats FP** on CYP1A2 (+1.7) and CYP2C19 (+5.8) | **Combined TT+FP** best on 3/6 endpoints")

st.markdown("---")

st.subheader("5. Context Conditioning")
st.markdown("""
**When you provide known test results, predictions improve monotonically**
Bootstrap (in-sample), 2,227 properties across 11 sources
""")

context_data = {
    "Context Properties": ["0", "5", "10", "20"],
    "Median AUC": ["0.862", "0.879", "0.910", "**0.936**"],
    "Δ vs No Context": ["—", "+1.7", "+4.8", "**+7.4**"],
}
df_context = pd.DataFrame(context_data)
st.dataframe(df_context, hide_index=True)

st.markdown("""
**Confirmed out-of-sample**: Matched holdout set (1,012 properties) improves from 0.942 → 0.964, with 60% of properties gaining.
""")

st.markdown("---")

st.markdown("""
### Citation
If you use ToxTransformer in your research, please cite:
```
ToxTransformer: A decoder-only transformer for multi-task toxicity prediction
Thomas Luechtefeld, et al. (2026)
```
""")

st.markdown("---")
st.caption("Powered by ToxTransformer | Deployed on Google Cloud Platform")

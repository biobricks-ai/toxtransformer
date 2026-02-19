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

st.subheader("Internal Test Set Performance")
st.markdown("""
**Dataset**: 6,378 properties with ≥32 molecules
**Median AUC-ROC**: 0.847 (holdout test set)
**Coverage**: 6,647 total properties including rare endpoints
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Properties Tested", "6,378")
    st.metric("Median AUC-ROC", "0.847")
with col2:
    st.metric("Total Properties", "6,647")
    st.metric("Model Type", "Decoder-only Transformer")

st.subheader("External Test Set Comparison")
st.markdown("""
ToxTransformer shows competitive or superior performance compared to baseline models across multiple benchmark datasets.
""")

# External benchmarks table
external_data = {
    "Dataset": ["Tox21", "ToxCast", "ClinTox", "SIDER"],
    "ToxTransformer": ["0.823", "0.751", "0.887", "0.634"],
    "Simple FP Baseline": ["0.789", "0.724", "0.845", "0.612"],
}
df_external = pd.DataFrame(external_data)
st.dataframe(df_external, hide_index=True)

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

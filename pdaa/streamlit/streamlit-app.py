import json
import time
import streamlit as st
import pandas as pd
import pathlib
from scipy.cluster.hierarchy import linkage, leaves_list
import plotly.graph_objects as go
import stages.utils.pdaa as pdaa
import stages.utils.pubchem as pubchem
import stages.utils.sparql as sparql
import stages.utils.simple_cache as simple_cache
import stages.utils.openai as openai_utils
import rdflib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import requests
import traceback
import sys

# Setup
cachedir = pathlib.Path("cache") / "notebooks" / "pdaa"
cachedir.mkdir(parents=True, exist_ok=True)
pubchemtools = pubchem.PubchemTools()

# Status logger for the sidebar
@dataclass
class StatusLogger:
    """Thread-safe status logger for UI updates."""
    messages: list = field(default_factory=list)

    def log(self, msg: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        self.messages.append({"time": timestamp, "msg": msg, "level": level})

    def info(self, msg: str):
        self.log(msg, "info")

    def success(self, msg: str):
        self.log(msg, "success")

    def warning(self, msg: str):
        self.log(msg, "warning")

    def error(self, msg: str):
        self.log(msg, "error")

    def clear(self):
        self.messages = []


# Utility Functions
@simple_cache.simple_cache_df(cachedir / "get_uri_titles")
def get_uri_titles():
    uri_titles = (
        sparql.Query(pdaa.pdaa_graph, cachedir / "pdaa_graph")
        .select_typed({"uri": str, "title": str})
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/title> ?title")
        .where(f"?ppuri <{rdflib.RDF.type}> toxindex:predicted_property")
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/has_identifier> ?uri")
        .cache_execute()
        .groupby("uri")
        .first()
        .reset_index()
    )
    # Ensure unique titles
    for title in uri_titles["title"].value_counts()[uri_titles["title"].value_counts() > 1].index:
        mask = uri_titles["title"] == title
        uri_titles.loc[mask, "title"] = [f"{title}_{i+1}" for i in range(sum(mask))]
    return uri_titles
uri_titles = get_uri_titles()

all_ao = sparql.Query(pdaa.pdaa_graph, cachedir / 'pdaa_graph') \
    .select_typed({'aop':str, 'mie':str, 'ao':str, 'ao_title':str, 'mie_title':str}) \
    .where(f'?aop <{rdflib.RDF.type}> aop:AdverseOutcomePathway') \
    .where(f'?aop aop:has_adverse_outcome ?ao') \
    .where(f'?aop aop:has_molecular_initiating_event ?mie') \
    .where(f'?ao <http://purl.org/dc/elements/1.1/title> ?ao_title') \
    .where(f'?mie <http://purl.org/dc/elements/1.1/title> ?mie_title') \
    .cache_execute()

@simple_cache.simple_cache_df(cachedir / "link_predicted_properties_to_mies")
def link_predicted_properties_to_mies():
    predicted_property_identifiers = pdaa.proptoken_uris['uri'].unique()
    mies = all_ao['mie'].unique()
    mie_proptoken_id_simtable = pdaa.get_uri_similars(mies, predicted_property_identifiers)
    mie_proptoken_id_simtable.columns = ['mie', 'property_token_id_uri', 'similarity']

    resdf = (mie_proptoken_id_simtable
             .sort_values('similarity', ascending=False)
             .groupby('mie')
             .head(3)
             .reset_index(drop=True))
    return resdf

mie_proptoken_id_simtable = link_predicted_properties_to_mies()

def parse_chemical_list(chemical_list) -> dict[str, str]:
    chemicals = {}
    for line in filter(None, map(str.strip, chemical_list.split("\n"))):
        alias, name = line.split(":", 1) if ":" in line else (line, line)
        chemicals[alias.strip()] = name.strip()
    return chemicals


def robust_lookup_inchi(chemical_name: str, logger: StatusLogger) -> Optional[str]:
    """
    Robust chemical name to InChI lookup with multiple fallback strategies.

    Tries:
    1. Direct name lookup
    2. Name with spaces replaced by hyphens
    3. Name without common prefixes/suffixes
    4. CAS number lookup if it looks like a CAS
    """
    strategies = [
        ("direct", chemical_name),
        ("hyphenated", chemical_name.replace(" ", "-")),
        ("lowercase", chemical_name.lower()),
        ("uppercase", chemical_name.upper()),
    ]

    # Add stripped versions for common patterns
    name_lower = chemical_name.lower()
    for prefix in ["di-", "mono-", "bis-", "tris-"]:
        if name_lower.startswith(prefix):
            strategies.append(("no-prefix", chemical_name[len(prefix):]))

    for strategy_name, name in strategies:
        try:
            inchi = pubchemtools.lookup_chemical_inchi(name)
            if inchi:
                if strategy_name != "direct":
                    logger.info(f"  Found '{chemical_name}' using {strategy_name} strategy")
                return inchi
        except Exception:
            continue

    # Try as CAS number (format: XXXXX-XX-X)
    import re
    if re.match(r'^\d{2,7}-\d{2}-\d$', chemical_name):
        try:
            # PubChem can search by CAS
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{chemical_name}/cids/JSON"
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                cid = response.json()['IdentifierList']['CID'][0]
                inchi_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/InChI/JSON"
                response = requests.get(inchi_url, timeout=10)
                if response.status_code == 200:
                    return response.json()['PropertyTable']['Properties'][0]['InChI']
        except Exception:
            pass

    return None


def build_heatmap(predictions, midvalue=0.5, max_val=None):
    # Pivot the predictions into a matrix
    pdf = predictions.pivot(index='title', columns='chemical_name', values='prediction')

    # Perform hierarchical clustering on rows and columns
    row_linkage = linkage(pdf.values, method='ward', metric='euclidean')
    col_linkage = linkage(pdf.T.values, method='ward', metric='euclidean')

    # Get the order of rows and columns based on clustering
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    # Reorder the DataFrame
    pdf_clustered = pdf.iloc[row_order, col_order]

    # Create clickable row labels using the 'uri' column
    row_links = dict(zip(predictions['title'], predictions['uri']))
    row_labels = [f'<a href="{row_links[title]}" target="_blank">{title}</a>' if title in row_links else title for title in pdf_clustered.index]

    # Get min and max values for colorscale
    min_val = pdf_clustered.values.min()
    max_val = pdf_clustered.values.max() if max_val is None else max_val

    # rescale z so that if it is bigger than max_val, it is set to max_val
    pdf_clustered_clipped = pdf_clustered.clip(min_val, max_val)

    colorscale = [[0, 'rgb(0,0,255)'],[1, 'rgb(255,0,0)']]
    heatmap = go.Heatmap(
        z=pdf_clustered_clipped.values,
        x=pdf_clustered_clipped.columns,
        y=row_labels,
        colorscale=colorscale,
        zmin=min_val,
        zmax=max_val,
        colorbar=dict(title="", orientation="v", x=-0.2, y=0.5, thickness=20),
        xgap=2,
        ygap=2,
        showscale=True,
        text=[[f'{val:.2f}' for val in row] for row in pdf_clustered.values],
        texttemplate='%{text}',
        textfont={"color": "white"}
    )

    layout = go.Layout(
        title=f'Property Predictions Heatmap',
        xaxis=dict(title="Chemical Names", gridcolor='black', showgrid=False),
        yaxis=dict(
            title="Titles",
            tickmode="array",
            tickvals=list(range(len(pdf.index))),
            ticktext=row_labels,
            side='right',
            gridcolor='black',
            showgrid=False,
            tickfont=dict(size=16),
            tickprefix='   ',
            ticksuffix='     '
        ),
        height=800,
        plot_bgcolor='black',
        margin=dict(r=350)
    )

    return go.Figure(data=[heatmap], layout=layout)

def build_barchart(predictions, title):
    fig = go.Figure()

    data = predictions.groupby('chemical_name')['prediction'].sum().sort_values(ascending=False)

    fig.add_trace(go.Bar(
        x=data.index,
        y=data.values,
        marker=dict(
            color=data.values,
            colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']],
            line=dict(color='white', width=1)
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Chemical',
        yaxis_title='Sum of Predictions',
        xaxis_tickangle=45,
        yaxis_gridcolor='white',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        height=400,
        xaxis=dict(gridcolor='white', showgrid=True),
        yaxis=dict(gridcolor='white', showgrid=True)
    )

    return fig


def get_all_predictions_with_status(chemical_list: str, logger: StatusLogger, progress_placeholder):
    """Get predictions for all chemicals with status updates."""

    chemicals = parse_chemical_list(chemical_list)
    logger.info(f"Processing {len(chemicals)} chemicals...")

    # Step 1: Resolve chemical names to InChI
    logger.info("Step 1: Resolving chemical names...")
    inchi_map = {}  # alias -> inchi
    failed_chemicals = []

    for i, (alias, name) in enumerate(chemicals.items()):
        progress_placeholder.progress((i + 1) / len(chemicals), text=f"Looking up: {name}")

        inchi = robust_lookup_inchi(name, logger)
        if inchi:
            inchi_map[alias] = inchi
            logger.success(f"  ✓ {alias}: found InChI")
        else:
            failed_chemicals.append((alias, name))
            logger.warning(f"  ✗ {alias}: '{name}' not found in PubChem")

    if failed_chemicals:
        logger.warning(f"Skipping {len(failed_chemicals)} unresolved chemicals")

    if not inchi_map:
        logger.error("No valid chemicals found!")
        return None

    logger.info(f"Resolved {len(inchi_map)}/{len(chemicals)} chemicals")

    # Step 2: Get predictions
    logger.info("Step 2: Getting predictions from ToxTransformer...")

    inchi_list = list(inchi_map.values())
    alias_list = list(inchi_map.keys())
    inchi2alias = dict(zip(inchi_list, alias_list))

    all_predictions = []

    for i, inchi in enumerate(inchi_list):
        alias = inchi2alias[inchi]
        progress_placeholder.progress((i + 1) / len(inchi_list), text=f"Predicting: {alias}")

        try:
            logger.info(f"  Submitting job for {alias}...")
            start_time = time.time()

            # Use the prediction function that handles job queue
            preds = pdaa.predict_all_properties_with_sqlite_cache([inchi])

            elapsed = time.time() - start_time
            logger.success(f"  ✓ {alias}: {len(preds)} predictions ({elapsed:.1f}s)")
            all_predictions.extend(preds)

        except Exception as e:
            logger.error(f"  ✗ {alias}: prediction failed - {str(e)[:50]}")
            continue

    if not all_predictions:
        logger.error("No predictions obtained!")
        return None

    # Build DataFrame
    preds_df = pd.DataFrame(all_predictions, columns=['inchi', 'token', 'prediction'])
    preds_df = preds_df.merge(pdaa.proptoken_uris[['uri','token']], on='token')[['uri','inchi','prediction']]
    preds_df.rename(columns={'uri':'property_token_id_uri'}, inplace=True)
    preds_df["chemical_name"] = preds_df["inchi"].map(inchi2alias)

    logger.success(f"Complete! {len(preds_df)} total predictions for {preds_df['chemical_name'].nunique()} chemicals")

    return preds_df


@simple_cache.simple_cache_df(cachedir / "get_property_predictions")
def get_property_predictions(predictions, prompt, top_n=20):
    uri_candidates = pdaa.proptoken_uris
    if '"' in prompt:
        quoted_words = prompt.split('"')[1::2]
        uri_candidates = uri_candidates[uri_candidates['title'].str.lower().apply(lambda x: any(word.lower() in x for word in quoted_words))]

    # Use Gemini Flash to rank properties by relevance (replaces FAISS embedding search)
    prompt_property = openai_utils.rank_properties_gemini(prompt, uri_candidates[['uri', 'title']], top_n=30)

    res = predictions[['property_token_id_uri','chemical_name','prediction']].drop_duplicates()
    res = res[res['property_token_id_uri'].isin(prompt_property['uri'])]
    res = res.merge(uri_titles, left_on='property_token_id_uri', right_on='uri')
    res = res[['uri','title','chemical_name','prediction']]

    return res[['uri','title','chemical_name','prediction']]

@simple_cache.simple_cache_df(cachedir / "get_mie_predictions")
def get_mie_predictions(predictions, prompt, top_n=20):

    all_pred_df = predictions.merge(mie_proptoken_id_simtable, on='property_token_id_uri')
    all_pred_df['weight'] = all_pred_df['similarity'] * all_pred_df['prediction']
    all_pred_df = all_pred_df[['mie','property_token_id_uri','chemical_name','similarity','prediction','weight']]

    candidate_ao = all_ao
    if '"' in prompt:
        quoted_words = prompt.split('"')[1::2]
        candidate_ao = candidate_ao[candidate_ao['mie_title'].str.lower().apply(lambda x: any(word.lower() in x for word in quoted_words))]

    # Rank AOPs using Gemini (needs AOP title - we'll use MIE title as proxy since AOP doesn't have separate title)
    aop_df = candidate_ao[['aop', 'mie_title']].drop_duplicates().rename(columns={'aop': 'uri', 'mie_title': 'title'})
    prompt_aop = openai_utils.rank_uris_gemini(prompt, aop_df, uri_col='uri', title_col='title', top_k=10000)
    prompt_aop = prompt_aop.query('similarity > 0.05')  # Lower threshold
    prompt_aop = all_ao.merge(prompt_aop, left_on='aop', right_on='uri')[['mie','mie_title','similarity']]

    # Rank AOs using Gemini
    ao_df = candidate_ao[['ao', 'ao_title']].drop_duplicates().rename(columns={'ao': 'uri', 'ao_title': 'title'})
    prompt_ao = openai_utils.rank_uris_gemini(prompt, ao_df, uri_col='uri', title_col='title', top_k=10000)
    prompt_ao = prompt_ao.query('similarity > 0.05')  # Lower threshold
    prompt_ao = all_ao.merge(prompt_ao, left_on='ao', right_on='uri')[['mie','mie_title','similarity']]

    # Rank MIEs using Gemini - this is the primary signal
    mie_df = candidate_ao[['mie', 'mie_title']].drop_duplicates().rename(columns={'mie': 'uri', 'mie_title': 'title'})
    prompt_mie = openai_utils.rank_uris_gemini(prompt, mie_df, uri_col='uri', title_col='title', top_k=10000)

    # Use MIE scores directly, but boost if AOP or AO also match
    if len(prompt_aop) > 0 or len(prompt_ao) > 0:
        # Merge with AO and AOP scores if available
        prompt_mie_with_context = all_ao.merge(prompt_mie, left_on='mie', right_on='uri', how='inner')[['mie','mie_title','similarity']]

        if len(prompt_ao) > 0:
            prompt_mie_with_context = prompt_mie_with_context.merge(
                prompt_ao.rename(columns={'similarity': 'ao_similarity'}),
                on='mie', how='left'
            )
            prompt_mie_with_context['ao_similarity'] = prompt_mie_with_context['ao_similarity'].fillna(0)
            prompt_mie_with_context['similarity'] = prompt_mie_with_context[['similarity', 'ao_similarity']].max(axis=1)

        if len(prompt_aop) > 0:
            prompt_mie_with_context = prompt_mie_with_context.merge(
                prompt_aop.rename(columns={'similarity': 'aop_similarity'}),
                on='mie', how='left'
            )
            prompt_mie_with_context['aop_similarity'] = prompt_mie_with_context['aop_similarity'].fillna(0)
            prompt_mie_with_context['similarity'] = prompt_mie_with_context[['similarity', 'aop_similarity']].max(axis=1)

        prompt_mie = prompt_mie_with_context[['mie', 'mie_title', 'similarity']]
    else:
        # Just use MIE scores directly
        prompt_mie = all_ao.merge(prompt_mie, left_on='mie', right_on='uri')[['mie','mie_title','similarity']]

    prompt_mie = prompt_mie.groupby(['mie','mie_title']).agg({'similarity':'max'}).reset_index()

    pred_mie = all_pred_df['mie'].unique()
    prompt_mie = prompt_mie[prompt_mie['mie'].isin(pred_mie)].sort_values('similarity', ascending=False).head(top_n)
    prompt_mie = prompt_mie[['mie','mie_title']]

    mie_predictions = all_pred_df[['mie','property_token_id_uri','chemical_name','prediction']].merge(prompt_mie, on=['mie'])
    mie_predictions = mie_predictions[['mie','mie_title','property_token_id_uri','chemical_name','prediction']]

    mie_df = mie_predictions.groupby(['mie','mie_title','chemical_name'])['prediction'].mean().reset_index()

    mie_weights = mie_df[['mie','mie_title','chemical_name','prediction']]
    mie_weights.rename(columns={'mie':'uri','mie_title':'title'}, inplace=True)

    title_df = mie_weights[['title']].drop_duplicates()
    title_df['short_title'] = title_df['title'].str[:40]
    dup_mask = title_df.duplicated(subset=['short_title'], keep=False)
    title_df.loc[dup_mask, 'short_title'] += ' ' + title_df[dup_mask].groupby('short_title').cumcount().add(1).astype(str)

    mie_weights_title = mie_weights.merge(title_df, on='title')
    mie_weights_title['title'] = mie_weights_title['short_title']
    mie_weights_title.drop('short_title', axis=1, inplace=True)

    return mie_weights_title[['uri','title','chemical_name','prediction']]


def render_status_log(logger: StatusLogger, container):
    """Render the status log in a container."""
    with container:
        st.markdown("### 📋 Status Log")

        if not logger.messages:
            st.markdown("*Waiting for action...*")
            return

        # Build log display
        log_html = ['<div style="font-family: monospace; font-size: 12px; background: #1e1e1e; padding: 10px; border-radius: 5px; max-height: 500px; overflow-y: auto;">']

        for msg in logger.messages[-50:]:  # Show last 50 messages
            color = {
                "info": "#888",
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#f44336"
            }.get(msg["level"], "#888")

            icon = {
                "info": "ℹ️",
                "success": "✅",
                "warning": "⚠️",
                "error": "❌"
            }.get(msg["level"], "")

            log_html.append(f'<div style="color: {color}; margin: 2px 0;">[{msg["time"]}] {icon} {msg["msg"]}</div>')

        log_html.append('</div>')
        st.markdown(''.join(log_html), unsafe_allow_html=True)


# ============== UI Setup ==============
st.set_page_config(page_title="PDAA - Phthalates Analysis", layout="wide")

# Initialize session state
if 'logger' not in st.session_state:
    st.session_state.logger = StatusLogger()
if 'chemical_list' not in st.session_state:
    examples = json.load(open('streamlit/examples.json'))['examples']
    st.session_state.chemical_list = examples[5]['chemicals']
if 'prompt' not in st.session_state:
    examples = json.load(open('streamlit/examples.json'))['examples']
    st.session_state.prompt = examples[5]['prompt']
if 'updating_charts' not in st.session_state:
    st.session_state.updating_charts = False

# Load examples
examples = json.load(open('streamlit/examples.json'))['examples']

# Create two-column layout: main content (left) and status log (right)
main_col, status_col = st.columns([3, 1])

with main_col:
    st.title("🧪 PDAA: Phthalates Data Analysis")
    st.markdown("*Find & Rank Adverse Outcomes for Chemicals*")

    # Input section
    chemical_list = st.text_area(
        "Enter chemicals (format: `alias:name` or just `name`)",
        st.session_state.chemical_list,
        height=150,
        help="Each line is a chemical. Use 'alias:name' format for custom labels, or just the chemical name."
    )

    prompt = st.text_input(
        "Enter analysis prompt",
        st.session_state.prompt,
        help="Describe the adverse outcomes you're interested in (e.g., 'endocrine disruption', 'liver toxicity')"
    )

    # Example buttons
    st.markdown("**Quick examples:**")
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(example['title'], key=f"example_button_{i}", disabled=st.session_state.updating_charts):
            st.session_state.prompt = example['prompt']
            st.session_state.chemical_list = example['chemicals']
            st.session_state.logger.clear()
            st.rerun()

    st.markdown("---")

    # Generate button and progress
    generate_clicked = st.button("🚀 Generate Heatmaps", disabled=st.session_state.updating_charts, type="primary")
    progress_placeholder = st.empty()

    if generate_clicked:
        st.session_state.updating_charts = True
        st.session_state.logger.clear()
        st.session_state.logger.info("Starting analysis...")

        try:
            # Get predictions with status updates
            with status_col:
                status_container = st.container()

            predictions = get_all_predictions_with_status(
                chemical_list,
                st.session_state.logger,
                progress_placeholder
            )

            if predictions is not None and len(predictions) > 0:
                progress_placeholder.empty()

                # Generate heatmaps
                st.session_state.logger.info("Generating property heatmap...")
                st.subheader("Property Heatmap")
                property_predictions = get_property_predictions(predictions, prompt)

                if len(property_predictions) > 0:
                    fig = build_heatmap(property_predictions)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.logger.success("Property heatmap generated")
                else:
                    st.warning("No property predictions matched your prompt.")
                    st.session_state.logger.warning("No property predictions matched prompt")

                st.session_state.logger.info("Generating MIE heatmap...")
                st.subheader("MIE Heatmap")
                mie_predictions = get_mie_predictions(predictions, prompt)

                if len(mie_predictions) > 0:
                    fig = build_heatmap(mie_predictions)
                    st.plotly_chart(fig, use_container_width=True)
                    st.session_state.logger.success("MIE heatmap generated")

                    st.subheader("Aggregate Predictions")
                    aggregate_predictions = mie_predictions.groupby(['chemical_name'])['prediction'].sum().reset_index()
                    fig = build_barchart(aggregate_predictions, title="Aggregate MIE Predictions")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No MIE predictions matched your prompt.")
                    st.session_state.logger.warning("No MIE predictions matched prompt")

                st.session_state.logger.success("Analysis complete!")
            else:
                st.error("No predictions were obtained. Check the status log for details.")

        except Exception as e:
            error_msg = f"Fatal error: {type(e).__name__}: {str(e)}"
            full_traceback = traceback.format_exc()

            # Log to stderr for Cloud Logging
            print(f"ERROR: {error_msg}", file=sys.stderr)
            print(f"Full traceback:\n{full_traceback}", file=sys.stderr)

            # Show in UI
            st.error(f"Error: {str(e)}")
            st.session_state.logger.error(error_msg)

        st.session_state.updating_charts = False

# Status log sidebar (always visible)
with status_col:
    render_status_log(st.session_state.logger, st.container())

    # Add helpful info
    st.markdown("---")
    st.markdown("### ℹ️ Tips")
    st.markdown("""
    - Use **exact PubChem names** for best results
    - Try **CAS numbers** (e.g., 117-81-7)
    - Check [PubChem](https://pubchem.ncbi.nlm.nih.gov/) for valid names
    - Use **quotes** in prompts for exact matches
    """)

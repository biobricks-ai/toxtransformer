import sys
import dotenv
import stages.utils.simple_cache as simple_cache
import pathlib
import pandas as pd
import json
import os
dotenv.load_dotenv()

cache_dir = pathlib.Path('cache/util/openai') / 'cache'
cache_dir.mkdir(parents=True, exist_ok=True)

@simple_cache.simple_cache(cache_dir / 'rerank')
def rerank(prompt, values):
    """Rerank values by relevance to prompt using Gemini Flash."""
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'toxindex')
        vertexai.init(project=project_id, location='us-central1')
        
        model = GenerativeModel("gemini-2.0-flash")
        
        system_prompt = """You are a helpful assistant that evaluates how strongly items relate to a given topic.
For each item, assign a strength score from 1-10 where:
- 10 means extremely relevant/strong relationship 
- 1 means minimal/weak relationship

Return JSON only in this format:
{"table": [{"value": "item1", "strength": 8}, {"value": "item2", "strength": 5}, ...]}"""

        user_prompt = f"""Topic: {prompt}

Items to evaluate:
{chr(10).join(values)}"""

        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": 0}
        )
        
        result = json.loads(response.text)
        return result
    except Exception as e:
        # Fallback: return all items with neutral score
        return {"table": [{"value": v, "strength": 5} for v in values]}

@simple_cache.simple_cache(cache_dir / 'link_molecular_terms')
def link_molecular_terms(assays, mie):
    """Link assays to molecular initiating events using Gemini Flash."""
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai

        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'toxindex')
        vertexai.init(project=project_id, location='us-central1')

        model = GenerativeModel("gemini-2.0-flash")

        system_prompt = """You are a helpful assistant that evaluates how well different assays can measure a molecular initiating event (MIE).
For each assay, determine how relevant it is for measuring the given MIE on a scale of 1-10:
- 10: Assay directly and specifically measures the MIE
- 7-9: Assay closely related to measuring the MIE
- 4-6: Assay indirectly measures or is moderately related to the MIE
- 1-3: Assay has minimal or tangential relationship to measuring the MIE

Return JSON only in this format:
{"matches": [{"assay": "assay1", "relevance_score": 8, "explanation": "..."}, ...]}"""

        user_prompt = f"""Molecular Initiating Event (MIE): {mie}

Available Assays:
{chr(10).join(assays)}

For each assay, rate how well it measures this MIE and explain your reasoning."""

        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": 0}
        )

        result = json.loads(response.text)
        return result
    except Exception as e:
        # Fallback: return empty matches
        return {"matches": []}

def score_properties_batch(prompt, property_titles):
    """Score a batch of properties using Gemini Flash."""
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai

        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'toxindex')
        vertexai.init(project=project_id, location='us-central1')

        model = GenerativeModel("gemini-2.0-flash")

        system_prompt = """You are a toxicology expert evaluating how well different biological assays/properties relate to a specific topic.

For each numbered property, assign a relevance score from 0-10 where:
- 10: Extremely relevant, directly measures the concept
- 7-9: Highly relevant, closely related
- 4-6: Moderately relevant, indirect relationship
- 1-3: Weakly relevant, tangential connection
- 0: Not relevant

Return ONLY valid JSON in this exact format (use the index number, not the title):
{"scores": [{"index": 1, "score": 8}, {"index": 2, "score": 3}, ...]}
You MUST include every index from 1 to N in your response."""

        # Format properties as numbered list
        property_list = "\n".join([f"{i+1}. {title}" for i, title in enumerate(property_titles)])

        user_prompt = f"""Topic: {prompt}

Properties to evaluate:
{property_list}

Rate each property's relevance to the topic. Return index numbers, not titles."""

        response = model.generate_content(
            [system_prompt, user_prompt],
            generation_config={"temperature": 0, "response_mime_type": "application/json"}
        )

        result = json.loads(response.text)
        return result.get("scores", [])
    except Exception as e:
        # Fallback: return zero scores by index
        return [{"index": i + 1, "score": 0} for i in range(len(property_titles))]

@simple_cache.simple_cache_df(cache_dir / 'rank_properties_gemini')
def rank_properties_gemini(prompt, properties_df, top_n=30):
    """Rank properties by relevance using parallel Gemini Flash requests.

    Args:
        prompt: User's search query
        properties_df: DataFrame with 'uri' and 'title' columns
        top_n: Number of top properties to return

    Returns:
        DataFrame with 'uri' and 'relevance_score' columns, sorted by score
    """
    import concurrent.futures
    import numpy as np

    if len(properties_df) == 0:
        return pd.DataFrame(columns=['uri', 'relevance_score'])

    # Split properties into batches of 100 for reliable Gemini JSON output
    batch_size = 100
    num_workers = 10

    batches = []
    for i in range(0, len(properties_df), batch_size):
        batch = properties_df.iloc[i:i+batch_size]
        batches.append(batch)

    # Score batches in parallel
    all_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(score_properties_batch, prompt, batch['title'].tolist())
            futures.append((future, batch))

        for future, batch in futures:
            try:
                scores = future.result(timeout=30)
            except Exception:
                scores = [{"index": i + 1, "score": 0} for i in range(len(batch))]
            # Match scores back to URIs by index (not title string matching)
            for score_obj in scores:
                idx = score_obj.get('index', 0) - 1  # convert to 0-based
                score = score_obj.get('score', 0)
                if 0 <= idx < len(batch):
                    all_scores.append({
                        'uri': batch.iloc[idx]['uri'],
                        'relevance_score': float(score) / 10.0  # Normalize to 0-1
                    })

    # Sort by score and return top N
    result_df = pd.DataFrame(all_scores)
    if len(result_df) == 0:
        return pd.DataFrame(columns=['uri', 'relevance_score'])

    result_df = result_df.sort_values('relevance_score', ascending=False).head(top_n)
    return result_df

@simple_cache.simple_cache(cache_dir / 'rank_uris_gemini')
def _rank_uris_gemini_impl(prompt, uri_title_pairs, top_k=100):
    """Internal implementation for ranking URIs (for caching)."""
    import concurrent.futures
    import numpy as np

    if len(uri_title_pairs) == 0:
        return []

    # Split into batches of 100 for reliable Gemini JSON output
    batch_size = 100
    num_workers = 10

    batches = []
    for i in range(0, len(uri_title_pairs), batch_size):
        batch = uri_title_pairs[i:i+batch_size]
        batches.append(batch)

    # Score batches in parallel
    all_scores = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for batch in batches:
            titles = [title for uri, title in batch]
            future = executor.submit(score_properties_batch, prompt, titles)
            futures.append((future, batch))

        for future, batch in futures:
            try:
                scores = future.result(timeout=30)
            except Exception:
                scores = [{"index": i + 1, "score": 0} for i in range(len(batch))]
            # Match scores back to URIs by index (not title string matching)
            for score_obj in scores:
                idx = score_obj.get('index', 0) - 1  # convert to 0-based
                score = score_obj.get('score', 0)
                if 0 <= idx < len(batch):
                    uri, batch_title = batch[idx]
                    all_scores.append({
                        'uri': uri,
                        'similarity': float(score) / 10.0
                    })

    # Sort by score and return top k
    all_scores.sort(key=lambda x: x['similarity'], reverse=True)
    return all_scores[:top_k]

def rank_uris_gemini(prompt, uris_df, uri_col='uri', title_col='title', top_k=100):
    """Rank URIs by relevance using Gemini Flash (for AOPs, MIEs, AOs).

    Args:
        prompt: User's search query
        uris_df: DataFrame containing URIs and their titles
        uri_col: Column name containing URIs
        title_col: Column name containing titles
        top_k: Maximum number of results to return

    Returns:
        DataFrame with 'uri' and 'similarity' columns (similarity 0-1), sorted by score
    """
    if len(uris_df) == 0:
        return pd.DataFrame(columns=['uri', 'similarity'])

    # Extract unique uri-title pairs as tuples for better caching
    unique_pairs = uris_df[[uri_col, title_col]].drop_duplicates()
    uri_title_tuples = [(row[uri_col], row[title_col]) for _, row in unique_pairs.iterrows()]
    uri_title_tuples = tuple(sorted(set(uri_title_tuples)))  # Sort for deterministic caching

    # Use cached implementation
    results = _rank_uris_gemini_impl(prompt, uri_title_tuples, top_k)

    # Convert list of dicts to DataFrame
    return pd.DataFrame(results)

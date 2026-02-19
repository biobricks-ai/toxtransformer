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

# Lazy client initialization - only create when actually needed
_client = None

def _get_client():
    """Get Vertex AI client for embeddings."""
    global _client
    if _client is None:
        try:
            from google.cloud import aiplatform
            from vertexai.language_models import TextEmbeddingModel
            
            # Initialize Vertex AI (uses Application Default Credentials in Cloud Run)
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'toxindex')
            aiplatform.init(project=project_id, location='us-central1')
            
            _client = TextEmbeddingModel.from_pretrained("text-embedding-004")
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Vertex AI: {e}\n"
                "Semantic search features require Vertex AI access."
            )
    return _client

@simple_cache.simple_cache(cache_dir / 'embed')
def embed(text):
    """Generate embeddings using Vertex AI text-embedding-004."""
    client = _get_client()
    embeddings = client.get_embeddings([text])
    return embeddings[0].values

@simple_cache.simple_cache(cache_dir / 'rerank')
def rerank(prompt, values):
    """Rerank values by relevance to prompt using Gemini Flash."""
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'toxindex')
        vertexai.init(project=project_id, location='us-central1')
        
        model = GenerativeModel("gemini-2.0-flash-exp")
        
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
        
        model = GenerativeModel("gemini-2.0-flash-exp")
        
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

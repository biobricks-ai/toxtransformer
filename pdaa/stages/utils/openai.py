import sys
import dotenv
import openai
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
    global _client
    if _client is None:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError(
                "OpenAI API key not set. Set OPENAI_API_KEY environment variable "
                "to use semantic search features."
            )
        _client = openai.OpenAI(api_key=api_key)
    return _client

@simple_cache.simple_cache(cache_dir / 'embed')
def embed(text):
    client = _get_client()
    return client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding

@simple_cache.simple_cache(cache_dir / 'rerank')
def rerank(prompt, values):
    # Define JSON schema for response validation
    json_schema = {
        "name": "rerank",
        "schema": {
            "type": "object",
            "properties": {
                "table": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "strength": {
                                "type": "integer"
                            }
                        },
                        "required": ["value", "strength"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["table"],
            "additionalProperties": False
        },
        "strict": True
    }

    # Create structured prompt for reranking
    system_prompt = """You are a helpful assistant that evaluates how strongly items relate to a given topic.
For each item, assign a strength score from 1-10 where:
- 10 means extremely relevant/strong relationship 
- 1 means minimal/weak relationship"""

    user_prompt = f"""Topic: {prompt}

Items to evaluate:
{chr(10).join(values)}"""

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    
    result = json.loads(response.choices[0].message.content)
    return result

@simple_cache.simple_cache(cache_dir / 'link_molecular_terms')
def link_molecular_terms(assays, mie):
    # Define JSON schema for response validation
    json_schema = {
        "name": "assay_mie_link",
        "schema": {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "assay": {"type": "string"},
                            "relevance_score": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10
                            },
                            "explanation": {"type": "string"}
                        },
                        "required": ["assay", "relevance_score", "explanation"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["matches"],
            "additionalProperties": False
        },
        "strict": True
    }

    system_prompt = """You are a helpful assistant that evaluates how well different assays can measure a molecular initiating event (MIE).
For each assay, determine how relevant it is for measuring the given MIE on a scale of 1-10:
- 10: Assay directly and specifically measures the MIE
- 7-9: Assay closely related to measuring the MIE
- 4-6: Assay indirectly measures or is moderately related to the MIE
- 1-3: Assay has minimal or tangential relationship to measuring the MIE

Provide a brief explanation for each score."""

    user_prompt = f"""Molecular Initiating Event (MIE): {mie}

Available Assays:
{chr(10).join(assays)}

For each assay, rate how well it measures this MIE and explain your reasoning."""

    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4",
        response_format={
            "type": "json_schema",
            "json_schema": json_schema
        },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    result = json.loads(response.choices[0].message.content)
    return result
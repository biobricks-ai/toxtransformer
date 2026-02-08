# ToxJobs MCP Server: Required Property Metadata Enhancement

## Problem Statement

The current toxjobs MCP server `toxtransformer` tool returns predictions as a **flat array of floats** without any property metadata. This makes it impossible to:

1. Know which property each prediction corresponds to
2. Match predictions to external evaluation endpoints
3. Understand what is being predicted
4. Use predictions for scientific validation

### Current Response Structure (WRONG)

```json
{
  "tool": "toxtransformer",
  "success": true,
  "result": {
    "smiles": "CCO",
    "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    "predictions": [
      0.9058858156204224,
      0.9584697484970093,
      0.877509593963623,
      ...  // 6647 floats with NO context
    ]
  }
}
```

### Required Response Structure (CORRECT)

The Flask API at `http://136.111.102.10:6515/predict_all` already returns the correct format:

```json
[
  {
    "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    "property_token": 0,
    "property": {
      "property_token": 0,
      "source": "pubchem",
      "title": "identification of small-molecule activators for gpr151 receptor in human cell-based assay",
      "metadata": {
        "BioAssay Name": "...",
        "Source ID": "...",
        "Deposit Date": "...",
        ...
      },
      "categories": [
        {
          "category": "other",
          "reason": "...",
          "strength": "7.5"
        }
      ]
    },
    "value": 0.9058858156204224
  },
  {
    "inchi": "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
    "property_token": 1,
    "property": {
      "property_token": 1,
      "source": "pubchem",
      "title": "high-throughput biochemical assay for mitf inhibition detection",
      ...
    },
    "value": 0.9584697484970093
  },
  ...
]
```

## Required Changes to toxjobs MCP Server

### Option 1: Full Metadata (Recommended)

Update the `toxtransformer` tool to return the same structure as the Flask API:

```python
# In toxjobs MCP server code
def run_toxtransformer(smiles: str, property_token: Optional[int] = None):
    """Run ToxTransformer predictions with full property metadata."""

    # Convert SMILES to InChI
    inchi = smiles_to_inchi(smiles)

    # Call Flask API
    if property_token:
        response = requests.get(f"{API_URL}/predict?inchi={quote(inchi)}&property_token={property_token}")
    else:
        response = requests.get(f"{API_URL}/predict_all?inchi={quote(inchi)}")

    predictions = response.json()

    # Return full structure
    return {
        "smiles": smiles,
        "inchi": inchi,
        "success": True,
        "predictions": predictions  # Full objects with metadata!
    }
```

### Option 2: Minimal Metadata

If bandwidth is a concern, return at minimum:

```json
{
  "smiles": "CCO",
  "inchi": "InChI=1S/...",
  "predictions": [
    {
      "property_token": 0,
      "title": "GPR151 receptor activators",
      "source": "pubchem",
      "value": 0.905
    },
    ...
  ]
}
```

### Option 3: Separate Metadata Endpoint

Add a new tool to retrieve property metadata:

```python
@tool
def toxtransformer_properties():
    """List all ToxTransformer property tokens with metadata."""
    return {
        "properties": [
            {
                "property_token": 0,
                "title": "...",
                "source": "...",
                "categories": [...]
            },
            ...
        ]
    }
```

## Why This Matters

### External Evaluation Requires Property Matching

To validate ToxTransformer against external datasets, we must match:
- External endpoint (e.g., "CYP1A2 inhibition")
- ToxTransformer property token (e.g., token 1610: "assessment of cyp1a2 antagonists")

Without property tokens in the response, we cannot:
1. Find predictions for specific endpoints
2. Calculate performance metrics
3. Verify semantic matching

### Current Workaround

The `scripts/run_semantic_evaluation.py` script calls the Flask API directly instead of using the MCP server because the MCP server lacks property metadata.

### Key Property Categories Needed

At minimum, the following metadata should be included:

| Field | Purpose |
|-------|---------|
| `property_token` | Unique identifier for matching |
| `title` | Human-readable description |
| `source` | Data source (pubchem, chembl, tox21, etc.) |
| `categories` | Toxicity category classification |

## Implementation Priority

1. **Immediate**: Return `property_token` with each prediction value
2. **Short-term**: Add `title` and `source`
3. **Long-term**: Include full metadata and categories

## Testing

After implementing, verify with:

```python
result = toxjobs_run(tool="toxtransformer", smiles="CCO")
predictions = result["predictions"]

# Should work:
assert isinstance(predictions[0], dict)
assert "property_token" in predictions[0]
assert "value" in predictions[0]
assert "title" in predictions[0]
```

## Files to Update

1. toxjobs MCP server source (location TBD - not in this repo)
2. jobs.toxindex.com backend handler for `toxtransformer`

## Contact

If you need access to the Flask API response format for reference:
- API endpoint: `http://136.111.102.10:6515/predict_all?inchi=<inchi>`
- Source code: `/mnt/ssd/git/chemprop-transformer/flask_cvae/predictor.py`

# ToxTransformer API Usage Guide

## Base URL

```
http://136.111.102.10:6515
```

Or via load balancer (once SSL certificate is active):
```
https://toxtransformer.toxindex.com
```

## Endpoints

### Health Check

```bash
curl http://136.111.102.10:6515/health
```

**Response:**
```json
{"status": "ok"}
```

### Predict All Properties

Get predictions for all 6,647 toxicity properties for a compound.

**Endpoint:** `GET /predict_all`

**Parameters:**
- `inchi` (required): InChI string of the compound

**Example:**

```bash
# Caffeine
curl "http://136.111.102.10:6515/predict_all?inchi=InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
```

**Response Structure:**

```json
[
  {
    "inchi": "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3",
    "property": {
      "categories": [
        {
          "category": "carcinogenicity",
          "reason": "The assay measures tumor cell line growth inhibition...",
          "strength": 10.0
        }
      ],
      "metadata": {
        "assay_id": 809035,
        "description": "PUBCHEM_BIOASSAY: NCI human tumor cell line...",
        "standard_type": "GI50",
        "source": "chembl",
        ...
      },
      "property_token": 1000,
      "source": "chembl",
      "title": "nci sk-mel-5 melanoma cell growth inhibition"
    },
    "property_token": 1000,
    "value": 0.391
  },
  ...
  // 6,646 more predictions
]
```

**Response Fields:**
- `value`: Predicted probability (0-1) for the property
- `property.title`: Human-readable property name
- `property.categories`: Toxicity categories with reasoning
- `property.metadata`: Full assay metadata

## Python Example

```python
import requests
import json

API_URL = "http://136.111.102.10:6515"

def predict_toxicity(inchi: str):
    """Get toxicity predictions for a compound"""
    response = requests.get(
        f"{API_URL}/predict_all",
        params={"inchi": inchi},
        timeout=120
    )
    return response.json()

# Example: Caffeine
caffeine_inchi = "InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"
predictions = predict_toxicity(caffeine_inchi)

# Filter high-risk predictions (probability > 0.7)
high_risk = [p for p in predictions if p.get('value', 0) > 0.7]

print(f"Total properties: {len(predictions)}")
print(f"High-risk properties: {len(high_risk)}")

# Show top 5 high-risk properties
for pred in sorted(high_risk, key=lambda x: x['value'], reverse=True)[:5]:
    print(f"  {pred['property']['title']}: {pred['value']:.3f}")
```

## cURL Examples

### Get predictions and filter by category

```bash
# Get all carcinogenicity-related predictions
curl -s "http://136.111.102.10:6515/predict_all?inchi=InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3" \
  | jq '[.[] | select(.property.categories[]?.category == "carcinogenicity")] | .[0:5]'
```

### Count predictions above threshold

```bash
# Count how many properties have probability > 0.5
curl -s "http://136.111.102.10:6515/predict_all?inchi=InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3" \
  | jq '[.[] | select(.value > 0.5)] | length'
```

### Get top 10 highest risk predictions

```bash
curl -s "http://136.111.102.10:6515/predict_all?inchi=InChI=1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3" \
  | jq 'sort_by(-.value) | .[0:10] | .[] | {title: .property.title, value: .value}'
```

## Common InChI Examples

### Aspirin
```
InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)
```

### Ibuprofen
```
InChI=1S/C13H18O2/c1-9(2)8-11-4-6-12(7-5-11)10(3)13(14)15/h4-7,9-10H,8H2,1-3H3,(H,14,15)
```

### Benzene
```
InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H
```

## Notes

- **Request timeout**: Set a generous timeout (120s recommended) as predictions can take time for complex molecules
- **Response size**: Each response contains 6,647 predictions (~4-5 MB JSON)
- **Rate limiting**: Currently no rate limits, but please be respectful
- **SMILES conversion**: If you have SMILES, convert to InChI first using RDKit or similar

## Converting SMILES to InChI

```python
from rdkit import Chem

def smiles_to_inchi(smiles: str) -> str:
    """Convert SMILES to InChI"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToInchi(mol)

# Example
inchi = smiles_to_inchi("CC(=O)OC1=CC=CC=C1C(=O)O")  # Aspirin
print(inchi)
```

## Web UI

A Streamlit web interface is available at:
```
https://toxtransformer.toxindex.com
```

The UI provides:
- Molecule structure visualization
- Interactive predictions table
- Category-based breakdown
- Export to CSV/JSON
- Batch processing

## Support

For issues or questions:
- GitHub: https://github.com/biobricks-ai/chemprop-transformer
- API Status: http://136.111.102.10:6515/health

# Improved Property Matching Plan for ToxTransformer Benchmark

## Problem Statement

The current evaluation uses crude keyword matching that maps semantically different endpoints
to inappropriate ToxTransformer tokens. This leads to:
- Ames mutagenicity being evaluated against cytotoxicity predictions
- Organ-specific toxicity being evaluated against generic cell viability
- CYP isoform mismatches (CYP1A2 external data vs CYP3A4 predictions)

## Evidence That Match Quality Affects Performance

Within-category analysis shows matching matters:
- CYP3A4 (correctly matched): AUC 0.62-0.66
- CYP1A2/2E1 (mismatched): AUC 0.45-0.49
- Difference: ~0.15 AUC improvement with proper matching

## Proposed Approach

### Phase 1: Semantic Token Discovery

Query ToxTransformer for ALL available tokens and create a comprehensive token catalog:

```python
# Pseudo-code for token discovery
def discover_tokens():
    """Run inference on diverse compounds and collect all returned tokens."""
    diverse_compounds = get_diverse_chemical_set(n=1000)  # Structurally diverse
    all_tokens = {}

    for compound in diverse_compounds:
        predictions = api.predict_all(compound)
        for pred in predictions:
            token_id = pred['property_token']
            if token_id not in all_tokens:
                all_tokens[token_id] = {
                    'title': pred['property']['title'],
                    'categories': pred['property']['categories'],
                    'metadata': pred['property']['metadata']
                }

    return all_tokens
```

### Phase 2: Curated Endpoint Mapping

Create explicit, curated mappings for key toxicology endpoints:

```python
CURATED_MAPPINGS = {
    # Ames Mutagenicity - ONLY use mutagenicity-specific tokens
    'ames_mutagenicity': {
        'required_keywords': ['ames', 'mutagen', 'salmonella', 'genotox'],
        'excluded_keywords': ['cytotox', 'viability'],
        'fallback_tokens': [3830],  # Known Salmonella mutagenicity token
    },

    # CYP450 - Isoform-specific matching
    'cyp1a2': {
        'required_keywords': ['cyp1a2'],
        'excluded_keywords': ['cyp2', 'cyp3'],
        'fallback_tokens': None,  # Don't evaluate if no specific token
    },
    'cyp2c9': {
        'required_keywords': ['cyp2c9'],
        'excluded_keywords': ['cyp1', 'cyp3', 'cyp2d', 'cyp2c19'],
        'fallback_tokens': [1854],
    },
    # ... etc for each isoform

    # hERG - Cardiac ion channel
    'herg': {
        'required_keywords': ['herg', 'kcnh2', 'cardiac ion'],
        'excluded_keywords': [],
        'fallback_tokens': [396, 1866, 2223, 2228, 5768],
    },

    # Hepatotoxicity/DILI
    'hepatotoxicity': {
        'required_keywords': ['hepat', 'liver', 'dili'],
        'excluded_keywords': ['cytotox'],
        'fallback_tokens': None,  # Discover or skip
    },

    # Carcinogenicity
    'carcinogenicity': {
        'required_keywords': ['carcino', 'tumor', 'cancer'],
        'excluded_keywords': ['cytotox'],
        'fallback_tokens': [2881, 3128, 3291, 3574],
    },
}
```

### Phase 3: Strict Matching Evaluation

```python
def evaluate_with_strict_matching(external_property, prediction_tokens, curated_mapping):
    """Only evaluate if we have semantically appropriate tokens."""

    # Find the endpoint category
    endpoint_type = categorize_external_property(external_property)
    mapping = curated_mapping.get(endpoint_type)

    if mapping is None:
        return None, "No mapping defined"

    # Find matching tokens in predictions
    matched_tokens = []
    for token in prediction_tokens:
        token_desc = get_token_description(token)

        # Must have required keywords
        if not any(kw in token_desc.lower() for kw in mapping['required_keywords']):
            continue

        # Must not have excluded keywords
        if any(kw in token_desc.lower() for kw in mapping['excluded_keywords']):
            continue

        matched_tokens.append(token)

    # Use fallback if no semantic match found
    if not matched_tokens and mapping['fallback_tokens']:
        matched_tokens = [t for t in mapping['fallback_tokens'] if t in prediction_tokens]

    if not matched_tokens:
        return None, "No semantically appropriate tokens found"

    return aggregate_predictions(matched_tokens), "Matched"
```

### Phase 4: Transparent Reporting

Report match quality alongside performance:

| External Endpoint | ToxTransformer Token | Match Quality | AUC | Note |
|-------------------|---------------------|---------------|-----|------|
| Ames (TDC) | Token 3830 (Salmonella mutagen) | Exact | 0.XX | Direct semantic match |
| CYP3A4 (Veith) | Token 1521 (CYP3A4 inhibition) | Exact | 0.65 | Isoform match |
| CYP1A2 (Veith) | None | Not evaluated | N/A | No CYP1A2 token available |
| Liver toxicity | Token 760 (cytotoxicity) | Poor | 0.52 | Only general cytotox available |

## Implementation Steps

1. **Token Discovery Script** (2-3 hours API time)
   - Run predictions on 1000+ diverse compounds
   - Catalog all returned tokens with full metadata
   - Categorize by endpoint type

2. **Curated Mapping File** (manual, 2-4 hours)
   - Create JSON/YAML mapping file
   - Expert review of token-endpoint alignments
   - Document rationale for each mapping

3. **Re-evaluation Script** (6+ hours API time)
   - Re-run benchmark with strict matching
   - Only evaluate endpoints with valid mappings
   - Report match quality with results

4. **Publication Table**
   - Separate results by match quality
   - Clearly state which endpoints could/couldn't be evaluated
   - Honest about model coverage vs claimed performance

## Expected Outcomes

- Fewer endpoints evaluated (exclude mismatched ones)
- Higher AUC on properly matched endpoints
- Clear documentation of model capabilities and limitations
- More scientifically valid benchmark for publication

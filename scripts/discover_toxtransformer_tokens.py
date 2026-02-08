#!/usr/bin/env python3
"""
Discover all available tokens in ToxTransformer by running predictions
on diverse compounds and cataloging the returned property tokens.
"""

import requests
import json
from pathlib import Path
from collections import defaultdict
import time

API_URL = "http://136.111.102.10:6515"

# Diverse set of compounds for token discovery (structurally varied)
DIVERSE_COMPOUNDS = [
    # Simple organics
    "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H",  # Benzene
    "InChI=1S/CH4O/c1-2/h2H,1H3",  # Methanol
    "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",  # Ethanol

    # Drugs
    "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",  # Aspirin
    "InChI=1S/C13H18O2/c1-9(2)8-11-4-6-12(7-5-11)10(3)13(14)15/h4-7,9-10H,8H2,1-3H3,(H,14,15)",  # Ibuprofen
    "InChI=1S/C8H9NO2/c1-6(10)9-7-2-4-8(11)5-3-7/h2-5,11H,1H3,(H,9,10)",  # Acetaminophen

    # Known toxicants
    "InChI=1S/C7H5N3O6/c11-9(12)6-3-1-5(8(13)14)2-4-7(6)10(15)16/h1-4H,(H2,8,13,14)",  # TNT
    "InChI=1S/CHCl3/c2-1(3)4/h1H",  # Chloroform

    # Heterocycles
    "InChI=1S/C5H5N/c1-2-4-6-5-3-1/h1-5H",  # Pyridine
    "InChI=1S/C4H4O/c1-2-4-5-3-1/h1-4H",  # Furan
    "InChI=1S/C4H5N/c1-2-4-5-3-1/h1-3,5H,4H2",  # Pyrrole

    # Amines
    "InChI=1S/C6H7N/c7-6-4-2-1-3-5-6/h1-5H,7H2",  # Aniline
    "InChI=1S/C7H9N/c1-6-4-2-3-5-7(6)8/h2-5H,8H2,1H3",  # o-Toluidine

    # Carboxylic acids
    "InChI=1S/C7H6O2/c8-7(9)6-4-2-1-3-5-6/h1-5H,(H,8,9)",  # Benzoic acid

    # Steroids
    "InChI=1S/C27H44O/c1-19(2)8-6-9-21(4)25-15-16-26-24-14-13-22-12-10-11-20(3)27(22,5)23(24)17-18-28-25-26/h6,8,20-26H,9-18H2,1-5H3",  # Cholesterol

    # Complex drugs
    "InChI=1S/C17H18FN3O3/c1-10-9-21(11-3-6-12(18)7-4-11)15(20-10)8-14(22)13-5-2-19-16(23)17(13)24/h2-7,9,14,22H,8H2,1H3,(H,19,23,24)",  # Ofloxacin
]

def discover_tokens():
    """Discover all tokens by running predictions on diverse compounds."""
    all_tokens = {}
    token_counts = defaultdict(int)

    print("Discovering ToxTransformer tokens...")
    print("=" * 60)

    for i, inchi in enumerate(DIVERSE_COMPOUNDS):
        try:
            r = requests.get(f"{API_URL}/predict_all", params={"inchi": inchi}, timeout=120)
            if r.status_code == 200:
                predictions = r.json()
                print(f"Compound {i+1}/{len(DIVERSE_COMPOUNDS)}: {len(predictions)} predictions")

                for pred in predictions:
                    token_id = pred['property_token']
                    token_counts[token_id] += 1

                    if token_id not in all_tokens:
                        all_tokens[token_id] = {
                            'title': pred['property'].get('title', 'Unknown'),
                            'categories': pred['property'].get('categories', []),
                            'metadata': pred['property'].get('metadata', {}),
                        }
            else:
                print(f"Compound {i+1}: API error {r.status_code}")
        except Exception as e:
            print(f"Compound {i+1}: Error - {e}")

        time.sleep(0.5)  # Rate limit

    return all_tokens, token_counts

def categorize_tokens(all_tokens):
    """Categorize tokens by endpoint type."""
    categories = defaultdict(list)

    for token_id, info in all_tokens.items():
        title = info['title'].lower()
        token_cats = [c['category'] for c in info.get('categories', [])]

        # Categorize based on title keywords
        if any(kw in title for kw in ['ames', 'mutagen', 'salmonella', 'genotox']):
            categories['mutagenicity'].append((token_id, info))
        elif any(kw in title for kw in ['herg', 'kcnh', 'kcnq', 'ion channel']):
            categories['herg_ion_channel'].append((token_id, info))
        elif 'cyp' in title:
            # Subclassify by isoform
            for iso in ['1a2', '2c9', '2c19', '2d6', '2e1', '3a4']:
                if iso in title:
                    categories[f'cyp_{iso}'].append((token_id, info))
                    break
            else:
                categories['cyp_other'].append((token_id, info))
        elif any(kw in title for kw in ['hepat', 'liver', 'dili']):
            categories['hepatotoxicity'].append((token_id, info))
        elif any(kw in title for kw in ['cardio', 'cardiac', 'heart']):
            categories['cardiotoxicity'].append((token_id, info))
        elif any(kw in title for kw in ['cytotox', 'viability', 'cell death', 'apoptosis']):
            categories['cytotoxicity'].append((token_id, info))
        elif any(kw in title for kw in ['carcino', 'tumor']):
            categories['carcinogenicity'].append((token_id, info))
        elif any(kw in title for kw in ['solubil']):
            categories['solubility'].append((token_id, info))
        elif any(kw in title for kw in ['zebrafish', 'developmental', 'teratogen']):
            categories['developmental'].append((token_id, info))
        else:
            categories['other'].append((token_id, info))

    return categories

def main():
    # Check API health
    try:
        r = requests.get(f"{API_URL}/health", timeout=10)
        if r.status_code != 200:
            print("API not available")
            return
    except Exception as e:
        print(f"API connection error: {e}")
        return

    # Discover tokens
    all_tokens, token_counts = discover_tokens()

    print(f"\n{'=' * 60}")
    print(f"Total unique tokens discovered: {len(all_tokens)}")
    print(f"{'=' * 60}")

    # Categorize
    categories = categorize_tokens(all_tokens)

    print("\n=== Token Categories ===\n")
    for cat, tokens in sorted(categories.items()):
        print(f"\n{cat.upper()} ({len(tokens)} tokens):")
        for token_id, info in sorted(tokens, key=lambda x: token_counts[x[0]], reverse=True)[:5]:
            print(f"  Token {token_id} (seen {token_counts[token_id]}x): {info['title'][:70]}...")

    # Save full catalog
    output = {
        'total_tokens': len(all_tokens),
        'tokens': {str(k): v for k, v in all_tokens.items()},
        'categories': {cat: [(tid, info['title']) for tid, info in tokens]
                       for cat, tokens in categories.items()},
        'token_frequencies': dict(token_counts),
    }

    output_path = Path('publication/benchmark/data/toxtransformer_token_catalog.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n\nFull catalog saved to: {output_path}")

if __name__ == "__main__":
    main()

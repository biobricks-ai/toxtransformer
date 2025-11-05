#!/usr/bin/env python3
"""
Analyze datasets and generate statistics for the ToxTransformer project.

This script analyzes the distribution of SELFIES token lengths across all substances
using efficient sampling to handle large datasets (117M+ substances).

Usage:
    source .venv/bin/activate
    python code/2_4_analyze_datasets.py

Outputs:
    - cache/analyze_datasets/token_length_analysis.md: Markdown report with tables
    - cache/analyze_datasets/token_length_stats.json: JSON statistics
"""

import pandas as pd
import numpy as np
import pathlib
import json
import logging
import sqlite3
from typing import Dict, Any, Tuple
from collections import Counter
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer

# Set up logging
def setup_logging():
    logdir = pathlib.Path('cache/analyze_datasets/log')
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logdir / 'analyze_datasets.log', mode='w'),
            logging.StreamHandler()
        ]
    )

def load_tokenizer():
    """Load the SELFIES tokenizer"""
    tokenizer_path = "brick/selfies_property_val_tokenizer/selfies_tokenizer.json"
    return SelfiesPropertyValTokenizer.from_file(tokenizer_path)

def sample_from_parquet(parquet_path: str, sample_size: int = 100000) -> Tuple[pd.DataFrame, int]:
    """Sample substances from Parquet file efficiently."""
    logging.info(f"Loading data from Parquet: {parquet_path}")
    
    # First, get the total count by reading just the shape
    logging.info("Getting total count of substances...")
    df_full = pd.read_parquet(parquet_path)
    total_count = len(df_full)
    logging.info(f"Total substances in parquet: {total_count:,}")
    
    # Sample randomly
    logging.info(f"Sampling {sample_size:,} substances randomly...")
    if total_count <= sample_size:
        sample_df = df_full
        logging.info(f"Using all {total_count:,} substances (smaller than sample size)")
    else:
        sample_df = df_full.sample(n=sample_size, random_state=42)
        logging.info(f"Successfully sampled {len(sample_df):,} substances")
    
    # Make sure we have the required columns
    required_cols = ['selfies']
    missing_cols = [col for col in required_cols if col not in sample_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(sample_df.columns)}")
    
    return sample_df[required_cols], total_count

def analyze_token_lengths(sample_df: pd.DataFrame, total_count: int, sample_size: int) -> Tuple[Dict[str, Any], np.ndarray]:
    """Analyze token lengths from SELFIES strings using tokenizer."""
    logging.info("Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    logging.info("Analyzing token lengths from SELFIES strings...")
    
    token_lengths = []
    actual_lengths = []  # lengths without padding
    
    for i, selfies in enumerate(sample_df['selfies']):
        if i % 10000 == 0:
            logging.info(f"Processed {i:,}/{len(sample_df):,} substances...")
            
        if pd.isna(selfies) or selfies == '':
            continue
            
        try:
            # Tokenize the SELFIES string
            tokens = tokenizer.encode(selfies)
            token_lengths.append(len(tokens))
            
            # Count actual tokens (excluding padding)
            # Find actual length by counting non-padding tokens
            if hasattr(tokenizer, 'pad_token_id'):
                pad_id = tokenizer.pad_token_id
            else:
                pad_id = 0  # assume 0 is padding
            
            actual_length = sum(1 for token_id in tokens if token_id != pad_id)
            actual_lengths.append(actual_length)
            
        except Exception as e:
            logging.warning(f"Error tokenizing SELFIES '{selfies}': {e}")
            continue
    
    logging.info(f"Successfully analyzed {len(actual_lengths):,} substances")
    
    if len(actual_lengths) == 0:
        raise ValueError("No valid SELFIES strings found to analyze")
    
    actual_lengths = np.array(actual_lengths)
    token_lengths = np.array(token_lengths)
    
    # Basic statistics for actual lengths
    stats = {
        'total_substances_in_db': total_count,
        'sample_size': len(actual_lengths),
        'sampling_percentage': (len(actual_lengths) / total_count) * 100,
        'padded_length': int(token_lengths[0]) if len(token_lengths) > 0 else 0,
        'min_length': int(actual_lengths.min()),
        'max_length': int(actual_lengths.max()),
        'mean_length': float(actual_lengths.mean()),
        'median_length': float(np.median(actual_lengths)),
        'std_length': float(actual_lengths.std()),
        'q25_length': float(np.percentile(actual_lengths, 25)),
        'q75_length': float(np.percentile(actual_lengths, 75)),
        'q95_length': float(np.percentile(actual_lengths, 95)),
        'q99_length': float(np.percentile(actual_lengths, 99)),
    }
    
    # Count distribution for actual lengths
    length_counts = Counter(actual_lengths)
    stats['length_distribution'] = dict(sorted(length_counts.items()))
    
    # Binned distribution for ranges
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, np.inf]
    bin_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-74', '75-99', '100-149', '150-199', '200+']
    binned_counts = pd.cut(actual_lengths, bins=bins, labels=bin_labels, right=False).value_counts().sort_index()
    
    stats['binned_distribution'] = {}
    for label, count in binned_counts.items():
        stats['binned_distribution'][str(label)] = int(count)
    
    # Percentage distributions
    total = len(actual_lengths)
    stats['binned_percentages'] = {}
    for label, count in stats['binned_distribution'].items():
        stats['binned_percentages'][label] = round(100 * count / total, 2)
    
    # Add padding analysis
    padded_length = stats['padded_length']
    stats['padding_analysis'] = {
        'sequences_at_max_padding': int(np.sum(actual_lengths >= padded_length)),
        'sequences_needing_truncation': int(np.sum(actual_lengths > padded_length)),
        'average_padding_used': float(padded_length - actual_lengths.mean()),
        'median_padding_used': float(padded_length - np.median(actual_lengths))
    }
    
    logging.info(f"Analysis complete. Mean actual token length: {stats['mean_length']:.2f}")
    return stats, actual_lengths

def generate_markdown_report(stats: Dict[str, Any], output_path: pathlib.Path) -> None:
    """Generate a markdown report with analysis results."""
    logging.info(f"Generating markdown report at {output_path}")
    
    with open(output_path, 'w') as f:
        f.write("# SELFIES Token Length Analysis\n\n")
        f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset overview
        f.write("## Dataset Overview\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total Substances in Database | {stats['total_substances_in_db']:,} |\n")
        f.write(f"| Sample Size Analyzed | {stats['sample_size']:,} |\n")
        f.write(f"| Sampling Percentage | {stats['sampling_percentage']:.3f}% |\n")
        f.write(f"| Padded Length (All Sequences) | {stats['padded_length']} |\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics (Actual Token Lengths)\n\n")
        f.write("| Statistic | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Mean Actual Length | {stats['mean_length']:.2f} |\n")
        f.write(f"| Median Actual Length | {stats['median_length']:.2f} |\n")
        f.write(f"| Standard Deviation | {stats['std_length']:.2f} |\n")
        f.write(f"| Minimum Length | {stats['min_length']} |\n")
        f.write(f"| Maximum Length | {stats['max_length']} |\n")
        f.write(f"| 25th Percentile | {stats['q25_length']:.2f} |\n")
        f.write(f"| 75th Percentile | {stats['q75_length']:.2f} |\n")
        f.write(f"| 95th Percentile | {stats['q95_length']:.2f} |\n")
        f.write(f"| 99th Percentile | {stats['q99_length']:.2f} |\n\n")
        
        # Padding analysis
        f.write("## Padding Analysis\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Average Padding Used | {stats['padding_analysis']['average_padding_used']:.2f} tokens |\n")
        f.write(f"| Median Padding Used | {stats['padding_analysis']['median_padding_used']:.2f} tokens |\n")
        f.write(f"| Sequences at Max Length | {stats['padding_analysis']['sequences_at_max_padding']:,} |\n")
        f.write(f"| Sequences Needing Truncation | {stats['padding_analysis']['sequences_needing_truncation']:,} |\n\n")
        
        # Binned distribution
        f.write("## Actual Token Length Distribution by Ranges\n\n")
        f.write("| Length Range | Count | Percentage |\n")
        f.write("|--------------|-------|------------|\n")
        for bin_label in stats['binned_distribution'].keys():
            count = stats['binned_distribution'][bin_label]
            percentage = stats['binned_percentages'][bin_label]
            f.write(f"| {bin_label} | {count:,} | {percentage:.2f}% |\n")
        f.write("\n")
        
        # Top 20 most common actual lengths
        f.write("## Top 20 Most Common Actual Token Lengths\n\n")
        f.write("| Length | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        
        # Sort by count descending and take top 20
        sorted_lengths = sorted(stats['length_distribution'].items(), 
                              key=lambda x: x[1], reverse=True)[:20]
        total = stats['total_substances']
        
        for length, count in sorted_lengths:
            percentage = 100 * count / total
            f.write(f"| {length} | {count:,} | {percentage:.2f}% |\n")
        f.write("\n")
        
        # Analysis insights
        f.write("## Key Insights\n\n")
        
        # Most common length
        most_common_length = max(stats['length_distribution'].items(), key=lambda x: x[1])
        total = stats['sample_size']
        f.write(f"- **Most common actual token length:** {most_common_length[0]} tokens "
                f"({most_common_length[1]:,} substances, "
                f"{100 * most_common_length[1] / total:.2f}%)\n")
        
        # Padding efficiency
        avg_padding = stats['padding_analysis']['average_padding_used']
        padding_efficiency = 100 * (1 - avg_padding / stats['padded_length'])
        f.write(f"- **Padding efficiency:** {padding_efficiency:.1f}% (average {avg_padding:.1f} padding tokens per sequence)\n")
        
        # Truncation analysis
        truncated = stats['padding_analysis']['sequences_needing_truncation']
        if truncated > 0:
            f.write(f"- **Sequences requiring truncation:** {truncated:,} ({100 * truncated / total:.2f}%)\n")
        else:
            f.write(f"- **No sequences require truncation** (all fit within {stats['padded_length']} tokens)\n")
        
        # Short sequences
        short_count = sum(count for length, count in stats['length_distribution'].items() if length <= 20)
        f.write(f"- **Short sequences (≤20 tokens):** {short_count:,} "
                f"({100 * short_count / total:.2f}%)\n")
        
        # Long sequences  
        long_count = sum(count for length, count in stats['length_distribution'].items() if length >= 100)
        f.write(f"- **Long sequences (≥100 tokens):** {long_count:,} "
                f"({100 * long_count / total:.2f}%)\n")
        
        # IQR analysis
        iqr = stats['q75_length'] - stats['q25_length']
        f.write(f"- **Interquartile Range (IQR):** {iqr:.2f} tokens\n")
        f.write(f"- **50% of sequences are between:** {stats['q25_length']:.0f} and {stats['q75_length']:.0f} tokens\n")
        
        # Memory usage insight (extrapolated to full dataset)
        total_db_tokens = stats['total_substances_in_db'] * stats['padded_length']
        actual_db_tokens = stats['total_substances_in_db'] * stats['mean_length']
        memory_overhead = 100 * (total_db_tokens - actual_db_tokens) / total_db_tokens
        f.write(f"- **Estimated memory overhead from padding (full DB):** {memory_overhead:.1f}% ({(total_db_tokens - actual_db_tokens)/1e9:,.1f}B padding tokens)\n")

def analyze_selfies_tokens(sample_df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
    """Analyze the actual SELFIES tokens in a sample of the data."""
    logging.info(f"Analyzing SELFIES tokens from sample of {min(sample_size, len(sample_df))} substances...")
    
    # Take a subsample for token analysis if needed
    if len(sample_df) > sample_size:
        token_sample_df = sample_df.sample(n=sample_size, random_state=42)
    else:
        token_sample_df = sample_df
    
    # Extract all tokens from the sample
    all_tokens = []
    for selfies_str in token_sample_df['selfies'].dropna():
        # Split SELFIES string into tokens (similar to tokenizer logic)
        try:
            import selfies as sf
            tokens = sf.split_selfies(selfies_str)
            all_tokens.extend(tokens)
        except:
            # Fallback: simple bracket-based splitting
            tokens = []
            current_token = ""
            in_bracket = False
            for char in selfies_str:
                if char == '[':
                    if current_token:
                        tokens.append(current_token)
                        current_token = ""
                    current_token = char
                    in_bracket = True
                elif char == ']':
                    current_token += char
                    tokens.append(current_token)
                    current_token = ""
                    in_bracket = False
                else:
                    current_token += char
            if current_token:
                tokens.append(current_token)
            all_tokens.extend(tokens)
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Vocabulary statistics
    vocab_stats = {
        'total_tokens': len(all_tokens),
        'unique_tokens': len(token_counts),
        'sample_size': len(token_sample_df),
        'top_tokens': dict(token_counts.most_common(50)),
        'singleton_tokens': sum(1 for count in token_counts.values() if count == 1)
    }
    
    logging.info(f"Found {vocab_stats['unique_tokens']} unique tokens in sample")
    return vocab_stats

def save_statistics(stats: Dict[str, Any], vocab_stats: Dict[str, Any], output_path: pathlib.Path) -> None:
    """Save statistics to JSON file."""
    logging.info(f"Saving statistics to {output_path}")
    
    combined_stats = {
        'token_length_analysis': stats,
        'vocabulary_analysis': vocab_stats,
        'metadata': {
            'generated_at': pd.Timestamp.now().isoformat(),
            'analysis_version': '2.0_sampled',
            'sampling_info': {
                'total_db_size': stats['total_substances_in_db'],
                'sample_size': stats['sample_size'],
                'sampling_percentage': stats['sampling_percentage']
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_stats, f, indent=2)

def main(sample_size: int = 100000):
    """Main analysis function."""
    setup_logging()
    logging.info("Starting dataset analysis with sampling...")
    
    # Create output directory
    output_dir = pathlib.Path('cache/analyze_datasets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample data from Parquet file (the one with SELFIES data)
    parquet_path = "cache/preprocess_tokenizer/substances.parquet"
    if not pathlib.Path(parquet_path).exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
    
    sample_df, total_count = sample_from_parquet(parquet_path, sample_size)
    
    # Analyze token lengths
    stats, token_lengths = analyze_token_lengths(sample_df, total_count, sample_size)
    
    # Analyze vocabulary (on a smaller sample for efficiency)
    vocab_stats = analyze_selfies_tokens(sample_df, sample_size=min(10000, len(sample_df)))
    
    # Generate outputs
    markdown_path = output_dir / 'token_length_analysis.md'
    json_path = output_dir / 'token_length_stats.json'
    
    generate_markdown_report(stats, markdown_path)
    save_statistics(stats, vocab_stats, json_path)
    
    logging.info("Analysis complete!")
    logging.info(f"Markdown report: {markdown_path}")
    logging.info(f"JSON statistics: {json_path}")
    
    # Print quick summary
    print("\n" + "="*60)
    print("SELFIES TOKEN LENGTH ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total substances in DB: {stats['total_substances_in_db']:,}")
    print(f"Sample size analyzed: {stats['sample_size']:,} ({stats['sampling_percentage']:.3f}%)")
    print(f"Mean token length: {stats['mean_length']:.2f}")
    print(f"Median token length: {stats['median_length']:.0f}")
    print(f"Token length range: {stats['min_length']} - {stats['max_length']}")
    print(f"Unique vocabulary size (sample): {vocab_stats['unique_tokens']:,}")
    print(f"Most common length: {max(stats['length_distribution'].items(), key=lambda x: x[1])[0]} tokens")
    print("="*60)

if __name__ == "__main__":
    main()

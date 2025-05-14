import pathlib
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
from sklearn.metrics import roc_auc_score, average_precision_score

from cvae.models.datasets import PropertyGuaranteeDataset, SharedSampleTracker
from cvae.models.multitask_transformer import SelfiesPropertyValTokenizer
from cvae.training.trainer import Trainer

# Set paths
benchmark_dir = pathlib.Path('cache/benchmark_evaluations')
results_dir = benchmark_dir / 'results'
results_dir.mkdir(parents=True, exist_ok=True)

# Load the benchmark properties
benchmark_props = pd.read_parquet(benchmark_dir / 'benchmark_properties.parquet')
property_tokens = benchmark_props['property_token'].unique().tolist()

def evaluate_benchmark_properties(model, dataset, device='cuda', batch_size=32):
    """
    Evaluate model on benchmark properties at specified positions
    """
    model.eval()
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    
    all_results = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items() if k != 'metadata'}
            
            # Get property positions for evaluation
            property_positions = batch['metadata']['property_positions']
            property_tokens = batch['metadata']['property_tokens']
            true_values = batch['metadata']['property_values']
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Extract predictions at property positions
            batch_size = logits.shape[0]
            for b in range(batch_size):
                for pos, token, true_val in zip(
                    property_positions[b], 
                    property_tokens[b], 
                    true_values[b]
                ):
                    if pos < 0:  # Skip invalid positions
                        continue
                    
                    # Get prediction at this position for this property token
                    pred = torch.sigmoid(logits[b, pos, token]).item()
                    
                    # Record the result
                    if token not in all_results:
                        all_results[token] = {'preds': [], 'targets': []}
                    
                    all_results[token]['preds'].append(pred)
                    all_results[token]['targets'].append(true_val)
    
    # Calculate metrics for each property token
    metrics = {}
    for token, data in all_results.items():
        preds = np.array(data['preds'])
        targets = np.array(data['targets'])
        
        if len(np.unique(targets)) < 2:
            # Skip if all targets are the same
            continue
        
        try:
            auroc = roc_auc_score(targets, preds)
            auprc = average_precision_score(targets, preds)
            
            metrics[token] = {
                'auroc': auroc,
                'auprc': auprc,
                'num_samples': len(targets)
            }
        except:
            # Handle edge cases
            print(f"Could not calculate metrics for token {token}")
    
    return metrics

def run_benchmark_evaluation(model_path, run_id):
    """Run benchmark evaluation for a given model"""
    # Load the model
    model, config = get_model_from_run(model_path, run_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create tokenizer
    tokenizer = SelfiesPropertyValTokenizer.from_config(config.tokenizer_config)
    
    # Create dataset with benchmark properties
    tracker = SharedSampleTracker()
    benchmark_dataset = PropertyGuaranteeDataset(
        tokenizer=tokenizer,
        max_length=config.model_config.get('max_length', 512),
        property_tokens=property_tokens,
        sample_tracker=tracker,
        split='test'
    )
    
    # Run evaluation
    metrics = evaluate_benchmark_properties(model, benchmark_dataset, device)
    
    # Save results
    with open(results_dir / f"benchmark_eval_{run_id}.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Calculate and print aggregate metrics by source
    source_metrics = {}
    for _, row in benchmark_props.iterrows():
        token = row['property_token']
        source = row['source']
        weight = row['weight']
        
        if token in metrics:
            if source not in source_metrics:
                source_metrics[source] = {
                    'weighted_auroc': 0,
                    'weighted_auprc': 0,
                    'total_weight': 0,
                    'property_count': 0
                }
            
            source_metrics[source]['weighted_auroc'] += metrics[token]['auroc'] * weight
            source_metrics[source]['weighted_auprc'] += metrics[token]['auprc'] * weight
            source_metrics[source]['total_weight'] += weight
            source_metrics[source]['property_count'] += 1
    
    # Normalize by total weight
    for source in source_metrics:
        if source_metrics[source]['total_weight'] > 0:
            source_metrics[source]['auroc'] = source_metrics[source]['weighted_auroc'] / source_metrics[source]['total_weight']
            source_metrics[source]['auprc'] = source_metrics[source]['weighted_auprc'] / source_metrics[source]['total_weight']
    
    # Save source metrics
    with open(results_dir / f"source_metrics_{run_id}.json", 'w') as f:
        json.dump(source_metrics, f, indent=2)
    
    print("Benchmark evaluation complete. Source-level metrics:")
    for source, metrics in source_metrics.items():
        print(f"{source}: AUROC={metrics.get('auroc', 'N/A'):.4f}, AUPRC={metrics.get('auprc', 'N/A'):.4f}")
    
    return source_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run benchmark evaluations')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model directory')
    parser.add_argument('--run_id', type=str, required=True, help='Run ID for model')
    
    args = parser.parse_args()
    run_benchmark_evaluation(args.model_path, args.run_id)

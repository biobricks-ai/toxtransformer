import pandas as pd
import torch
import pathlib
import numpy as np
import sqlite3
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import cvae.models.mixture_experts as me
from cvae.tokenizer import SelfiesPropertyValTokenizer

# Setup paths
outdir = pathlib.Path("cache/nprops0")
outdir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(f'cuda:0')
model: me.MoE = me.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)
tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer

hld_df = pd.read_parquet("cache/property_benchmarks/hld_df.parquet")
# Load property metadata
conn = sqlite3.connect('brick/cvae.sqlite')
prop_src = pd.read_sql("SELECT property_token, title, source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)
prop_src = prop_src.groupby('property_token').first().reset_index()
# Ensure 'value' column is binary
hld_df['value'] = hld_df['value'].apply(lambda v: 0 if v == tokenizer.value_indexes()[0] else 1)

def build_av_batch_nprop0(batch_size, property_tokens):
    av_batch = torch.full((batch_size, 4), tokenizer.PAD_IDX, dtype=torch.long)
    av_batch[:, 0] = tokenizer.SEP_IDX  # Start token
    av_batch[:, 1] = property_tokens.view(-1)  # Ensure correct shape
    av_batch[:, 2] = tokenizer.PAD_IDX  # Space for predicted value
    av_batch[:, 3] = tokenizer.END_IDX  # End token
    return av_batch.to(DEVICE)

def evaluate_model_nprop0(df):
    dataset = [(torch.tensor(sf, dtype=torch.long), torch.tensor(assay, dtype=torch.long), torch.tensor(val, dtype=torch.long))
               for sf, assay, val in zip(df['selfies'], df['assay'], df['value'])]
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=640, shuffle=False, collate_fn=lambda x: list(zip(*x)))
    preds = []
    with torch.no_grad():
        for sf_batch, property_tokens, values in tqdm(dataset_loader, desc="Evaluating model"):
            sf_batch = torch.stack(sf_batch).to(DEVICE)
            property_tokens = torch.stack(property_tokens).to(DEVICE)  # Ensure tensor format
            av_batch = build_av_batch_nprop0(sf_batch.shape[0], property_tokens)
            batch_preds = model(sf_batch, av_batch)[:, 1, [tokenizer.value_indexes()[0], tokenizer.value_indexes()[1]]]
            batch_preds = torch.nn.functional.softmax(batch_preds, dim=1)
            preds.extend(batch_preds.cpu()[:, 1].tolist())
    return preds


hld_df['selfies_tuple'] = hld_df['selfies'].apply(tuple)
pair_counts = hld_df.groupby(['selfies_tuple', 'assay']).size().reset_index(name='count')
count_dist = pair_counts['count'].value_counts().sort_index()
# Run evaluation
hld_df['pred_nprop0'] = evaluate_model_nprop0(hld_df)
hld_df.to_parquet("data/property_benchmarks/temp_filtered_10_nprop0_from_hld.parquet")
safe_roc = lambda x, y: roc_auc_score(x, y) if len(set(x)) > 1 else None
aucs = hld_df.groupby('assay').apply(lambda x: safe_roc(x['value'], x['pred_nprop0'])).reset_index()
aucs.columns = ['property_token', 'auc']
propeval = prop_src.merge(aucs, on='property_token', how='left')
med_aucs = propeval.groupby('source').apply(lambda x: x['auc'].median()).sort_values(ascending=False)
print(med_aucs)

# bindingdb    0.853648
# tox21        0.790215
# sider        0.731023
# CLINTOX      0.724216
# BACE         0.721796
# BBBP         0.711432
# pubchem      0.680955
# Tox21        0.678501
# toxvaldb     0.666288
# ice          0.658655
# toxcast      0.650775
# reach        0.629383
# chembl       0.616117
# ctdbase      0.284722


# ================================
# Try to use the pickle files as df for evaluate_model_nprop0
import pickle
pickle_files = list(pathlib.Path("cache/property_benchmarks/temp_filtered_10").glob("*.pkl"))
evaldf = pd.DataFrame()
print("\nProcessing pickle files...")
pickle_files = tqdm(pickle_files, desc="Processing pickle files")

# Process each pickle file and create a DataFrame for evaluation
for pickle_file in pickle_files:
    res = pickle.load(open(pickle_file, "rb"))
    prop_token, sf_str, values, sf_tensors, av_tensors = res.values()
    
    # Convert to tensors and reshape
    sf_tensor = torch.stack(sf_tensors)
    
    # Create dataset with selfies tensors and property tokens
    property_tokens = torch.full((len(sf_tensor),), prop_token, dtype=torch.long)
    dataset = [(sf, prop) for sf, prop in zip(sf_tensor, property_tokens)]
    
    # Create DataLoader
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=640, shuffle=False, collate_fn=lambda x: list(zip(*x)))
    
    # Get predictions
    preds = []
    with torch.no_grad():
        for sf_batch, prop_batch in dataset_loader:
            sf_batch = torch.stack(sf_batch).to(DEVICE)
            prop_batch = torch.stack(prop_batch)
            av_batch = build_av_batch_nprop0(sf_batch.shape[0], prop_batch)
            batch_preds = model(sf_batch, av_batch)[:, 1, [tokenizer.value_indexes()[0], tokenizer.value_indexes()[1]]]
            batch_preds = torch.nn.functional.softmax(batch_preds, dim=1)
            preds.extend(batch_preds.cpu()[:, 1].tolist())

    # Create temporary DataFrame for this property
    temp_df = pd.DataFrame({
        'selfies_str': sf_str,
        'value': values,
        'pred': preds,
        'property_token': prop_token    
    })
    evaldf = pd.concat([evaldf, temp_df])
evaldf.to_parquet("data/property_benchmarks/temp_filtered_10_nprop0_from_pickle.parquet")
# Calculate AUC scores for nprops=0
safe_roc = lambda x, y: roc_auc_score(x, y) if len(set(x)) > 1 else None

aucs = evaldf.groupby('property_token').apply(lambda x: safe_roc(x['value'], x['pred'])).reset_index()
aucs.columns = ['property_token', 'auc']

# Merge with property metadata
propeval = prop_src.merge(aucs, on='property_token', how='left')
med_aucs = propeval.groupby('source').apply(lambda x: x['auc'].median()).sort_values(ascending=False)
print("\nMedian AUC scores by source for nprops=0:")
print(med_aucs)

# >>> print(med_aucs)
# source
# tox21        0.708384
# ice          0.667708
# bindingdb    0.666667
# toxcast      0.663529
# BACE         0.583270
# BBBP         0.583053
# CLINTOX      0.572533
# pubchem      0.556346
# Tox21        0.537908
# sider        0.530816
# chembl       0.516667
# ctdbase      0.500000
# toxvaldb     0.455840
# reach        0.444553

# Count distribution analysis
pair_counts = evaldf.groupby(['selfies_str', 'property_token']).size().reset_index(name='count')
count_dist = pair_counts['count'].value_counts().sort_index()
print("\nDistribution of selfies-property pair counts for nprops=0:")
print(count_dist)




# ================================


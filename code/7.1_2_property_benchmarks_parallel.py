import pickle, uuid
import pandas as pd, sqlite3, seaborn as sns, matplotlib.pyplot as plt, os, pathlib, numpy as np
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
from concurrent.futures import ProcessPoolExecutor, as_completed
from cvae.tokenizer import SelfiesPropertyValTokenizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, recall_score
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import logging

tqdm.pandas()

# Create Spark environment
from pyspark.sql import SparkSession

# Initialize Spark session
spark = cvae.utils.get_spark_session()

outdir = pathlib.Path("cache/property_benchmarks_parallel")
outdir.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(f'cuda:0')
model : me.MoE = me.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)
tokenizer : SelfiesPropertyValTokenizer = model.module.tokenizer

#%% GET PROPERTIES OF INTEREST ===========================================================
conn = sqlite3.connect('brick/cvae.sqlite')

# get all the property_tokens for tox21 properties
prop_src = pd.read_sql("SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)

# assert that each property_token has only one title
assert prop_src.groupby('property_token').size().max() == 1

# Converts 2_build_tensordataset.py .pt files into df 
# @return dataframe with columns -> [ selfies_str, selfies : tensor, assay : int, value : int ]
def tensors_to_df(tensor_dir = pathlib.Path("cache/build_tensordataset/multitask_tensors/hld")):
    tensors = [torch.load(file) for file in tqdm(list(tensor_dir.iterdir()))]
    df = pd.DataFrame([
        {'selfies': selfie.tolist(), 'assay_vals': assay_val.tolist()}
        for tensor in tqdm(tensors)
        for selfie, assay_val in zip(tensor['selfies'], tensor['assay_vals'])
    ])

    df['selfies_str'] = df['selfies'].progress_apply(lambda x: ' '.join(map(str, x)))

    def process_assay_vals(assay_vals):
        # Ignore the first and last values, and remove all zeros
        processed = [val for val in assay_vals[1:] if val != 0]
        processed = processed[:-1]
        
        # Group into pairs of assay and value
        return [(processed[i], processed[i+1]) for i in range(0, len(processed), 2)]

    # Apply the processing function and explode the result
    exploded_df = df.progress_apply(lambda row: pd.Series({
        'selfies_str': row['selfies_str'],
        'selfies': row['selfies'],
        'assay_value_pairs': process_assay_vals(row['assay_vals'])
    }), axis=1).explode('assay_value_pairs')

    # Split the pairs into separate columns
    exploded_df[['assay', 'value']] = pd.DataFrame(exploded_df['assay_value_pairs'].tolist(), index=exploded_df.index)
    exploded_df = exploded_df.drop('assay_value_pairs', axis=1).reset_index(drop=True)

    return exploded_df

trn_df = tensors_to_df(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn"))
hld_df = tensors_to_df(pathlib.Path("cache/build_tensordataset/multitask_tensors/hld"))

trn_df.to_parquet((outdir / "trn_df.parquet").as_posix())
hld_df.to_parquet((outdir / "hld_df.parquet").as_posix())

# assert that there is no overlap in selfies_str between trn_df and hld_df
assert not trn_df['selfies_str'].isin(hld_df['selfies_str']).any(), "There is overlap in selfies_str between trn_df and hld_df"

#%% EVALUATE PROPERTIES ===========================================================
trn_df = pd.read_parquet(outdir / "trn_df.parquet")
hld_df = pd.read_parquet(outdir / "hld_df.parquet")

# this finds informative, non-redundant features for a given property-token
def find_features_for_properties(property_token : int, trn_df : pd.DataFrame):
    # 1. Filter trndf to the given property
    property_df = trn_df[trn_df['assay'] == property_token]
    
    # 2. Join it with trndf on selfies_str
    joined_df = property_df.merge(trn_df[['selfies_str', 'assay','value']], on='selfies_str', suffixes=('_x', '_y'))
    counts_df = joined_df.groupby(['assay_y']).size().reset_index(name='counts')
    counts_df = counts_df[counts_df['counts'] > 30]
    joined_counts = joined_df.merge(counts_df, on=['assay_y'], how='inner')

    # check that nunique(assay_y) == counts_df.shape[0]
    assert joined_counts['assay_y'].nunique() == counts_df.shape[0], "Assay_y is not unique"

    # 3. Evaluate the entropy of the results by comparing the value_x and value_y
    def calculate_entropy(group):
        probs = group['value_y'].value_counts(normalize=True)
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy

    entropy_df = joined_counts.groupby(['assay_y','counts']).apply(calculate_entropy).reset_index()
    entropy_df.columns = ['assay_y', 'counts','entropy']
    entropy_df = entropy_df.sort_values('entropy', ascending=True)

    non_redundant_features = entropy_df[entropy_df['entropy'] > 0.025]['assay_y']
    
    # 5. Remove features that provide no information
    max_entropy = np.log2(2)  # Entropy of a binary random variable (completely non-informative)
    informative_features = entropy_df[entropy_df['entropy'] <= 0.9 * max_entropy]['assay_y']
    
    # Combine the filters
    useful_features = set(non_redundant_features) & set(informative_features)
    
    return list(useful_features)

def build_selfies_assay_vals_tensors(property_token, features, values, nprops=5, ntensors=1):
    # Filter features_df to the input selfies_str
    filtered_df = pd.DataFrame({'assay_y': features, 'value_y': values})
    
    # Initialize list to store our tensors
    X = []
    
    for _ in range(ntensors):
        # Randomly select a subset of features
        selected_features = filtered_df.sample(n=min(nprops, len(filtered_df)), replace=False)
        
        # Create a tensor for the selected features
        feature_tensor = torch.zeros(nprops*2+4, dtype=torch.long)
        feature_tensor[0] = tokenizer.SEP_IDX
        idx = 1
        for _, row in selected_features.iterrows():
            feature_tensor[idx] = row['assay_y']
            feature_tensor[idx + 1] = row['value_y']
            idx += 2
        
        # Add property_token
        feature_tensor[idx] = property_token
        idx += 1
        
        # Add padding value
        feature_tensor[idx] = tokenizer.PAD_IDX
        idx += 1
        
        feature_tensor[-1] = tokenizer.END_IDX
        
        # Pad the tensor if necessary
        if idx < nprops*2+3:
            feature_tensor[idx:-1] = tokenizer.PAD_IDX
        
        X.append(feature_tensor)
    
    # Convert list to tensor
    X = torch.stack(X)
    
    return X

def build_property_token_tensors(property_token : int, hld_df : pd.DataFrame, trn_df : pd.DataFrame):
    try:
        features = find_features_for_properties(property_token, trn_df)
        if features is None:
            return None
        df = hld_df[hld_df['assay'] == property_token]
        features_df = hld_df[hld_df['assay'].isin(features)][['selfies_str','assay','value']]
        features_df = df.merge(features_df, on='selfies_str')
        
        df = df.sample(n=min(2000, len(df)), replace=False)
        selfies, values, sf_tensors, av_tensors = [], [], [], []
        for index in range(len(df)):
            
            # get all the features for this selfies_str
            selfies_str = df['selfies_str'].iloc[index]
            selfies_df = features_df[features_df['selfies_str'] == selfies_str]
            
            features, feature_values = selfies_df['assay_y'].tolist(), selfies_df['value_y'].tolist()
            sf_tensor = torch.tensor(df['selfies'].iloc[index], dtype=torch.long)
            av_tensor = build_selfies_assay_vals_tensors(property_token, features, feature_values, nprops=4, ntensors=10)
            
            value = 0 if df['value'].iloc[index] == tokenizer.value_indexes()[0] else 1
            selfies.append(selfies_str)
            values.append(value)
            sf_tensors.append(sf_tensor)
            av_tensors.append(av_tensor)
            
        return {'property_token': property_token, 'selfies': selfies, 'values': values, 'sf_tensors': sf_tensors, 'av_tensors': av_tensors}
    except Exception as e:
        print(f"Error for property_token {property_token}: {e}")
        return None

tempdir = outdir / "temp"
tempdir.mkdir(parents=True, exist_ok=True)

def init_worker(trn_df, hld_df):
    global shared_trn_df, shared_hld_df
    shared_trn_df, shared_hld_df = trn_df, hld_df

def worker_fn(property_tokens):
    for token in property_tokens:
        result = build_property_token_tensors(token, shared_hld_df, shared_trn_df)
        if result is not None:
            with open(tempdir / f"{token}.pkl", "wb") as f:
                pickle.dump(result, f)

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait

workers = 32
with ProcessPoolExecutor(initializer=init_worker, initargs=(trn_df, hld_df)) as executor:
    property_tokens = prop_src['property_token'].tolist()
    token_chunks = [property_tokens[i::workers] for i in range(workers)]
    futures = [executor.submit(worker_fn, token_chunk) for token_chunk in token_chunks]
    # wait on futures to finish
    wait(futures)
    
evaldf = pd.DataFrame()
pickles = list(tempdir.iterdir())
for pickle_file in tqdm(pickles):
    res = pickle.load(open(pickle_file, "rb"))
    prop_token, sf_str, values, sf_tensor, av_tensors = res.values()
    av_tensor = torch.stack(av_tensors)
    sf_tensor = torch.stack(sf_tensor).repeat_interleave(10, dim=0)
    av_tensor = av_tensor.reshape(-1, av_tensor.shape[-1])
    
    val0index, val1index = tokenizer.value_indexes()[0], tokenizer.value_indexes()[1]
    dataset_loader = torch.utils.data.DataLoader(list(zip(sf_tensor, av_tensor)), batch_size=640, shuffle=False)
    with torch.no_grad():
        preds = []
        for sf_batch, av_batch in dataset_loader:
            sf_batch = sf_batch.to(DEVICE)
            av_batch = av_batch.to(DEVICE)
            batch_preds = model(sf_batch, av_batch)[:, 10, [val0index,val1index]]
            batch_preds = torch.nn.functional.softmax(batch_preds, dim=1)
            preds.extend(batch_preds.cpu()[:,1])
        
    meanpreds = torch.stack(preds).reshape(-1,10).mean(dim=1)
    resdf = pd.DataFrame({'selfies_str': sf_str, 'value': values, 'pred': meanpreds})
    resdf['property_token'] = prop_token
    evaldf = pd.concat([evaldf, resdf])


path = (outdir / "evaluation.parquet").as_posix()
evaldf.to_parquet(path)

evaldf = pd.read_parquet(path)

safe_roc = lambda x, y: roc_auc_score(x, y) if len(set(x)) > 1 else None
aucs = evaldf.groupby('property_token').apply(lambda x: safe_roc(x['value'], x['pred'])).reset_index()
aucs.columns = ['property_token', 'auc']
propeval = prop_src.merge(aucs, on='property_token', how='left')

# get median auc by source
med_aucs = propeval.groupby('source').apply(lambda x: x['auc'].median()).sort_values(ascending=False)
print(med_aucs)

# save median aucs by source
med_aucs.to_parquet((outdir / "median_aucs.parquet").as_posix())
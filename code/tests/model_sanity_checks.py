# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/5_1_eval_multi_properties.py

import itertools, uuid, pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.models.mixture_experts as me
import logging
from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss
from cvae.models.datasets.inmemory_sequence_shift_dataset import InMemorySequenceShiftDataset, PreloadedSequenceShiftDataset, StratifiedPropertySequenceShiftWrapper, ComprehensiveStratifiedPropertySequenceShiftWrapper
from torch.nn import functional as F
from tqdm import tqdm

import importlib
importlib.reload(me)

tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

model : me.MoE = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")
model.gating_network.all_props_tensor.device
model.train()

tmp = InMemorySequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=5)
tmpdl = torch.utils.data.DataLoader(tmp, batch_size=1, shuffle=False, num_workers=1)
tmpdl = iter(tmpdl)
eginp, egtch, egout = next(tmpdl)
model(eginp, egtch)  # [B, T, V]
model.num_forward_calls 

model.gating_network.get_routing_stats()
# {'expert_loads': array([0.06274803, 0.06076939, 0.0666521 , 0.05664491, 0.06364259,
#        0.06398162, 0.05714184, 0.06058828, 0.06497426, 0.06943446,
#        0.06012474, 0.06455599, 0.06177576, 0.06153877, 0.06236126,
#        0.06306602], dtype=float32), 
# 'avg_active_experts_per_property': 1.6202155351638794, 
# 'load_balance_std': 0.003228102345019579, 
# 'most_used_expert': 9, 
# 'least_used_expert': 3, 
# 'max_load': 0.06943446397781372, 
# 'min_load': 0.05664490535855293}

# Check if experts produce different outputs
with torch.no_grad():
    expert_outs = [expert(eginp, egtch) for expert in model.experts]
    
# Compute pairwise differences
diffs = []
for i in range(len(expert_outs)):
    for j in range(i+1, len(expert_outs)):
        diff = (expert_outs[i] - expert_outs[j]).abs().mean()
        diffs.append(diff.item())

print(f"Average expert output difference: {np.mean(diffs)}")
assert np.mean(diffs) > 0.05, "Experts are not producing different outputs!"

# CHECK LOSS FUNCTION ===========================================================================

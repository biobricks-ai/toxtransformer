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
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset
from torch.nn import functional as F
from tqdm import tqdm

import importlib
importlib.reload(mt)

tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
dataset_path = "cache/build_tensordataset/multitask_tensors/tmp"
ds = InMemorySelfiesPropertiesValuesDataset(dataset_path, tokenizer, nprops=5)
dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=1)

sf, props, vals, mask = next(iter(dl))

model = mt.ToxTransformer(tokenizer)
backbone_param = model.shared_base.classification_layers[-1].weight  # nn.Parameter

assay_tokens = sorted(tokenizer.assay_indexes().values())
prop_to_index = {p: (p - assay_tokens[0]) for p in assay_tokens}
num_tasks = len(assay_tokens)

# CHECK LOSS FUNCTION ===========================================================================

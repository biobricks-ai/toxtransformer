# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import pandas as pd
# import os
# import collections
# import torch
# import sqlite3
# import cvae.models.multitask_transformer as mt
# import cvae.tokenizer
# from tqdm import tqdm


# # setup
# pd.set_option('display.width', 150)
# pd.set_option('display.max_columns', None)
# tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

# # Load dataset
# dataset = mt.FastPackedSequenceShiftDataset("cache/pack_multitask_tensors/packed_trn")

# # Initialize counters - map property tokens to sets of input hashes
# property_value_inputs = collections.defaultdict(set)
# molecule_count = 0

# for idx in tqdm(range(len(dataset))):
#     inp, _, out = dataset[idx]
#     molecule_count += 1
    
#     # Hash the input tensor to use as unique identifier
#     input_hash = hash(inp.numpy().tobytes())
    
#     # Extract properties and values from the teaching sequence
#     j = 1  # Start after <sos>
#     while j < len(out) - 1 and out[j] != tokenizer.END_IDX and out[j] != tokenizer.PAD_IDX:
#         # Every odd position is a property token, followed by its value
#         if j % 2 == 1 and j + 1 < len(out):
#             prop_token = out[j].item()
#             value_token = out[j+1].item()
#             property_value_inputs[(prop_token, value_token)].add(input_hash)
#         j += 2  # Move to next property-value pair

# token_assays = {idx: assay for assay, idx in tokenizer.assay_indexes().items()}
# data = [{'property': p, 'value': v, 'nchem': len(inp) } for (p, v), inp in property_value_inputs.items()]
# pvdf = pd.DataFrame(data).sort_values(by='nchem', ascending=False)
# pvdf['property_token'] = pvdf['property'].astype(int)

# # Connect to SQLite database
# conn = sqlite3.connect('brick/cvae.sqlite')
# props = pd.read_sql_query("SELECT * FROM property inner join source on property.source_id = source.source_id", conn)
# minprop = min(props['property_token'])
# maxprop = max(props['property_token'])

# # assert that every property_token is in the props dataframe
# assert pvdf['property_token'].isin(props['property_token']).all()
# missing = pvdf[~pvdf['property_token'].isin(props['property_token'])]

# props = props.merge(pvdf, on='property_token', how='inner')
# props[['source','value']].value_counts()

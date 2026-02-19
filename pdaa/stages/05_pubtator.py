import pandas as pd
import os
import glob
import biobricks as bb

pubtator = bb.assets('pubtator')
# print keys in pubtator namespace
for key in pubtator.__dict__.keys():
    print(key)

pubtator_chem = pd.read_parquet(pubtator.chemical2pubtator3_parquet)
pubtator_bioC = pd.read_parquet(pubtator.BioC_parquet)

pubtator_chem.iloc[0]
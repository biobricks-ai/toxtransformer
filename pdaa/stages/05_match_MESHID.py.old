import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool
import os
import glob
import json

import biobricks as bb
tqdm.pandas()

def addCasRN(args):
    path, substances_parquet = args
    
    df = pd.read_parquet(path)
    
    mesh_df = pd.merge(substances_parquet, df, on='sid')
    output_path = path.replace('raw/harmonized_phthalates/', 'raw/CTD_MESH_phthalates/')
    mesh_df = mesh_df.dropna(subset=['CasRN'])
    mesh_df = mesh_df.drop_duplicates(subset=['sid'])
    mesh_df.to_parquet(output_path, index=False)
    
def addMESH(args):
    path, CTD_chemicals_parquet = args
    
    df = pd.read_parquet(path)
    
    mesh_df = pd.merge(CTD_chemicals_parquet, df, on='CasRN')
    mesh_df.to_parquet(path, index=False)
    
def processInParallel(argsm, function, n=16):
    with Pool(n) as pool:
        list(tqdm(pool.imap(function, args), total=len(file_paths)))
    

def getMESHID(json_string):
    try:
        json_data = json.loads(json_string)
        if type(json_data['ChemicalID']) == list:
            return json_data['ChemicalID'][0]
        else:
            return json_data['ChemicalID']
    except:
        return None
    
def getCasRN(json_string):
    try:
        json_data = json.loads(json_string)
        for key in ['casrn', 'CasRN', 'CAS']:
            if key in json_data:
                if type(json_data[key]) == list:
                    return json_data[key][0]
                else:
                    return json_data[key]
        return None
    except:
        return None

chemharmony = bb.assets('chemharmony')
substances_parquet = pd.read_parquet(chemharmony.substances_parquet)
substances_parquet['CasRN'] = substances_parquet['data'].progress_apply(lambda x: getCasRN(x))

os.makedirs('raw/CTD_MESH_phthalates', exist_ok=True)
file_paths = glob.glob('raw/harmonized_phthalates/*.parquet')
args = [(path, substances_parquet) for path in file_paths]
processInParallel(args, addCasRN)

ctdbase = bb.assets('ctdbase')
CTD_chemicals_parquet = pd.read_parquet(ctdbase.CTD_chemicals_parquet)[['ChemicalID', 'CasRN']]
CTD_chemicals_parquet = CTD_chemicals_parquet.dropna(subset='CasRN')
CTD_chemicals_parquet = CTD_chemicals_parquet.drop_duplicates(subset='CasRN')
file_paths = glob.glob('raw/CTD_MESH_phthalates/*.parquet')
args = [(path, CTD_chemicals_parquet) for path in file_paths]
processInParallel(args, addMESH)

    
    
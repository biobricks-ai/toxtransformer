import pandas as pd, biobricks as bb, glob, time, pathlib
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool

tqdm.pandas()

outdir = pathlib.Path('cache/chemarmony_properties')
outdir.mkdir(exist_ok=True)

chemharmony = bb.assets('chemharmony')
chemharmony = pd.read_parquet(chemharmony.activities_parquet)

phthalates = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')
phthalates['inchi'] = phthalates['smiles'].progress_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)

# join on inchi
joined = phthalates.merge(chemharmony, on='inchi', how='inner')

outfile = outdir / 'phthalates_chemharmony.parquet'
joined.to_parquet(outfile)
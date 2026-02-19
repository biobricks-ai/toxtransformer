# process all phthalates and find any that are highly similar to example_phthalates
# deps = [ cache/ ]
# outs = [ cache/priority_phthalates/priority_phthalates.parquet ]
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.DataStructs, rdkit.Chem.rdFingerprintGenerator
import boltons.funcutils, pandas as pd, pathlib
from tqdm import tqdm
tqdm.pandas()

outdir = pathlib.Path('cache/priority_phthalates')
outdir.mkdir(parents=True, exist_ok=True)

dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')
example_phthalates = [dehp, diup, dtdp, didp, dinp]
morgan = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
example_fps = [morgan.GetFingerprint(mol) for mol in example_phthalates]

# 1. go through all phthalates
# 2. filter for top 1000 most similar
# 3. save to parquet

phthalates = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')

@boltons.funcutils.lru_cache(default=0)
def get_max_similarity(mol_smiles):

    mol = rdkit.Chem.MolFromSmiles(mol_smiles)
    if mol is None: return 0.0
    
    # Calculate fingerprint for test molecule
    fp = morgan.GetFingerprint(mol)
    
    # Calculate similarity to each example and take max
    sims = [rdkit.DataStructs.TanimotoSimilarity(fp, ex_fp) for ex_fp in example_fps]
    return max(sims)

# Calculate similarities
phthalates['max_similarity'] = phthalates['smiles'].progress_apply(get_max_similarity)
phthalates.to_parquet(outdir / 'priority_phthalates.parquet')
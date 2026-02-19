import pathlib
import shutil
from pyspark.sql import SparkSession
import biobricks as bb
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

# TODO start moving bricks to iceberg and do this in cloud. 

# REPARTITION ZINC INTO LOCAL TEMPORARY DIRECTORY ==========================================
# Set up directories and Spark session
outdir = pathlib.Path('cache/zinc_phthalates')
outdir.mkdir(parents=True, exist_ok=True)

# Initialize Spark session with optimized settings for shuffling and partitioning
spark = (SparkSession.builder
         .appName("zinc_phthalates_partitioning")
         .config("spark.driver.memory", "32g")  # More memory for driver
         .config("spark.executor.memory", "32g")  # More memory for executors
         .config("spark.sql.shuffle.partitions", "1000")  # Matching target file count
         .config("spark.driver.maxResultSize", "32g")  # Increased max result size
         .getOrCreate())

# Path to the ZINC dataset and temp output directory
zinc_path = pathlib.Path(bb.assets('zinc').zinc_parquet)
tempdir = outdir / 'temp'
tempdir.mkdir(exist_ok=True)

# Read the original dataset
zinc_df = spark.read.parquet(str(zinc_path))

# Repartition the DataFrame into 1000 partitions to create 1000 smaller files
zinc_df = zinc_df.repartition(1000)

# Write the repartitioned DataFrame to the temp directory
zinc_df.write.mode("overwrite").parquet(str(tempdir / 'zinc_partitioned'))

print("Data has been successfully repartitioned and saved to 1000 smaller parquet files.")

# FILTER FOR PHTHALATES ======================================================================
temp_phthalates = tempdir / 'temp_phthalates'
shutil.rmtree(temp_phthalates, ignore_errors=True) 
temp_phthalates.mkdir(exist_ok=True)

def has_substructure(smiles, substructure):
    if pd.isna(smiles):  # Check for NaN/NA values
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return bool(mol.HasSubstructMatch(substructure)) if mol else False
    except:
        return False

# process each file from partitioned zinc 
# output to `temp_phthalates`
def process_file_with_index(index_parquet_file_tuple):
    index, parquet_file = index_parquet_file_tuple
    outfile = temp_phthalates / f'phthalates_{index}.parquet'

    try:
        df = pd.read_parquet(parquet_file)
        substructure = Chem.MolFromSmiles('OC(=O)C1=CC=CC=C1C(=O)O')

        # Filter for phthalates
        df['phthalate'] = df['smiles'].apply(lambda x: has_substructure(x, substructure))
        phthalates_df = df[df['phthalate'] == True]  # Explicit boolean comparison
        
        # Save intermediate results if we found any phthalates
        if len(phthalates_df) > 0:
            phthalates_df.to_parquet(outfile)
    except Exception as e:
        print(f"Error processing file {parquet_file}: {str(e)}")


files = list((tempdir / 'zinc_partitioned').glob('*.parquet'))
with ProcessPoolExecutor(max_workers=30) as executor:
    file_tuples = list(enumerate(files))
    list(tqdm(
        executor.map(process_file_with_index, file_tuples),
        total=len(files),
        desc='Processing files'
    ))

# Combine all results
phthalates_files = list(temp_phthalates.glob('*.parquet'))
final_df = pd.concat([pd.read_parquet(f) for f in phthalates_files])

# generate inchi for all phthalates
final_df['inchi'] = final_df['smiles'].progress_apply(lambda x: Chem.MolToInchi(Chem.MolFromSmiles(x)) if pd.notnull(x) else None)

final_df.to_parquet(outdir / 'zinc_phthalates.parquet')

final_df = pd.read_parquet(outdir / 'zinc_phthalates.parquet')
final_df.to_parquet(outdir / 'zinc_phthalates.parquet')

# Cleanup temporary files
# TODO actually execute a cleanup
# shutil.rmtree(tempdir)

# cmd: 
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events --conf spark.local.dir=/data/tmp/spark-local code/1_3_preprocess_activities_augmented.py

import biobricks
import cvae
import cvae.utils
import logging
import pandas as pd
import pathlib
import pyspark.ml.feature
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_json, first, udf, array, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
from pyspark.sql.window import Window
from rdkit import Chem
from rdkit.Chem import AllChem
import random

# set up logging
logdir = pathlib.Path('cache/preprocess_activities/log')
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   filename=logdir / 'preprocess_activities_augmented.log', filemode='w')

outdir = pathlib.Path('cache/preprocess_activities')
outdir.mkdir(parents=True, exist_ok=True)

# set up spark session
spark = cvae.utils.get_spark_session()

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================
import selfies as sf
import pickle

def generate_alternative_smiles(smiles, num_alternatives=5, max_attempts=50, seed=42):
    """
    Generate alternative valid SMILES representations of the same molecule.
    Returns a list including the original SMILES plus alternatives.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]  # Return original if parsing fails
        
        # Set seed for reproducibility per molecule
        # Use hash of SMILES to ensure same alternatives are generated for same molecule
        mol_seed = hash(smiles) % (2**32) + seed
        random.seed(mol_seed)
        
        alternatives = set()
        alternatives.add(smiles)  # Include original
        
        attempts = 0
        while len(alternatives) < num_alternatives + 1 and attempts < max_attempts:
            # Randomize atom order to get different SMILES
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, new_atom_order)
            
            # Generate non-canonical SMILES with randomization
            random_smiles = Chem.MolToSmiles(random_mol, canonical=False, doRandom=True)
            
            # Verify the SMILES is valid by parsing it
            if Chem.MolFromSmiles(random_smiles) is not None:
                alternatives.add(random_smiles)
            
            attempts += 1
        
        return list(alternatives)
        
    except Exception as e:
        logging.warning(f"Augmentation failed for SMILES {smiles}: {str(e)}")
        return [smiles]

def generate_alternative_selfies_and_encode(smiles, tokenizer, num_alternatives=5, max_attempts=50, seed=42):
    """
    Generate alternative SMILES, convert to SELFIES, and encode them.
    Returns a list of encoded SELFIES arrays.
    """
    try:
        # Get alternative SMILES
        alternative_smiles = generate_alternative_smiles(smiles, num_alternatives, max_attempts, seed)
        
        # Convert each SMILES to SELFIES and encode
        encoded_alternatives = []
        for alt_smiles in alternative_smiles:
            try:
                selfies_str = sf.encoder(alt_smiles)
                if selfies_str:
                    # Use selfies_to_indices method from your tokenizer
                    encoded = tokenizer.selfies_to_indices(selfies_str)
                    encoded_alternatives.append(encoded)
            except Exception as e:
                logging.debug(f"Failed to encode SMILES {alt_smiles}: {e}")
                continue
        
        # If no alternatives could be encoded, return empty list
        if not encoded_alternatives:
            logging.warning(f"Could not encode any alternatives for SMILES: {smiles}")
        
        return encoded_alternatives
        
    except Exception as e:
        logging.warning(f"Alternative generation failed for SMILES {smiles}: {str(e)}")
        return []

# Load tokenizer for use in UDF
import cvae.tokenizer.selfies_tokenizer
tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer.load('cache/preprocess_tokenizer/selfies_tokenizer.json')

# Broadcast tokenizer to all workers
tokenizer_broadcast = spark.sparkContext.broadcast(tokenizer)

# Create UDF for generating alternative SELFIES (as strings for inspection)
def generate_alternative_selfies_strings(smiles, num_alternatives=3):
    """Generate alternative SELFIES strings for inspection/debugging."""
    alternative_smiles = generate_alternative_smiles(smiles, num_alternatives)
    alternative_selfies = []
    for alt_smiles in alternative_smiles:
        try:
            selfies_str = sf.encoder(alt_smiles)
            if selfies_str:
                alternative_selfies.append(selfies_str)
        except Exception as e:
            logging.debug(f"Failed to encode SMILES to SELFIES: {alt_smiles}, error: {e}")
            continue
    
    # If we have some alternatives, return them
    if alternative_selfies:
        return alternative_selfies
    
    # If no alternatives worked, try the original SMILES
    try:
        original_selfies = sf.encoder(smiles)
        if original_selfies:
            return [original_selfies]
    except Exception as e:
        logging.warning(f"Could not encode original SMILES to SELFIES: {smiles}, error: {e}")
    
    # If nothing worked, return empty list
    return []

generate_alt_selfies_udf = udf(
    lambda smiles: generate_alternative_selfies_strings(smiles, num_alternatives=3),
    ArrayType(StringType())
)

# Create UDF for generating encoded alternatives
def generate_encoded_alternatives(smiles, num_alternatives=3):
    """Generate alternative SMILES, convert to SELFIES, and encode them."""
    tok = tokenizer_broadcast.value
    return generate_alternative_selfies_and_encode(smiles, tok, num_alternatives)

# This UDF returns an array of arrays (each inner array is an encoded SELFIES)
from pyspark.sql.types import ArrayType, IntegerType
generate_encoded_alts_udf = udf(
    lambda smiles: generate_encoded_alternatives(smiles, num_alternatives=3),
    ArrayType(ArrayType(IntegerType()))
)

# =============================================================================
# LOAD AND PREPROCESS DATA
# =============================================================================

logging.info("Loading ChemHarmony activities data...")
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(['source','smiles','pid','sid','binary_value'])
substances = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')
data = activities.join(substances.select('sid','selfies','encoded_selfies'), 'sid', 'inner')
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')

# =============================================================================
# GENERATE ALTERNATIVE SELFIES REPRESENTATIONS AND ENCODINGS
# =============================================================================

logging.info("Generating alternative SELFIES representations and encodings...")

# First, get unique SMILES to avoid redundant augmentation
unique_smiles = data.select('smiles').distinct()

# Generate alternative SELFIES strings (for inspection)
unique_smiles_with_selfies = unique_smiles.withColumn(
    'alternative_selfies', 
    generate_alt_selfies_udf(col('smiles'))
)

# Generate encoded alternatives (what we'll actually use)
unique_smiles_with_encodings = unique_smiles_with_selfies.withColumn(
    'alternative_encoded_selfies',
    generate_encoded_alts_udf(col('smiles'))
)

# Cache this as it will be reused
unique_smiles_with_encodings.cache()

# Log some statistics about augmentation
total_unique = unique_smiles_with_encodings.count()
avg_alternatives = unique_smiles_with_encodings.select(
    F.avg(F.size('alternative_selfies')).alias('avg_alternatives')
).collect()[0]['avg_alternatives']

avg_encoded = unique_smiles_with_encodings.filter(F.size('alternative_encoded_selfies') > 0).select(
    F.avg(F.size('alternative_encoded_selfies')).alias('avg_encoded')
).collect()[0]['avg_encoded']

logging.info(f"Generated alternatives for {total_unique} unique SMILES")
logging.info(f"Average number of SELFIES alternatives per molecule: {avg_alternatives:.2f}")
logging.info(f"Average number of successfully encoded alternatives: {avg_encoded:.2f}")

# Join alternatives back to main data
data = data.join(
    unique_smiles_with_encodings.select('smiles', 'alternative_selfies', 'alternative_encoded_selfies'), 
    'smiles', 
    'left'
)

# For molecules where augmentation failed, ensure we have at least the original encoded SELFIES
# Also filter out any empty alternative arrays
data = data.withColumn(
    'alternative_selfies',
    F.when(
        F.col('alternative_selfies').isNull() | (F.size('alternative_selfies') == 0),
        F.array()  # Empty array if no alternatives
    ).otherwise(F.col('alternative_selfies'))
)

data = data.withColumn(
    'alternative_encoded_selfies',
    F.when(
        F.col('alternative_encoded_selfies').isNull() | (F.size('alternative_encoded_selfies') == 0),
        F.array(F.col('encoded_selfies'))  # Use original if no alternatives could be encoded
    ).otherwise(F.col('alternative_encoded_selfies'))
)

# =============================================================================
# CONTINUE WITH ORIGINAL PREPROCESSING
# =============================================================================

data = data.orderBy(F.rand(52)) # Randomly shuffle data with seed 52
data.cache()

initial_count = data.count()

source_counts = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
logging.info(str(source_counts))
assert min(source_counts['count']) >= 1000

## FILTER TO USEFUL OR CLASSIFIABLE PROPERTIES
logging.info("Filtering assays based on number of unique selfies")

assay_counts = data.groupBy('assay').agg(F.countDistinct('selfies').alias('unique_selfies_count'))
assay_counts = assay_counts.filter(F.col('unique_selfies_count') >= 100)
data = data.join(assay_counts, 'assay', 'inner')

logging.info(f"Data size after filtering: {data.count():,} rows ({(data.count() / initial_count * 100):.1f}% of initial)")

# Filter to assays with at least 50 positives and 50 negatives
logging.info("Filtering assays for balanced positive/negative samples")
assay_balance = data.groupBy('assay').agg(
    F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
    F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
)

# Filter assays with at least 50 positives AND 50 negatives
balanced_assays = assay_balance.filter((F.col('positive_count') >= 50) & (F.col('negative_count') >= 50))
data = data.join(balanced_assays.select('assay'), 'assay', 'inner')

# Check for conflicts (same SELFIES-assay pair with different values)
conflicts = data.groupBy('selfies', 'assay').agg(
    F.countDistinct('value').alias('distinct_values'),
    F.count('*').alias('duplicate_count')
).filter(F.col('distinct_values') > 1)

conflicting_pairs = conflicts.select('selfies', 'assay')
data = data.join(conflicting_pairs, on=['selfies', 'assay'], how='left_anti')

## Map assay UUIDs to integers
logging.info("Converting assay IDs to indices...")
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

# =============================================================================
# WRITE AUGMENTED DATA
# =============================================================================

## write out the processed data with alternative SMILES
logging.info("Writing processed data with augmentation to parquet...")
data.write.parquet((outdir / 'activities_augmented.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'activities_augmented.parquet'}")

## TEST count activities
data = spark.read.parquet((outdir / 'activities_augmented.parquet').as_posix())
assert data.count() > 10e6 # should have more than 10m activities

# Verify alternative encodings exist and have expected content
sample_with_alts = data.filter(F.size('alternative_encoded_selfies') > 1).limit(5).collect()
logging.info("Sample of augmented SELFIES:")
for row in sample_with_alts:
    logging.info(f"  Original SMILES: {row['smiles']}")
    logging.info(f"  Alternative SELFIES strings ({len(row['alternative_selfies'])}): {row['alternative_selfies'][:2]}...")
    logging.info(f"  Alternative encodings count: {len(row['alternative_encoded_selfies'])}")
    if row['alternative_encoded_selfies']:
        logging.info(f"    First encoding length: {len(row['alternative_encoded_selfies'][0])}")

source_counts_2 = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()

# Calculate percentage drop for each source
source_counts_2['initial_count'] = source_counts_2['source'].map(source_counts.set_index('source')['count'])
source_counts_2['pct_drop'] = ((source_counts_2['initial_count'] - source_counts_2['count']) / source_counts_2['initial_count'] * 100)
source_counts_2['pct_remaining'] = 100 - source_counts_2['pct_drop']
logging.info("Source counts after filtering:")
logging.info(f"\n{source_counts_2.to_string(index=False, float_format=lambda x: '{:.1f}'.format(x))}")
assert min(source_counts_2['count']) >= 1000

# Log augmentation statistics
aug_stats = data.select(
    F.min(F.size('alternative_encoded_selfies')).alias('min_alts'),
    F.max(F.size('alternative_encoded_selfies')).alias('max_alts'),
    F.avg(F.size('alternative_encoded_selfies')).alias('avg_alts')
).collect()[0]

logging.info("=== AUGMENTATION STATISTICS ===")
logging.info(f"Min encoded alternatives per molecule: {aug_stats['min_alts']}")
logging.info(f"Max encoded alternatives per molecule: {aug_stats['max_alts']}")
logging.info(f"Avg encoded alternatives per molecule: {aug_stats['avg_alts']:.2f}")
logging.info(f"Successfully added alternative encoded SELFIES to {data.count()} activity records")
#!/usr/bin/env python
"""
Preprocess ChemHarmony activities with SMILES augmentation for improved model training.

Usage:
PYTHONPATH=./ spark-submit --master local[240] \
    --driver-memory 512g \
    --conf spark.memory.fraction=0.6 \
    --conf spark.memory.storageFraction=0.3 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.sql.adaptive.coalescePartitions.enabled=true \
    --conf spark.sql.shuffle.partitions=400 \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:///data/tmp/spark-events \
    --conf spark.local.dir=/data/tmp/spark-local \
    code/1_3_preprocess_activities.py
"""

import biobricks
import cvae
import cvae.utils
import cvae.tokenizer.selfies_tokenizer
import cvae.tokenizer.selfies_property_val_tokenizer as spt
import logging
import pandas as pd
import pathlib
import pyspark.ml.feature
import random
import selfies as sf
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, array
from pyspark.sql.types import ArrayType, IntegerType, StringType
from rdkit import Chem

# Setup
logdir = pathlib.Path('cache/preprocess_activities/log')
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=logdir / 'preprocess_activities_augmented.log', 
    filemode='w'
)

outdir = pathlib.Path('cache/preprocess_activities')
outdir.mkdir(parents=True, exist_ok=True)

# Initialize Spark
spark = cvae.utils.get_spark_session()
# Enable adaptive query execution for better performance
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def generate_alternative_smiles(smiles, num_alternatives=3, max_attempts=30):
    """Generate alternative SMILES representations of the same molecule."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [smiles]
        
        # Deterministic seed based on SMILES
        mol_seed = hash(smiles) % (2**32)
        random.seed(mol_seed)
        
        alternatives = {smiles}  # Include original
        attempts = 0
        
        while len(alternatives) < num_alternatives + 1 and attempts < max_attempts:
            # Randomize atom order
            new_atom_order = list(range(mol.GetNumAtoms()))
            random.shuffle(new_atom_order)
            random_mol = Chem.RenumberAtoms(mol, new_atom_order)
            
            # Generate non-canonical SMILES
            random_smiles = Chem.MolToSmiles(random_mol, canonical=False, doRandom=True)
            
            # Verify validity
            if Chem.MolFromSmiles(random_smiles):
                alternatives.add(random_smiles)
            
            attempts += 1
        
        return list(alternatives)
    except:
        return [smiles]

def generate_encoded_alternatives(smiles, tokenizer, num_alternatives=3):
    """Generate alternative SMILES, convert to SELFIES, and encode."""
    alternative_smiles = generate_alternative_smiles(smiles, num_alternatives)
    encoded_alternatives = []
    
    for alt_smiles in alternative_smiles:
        try:
            selfies_str = sf.encoder(alt_smiles)
            if selfies_str:
                encoded = tokenizer.selfies_to_indices(selfies_str)
                encoded_alternatives.append(encoded)
        except:
            continue
    
    return encoded_alternatives if encoded_alternatives else []

# =============================================================================
# MAIN PROCESSING
# =============================================================================

logging.info("Loading ChemHarmony activities data...")

# Load data
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(
    'source', 'smiles', 'pid', 'sid', 'binary_value'
)
substances = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')

# Join and rename columns
data = activities.join(
    substances.select('sid', 'selfies', 'encoded_selfies'), 
    'sid', 
    'inner'
)
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')

# Shuffle data with seed for reproducibility
data = data.orderBy(F.rand(52))

# Don't cache the full dataset - process in streaming fashion
initial_count = data.count()
logging.info(f"Initial data count: {initial_count:,}")

# =============================================================================
# FILTER ASSAYS
# =============================================================================

logging.info("Filtering assays...")

# Filter to assays with >= 100 unique molecules
assay_counts = data.groupBy('assay').agg(
    F.countDistinct('selfies').alias('unique_selfies_count')
).filter(F.col('unique_selfies_count') >= 100)

data = data.join(assay_counts.select('assay'), 'assay', 'inner')

# Filter to balanced assays (>= 50 positives and >= 50 negatives)
assay_balance = data.groupBy('assay').agg(
    F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
    F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
).filter((F.col('positive_count') >= 50) & (F.col('negative_count') >= 50))

data = data.join(assay_balance.select('assay'), 'assay', 'inner')

# Remove conflicting data points (same molecule-assay with different labels)
conflicts = data.groupBy('selfies', 'assay').agg(
    F.countDistinct('value').alias('distinct_values')
).filter(F.col('distinct_values') > 1).select('selfies', 'assay')

data = data.join(conflicts, on=['selfies', 'assay'], how='left_anti')

# Convert assay IDs to indices
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

filtered_count = data.count()
logging.info(f"Data after filtering: {filtered_count:,} ({filtered_count/initial_count*100:.1f}%)")

# =============================================================================
# AUGMENTATION - Process in batches to manage memory
# =============================================================================

logging.info("Adding SMILES augmentation...")

# Load and broadcast tokenizer
tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer.load(
    'cache/preprocess_tokenizer/selfies_tokenizer.json'
)
tokenizer_broadcast = spark.sparkContext.broadcast(tokenizer)

# Create UDF for encoded alternatives
generate_encoded_alts_udf = udf(
    lambda smiles: generate_encoded_alternatives(
        smiles, tokenizer_broadcast.value, num_alternatives=3
    ),
    ArrayType(ArrayType(IntegerType()))
)

# Process unique SMILES separately to avoid redundant augmentation
unique_smiles = data.select('smiles').distinct()

# Generate augmentations without caching intermediate results
unique_smiles_augmented = unique_smiles.withColumn(
    'alternative_encoded_selfies',
    generate_encoded_alts_udf(col('smiles'))
)

# Join back to main data without caching
data = data.join(
    unique_smiles_augmented,
    'smiles',
    'left'
)

# Handle failed augmentations - use original encoding
data = data.withColumn(
    'alternative_encoded_selfies',
    F.when(
        F.col('alternative_encoded_selfies').isNull() | 
        (F.size('alternative_encoded_selfies') == 0),
        F.array(F.col('encoded_selfies'))
    ).otherwise(F.col('alternative_encoded_selfies'))
)

# =============================================================================
# WRITE OUTPUT
# =============================================================================

logging.info("Writing augmented data...")

# Write directly without caching - use coalesce to reduce number of output files
output_path = (outdir / 'activities_augmented.parquet').as_posix()
data.coalesce(200).write.parquet(output_path, mode='overwrite')

logging.info(f"Wrote augmented data to {output_path}")

# =============================================================================
# CREATE PROPERTY VALUE TOKENIZER
# =============================================================================

logging.info("Creating SelfiesPropertyValTokenizer...")

# Create and save the property value tokenizer
# This is needed for the next step in the pipeline (2_2_build_sqlite.py)
tokenizer_dir = pathlib.Path('brick/selfies_property_val_tokenizer')
tokenizer_dir.mkdir(parents=True, exist_ok=True)

# Load the augmented data to get assay information
augmented_data = spark.read.parquet(output_path)

# Get the maximum assay index (number of assays)
num_assays = int(augmented_data.agg(F.max('assay_index')).collect()[0][0] + 1)
# For binary classification, num_values is 2 (0 and 1)
num_values = 2

# Create the tokenizer with the selfies tokenizer and assay counts
selfies_tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer.load(
    'cache/preprocess_tokenizer/selfies_tokenizer.json'
)
property_val_tokenizer = spt.SelfiesPropertyValTokenizer(
    selfies_tokenizer, 
    num_assays, 
    num_values
)

# Save the tokenizer
property_val_tokenizer.save(tokenizer_dir)
logging.info(f"Saved SelfiesPropertyValTokenizer to {tokenizer_dir}")

# =============================================================================
# VALIDATION
# =============================================================================

logging.info("Validating output...")

# Read back and validate
final_data = spark.read.parquet(output_path)
final_count = final_data.count()

assert final_count > 10e6, f"Expected >10M activities, got {final_count}"

# Check augmentation stats
aug_stats = final_data.select(
    F.min(F.size('alternative_encoded_selfies')).alias('min_alts'),
    F.max(F.size('alternative_encoded_selfies')).alias('max_alts'),
    F.avg(F.size('alternative_encoded_selfies')).alias('avg_alts')
).collect()[0]

logging.info(f"=== FINAL STATISTICS ===")
logging.info(f"Total activities: {final_count:,}")
logging.info(f"Encoded alternatives per molecule: min={aug_stats['min_alts']}, "
             f"max={aug_stats['max_alts']}, avg={aug_stats['avg_alts']:.2f}")

# Check source distribution
source_counts = final_data.groupBy('source').count().orderBy('count', ascending=False)
source_df = source_counts.toPandas()
logging.info(f"Source distribution:\n{source_df.to_string(index=False)}")
assert min(source_df['count']) >= 1000, "Some sources have too few activities"

logging.info("Processing complete!")

# Clean up
spark.stop()
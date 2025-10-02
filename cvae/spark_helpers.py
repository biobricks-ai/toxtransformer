from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType, ArrayType, IntegerType, StringType
from rdkit import Chem, RDLogger
import selfies
from selfies.exceptions import EncoderError, SMILESParserError
import cvae.models.tokenizer as tokenizer
import logging
import random
import socket

# Suppress RDKit warnings globally
RDLogger.DisableLog('rdApp.*')

# Setup executor-local logger
host = socket.gethostname()
logger = logging.getLogger(f"spark_executor_{host}")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)

def _log_sample(msg, sample=0.0001):
    if random.random() < sample:
        logger.info(msg)

# --- SMILES validity check ---
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None and mol.GetNumAtoms() > 0
    except:
        return False

is_valid_smiles_udf = udf(is_valid_smiles, BooleanType())

# --- SMILES to Morgan fingerprint ---
def smiles_to_morgan(smiles, nBits=2048):
    fp = tokenizer.smiles_to_morgan_fingerprint(smiles, nBits=nBits)
    return fp.tolist() if fp is not None else []

smiles_to_morgan_udf = udf(smiles_to_morgan, ArrayType(IntegerType()))

# --- InChI to SMILES ---
def inchi_to_smiles_safe(inchi):
    try:
        if inchi:
            mol = Chem.MolFromInchi(inchi)
            if mol:
                _log_sample(f"[{host}] InChI→SMILES working")
                return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.warning(f"[{host}] InChI→SMILES error: {e}")
    return None

inchi_to_smiles_udf = udf(inchi_to_smiles_safe, StringType())

# --- SMILES to SELFIES ---
def smiles_to_selfies_safe(smiles):
    try:
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                _log_sample(f"[{host}] SMILES→SELFIES working")
                return selfies.encoder(smiles)
    except (SMILESParserError, EncoderError):
        return None
    except Exception as e:
        logger.warning(f"[{host}] SMILES→SELFIES error: {e}")
    return None

smiles_to_selfies_udf = udf(smiles_to_selfies_safe, StringType())

# --- SELFIES splitter ---
def split_selfies(selfies_string):
    if selfies_string:
        _log_sample(f"[{host}] Splitting SELFIES")
        return selfies.split_selfies(selfies_string)
    return []

split_selfies_udf = udf(split_selfies, ArrayType(StringType()))

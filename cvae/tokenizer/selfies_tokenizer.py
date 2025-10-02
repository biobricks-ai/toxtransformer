import json
from pyspark.sql.types import BooleanType, ArrayType, IntegerType, FloatType, StringType
import pyspark.sql.functions as F
import logging
import socket
import random
import selfies as sf

# Setup executor-local logger
host = socket.gethostname()
logger = logging.getLogger(f"spark_executor_{host}")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)

def split_selfies(selfies_string, sample=0.0001):
    if selfies_string:
        if random.random() < sample:
            logger.info(f"[{host}] Splitting SELFIES")
        return sf.split_selfies(selfies_string)
    return []

class SelfiesTokenizer:
    
    PAD_TOKEN = '<pad>'
    SOS_TOKEN = '<sos>'
    END_TOKEN = '<eos>'
    
    def __init__(self):
        # Initialize with special tokens
        self.special_tokens = [SelfiesTokenizer.PAD_TOKEN, SelfiesTokenizer.SOS_TOKEN, SelfiesTokenizer.END_TOKEN]
        self.symbol_to_index = {token: i for i, token in enumerate(self.special_tokens)}
        self.index_to_symbol = {i: token for i, token in enumerate(self.special_tokens)}

    def fit(self, dataset, column):
        # Extract unique symbols from the dataset in a distributed manner
        unique_symbols_rdd = dataset.select(column).rdd \
            .flatMap(lambda row: split_selfies(row[column], sample=0.0001)) \
            .distinct()

        # Collect unique symbols to the driver and merge with existing mappings
        unique_symbols = unique_symbols_rdd.collect()
        start_idx = len(self.symbol_to_index)  # Start indexing after special tokens
        new_mappings = {symbol: idx + start_idx for idx, symbol in enumerate(unique_symbols) if symbol not in self.symbol_to_index}
        self.symbol_to_index.update(new_mappings)
        self.index_to_symbol = {idx: symbol for symbol, idx in self.symbol_to_index.items()}

        return self

    def selfies_to_indices(self, selfies_string):
        symbols = [self.SOS_TOKEN] + list(sf.split_selfies(selfies_string)) + [self.END_TOKEN]
        indices = [self.symbol_to_index.get(symbol, self.symbol_to_index[self.PAD_TOKEN]) for symbol in symbols]
        return indices
            
    def transform(self, dataset, selfies_column, new_column, pad_length=120, sample_rate=0.0001):
        # Function to convert selfies string to indices and pad
        def selfies_to_indices(selfies_string):
            if random.random() < sample_rate:
                logger.info(f"[{host}] Transforming SELFIES")
            if selfies_string is not None:
                symbols = [self.SOS_TOKEN] + list(sf.split_selfies(selfies_string)) + [self.END_TOKEN]
                indices = [self.symbol_to_index.get(symbol, self.symbol_to_index[self.PAD_TOKEN]) for symbol in symbols]
                padded_indices = indices[:pad_length] + [self.symbol_to_index[self.PAD_TOKEN]] * max(0, pad_length - len(indices))
                return padded_indices[:pad_length]
            else:
                return [self.symbol_to_index[self.PAD_TOKEN]] * pad_length

        selfies_to_indices_udf = F.udf(selfies_to_indices, ArrayType(IntegerType()))

        # Apply the function to the dataset
        return dataset.withColumn(new_column, selfies_to_indices_udf(F.col(selfies_column)))

    def indexes_to_selfies(self, indexes):
        symbols = [self.index_to_symbol[i] for i in indexes]
        symbols = [ s for s in symbols if not s in self.special_tokens ]
        return ''.join(symbols)
    
    def indexes_to_smiles(self, indexes):
        symbols = [self.index_to_symbol[i] for i in indexes]
        symbols = [ s for s in symbols if not s in self.special_tokens ]
        return sf.decoder(''.join(symbols))
    
    def save(self, filepath):
        with open(filepath, 'w') as file:
            json.dump({
                'symbol_to_index': self.symbol_to_index,
                'index_to_symbol': self.index_to_symbol
            }, file)
        return filepath
            
    @staticmethod
    def load(filepath):
        tokenizer = SelfiesTokenizer()
        with open(filepath, 'r') as file:
            data = json.load(file)
            tokenizer.symbol_to_index = data['symbol_to_index']
            tokenizer.index_to_symbol = data['index_to_symbol']
            tokenizer.index_to_symbol = {int(k):v for k,v in tokenizer.index_to_symbol.items()}
        return tokenizer
            

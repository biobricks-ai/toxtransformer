import torch, json, pathlib, numpy as np
from cvae.tokenizer.selfies_tokenizer import SelfiesTokenizer

# This tokenizer combines SELFIES encoding with property and value information.
# It extends the SELFIES tokenization by adding tokens for assays and their values.
# The tokenization process includes:
# 1. Encoding the SELFIES string
# 2. Adding a separator token
# 3. Encoding assay IDs and their corresponding values
# 4. Adding an end token
# This allows for joint representation of molecular structure and associated properties.

class SelfiesPropertyValTokenizer:
    
    def __init__(self, selfies_tokenizer: SelfiesTokenizer, num_assays, num_vals):
        self.selfies_tokenizer = selfies_tokenizer
        self.pad_idx = selfies_tokenizer.symbol_to_index[selfies_tokenizer.PAD_TOKEN]
        self.selfies_offset = len(self.selfies_tokenizer.symbol_to_index)
        self.num_assays = num_assays
        self.properties_offset = self.selfies_offset + num_assays
        self.num_vals = num_vals
        
        self.PAD_TOKEN = self.selfies_tokenizer.PAD_TOKEN
        self.PAD_IDX = self.selfies_tokenizer.symbol_to_index[self.PAD_TOKEN]
        self.SEP_IDX = self.selfies_offset + num_assays + num_vals
        self.END_IDX = self.selfies_offset + num_assays + num_vals + 1 
        self.vocab_size = self.selfies_offset + num_assays + num_vals + 2
    
    def value_indexes(self) -> dict:
        "returns a dictionary from value token to its tokenizer index"
        vals = range(self.num_vals)
        idxs = [self.value_id_to_token_idx(x) for x in vals]
        return {x: i for x, i in zip(vals, idxs)}
    
    def assay_indexes(self) -> dict:
        "returns a dictionary from assay token to its tokenizer index"
        assays = [x for x in range(self.num_assays)]
        idxs = [self.assay_id_to_token_idx(x) for x in assays]
        return {f"assay_{assay_id}": token_id for assay_id, token_id in zip(assays, idxs)}
    
    def assay_id_to_token_idx(self, assay_id):
        return self.selfies_offset + assay_id
    
    def value_id_to_token_idx(self, value_id):
        return self.selfies_offset + self.num_assays + value_id
    
    def tokenize(self, raw_selfies_encoding, assay_vals):
        
        selfies_pad_length = len(raw_selfies_encoding)
        selfies_encoding = torch.LongTensor([x for x in raw_selfies_encoding if x != self.pad_idx]) # remove padding
        
        assay_val_tensor = self.tokenize_assay_values(assay_vals)
        
        tensor = torch.cat([selfies_encoding, assay_val_tensor])
        padded = torch.cat([tensor, torch.LongTensor([self.pad_idx] * (selfies_pad_length + self.num_assays - len(tensor)))])
        
        return padded
    
    def tokenize_assay_values(self, assay_vals):
        assay_val_tokens = [(self.assay_id_to_token_idx(int(x.assay_index)), self.value_id_to_token_idx(int(x.value))) for x in assay_vals]
        assay_val_tensor = torch.LongTensor([x for tupl in assay_val_tokens for x in tupl]) # flatten
        return torch.cat([torch.LongTensor([self.SEP_IDX]), assay_val_tensor, torch.LongTensor([self.END_IDX])])
        
    def symbol_to_index(self, symbol):
        if symbol in self.selfies_tokenizer.symbol_to_index:
            return self.selfies_tokenizer.symbol_to_index[symbol]
        elif symbol == self.SEP_TOKEN:
            return self.SEP_IDX
        elif symbol == self.END_TOKEN:
            return self.END_IDX
        else:
            raise ValueError(f"Symbol {symbol} not in tokenizer")

    def norm_properties(self, properties, bool_mask = None):
        properties = properties - self.selfies_offset
        if bool_mask is not None:
            properties[~bool_mask] = 0
        return properties

    def norm_values(self, values, bool_mask):
        """
        takes tokenized values and make them 0 indexed eg
        if the min value is 16368 [16368,16369] -> [0,1]
        """
        values = values - self.properties_offset
        values[~bool_mask] = 0
        return values

    def save(self, path):
        # Save the necessary attributes to a JSON file
        data = {
            'selfies_tokenizer': str(self.selfies_tokenizer.save(path / "selfies_tokenizer.json")),
            'num_assays': self.num_assays,
            'num_vals': self.num_vals,
            'selfies_offset': self.selfies_offset,
            'SEP_IDX': self.SEP_IDX,
            'END_IDX': self.END_IDX
        }
        with open(path / 'selfies_property_val_tokenizer.json', 'w') as file:
            json.dump(data, file)
        return path

    def indexes_to_symbols(self, indexes):
        if isinstance(indexes, torch.Tensor):
            indexes = indexes.cpu().numpy()
        elif isinstance(indexes, list):
            indexes = np.array(indexes)
        symbols = [self.selfies_tokenizer.index_to_symbol[i] for i in indexes if i < self.selfies_offset]
        remainder = [str(s) for s in indexes[indexes >= self.selfies_offset]]
        return symbols + remainder
    
    @staticmethod
    def load(path):
        path = pathlib.Path(path)
        with open(path / 'selfies_property_val_tokenizer.json', 'r') as file:
            data = json.load(file)

        selfies_tokenizer = SelfiesTokenizer.load(path / "selfies_tokenizer.json")
        tokenizer = SelfiesPropertyValTokenizer(
            selfies_tokenizer, 
            data['num_assays'], 
            data['num_vals']
        )

        # Load other attributes
        tokenizer.selfies_offset = data['selfies_offset']
        tokenizer.SEP_IDX = data['SEP_IDX']
        tokenizer.END_IDX = data['END_IDX']

        return tokenizer
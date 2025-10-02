import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn, torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, sequence_length, d_model]
        # Update positional encoding to match batch size and sequence length
        x = x + self.pe[:, :x.size(1)].repeat(x.size(0), 1, 1)
        return self.dropout(x)

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """ Generate the attention mask for causal decoding """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerVAE(nn.Module):
    
    def __init__(self, len_charset, tokenizer):
        super(TransformerVAE, self).__init__()
        
        self.vocab_size = len_charset
        self.hdim = 128  # Increased hidden dimension
        self.nhead = 16  # You might experiment with this number
        self.dim_feedforward = self.hdim * 6  # Consider increasing if necessary
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.symbol_to_index[tokenizer.PAD_TOKEN]
        
        self.embedding = nn.Embedding(len_charset, self.hdim*2)
        self.positional_encoding = PositionalEncoding(self.hdim*2)
        
        # Increase the number of layers if necessary, but watch out for training difficulties
        encode_layer = nn.TransformerEncoderLayer(d_model=self.hdim*2, nhead=self.nhead, dim_feedforward=self.dim_feedforward, batch_first=True, dropout=0.1)  # Added dropout
        self.encoder = nn.TransformerEncoder(encode_layer, num_layers=4)  # Increased number of layers
        
        decode_layer = nn.TransformerDecoderLayer(d_model=self.hdim, nhead=self.nhead, dim_feedforward=self.dim_feedforward, batch_first=True, dropout=0.1)  # Added dropout
        self.decoder = nn.TransformerDecoder(decode_layer, num_layers=4)  # Increased number of layers
        
        self.classification_layer = nn.Linear(self.hdim, self.vocab_size)


    def forward(self, insmi, temp=0.1):
        
        memory_mask = insmi == self.token_pad_idx
        
        smiles_embedding = self.positional_encoding(self.embedding(insmi))
        smiles_encoding = self.encoder(smiles_embedding, src_key_padding_mask=insmi == memory_mask)
        zmean, zlogvar = self.vae_encode(smiles_encoding)
        z = self.sample(zmean, zlogvar, temp)
        
        teach_forcing_embed = smiles_embedding.clone()[:,:,:self.hdim]
        tgt_mask = generate_square_subsequent_mask(teach_forcing_embed.size(1))
        decoded = self.decoder(teach_forcing_embed, z, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        logits = self.classification_layer(decoded)
        
        return logits, zmean, zlogvar
    
    def vae_encode(self, smiles_encoding):
        # split smi_encode into zmean and zlogvar
        zmean = smiles_encoding[:,:,:self.hdim]
        zlogvar = smiles_encoding[:,:,self.hdim:]
        
        return zmean, zlogvar
    
    def sample(self, zmean, zlogvar, temp=0.1):
        epsilon = temp*torch.randn_like(zlogvar)
        std = torch.exp(0.5 * zlogvar)
        z = (std * epsilon) + zmean
        return z    
    
    def generate(self, insmi, device, temp = 0.0):
        
        memory_mask = insmi == self.token_pad_idx
        
        selfies_embedding = self.positional_encoding(self.embedding(insmi))
        smiles_encoding = self.encoder(selfies_embedding, src_key_padding_mask=memory_mask)
        zmean, zlogvar = self.vae_encode(smiles_encoding)  
        z = self.sample(zmean, zlogvar, temp)
        
        sos_token_idx = self.tokenizer.symbol_to_index[self.tokenizer.END_TOKEN]
        decoded_tokens = torch.LongTensor([sos_token_idx]).to(device).repeat(insmi.size(0),1).to(device)
        
        for i in range(120):  # max length of generation
            
            decoded_embedding = self.positional_encoding(self.embedding(decoded_tokens))[:,:,:self.hdim]
            tgt_mask = generate_square_subsequent_mask(decoded_embedding.size(1))
            next_dec = self.decoder(decoded_embedding, z, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
            logits = self.classification_layer(next_dec)
            new_tokens = torch.argmax(logits[:,i], dim=1).unsqueeze(1)
            
            decoded_tokens = torch.cat([decoded_tokens, new_tokens],dim=1)

        return decoded_tokens
    
    @staticmethod
    def loss(decsmi, insmi, z_mean, z_logvar, tokenizer, beta=1.0):
        
        pad_idx = tokenizer.symbol_to_index[tokenizer.PAD_TOKEN]
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_idx)
        recon_loss = criterion(decsmi, insmi)
        
        kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        
        return  total_loss, recon_loss, kl_loss
    
    @staticmethod
    def load(path = "brick/pvae.pt"):
        pass

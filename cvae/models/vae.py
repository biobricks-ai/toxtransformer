import torch, torch.nn as nn, torch.nn.functional as F
import pathlib

transformer_model = nn.Transformer(
    nhead=2, num_encoder_layers=12, batch_first=True,
    d_model=58)
src = torch.rand((32, 120, 58))
# tgt = torch.rand((32, 20, 512))
out = transformer_model(src, src)

class VAE(nn.Module):
    
    def __init__(self, len_charset):
        super(VAE, self).__init__()
        ldim = 1024
        
        self.encoder_transformer = nn.TransformerEncoderLayer(
            nhead=2,batch_first=True,d_model=58)
        
        self.encoder = nn.Sequential(
            nn.Conv1d(58, 1024, kernel_size=11),
            nn.MaxPool1d(110, stride=110))
        
        self.fcencoder = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
        )
        
        self.linear_mean = nn.Linear(1024, ldim)
        self.linear_logvar = nn.Linear(1024, ldim)
        
        self.dec_seq = nn.LSTM(1024, 250, 3, batch_first=True, bidirectional=True)
        self.dec_sig = nn.Sequential(nn.Linear(500, len_charset), nn.Sigmoid())

    def encode(self, insmi):
        x = self.encoder_transformer(insmi.permute(0,2,1)).permute(0,2,1)
        x = self.encoder(x)
        x = self.fcencoder(x.view(x.shape[0], -1))
        
        mean = self.linear_mean(x)
        logvar = self.linear_logvar(x)
        
        return mean, logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = 0.1*torch.randn_like(z_logvar)
        std = torch.exp(0.5 * z_logvar)
        z = std * epsilon + z_mean
        return z

    def decode(self, z):
        zrep = z.unsqueeze(1).repeat(1, 120, 1)
        dec = self.dec_seq(zrep)[0]
        dec = self.dec_sig(dec)
        return dec.permute(0,2,1)
    
    def forward(self, insmi):
        z_mean, z_logvar = self.encode(insmi)
        z = self.sampling(z_mean, z_logvar)
        decoded = self.decode(z)
        return decoded, z_mean, z_logvar
    
    def loss(self, decoded, insmi, z_mean, z_logvar):
        recon_loss = F.binary_cross_entropy(decoded, insmi, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
            
        return recon_loss + kl_loss, recon_loss, kl_loss
    
    @staticmethod
    def load(vaepath = "brick/vae.pt"):
        state = torch.load(vaepath)
        len_charset = state['encoder.0.weight'].shape[1]
        model = VAE(len_charset)
        model.load_state_dict(state)
        return model
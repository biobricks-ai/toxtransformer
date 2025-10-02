import torch
import torch.nn as nn
import torch.nn.functional as F
import cvae.models.tokenizer

class PredVAE(nn.Module):
    
    def __init__(self, len_charset, ldim=1024):
        super(PredVAE, self).__init__()
        
        self.ldim = ldim
        
        self.embedding_layer = nn.Embedding(len_charset, 10)
        self.encoder_tf = nn.TransformerEncoderLayer(nhead=2,batch_first=True,d_model=10) # BATCH, SEQLEN, 58
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=1024, kernel_size=11),
            nn.MaxPool1d(110, stride=110))
            
        
        self.fcencoder = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024),
        )
        
        self.linear_mean = nn.Linear(1024, ldim)
        self.linear_logvar = nn.Linear(1024, ldim)
        
        self.dec_seq = nn.LSTM(ldim, 250, 3, batch_first=True, bidirectional=True)
        self.dec_sig = nn.Sequential(nn.Linear(500, len_charset), nn.Sigmoid())
        
    def encode(self, smiles_embedding):
        esmi = self.encoder_tf(smiles_embedding)
        esmi = self.encoder(esmi.transpose(1,2))
        esmi = esmi.view(esmi.shape[0], -1)
        esmi = self.fcencoder(esmi)
        
        zmean = self.linear_mean(esmi)
        zlogvar = self.linear_logvar(esmi)
        return zmean, zlogvar
    
    def sample(self, zmean, zlogvar):
        epsilon = 0.1*torch.randn_like(zlogvar)
        std = torch.exp(0.5 * zlogvar)
        z = (std * epsilon) + zmean
        return z
    
    def decode(self, z):
        zrep = z.unsqueeze(1).repeat(1, 120, 1)
        dec = self.dec_seq(zrep)[0]
        dec = self.dec_sig(dec)
        return dec
    
    def forward(self, insmi):
        smiles_embedding = self.embedding_layer(insmi)
        zmean, zlogvar = self.encode(smiles_embedding)
        z = self.sample(zmean, zlogvar)
        decsmi = self.decode(z)
        return decsmi, zmean, zlogvar
    
    def loss(self, decsmi, insmi, z_mean, z_logvar):
        
        insmi_one_hot = F.one_hot(insmi, num_classes=decsmi.size(2)).to(dtype=torch.float32)

        decsmi = torch.clamp(decsmi, 1e-5, 1-(1e-5))
        rl = F.binary_cross_entropy(decsmi, insmi_one_hot, reduction='sum')
        
        Bkl = 1
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = Bkl * kl_loss
        
        return  rl + kl_loss, rl, kl_loss
    
    @staticmethod
    def load(path = "brick/pvae.pt"):
        state = torch.load(path)
        ldim = state['linear_mean.weight'].shape[0] # number of labels
        nchrs = state['encoder.0.weight'].shape[1] # number of characters
        mod = PredVAE(nchrs, ldim)
        mod.load_state_dict(state)
        return mod

# def recon_acc(self, decsmi, insmi):
#     rdecoded = torch.round(decsmi)
#     truepositive = torch.sum(rdecoded * insmi)
#     truenegative = torch.sum((1 - rdecoded) * (1 - insmi))
#     positive = torch.sum(inchem)
#     negative = torch.sum(1 - inchem)
    
#     se = truepositive / positive
#     sp = truenegative / negative
#     return 100.0 * (se + sp)/2.0

# def valacc(self, decval, inval):
#     pval = torch.round(decval)
#     tp = torch.sum(pval * inval)
#     tn = torch.sum((1 - pval) * (1 - inval))
#     pos,neg = torch.sum(inval), torch.sum(1 - inval)
    
#     se, sp = tp/pos, tn/neg
#     return 100.0 * (se + sp)/2.0


def rand(self, n):
    
    pvae = cvae.models.predvae.PredVAE.load().to(device)
    pvae.eval()
    
    smi = 'C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O'
    smi2 = 'CN1CCN(CC1)C2=C(C=C3C(=C2F)N(C=C(C3=O)C(=O)O)CCF)F'
    smi3 = 'CC=CC=CC(=O)[O-].[K+]'
    smi4 = 'CC=CC=CC(=O)[O-]'
    insmi = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi)).unsqueeze(0).to(device)
    insmi2 = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi2)).unsqueeze(0).to(device)
    insmi3 = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi3)).unsqueeze(0).to(device)
    insmi4 = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi4)).unsqueeze(0).to(device)
    inlbl = torch.Tensor([0]).long().to(device)
    z1 = pvae.encode(insmi, inlbl)[0] 
    z2 = pvae.encode(insmi2, inlbl)[0]
    z3 = pvae.encode(insmi3, inlbl)[0]
    z4 = pvae.encode(insmi4, inlbl)[0]
    F.cosine_similarity(z1,z2)
    F.cosine_similarity(z1,z3)
    F.cosine_similarity(z1,z4)
    F.cosine_similarity(z2,z3)
    F.cosine_similarity(z2,z4)
    F.cosine_similarity(z3,z4)
    
    
    smi = 'C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O'
    insmi = torch.Tensor(cvae.models.tokenizer.smiles_one_hot(smi)).unsqueeze(0).to(device)
    z = pvae.encode(insmi, torch.Tensor([0]).long().to(device))[0]
    z = z.repeat(100,1)
    z = z + 0.2*torch.randn_like(z)
    decsmi = self.decode(z)[0].permute(0,2,1)
    argmax = torch.argmax(decsmi,dim=2).cpu().numpy()
    osmi = [cvae.models.tokenizer.decode_smiles_from_indexes(x) for x in argmax]
    print("\n".join(osmi))
    
    def isvalid(smi): 
        try:
            return cvae.models.tokenizer.valid_smiles(smi)
        except:
            return False
    
    vsmi = [x for x in osmi if isvalid(x)]
    
    return decsmi

# def sample_mol(self, smiles, sample_coeff=0.1):
#     insmi = cvae.models.tokenizer.smiles_one_hot(smiles)
#     zmean, zlogvar = self.encode(insmi)
    
#     epsilon = sample_coeff*torch.randn_like(zlogvar)
#     std = torch.exp(0.5 * zlogvar)
#     z = std * epsilon + zmean
    
#     z = self.sample(zmean, zlogvar)
#     decsmi, _ = self.decode(z)
#     return decsmi
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvae.models.predvae import PredVAE


class TripletModel(nn.Module):
    
    def __init__(self, numlabel, pvae):
        super(TripletModel, self).__init__()
        for param in pvae.parameters():
            param.requires_grad = False
        
        self.numlabel = numlabel
        self.lblembed = nn.Embedding(numlabel, numlabel)
        self.lblembed.requires_grad = False
        self.lblembed.weight.data = torch.eye(numlabel,numlabel)
        self.pvae = pvae
        
        self.fc = nn.Sequential(
            nn.Linear(pvae.ldim*numlabel, 1024),nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, pvae.ldim),nn.Sigmoid())
            

    def encode(self, lbl, smi):
        esmi = self.pvae.encode(smi)[0]
        
        rep_esmi = esmi.unsqueeze(1).repeat(1,self.numlabel,1)
        elbl = self.lblembed(lbl).unsqueeze(2).expand_as(rep_esmi)
        rep_esmi = rep_esmi * elbl
        rep_esmi = rep_esmi.view(rep_esmi.shape[0],-1)

        enew = esmi + self.fc(rep_esmi)
        
        return enew
    
    def forward(self, lbl, anc, pos, neg):
        embanc = self.encode(lbl, anc)
        embpos = self.encode(lbl, pos)
        embneg = self.encode(lbl, neg)
        return embanc, embpos, embneg
    
    def loss(self, embanc, embpos, embneg):
       
        ancposip = (1.+F.cosine_similarity(embanc, embpos))/2. # 0 to 1
        ancnegip = (1.+F.cosine_similarity(embanc, embneg))/2. # 0 to 1
        diff = ancnegip - ancposip  + 1. # 0 to 2 
        tripleloss = torch.sum(diff)
        
        return tripleloss, torch.mean(ancposip).item(), torch.mean(ancnegip).item()
    
    @staticmethod
    def load(path = "brick/vaecontrast.pt", smilesvaepath = "brick/pvae.pt", device = 'cpu'):
        pvae = PredVAE.load().to(device)
        state = torch.load(path)
        ldim = state['lblembed.weight'].shape[0] # number of labels
        return TripletModel(ldim, pvae).to(device)


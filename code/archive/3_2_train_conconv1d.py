from tqdm import tqdm

import pandas as pd
import json, os, signal, dvc.api, numpy as np, pathlib, time
import torch, torch.nn.functional as F, torch.optim as optim
import torch.nn as nn

import torch.utils.data
from torch.utils.data import Dataset, Subset
import gc

import cvae.models, cvae.models.predvae
import cvae.utils as utils
import cvae.models.tokenizer

import seaborn as sns
import matplotlib.pyplot as plt
import faiss, numpy as np, torch, pandas, csv
import cvae.models.triplet, importlib
import signal

device = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

# load data ===========================================================================
ismiles, ilbl, ivalue = (0,1,2)
tst = torch.load(utils.res("data/processed/validation.pt")).tensors
tst = [tst[i] for i in [ismiles, ilbl, ivalue]]

trn = torch.load(utils.res("data/processed/train.pt")).tensors
trn = [trn[i] for i in [ismiles, ilbl, ivalue]]

hld = torch.load(utils.res("data/processed/holdout.pt")).tensors
hld = [tst[i] for i in [ismiles, ilbl, ivalue]]
ismiles, ilbl, ivalue = (0,1,2,3)

def mkfaiss(emb):
    emb = np.ascontiguousarray(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

class ContrastiveLearningDataset(Dataset):

    def __init__(self, emb, smi, labels, values, batchsize=150):
        self.emb, self.smi = emb, smi
        self.dim = self.emb.shape[1]
        
        self.batchsize = batchsize
        self.labels = labels
        self.unique_labels = torch.unique(self.labels).numpy()
        self.values = values
        self.posmask = self.values == 1
        
        self.posfaiss, self.negfaiss = {}, {}
        for lblidx in tqdm(self.unique_labels):
            egpos = self.emb[self.posmask & (labels == lblidx)]
            self.posfaiss[lblidx.item()] = mkfaiss(egpos)
            egneg = self.emb[~self.posmask & (labels == lblidx)]
            self.negfaiss[lblidx.item()] = mkfaiss(egneg)
            
    def __len__(self):
        return len(self.unique_labels)

    def __getitem__(self, lblidx):
        
        idx = torch.where(self.labels == lblidx)[0]
        
        ancidx = idx[torch.randperm(idx.shape[0])[:self.batchsize]]
        ancemb, ancchem, ancval = self.emb[ancidx], self.smi[ancidx], self.values[ancidx]
        
        idxposchem = self.smi[(self.labels == lblidx) & self.posmask]
        idxnegchem = self.smi[(self.labels == lblidx) & ~self.posmask]
        
        k = 5
        sim1 = idxposchem[self.posfaiss[lblidx].search(ancemb.numpy(), k)[1]].reshape(-1,58,120)
        sim0 = idxnegchem[self.negfaiss[lblidx].search(ancemb.numpy(), k)[1]].reshape(-1,58,120)
        
        repchem = ancchem.repeat_interleave(k,dim=0)
        
        repval = ancval.repeat_interleave(k,dim=0)
        poschem = torch.where(repval.unsqueeze(1).unsqueeze(2) == 1, sim1, sim0)
        negchem = torch.where(repval.unsqueeze(1).unsqueeze(2) == 1, sim0, sim1)
        
        lbl = torch.tensor([lblidx] * repchem.shape[0])
        return (lbl, repchem, poschem, negchem)
    
    def mkloader(self):
        return torch.utils.data.DataLoader(
            self, batch_size=1, shuffle=True, pin_memory=True, 
            num_workers=4, persistent_workers=True)


# make model ===========================================================================
pvae = cvae.models.predvae.PredVAE.load().to(device)

class TripletModel(nn.Module):
    
    def __init__(self, numlabel):
        super(TripletModel, self).__init__()
        for param in pvae.parameters():
            param.requires_grad = False
        
        self.numlabel = numlabel
        self.lblembed = nn.Embedding(numlabel, numlabel)
        self.lblembed.requires_grad = False
        self.lblembed.weight.data = torch.eye(numlabel,numlabel)
        
        self.fc = nn.Sequential(
            nn.Linear(pvae.ldim*numlabel, 1024),nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),nn.ReLU(), nn.BatchNorm1d(1024),
            nn.Linear(1024, pvae.ldim),nn.Sigmoid())
            

    def encode(self, lbl, smi):
        esmi = pvae.encode(smi)[0]
        
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
   
def update_triplet(model, insmi, inlbl, inval):
    emb = []
    ds = torch.utils.data.TensorDataset(insmi, inlbl, inval)
    dl = torch.utils.data.DataLoader(ds, batch_size=1000)
    for i, (smi, lbl, val) in enumerate(dl):
        smi, lbl = smi.to(device), lbl.to(device).int()
        embi = model.encode(lbl,smi)
        emb.append(embi.detach().cpu())

    emb = torch.cat(emb)
    return ContrastiveLearningDataset(emb, insmi, inlbl, inval)

model = TripletModel(152).to(device)
trndl = update_triplet(model, trn[ismiles], trn[ilbl], trn[ivalue]).mkloader()
tstdl = update_triplet(model, tst[ismiles], tst[ilbl], tst[ivalue]).mkloader()

def evaluate(model, dl):
    _ = model.eval()
    tstloss, avgposip, avgnegip = 0, 0, 0
    with torch.no_grad():
        for i, tsttensors in enumerate(dl):
            lbl, anc, pos, neg = [x.to(device).squeeze(0) for x in tsttensors]
            embanc, embpos, embneg = model(lbl,anc,pos,neg)
            iloss, iposip, inegip = model.loss(embanc, embpos, embneg)
            tstloss += iloss.item() / len(dl)
            avgposip += iposip / len(dl)
            avgnegip += inegip / len(dl)
    return tstloss, avgposip, avgnegip


# INIT MODEL =============================================================================
model = model.to(device)
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

# TRAIN MODEL ============================================================================
best_trnloss, best_tstloss = np.inf, np.inf
tstloss, tstpip, tstnip = evaluate(model, tstdl)
for epoch in range(1000):
    
    trnloss, avgdif, avgposip = 0, 0, 0  
    pbar = tqdm(enumerate(trndl), total=len(trndl))    
    _ = model.train()
    for i, trntensors in pbar:
        
        optimizer.zero_grad()
        lbl, anc, pos, neg = [x.to(device).squeeze(0) for x in trntensors]
        
        embanc, embpos, embneg = model(lbl,anc,pos,neg)
        loss, posip, negip = model.loss(embanc, embpos, embneg)
        diff = negip - posip
        
        loss.backward()
        optimizer.step()
        
        trnloss += loss.item() / len(trndl)
        avgdif, avgposip = avgdif + (diff / len(trndl)), avgposip + (posip / len(trndl))
        
        msg = f"{epoch} tst: {tstloss:.1f} trn: {loss.item():.1f} dif: {(negip-posip):.2f} posip: {posip:.2f}"
        if i==len(trndl)-1:
            msg = f"{epoch} tst: {tstloss:.1f} trn: {trnloss:.1f} dif: {avgdif:.2f} posip: {avgposip:.2f}"
        pbar.set_description(msg)
        
        if signal_received:
            break
        
    if signal_received:
        signal_received = False
        break
    
    tstloss, tstpip, tstnip = evaluate(model, tstdl)
    scheduler.step(tstloss)
    if trnloss < best_trnloss and tstloss < best_tstloss:
        print(f"saving and updating to tst: {tstloss:.1f} trn: {trnloss:.1f}")
        best_trnloss = trnloss
        best_tstloss = tstloss
        torch.save(model.state_dict(), utils.res("brick/vaecontrast.pt"))
    
    if epoch > 5 and (epoch % 5) == 0:
        print("updating triplets")
        trndl = update_triplet(model, trn[ismiles], trn[ilbl], trn[ivalue]).mkloader()
        tstdl = update_triplet(model, tst[ismiles], tst[ilbl], tst[ivalue]).mkloader()
      

# KNN RESULTS =============================================================================
model.eval()

def get_embeddings(model, insmi, inlbl):
    ds = torch.utils.data.TensorDataset(insmi, inlbl)
    dl = torch.utils.data.DataLoader(ds, batch_size=1000)
    emb = []
    for i, (smi, lbl) in tqdm(enumerate(dl), total=len(dl)):
        smi, lbl = smi.to(device), lbl.to(device).int()
        embi = model.encode(lbl,smi)
        emb.append(embi.detach().cpu())

    emb = torch.cat(emb)
    return emb

trnemb = get_embeddings(model, trn[ismiles], trn[ilbl])
tstemb = get_embeddings(model, tst[ismiles], tst[ilbl])
hldemb = get_embeddings(model, hld[ismiles], hld[ilbl])

import cvae.knn.knn as knn
results = []
for assay in tqdm(tst[ilbl].unique()):
    
    itstidx = torch.where(tst[ilbl] == assay)[0]
    itstemb = tstemb[itstidx]
    itstval = tst[ivalue][itstidx]
    
    ihldidx = torch.where(hld[ilbl] == assay)[0]
    ihldemb = torch.vstack([hldemb[ihldidx], itstemb])
    ihldval = torch.cat([hld[ivalue][ihldidx], itstval])
    
    itrnidx = torch.where(trn[ilbl] == assay)[0]
    itrnemb = torch.vstack([trnemb[itrnidx], ihldemb])
    itrnval = torch.cat([trn[ivalue][itrnidx], ihldval])
    
    # trnemb, trnval, tstemb, tstval = itrnemb, itrnval, ihldemb, ihldval
    res = knn.evaluate_embeddings(itrnemb, itrnval, ihldemb, ihldval, K=5)
    res['assay'] = assay.item()
    results = results + [res]

results = pd.DataFrame(results)
results.to_csv(utils.res("metrics/vaecontrast.csv"), index=False)
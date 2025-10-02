import signal, numpy as np, pathlib, tqdm
import torch, torch.optim as optim, torch.nn
import cvae.tokenizer.selfies_tokenizer, cvae.models.transformervae
from torch.utils.data import Dataset, DataLoader
import random
import sys

DEVICE = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

class SelfiesDataset(Dataset):
    
    def __init__(self, device, tokenizer, data_dir='data/processed/all_selfies'):
        self.filepaths = list(pathlib.Path(data_dir).glob('*.pt'))
        self.device = device
        self.pad_index = tokenizer.symbol_to_index[tokenizer.PAD_TOKEN]
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        tselfies = torch.load(self.filepaths[idx])
        padder = torch.tensor([[self.pad_index]], dtype=tselfies.dtype).repeat(tselfies.size(0), 1)
        output = torch.cat([tselfies[:,1:], padder], dim=1)
        return tselfies.to(self.device), output.to(self.device)

tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('data/processed/selfies_tokenizer.json')
selfies_ds = SelfiesDataset(DEVICE, tokenizer)
selfies_dl = DataLoader(selfies_ds, batch_size=1, shuffle=True, num_workers=0)

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

import importlib
importlib.reload(cvae.models.transformervae)

vocabsize= len(tokenizer.index_to_symbol)
model = cvae.models.transformervae.TransformerVAE(vocabsize, tokenizer).to(DEVICE)
model = torch.nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=21, factor=0.9, verbose=True, min_lr=1e-6)
epochs, best_trn_loss, best_tst_loss = 100, np.inf, np.inf

def example(module, smisample, temp=0.25):
    _ = module.eval()
    output = module.generate(smisample.unsqueeze(0), DEVICE, temp=temp).squeeze(0)
    return tokenizer.indexes_to_smiles(output.cpu().numpy())

def example2(module, smisample):
    _ = module.eval()
    output, _, _ = module(smisample.unsqueeze(0))
    outputidx = torch.argmax(output, dim=2)[0].detach().cpu().numpy()
    return tokenizer.indexes_to_smiles(outputidx)

print(example2(model.module, smisample=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=torch.long).to(DEVICE)))
print(example(model.module, smisample=torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], dtype=torch.long).to(DEVICE)))

writepath = pathlib.Path("metrics/vaeloss.tsv")
_ = writepath.write_text("epoch\tloss\trecloss\tkloss\n")

epochs, best_trn_loss, best_tst_loss = 100, np.inf, np.inf
for epoch in range(0,1000):
    
    # eloss, erec_loss, ekl_loss = evaluate()
    enum = enumerate(selfies_dl)
    pbar1 = tqdm.tqdm(enum, total=len(selfies_dl), unit="batch")
    
    _ = model.train()
    epochloss, recloss, kloss, l1loss = 0, 0, 0, 0
    
    # i, (insmi, output) = next(enum)
    
    schedulerloss = []
    for _,(insmi,output) in pbar1:
        
        insmi, output = insmi.squeeze(0), output.squeeze(0) # [VERY LARGE, 120] and [VERY LARGE, 120]
        batch_size = 128 # define your batch size
        
        pbar2 = tqdm.tqdm(range(0, insmi.size(0), batch_size), total=insmi.size(0)//batch_size, unit="batch")
        batchloss, recloss, kloss = [], [], []
        for i in pbar2:
            
            optimizer.zero_grad()
            
            insmi_batch = insmi[i:i + batch_size,:]
            mask = torch.rand(insmi_batch.shape).to(DEVICE) < 0.15
            mask[:,0] = False # don't mask the start token
            insmi_batch = insmi_batch.masked_fill(mask, tokenizer.symbol_to_index[tokenizer.PAD_TOKEN])
            output_batch = output[i:i + batch_size]
            
            # outputs and loss
            decoded, zmean, zlogvar = model(insmi_batch)
            lossfn = cvae.models.transformervae.TransformerVAE.loss
            loss, rec_loss, kl_loss = lossfn(decoded.permute(0,2,1), output_batch, zmean, zlogvar, tokenizer)
            
            # update model
            loss.backward()
            optimizer.step()
            
            batchloss.append(loss.item())
            recloss.append(rec_loss.item())
            kloss.append(kl_loss.item())            
            
            msg = f"{epoch} totloss: { loss.item() :.4f} "
            msg = msg + f"REC: {rec_loss.item():.4f} KL: {kl_loss.item():.4f}"
            pbar2.set_description(msg)
            
            if (i+1) // batch_size % 500 == 0:
                
                batchloss, recloss, kloss = [ np.mean(l) for l in [batchloss, recloss, kloss] ]                
                
                scheduler.step(batchloss)
                randsample = insmi[random.randint(0,insmi.size(0)-1)]
                tqdm.tqdm.write(f"input: {tokenizer.indexes_to_smiles(randsample.cpu().numpy())}")
                tqdm.tqdm.write(f"autor: {example(model.module, randsample)}")
                tqdm.tqdm.write(f"teach: {example2(model.module, randsample)}")
                tqdm.tqdm.write(f"epoch: {epoch}\tbatch:{i // batch_size}\tloss:{batchloss:.4f}\trecloss:{recloss:.4f}\tkl:{kloss:.4f}\tlr:{optimizer.param_groups[0]['lr']}\n")
                sys.stdout.flush()
                
                _ = model.train()
                with open(writepath, "a") as f:
                    _ = f.write(f"{epoch}\t{batchloss}\t{recloss}\t{kloss}\n")
                
                batchloss, recloss, kloss = [], [], []
            
            if signal_received:
                break
                
        if signal_received:
            break
    
    if signal_received:
            signal_received = False
            break
        
    _ = model.eval()
    # example()

    if epochloss < best_trn_loss:
        print('saving!')
        best_trn_loss = epochloss
        # best_tst_loss = eloss
        path = pathlib.Path("brick/tvae.pt")
        torch.save(model.state_dict(), path) 


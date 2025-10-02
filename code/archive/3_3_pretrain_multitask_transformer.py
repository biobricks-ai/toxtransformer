import signal, numpy as np, pathlib, tqdm
import torch, torch.optim as optim, torch.nn
import cvae.tokenizer.selfies_property_val_tokenizer, cvae.models.transformervae
from torch.utils.data import Dataset, DataLoader
import random
import sys
import cvae.utils

DEVICE = torch.device(f'cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(42)

class SelfiesDataset(Dataset):
    
    def __init__(self, device, tokenizer, data_dir='data/processed/all_selfies'):
        self.filepaths = list(pathlib.Path(data_dir).glob('*.pt'))
        self.device = device
        self.pad_index = tokenizer.PAD_IDX
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        tselfies = torch.load(self.filepaths[idx])
        
        padder = torch.tensor([[self.pad_index]], dtype=tselfies.dtype).repeat(tselfies.size(0), 1)
        output = torch.cat([tselfies[:,1:], padder], dim=1)
        return tselfies.to(self.device), output.to(self.device)

import importlib
importlib.reload(cvae.tokenizer.selfies_property_val_tokenizer)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')

selfies_ds = SelfiesDataset(DEVICE, tokenizer)
selfies_dl = DataLoader(selfies_ds, batch_size=1, shuffle=True, num_workers=0)

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

importlib.reload(cvae.models.multitask_transformer)
model = cvae.models.multitask_transformer.MultitaskTransformer(tokenizer).to(DEVICE)
model = torch.nn.DataParallel(model)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.9, verbose=True, min_lr=1e-6)


writepath = pathlib.Path("metrics/pretrain_multitask_transformer.tsv")
_ = writepath.write_text("batch\tepochs_since_best\tloss\n")

epochs, best_trn_loss = 100, np.inf
# LOSS GETS DOWN TO 0.05 when ignoring pad index
while scheduler.num_bad_epochs < 21:
    
    for _,(insmi,output) in tqdm.tqdm(enumerate(selfies_dl), total=len(selfies_dl), unit="batch"):
        if signal_received: break
        
        insmi, output = insmi.squeeze(0), output.squeeze(0) # [VERY LARGE, 120] and [VERY LARGE, 120]
        batch_size = 128 # define your batch size
        
        pbar2 = tqdm.tqdm(range(0, insmi.size(0), batch_size), total=insmi.size(0)//batch_size, unit="batch")
        batchloss = []
        for i in pbar2:
            if signal_received: break
            
            optimizer.zero_grad()
            _ = model.train()
            
            insmi_batch = insmi[i:i + batch_size,:]
            mask = torch.rand(insmi_batch.shape).to(DEVICE) < 0.15
            mask[:,0] = False # don't mask the start token
            insmi_batch = insmi_batch.masked_fill(mask, tokenizer.PAD_IDX)
            output_batch = output[i:i + batch_size]
            
            # outputs and loss
            decoded = model(insmi_batch)
            lossfn = cvae.models.multitask_transformer.MultitaskTransformer.lossfn()
            loss = lossfn(decoded.permute(0,2,1), output_batch)
            
            # update model
            loss.backward()
            
            # Gradient Clipping
            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # update weights
            optimizer.step()
            
            batchloss.append(loss.item()) 
            
            pbar2.set_description( f"epochs_since_best: {scheduler.num_bad_epochs}\t totloss: { loss.item() :.4f} ")
            
            if (i+1) // batch_size % 500 == 0:
                _ = model.eval()
                mean_bl = np.mean(batchloss)
                
                scheduler.step(mean_bl)
                randsample = insmi[random.randint(0,insmi.size(0)-1)]
                
                sample_str = ''.join(tokenizer.indexes_to_symbols(randsample.cpu().numpy())).replace('<sos>','')
                output_idx = torch.argmax(model(randsample.unsqueeze(0)), dim=2)[0].detach().cpu().numpy()
                output_str = ''.join(tokenizer.indexes_to_symbols(output_idx))
                
                # write input and output characters in 40 character batches
                for si in range(0, len(sample_str), 120):
                    tqdm.tqdm.write(f"input: {sample_str[si:si+120]}")                
                    tqdm.tqdm.write(f"teach: {output_str[si:si+120]}")
                    tqdm.tqdm.write(f"-")
                    
                tqdm.tqdm.write(f"bad_epochs: {scheduler.num_bad_epochs} loss:{mean_bl:.4f}\tlr:{optimizer.param_groups[0]['lr']:.4f}\n")
                
                with open(writepath, "a") as f:
                    _ = f.write(f"{i}\t{scheduler.num_bad_epochs}\t{mean_bl}\n")
                
                batchloss = []
        
                if mean_bl < best_trn_loss:
                    tqdm.tqdm.write('saving!\n')
                    best_trn_loss = mean_bl
                    path = pathlib.Path("brick/mtransform1")
                    cvae.utils.mk_empty_directory(path, overwrite=True)
                    model.module.save(path)
                
                sys.stdout.flush()

    if signal_received:
            signal_received = False
            break
        



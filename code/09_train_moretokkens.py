# add tokens for inp
import importlib, inspect, itertools
import pathlib, signal, random, numpy as np, tqdm, sys
import cvae.models, cvae.models, cvae.tokenizer, cvae.utils as utils
import torch, torch.utils.data, torch.nn.functional as F, torch.optim as optim
import cvae.models.multitask_transformer as mt 

from tqdm import tqdm
import itertools
import pathlib

DEVICE = torch.device(f'cuda:0')

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

# load data ===========================================================================
importlib.reload(cvae.tokenizer.selfies_property_val_tokenizer)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')
class Trainer():
    
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(),lr=1e-3)
        self.lossfn = cvae.models.multitask_transformer.MultitaskTransformer.lossfn()
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.9, verbose=True, min_lr=1e-6)
        self.scheduler_loss = []
        self.scheduler_loss_interval = 100
        
        self.savepath = "brick/mtransform_addtokens2"
        self.test_losses = [np.inf]
        self.best_test_loss = np.inf
    
    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        return self
    
    # def set_evaluation_interval(self, interval):
        # self.evaluation_interval = interval
        # return self
    
    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        return self
    
    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        return self
    
    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = metrics_path
        if overwrite: utils.write_path(self.metrics_path,"epoch\tloss\tlearning_rate\n", mode='w')
        return self
    
    def set_values_mask_flag(self, bool_flag):
        self.values_mask = bool_flag
        return self
    
    def _evaluation_loss(self, valdl):
        self.model.eval()
        epochloss = []
        for _, (inp, teach, out) in tqdm(enumerate(valdl), total=len(valdl)):
            if signal_received: return
            inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
            pred = self.model(inp, teach)
            loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
            epochloss.append(loss.item())
        
        return np.mean(epochloss)
                
    def _epochs_since_improvement(self):
        return len(self.test_losses) - np.argmin(self.test_losses)
    
    def _train_batch(self, inp, teach, out):
        _ = self.model.train()
        self.optimizer.zero_grad()
        
        # outputs and loss
        pred = self.model(inp, teach)
        loss = self.lossfn(self.model.parameters(), pred.permute(0,2,1), out)
        
        # update model
        loss.backward()
        self.optimizer.step()
            
        return loss.item()
    
    def start(self):
        global signal_received
        signal_received = False
        #not sure about this line
        # self.trn_iterator = (item for item in tqdm(self.trn_iterator, total=len(trnds) // 124))
        i, (inp, teach, out) = next(self.trn_iterator)
        batch_size = inp.size(0)
        # evaluate twice per epoch
        evaluation_interval = ((len(trnds)-1) // batch_size) // 2
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        value_indexes = torch.LongTensor(list(tokenizer.value_indexes().values())).to(DEVICE)
        
        epoch = 0
        trn_loss = []
        
        while self._epochs_since_improvement() < 4000:
            if signal_received: return    
            i, (inp, teach, out) = next(self.trn_iterator)
            inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
            this_batch_size = inp.size(0)
            
            mask = torch.rand(inp.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            inp = inp.masked_fill(mask, tokenizer.pad_idx)
            mask = torch.rand(teach.shape, device=DEVICE) < self.mask_percent
            mask[:,0] = False # don't mask the first token
            teach = teach.masked_fill(mask, tokenizer.pad_idx)
            
            if self.values_mask:
                teach = teach.masked_fill(torch.isin(teach, value_indexes), tokenizer.pad_idx)
            
            loss = self._train_batch(inp, teach, out)
            trn_loss.append(loss)
            utils.write_path(self.metrics_path,f"train\t{i}\t{loss/this_batch_size}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
            
            # SCHEDULER UPDATE 
            if (i + 1) % self.scheduler_loss_interval == 0:
                mean_loss = np.mean(trn_loss)
                utils.write_path(self.metrics_path,f"scheduler\t{i}\t{mean_loss/this_batch_size}\t{self.optimizer.param_groups[0]['lr']:.4f}\n")
                trn_loss = []
            
            # EVALUATION UPDATE
            if (i + 1) % evaluation_interval == 0:
                epoch += 1
                eval_loss = self._evaluation_loss(valdl)
                self.scheduler.step(mean_loss)
                self.test_losses.append(eval_loss)
                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.model.module.save(self.savepath)
                
                utils.write_path(self.metrics_path,f"eval\t{i}\t{self.test_losses[-1]/batch_size}\n")
                print(f"epoch: {epoch}\t eval_loss: {self.best_test_loss/batch_size:.4f}")
        
        
importlib.reload(mt)
# model = cvae.models.MultitaskTransformer(tokenizer).to(DEVICE)
# model = mt.MultitaskDecoderTransformer(tokenizer).to(DEVICE)
# model = torch.nn.DataParallel(model)
model = mt.MultitaskTransformer.load("brick/mtransform_addtokens2").to(DEVICE)

trnds = mt.SequenceShiftDataset("data/processed/multitask_tensors/trn", tokenizer)
trndl = torch.utils.data.DataLoader(trnds, batch_size=124, shuffle=True, prefetch_factor=100, num_workers=20)
valds = mt.SequenceShiftDataset("data/processed/multitask_tensors/tst", tokenizer)
valdl = torch.utils.data.DataLoader(valds, batch_size=124, shuffle=True, prefetch_factor=100, num_workers=20)
# (i,o) = valds[0]

# inp, teach, out = valds[0]
# inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
# model(inp.unsqueeze(0),teach.unsqueeze(0)).shape

trainer = Trainer(model)\
    .set_trn_iterator(itertools.cycle(enumerate(trndl)))\
    .set_validation_dataloader(valdl)\
    .set_mask_percent(0.0)\
    .set_metrics_file(pathlib.Path("metrics/multitask_loss_addtokens2.tsv"))\
    .set_values_mask_flag(False)

trainer.start()

# EVALUATE ===================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
import tqdm
import importlib, inspect, itertools
import pathlib, signal, random, numpy as np, tqdm, sys
import cvae.models, cvae.models, cvae.tokenizer, cvae.utils as utils
import cvae.models.multitask_transformer as mt
import torch, torch.utils.data, torch.nn.functional as F, torch.optim as optim

DEVICE = torch.device(f'cuda:0')

signal_received = False
def handle_interrupt(signal_number, frame):
    global signal_received
    signal_received = True
signal.signal(signal.SIGINT, handle_interrupt)

importlib.reload(cvae.tokenizer.selfies_property_val_tokenizer)
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('data/processed/selfies_property_val_tokenizer')
importlib.reload(cvae.models.multitask_transformer)
model = mt.MultitaskTransformer.load("brick/mtransform2").to(DEVICE)
    
def extract_ordered_assays(tensor):
    assay_indexes = tokenizer.assay_indexes()
    index_assays = {v: k for k, v in assay_indexes.items()}
    return [index_assays[x.item()] for x in tensor if x.item() in assay_indexes.values()]

def extract_ordered_values(tensor):
    value_indexes = tokenizer.value_indexes()
    index_values = {v: k for k, v in value_indexes.items()}
    return [index_values[x.item()] for x in tensor if x.item() in value_indexes.values()]

def extract_assay_value_dict(tensor):
    assay_indexes = torch.LongTensor(list(tokenizer.assay_indexes().values())).to(DEVICE)
    # torch where tensor value is in assay_indexes
    indexes = torch.isin(tensor, assay_indexes)
    # shift index one to right to get value indexes
    value_indexes = torch.where(indexes)[0] + 1
    tensor_assays = tensor[indexes].cpu().numpy()
    tensor_values = tensor[value_indexes].cpu().numpy()
    # zip together to get assay:value pairs
    return dict(zip(tensor_assays, tensor_values))

def extract_probabilities_of_one(out, probs):
    # get the index of each assay token in the out tensor
    npout = out.cpu().numpy()
    out_value_sequence_indexes = np.where(np.isin(npout, list(tokenizer.value_indexes().values())))[0]
    
    value_indexes = list(tokenizer.value_indexes().values())
    assay_value_probs = probs[out_value_sequence_indexes][:,value_indexes]
    normalized_probs = assay_value_probs / assay_value_probs.sum(axis=1, keepdims=True)
    one_probs = normalized_probs[:,1]
    
    return one_probs

def evaluate(model):

    tst = mt.SequenceShiftDataset("data/processed/multitask_tensors/tst", tokenizer.pad_idx, tokenizer.SEP_IDX, tokenizer.END_IDX)
    
    global signal_received 
    signal_received = False
    
    out_df = pd.DataFrame()
    # for i in tqdm.tqdm(range(len(tst))):
    for i in tqdm.tqdm(range(10000)):
        if signal_received: break
        inp, teach, out = tst[i]
        inp, teach, out = inp.to(DEVICE), teach.to(DEVICE), out.to(DEVICE)
        
        # mask all the value tokens
        value_indexes = torch.LongTensor(list(tokenizer.value_indexes().values())).to(DEVICE)
        # teach = teach.masked_fill(torch.isin(teach, value_indexes), tokenizer.pad_idx)
        pred = model(inp.unsqueeze(0), teach.unsqueeze(0))
        probs = torch.softmax(pred[0],dim=1).detach()
        
        maxprobs = torch.argmax(probs, dim=1)
        probs = extract_probabilities_of_one(out, probs).cpu().numpy()
        assays = extract_ordered_assays(out)
        values = extract_ordered_values(out)
        
        i_out_df = pd.DataFrame({"i":i, "assay": assays, "value": values, "prob_1": probs })
        out_df = pd.concat([out_df, i_out_df])

    # 1.0 very large model ------------------------------- AUC: 0.7283     ACC: 0.6440     BAC: 0.6447  LOSS: 0.25
    # 1.1 much smaller model & sum loss & L2  ------------ AUC: 0.6914     ACC: 0.6185     BAC: 0.6178  LOSS: 921.94
    # 1.2 much larger model  & sum loss & 15000 iter ----- AUC: 0.6653     ACC: 0.6005     BAC: 0.6029  MEAN_EVAL_LOSS: 1.76361 BATCH=128
    # 1.2.1 much smaller model & sum loss & 15000 iter --- AUC: 0.7523     ACC: 0.6707     BAC: 0.6737  MEAN_EVAL_LOSS: 1.5625  BATCH=2048
    # 1.2.1.1 a few more iterations ---------------------- AUC: 0.7719     ACC: 0.6831     BAC: 0.6831  MEAN_EVAL_LOSS: 1.5454  BATCH=2048
    # 1.3 single property-value output ------------------- AUC: 0.7684     ACC: 0.6773     BAC: 0.6790  MEAN_EVAL_LOSS: 0.55
    # 1.4 decoder-only selfies in, selfies + pv out 40k -- AUC: 0.6984     ACC: 0.6165     BAC: 0.6152  MEAN_EVAL_LOSS: 17.14 BATCH=256
    # 1.5 decoder-only trunc-selfies more iterations ------AUC: 0.7581     ACC: 0.6759     BAC: 0.6759  MEAN_EVAL_LOSS: 32.53 BATCH=256 Iterations = 60000
    y_true, y_pred = out_df['value'].values, out_df['prob_1'].values
    auc, acc, bac = roc_auc_score(y_true, y_pred), accuracy_score(y_true, y_pred > 0.5), balanced_accuracy_score(y_true, y_pred > 0.5)
    print(f"Total Metrics:\tAUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")
    
    # Group by assay and calculate metrics
    assay_metrics = []
    grouped = out_df.groupby('assay')
    for assay, group in tqdm.tqdm(grouped):
        y_true = group['value'].values
        y_pred = group['prob_1'].values
        if len(np.unique(y_true)) < 2: continue
        if sum(y_true==0) < 10 or sum(y_true==1) < 10 : continue
        
        auc = roc_auc_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred > 0.5)
        bac = balanced_accuracy_score(y_true, y_pred > 0.5)
        assay_metrics.append({'assay': assay, 'AUC': auc, 'ACC': acc, 'BAC': bac, "NUM_POS": sum(y_true==1), "NUM_NEG": sum(y_true==0)})

    metrics_df = pd.DataFrame(assay_metrics)
    # sort metrics_df by AUC
    metrics_df.sort_values(by=['AUC'], inplace=True, ascending=False)
    metrics_df
    auc, acc, bac = metrics_df['AUC'].median(), metrics_df['ACC'].median(), metrics_df['BAC'].median()
    print(f"Metrics over assays:\tAUC: {auc:.4f}\tACC: {acc:.4f}\tBAC: {bac:.4f}")
    print(f"number of assays {len(metrics_df)}")
    # 1.5.1 AUC: 0.8033     ACC: 0.7460     BAC: 0.6979
    # Plotting
    from matplotlib import pyplot as plt

    # Set the style
    plt.style.use('dark_background')

    # Create the figure and the histogram
    plt.figure(figsize=(20, 10))
    n, bins, patches = plt.hist(metrics_df['AUC'], bins=20, alpha=0.5, edgecolor='white', linewidth=1.5, color='turquoise')

    # Add a line for the median AUC value
    plt.axvline(auc, color='yellow', linestyle='dashed', linewidth=2)

    # Annotate the median AUC value
    median_annotation = f'Median AUC: {auc:.4f}'
    plt.annotate(median_annotation, xy=(auc, max(n)), xytext=(auc, max(n) + max(n)*0.1),
                arrowprops=dict(facecolor='yellow', shrink=0.05),
                fontsize=18, color='yellow', fontweight='bold', ha='center')

    # Label the bars with the count
    for (rect, label) in zip(patches, n):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 5, f'{int(label)}', ha='center', va='bottom', color='white', fontsize=12)

    # Enhance titles and labels
    plt.title('Histogram of AUC per Property', color='white', fontsize=26)
    plt.xlabel('AUC', color='white', fontsize=22)
    plt.ylabel('Number of Properties', color='white', fontsize=22)

    # Improve tick marks
    plt.xticks(fontsize=18, color='white')
    plt.yticks(fontsize=18, color='white')

    # Show grid
    plt.grid(color='gray', linestyle='dashed', linewidth=0.5, alpha=0.5)

    # Adjust the layout
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig('notebook/plots/multitask_transformer_metrics.png', facecolor='black')



    # Display the plot
    plt.show()
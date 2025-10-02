import faiss, numpy as np, torch, pandas, csv
import cvae.utils as utils
from tqdm import tqdm

def knn_search(index, values, query, k, minsim=0.9):
    query = np.ascontiguousarray(query)
    
    weight, indices = index.search(query, k)
    index_values = values[indices]
    
    filtered_weight = np.where(weight>minsim, weight, 0)
    nonzero_weights = np.count_nonzero(filtered_weight, axis=1)
    filtered_values = np.where(weight>minsim, index_values, 0)
    weighted_values = np.sum(filtered_values,axis=1)
    
    result = np.divide(weighted_values, nonzero_weights, where=(nonzero_weights > 0))
    result = np.where(nonzero_weights > 0, result, np.nan)

    return result

def knn_dist_search(index, values, query, k, maxdist=4):
    query = np.ascontiguousarray(query)
    
    weight, indices = index.search(query, k)
    index_values = values[indices]
    
    filtered_weight = np.where(weight < maxdist, weight, 0)
    nonzero_weights = np.count_nonzero(filtered_weight, axis=1)
    filtered_values = np.where(weight < maxdist, index_values, 0)
    weighted_values = np.sum(filtered_values,axis=1)
    
    result = np.divide(weighted_values, nonzero_weights, where=(nonzero_weights > 0))
    result = np.where(nonzero_weights > 0, result, np.nan)

    return result

# smiles, assay, value, morgan, mvae, trnflag, biosim
trainds = torch.load(utils.res("data/biosim/train.pt"))
evalds = torch.load(utils.res("data/biosim/validation.pt"))
imorgan, ipvae = 3, 6

# build predictions for morgan, mvae, biosim by assay
assays = trainds.tensors[1].unique()
allval = trainds.tensors[2].numpy()

def getV(ds, assay, idx):
    V = ds.tensors[idx][torch.where(ds.tensors[1] == assay)[0]].cpu()
    V = V.detach().numpy().astype(np.float32)
    V = np.ascontiguousarray(V)
    return V

def mkfaiss(emb):
    emb = np.ascontiguousarray(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index

K = 5
outdf = pandas.DataFrame(columns=['assay','tstidx','tstval','fingerprint','prediction'])

for assay in tqdm(assays):
    trnidx = torch.where(trainds.tensors[1] == assay)[0]
    trnval = allval[trnidx]
    
    trnmorgan = getV(trainds, assay, imorgan)
    trnmorgan = trnmorgan / np.linalg.norm(trnmorgan,axis=1,keepdims=True)
    
    trnbiosim = getV(trainds, assay, ipvae)
    trnbiosim = trnbiosim / np.linalg.norm(trnbiosim,axis=1,keepdims=True)
    
    morgan_faiss = mkfaiss(trnmorgan)
    biosim_faiss = mkfaiss(trnbiosim)
    
    tstmorgan = getV(evalds, assay, imorgan)
    tstmorgan = tstmorgan / np.linalg.norm(tstmorgan,axis=1,keepdims=True)
    
    tstbiosim = getV(evalds, assay, ipvae)
    tstbiosim = tstbiosim / np.linalg.norm(tstbiosim,axis=1,keepdims=True)
    
    morgan_pred = knn_search(morgan_faiss, trnval, tstmorgan, k=K, minsim=0.5)
    biosim_pred = knn_search(biosim_faiss, trnval, tstbiosim, k=K, minsim=0.8)
    # index, values, query, k, maxdist = biosim_faiss, trnval, tstbiosim, 5, 0.5
    # write the assay, test values, and all predictions to file
    tstidx = torch.where(evalds.tensors[1] == assay)
    tstval = evalds.tensors[2][tstidx].numpy()
    
    adf = np.vstack([tstval, morgan_pred, biosim_pred]).T
    columns = ['tstval', 'morgan_pred', 'biosim_pred']
    adf = pandas.DataFrame(adf, columns=columns)
    adf['assay'] = assay.item()
    adf['tstidx'] = tstidx[0]
    adf = adf.melt(
        id_vars=['assay','tstidx', 'tstval'], 
        var_name='fingerprint', value_name='prediction')
    outdf = pandas.concat([outdf,adf], ignore_index=True)

outpath = utils.res("metrics/knn/predictions.csv")
outpath.parent.mkdir(parents=True, exist_ok=True)
outdf.to_csv(outpath, index=False, header=True)


from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

# Initialize an empty DataFrame for the results
results = pandas.DataFrame(columns=['assay', 'fingerprint', 'sensitivity', 'specificity', 'auc', 'acc', 'bac', 'count', 'groupsize'])
grouped = outdf.groupby(['assay','fingerprint'])

# enumg = enumerate(grouped)
# for i, ((assay, finger), group) in enumerate(grouped):
#     if i == 28:
#         break

for i, ((assay, finger), group) in enumerate(grouped):
    print(i)
    # Filter out rows with NaN predictions
    groupsize = len(group)
    group = group.dropna(subset=['prediction'])

    # if group is empty, skip
    if len(group) < 30 or len(group['tstval'].unique()) == 1:
        continue
    
    # # Calculate metrics
    y_true = group['tstval']
    y_pred = group['prediction']
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    # if there is only one y_true, continue
    if len(y_true.unique()) == 1 and len(y_pred.unique()) == 1:
        continue
    
    # # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # # Calculate accuracy and balanced accuracy
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred_binary)
    
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    # Append results to the results DataFrame
    res = pandas.DataFrame({
        'assay': assay,
        'fingerprint': finger,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'acc': accuracy,
        'bac': balanced_accuracy,
        'count': len(group),
        'groupsize': groupsize
    },index=[0])
    
    results = pandas.concat([results,res], ignore_index=True)

newpath = utils.res("metrics/knn.csv".format(assay))
results.to_csv(newpath, index=False, header=True)
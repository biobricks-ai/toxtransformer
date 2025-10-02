import faiss, numpy as np, torch, pandas, csv
import cvae.utils as utils
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix

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

def evaluate_embeddings(trnemb, trnval, tstemb, tstval, K=5):

    def mkfaiss(emb):
        emb = np.ascontiguousarray(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return index

    outdf = pandas.DataFrame(columns=['tstidx','tstval','prediction'])
    biosim_faiss = mkfaiss(trnemb)
    biosim_pred = knn_search(biosim_faiss, trnval, tstemb, k=K, minsim=0.8)
    
    pdf = pandas.DataFrame({"tstval": tstval, "prediction": biosim_pred})
    
    pdf['nompred'] = (pdf['prediction'] > 0.5).astype(float)

    tn, fp, fn, tp = confusion_matrix(pdf['tstval'], pdf['nompred']).ravel()
    
    return {
        'tp': tp,
        'tn': tn,
        'se': tp / (tp + fn),
        'sp': tn / (tn + fp),
        'acc': accuracy_score(pdf['tstval'], pdf['nompred']),
        'bac': balanced_accuracy_score(pdf['tstval'], pdf['nompred']),
        'auc': roc_auc_score(pdf['tstval'], pdf['prediction']),
        'tot': len(pdf)
    }
    
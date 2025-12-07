import itertools, uuid, pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when

# Create all necessary directories
outdir = pathlib.Path("cache/eval_multi_properties")
outdir.mkdir(exist_ok=True, parents=True)

temp_dir = outdir / "temp"
temp_dir.mkdir(exist_ok=True)
# clear all files in temp_dir
for file in temp_dir.iterdir():
    file.unlink()

metrics_dir = outdir / "metrics"
metrics_dir.mkdir(exist_ok=True)

tqdm.tqdm.pandas()
DEVICE = torch.device(f'cuda:0')

model : me.MoE = me.MoE.load("brick/moe").to(DEVICE)
model = torch.nn.DataParallel(model)
tokenizer : SelfiesPropertyValTokenizer = model.module.tokenizer

spark = cvae.utils.get_spark_session()


# EVALUATION LOOP ===================================================================
assay_indexes = torch.tensor(list(tokenizer.assay_indexes().values()), device=DEVICE)
value_indexes = torch.tensor(list(tokenizer.value_indexes().values()), device= DEVICE)

def run_eval(i, raw_inp, raw_out, out_df, nprops):
    inp, raw_out = raw_inp.to(DEVICE), raw_out.to(DEVICE)
        
    # filter to instances with at least nprops properties
    x = torch.greater_equal(torch.sum(torch.isin(raw_out, value_indexes),dim=1),nprops)
    chemical_id = torch.where(x)[0] + (i * batch_size)
    inp, trunc_out = inp[x], raw_out[x,1:(2*nprops + 1)].reshape(-1,nprops,2)
    
    # if all of x is false skip
    if len(chemical_id) == 0: 
        return out_df
    
    # get all permutations
    perm_indices = list(itertools.permutations(range(nprops)))
    perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in perm_indices],dim=0).reshape(-1,nprops*2)
    sep_tensor = torch.full((perm_out.size(0),1), tokenizer.SEP_IDX, device=raw_out.device)
    zer_tensor = torch.zeros_like(sep_tensor, device=raw_out.device)
    out = torch.cat([sep_tensor,perm_out,zer_tensor],dim=1)
    
    # make teach tensor
    one_tensor = torch.ones_like(sep_tensor, device=out.device)
    teach = torch.cat([one_tensor, out[:,:-1]], dim=1)
    
    # repeat interleave input for all the permutations. if inp has idxs 1,2 then the below gives us 1,1,2,2
    rep_inp = inp.repeat(len(perm_indices),1)
    
    # get model predictions as a prob
    prob = torch.softmax(model(rep_inp, teach),dim=2).detach()
    
    # get out assays and the assay with the highest prob
    assays = out[torch.isin(out, assay_indexes)].cpu().numpy()
    prob_assays = torch.argmax(prob, dim=2)[torch.isin(out, assay_indexes)].cpu().numpy()
    
    # get out values and the value with the highest prob and the prob of the `1`` value
    values = out[torch.isin(out, value_indexes)].cpu().numpy()
    
    probmax_vals = torch.argmax(prob, dim=2)[torch.isin(out, value_indexes)].cpu().numpy()
    rawprobs = prob[torch.isin(out, value_indexes)][:,value_indexes]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:,1].cpu().numpy()
    
    # get position of each value in the out tensor
    num_props = torch.sum(torch.isin(out, assay_indexes), dim=1)
    position = torch.cat([torch.arange(size.item()) for size in num_props]).cpu().numpy()
    
    # repeat chemical_id 10x
    chemical_id = torch.repeat_interleave(chemical_id, len(perm_indices))
    chemical_id = torch.repeat_interleave(chemical_id, num_props).cpu().numpy()
    
    # cut assays up into groups of nprops then build 10 strings with assay 0, assay 0 + assay 1, assay 0 + assay 1 + assay 2, etc.
    assays_reshaped = assays.reshape(-1, nprops).astype(str)
    values_reshaped = values.reshape(-1, nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[i, :j+1]) for i in range(len(assays_reshaped)) for j in range(nprops)]
    prior_values = [values_reshaped[i, :j+1] for i in range(len(values_reshaped)) for j in range(nprops)]
    batch_df = pd.DataFrame({'batch': i, 'chemical_id': chemical_id, 
                                'prior_assays': prior_assays, 'prior_values': prior_values,
                                'assay': assays, 
                                'value': values, 'probs':probs, 'nprops':position,
                                'prob_assays': prob_assays, 'prob_vals': probmax_vals})
    
    return pd.concat([out_df, batch_df]) if len(out_df) > 0 else batch_df

batch_size = 20
nprops = 5
val = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer, nprops=nprops)
valdl = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
seen_inputs = set()

# just keep going for 4 hours
import time
start_time = time.time()
max_time = 1 * 60 * 60  # 1 hours in seconds

for iter in tqdm.tqdm(range(24)):
    out_df = pd.DataFrame({'chemical_id':[], 'prior_assays':[], 'prior_values':[], 'assay':[], 'value':[], 'probs':[], 'nprops':[], 'prob_assays':[], 'prob_vals':[]})
    for i, (raw_inp, _, raw_out) in tqdm.tqdm(enumerate(valdl), total=len(val)/batch_size):
        # Check if 4 hours have elapsed
        if time.time() - start_time > max_time:
            if not out_df.empty:
                path = temp_dir / f"multitask_predictions_{str(uuid.uuid4())}.parquet"
                out_df.to_parquet(path, index=False)
            break
            
        batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
        new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
        seen_inputs.update(batch_tuples)
        
        if any(new_inputs_mask):
            new_raw_inp = raw_inp[new_inputs_mask]
            new_raw_out = raw_out[new_inputs_mask]
            current_df = run_eval(i, new_raw_inp, new_raw_out, pd.DataFrame(), nprops)
            out_df = pd.concat([out_df, current_df], ignore_index=True)
        
        if len(out_df) > 1000000:
            out_df.to_parquet(temp_dir / f"multitask_predictions_{str(uuid.uuid4())}.parquet", index=False)
            out_df = pd.DataFrame({'chemical_id':[], 'prior_assays':[], 'prior_values':[], 'assay':[], 'value':[], 'probs':[], 'nprops':[], 'prob_assays':[], 'prob_vals':[]})

        if time.time() - start_time > max_time:
            break
        
    if not out_df.empty:
        path = temp_dir / f"multitask_predictions_{str(uuid.uuid4())}.parquet"
        out_df.to_parquet(path, index=False)

df0 = spark.read.parquet(str(temp_dir / "*.parquet"))
df1 = df0.withColumn("prior_assays", split(col("prior_assays"), " \+ ")) 
df1.write.parquet((outdir / "multitask_predictions.parquet").as_posix(), mode="overwrite")

# GENERATE STRATIFIED EVALUATIONS FOR POSITION 0-5 ===============================
outdf = spark.read.parquet((outdir / "multitask_predictions.parquet").as_posix())

# Calculate metrics
value_indexes = torch.tensor(list(tokenizer.value_indexes().values()), device=DEVICE)
val0_index, val1_index = value_indexes[0].item(), value_indexes[1].item()

from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_true_binary = (y_true != val0_index).astype(int)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    auc = float(roc_auc_score(y_true_binary, y_pred))
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    bac = float(balanced_accuracy_score(y_true_binary, y_pred_binary))
    ce_loss = float(log_loss(y_true_binary, y_pred))
    return auc, acc, bac, ce_loss


calculate_metrics_udf = F.udf(calculate_metrics, "struct<AUC:double, ACC:double, BAC:double, cross_entropy_loss:double>")
large_properties_df = outdf.groupBy('nprops', 'assay').agg(
    F.collect_list('value').alias('y_true'),
    F.collect_list('probs').alias('y_pred'),
    countDistinct('chemical_id').alias('nchem'),
    F.sum(when(col('value') == val1_index, 1).otherwise(0)).alias('NUM_POS'),
    F.sum(when(col('value') == val0_index, 1).otherwise(0)).alias('NUM_NEG')) \
    .filter((col('NUM_POS') >= 10) & (col('NUM_NEG') >= 10) & (col('nchem') >= 20)).cache()

metrics_df = large_properties_df.repartition(800) \
    .withColumn('metrics', calculate_metrics_udf(F.col('y_true'), F.col('y_pred'))) \
    .select('nprops', 'assay', col('metrics.AUC').alias('AUC'), col('metrics.ACC').alias('ACC'), col('metrics.BAC').alias('BAC'), col('metrics.cross_entropy_loss').alias('cross_entropy_loss'), 'NUM_POS', 'NUM_NEG')

df = metrics_df.toPandas()
df['AUC'].median()

metrics_df.write.parquet((outdir / "multitask_metrics.parquet").as_posix(), mode="overwrite")
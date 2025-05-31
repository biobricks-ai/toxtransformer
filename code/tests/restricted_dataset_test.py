import cvae.models.datasets.restricted_dataset as rd
import cvae.tokenizer
import pandas as pd
import shutil
import cvae.models.mixture_experts as me
import pathlib

tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
target_props = bp['property_token'].tolist()
target_positions = [0, 4]  # Positions to focus evaluation on
sampling_weights = {prop: weight for prop, weight in zip(target_props, bp['weight'].tolist())}
nprops = 5

# make a tmpdir and just copy two files into it

import tempfile
from pathlib import Path

tmpdir = Path(tempfile.mkdtemp())
tmpdir.mkdir(exist_ok=True)
files = pathlib.Path("cache/build_tensordataset/multitask_tensors/tst").glob("*.pt")
f1 = next(files)
f2 = next(files)

shutil.copy(f1, tmpdir / f1.name)
shutil.copy(f2, tmpdir / f2.name)

valds = rd.PropertyGuaranteeDataset(
    path=tmpdir,
    tokenizer=tokenizer,
    nprops=nprops,
    target_props=target_props,
    target_positions=target_positions,
    sampling_weights=sampling_weights,
    distributed=True,
    rank=0,
    world_size=8
)

valds.set_sampling_weights({2600:1.0})
inp, teach, out = valds[0]

# Keep selecting until we get a sequence with token 16369
found = False
while not found:
    inp, teach, out = valds[0]
    if 16369 in out:
        found = True


model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")

print(inp.shape)
print(teach.shape)
print(out.shape)

inp = inp.unsqueeze(0)
teach = teach.unsqueeze(0)
pred = model(inp, teach)

value_indexes = list(tokenizer.value_indexes().values())
test = pred[:,list([3,5]), :][:, :, value_indexes]

print(test.shape)

print(test)


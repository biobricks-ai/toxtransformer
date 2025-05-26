# PYTHONPATH=./ python code/tests/hldout_properties.py
from collections import defaultdict
import torch
import pathlib
import shutil
import pandas as pd
from cvae.models.mixture_experts import MoE
from cvae.models.datasets.restricted_dataset import PropertyGuaranteeDataset
from tqdm import tqdm

# --- Setup ---
tmpdir = pathlib.Path("cache/tests/hldout_properties/tmp")
tmpdir.mkdir(exist_ok=True, parents=True)

source_dir = pathlib.Path("cache/build_tensordataset/multitask_tensors/hld")
files = list(source_dir.glob("*.pt"))
shutil.copy(files[0], tmpdir / "0.pt")
shutil.copy(files[1], tmpdir / "1.pt")
shutil.copy(files[2], tmpdir / "2.pt")
shutil.copy(files[3], tmpdir / "3.pt")
shutil.copy(files[4], tmpdir / "4.pt")

tokenizer = MoE.load('brick/moe/').tokenizer
tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()))

bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
target_props = bp['property_token'].tolist()
sampling_weights = {prop: weight for prop, weight in zip(target_props, bp['weight'].tolist())}
target_positions = [0, 4]
nprops = 500

dataset = PropertyGuaranteeDataset(
    path=source_dir,
    tokenizer=tokenizer,
    nprops=nprops,
    target_props=target_props,
    target_positions=target_positions,
    sampling_weights=sampling_weights,
    distributed=False
)

# --- Test ---
token_inputs = defaultdict(set)
mismatches = []

def extract_tokens(tensor, valid_tokens):
    return set(tensor[torch.isin(tensor, valid_tokens)].tolist())

def hash_input(tensor):
    return "_".join(map(str, tensor.tolist()))

# Run test
mismatches = []
for _ in tqdm(range(500)):  # Feel free to increase
    idx, target_prop = dataset._sample_index_and_property()

    file_idx = next(j for j in range(len(dataset.cumulative_lengths) - 1)
                    if dataset.cumulative_lengths[j+1] > idx)
    local_idx = idx - dataset.cumulative_lengths[file_idx]

    selfies = dataset.data[file_idx][0][local_idx]
    assay_vals = dataset.data[file_idx][1][local_idx]
    properties = dataset._extract_property_values(assay_vals)
    target_pos = target_positions[0]

    # Generate output exactly like __getitem__
    rearranged = dataset._rearrange_properties(properties, target_prop, target_pos)
    flat = [item for pair in rearranged for item in pair][:nprops*2]
    raw_out = torch.tensor([tokenizer.SEP_IDX] + flat + [tokenizer.END_IDX])
    if raw_out.size(0) < (nprops * 2 + 2):
        raw_out = torch.nn.functional.pad(raw_out, (0, nprops * 2 + 2 - raw_out.size(0)), value=tokenizer.PAD_IDX)

    # Compare tokens
    output_tokens = extract_tokens(raw_out, tokenizer.assay_indexes_tensor)
    true_tokens = extract_tokens(assay_vals, tokenizer.assay_indexes_tensor)
    for token in output_tokens:
        if token not in true_tokens:
            mismatches.append((token, hash_input(selfies)))

# Report
if mismatches:
    df = pd.DataFrame(mismatches, columns=["token", "input_hash"])
    print("❌ Found property mismatches (properties not in original data):")
    print(df.drop_duplicates().to_string(index=False))
else:
    print("✅ No mismatches: all output properties were present in the original inputs.")
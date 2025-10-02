import pathlib
import torch
import numpy as np
import json
import tqdm
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from cvae.models.multitask_transformer import process_assay_vals
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
import cvae.tokenizer

# Set up logging
outdir = pathlib.Path("cache/pack_multitask_tensors")
outdir.mkdir(exist_ok=True, parents=True)

log_dir = outdir / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "log.txt"),
        logging.StreamHandler()
    ]
)

def process_file(file_path, pad_idx, sep_idx, end_idx, nprops, epochs):
    file_data = torch.load(file_path, map_location="cpu", weights_only=False)
    selfies = file_data["selfies"]
    assay_vals = file_data["assay_vals"]

    selfies_out = []
    tch_out = []
    out_out = []

    for i in range(len(assay_vals)):
        for _ in range(epochs):
            t, o = process_assay_vals(assay_vals[i], pad_idx, sep_idx, end_idx, nprops)
            tch_out.append(t)
            out_out.append(o)
            selfies_out.append(selfies[i])

    return (
        torch.stack(selfies_out),  # [M, seq_len]
        torch.stack(tch_out),      # [M, prop_len]
        torch.stack(out_out),      # [M, prop_len]
    )

def pack_dataset(split_dir: pathlib.Path, out_path: pathlib.Path, tokenizer, nprops: int, epochs: int):
    logging.info(f"Packing dataset from {split_dir} → {out_path.stem}")
    pad_idx, sep_idx, end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX
    file_paths = sorted(split_dir.glob("*.pt"))
    logging.info(f"Found {len(file_paths)} files")

    selfies_list, tch_list, out_list = [], [], []

    # Set multiprocessing start method to 'spawn' to avoid deadlocks
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as executor:
        futures = {
            executor.submit(process_file, fp, pad_idx, sep_idx, end_idx, nprops, epochs): fp.name
            for fp in file_paths
        }
        with tqdm.tqdm(total=len(futures), desc=f"Packing {split_dir.name}") as pbar:
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    s, t, o = fut.result()
                    selfies_list.append(s)
                    tch_list.append(t)
                    out_list.append(o)
                except Exception as e:
                    logging.error(f"[{name}] {e}")
                pbar.update(1)

    # concatenate once
    selfies_all = torch.cat(selfies_list, dim=0)
    tch_all     = torch.cat(tch_list,     dim=0)
    out_all     = torch.cat(out_list,     dim=0)

    logging.info(f"Final shapes: selfies={selfies_all.shape}, tch={tch_all.shape}, out={out_all.shape}")

    # convert to numpy and dump raw .dat
    prefix = str(out_path.with_suffix(""))
    selfies_all.cpu().numpy().tofile(f"{prefix}_selfies.dat")
    tch_all.cpu().numpy().tofile(    f"{prefix}_tch.dat")
    out_all.cpu().numpy().tofile(    f"{prefix}_out.dat")

    # write meta.json with shapes and dtypes
    meta = {
        "selfies": {"shape": tuple(selfies_all.shape), "dtype": str(selfies_all.dtype)},
        "tch":     {"shape": tuple(tch_all.shape),     "dtype": str(tch_all.dtype)},
        "out":     {"shape": tuple(out_all.shape),     "dtype": str(out_all.dtype)},
    }
    with open(f"{prefix}_meta.json", "w") as f:
        json.dump(meta, f)

    logging.info(f"Dumped .dat + meta.json for split {split_dir.name}")

if __name__ == "__main__":
    indir   = pathlib.Path("cache/build_tensordataset/multitask_tensors")
    nprops  = 20
    epochs  = 2

    logging.info("Starting tensor packing")
    logging.info(f"Input:  {indir}")
    logging.info(f"Output: {outdir}")

    tokenizer = SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    logging.info("Loaded tokenizer")

    for split in ["trn", "tst", "hld"]:
        logging.info(f"→ {split}")
        split_dir = indir / split
        out_pt    = outdir / f"packed_{split}.pt"  # used only for naming
        pack_dataset(split_dir, out_pt, tokenizer, nprops, epochs)

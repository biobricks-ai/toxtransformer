# PYTHONPATH=./ python code/2_4_merge_tensordataset.py
import torch
import pathlib
import shutil
from tqdm import tqdm

def merge_pt_files(input_dir: str, output_dir: pathlib.Path, nchunks: int = 16):
    input_path = pathlib.Path(input_dir)
    assert input_path.exists() and input_path.is_dir(), f"Invalid path: {input_dir}"

    pt_files = sorted(input_path.glob("*.pt"))
    print(f"Found {len(pt_files)} files in {input_dir}. Splitting into {nchunks} chunks...")

    file_chunks = [[] for _ in range(nchunks)]
    for i, f in enumerate(pt_files):
        file_chunks[i % nchunks].append(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, chunk in enumerate(file_chunks):
        merged = {"selfies": [], "assay_vals": []}
        print(f"Merging chunk {i+1}/{nchunks} with {len(chunk)} files...")
        for pt_file in tqdm(chunk, leave=False):
            data = torch.load(pt_file, map_location="cpu")
            merged["selfies"].extend(data["selfies"])
            merged["assay_vals"].extend(data["assay_vals"])

        out_file = output_dir / f"chunk_{i:02d}.pt"
        print(f"Saving to {out_file}")
        torch.save(merged, out_file)

    print(f"✅ Saved {nchunks} chunks to {output_dir}")

# Output base directory
base_outdir = pathlib.Path('cache/merge_tensordataset/multitask_tensors')
shutil.rmtree(base_outdir, ignore_errors=True)
base_outdir.mkdir(parents=True, exist_ok=True)

# Merge each split into its own subdirectory
merge_pt_files("cache/build_tensordataset/multitask_tensors/trn", base_outdir / "trn")
merge_pt_files("cache/build_tensordataset/multitask_tensors/tst", base_outdir / "tst")
merge_pt_files("cache/build_tensordataset/multitask_tensors/hld", base_outdir / "hld")

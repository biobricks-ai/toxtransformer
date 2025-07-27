# import tqdm
# import time
# import torch
# import numpy as np
# import json
# from torch.utils.data import DataLoader
# import cvae.tokenizer
# from cvae.models.mixture_experts import MoE
# import pathlib
# import logging
# import cvae.utils

# cachepath = pathlib.Path("cache/tests/speed_benchmarks")
# cachepath.mkdir(parents=True, exist_ok=True)

# logdir = pathlib.Path("cache/tests/speed_benchmarks/logs")
# logdir.mkdir(exist_ok=True, parents=True)
# cvae.utils.setup_logging(logdir / "speed_benchmarks.log", logging)

# logging.info("Starting speed benchmark")

# # Update this to match your actual dataset location
# path_prefix = "cache/pack_multitask_tensors/packed_trn"

# # Load tokenizer
# logging.info("Loading tokenizer...")
# tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

# # Load dataset
# logging.info("Loading dataset...")
# dataset = FastPackedSequenceShiftDataset(path_prefix)

# # Set up DataLoader
# batch_size = 128
# num_workers = 8
# logging.info(f"Creating DataLoader with batch_size={batch_size}, num_workers={num_workers}")
# loader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     pin_memory=True,
#     prefetch_factor=4,
#     persistent_workers=True
# )

# # Load model and move to device
# # device = torch.device("cuda")
# device = torch.device("cpu")
# logging.info(f"Loading model to {device}...")
# model = MoE.load("brick/moe")
# model.to(device)
# model.eval()

# # Benchmark config
# n_batches = 50
# timings = {
#     "data_load": [],
#     "cuda_transfer": [],
#     "forward": [],
#     "total": []
# }

# # Run benchmark
# loader_iter = iter(loader)
# logging.info(f"Starting benchmark loop for {n_batches} batches")
# pbar = tqdm.tqdm(range(n_batches), desc="Benchmarking")
# for i in pbar:
#     t0 = time.time()
#     batch = next(loader_iter)
#     t1 = time.time()

#     batch = [x.to(device, non_blocking=True) for x in batch]
#     # torch.cuda.synchronize()
#     t2 = time.time()

#     # with torch.no_grad():
#     #     _ = model(*batch)
#     # torch.cuda.synchronize()
#     t3 = time.time()

#     timings["data_load"].append(t1 - t0)
#     timings["cuda_transfer"].append(t2 - t1)
#     timings["forward"].append(t3 - t2)
#     timings["total"].append(t3 - t0)

#     pbar.set_postfix({
#         'total': f'{t3-t0:.3f}s',
#         'load': f'{t1-t0:.3f}s',
#         'transfer': f'{t2-t1:.3f}s',
#         'forward': f'{t3-t2:.3f}s'
#     })
#     logging.debug(f"Batch {i}: total={t3-t0:.3f}s, load={t1-t0:.3f}s, transfer={t2-t1:.3f}s, forward={t3-t2:.3f}s")

# # Print summary
# logging.info("\nBenchmark Summary:")
# for key in timings:
#     arr = np.array(timings[key])
#     msg = f"{key:>12}: mean={arr.mean():.4f}s | std={arr.std():.4f}s"
#     print(msg)
#     logging.info(msg)

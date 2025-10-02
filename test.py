# Simplified masking test script
import os, pathlib, torch
import cvae.tokenizer, cvae.models.multitask_encoder as mte
from cvae.models.datasets import SimplePropertyMappedDataset
import cvae.models.datasets.custom_sampler as custom_sampler

# Setup
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Load stuff
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load("brick/selfies_property_val_tokenizer")
config = mte.MultitaskEncoderConfig(hdim=64, nhead=4, num_layers=2, output_size=2, dropout_rate=0.0, attention_dropout=0.0, layer_dropout=0.0, drop_path_rate=0.0)
model = mte.MultitaskEncoder(tokenizer=tokenizer, config=config)
model.eval()

# Quick dataset
paths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))[:1]
dataset = SimplePropertyMappedDataset(paths=paths, tokenizer=tokenizer, target_properties=list(range(10)), nprops=2)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, sampler=custom_sampler.FastDistributedSampler(dataset, 1, 0, False), num_workers=0)

# Get batch
itdl = iter(dataloader)
selfies, properties, values, mask = next(itdl)
print("Shapes:", selfies.shape, properties.shape, values.shape, mask.shape)

# test model
res = model(selfies, properties, values, mask)

# Test masking
with torch.no_grad():
    # Create masks
    molecule_mask = selfies != model.token_pad_idx
    pv_embeddings = model.create_pv_teacher_forcing(properties, values)
    
    # SELFIES-only mask (all PV tokens visible)
    pv_mask = mask.repeat_interleave(2, dim=1)
    selfies_only_mask = torch.cat([molecule_mask, pv_mask], dim=1)
    
    # No mask
    no_mask = None
    
    print("Molecule mask sample:", molecule_mask[0].tolist())
    print("SELFIES-only mask sample:", selfies_only_mask[0].tolist())

# Test assertion: SELFIES masking should make a difference
with torch.no_grad():
    full_sequence = torch.cat([
        model.embedding_selfies(selfies) * model.embed_scale,
        pv_embeddings * model.embed_scale
    ], dim=1)
    
    # Compare with and without SELFIES masking
    decoded_masked = model.decoder(full_sequence, mask=selfies_only_mask)
    decoded_unmasked = model.decoder(full_sequence, mask=no_mask)
    
    max_diff = torch.max(torch.abs(decoded_masked - decoded_unmasked)).item()
    print(f"Max difference with/without SELFIES masking: {max_diff:.6f}")
    
    # Assertion: masking should make a measurable difference
    assert max_diff > 1e-6, f"SELFIES masking had no effect! Max diff: {max_diff}"
    print("✅ SELFIES masking is working - outputs differ significantly")

# Test final model output
with torch.no_grad():
    logits = model(selfies, properties, values, mask)
    print("Final logits shape:", logits.shape)
    print("No NaN:", not torch.any(torch.isnan(logits)).item())
    print("No Inf:", not torch.any(torch.isinf(logits)).item())

print("✅ All tests passed!")
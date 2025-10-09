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

# do different smiles of the same compound get different selfies?
import rdkit, rdkit.Chem
import selfies, random
import selfies as sf

from rdkit import Chem
import cvae.tokenizer.selfies_tokenizer
tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('cache/preprocess_tokenizer/selfies_tokenizer.json')

def generate_alternative_smiles(smiles, num_alternatives=5, max_attempts=50, seed=42):
    """
    Generate alternative valid SMILES representations of the same molecule.
    Returns a list including the original SMILES plus alternatives.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]  # Return original if parsing fails
    
    # Set seed for reproducibility per molecule
    # Use hash of SMILES to ensure same alternatives are generated for same molecule
    mol_seed = hash(smiles) % (2**32) + seed
    random.seed(mol_seed)
    
    alternatives = set()
    alternatives.add(smiles)  # Include original
    
    attempts = 0
    while len(alternatives) < num_alternatives + 1 and attempts < max_attempts:
        
        # Generate non-canonical SMILES with randomization
        random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        
        # Verify the SMILES is valid by parsing it
        if Chem.MolFromSmiles(random_smiles) is not None:
            alternatives.add(random_smiles)
        
        attempts += 1
    
    return list(alternatives)

def generate_alternative_selfies_and_encode(smiles, tokenizer, num_alternatives=5, max_attempts=50, seed=42):
    """
    Generate alternative SMILES, convert to SELFIES, and encode them.
    Returns a list of encoded SELFIES arrays.
    """
    try:
        # Get alternative SMILES
        alternative_smiles = generate_alternative_smiles(smiles, num_alternatives, max_attempts, seed)
        
        # Convert each SMILES to SELFIES and encode
        encoded_alternatives = []
        for alt_smiles in alternative_smiles:
            try:
                selfies_str = sf.encoder(alt_smiles)
                if selfies_str:
                    encoded = tokenizer.encode(selfies_str)
                    encoded_alternatives.append(encoded)
            except Exception as e:
                logging.debug(f"Failed to encode SMILES {alt_smiles}: {e}")
                continue
        
        # If no alternatives could be encoded, return empty list
        if not encoded_alternatives:
            logging.warning(f"Could not encode any alternatives for SMILES: {smiles}")
        
        return encoded_alternatives
        
    except Exception as e:
        logging.warning(f"Alternative generation failed for SMILES {smiles}: {str(e)}")
        return []

apap = 'CC(=O)NC1=CC=C(C=C1)O'
mol = rdkit.Chem.MolFromSmiles(apap)
apap2 = rdkit.Chem.MolToSmiles(mol, doRandom=True)

sf1 = selfies.encoder(apap)
sf2 = selfies.encoder(apap2)

alts = generate_alternative_smiles(apap, num_alternatives=5)

smiles = "NS(=O)(=O)c1ccc(OCCN2CCN(c3ccc(C(F)(F)F)cc3[N+](=O)[O-])CC2)cc1"
alts = generate_alternative_smiles(test, num_alternatives=5)
altsf = generate_alternative_selfies_and_encode(test, tokenizer, num_alternatives=5)
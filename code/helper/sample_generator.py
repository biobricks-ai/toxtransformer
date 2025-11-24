"""
Helper function to generate distinct samples ready for multitask encoder inference.

Given a SELFIES string, nprops, and a property_token, this function queries the
brick/cvae.sqlite database to retrieve actual property values for the molecule,
then generates distinct samples with different property combinations that can be
directly fed to a MultitaskEncoder model.
"""

import torch
from typing import List, Tuple, Optional, Dict
import random
import sqlite3
import pandas as pd
from pathlib import Path


def get_molecule_properties_by_smiles(
    smiles: str,
    db_path: str = "brick/cvae.sqlite"
) -> pd.DataFrame:
    """
    Query the database for all property-value pairs for a given SMILES string.

    Args:
        smiles: SMILES string representation of molecule
        db_path: Path to SQLite database (default: brick/cvae.sqlite)

    Returns:
        DataFrame with columns: property_token, value, smiles
        Empty DataFrame if molecule not found

    Example:
        >>> df = get_molecule_properties_by_smiles("CCO")
        >>> print(df)
           property_token  value smiles
        0             100      1    CCO
        1             200      0    CCO
        ...
    """
    conn = sqlite3.connect(db_path)

    query = """
    SELECT property_token, value, smiles
    FROM activity
    WHERE smiles = ?
    """

    df = pd.read_sql_query(query, conn, params=(smiles,))
    conn.close()

    return df


def canonicalize_smiles(smiles: str) -> str:
    """
    Canonicalize a SMILES string using RDKit.

    Args:
        smiles: SMILES string

    Returns:
        Canonical SMILES string
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)
    except ImportError:
        # If RDKit not available, return as-is
        return smiles


def get_molecule_properties(
    selfies_str: str,
    db_path: str = "brick/cvae.sqlite"
) -> pd.DataFrame:
    """
    Query the database for all property-value pairs for a given SELFIES string.

    Args:
        selfies_str: SELFIES string representation of molecule
        db_path: Path to SQLite database (default: brick/cvae.sqlite)

    Returns:
        DataFrame with columns: property_token, value, smiles
        Empty DataFrame if molecule not found

    Example:
        >>> df = get_molecule_properties("[C][C][O]")
        >>> print(df)
           property_token  value              smiles
        0             100      1           CCO
        1             200      0           CCO
        ...
    """
    # Convert SELFIES to SMILES for lookup
    try:
        import selfies as sf
        smiles = sf.decoder(selfies_str)
        # Canonicalize to match database format
        smiles = canonicalize_smiles(smiles)
    except Exception as e:
        raise ValueError(f"Could not decode SELFIES '{selfies_str}': {e}")

    # Try canonical SMILES first
    df = get_molecule_properties_by_smiles(smiles, db_path)

    # If not found, the database might have non-canonical SMILES
    # We need to search using InChI or other method
    if len(df) == 0:
        # Try to find by InChI
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                inchi = Chem.MolToInchi(mol)
                conn = sqlite3.connect(db_path)
                query = """
                SELECT property_token, value, smiles
                FROM activity
                WHERE inchi = ?
                """
                df = pd.read_sql_query(query, conn, params=(inchi,))
                conn.close()
        except ImportError:
            pass

    return df


def generate_encoder_samples_from_db(
    selfies_str: str,
    nprops: int,
    property_token: int,
    tokenizer,
    num_samples: int = 10,
    pad_idx: int = 0,
    seed: Optional[int] = None,
    device: str = 'cpu',
    db_path: str = "brick/cvae.sqlite",
    exclude_target_from_context: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate samples using actual property values from the database.

    This function:
    1. Queries brick/cvae.sqlite for all known property-value pairs for the molecule
    2. Separates the target property from context properties
    3. Generates num_samples distinct combinations by randomly selecting context properties
    4. Returns tensors ready for model inference

    Args:
        selfies_str: SELFIES string representation of molecule
        nprops: Number of properties per sample (includes target property)
        property_token: Target property token to predict (placed at position nprops-1)
        tokenizer: SelfiesPropertyValTokenizer instance
        num_samples: Number of distinct samples to generate
        pad_idx: Padding index (default: 0)
        seed: Random seed for reproducibility
        device: Device to place tensors on ('cpu' or 'cuda')
        db_path: Path to SQLite database
        exclude_target_from_context: If True, don't use target property in context positions

    Returns:
        Tuple of (selfies_batch, properties_batch, values_batch, mask_batch)

    Raises:
        ValueError: If molecule not found in database or insufficient properties available

    Example:
        >>> import cvae.tokenizer.selfies_property_val_tokenizer as spt
        >>> tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        >>>
        >>> # Generate samples using real data from database
        >>> selfies_batch, props, vals, mask = generate_encoder_samples_from_db(
        ...     selfies_str="[C][C][O]",
        ...     nprops=5,
        ...     property_token=100,
        ...     tokenizer=tokenizer,
        ...     num_samples=50
        ... )
        >>> # All property values come from actual database records
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device_obj = torch.device(device)

    # Query database for molecule's properties
    prop_df = get_molecule_properties(selfies_str, db_path)

    if len(prop_df) == 0:
        raise ValueError(f"Molecule with SELFIES '{selfies_str}' not found in database")

    # Check if target property exists for this molecule
    target_data = prop_df[prop_df['property_token'] == property_token]
    if len(target_data) == 0:
        raise ValueError(
            f"Property token {property_token} not found for this molecule. "
            f"Available properties: {sorted(prop_df['property_token'].unique().tolist())}"
        )

    # Get target value
    target_value = target_data['value'].iloc[0]

    # Get context properties (excluding target if specified)
    if exclude_target_from_context:
        context_df = prop_df[prop_df['property_token'] != property_token]
    else:
        context_df = prop_df.copy()

    if len(context_df) < nprops - 1:
        raise ValueError(
            f"Insufficient properties for this molecule. Need {nprops-1} context properties, "
            f"but only {len(context_df)} available (excluding target)."
        )

    # Encode SELFIES
    encoded_selfies = tokenizer.selfies_tokenizer.encode_selfies(selfies_str)
    selfies_tensor = torch.tensor(encoded_selfies, dtype=torch.long, device=device_obj)
    seq_len = selfies_tensor.size(0)

    # Prepare output tensors
    selfies_batch = selfies_tensor.unsqueeze(0).expand(num_samples, seq_len).clone()
    properties_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device_obj)
    values_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device_obj)
    mask_batch = torch.zeros((num_samples, nprops), dtype=torch.bool, device=device_obj)

    # Generate each sample with different property combinations
    for i in range(num_samples):
        # Randomly select (nprops - 1) context properties
        if len(context_df) == nprops - 1:
            # Use all available context properties
            sampled_context = context_df
        else:
            # Randomly sample without replacement
            sampled_context = context_df.sample(n=nprops - 1, replace=False)

        # Shuffle for variety
        sampled_context = sampled_context.sample(frac=1)

        # Build property and value lists
        context_props = sampled_context['property_token'].tolist()
        context_vals = sampled_context['value'].tolist()

        # Place target property at the end
        all_props = context_props + [property_token]
        all_values = context_vals + [target_value]

        # Fill tensors
        properties_batch[i, :nprops] = torch.tensor(all_props, dtype=torch.long, device=device_obj)
        values_batch[i, :nprops] = torch.tensor(all_values, dtype=torch.long, device=device_obj)
        mask_batch[i, :nprops] = True

    return selfies_batch, properties_batch, values_batch, mask_batch


def generate_encoder_samples(
    selfies: torch.Tensor,
    nprops: int,
    property_token: int,
    available_properties: List[int],
    available_values: List[int],
    num_samples: int = 10,
    pad_idx: int = 0,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate distinct samples ready for multitask encoder inference.

    Args:
        selfies: Encoded SELFIES tensor [seq_len], the molecular representation
        nprops: Number of properties to include in each sample
        property_token: The target property token to predict (will be placed at position nprops-1)
        available_properties: List of property tokens to randomly sample from (excluding property_token)
        available_values: List of value tokens to randomly assign to properties
        num_samples: Number of distinct samples to generate
        pad_idx: Padding index for properties and values tensors (default: 0)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (selfies_batch, properties_batch, values_batch, mask_batch):
        - selfies_batch: [num_samples, seq_len] - replicated SELFIES
        - properties_batch: [num_samples, nprops] - property tokens
        - values_batch: [num_samples, nprops] - value tokens
        - mask_batch: [num_samples, nprops] - boolean mask for valid property-value pairs

    Example:
        >>> tokenizer = SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        >>> selfies = torch.tensor([1, 2, 3, ...])  # encoded SELFIES
        >>> property_token = 500  # target property to predict
        >>> available_props = [100, 200, 300, 400]  # other properties
        >>> available_vals = [0, 1]  # binary values
        >>>
        >>> selfies_batch, props, vals, mask = generate_encoder_samples(
        ...     selfies, nprops=5, property_token=500,
        ...     available_properties=available_props,
        ...     available_values=available_vals,
        ...     num_samples=10
        ... )
        >>> # Now ready for model inference:
        >>> logits = model(selfies_batch, props, vals, mask)
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # Validate inputs
    if nprops < 1:
        raise ValueError(f"nprops must be at least 1, got {nprops}")

    if property_token in available_properties:
        raise ValueError(f"property_token {property_token} should not be in available_properties")

    if len(available_properties) < nprops - 1:
        raise ValueError(
            f"Need at least {nprops-1} available properties to fill {nprops} slots "
            f"(with target property), but only {len(available_properties)} provided"
        )

    if len(available_values) < 1:
        raise ValueError("Need at least 1 available value")

    # Prepare output tensors
    device = selfies.device
    seq_len = selfies.size(0)

    selfies_batch = selfies.unsqueeze(0).expand(num_samples, seq_len).clone()
    properties_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device)
    values_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device)
    mask_batch = torch.zeros((num_samples, nprops), dtype=torch.bool, device=device)

    # Generate each sample with different property combinations
    for i in range(num_samples):
        # Randomly select (nprops - 1) properties from available_properties
        if len(available_properties) == nprops - 1:
            # Use all available properties
            selected_props = available_properties.copy()
        else:
            # Randomly sample without replacement
            selected_props = random.sample(available_properties, nprops - 1)

        # Shuffle the selected properties for variety
        random.shuffle(selected_props)

        # Assign random values to the selected properties
        selected_values = [random.choice(available_values) for _ in range(len(selected_props))]

        # Place target property at the last position (nprops - 1)
        # Note: We use a placeholder value for the target since we're predicting it
        # The actual value doesn't matter as the model will predict position nprops-1
        target_value = random.choice(available_values)  # placeholder

        # Combine: other properties + target property at end
        all_props = selected_props + [property_token]
        all_values = selected_values + [target_value]

        # Fill the tensors
        actual_len = len(all_props)
        properties_batch[i, :actual_len] = torch.tensor(all_props, dtype=torch.long, device=device)
        values_batch[i, :actual_len] = torch.tensor(all_values, dtype=torch.long, device=device)
        mask_batch[i, :actual_len] = True

    return selfies_batch, properties_batch, values_batch, mask_batch


def generate_encoder_samples_from_molecule(
    selfies_str: str,
    nprops: int,
    property_token: int,
    tokenizer,
    available_properties: Optional[List[int]] = None,
    available_values: Optional[List[int]] = None,
    num_samples: int = 10,
    pad_idx: int = 0,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience wrapper that encodes a SELFIES string and generates samples.

    Args:
        selfies_str: SELFIES string representation of molecule
        nprops: Number of properties per sample
        property_token: Target property token to predict
        tokenizer: SelfiesPropertyValTokenizer instance
        available_properties: List of property tokens to sample from. If None, will use
                            a subset of tokenizer's available properties (excluding property_token)
        available_values: List of value tokens. If None, will use [0, 1] for binary classification
        num_samples: Number of distinct samples to generate
        pad_idx: Padding index (default: 0)
        seed: Random seed for reproducibility
        device: Device to place tensors on ('cpu' or 'cuda')

    Returns:
        Same as generate_encoder_samples()

    Example:
        >>> import cvae.tokenizer.selfies_property_val_tokenizer as spt
        >>> tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        >>>
        >>> # Generate samples for a specific molecule
        >>> selfies_str = "[C][C][O]"
        >>> samples = generate_encoder_samples_from_molecule(
        ...     selfies_str, nprops=5, property_token=500, tokenizer=tokenizer,
        ...     num_samples=20, device='cuda'
        ... )
        >>> selfies_batch, props, vals, mask = samples
        >>>
        >>> # Run inference
        >>> logits = model(selfies_batch, props, vals, mask)
        >>> probs = torch.softmax(logits[:, -1, :], dim=-1)  # probabilities for target property
    """
    device_obj = torch.device(device)

    # Encode SELFIES string
    encoded_selfies = tokenizer.selfies_tokenizer.encode_selfies(selfies_str)
    selfies_tensor = torch.tensor(encoded_selfies, dtype=torch.long, device=device_obj)

    # Set default available_properties if not provided
    if available_properties is None:
        # Use a subset of all available properties from tokenizer
        # Exclude the target property_token
        all_props = list(range(tokenizer.num_assays))
        if property_token in all_props:
            all_props.remove(property_token)

        # If we have too many properties, randomly sample a reasonable subset
        if len(all_props) > 100:
            random.seed(seed)
            available_properties = random.sample(all_props, 100)
        else:
            available_properties = all_props

    # Set default values (binary classification)
    if available_values is None:
        available_values = [0, 1]

    return generate_encoder_samples(
        selfies=selfies_tensor,
        nprops=nprops,
        property_token=property_token,
        available_properties=available_properties,
        available_values=available_values,
        num_samples=num_samples,
        pad_idx=pad_idx,
        seed=seed,
    )


def generate_property_permutations(
    selfies: torch.Tensor,
    property_token: int,
    other_properties: List[int],
    other_values: List[int],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate samples with all possible property orderings for a fixed set of properties.

    This is useful when you want to test how property order affects predictions,
    or to generate diverse samples that cover different property contexts.

    Args:
        selfies: Encoded SELFIES tensor [seq_len]
        property_token: The target property token (always placed last)
        other_properties: List of other property tokens to include
        other_values: Corresponding values for other_properties (must match length)
        pad_idx: Padding index (default: 0)

    Returns:
        Tuple of (selfies_batch, properties_batch, values_batch, mask_batch)
        Number of samples = number of permutations of other_properties

    Example:
        >>> # Test how 3 different property orderings affect the prediction
        >>> samples = generate_property_permutations(
        ...     selfies=encoded_selfies,
        ...     property_token=500,
        ...     other_properties=[100, 200, 300],
        ...     other_values=[1, 0, 1]
        ... )
        >>> # This will generate 6 samples (3! = 6 permutations)
    """
    if len(other_properties) != len(other_values):
        raise ValueError(
            f"Length mismatch: other_properties has {len(other_properties)} elements "
            f"but other_values has {len(other_values)} elements"
        )

    from itertools import permutations as perm

    device = selfies.device
    seq_len = selfies.size(0)

    # Generate all permutations of (property, value) pairs
    indices = list(range(len(other_properties)))
    all_perms = list(perm(indices))
    num_samples = len(all_perms)
    nprops = len(other_properties) + 1  # +1 for target property

    # Prepare output tensors
    selfies_batch = selfies.unsqueeze(0).expand(num_samples, seq_len).clone()
    properties_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device)
    values_batch = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device)
    mask_batch = torch.zeros((num_samples, nprops), dtype=torch.bool, device=device)

    # Fill each sample with a different permutation
    for i, perm_indices in enumerate(all_perms):
        # Reorder properties and values according to this permutation
        reordered_props = [other_properties[j] for j in perm_indices]
        reordered_vals = [other_values[j] for j in perm_indices]

        # Add target property at the end
        all_props = reordered_props + [property_token]
        all_values = reordered_vals + [0]  # placeholder value for target

        # Fill tensors
        properties_batch[i, :nprops] = torch.tensor(all_props, dtype=torch.long, device=device)
        values_batch[i, :nprops] = torch.tensor(all_values, dtype=torch.long, device=device)
        mask_batch[i, :nprops] = True

    return selfies_batch, properties_batch, values_batch, mask_batch

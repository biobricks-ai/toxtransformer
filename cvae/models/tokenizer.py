from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch, numpy as np

TOKEN_SOS = "^"
TOKEN_END = "$"
TOKEN_PAD = " "

charset = [
    TOKEN_SOS, TOKEN_END, TOKEN_PAD, 
    "(", ")", "[", "]", 
    ".", "=", "#", 
    "+", "-", 
    "@", "@@", 
    "/", "\\", 
    "%", 
    "*", 
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", 
    "b", "c", "n", "o", "p", "s", 
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Ne", 
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", 
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", 
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", 
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", 
    "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", 
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", 
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", 
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", 
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", 
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", 
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
]

TOKEN_SOS_IDX, TOKEN_PAD_IDX, TOKEN_END_IDX = [charset.index(t) for t in [TOKEN_SOS, TOKEN_PAD,TOKEN_END]]
nchar = len(charset)
char_to_index = {char: i for i, char in enumerate(charset)}

# smiles_char_to_idx("C[C@H]1[C@@H](Br)N(C)C(=O)NC1=O")
def smiles_char_to_idx(smiles, max_length=120):
    # Prepend the start character and append the end character
    modified_smiles = TOKEN_SOS + smiles + TOKEN_END

    # Initialize an empty list for indices
    indices = []

    # Parse SMILES string and handle two-character elements
    i = 0
    while i < len(modified_smiles):
        # Check if next two characters form a valid two-character element
        if i < len(modified_smiles) - 1 and modified_smiles[i:i+2] in char_to_index:
            element = modified_smiles[i:i+2]
            i += 2
        else:
            element = modified_smiles[i]
            i += 1

        # Add index of element to indices list
        indices.append(char_to_index.get(element, char_to_index[TOKEN_PAD]))

    # Truncate or pad indices array to ensure it's the right length
    indices = indices[:max_length] + [char_to_index[TOKEN_PAD]] * (max_length - len(indices))

    return indices

def idx_to_char(indices):
    return "".join(map(lambda x: charset[x], indices)).strip()

# smiles_one_hot("C[C@H]1[C@@H](Br)N(C)C(=O)NC1=O")
def smiles_one_hot(smiles, max_length=120):
    # Prepend the start character and append the end character
    modified_smiles = TOKEN_SOS + smiles + TOKEN_END

    # Initialize an empty list for indices
    indices = []

    # Parse SMILES string and handle two-character elements
    i = 0
    while i < len(modified_smiles):
        # Check if next two characters form a valid two-character element
        if i < len(modified_smiles) - 1 and modified_smiles[i:i+2] in char_to_index:
            element = modified_smiles[i:i+2]
            i += 2
        else:
            element = modified_smiles[i]
            i += 1

        # Add index of element to indices list
        indices.append(char_to_index.get(element, char_to_index[TOKEN_PAD]))

    # Truncate or pad indices array to ensure it's the right length
    indices = indices[:max_length] + [char_to_index[TOKEN_PAD]] * (max_length - len(indices))

    # Convert indices to one-hot encoding
    one_hot = np.eye(nchar, dtype=np.float32)[indices].T

    if one_hot.shape != (nchar, max_length):
        raise ValueError(f"Expected output shape {(nchar, max_length)}, but got {one_hot.shape}")

    return one_hot

def valid_smiles(smiles):
    is_mol = Chem.MolFromSmiles(smiles) is not None
    valid_char = set(smiles).issubset(set(charset))
    valid_length = len(smiles) <= 120
    return is_mol and valid_char and valid_length

      
def smiles_to_morgan_fingerprint(smiles, radius=2, nBits=2048):
    RDLogger.DisableLog('rdApp.*') # Disable RDKit warnings
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Generate the Morgan fingerprint
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    fingerprint_np = np.zeros((nBits,), dtype=np.uint8)
    for bit in fingerprint.GetOnBits():
        fingerprint_np[bit] = 1
        
    return fingerprint_np

def decode_smiles_from_indexes(vec):
    return "".join(map(lambda x: charset[x], vec)).strip()
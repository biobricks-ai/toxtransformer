import cvae.utils as utils
import os, re, flask, torch, faiss, numpy as np, pandas, csv
import torch.nn.functional as F
import rdkit
from tqdm import tqdm

from torch.utils.data import TensorDataset, DataLoader
from cvae.models import tokenizer, triplet
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Draw

import pickle, types

class TrainData:
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrainData, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    
    def __init__(self, tensors=None):
        self.ismiles, self.ilbl, self.ivalue, self.imorgan, self.ipvae = (0,1,2,3,4)
        if tensors:
            self.tensors = tensors
        elif not hasattr(self, 'tensors'):
            self.tensors = None
    
    @staticmethod
    def load():
        # Ensure the singleton instance is created or retrieved
        instance = TrainData()
        
        # Load tensors only if they haven't been loaded already
        if instance.tensors is None:
            instance.tensors = torch.load(utils.res("data/processed/train.pt")).tensors
        return instance
    
    def get_label_specific_tensors(self, label):
        indexes = torch.where(self.tensors[self.ilbl] == label)
        return [self.tensors[i][indexes] for i in [0,1,2,3,4]]


class AnalogueFinder:
    
    def __init__(self, contrastive_vae, vae, faiss_morgan, faiss_vae, faiss_contrastive_vae):
        self.contrastive_vae = contrastive_vae
        self.vae = vae
        self.faiss_morgan = faiss_morgan
        self.faiss_vae = faiss_vae
        self.faiss_contrastive_vae = faiss_contrastive_vae
        self.faiss_indexes = {
            "morgan": self.faiss_morgan,
            "vae": self.faiss_vae,
            "contrastive_vae": self.faiss_contrastive_vae
        }
        self.device = torch.device(f'cuda:0')
        self.train = TrainData.load()
    
    def to(self, device):
        self.contrastive_vae.to(device)
        self.vae.to(device)
        self.device = device
        return self
    
    def find_analogues(self, method, smiles, label, k=5):
        """
        Find analogues using the specified method (e.g., "morgan", "vae", "contrastive_vae"),
        smiles string, and label. Returns the top k nearest neighbors.
        """
        index_map = {
            "morgan": self.faiss_morgan,
            "vae": self.faiss_vae,
            "contrastive_vae": self.faiss_contrastive_vae
        }
        
        index = index_map[method][label]
        embedding = self.embed(method, smiles, label)
        D, I = index.search(embedding, k)
        return D, I
    
    def embed(self, method, smiles, label=None):
        
        if method == "morgan":
            return torch.Tensor(tokenizer.smiles_to_morgan_fingerprint(smiles)).to(self.device)
        
        tensor_smiles = torch.Tensor(tokenizer.smiles_one_hot(smiles)).to(self.device)
        tensor_smiles = tensor_smiles.unsqueeze(0)
        if method == "vae":
            return F.normalize(self.vae.encode(tensor_smiles)[0], p=2, dim=1)[0]
        
        tensor_label = torch.tensor([label]).long().to(self.device)
        if method == "contrastive_vae":
            return F.normalize(self.contrastive_vae.encode(tensor_label, tensor_smiles), p=2, dim=1)[0].detach()
        
        raise Exception("Invalid method")
    
    def knn(self, method, smiles, label, k=5):
        embedding = self.embed(method, smiles, label).detach().to('cpu').numpy()
        embedding = embedding.reshape(1,-1)
        
        index : faiss.IndexFlatIP = self.faiss_indexes[method][label]
        weights, indices = [x[0] for x in index.search(embedding, k)]
        
        label_tensors = self.train.get_label_specific_tensors(label)
        analogue_smiles_tensors = torch.argmax(label_tensors[self.train.ismiles][indices],dim=1)
        analogue_smiles = [tokenizer.decode_smiles_from_indexes(smi) for smi in analogue_smiles_tensors]
        analogue_values = label_tensors[self.train.ivalue][indices]
        
        prediction = torch.mean(analogue_values).item()
        res = {"weights":weights, "analogue_smiles":analogue_smiles, "analogue_values":analogue_values, "prediction":prediction}
        
        return types.SimpleNamespace(**res)
    
    def save(self, filename):
        self.train = None
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            instance = pickle.load(file)
            instance.train = TrainData.load()
            return instance
    
    @staticmethod
    def build(path):
        device = torch.device(f'cuda:0')
        
        # GET TRAIN DATA =================================================================================
        train = torch.load(utils.res("data/processed/train.pt")).tensors
        ismiles, ilbl, ivalue, imorgan, ipvae = (0,1,2,3,4)
            
        # GET MODELS =====================================================================================
        contrastive_vae = triplet.TripletModel.load("brick/vaecontrast.pt", "brick/pvae.pt", device)
        vae = contrastive_vae.pvae
        contrastive_vae.eval()
        vae.eval()
        
        # BUILD FAISS INDEXES =============================================================================
        faiss_morgan = {}
        for assay in tqdm(train[ilbl].unique()):
            lblmorgan = train[imorgan][torch.where(train[ilbl] == assay.item())].float()
            emb = np.ascontiguousarray(lblmorgan)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            faiss_morgan[assay.item()] = index
        
        faiss_vae = {}
        for assay in tqdm(train[ilbl].unique()):
            lbl_indices = torch.where(train[ilbl] == assay.item())[0]
            smiles, labels = train[ismiles][lbl_indices], train[ilbl][lbl_indices]
            loader = DataLoader(TensorDataset(smiles, labels), batch_size=1000)
            embeddings = []
            for i, (smi, lbl) in tqdm(enumerate(loader), total=len(loader)):
                smi = smi.to(device)
                embi = F.normalize(vae.encode(smi)[0], p=2, dim=1)
                embeddings.append(embi.detach().cpu())
            embeddings = torch.cat(embeddings)
            emb = np.ascontiguousarray(embeddings)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            faiss_vae[assay.item()] = index
                
        faiss_contrastive_vae = {}
        for assay in tqdm(train[ilbl].unique()):
            lbl_indices = torch.where(train[ilbl] == assay.item())[0]
            smiles, labels = train[ismiles][lbl_indices], train[ilbl][lbl_indices]
            loader = DataLoader(TensorDataset(smiles, labels), batch_size=1000)
            embeddings = []
            i, (smi, lbl) = next(enumerate(loader))
            for i, (smi, lbl) in tqdm(enumerate(loader), total=len(loader)):
                smi, lbl = smi.to(device), lbl.to(device).int()
                embi = F.normalize(contrastive_vae.encode(lbl, smi), p=2, dim=1)
                embeddings.append(embi.detach().cpu())
            embeddings = torch.cat(embeddings)
            emb = np.ascontiguousarray(embeddings)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            faiss_contrastive_vae[assay.item()] = index
            
        finder = AnalogueFinder(contrastive_vae=contrastive_vae,vae=vae,faiss_morgan=faiss_morgan, faiss_vae=faiss_vae, faiss_contrastive_vae=faiss_contrastive_vae)
        finder.save_to_file(path)

# AnalogueFinder.build("analoguefinder.pkl")

# device = torch.device(f'cuda:0')
# test = AnalogueFinder.load_from_file("analoguefinder.pkl").to(device)

# test.embed("morgan", "CC1=CC=C(C=C1)C(=O)O")

# test.embed("vae", "CC1=CC=C(C=C1)C(=O)O")
# test.embed("contrastive_vae", "CC1=CC=C(C=C1)C(=O)O", 1)

# from PIL import Image, ImageDraw

# input_smiles = "CC1=CC=C(C=C1)C(=O)O"

# input_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
# all_analogue_smiles = []
# all_analogue_values = []

# for method in ["morgan", "vae", "contrastive_vae"]:
#     weights, analogue_smiles, analogue_values, prediction = test.knn(method, input_smiles, label=20, k=4)
#     all_analogue_smiles += [input_smiles] + analogue_smiles
#     all_analogue_values += [0] + analogue_values.tolist()

# molecules = [Chem.MolFromSmiles(smi) for smi in all_analogue_smiles]
# img = Draw.MolsToGridImage(molecules, molsPerRow=5, subImgSize=(200, 200))

# # Convert RDKit Image to PIL Image
# img_pil = Image.new("RGB", img.size)
# img_pil.paste(img)

# # Create an overlay with colors
# overlay = Image.new("RGBA", img_pil.size, (255, 255, 255, 0))
# draw = ImageDraw.Draw(overlay)

# width, height = img_pil.size
# cell_width = width // 5
# cell_height = height // (len(all_analogue_smiles) // 5)

# for i, value in enumerate(all_analogue_values):
#     row, col = divmod(i, 5)
#     x1, y1 = col * cell_width, row * cell_height
#     x2, y2 = x1 + cell_width, y1 + cell_height
    
#     if value == 1:
#         color = (255, 0, 0, 64)  # Mostly transparent red
#     else:
#         color = (0, 255, 0, 64)  # Mostly transparent green
    
#     draw.rectangle([x1, y1, x2, y2], fill=color)

# # Merge the two images
# img_pil.paste(overlay, (0, 0), mask=overlay)
# img_pil.save("analogues.png")
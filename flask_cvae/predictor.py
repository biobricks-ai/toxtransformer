import json
import threading
import pandas as pd
import numpy as np
import cvae.models.multitask_encoder as mte
import cvae.spark_helpers as H
import torch
import torch.nn
import sqlite3
import itertools
import logging
import random
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

DEVICE = torch.device(f'cuda:0')

@dataclass
class Category:
    category: str
    reason: str
    strength: str

@dataclass
class Property:
    property_token: int
    source: str
    title: str
    metadata: dict
    categories: list[Category]

@dataclass
class Prediction:
    inchi: str
    property_token: int
    property: Property
    value: float

class Predictor:
    
    def __init__(self):
        self.dburl = 'brick/cvae.sqlite'
        self.dblock = threading.Lock()
        self.model = mte.MultitaskEncoder.load("brick/multitask_transformer_model").to(DEVICE)
        self.tokenizer = self.model.tokenizer
        self.model = torch.nn.DataParallel(self.model)  
        
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row 
        self.all_property_tokens = [r['property_token'] for r in conn.execute("SELECT DISTINCT property_token FROM property")]
        self.property_map = self.build_property_map()
        conn.close()
    
    def build_property_map(self):
        with self.dblock:
            conn = sqlite3.connect(self.dburl)
            conn.row_factory = lambda cursor, row: dict((cursor.description[i][0], value) for i, value in enumerate(row))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.property_token, p.title, p.data as metadata, s.source, c.category, pc.reason, pc.strength
                FROM property p
                INNER JOIN property_category pc ON p.property_id = pc.property_id 
                INNER JOIN category c ON pc.category_id = c.category_id
                INNER JOIN source s ON p.source_id = s.source_id
            """)
            res = cursor.fetchall()
            
            # Group results by property_token
            property_map = {}
            for property_token, group in itertools.groupby(res, key=lambda x: x['property_token']):
                group_list = list(group)
                categories = [Category(category=r['category'], reason=r['reason'], strength=r['strength']) 
                            for r in group_list]
                
                property = Property(property_token=property_token,
                                  title=group_list[0]['title'],
                                  metadata=json.loads(group_list[0]['metadata']),
                                  source=group_list[0]['source'],
                                  categories=categories)
                                  
                property_map[property_token] = property
                
            return property_map
    
    def _get_known_properties(self, inchi, category=None):
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()

        query = """
        SELECT source, inchi, prop.property_token, prop.data, cat.category, prop_cat.reason, prop_cat.strength, act.value_token, act.value 
        FROM activity act 
        INNER JOIN source src ON act.source_id = src.source_id 
        INNER JOIN property prop ON act.property_id = prop.property_id
        INNER JOIN property_category prop_cat ON prop.property_id = prop_cat.property_id
        INNER JOIN category cat ON prop_cat.category_id = cat.category_id
        WHERE inchi = ?"""
        
        params = [inchi]
        if category is not None:
            query += " AND cat.category = ?"
            params.append(category)
            
        cursor.execute(query, params)

        res = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return pd.DataFrame(res) if len(res) > 0 else pd.DataFrame(columns=['property_token'])
    
    # inchi = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"
    # property_token = 3022
    def predict_property_with_randomized_tensors(self, inchi, property_token, seed, num_rand_tensors=1000):
        if property_token not in self.all_property_tokens:
            logging.error(f"Property token {property_token} is not valid")
            return np.nan
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        selfies_tokens = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        selfies_input = selfies_tokens.view(1, -1).to(DEVICE)
        
        known_props = self._get_known_properties(inchi)
        known_props = known_props[known_props['property_token'] != property_token]
        
        if known_props.empty:
            # Simple case: no known properties, just predict the target property
            # Normalize the property token
            norm_property = property_token - self.tokenizer.selfies_offset
            properties = torch.LongTensor([[norm_property]]).to(DEVICE)
            values = torch.LongTensor([[0]]).to(DEVICE)  # Use normalized value 0 (which is value_token 1)
            mask = torch.BoolTensor([[True]]).to(DEVICE)
            
            with torch.no_grad():
                result_logit = self.model(selfies_input, properties, values, mask)
                return torch.softmax(result_logit, dim=-1).detach().cpu().numpy()

        # Complex case: randomize known properties and predict target
        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))
        
        rand_tensors_props = []
        rand_tensors_values = []
        rand_tensors_masks = []
        
        for i in range(num_rand_tensors):
            # Shuffle and truncate known properties
            shuffled_pairs = random.sample(property_value_pairs, min(4, len(property_value_pairs)))
            
            # Add target property
            target_value_token = self.tokenizer.value_id_to_token_idx(1)
            shuffled_pairs.append((property_token, target_value_token))
            
            # Prepare tensors
            props, vals = zip(*shuffled_pairs)
            
            # Normalize properties and values
            props_tensor = torch.LongTensor([props])
            vals_tensor = torch.LongTensor([vals])
            mask_tensor = torch.BoolTensor([[True] * len(props)])
            
            # Apply normalization
            norm_props = self.tokenizer.norm_properties(props_tensor.clone(), mask_tensor.clone())
            norm_vals = self.tokenizer.norm_values(vals_tensor.clone(), mask_tensor.clone())
            
            rand_tensors_props.append(norm_props)
            rand_tensors_values.append(norm_vals)
            rand_tensors_masks.append(mask_tensor)
        
        # Stack tensors
        properties_batch = torch.cat(rand_tensors_props, dim=0).to(DEVICE)
        values_batch = torch.cat(rand_tensors_values, dim=0).to(DEVICE)
        masks_batch = torch.cat(rand_tensors_masks, dim=0).to(DEVICE)
        selfies_batch = selfies_input.repeat(num_rand_tensors, 1)
        
        print(f"Batch shapes - selfies: {selfies_batch.shape}, properties: {properties_batch.shape}, values: {values_batch.shape}, masks: {masks_batch.shape}")
        
        with torch.no_grad():
            result_logit = self.model(selfies_batch, properties_batch, values_batch, masks_batch)
            # Get the last prediction (target property)
            target_logits = result_logit[:, -1, :]
            return torch.softmax(target_logits, dim=-1).detach().cpu().numpy()
    
    def predict_property(self, inchi, property_token, seed=137, num_rand_tensors=1000):
        predictions = self.predict_property_with_randomized_tensors(inchi, property_token, seed, num_rand_tensors=num_rand_tensors)
        
        if predictions.size == 0:
            logging.info(f"No predictions generated for InChI: {inchi} and property token: {property_token}")
            return np.nan
        
        token_property = self.property_map.get(property_token, None)
        # For binary classification (output_size=2), index 1 is the positive class
        meanpred = float(np.mean(predictions[:, 1], axis=0))
        prediction = Prediction(inchi=inchi, property_token=property_token, property=token_property, value=meanpred)
        return prediction
    
    def _build_random_tensors(self, inchi, seed, num_rand_tensors):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        selfies_tokens = torch.LongTensor(self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies))
        selfies_input = selfies_tokens.view(1, -1)
        
        known_props = self._get_known_properties(inchi)
        
        if known_props.empty:
            # Return empty tensors for properties/values when no known properties
            empty_props = torch.LongTensor([[0]])  # Normalized padding
            empty_values = torch.LongTensor([[0]])  # Normalized padding
            empty_mask = torch.BoolTensor([[False]])
            return selfies_input, empty_props, empty_values, empty_mask

        property_value_pairs = list(zip(known_props['property_token'], known_props['value_token']))
        
        rand_props_list = []
        rand_values_list = []
        rand_masks_list = []
        
        for _ in range(num_rand_tensors):
            # Shuffle and truncate properties
            shuffled_pairs = random.sample(property_value_pairs, min(4, len(property_value_pairs)))
            
            props, vals = zip(*shuffled_pairs)
            props_tensor = torch.LongTensor([props])
            vals_tensor = torch.LongTensor([vals])
            mask_tensor = torch.BoolTensor([[True] * len(props)])
            
            # Apply normalization
            norm_props = self.tokenizer.norm_properties(props_tensor.clone(), mask_tensor.clone())
            norm_vals = self.tokenizer.norm_values(vals_tensor.clone(), mask_tensor.clone())
            
            rand_props_list.append(norm_props)
            rand_values_list.append(norm_vals)
            rand_masks_list.append(mask_tensor)
        
        # Stack all tensors
        rand_props = torch.cat(rand_props_list, dim=0)
        rand_values = torch.cat(rand_values_list, dim=0)
        rand_masks = torch.cat(rand_masks_list, dim=0)
        
        print(f"Random tensors shapes - props: {rand_props.shape}, values: {rand_values.shape}, masks: {rand_masks.shape}")
        
        return selfies_input, rand_props, rand_values, rand_masks
    
    # test with inchi=InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H
    def predict_all_properties(self, inchi, seed=137, max_num_rand_tensors=100) -> list[Prediction]:
        selfies_input, rand_props, rand_values, rand_masks = self._build_random_tensors(inchi, seed, max_num_rand_tensors)
        
        # Remove duplicate rows from rand_props to get unique property combinations
        if rand_props.size(0) > 1:
            # Convert to tuples for uniqueness check
            prop_tuples = [tuple(row.tolist()) for row in rand_props]
            unique_indices = []
            seen = set()
            for i, tup in enumerate(prop_tuples):
                if tup not in seen:
                    seen.add(tup)
                    unique_indices.append(i)
            
            if unique_indices:
                rand_props = rand_props[unique_indices]
                rand_values = rand_values[unique_indices]
                rand_masks = rand_masks[unique_indices]
        
        num_rand_tensors = rand_props.size(0)
        
        # Create batches for all property predictions
        simultaneous_properties = 50  # Reduced batch size
        raw_preds = []
        
        for prop_start in tqdm(range(0, len(self.all_property_tokens), simultaneous_properties)):
            prop_end = min(prop_start + simultaneous_properties, len(self.all_property_tokens))
            current_prop_tokens = self.all_property_tokens[prop_start:prop_end]
            num_current_props = len(current_prop_tokens)
            
            # Build batch for current properties
            batch_selfies = []
            batch_props = []
            batch_values = []
            batch_masks = []
            
            for prop_token in current_prop_tokens:
                # Normalize the target property token
                norm_target_prop = prop_token - self.tokenizer.selfies_offset
                target_value_token = self.tokenizer.value_id_to_token_idx(1)
                norm_target_value = target_value_token - self.tokenizer.properties_offset
                
                for i in range(num_rand_tensors):
                    # Clone existing property context
                    existing_props = rand_props[i:i+1].clone()  # Keep batch dimension
                    existing_values = rand_values[i:i+1].clone()  # Keep batch dimension
                    existing_mask = rand_masks[i:i+1].clone()  # Keep batch dimension
                    
                    # Add the target property (already normalized)
                    new_prop = torch.LongTensor([[norm_target_prop]])
                    new_value = torch.LongTensor([[norm_target_value]])
                    new_mask = torch.BoolTensor([[True]])
                    
                    combined_props = torch.cat([existing_props, new_prop], dim=1)
                    combined_values = torch.cat([existing_values, new_value], dim=1)
                    combined_mask = torch.cat([existing_mask, new_mask], dim=1)
                    
                    batch_selfies.append(selfies_input)
                    batch_props.append(combined_props)
                    batch_values.append(combined_values)
                    batch_masks.append(combined_mask)
            
            # Stack batch tensors
            batch_selfies = torch.cat(batch_selfies, dim=0).to(DEVICE)
            batch_props = torch.cat(batch_props, dim=0).to(DEVICE)
            batch_values = torch.cat(batch_values, dim=0).to(DEVICE)
            batch_masks = torch.cat(batch_masks, dim=0).to(DEVICE)
            
            # Model inference
            with torch.no_grad():
                result_logit = self.model(batch_selfies, batch_props, batch_values, batch_masks)
                # Get predictions for the target property (last position)
                target_logits = result_logit[:, -1, :]  # Shape: [batch, 2] for binary classification
                target_probs = torch.softmax(target_logits, dim=-1)[:, 1]  # Get positive class probability
                
                # Calculate mean predictions across random contexts for each property
                target_probs_reshaped = target_probs.view(num_current_props, num_rand_tensors)
                batch_preds_mean = target_probs_reshaped.mean(dim=1)
                raw_preds.extend(batch_preds_mean.detach().cpu().numpy())
        
        raw_preds = [float(x) for x in raw_preds]
        property_tokens = [self.all_property_tokens[i] for i in range(len(raw_preds))]
        properties = [self.property_map.get(property_token, None) for property_token in property_tokens]
        preds = [Prediction(inchi=inchi, property_token=property_tokens[i], property=properties[i], value=raw_preds[i]) for i in range(len(raw_preds))]
        return preds

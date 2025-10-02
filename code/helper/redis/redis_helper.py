import redis
import pickle
import torch

class RedisDatasetCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.redis_client = redis.Redis(host=host, port=port, db=db)
    
    def save_dataset(self, dataset, key):
        """Save dataset to Redis"""
        print(f"Saving dataset to Redis key: {key}")
        
        # Save the core data structures
        data_to_save = {
            'target_properties': dataset.target_properties,
            'nprops': dataset.nprops,
            'property_to_samples': dataset.property_to_samples,
            'total_samples': dataset.total_samples
        }
        
        # Serialize with pickle
        serialized = pickle.dumps(data_to_save)
        self.redis_client.set(key, serialized)
        print(f"Dataset saved to Redis ({len(serialized)} bytes)")
    
    def load_dataset(self, key, tokenizer, nprops=1, assay_filter=[]):
        """Load dataset from Redis"""
        print(f"Loading dataset from Redis key: {key}")
        
        serialized = self.redis_client.get(key)
        if serialized is None:
            return None
            
        data = pickle.loads(serialized)
        
        # Reconstruct dataset object
        dataset = SimplePropertyMappedDataset.__new__(SimplePropertyMappedDataset)
        dataset.tokenizer = tokenizer
        dataset.pad_idx = tokenizer.PAD_IDX
        dataset.target_properties = data['target_properties']
        dataset.nprops = data['nprops']
        dataset.property_to_samples = data['property_to_samples']
        dataset.total_samples = data['total_samples']
        dataset.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None
        
        print(f"Dataset loaded from Redis")
        return dataset

# Usage in your dataset class
@staticmethod
def from_redis_or_create(paths, tokenizer, target_properties, nprops=1, 
                        assay_filter=[], redis_key="dataset"):
    cache = RedisDatasetCache()
    
    # Try loading from Redis first
    dataset = cache.load_dataset(redis_key, tokenizer, nprops, assay_filter)
    if dataset is not None:
        return dataset
    
    # Create new dataset
    print("Creating new dataset...")
    dataset = SimplePropertyMappedDataset(paths, tokenizer, target_properties, nprops, assay_filter)
    
    # Save to Redis
    cache.save_dataset(dataset, redis_key)
    return dataset
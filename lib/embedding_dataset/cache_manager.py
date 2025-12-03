"""
Cache management methods extracted from original EmbeddingDataset.
"""
import pickle
import torch
from pathlib import Path
import os


class CacheManager:
    """Handles cache loading and saving"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def _get_cache_path(self):
        """Get the cache file path for the processed dataset"""
        cache_dir = None
        if hasattr(self.dataset.conf.path, 'cache'):
            cache_dir = Path(self.dataset.conf.path.cache) / self.dataset.dataset_nickname
        elif hasattr(self.dataset.conf.path, 'working_dir'):
            cache_dir = Path(self.dataset.conf.path.working_dir) / 'cache' / self.dataset.dataset_nickname
        else:
            return None
        
        if self.dataset.embedding_type == 'multimodal':
            cache_id = "multimodal"
        else:
            cache_id = f"{self.dataset.embedding_type}_{self.dataset.embedding_format}"
        
        if self.dataset.debug:
            cache_id += "_debug"
        
        cache_file = cache_dir / f'processed_dataset_{cache_id}.pkl'
        return cache_file
    
    def _load_from_cache(self):
        """Load complete processed dataset from cache if available"""
        cache_file = self._get_cache_path()
        
        if not cache_file or not cache_file.exists():
            return False
        
        try:
            if self.verbose:
                print(f"Loading processed dataset from cache: {cache_file}")
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.dataset.info = cached_data['info']
            self.dataset.splitdict = cached_data['splitdict']
            self.dataset.clique2id = cached_data['clique2id']
            self.dataset._loaded_from_cache = True
            
            if self.verbose:
                total_versions = len(self.dataset.info)
                print(f"Loaded processed dataset with {total_versions} versions")
                for split_name, split_data in self.dataset.splitdict.items():
                    clique_count = len(split_data)
                    version_count = sum(len(versions) for versions in split_data.values())
                    print(f"  {split_name}: {clique_count} cliques, {version_count} versions")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading cache file {cache_file}: {e}")
                print("Will rebuild dataset...")
            return False
    
    def _save_to_cache(self):
        """Save the final processed dataset to cache"""
        cache_file = self._get_cache_path()
        
        if not cache_file:
            if self.verbose:
                print("Warning: Cannot save to cache - working_dir not configured")
            return
        
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'info': self.dataset.info,
                'splitdict': self.dataset.splitdict,
                'clique2id': self.dataset.clique2id,
                'embedding_type': self.dataset.embedding_type,
                'embedding_format': self.dataset.embedding_format
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            if self.verbose:
                print(f"✓ Saved processed dataset to cache: {cache_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not save to cache {cache_file}: {e}")
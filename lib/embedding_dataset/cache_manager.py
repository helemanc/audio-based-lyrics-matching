"""
Cache management for embeddings datasets.

Handles loading and saving processed dataset structures to disk for fast
initialization on subsequent runs.
"""

import pickle
from pathlib import Path
from typing import Optional, Dict, Any


class CacheManager:
    """
    Manages cache loading and saving for processed datasets.
    
    Cache stores: info dict, splitdict, clique2id mappings
    
    Args:
        dataset: Parent EmbeddingDataset instance
        verbose: Print cache operations
    """
    
    def __init__(self, dataset: Any, verbose: bool = True) -> None:
        self.dataset = dataset
        self.verbose = verbose
    
    def _get_cache_path(self) -> Optional[Path]:
        """
        Get cache file path for processed dataset.
        
        Cache ID includes: embedding_type, embedding_format, debug flag
        
        Returns:
            Path to cache file or None if working_dir not configured
        """
        cache_dir = None
        if hasattr(self.dataset.conf.path, 'cache'):
            cache_dir = Path(self.dataset.conf.path.cache) / self.dataset.dataset_nickname
        elif hasattr(self.dataset.conf.path, 'working_dir'):
            cache_dir = Path(self.dataset.conf.path.working_dir) / 'cache' / self.dataset.dataset_nickname
        else:
            return None
        
        # Build cache ID
        if self.dataset.embedding_type == 'multimodal':
            cache_id = "multimodal"
        else:
            cache_id = f"{self.dataset.embedding_type}_{self.dataset.embedding_format}"
        
        if self.dataset.debug:
            cache_id += "_debug"
        
        cache_file = cache_dir / f'processed_dataset_{cache_id}.pkl'
        return cache_file
    
    def _load_from_cache(self) -> bool:
        """
        Load processed dataset from cache.
        
        Populates: info, splitdict, clique2id
        
        Returns:
            True if cache loaded successfully, False otherwise
        """
        cache_file = self._get_cache_path()
        
        if not cache_file or not cache_file.exists():
            return False
        
        try:
            if self.verbose:
                print(f"Loading from cache: {cache_file}")
            
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            
            self.dataset.info = cached_data['info']
            self.dataset.splitdict = cached_data['splitdict']
            self.dataset.clique2id = cached_data['clique2id']
            self.dataset._loaded_from_cache = True
            
            if self.verbose:
                total = len(self.dataset.info)
                print(f"Loaded {total} versions from cache")
                for split_name, split_data in self.dataset.splitdict.items():
                    cliques = len(split_data)
                    versions = sum(len(v) for v in split_data.values())
                    print(f"  {split_name}: {cliques} cliques, {versions} versions")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading cache: {e}")
                print("Rebuilding dataset...")
            return False
    
    def _save_to_cache(self) -> None:
        """
        Save processed dataset to cache.
        
        Saves: info, splitdict, clique2id, embedding_type, embedding_format
        """
        cache_file = self._get_cache_path()
        
        if not cache_file:
            if self.verbose:
                print("Warning: Cannot save cache - working_dir not configured")
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
                print(f"✓ Saved to cache: {cache_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not save cache: {e}")
"""
Embeddings dataset module - simplified structure.

Public API for loading pre-extracted embeddings for version identification.
"""

from .base_dataset import EmbeddingDataset
from .multimodal_dataset import (
    MultimodalEmbeddingDataset_WEALYCLEWS,
    MultimodalEmbeddingDataset_WHISPERCLEWS
)
from .collate_functions import create_collate_fn, collate_embeddings_fixed_length
from .cache_manager import CacheManager

# Expose helper modules for advanced usage
from . import data_processing
from . import utils

__all__ = [
    # Main classes
    'EmbeddingDataset',
    'MultimodalEmbeddingDataset_WEALYCLEWS',
    'MultimodalEmbeddingDataset_WHISPERCLEWS',
    
    # Collation
    'create_collate_fn',
    'collate_embeddings_fixed_length',
    
    # Cache
    'CacheManager',
    
    # Helper modules
    'data_processing',
    'utils',
]
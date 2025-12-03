"""
Embedding dataset module for version identification.
"""
from .base_dataset import EmbeddingDataset
from .multimodal_dataset import (
    MultimodalEmbeddingDataset_WEALYCLEWS,
    MultimodalEmbeddingDataset_WHISPERCLEWS
)
from .collate_functions import (
    collate_embeddings_fixed_length,
    create_collate_fn,
    load_wealy_with_chunking,
    handle_wealy_test_mode
)
from .utils import create_deterministic_song_id

__all__ = [
    # Dataset classes
    'EmbeddingDataset',
    'MultimodalEmbeddingDataset_WEALYCLEWS',
    'MultimodalEmbeddingDataset_WHISPERCLEWS',
    
    # Collate functions
    'collate_embeddings_fixed_length',
    'create_collate_fn',
    'load_wealy_with_chunking',
    'handle_wealy_test_mode',
    
    # Utilities
    'create_deterministic_song_id',
]
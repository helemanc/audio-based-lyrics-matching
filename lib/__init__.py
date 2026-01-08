"""
Library module for lyrics-version-identification project.

Provides unified access to audio datasets, embedding datasets, and model utilities.
"""

# Core modules
from . import audio_dataset
from . import embedding_dataset
from . import layers
from . import losses
from . import tensor_ops


class dataset:
    """
    Unified dataset namespace for convenience and backward compatibility.
    
    Examples:
        >>> from lib import dataset
        >>> 
        >>> # Audio dataset
        >>> audio_ds = dataset.AudioDataset("shs", "./datasets", "/data", split="train")
        >>> loader = dataset.create_dataloader("shs", "./datasets", "/data", split="train")
        >>> 
        >>> # Embedding dataset
        >>> embed_ds = dataset.EmbeddingDataset(conf, split='train')
        >>> collate_fn = dataset.create_collate_fn(conf)
        >>> 
        >>> # Multimodal dataset
        >>> wealy_ds = dataset.MultimodalEmbeddingDataset_WEALYCLEWS(conf, split='train')
    """
    
    # ========================================================================
    # AUDIO DATASET
    # ========================================================================
    
    # Main classes
    AudioDataset = audio_dataset.AudioDataset
    TranscriptionCache = audio_dataset.TranscriptionCache
    TranscriptionValidator = audio_dataset.TranscriptionValidator
    
    # DataLoader creation
    create_dataloader = audio_dataset.create_dataloader
    collate_fn = audio_dataset.collate_fn
    
    # Helper modules (advanced usage)
    audio_data_processing = audio_dataset.data_processing
    audio_utils = audio_dataset.utils
    
    # ========================================================================
    # EMBEDDINGS DATASET
    # ========================================================================
    
    # Main classes
    EmbeddingDataset = embedding_dataset.EmbeddingDataset
    MultimodalEmbeddingDataset_WEALYCLEWS = embedding_dataset.MultimodalEmbeddingDataset_WEALYCLEWS
    MultimodalEmbeddingDataset_WHISPERCLEWS = embedding_dataset.MultimodalEmbeddingDataset_WHISPERCLEWS
    CacheManager = embedding_dataset.CacheManager
    
    # Collate functions (from collate_functions module)
    create_collate_fn = embedding_dataset.create_collate_fn
    collate_embeddings_fixed_length = embedding_dataset.collate_embeddings_fixed_length
    load_wealy_with_chunking = embedding_dataset.collate_functions.load_wealy_with_chunking
    handle_wealy_test_mode = embedding_dataset.collate_functions.handle_wealy_test_mode
    
    # Helper modules (advanced usage)
    embedding_data_processing = embedding_dataset.data_processing
    embedding_utils = embedding_dataset.utils
    
    # Utility functions
    create_deterministic_song_id = embedding_dataset.utils.create_deterministic_song_id


# Public API
__all__ = [
    'dataset',
    'audio_dataset',
    'embedding_dataset',
    'layers',
    'losses',
    'tensor_ops',
]
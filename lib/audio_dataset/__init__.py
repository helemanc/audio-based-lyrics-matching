"""
Audio dataset module - simplified structure.
"""

from .dataset import AudioDataset
from .cache import TranscriptionCache
from .validator import TranscriptionValidator
from .dataloader import create_dataloader, collate_fn

# Expose helper modules for advanced use
from . import data_processing
from . import utils

__all__ = [
    'AudioDataset',
    'TranscriptionCache',
    'TranscriptionValidator',
    'create_dataloader',
    'collate_fn',
    'data_processing',
    'utils',
]
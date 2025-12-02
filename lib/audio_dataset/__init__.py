"""
Audio dataset module for cover song identification with transcription support.
"""
from .dataset import AudioDataset
from .dataloader import create_dataloader, collate_fn
from .validator import TranscriptionValidator
from .cache import TranscriptionCache

__all__ = [
    'AudioDataset',
    'create_dataloader',
    'collate_fn',
    'TranscriptionValidator',
    'TranscriptionCache',
]
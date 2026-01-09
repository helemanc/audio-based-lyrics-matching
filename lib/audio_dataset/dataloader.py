"""
DataLoader creation and batch collation for audio datasets.

This module provides utilities for creating PyTorch DataLoaders with
custom collation functions optimized for audio waveforms and transcriptions.
Includes robust error handling and support for variable-length audio.

Example:
    >>> from lib.audio_dataset.dataloader import create_dataloader
    >>> dataloader = create_dataloader(
    ...     dataset_name="shs",
    ...     base_path="datasets/shs",
    ...     data_folder="/data",
    ...     split="train",
    ...     batch_size=8
    ... )
"""

import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Any
import signal

from .dataset import AudioDataset


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor, str]], 
    enforce_max_duration: bool = False, 
    max_duration_seconds: float = 300, 
    sample_rate: int = 16000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           List[str], torch.Tensor, List[str]]:
    """
    Efficient collate function for batching audio samples with transcriptions.
    
    Pads variable-length waveforms to the maximum length in the batch and
    creates attention masks. Includes robust error handling for empty batches
    or invalid items.
    
    Args:
        batch: List of tuples, each containing:
            - clique_id (torch.Tensor): Clique identifier
            - version_id (torch.Tensor): Version identifier
            - waveform (torch.Tensor): Audio waveform (1D tensor)
            - transcription (str): Transcription text
            - has_valid_transcription (torch.Tensor): Boolean flag
            - audio_path (str): Path to audio file
        enforce_max_duration: If True, clips audio to maximum duration
        max_duration_seconds: Maximum duration in seconds (default: 300 = 5 minutes)
        sample_rate: Sample rate of audio (default: 16000, Whisper default)
    
    Returns:
        Tuple containing:
            - clique_ids_tensor (torch.Tensor): Stacked clique IDs, shape (batch_size,)
            - version_ids_tensor (torch.Tensor): Stacked version IDs, shape (batch_size,)
            - padded_waveforms (torch.Tensor): Padded waveforms, shape (batch_size, max_length)
            - waveform_lengths (torch.Tensor): Original lengths, shape (batch_size,)
            - attention_mask (torch.Tensor): Boolean mask, shape (batch_size, max_length)
            - transcriptions (List[str]): List of transcription texts
            - valid_transcription_flags (torch.Tensor): Boolean flags, shape (batch_size,)
            - audio_paths (List[str]): List of audio file paths
    
    Raises:
        TypeError: If batch items have incorrect format
    
    Example:
        >>> batch = [(clique_id, version_id, waveform, text, valid, path), ...]
        >>> collated = collate_fn(batch, enforce_max_duration=True)
        >>> clique_ids, version_ids, waveforms, lengths, mask, texts, flags, paths = collated
    """
    # Handle empty batch or None batch
    if batch is None or len(batch) == 0:
        print("Warning: Empty batch received in collate_fn")
        # Return empty tensors
        return (torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.bool),
                [],
                torch.tensor([], dtype=torch.bool),
                [])

    # Unzip the batch to get separate lists - with error handling
    try:
        # Check if any item in batch is None or malformed
        for i, item in enumerate(batch):
            if item is None or not isinstance(item, tuple) or len(item) != 6:
                print(f"Warning: Invalid batch item at index {i}: {item}")
                # Remove invalid items
                batch = [b for b in batch if b is not None and isinstance(b, tuple) and len(b) == 6]
                break

        # If batch is now empty, return empty tensors
        if not batch:
            print("Warning: All items in batch were invalid")
            return (torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.float32),
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.bool),
                    [],
                    torch.tensor([], dtype=torch.bool),
                    [])

        # Safely unzip the batch
        clique_ids, version_ids, waveforms, transcriptions, valid_transcription_flags, audio_paths = zip(*batch)

    except TypeError as e:
        print(f"Error unpacking batch: {e}")
        print(f"Batch type: {type(batch)}, length: {len(batch)}")
        if len(batch) > 0:
            print(f"First item type: {type(batch[0])}")
        # Return empty tensors on error
        return (torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.bool),
                [],
                torch.tensor([], dtype=torch.bool),
                [])

    try:
        # Stack the ID tensors
        clique_ids_tensor = torch.stack(clique_ids)
        version_ids_tensor = torch.stack(version_ids)
        valid_transcription_flags_tensor = torch.stack(valid_transcription_flags)
        audio_paths_list = [audio_path for audio_path in audio_paths]

        # Get waveform lengths and check for valid waveforms
        valid_waveforms = []
        valid_indices = []
        for i, waveform in enumerate(waveforms):
            if isinstance(waveform, torch.Tensor) and waveform.numel() > 0:
                valid_waveforms.append(waveform)
                valid_indices.append(i)

        # If no valid waveforms, return empty tensors
        if not valid_waveforms:
            print("Warning: No valid waveforms in batch")
            return (clique_ids_tensor, version_ids_tensor,
                    torch.zeros((len(batch), 1), dtype=torch.float32),
                    torch.ones(len(batch), dtype=torch.long),
                    torch.zeros((len(batch), 1), dtype=torch.bool),
                    list(transcriptions),
                    valid_transcription_flags_tensor,
                    audio_paths_list)

        # Get waveform lengths
        waveform_lengths = torch.tensor([w.shape[0] for w in valid_waveforms], dtype=torch.long)

        # Determine max length based on enforce_max_duration setting
        if enforce_max_duration:
            max_length = int(max_duration_seconds * sample_rate)  # Force exactly 5 minutes
            waveform_lengths = torch.clamp(waveform_lengths, max=max_length)
        else:
            max_length = waveform_lengths.max().item()  # Otherwise use the longest in batch

        # Create padded waveforms tensor directly
        padded_waveforms = torch.zeros(len(valid_waveforms), max_length, dtype=torch.float32)
        for i, waveform in enumerate(valid_waveforms):
            # Ensure waveform is float32
            if waveform.dtype != torch.float32:
                waveform = waveform.to(torch.float32)
            # Handle case where waveform might be empty or have wrong shape
            if waveform.numel() > 0 and waveform.dim() == 1:
                # Clip waveform to max_length if needed
                actual_length = min(waveform.shape[0], max_length)
                padded_waveforms[i, :actual_length] = waveform[:actual_length]

        # Create attention mask (True for valid positions, False for padding)
        attention_mask = torch.arange(max_length).unsqueeze(0) < waveform_lengths.unsqueeze(1)

        # Return the batch components including transcriptions
        return (clique_ids_tensor, version_ids_tensor, padded_waveforms, 
                waveform_lengths, attention_mask, list(transcriptions), 
                valid_transcription_flags_tensor, audio_paths_list)

    except Exception as e:
        print(f"Error in collate_fn: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal valid batch on error
        return (clique_ids_tensor, version_ids_tensor,
                torch.zeros((len(batch), 1), dtype=torch.float32),
                torch.ones(len(batch), dtype=torch.long),
                torch.zeros((len(batch), 1), dtype=torch.bool),
                list(transcriptions),
                valid_transcription_flags_tensor,
                audio_paths_list)


def create_dataloader(
    dataset_name: str, 
    base_path: str, 
    data_folder: str, 
    split: str = 'train',
    whisper_set: str = "turbo_nothing_whisper_42", 
    batch_size: int = 8,
    evaluation_mode: bool = False, 
    debug_mode: bool = False,
    use_whisper_loader: bool = True,
    use_transcriptions: bool = True,  
    num_workers: int = 8, 
    pin_memory: bool = False,
    debug_num_cliques: Optional[int] = None, 
    enforce_max_duration: bool = False
) -> DataLoader:
    """
    Create a DataLoader with optimized settings for audio dataset.
    
    Configures DataLoader with custom collation, signal handling, and optional
    debugging features. Supports limiting dataset to specific number of cliques
    for faster debugging iterations.
    
    Args:
        dataset_name: Name of the dataset. Must be one of:
            - 'lyric-covers': Lyric Covers dataset
            - 'shs': SHS100K dataset
            - 'discogs-vi': Discogs-VI dataset
        base_path: Base path to the dataset metadata directory
        data_folder: Path to the data folder containing audio files and transcriptions
        split: Data split to use. One of 'train', 'val', or 'test'
        whisper_set: Name of the whisper model set for transcriptions
            (e.g., "turbo_nothing_whisper_42", "whisper-turbo")
        batch_size: Batch size for the dataloader
        evaluation_mode: If True, runs in evaluation mode (no shuffling, no dropout)
        debug_mode: If True, filters out items without valid transcriptions
        use_whisper_loader: If True, uses Whisper's audio loading function
        use_transcriptions: If True, loads pre-existing transcriptions from disk.
            Set to False for Whisper extraction (generates its own) or audio-only
            methods (WEALY, CLEWS). Set to True for SBERT or text baselines.
        num_workers: Number of worker processes for data loading
        pin_memory: If True, pins memory for faster GPU transfer
        debug_num_cliques: If set, limits dataset to samples from this many cliques
            for faster debugging. Useful for testing with small data subsets.
        enforce_max_duration: If True, clips audio to maximum 5 minutes
    
    Returns:
        Configured PyTorch DataLoader for the audio dataset
    
    Raises:
        ValueError: If dataset_name is not recognized
        FileNotFoundError: If base_path or data_folder does not exist
    
    DataLoader Behavior:
        - Training split: Shuffling enabled, incomplete batches dropped
        - Validation/Test splits: No shuffling, all samples included
        - Debug mode: Limits to first N cliques, adjusts batch size accordingly
    
    Performance Tips:
        - For Whisper extraction: Set use_transcriptions=False (saves 5-10 min)
        - For SBERT extraction: Set use_transcriptions=True (needs transcriptions)
        - For audio-only methods: Set use_transcriptions=False
    
    Example:
        >>> # Standard training dataloader
        >>> train_loader = create_dataloader(
        ...     dataset_name="shs",
        ...     base_path="datasets/shs",
        ...     data_folder="/data",
        ...     split="train",
        ...     batch_size=16
        ... )
        
        >>> # Whisper extraction (fast initialization)
        >>> whisper_loader = create_dataloader(
        ...     dataset_name="shs",
        ...     base_path="datasets/shs",
        ...     data_folder="/data",
        ...     split="train",
        ...     batch_size=8,
        ...     use_transcriptions=False  # Skip loading transcriptions
        ... )
        
        >>> # SBERT extraction (needs transcriptions)
        >>> sbert_loader = create_dataloader(
        ...     dataset_name="shs",
        ...     base_path="datasets/shs",
        ...     data_folder="/data",
        ...     split="train",
        ...     batch_size=16,
        ...     use_transcriptions=True  # Load transcriptions
        ... )
        
        >>> # Debug with limited cliques
        >>> debug_loader = create_dataloader(
        ...     dataset_name="shs",
        ...     base_path="datasets/shs",
        ...     data_folder="/data",
        ...     split="train",
        ...     batch_size=8,
        ...     debug_num_cliques=5  # Only 5 cliques
        ... )
    """

    # Define the custom collate function with closure over enforce_max_duration
    def custom_collate_fn(batch: List[Any]) -> Tuple:
        """Wrapper around collate_fn with preset max duration settings."""
        return collate_fn(
            batch,
            enforce_max_duration=enforce_max_duration,
            max_duration_seconds=300,  # 5 minutes
            sample_rate=16000  # Whisper default
        )

    # Adjust batch size based on debug_num_cliques
    effective_batch_size = batch_size
    if debug_num_cliques is not None and debug_num_cliques > 0:
        # For debugging with limited cliques, use smaller batch size
        # Ensure batch size doesn't exceed number of cliques * samples per clique
        effective_batch_size = min(batch_size, debug_num_cliques)
        print(
            f"Adjusted batch size from {batch_size} to {effective_batch_size} "
            f"for debugging with {debug_num_cliques} cliques"
        )

    batch_size = effective_batch_size

    # Configure signal handling for graceful termination
    def sig_handler(signum: int, frame: Any) -> None:
        """Handle interrupt signals gracefully."""
        print(f"Signal {signum} received. Exiting gracefully.")
        exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Create the dataset instance with use_transcriptions parameter
    dataset = AudioDataset(
        dataset_name,
        base_path,
        data_folder,
        split,
        whisper_set,
        evaluation_mode,
        debug_mode,
        use_whisper_loader,
        use_transcriptions 
    )

    # If debug_num_cliques is set, filter to keep only samples from limited cliques
    if debug_num_cliques is not None and debug_num_cliques > 0 and not dataset.df.empty:
        print(f"DEBUG MODE: Limiting dataset to samples from {debug_num_cliques} cliques")

        # Get unique clique IDs
        unique_cliques = dataset.df['clique_id'].unique()

        # Select a subset of cliques
        selected_cliques = unique_cliques[:min(debug_num_cliques, len(unique_cliques))]

        # Count versions in the selected cliques
        versions_count = dataset.df[
            dataset.df['clique_id'].isin(selected_cliques)
        ].groupby('clique_id')['version_id'].nunique()

        # Print statistics about the selected cliques
        print(f"Selected {len(selected_cliques)} cliques with the following version counts:")
        for clique_id, count in versions_count.items():
            print(f"  Clique {clique_id}: {count} versions")

        # Create a subset dataset that maintains the original dataset's structure
        # but only includes the selected indices
        filtered_indices = dataset.df[
            dataset.df['clique_id'].isin(selected_cliques)
        ].index.tolist()
        subset_dataset = torch.utils.data.Subset(dataset, filtered_indices)

        print(
            f"Reduced dataset to {len(filtered_indices)} samples "
            f"from {len(selected_cliques)} cliques"
        )

        # Configure DataLoader settings
        dataloader_config = {
            'batch_size': batch_size,
            'shuffle': (split == 'train'),  # Shuffle only for training
            'collate_fn': custom_collate_fn,
            'drop_last': (split == 'train'),  # Drop incomplete batches only in training
        }

        print(
            f"Creating DataLoader with {len(subset_dataset)} samples, "
            f"batch_size={batch_size}, num_workers={num_workers}"
        )

        # Return a dataloader with the subset dataset
        return DataLoader(subset_dataset, **dataloader_config)

    # If not using debug_num_cliques, proceed with the full dataset
    dataloader_config = {
        'batch_size': batch_size,
        'shuffle': (split == 'train'),  # Shuffle only for training
        'collate_fn': custom_collate_fn,
        'drop_last': (split == 'train'),  # Drop incomplete batches only in training
    }

    print(
        f"Creating DataLoader with {len(dataset)} samples, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    # Create and return the dataloader with configured settings
    return DataLoader(dataset, **dataloader_config)
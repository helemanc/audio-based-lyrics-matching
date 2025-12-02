"""
DataLoader creation and batch collation for audio datasets.
"""
import torch
from torch.utils.data import DataLoader
import signal

from .dataset import AudioDataset

def collate_fn(batch, enforce_max_duration=False, max_duration_seconds=300, sample_rate=16000):
    """
    Efficient collate function for batching samples with transcriptions.
    Includes robust error handling for empty batches or None values.

    Args:
        batch: List of tuples (clique_id, version_id, waveform, transcription, has_valid_transcription)
        enforce_max_duration (bool): If True, clips audio to maximum duration. Default is False.
        max_duration_seconds (float): Maximum duration in seconds. Default is 300 (5 minutes).
        sample_rate (int): Sample rate of audio. Default is 16000 (whisper default).

    Returns:
        tuple: Processed batch tensors and transcriptions
    """
    # Handle empty batch or None batch
    if batch is None or len(batch) == 0:
        print("Warning: Empty batch received in collate_fn")
        # Return empty tensors
        return (torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.bool))

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
                    torch.tensor([], dtype=torch.bool))

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
                torch.tensor([], dtype=torch.bool))

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
                    transcriptions)

        # Get waveform lengths
        waveform_lengths = torch.tensor([w.shape[0] for w in valid_waveforms], dtype=torch.long)

        # Do this:
        if enforce_max_duration:
            max_length = int(max_duration_seconds * sample_rate)  # Force exactly 5 minutes
            max_samples = int(max_duration_seconds * sample_rate)

            waveform_lengths = torch.clamp(waveform_lengths, max=max_samples)

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

        # Create attention mask
        attention_mask = torch.arange(max_length).unsqueeze(0) < waveform_lengths.unsqueeze(1)

        # Return the batch components including transcriptions
        return clique_ids_tensor, version_ids_tensor, padded_waveforms, waveform_lengths, attention_mask, transcriptions, valid_transcription_flags_tensor, audio_paths_list

    except Exception as e:
        print(f"Error in collate_fn: {e}")
        import traceback
        traceback.print_exc()
        # Return a minimal valid batch on error
        return (clique_ids, version_ids,
                torch.zeros((len(batch), 1), dtype=torch.float32),
                torch.ones(len(batch), dtype=torch.long),
                torch.zeros((len(batch), 1), dtype=torch.bool),
                transcriptions,
                torch.zeros((len(batch), 1), dtype=torch.bool),
                torch.zeros((len(batch), 1), dtype=torch.bool))

def create_dataloader(dataset_name, base_path, data_folder, split='train',
                      whisper_set="turbo_nothing_whisper_42", batch_size=8,
                      evaluation_mode=False, debug_mode=False,
                      use_whisper_loader=True, num_workers=8, pin_memory=False,
                      debug_num_cliques=None, enforce_max_duration=False):
    """
    Create a dataloader with optimized settings for audio dataset.

    Args:
        dataset_name (str): Name of the dataset ('lyric-covers', 'shs', or 'discogs-vi')
        base_path (str): Base path to the dataset metadata
        data_folder (str): Path to the data folder containing audio files and transcriptions
        split (str): Data split to use ('train', 'val', or 'test')
        whisper_set (str): Name of the whisper set to use for transcriptions
        batch_size (int): Batch size for the dataloader
        evaluation_mode (bool): Whether to run in evaluation mode
        debug_mode (bool): Whether to run in debug mode (filtering out items without transcriptions)
        use_whisper_loader (bool): Whether to use whisper's audio loader
        num_workers (int): Number of worker processes for the dataloader
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        debug_num_cliques (int, optional): If set, limit dataset to samples from this many cliques
        enforce_max_duration (bool): If True, clips audio to maximum 5 minutes. Default is False.

    Returns:
        torch.utils.data.DataLoader: Configured dataloader for the dataset
    """

    # Define the custom collate function with closure over enforce_max_duration
    def custom_collate_fn(batch):
        return collate_fn(batch,
                         enforce_max_duration=enforce_max_duration,
                         max_duration_seconds=300,  # 5 minutes
                         sample_rate=16000)  # whisper default

    # Adjust batch size based on debug_num_cliques
    effective_batch_size = batch_size
    if debug_num_cliques is not None and debug_num_cliques > 0:
        # For debugging with limited cliques, use smaller batch size
        # Ensure batch size doesn't exceed number of cliques * samples per clique
        effective_batch_size = min(batch_size, debug_num_cliques)
        print(f"Adjusted batch size from {batch_size} to {effective_batch_size} for debugging with {debug_num_cliques} cliques")

    batch_size = effective_batch_size

    # Configure signal handling for graceful termination
    def sig_handler(signum, frame):
        print(f"Signal {signum} received. Exiting gracefully.")
        exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Create the dataset instance
    dataset = AudioDataset(
        dataset_name,
        base_path,
        data_folder,
        split,
        whisper_set,
        evaluation_mode,
        debug_mode,
        use_whisper_loader
    )

    # If debug_num_cliques is set, filter to keep only samples from a limited number of cliques
    if debug_num_cliques is not None and debug_num_cliques > 0 and not dataset.df.empty:
        print(f"DEBUG MODE: Limiting dataset to samples from {debug_num_cliques} cliques")

        # Get unique clique IDs
        unique_cliques = dataset.df['clique_id'].unique()

        # Select a subset of cliques
        selected_cliques = unique_cliques[:min(debug_num_cliques, len(unique_cliques))]

        # Count versions in the selected cliques
        versions_count = dataset.df[dataset.df['clique_id'].isin(selected_cliques)].groupby('clique_id')['version_id'].nunique()

        # Print statistics about the selected cliques
        print(f"Selected {len(selected_cliques)} cliques with the following version counts:")
        for clique_id, count in versions_count.items():
            print(f"  Clique {clique_id}: {count} versions")

        # Create a subset dataset that maintains the original dataset's structure
        # but only includes the selected indices
        filtered_indices = dataset.df[dataset.df['clique_id'].isin(selected_cliques)].index.tolist()
        subset_dataset = torch.utils.data.Subset(dataset, filtered_indices)

        print(f"Reduced dataset to {len(filtered_indices)} samples from {len(selected_cliques)} cliques")

        # Configure DataLoader settings
        dataloader_config = {
            'batch_size': batch_size,
            'shuffle': (split == 'train'),  # Shuffle only for training
            'collate_fn': custom_collate_fn,
            'drop_last': (split == 'train'),  # Drop incomplete batches only in training
        }

        print(f"Creating DataLoader with {len(subset_dataset)} samples, batch_size={batch_size}, num_workers={num_workers}")

        # Return a dataloader with the subset dataset
        return DataLoader(subset_dataset, **dataloader_config)

    # If not using debug_num_cliques, proceed with the full dataset

    dataloader_config = {
        'batch_size': batch_size,
        'shuffle': (split == 'train'),  # Shuffle only for training
        'collate_fn': custom_collate_fn,
        'drop_last': (split == 'train'),  # Drop incomplete batches only in training
    }

    print(f"Creating DataLoader with {len(dataset)} samples, batch_size={batch_size}, num_workers={num_workers}")

    # Create and return the dataloader with configured settings
    return DataLoader(dataset, **dataloader_config)

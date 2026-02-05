"""
Utility functions for extracting and saving latent representations.

Provides helpers for organizing file paths, extracting dataset-specific metadata,
and saving various types of embeddings (encoder, hidden states, SBERT) across
different dataset formats (SHS, Lyric-Covers, Discogs-VI).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def define_save_folders(dataset_name: str, data_folder: str) -> Tuple[Path, Path]:
    """
    Define folder names for saving transcriptions and hidden states.

    Args:
        dataset_name: Name of dataset ('shs', 'lyric-covers', 'discogs-vi')
        data_folder: Base data directory path

    Returns:
        Tuple of (transcription_path, hidden_states_path)

    Raises:
        ValueError: If dataset_name is not supported

    Example:
        >>> transcription_path, hidden_states_path = define_save_folders('shs', '/data')
        >>> # Returns: (/data/SHS100K-transcriptions, /data/SHS100K-hidden-states)
    """
    # Define the folder names based on the dataset name
    if dataset_name == "shs":
        dataset_transcription_folder = "SHS100K-transcriptions"
        dataset_encoder_embeddings_folder = "SHS100K-encoder-embeddings"
        dataset_hidden_states_folder = "SHS100K-hidden-states"
    elif dataset_name == "lyric-covers":
        dataset_transcription_folder = "LyricCovers-transcriptions"
        dataset_encoder_embeddings_folder = "LyricCovers-encoder-embeddings"
        dataset_hidden_states_folder = "LyricCovers-hidden-states"
    elif dataset_name == "discogs-vi":
        dataset_transcription_folder = "DiscogsVI-transcriptions"
        dataset_encoder_embeddings_folder = "DiscogsVI-encoder-embeddings"
        dataset_hidden_states_folder = "DiscogsVI-hidden-states"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create the full paths for the transcription and hidden states folders
    transcription_path = os.path.join(data_folder, dataset_transcription_folder)
    hidden_states_path = os.path.join(data_folder, dataset_hidden_states_folder)

    # Create the directories if they do not exist
    transcription_path = Path(transcription_path)
    hidden_states_path = Path(hidden_states_path)
    transcription_path.mkdir(parents=True, exist_ok=True)
    hidden_states_path.mkdir(parents=True, exist_ok=True)

    return transcription_path, hidden_states_path


def get_save_path_for_dataset(
    hidden_states_folder: str,
    dataset_name: str,
    clique_id: str,
    version_id: str,
    save_components: Tuple[str, ...],
) -> Path:
    """
    Build the save path based on dataset structure.

    Different datasets organize files differently:
    - SHS: hidden_states/{clique_folder}/{version_folder}/
    - Lyric-Covers: hidden_states/{version_id}/
    - Discogs-VI: hidden_states/{dir}/{filename}/

    Args:
        hidden_states_folder: Base folder for hidden states
        dataset_name: Name of dataset ('shs', 'lyric-covers', 'discogs-vi')
        clique_id: Clique identifier
        version_id: Version identifier
        save_components: Tuple of path components for organizing files

    Returns:
        Full path for saving embeddings

    Raises:
        ValueError: If dataset_name is not supported

    Example:
        >>> path = get_save_path_for_dataset(
        ...     '/data/hidden-states', 'shs', '0-', '0-1', ('0-', '0-1')
        ... )
        >>> # Returns: /data/hidden-states/0-/0-1
    """
    base_path = Path(hidden_states_folder)

    if dataset_name == "shs":
        # SHS: hidden_states/{set_folder}/{set_id-ver_id}/
        clique_folder, version_folder = save_components
        return base_path / clique_folder / version_folder

    elif dataset_name == "lyric-covers":
        # Lyric Covers: hidden_states/{id}/
        version_folder = save_components[0]
        return base_path / version_folder

    elif dataset_name == "discogs-vi":
        # Discogs-VI: hidden_states/{base_filename_path}/
        # Handle potential subdirectories
        return base_path / Path(*save_components)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def get_tensor_filename(embedding_type: str, embedding_format: str) -> Optional[str]:
    """
    Get tensor filename based on embedding type and format.

    Args:
        embedding_type: Type of embedding ('encoder', 'hidden_states',
                       'last_hidden_states', 'last_hidden_states_en', 'sbert')
        embedding_format: Format of embedding ('concat', 'all')

    Returns:
        Filename string or None if combination is invalid

    Example:
        >>> get_tensor_filename('last_hidden_states', 'concat')
        'hs_last_seq.pt'
        >>> get_tensor_filename('sbert', 'all')
        'hs_sbert.pt'
    """
    if embedding_type == "encoder":
        if embedding_format == "concat":
            return "x_concat.pt"
        elif embedding_format == "all":
            return "x_all.pt"
    elif embedding_type == "hidden_states":
        if embedding_format == "all":
            return "hs_all.pt"
    elif embedding_type == "last_hidden_states":
        if embedding_format == "concat":
            return "hs_last_seq.pt"
        elif embedding_format == "all":
            return "hs_last_all.pt"
    elif embedding_type == "last_hidden_states_en":
        if embedding_format == "concat":
            return "hs_last_seq_en.pt"
        elif embedding_format == "all":
            return "hs_last_all_en.pt"
    elif embedding_type == "sbert":
        return "hs_sbert.pt"

    return None


def extract_path_info_for_dataset(
    audio_path: str, dataset_name: str
) -> Tuple[str, str, Tuple[str, ...]]:
    """
    Extract clique_id, version_id, and save path components based on dataset type.

    Different datasets have different file organization:
    - SHS: /path/to/SHS100K/audio/{clique_folder}/{version_id}.mp3
    - Lyric-Covers: /path/to/LyricCovers/audio/{version_id}/{version_id}_audio.mp3
    - Discogs-VI: /path/to/DiscogsVI/audio/{dir}/{filename}.mp3

    Args:
        audio_path: Path to audio file
        dataset_name: Name of dataset ('shs', 'lyric-covers', 'discogs-vi')

    Returns:
        Tuple of (clique_id, version_id, save_base_path_components)

    Raises:
        ValueError: If dataset_name is not supported

    Example:
        >>> extract_path_info_for_dataset('/data/SHS100K/audio/0-/0-1.mp3', 'shs')
        ('0-', '0-1', ('0-', '0-1'))
    """
    audio_path = Path(audio_path)

    if dataset_name == "shs":
        # SHS structure: /path/to/SHS100K/audio/{set_folder}/{set_id-ver_id}.mp3
        # Example: /data/SHS100K/audio/0-/0-1.mp3
        clique_id = audio_path.parent.name  # "0-"
        version_id = audio_path.stem  # "0-1"
        # For saving: use the clique_id (set_folder) and version_id
        return clique_id, version_id, (clique_id, version_id)

    elif dataset_name == "lyric-covers":
        # Lyric Covers structure: /path/to/LyricCovers/audio/{id}/{id}_audio.mp3
        # Example: /data/LyricCovers/audio/12345/12345_audio.mp3
        version_id = audio_path.parent.name  # "12345"
        clique_id = version_id  # For lyric covers, clique comes from dataframe
        # For saving: use just the version_id as folder
        return clique_id, version_id, (version_id,)

    elif dataset_name == "discogs-vi":
        # Discogs-VI structure: /path/to/DiscogsVI/audio/something/something_1.mp3
        # We need the last 2 components: something/something_1.mp3
        version_id = audio_path.stem  # "something_1"
        clique_id = version_id  # For discogs-vi, clique comes from dataframe

        # Get the directory name and filename as the last 2 components
        dir_name = audio_path.parent.name  # "something"
        filename = audio_path.stem  # "something_1"
        save_components = (dir_name, filename)

        return clique_id, version_id, save_components

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def save_sbert_embeddings(
    sbert_embedding: torch.Tensor,
    save_base_path: Path,
    decoding_config_name: Optional[str] = None,
) -> None:
    """
    Save SBERT embeddings to disk.

    Args:
        sbert_embedding: SBERT embedding tensor to save
        save_base_path: Base path for saving
        decoding_config_name: Optional config name (for compatibility, currently unused)

    Example:
        >>> embedding = torch.randn(384)  # SBERT embedding dimension
        >>> save_sbert_embeddings(embedding, Path('/data/embeddings/track_001'))
        # Saves to: /data/embeddings/track_001/hs_sbert.pt
    """
    save_base_path.mkdir(parents=True, exist_ok=True)
    sbert_path = save_base_path / "hs_sbert.pt"
    torch.save(sbert_embedding, sbert_path)


def save_transcription_and_latents(
    dataset_name: str,
    result: Dict[str, Any],
    frames_audio_features: List[torch.Tensor],
    lyric_feature_seq: torch.Tensor,
    transcription_folder: str,
    hidden_states_folder: str,
    save_base_path: Path,
    save_components: Tuple[str, ...],
    decoding_config_name: str,
    save_transcription: bool = False,
    save_encoder_embeddings: bool = False,
    save_encoder_embeddings_seq: bool = False,
    save_hidden_states: bool = False,
    save_last_hidden_states: bool = False,
    save_last_hidden_states_seq: bool = False,
    language: Optional[str] = None,
) -> None:
    """
    Save various model outputs based on flags. Skips existing files.

    Supports saving:
    - Transcription text and detected language
    - Encoder embeddings (individual and concatenated)
    - Hidden states (all layers)
    - Last hidden states (final layer)

    Args:
        dataset_name: Name of dataset for path organization
        result: Dictionary containing model outputs with keys:
                - 'text': Transcription text
                - 'language': Detected language
                - 'frames_hidden_states': Hidden states per frame
                - 'frames_last_hidden_states': Last hidden states per frame
        frames_audio_features: List of encoder embeddings per frame
        lyric_feature_seq: Concatenated sequence of last hidden states
        transcription_folder: Base folder for transcriptions
        hidden_states_folder: Base folder for hidden states
        save_base_path: Base path for current file
        save_components: Path components for dataset-specific organization
        decoding_config_name: Name of decoding configuration
        save_transcription: Save transcription and language
        save_encoder_embeddings: Save all encoder embeddings
        save_encoder_embeddings_seq: Save concatenated encoder embeddings
        save_hidden_states: Save all hidden states
        save_last_hidden_states: Save all last hidden states
        save_last_hidden_states_seq: Save concatenated last hidden states
        language: Optional language suffix for filenames

    Example:
        >>> save_transcription_and_latents(
        ...     dataset_name='shs',
        ...     result={'text': 'Hello world', 'language': 'en', ...},
        ...     frames_audio_features=[...],
        ...     lyric_feature_seq=torch.randn(100, 1280),
        ...     transcription_folder='/data/transcriptions',
        ...     hidden_states_folder='/data/hidden-states',
        ...     save_base_path=Path('/data/hidden-states/0-/0-1'),
        ...     save_components=('0-', '0-1'),
        ...     decoding_config_name='large-v2',
        ...     save_last_hidden_states_seq=True
        ... )
    """
    save_base_path.mkdir(parents=True, exist_ok=True)

    if save_transcription:
        if dataset_name == "shs":
            transcription_save_path = (
                Path(transcription_folder)
                / "transcriptions"
                / save_components[0]
                / save_components[1]
            )
            detected_language_save_path = (
                Path(transcription_folder)
                / "detected_language"
                / save_components[0]
                / save_components[1]
            )
        elif dataset_name == "lyric-covers":
            transcription_save_path = (
                Path(transcription_folder) / "transcriptions" / save_components[0]
            )
            detected_language_save_path = (
                Path(transcription_folder) / "detected_language" / save_components[0]
            )
        elif dataset_name == "discogs-vi":
            transcription_save_path = (
                Path(transcription_folder)
                / "transcriptions"
                / save_components[0]
                / save_components[1]
            )
            detected_language_save_path = (
                Path(transcription_folder)
                / "detected_language"
                / save_components[0]
                / save_components[1]
            )

        transcription_save_path.mkdir(parents=True, exist_ok=True)
        txt_path = transcription_save_path / f"{decoding_config_name}.txt"
        detected_language_save_path.mkdir(parents=True, exist_ok=True)
        lang_path = detected_language_save_path / "detected_language.txt"

        if not txt_path.exists():
            with open(txt_path, "w") as f:
                f.write(result["text"])

        if not lang_path.exists():
            with open(lang_path, "w") as f:
                f.write(result["language"])

    if save_encoder_embeddings:
        # Save as a single dictionary (faster I/O)
        if language is None:
            dict_path = save_base_path / "x_all.pt"
        else:
            dict_path = save_base_path / f"x_all_{language}.pt"
        if not dict_path.exists():
            embeddings_dict = {
                f"x_{i}": emb.half() for i, emb in enumerate(frames_audio_features)
            }
            torch.save(embeddings_dict, dict_path)

    if save_encoder_embeddings_seq:
        if language is None:
            concat_path = save_base_path / "x_concat.pt"
        else:
            concat_path = save_base_path / f"x_concat_{language}.pt"
        if not concat_path.exists():
            concat_embedding = torch.cat(frames_audio_features, dim=0)
            torch.save(concat_embedding.half(), concat_path)

    if save_hidden_states:
        # Save all hidden states in one file
        if language is None:
            hs_path = save_base_path / "hs_all.pt"
        else:
            hs_path = save_base_path / f"hs_all_{language}.pt"
        if not hs_path.exists():
            hs_dict = {}
            for chunk_idx, chunk in enumerate(result["frames_hidden_states"]):
                for step_idx, step_hidden in enumerate(chunk):
                    hs = torch.stack(step_hidden, dim=0)
                    hs_dict[f"hs_{chunk_idx}_{step_idx}"] = hs.half()
            torch.save(hs_dict, hs_path)

    if save_last_hidden_states:
        # Save all in one file
        if language is None:
            last_hs_path = save_base_path / "hs_last_all.pt"
        else:
            last_hs_path = save_base_path / f"hs_last_all_{language}.pt"
        if not last_hs_path.exists():
            last_hs_dict = {
                f"hs_last_{i}_{j}": step_hidden.half()
                for i, chunk in enumerate(result["frames_last_hidden_states"])
                for j, step_hidden in enumerate(chunk)
            }
            torch.save(last_hs_dict, last_hs_path)

    if save_last_hidden_states_seq:
        if language is None:
            seq_path = save_base_path / "hs_last_seq.pt"
        else:
            seq_path = save_base_path / f"hs_last_seq_{language}.pt"
        if not seq_path.exists():
            torch.save(lyric_feature_seq.half(), seq_path)

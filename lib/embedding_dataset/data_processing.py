"""
Data processing for embeddings datasets.

Consolidates metadata loading, path construction, filtering, and embedding verification
for SHS100K, Lyric Covers, and Discogs-VI datasets.
"""

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict, Optional
from omegaconf import DictConfig


# ============================================================================
# DISTRIBUTED PRINTING HELPER
# ============================================================================

def is_main_process() -> bool:
    """Check if this is the main process (rank 0) in distributed training."""
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    if 'RANK' in os.environ:
        return int(os.environ['RANK']) == 0
    return True


def dist_print(message: str, verbose: bool = True, end: str = '\n') -> None:
    """Print only from main process."""
    if verbose and is_main_process():
        print(message, end=end)


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_shs_metadata(conf: DictConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Load SHS100K metadata from CSV and split files.
    
    Args:
        conf: Configuration object with paths
        verbose: Print loading progress (only from rank 0)
    
    Returns:
        DataFrame with columns: set_id, ver_id, split, clique_id, version_id
    """
    dist_print("Building metadata from SHS100K CSV files...", verbose)
    
    shs_df = pd.read_csv(conf.path.shs_data)
    
    split_files = {
        "train": os.path.join(conf.path.shs_splits, "SHS100K-TRAIN"),
        "val": os.path.join(conf.path.shs_splits, "SHS100K-VAL"),
        "test": os.path.join(conf.path.shs_splits, "SHS100K-TEST")
    }
    
    split_dfs = []
    for split_name, split_file in split_files.items():
        with open(split_file, 'r') as f:
            split_data = []
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    split_data.append({
                        'set_id': int(parts[0]),
                        'ver_id': int(parts[1]),
                        'split': split_name
                    })
            if split_data:
                split_dfs.append(pd.DataFrame(split_data))
    
    if split_dfs:
        all_splits = pd.concat(split_dfs, ignore_index=True)
        df = shs_df.merge(all_splits, on=['set_id', 'ver_id'], how='inner')
    else:
        df = shs_df.copy()
        df['split'] = 'train'
    
    df["clique_id"] = df["set_id"]
    df["version_id"] = df["ver_id"]
    
    return df


def load_lyric_covers_metadata(conf: DictConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Load Lyric Covers metadata from split CSV files.
    
    Args:
        conf: Configuration object with paths
        verbose: Print loading progress (only from rank 0)
    
    Returns:
        DataFrame with columns: id, label, split, clique_id, version_id
    """
    dist_print("Building metadata from Lyric Covers CSV files...", verbose)
    
    split_files = {
        "train": "train_no_dup.csv",
        "val": "val_no_dup.csv",
        "test": "test_no_dup.csv"
    }
    
    split_dfs = []
    for split_name, split_file in split_files.items():
        split_path = os.path.join(conf.path.lyric_covers_data, split_file)
        split_df = pd.read_csv(split_path)
        split_df['split'] = split_name
        split_dfs.append(split_df)
    
    df = pd.concat(split_dfs, ignore_index=True)
    df["clique_id"] = df["label"]
    df["version_id"] = df["id"]
    
    return df


def load_discogs_vi_metadata(conf: DictConfig, verbose: bool = True) -> pd.DataFrame:
    """
    Load Discogs-VI metadata from CSV.
    
    Args:
        conf: Configuration object with paths
        verbose: Print loading progress (only from rank 0)
    
    Returns:
        DataFrame with columns: split, clique_id, version_id, youtube_id, base_filename
    """
    dist_print("Building metadata from Discogs-VI CSV files...", verbose)
    
    csv_path = os.path.join(conf.path.discogs_vi_data, "id-to-file-mapping.csv")
    df = pd.read_csv(csv_path, names=['split', 'clique_id', 'version_id', 
                                       'youtube_id', 'base_filename'])
    
    df["clique_id"] = df["clique_id"].astype(str)
    df["version_id"] = df["version_id"].astype(str)
    
    return df


def load_metadata(dataset_name: str, conf: DictConfig, 
                 verbose: bool = True) -> pd.DataFrame:
    """
    Load metadata for specified dataset.
    
    Args:
        dataset_name: 'shs', 'lyric-covers', or 'discogs-vi'
        conf: Configuration object
        verbose: Print loading progress (only from rank 0)
    
    Returns:
        DataFrame with standardized columns including clique_id, version_id, split
    
    Raises:
        ValueError: If dataset_name not recognized
    """
    loaders = {
        'shs': load_shs_metadata,
        'lyric-covers': load_lyric_covers_metadata,
        'discogs-vi': load_discogs_vi_metadata
    }
    
    loader = loaders.get(dataset_name)
    if not loader:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loader(conf, verbose)


def build_info_and_splitdict(
    df: pd.DataFrame, 
    dataset_name: str, 
    verbose: bool = True
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, List[str]]]]:
    """
    Build info dict and splitdict from dataframe.
    
    Args:
        df: DataFrame with metadata
        dataset_name: Dataset identifier
        verbose: Print progress (only from rank 0)
    
    Returns:
        Tuple of (info, splitdict):
            - info: {version_id: {metadata}}
            - splitdict: {split: {clique_id: [version_ids]}}
    """
    info: Dict[str, Dict] = {}
    splitdict: Dict[str, DefaultDict[str, List[str]]] = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list)
    }
    
    for idx, row in df.iterrows():
        # Create version key
        if dataset_name == 'shs':
            version_key = f"{row['set_id']}-{row['ver_id']}"
            filename = f"{version_key}.mp3"
        elif dataset_name == 'lyric-covers':
            version_key = str(row['id'])
            filename = f"{version_key}_audio.mp3"
        elif dataset_name == 'discogs-vi':
            version_key = str(row['base_filename'])
            filename = f"{version_key}.mp3"
        else:
            continue
        
        # Build info entry
        info[version_key] = {
            'id': idx,
            'clique': str(row['clique_id']),
            'clique_idx': row['clique_idx'],
            'version_idx': row['version_idx'],
            'filename': filename,
            'version_key': version_key
        }
        
        # Add dataset-specific fields
        if dataset_name == 'shs':
            info[version_key].update({
                'set_id': int(row['set_id']),
                'ver_id': int(row['ver_id'])
            })
        elif dataset_name == 'lyric-covers':
            info[version_key]['version_id'] = str(row['id'])
        elif dataset_name == 'discogs-vi':
            info[version_key].update({
                'base_filename': str(row['base_filename']),
                'version_id': str(row['version_id'])
            })
        
        # Add to splitdict
        split_name = str(row['split']).lower()
        if split_name in splitdict:
            clique_key = str(row['clique_id'])
            splitdict[split_name][clique_key].append(version_key)
    
    # Convert to regular dicts
    final_splitdict: Dict[str, Dict[str, List[str]]] = {}
    for split in ["train", "val", "test"]:
        final_splitdict[split] = dict(splitdict[split])
    
    return info, final_splitdict


# ============================================================================
# PATH CONSTRUCTION
# ============================================================================

def get_embedding_path(
    dataset_name: str,
    hidden_states_path: str,
    version: str,
    required_filename: str
) -> Optional[Path]:
    """
    Get embedding file path for version.
    
    Args:
        dataset_name: Dataset identifier
        hidden_states_path: Base path to embeddings
        version: Version identifier
        required_filename: Embedding filename (e.g., 'hs_sbert.pt')
    
    Returns:
        Path to embedding file if exists, None otherwise
    """
    base_path = Path(hidden_states_path)
    
    if dataset_name == 'shs':
        if '-' not in version:
            return None
        set_id, ver_id = version.split('-', 1)
        
        possible_folders = [
            set_id,
            f"{set_id}-" if set_id.isdigit() and int(set_id) < 10 else set_id,
            set_id[:2] if len(set_id) > 2 else set_id
        ]
        
        for folder in possible_folders:
            path = base_path / folder / f"{set_id}-{ver_id}" / required_filename
            if path.exists():
                return path
    
    elif dataset_name == 'lyric-covers':
        path = base_path / version / required_filename
        if path.exists():
            return path
    
    elif dataset_name == 'discogs-vi':
        path = base_path / version.replace('/', os.sep) / required_filename
        if path.exists():
            return path
    
    return None


# ============================================================================
# FILTERING
# ============================================================================

def check_audio_exists(dataset_name: str, audio_base_path: Path, version: str) -> bool:
    """
    Check if audio file exists for version.
    
    Args:
        dataset_name: Dataset identifier
        audio_base_path: Base path to audio files
        version: Version identifier
    
    Returns:
        True if audio file exists
    """
    if dataset_name == 'shs':
        if '-' not in version:
            return False
        set_id, ver_id = version.split('-', 1)
        
        possible_folders = [
            set_id,
            f"{set_id}-" if set_id.isdigit() and int(set_id) < 10 else set_id,
            set_id[:2] if len(set_id) > 2 else set_id
        ]
        
        for folder in possible_folders:
            if (audio_base_path / folder / f"{version}.mp3").exists():
                return True
        return False
    
    elif dataset_name == 'lyric-covers':
        return (audio_base_path / version / f"{version}_audio.mp3").exists()
    
    elif dataset_name == 'discogs-vi':
        return (audio_base_path / f"{version}.mp3").exists()
    
    return False


def filter_by_audio(
    splitdict: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    audio_path: str,
    verbose: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Remove versions without audio files.
    
    Args:
        splitdict: Split to clique to versions mapping
        dataset_name: Dataset identifier
        audio_path: Base path to audio
        verbose: Print progress (only from rank 0)
    
    Returns:
        Filtered splitdict
    """
    dist_print("Filtering by audio availability...", verbose)
    
    if dataset_name == 'shs':
        audio_base = Path(audio_path) / "SHS100K" / "audio"
    elif dataset_name == 'lyric-covers':
        audio_base = Path(audio_path) / "LyricCovers" / "audio"
    elif dataset_name == 'discogs-vi':
        audio_base = Path(audio_path) / "DiscogsVI" / "audio"
    else:
        return splitdict
    
    for split in ["train", "val", "test"]:
        filtered = {}
        for clique_id, versions in splitdict[split].items():
            valid_versions = [v for v in versions 
                            if check_audio_exists(dataset_name, audio_base, v)]
            if valid_versions:
                filtered[clique_id] = valid_versions
        splitdict[split] = filtered
        
        total_versions = sum(len(v) for v in splitdict[split].values())
        dist_print(f"  {split}: {len(splitdict[split])} cliques, {total_versions} versions", 
                   verbose)
    
    return splitdict


def filter_single_version_cliques(
    splitdict: Dict[str, Dict[str, List[str]]],
    verbose: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Remove cliques with only 1 version (requires ≥2 for version ID).
    
    Args:
        splitdict: Split to clique to versions mapping
        verbose: Print progress (only from rank 0)
    
    Returns:
        Filtered splitdict with only multi-version cliques
    """
    dist_print("Removing single-version cliques...", verbose)
    
    for split in ["train", "val", "test"]:
        filtered = {cid: vers for cid, vers in splitdict[split].items() 
                   if len(vers) >= 2}
        splitdict[split] = filtered
        
        total = sum(len(v) for v in splitdict[split].values())
        dist_print(f"  {split}: {len(splitdict[split])} cliques, {total} versions", 
                   verbose)
    
    return splitdict


def remove_overlapping_cliques(
    splitdict: Dict[str, Dict[str, List[str]]],
    verbose: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Remove overlapping cliques across splits (train takes priority).
    
    Args:
        splitdict: Split to clique to versions mapping
        verbose: Print progress (only from rank 0)
    
    Returns:
        Splitdict with no overlapping cliques
    """
    dist_print("Removing overlapping cliques...", verbose)
    
    train_cliques = set(splitdict["train"].keys())
    
    removed_val = 0
    removed_test = 0
    
    for cid in list(splitdict["val"].keys()):
        if cid in train_cliques:
            del splitdict["val"][cid]
            removed_val += 1
    
    for cid in list(splitdict["test"].keys()):
        if cid in train_cliques:
            del splitdict["test"][cid]
            removed_test += 1
    
    if verbose and (removed_val or removed_test):
        dist_print(f"  Removed {removed_val} from val, {removed_test} from test", verbose)
    
    return splitdict


# ============================================================================
# EMBEDDING VERIFICATION
# ============================================================================

def check_embedding_exists(
    dataset_name: str,
    hidden_states_path: str,
    version: str,
    filename: str
) -> bool:
    """Check if embedding file exists for version."""
    path = get_embedding_path(dataset_name, hidden_states_path, version, filename)
    return path is not None


def verify_embeddings(
    splitdict: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    hidden_states_path: str,
    required_filename: str,
    verbose: bool = True
) -> bool:
    """
    Verify embeddings exist for all versions.
    
    Args:
        splitdict: Split to clique to versions mapping
        dataset_name: Dataset identifier
        hidden_states_path: Base path to embeddings
        required_filename: Embedding filename
        verbose: Print progress (only from rank 0)
    
    Returns:
        True if all embeddings exist
    
    Note:
        Uses distributed-aware printing to avoid duplicate messages across GPUs.
    """
    dist_print(f"Verifying embeddings ({required_filename})...", verbose)
    
    all_good = True
    for split in ["train", "val", "test"]:
        missing = []
        for clique_id, versions in splitdict[split].items():
            for version in versions:
                if not check_embedding_exists(dataset_name, hidden_states_path,
                                            version, required_filename):
                    missing.append(version)
        
        if missing:
            all_good = False
            dist_print(f"  {split}: {len(missing)} missing embeddings", verbose)
        else:
            total = sum(len(v) for v in splitdict[split].values())
            dist_print(f"  {split}: ✓ All {total} versions have embeddings", verbose)
    
    return all_good


def filter_by_embeddings(
    splitdict: Dict[str, Dict[str, List[str]]],
    dataset_name: str,
    hidden_states_path: str,
    required_filename: str,
    verbose: bool = True
) -> Dict[str, Dict[str, List[str]]]:
    """
    Filter to only versions with embeddings.
    
    Args:
        splitdict: Split to clique to versions mapping
        dataset_name: Dataset identifier
        hidden_states_path: Base path to embeddings
        required_filename: Embedding filename
        verbose: Print progress (only from rank 0)
    
    Returns:
        Filtered splitdict
    """
    dist_print("Filtering by embedding availability...", verbose)
    
    for split in ["train", "val", "test"]:
        filtered = {}
        for clique_id, versions in splitdict[split].items():
            valid_versions = [v for v in versions 
                            if check_embedding_exists(dataset_name, hidden_states_path,
                                                    v, required_filename)]
            if len(valid_versions) >= 2:
                filtered[clique_id] = valid_versions
        splitdict[split] = filtered
        
        total = sum(len(v) for v in splitdict[split].values())
        dist_print(f"  {split}: {len(splitdict[split])} cliques, {total} versions", 
                   verbose)
    
    return splitdict
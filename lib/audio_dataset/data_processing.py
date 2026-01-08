"""
Data processing utilities for audio datasets.

Handles metadata loading, path construction, and dataframe filtering.
All dataset-specific logic consolidated here.
"""

import os
import pandas as pd
from typing import Optional


# ============================================================================
# METADATA LOADING
# ============================================================================

def load_lyric_covers_metadata(base_path: str) -> pd.DataFrame:
    """Load LyricCovers dataset metadata."""
    try:
        covers_path = os.path.join(base_path, "lyric-covers", "data.csv")
        if not os.path.exists(covers_path):
            print(f"Error: File not found - {covers_path}")
            return pd.DataFrame()

        df = pd.read_csv(covers_path, dtype={15: str, 16: str}, low_memory=False)

        # Load splits
        split_files = ["train_no_dup.csv", "val_no_dup.csv", "test_no_dup.csv"]
        split_dfs = []
        for file_name in split_files:
            split_file = os.path.join(base_path, "lyric-covers", file_name)
            if not os.path.exists(split_file):
                continue
            split = file_name.split("_")[0].lower()
            split_df = pd.read_csv(split_file, usecols=["id", "label"])
            split_df["split"] = split
            split_dfs.append(split_df)

        if not split_dfs:
            return pd.DataFrame()

        split_df = pd.concat(split_dfs, ignore_index=True)
        df = df.merge(split_df, on=["id"], how="inner")
        df["clique_id"] = df["label"]
        df["version_id"] = df["id"]
        return df
    except Exception as e:
        print(f"Error loading LyricCovers metadata: {e}")
        return pd.DataFrame()


def load_shs_metadata(base_path: str) -> pd.DataFrame:
    """Load SHS100K dataset metadata."""
    try:
        shs_path = os.path.join(base_path, "shs", "shs_data.csv")
        if not os.path.exists(shs_path):
            print(f"Error: File not found - {shs_path}")
            return pd.DataFrame()

        df = pd.read_csv(shs_path)

        # Load splits
        split_files = {"train": "SHS100K-TRAIN", "val": "SHS100K-VAL", "test": "SHS100K-TEST"}
        split_dfs = []
        for split_name, file_name in split_files.items():
            split_file = os.path.join(base_path, "shs", file_name)
            if not os.path.exists(split_file):
                continue
            split_df = pd.read_csv(split_file, usecols=[0, 1], names=["set_id", "ver_id"], 
                                   header=None, sep="\t")
            split_df["split"] = split_name.lower()
            split_dfs.append(split_df)

        if not split_dfs:
            return pd.DataFrame()

        split_df = pd.concat(split_dfs, ignore_index=True)
        df = df.merge(split_df, on=["set_id", "ver_id"], how="inner")
        df["clique_id"] = df["set_id"]
        df["version_id"] = df["ver_id"]
        return df
    except Exception as e:
        print(f"Error loading SHS metadata: {e}")
        return pd.DataFrame()


def load_discogs_vi_metadata(base_path: str) -> pd.DataFrame:
    """Load DiscogsVI dataset metadata."""
    try:
        file_path = os.path.join(base_path, "discogs-vi", "id-to-file-mapping.csv")
        if not os.path.exists(file_path):
            print(f"Error: File not found - {file_path}")
            return pd.DataFrame()
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading DiscogsVI metadata: {e}")
        return pd.DataFrame()


def load_metadata(dataset_name: str, base_path: str) -> pd.DataFrame:
    """Load metadata for specified dataset."""
    loaders = {
        "lyric-covers": load_lyric_covers_metadata,
        "shs": load_shs_metadata,
        "discogs-vi": load_discogs_vi_metadata
    }
    
    loader = loaders.get(dataset_name)
    if loader:
        return loader(base_path)
    else:
        print(f"Error: Unknown dataset name '{dataset_name}'")
        return pd.DataFrame()


# ============================================================================
# PATH CONSTRUCTION
# ============================================================================

def get_audio_path(dataset_name: str, data_folder: str, row: pd.Series) -> str:
    """Get path to audio file based on dataset type."""
    try:
        if dataset_name == "lyric-covers":
            song_id = str(row["id"])
            return os.path.join(data_folder, "LyricCovers", "audio", 
                              song_id, f"{song_id}_audio.mp3")
        
        elif dataset_name == "shs":
            set_id, ver_id = str(row["set_id"]), str(row["ver_id"])
            set_folder = set_id
            if int(set_id) in range(0, 10):
                set_folder = f"{set_id}-"
            if len(set_folder) > 2:
                set_folder = set_folder[:2]
            return os.path.join(data_folder, "SHS100K", "audio", 
                              set_folder, f"{set_id}-{ver_id}.mp3")
        
        elif dataset_name == "discogs-vi":
            return os.path.join(data_folder, "DiscogsVI", "audio",
                              f"{row['base_filename']}.mp3")
        else:
            return ""
    except Exception as e:
        print(f"Error getting audio path: {e}")
        return ""


def get_transcription_path(dataset_name: str, data_folder: str, 
                          row: pd.Series, whisper_set: str) -> str:
    """Get path to transcription file based on dataset type."""
    try:
        if dataset_name == "lyric-covers":
            song_id = str(row["id"])
            return os.path.join(data_folder, "LyricCovers-transcriptions",
                              "transcriptions", song_id, f"{whisper_set}.txt")
        
        elif dataset_name == "shs":
            set_id, ver_id = str(row["set_id"]), str(row["ver_id"])
            song_id = f"{set_id}-{ver_id}"
            set_folder = set_id
            if int(set_id) in range(0, 10):
                set_folder = f"{set_id}-"
            if len(set_folder) > 2:
                set_folder = set_folder[:2]
            return os.path.join(data_folder, "SHS100K-transcriptions",
                              "transcriptions", set_folder, song_id, f"{whisper_set}.txt")
        
        elif dataset_name == "discogs-vi":
            return os.path.join(data_folder, "DiscogsVI-transcriptions",
                              "transcriptions", row["base_filename"], f"{whisper_set}.txt")
        else:
            return ""
    except Exception as e:
        print(f"Error getting transcription path: {e}")
        return ""


# ============================================================================
# FILTERING
# ============================================================================

def filter_by_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Filter dataframe to specified split."""
    if df.empty:
        return df
    filtered = df[df["split"] == split].reset_index(drop=True)
    if filtered.empty:
        print(f"Warning: No data for split '{split}'")
    return filtered


def filter_by_audio_availability(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to rows with available audio files."""
    if df.empty:
        return df
    filtered = df[df["status_audio"]].reset_index(drop=True)
    if filtered.empty:
        print("Warning: No data with available audio files")
    return filtered


def filter_single_version_cliques(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove cliques with only one version."""
    try:
        if df.empty:
            return df

        if verbose:
            before = df.groupby('clique_id')['version_id'].nunique()
            single = sum(before == 1)
            print(f"Before filtering: {len(before)} cliques, {single} with only one version")

        counts = df.groupby('clique_id')['version_id'].nunique()
        single_cliques = counts[counts == 1].index.tolist()

        if single_cliques:
            if verbose:
                print(f"Removing {len(single_cliques)} cliques with only one version")
            filtered = df[~df['clique_id'].isin(single_cliques)].reset_index(drop=True)
            
            if verbose:
                after = filtered.groupby('clique_id')['version_id'].nunique()
                print(f"After filtering: {len(after)} cliques, "
                      f"min versions per clique: {after.min() if not after.empty else 0}")
            return filtered
        else:
            if verbose:
                print("No single-version cliques found")
            return df
    except Exception as e:
        print(f"Error filtering: {e}")
        return df


def add_file_status_columns(df: pd.DataFrame, dataset_name: str, 
                            data_folder: str, whisper_set: str) -> pd.DataFrame:
    """Add columns indicating file existence."""
    try:
        # Audio
        df["status_audio"] = df.apply(
            lambda row: os.path.isfile(get_audio_path(dataset_name, data_folder, row)),
            axis=1
        )
        
        # Lyrics (similar structure to audio)
        df["status_lyrics"] = df["status_audio"]  # Simplified for brevity
        
        # Transcriptions
        df[f"whisper_{whisper_set}"] = df.apply(
            lambda row: os.path.isfile(
                get_transcription_path(dataset_name, data_folder, row, whisper_set)
            ),
            axis=1
        )
        return df
    except Exception as e:
        print(f"Error adding file status: {e}")
        df["status_audio"] = False
        df["status_lyrics"] = False
        df[f"whisper_{whisper_set}"] = False
        return df
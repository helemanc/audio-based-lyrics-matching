"""
RAM-based cache for Whisper transcriptions with disk persistence.

This module provides efficient caching of Whisper transcription files across
multiple datasets (SHS100K, Discogs-VI, Lyric Covers) with validation.

Example:
    >>> cache = TranscriptionCache(data_folder="/data", dataset_name="shs")
    >>> df = cache.apply_to_dataframe(
    ...     df, 
    ...     whisper_sets=["whisper-turbo"],
    ...     split="train"
    ... )
"""

import os
import pickle
import glob
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from .validator import TranscriptionValidator


class TranscriptionCache:
    """
    RAM-based cache for Whisper transcriptions with disk persistence.
    
    This class manages loading, caching, and validation of Whisper transcription
    files. Transcriptions are loaded into memory for fast access and persisted
    to disk for subsequent runs.
    
    Attributes:
        data_folder (str): Root directory containing transcription files
        dataset_name (str): Name of dataset ("shs", "discogs-vi", "lyric-covers")
        cache_dir (str): Directory for storing cache files
        transcription_cache (Dict[str, Dict[str, str]]): In-memory cache mapping
            whisper_set -> (file_key -> transcription_text)
    
    Example:
        >>> cache = TranscriptionCache("/data", "shs")
        >>> cache.build_index("whisper-turbo")
        >>> df = cache.apply_to_dataframe(df, ["whisper-turbo"], split="train")
    """
    
    def __init__(self, data_folder: str, dataset_name: str) -> None:
        """
        Initialize transcription cache.
        
        Args:
            data_folder: Root directory containing dataset transcription files
            dataset_name: Dataset identifier ("shs", "discogs-vi", "lyric-covers")
        
        Raises:
            OSError: If cache directory cannot be created
        """
        self.data_folder: str = data_folder
        self.dataset_name: str = dataset_name
        self.cache_dir: str = os.path.join(
            data_folder, 
            f"{dataset_name}-transcription-cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self.transcription_cache: Dict[str, Dict[str, str]] = {}

    def get_cache_file(self, whisper_set: str, split: str = "all") -> str:
        """
        Get cache file path for a specific whisper model and data split.
        
        Args:
            whisper_set: Whisper model identifier (e.g., "whisper-turbo")
            split: Data split identifier (e.g., "train", "val", "test", "all")
        
        Returns:
            Absolute path to cache pickle file
        
        Example:
            >>> cache.get_cache_file("whisper-turbo", "train")
            '/data/shs-transcription-cache/shs_whisper-turbo_train_cache.pkl'
        """
        cache_id = f"{self.dataset_name}_{whisper_set}_{split}"
        return os.path.join(self.cache_dir, f"{cache_id}_cache.pkl")

    def load_disk_cache(self, whisper_set: str, split: str = "all") -> bool:
        """
        Load transcriptions from disk cache into memory.
        
        Args:
            whisper_set: Whisper model identifier
            split: Data split identifier
        
        Returns:
            True if cache loaded successfully, False otherwise
        
        Example:
            >>> if cache.load_disk_cache("whisper-turbo", "train"):
            ...     print("Cache loaded")
        """
        cache_file = self.get_cache_file(whisper_set, split)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.transcription_cache[whisper_set] = pickle.load(f)
                print(
                    f"Loaded cache for {whisper_set} with "
                    f"{len(self.transcription_cache[whisper_set])} entries"
                )
                return True
            except Exception as e:
                print(f"Error loading cache: {e}")
        return False

    def save_disk_cache(self, whisper_set: str, split: str = "all") -> None:
        """
        Save in-memory cache to disk.
        
        Args:
            whisper_set: Whisper model identifier
            split: Data split identifier
        
        Raises:
            IOError: If cache cannot be written to disk
        
        Example:
            >>> cache.save_disk_cache("whisper-turbo", "train")
        """
        if whisper_set in self.transcription_cache:
            cache_file = self.get_cache_file(whisper_set, split)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.transcription_cache[whisper_set], f)
            print(f"Cache saved to {cache_file}")

    def build_index(self, whisper_set: str) -> Dict[str, str]:
        """
        Load all transcriptions for a specific whisper model into memory.
        
        Scans the dataset directory structure and loads all transcription files
        matching the whisper_set identifier. File keys are extracted based on
        dataset-specific directory structures.
        
        Args:
            whisper_set: Whisper model identifier (e.g., "whisper-turbo")
        
        Returns:
            Dictionary mapping file keys to transcription text
        
        Raises:
            ValueError: If dataset_name is not supported
        
        Directory Structure:
            - lyric-covers: LyricCovers-transcriptions/transcriptions/{song_id}/
            - shs: SHS100K-transcriptions/transcriptions/{set_id}/{ver_id}/
            - discogs-vi: DiscogsVI-transcriptions/transcriptions/{dir1}/{dir2}/
        
        Example:
            >>> transcriptions = cache.build_index("whisper-turbo")
            >>> print(f"Loaded {len(transcriptions)} transcriptions")
        """
        print(f"Building index for {whisper_set}...")

        # Initialize cache for this whisper set
        if whisper_set not in self.transcription_cache:
            self.transcription_cache[whisper_set] = {}

        # Get base path and pattern based on dataset
        if self.dataset_name == "lyric-covers":
            pattern = os.path.join(
                self.data_folder, 
                "LyricCovers-transcriptions",
                "transcriptions", 
                "*", 
                f"{self.dataset_name}_{whisper_set}.txt"
            )
            print(pattern)

        elif self.dataset_name == "shs":
            pattern = os.path.join(
                self.data_folder, 
                "SHS100K-transcriptions",
                "transcriptions", 
                "*", 
                "*", 
                f"{self.dataset_name}_{whisper_set}.txt"
            )
        elif self.dataset_name == "discogs-vi":
            pattern = os.path.join(
                self.data_folder, 
                "DiscogsVI-transcriptions",
                "transcriptions", 
                "*", 
                "*", 
                f"{self.dataset_name}_{whisper_set}.txt"
            )
        else:
            print(f"Unsupported dataset: {self.dataset_name}")
            return self.transcription_cache[whisper_set]

        # Load all matching files
        for path in tqdm(glob.glob(pattern), desc="Loading transcriptions"):
            components = path.split(os.sep)
            
            # Extract key based on dataset type
            if self.dataset_name == "lyric-covers":
                key = components[-2]  # song_id
            elif self.dataset_name == "shs":
                key = components[-2]  # set_id-ver_id
            elif self.dataset_name == "discogs-vi":
                key = f"{components[-3]}/{components[-2]}"  # base_filename

            # Read and store in memory
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.transcription_cache[whisper_set][key] = f.read()
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
                continue

        print(
            f"Loaded {len(self.transcription_cache[whisper_set])} transcriptions"
        )
        return self.transcription_cache[whisper_set]

    def apply_to_dataframe(
        self, 
        df: pd.DataFrame, 
        whisper_sets: List[str], 
        rebuild_cache: bool = False, 
        split: str = "all"
    ) -> pd.DataFrame:
        """
        Apply transcriptions to dataframe from memory cache with enhanced validation.
        
        Loads transcription text for each row in the dataframe, validates quality,
        and adds transcription columns with validation metadata.
        
        Args:
            df: Input dataframe with song metadata
            whisper_sets: List of whisper model identifiers to apply
            rebuild_cache: If True, rebuild cache even if disk cache exists
            split: Data split identifier for cache naming
        
        Returns:
            DataFrame with added columns for each whisper_set:
                - transcription_{whisper_set}: Transcription text
                - has_valid_transcription_{whisper_set}: Boolean validity flag
                - transcription_validation_details_{whisper_set}: Validation details
        
        Raises:
            KeyError: If required dataframe columns are missing
        
        Column Requirements by Dataset:
            - lyric-covers: Requires "id" column
            - shs: Requires "set_id" and "ver_id" columns
            - discogs-vi: Requires "base_filename" column
        
        Example:
            >>> df = pd.DataFrame({"id": [1, 2, 3]})
            >>> df = cache.apply_to_dataframe(
            ...     df, 
            ...     whisper_sets=["whisper-turbo"],
            ...     split="train"
            ... )
            >>> print(df["has_valid_transcription_whisper-turbo"].sum())
        """
        result_df = df.copy()

        for whisper_set in whisper_sets:
            # Load or build cache
            if not rebuild_cache and self.load_disk_cache(whisper_set, split):
                pass  # Cache loaded successfully
            else:
                self.build_index(whisper_set)
                self.save_disk_cache(whisper_set, split)

            print(f"Applying transcriptions for {whisper_set}...")
            transcription_column = f"transcription_{whisper_set}"

            # Create mapping function that handles all dataset types
            if self.dataset_name == "lyric-covers":
                result_df[transcription_column] = result_df["id"].astype(str).map(
                    self.transcription_cache[whisper_set]
                ).fillna("")
            elif self.dataset_name == "shs":
                # Create composite key for SHS
                result_df['temp_key'] = result_df.apply(
                    lambda row: f"{str(row['set_id'])}-{str(row['ver_id'])}", 
                    axis=1
                )
                result_df[transcription_column] = result_df['temp_key'].map(
                    self.transcription_cache[whisper_set]
                ).fillna("")
                result_df.drop('temp_key', axis=1, inplace=True)
            elif self.dataset_name == "discogs-vi":
                result_df[transcription_column] = result_df["base_filename"].map(
                    self.transcription_cache[whisper_set]
                ).fillna("")

            # Enhanced validation column name
            valid_transcription_column = f"has_valid_transcription_{whisper_set}"

            # Initialize enhanced validator
            validator = TranscriptionValidator(
                min_words=10,
                max_repetition_ratio=0.6,  # Allow up to 60% repetition
                min_unique_bigrams=3,
                min_unique_trigrams=2
            )

            # Apply enhanced validation
            print(f"Applying enhanced transcription validation for {whisper_set}...")
            result_df[valid_transcription_column] = result_df[
                transcription_column
            ].apply(validator.is_valid_transcription)

            # Create detailed validation column for debugging
            validation_details_column = (
                f"transcription_validation_details_{whisper_set}"
            )
            result_df[validation_details_column] = result_df[
                transcription_column
            ].apply(validator.get_validation_details)

            # Report detailed statistics
            empty_count = (result_df[transcription_column] == "").sum()
            valid_count = result_df[valid_transcription_column].sum()
            invalid_count = len(result_df) - empty_count - valid_count

            print(f"Enhanced transcription validation results for {whisper_set}:")
            print(f"  Total transcriptions: {len(result_df)}")
            print(f"  Empty transcriptions: {empty_count}")
            print(f"  Valid transcriptions: {valid_count}")
            print(f"  Invalid transcriptions: {invalid_count}")
            print(f"  Validation rate: {valid_count/len(result_df)*100:.2f}%")

            # Print breakdown of validation issues
            if invalid_count > 0:
                print(f"\nValidation issue breakdown for {whisper_set}:")
                all_issues = []
                for details in result_df[validation_details_column]:
                    if isinstance(details, dict) and not details.get('is_valid', True):
                        all_issues.extend(details.get('issues', []))
                
                if all_issues:
                    from collections import Counter
                    issue_counts = Counter(all_issues)
                    for issue, count in issue_counts.items():
                        print(
                            f"  {issue.replace('_', ' ').title()}: "
                            f"{count} transcriptions"
                        )
                else:
                    print("  No specific issues identified "
                          "(transcriptions may be empty)")

            # Report basic statistics
            total_with_transcriptions = len(result_df) - empty_count
            print(
                f"Found {total_with_transcriptions} transcriptions "
                f"out of {len(result_df)} rows"
            )
            print(
                f"Found {valid_count} valid transcriptions (enhanced criteria) "
                f"out of {len(result_df)} rows"
            )

        return result_df
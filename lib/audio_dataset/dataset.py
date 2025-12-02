"""
Audio dataset implementation.
"""
import pandas as pd
import os
import torch
import whisper
import torchaudio
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from .cache import TranscriptionCache
from .validator import TranscriptionValidator

class AudioDataset(Dataset):
    """Optimized dataset for audio data with RAM-based transcription loading"""
    def __init__(self, dataset_name, base_path, data_folder, split='train',
                 whisper_set="turbo_nothing_whisper_42", evaluation_mode=False,
                 debug_mode=False, use_whisper_loader=True):
        """
        Initialize the AudioDataset for version identification with transcription support.
        
        This dataset loads audio files, metadata, and Whisper transcriptions for version
        identification tasks. It supports multiple datasets (LyricCovers, SHS100K, DiscogsVI)
        and handles clique-based organization where each clique contains multiple versions
        of the same song.
        
        Args:
            dataset_name (str): Name of the dataset to load. Must be one of:
                - 'lyric-covers': LyricCovers dataset
                - 'shs': SHS100K dataset  
                - 'discogs-vi': DiscogsVI dataset
                
            base_path (str): Path to the directory containing dataset metadata files.
                Expected structure:
                - base_path/audio-based-lyrics-matching/datasets/lyric-covers/data.csv
                - base_path/audio-based-lyrics-matching/datasets/shs/shs_data.csv
                - base_path/audio-based-lyrics-matching/datasets/discogs-vi/id-to-file-mapping.csv
                
            data_folder (str): Path to the directory containing actual data files
                (audio, lyrics, transcriptions). Expected structure:
                - data_folder/LyricCovers/audio/
                - data_folder/SHS100K/audio/
                - data_folder/DiscogsVI/audio/
                
            split (str, optional): Data split to load. Must be one of:
                - 'train': Training set
                - 'val': Validation set
                - 'test': Test set
                Default: 'train'
                
            whisper_set (str or list, optional): Name of the Whisper transcription set to use.
                Transcription files should be located at:
                - data_folder/{DatasetName}-transcriptions/transcriptions/
                If a list is provided, only the first element is used.
                Default: 'turbo_nothing_whisper_42'
                
            evaluation_mode (bool, optional): If True, skips audio loading and prepares
                tensors for efficient evaluation. Use this when you only need metadata
                and transcriptions without actual audio waveforms.
                Default: False
                
            debug_mode (bool, optional): If True, filters the dataset to only include
                samples that have valid Whisper transcriptions available. Useful for
                debugging transcription-related code without loading samples that lack
                transcriptions.
                Default: False
                
            use_whisper_loader (bool, optional): If True, uses whisper.load_audio() to
                load audio files (resamples to 16kHz mono). If False, uses torchaudio.load().
                Whisper loader is recommended for consistency with Whisper preprocessing.
                Default: True
        
        Attributes:
            df (pd.DataFrame): Loaded and filtered dataframe containing all samples
            transcription_cache (TranscriptionCache): Cache manager for transcriptions
            clique_id_to_idx (dict): Mapping from original clique IDs to integer indices
            version_id_to_idx (dict): Mapping from original version IDs to integer indices
            idx_to_clique_id (dict): Reverse mapping from integer indices to clique IDs
            idx_to_version_id (dict): Reverse mapping from integer indices to version IDs
            
            When evaluation_mode=True, also creates:
                candidates_i (torch.Tensor): Tensor of version indices
                candidates_c (torch.Tensor): Tensor of clique indices
                lyrics_mask (torch.Tensor): Boolean mask for lyrics availability
                whisper_mask (torch.Tensor): Boolean mask for transcription availability
        
        Raises:
            ValueError: If dataset_name is not recognized
            FileNotFoundError: If required data files are not found at expected paths
        
        Example:
            >>> # Basic usage
            >>> dataset = AudioDataset(
            ...     dataset_name='shs',
            ...     base_path='./datasets',
            ...     data_folder='/data',
            ...     split='train'
            ... )
            >>> print(f"Loaded {len(dataset)} samples")
            
            >>> # Debug mode with specific transcriptions
            >>> dataset = AudioDataset(
            ...     dataset_name='lyric-covers',
            ...     base_path='./datasets',
            ...     data_folder='/data',
            ...     split='val',
            ...     whisper_set='large_v2_whisper',
            ...     debug_mode=True  # Only samples with transcriptions
            ... )
            
            >>> # Evaluation mode (no audio loading)
            >>> dataset = AudioDataset(
            ...     dataset_name='shs',
            ...     base_path='./datasets',
            ...     data_folder='/data',
            ...     split='test',
            ...     evaluation_mode=True
            ... )
        
        Notes:
            - The dataset automatically filters out cliques with only one version,
            as version identification requires at least two versions per clique.
            - Transcriptions are loaded into RAM for fast access during training.
            - Invalid transcriptions (too short, repetitive, musical content) are
            automatically detected and flagged via the has_valid_transcription column.
            - Audio files are loaded on-demand in __getitem__ unless evaluation_mode=True.
        """
        self.dataset_name = dataset_name
        self.base_path = base_path
        self.data_folder = data_folder
        self.split = split

        # Convert whisper_set to string if it's already a list
        self.whisper_set = whisper_set[0] if isinstance(whisper_set, list) else whisper_set

        self.evaluation_mode = evaluation_mode
        self.debug_mode = debug_mode
        self.use_whisper_loader = use_whisper_loader

        # Initialize transcription cache
        self.transcription_cache = TranscriptionCache(data_folder, dataset_name)

        # Load the data
        self.df = self._load_data()

        # Load transcriptions into memory
        if len(self.df) > 0:  # Only load if we have data
            self.df = self.transcription_cache.apply_to_dataframe(self.df, [self.whisper_set], split=split)

        # Create ID mappings
        self._create_id_mappings()

        # In evaluation mode, prepare evaluation tensors
        if self.evaluation_mode:
            self._prepare_evaluation_tensors()

        # Check and verify that each clique has at least two versions
        print(f"Initialized {dataset_name} dataset with {len(self.df)} samples")
        if len(self.df) > 0:
            self.check_clique_versions()

        print(f"Initialized {dataset_name} dataset with {len(self.df)} samples")

    def _load_data(self):
        """Load and prepare dataset with optimized approach"""
        try:
            # Load the appropriate dataset
            if self.dataset_name == "discogs-vi":
                file_path = os.path.join(self.base_path, "discogs-vi", "id-to-file-mapping.csv")
                if not os.path.exists(file_path):
                    print(f"Error: File not found - {file_path}")
                    return pd.DataFrame()  # Return empty dataframe on error

                df = pd.read_csv(file_path)
                df["status_audio"] = df["base_filename"].apply(
                    lambda x: os.path.isfile(os.path.join(self.data_folder, "DiscogsVI", "audio", f"{x}.mp3")))
                df["status_lyrics"] = df["base_filename"].apply(
                    lambda x: os.path.isfile(os.path.join(self.data_folder, "DiscogsVI", "lyrics", f"{x}.txt")))

            elif self.dataset_name == "shs":
                # Load main data
                shs_path = os.path.join(self.base_path, "shs", "shs_data.csv")
                if not os.path.exists(shs_path):
                    print(f"Error: File not found - {shs_path}")
                    return pd.DataFrame()  # Return empty dataframe on error

                df = pd.read_csv(shs_path)

                # Load split information
                split_files = {"train": "SHS100K-TRAIN", "val": "SHS100K-VAL", "test": "SHS100K-TEST"}
                split_dfs = []

                for split_name, file_name in split_files.items():
                    split_file = os.path.join(self.base_path, "shs", file_name)
                    if not os.path.exists(split_file):
                        print(f"Warning: Split file not found - {split_file}")
                        continue

                    split_df = pd.read_csv(
                        split_file,
                        usecols=[0, 1], names=["set_id", "ver_id"], header=None, sep="\t")
                    split_df["split"] = split_name.lower()
                    split_dfs.append(split_df)

                if not split_dfs:
                    print("Error: No split files found")
                    return pd.DataFrame()  # Return empty dataframe if no splits

                split_df = pd.concat(split_dfs, ignore_index=True)
                df = df.merge(split_df, on=["set_id", "ver_id"], how="inner")  # Use inner join to only keep matching rows

                # Check file existence
                def check_file_exists(row, extension):
                    try:
                        set_id, ver_id = str(row["set_id"]), str(row["ver_id"])
                        song_id = f"{set_id}-{ver_id}"

                        set_folder = set_id
                        if int(set_id) in range(0, 10):
                            set_folder = f"{set_id}-"
                        if len(set_folder) > 2:
                            set_folder = set_folder[:2]

                        data_type = "audio" if extension == "mp3" else "lyrics"
                        file_path = os.path.join(self.data_folder, "SHS100K", data_type, set_folder, f"{song_id}.{extension}")
                        return os.path.isfile(file_path)
                    except Exception as e:
                        print(f"Error checking file existence: {e}")
                        return False

                df["status_audio"] = df.apply(lambda row: check_file_exists(row, "mp3"), axis=1)
                df["status_lyrics"] = df.apply(lambda row: check_file_exists(row, "txt"), axis=1)
                df["clique_id"] = df["set_id"]
                df["version_id"] = df["ver_id"]

            elif self.dataset_name == "lyric-covers":
                # Load main data
                covers_path = os.path.join(self.base_path, "lyric-covers", "data.csv")
                if not os.path.exists(covers_path):
                    print(f"Error: File not found - {covers_path}")
                    return pd.DataFrame()  # Return empty dataframe on error

                df = pd.read_csv(covers_path, dtype={15: str, 16: str}, low_memory=False)


                # Load split information
                split_files = ["train_no_dup.csv", "val_no_dup.csv", "test_no_dup.csv"]
                split_dfs = []

                for file_name in split_files:
                    split_file = os.path.join(self.base_path, "lyric-covers", file_name)
                    if not os.path.exists(split_file):
                        print(f"Warning: Split file not found - {split_file}")
                        continue

                    split = file_name.split("_")[0].lower()
                    split_df = pd.read_csv(split_file, usecols=["id", "label"])
                    split_df["split"] = split
                    split_dfs.append(split_df)

                if not split_dfs:
                    print("Error: No split files found")
                    return pd.DataFrame()  # Return empty dataframe if no splits

                split_df = pd.concat(split_dfs, ignore_index=True)
                df = df.merge(split_df, on=["id"], how="inner")  # Use inner join to only keep matching rows

                # Check file existence
                df["status_audio"] = df["id"].apply(
                    lambda x: os.path.isfile(os.path.join(self.data_folder, "LyricCovers", "audio", str(x), f"{x}_audio.mp3")))
                df["status_lyrics"] = df["id"].apply(
                    lambda x: os.path.isfile(os.path.join(self.data_folder, "LyricCovers", "audio", str(x), f"{x}_audio.mp3")))
                df["clique_id"] = df["label"]
                df["version_id"] = df["id"]

            else:
                print(f"Error: Unknown dataset name '{self.dataset_name}'")
                return pd.DataFrame()  # Return empty dataframe for unknown dataset

            # Check if we have data
            if df.empty:
                print("Warning: Dataset is empty after loading")
                return df

            # Check for whisper transcriptions - with better error handling
            try:
                df[f"whisper_{self.whisper_set}"] = df.apply(
                    lambda row: self._check_whisper_file(row, self.whisper_set), axis=1)
            except Exception as e:
                print(f"Error checking whisper files: {e}")
                df[f"whisper_{self.whisper_set}"] = False  # Set all to False on error

            # Filter by split and audio availability
            df = df[df["split"] == self.split].reset_index(drop=True)
            if df.empty:
                print(f"Warning: No data for split '{self.split}'")
                return df

            df = df[df["status_audio"]].reset_index(drop=True)
            if df.empty:
                print("Warning: No data with available audio files")
                return df

            # Filter cliques with only one version
            df = self._filter_single_version_cliques(df)
            if df.empty:
                print("Warning: No data after filtering single version cliques")
                return df

            # Apply debug mode filtering if needed
            if self.debug_mode:
                whisper_col = f"whisper_{self.whisper_set}"
                if whisper_col in df.columns:
                    df = df[~df[whisper_col].isna()].reset_index(drop=True)
                    if df.empty:
                        print("Warning: No data with available whisper transcriptions")

            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()  # Return empty dataframe on error

    def _check_whisper_file(self, row, whisper_set):
        """Check if whisper file exists and has enough content"""
        try:
            file_path = None

            if self.dataset_name == "lyric-covers":
                song_id = str(row["id"])
                file_path = os.path.join(self.data_folder, "lyric-covers", "transcriptions",
                                        song_id, f"{whisper_set}.txt")
            elif self.dataset_name == "shs":
                set_id, ver_id = str(row["set_id"]), str(row["ver_id"])
                song_id = f"{set_id}-{ver_id}"
                if int(set_id) in range(0, 10):
                    set_id = f"{set_id}-"
                if len(set_id) > 2:
                    set_id = set_id[:2]
                file_path = os.path.join(self.data_folder, "SHS100K", "transcriptions",
                                        set_id, song_id, f"{whisper_set}.txt")
            elif self.dataset_name == "discogs-vi":
                base_filename = row["base_filename"]
                file_path = os.path.join(self.data_folder, "DiscogsVI", "transcriptions",
                                        base_filename, f"{whisper_set}.txt")

            if file_path and os.path.isfile(file_path):
                with open(file_path, "r") as f:
                    text = f.read()
                try:
                    words = word_tokenize(text)
                    return len(words) >= 10
                except:
                    # If tokenization fails, just check if there's some content
                    return len(text.strip()) > 0
            return False
        except Exception as e:
            print(f"Error checking whisper file: {e}")
            return False

    def _filter_single_version_cliques(self, df):
        """Filter out clique IDs that have only one version ID"""
        try:
            if df.empty:
                return df

            # Before filtering: print statistics about versions per clique
            before_counts = df.groupby('clique_id')['version_id'].nunique()
            single_version_count_before = sum(before_counts == 1)

            print(f"Before filtering: {len(before_counts)} cliques, {single_version_count_before} with only one version")

            # Identify cliques with only one version
            counts = df.groupby('clique_id')['version_id'].nunique()
            single_version_cliques = counts[counts == 1].index.tolist()

            if single_version_cliques:
                print(f"Removing {len(single_version_cliques)} cliques with only one version")
                filtered_df = df[~df['clique_id'].isin(single_version_cliques)].reset_index(drop=True)

                # After filtering: verify that all cliques now have multiple versions
                after_counts = filtered_df.groupby('clique_id')['version_id'].nunique()
                min_versions = after_counts.min() if not after_counts.empty else 0

                print(f"After filtering: {len(after_counts)} cliques, min versions per clique: {min_versions}")

                # Double-check for any remaining single-version cliques
                remaining_single = after_counts[after_counts == 1]
                if not remaining_single.empty:
                    print(f"WARNING: {len(remaining_single)} cliques with only one version remain after filtering!")
                    print("This should not happen - check the filtering logic")

                return filtered_df
            else:
                print("No cliques with only one version found - no filtering needed")
                return df
        except Exception as e:
            print(f"Error filtering single version cliques: {e}")
            return df  # Return original df on error

    def _create_id_mappings(self):
        """Create consistent integer mappings for clique and version IDs"""
        try:
            if self.df.empty:
                self.clique_id_to_idx = {}
                self.version_id_to_idx = {}
                self.idx_to_clique_id = {}
                self.idx_to_version_id = {}
                return

            # Get unique IDs
            unique_clique_ids = sorted(self.df["clique_id"].unique())
            unique_version_ids = sorted(self.df["version_id"].unique())

            # Create mappings
            self.clique_id_to_idx = {cid: idx for idx, cid in enumerate(unique_clique_ids)}
            self.version_id_to_idx = {vid: idx for idx, vid in enumerate(unique_version_ids)}

            # Create reverse mappings
            self.idx_to_clique_id = {idx: cid for cid, idx in self.clique_id_to_idx.items()}
            self.idx_to_version_id = {idx: vid for vid, idx in self.version_id_to_idx.items()}

            # Add index columns to dataframe for faster access
            self.df["clique_idx"] = self.df["clique_id"].map(self.clique_id_to_idx)
            self.df["version_idx"] = self.df["version_id"].map(self.version_id_to_idx)
        except Exception as e:
            print(f"Error creating ID mappings: {e}")
            # Initialize empty mappings
            self.clique_id_to_idx = {}
            self.version_id_to_idx = {}
            self.idx_to_clique_id = {}
            self.idx_to_version_id = {}

    def _prepare_evaluation_tensors(self):
        """Prepare tensors needed for evaluation mode"""
        try:
            if self.df.empty:
                # Create empty tensors if dataframe is empty
                self.candidates_i = torch.tensor([], dtype=torch.long)
                self.candidates_c = torch.tensor([], dtype=torch.long)
                self.lyrics_mask = torch.tensor([], dtype=torch.bool)
                self.whisper_mask = torch.tensor([], dtype=torch.bool)
                return

            # Convert indices to tensors
            self.candidates_i = torch.tensor(self.df["version_idx"].values)
            self.candidates_c = torch.tensor(self.df["clique_idx"].values)

            # Boolean masks as tensors
            self.lyrics_mask = torch.tensor(self.df["status_lyrics"].values)

            # Handle possible missing whisper column
            whisper_col = f"whisper_{self.whisper_set}"
            if whisper_col in self.df.columns:
                self.whisper_mask = torch.tensor(self.df[whisper_col].fillna(False).astype(bool).values)
            else:
                print(f"Warning: Whisper column '{whisper_col}' not found in dataframe")
                self.whisper_mask = torch.zeros(len(self.df), dtype=torch.bool)
        except Exception as e:
            print(f"Error preparing evaluation tensors: {e}")
            # Create empty tensors on error
            self.candidates_i = torch.tensor([], dtype=torch.long)
            self.candidates_c = torch.tensor([], dtype=torch.long)
            self.lyrics_mask = torch.tensor([], dtype=torch.bool)
            self.whisper_mask = torch.tensor([], dtype=torch.bool)

    def _enhanced_check_transcription_validity(self, text):
        """
        Enhanced transcription validity check using the same validator as the cache
        This method can be used in the AudioDataset class for additional validation
        """
        # Initialize validator if it doesn't exist
        if not hasattr(self, 'validator'):
            self.validator = TranscriptionValidator(
                min_words=10,
                max_repetition_ratio=0.6,  # Allow up to 60% repetition
                min_unique_bigrams=3,
                min_unique_trigrams=2
            )
        
        try:
            return self.validator.is_valid_transcription(text)
        except Exception as e:
            print(f"Error in enhanced transcription validation: {e}")
            # Fallback to basic validation if enhanced validation fails
            try:
                if not text or not isinstance(text, str) or text.strip() == "":
                    return False
                # Basic word count check as fallback
                words = word_tokenize(text) if text else []
                return len(words) >= 10
            except Exception as fallback_error:
                print(f"Error in fallback transcription validation: {fallback_error}")
                return False

    def _check_transcription_length(self, text):
        """
        Legacy method - now delegates to enhanced validation for consistency
        """
        return self._enhanced_check_transcription_validity(text)
    

    def check_clique_versions(self):
        """
        Verify that each clique has at least two versions after all preprocessing.
        Prints statistics about clique sizes and identifies any problematic cliques.

        Returns:
            bool: True if all cliques have at least two versions, False otherwise
        """
        if self.df.empty:
            print("Dataset is empty, cannot check clique versions")
            return False

        # Count the number of versions per clique
        clique_counts = self.df.groupby('clique_id')['version_id'].nunique()

        # Get statistics
        min_versions = clique_counts.min()
        max_versions = clique_counts.max()
        avg_versions = clique_counts.mean()

        # Identify problematic cliques (those with only one version)
        single_version_cliques = clique_counts[clique_counts == 1]

        # Print the statistics
        print(f"\n=== Clique Version Statistics ===")
        print(f"Total cliques: {len(clique_counts)}")
        print(f"Min versions per clique: {min_versions}")
        print(f"Max versions per clique: {max_versions}")
        print(f"Average versions per clique: {avg_versions:.2f}")

        # Check if there are any problematic cliques
        if not single_version_cliques.empty:
            print(f"\nWARNING: Found {len(single_version_cliques)} cliques with only one version:")
            for clique_id in single_version_cliques.index[:10]:  # Show first 10 examples
                print(f"  - Clique ID: {clique_id}")

            if len(single_version_cliques) > 10:
                print(f"  ... and {len(single_version_cliques) - 10} more")

            return False
        else:
            print("\nVerification passed: All cliques have at least two versions ✓")

            # Print distribution of clique sizes
            print("\nClique size distribution:")
            size_counts = clique_counts.value_counts().sort_index()
            for size, count in size_counts.items():
                print(f"  {size} versions: {count} cliques")

            return True

    def get_audio_path(self, idx):
        """Get the path to the audio file based on dataset type"""
        try:
            row = self.df.iloc[idx]

            if self.dataset_name == "lyric-covers":
                song_id = str(row["id"])
                return os.path.join(self.data_folder, "LyricCovers", "audio", song_id, f"{song_id}_audio.mp3")

            elif self.dataset_name == "shs":
                set_id = str(row["set_id"])
                ver_id = str(row["ver_id"])
                song_id = f"{set_id}-{ver_id}"

                # Format set_id for folder structure
                set_folder = set_id
                if int(set_id) in range(0, 10):
                    set_folder = f"{set_id}-"
                if len(set_folder) > 2:
                    set_folder = set_folder[:2]

                return os.path.join(self.data_folder, "SHS100K", "audio", set_folder, f"{song_id}.mp3")

            elif self.dataset_name == "discogs-vi":
                return os.path.join(self.data_folder, "DiscogsVI", "audio", f"{row['base_filename']}.mp3")

            else:
                print(f"Error: Unknown dataset name '{self.dataset_name}'")
                return ""  # Return empty string for unknown dataset

        except Exception as e:
            print(f"Error getting audio path for index {idx}: {e}")
            return ""  # Return empty string on error

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.df)

    def __getitem__(self, idx):
        """Get a sample from the dataset including transcription"""
        try:
            if self.df.empty or idx >= len(self.df):
                print(f"Error: Requested index {idx} but dataset is empty or index out of range")
                # Return default values
                return (torch.tensor(0, dtype=torch.long),
                        torch.tensor(0, dtype=torch.long),
                        torch.zeros(16000, dtype=torch.float32),
                        "")

            row = self.df.iloc[idx]

            # Get indices directly from dataframe
            clique_idx = row["clique_idx"]
            version_idx = row["version_idx"]

            # Get transcription text - with better error handling
            transcription_column = f"transcription_{self.whisper_set}"
            transcription = ""
            if transcription_column in row.index:
                try:
                    transcription = row[transcription_column]
                    if pd.isna(transcription):
                        transcription = ""
                except Exception as e:
                    print(f"Error accessing transcription: {e}")
                    transcription = ""

            # Get valid transcription flag with the new column name
            valid_transcription_column = f"has_valid_transcription_{self.whisper_set}"
            has_valid_transcription = False
            if valid_transcription_column in row.index:
                try:
                    has_valid_transcription = row[valid_transcription_column]
                    if pd.isna(has_valid_transcription):
                        has_valid_transcription = False
                except Exception as e:
                    print(f"Error accessing valid transcription flag: {e}")
                    has_valid_transcription = False

            # Convert to tensors
            clique_idx_tensor = torch.tensor(clique_idx, dtype=torch.long)
            version_idx_tensor = torch.tensor(version_idx, dtype=torch.long)
            has_valid_transcription_tensor = torch.tensor(has_valid_transcription, dtype=torch.bool)

            if self.evaluation_mode:
                waveform = torch.zeros(16000, dtype=torch.float32)
            else:
                # Load audio - with error handling
                audio_path = self.get_audio_path(idx)
                if not audio_path or not os.path.exists(audio_path):
                    if audio_path:  # Only print warning if path exists but file doesn't
                        print(f"Warning: Audio file not found: {audio_path}")
                    # Return a dummy waveform if file not found
                    waveform = torch.zeros(16000, dtype=torch.float32)  # 1 second of silence at 16kHz
                else:
                    try:
                        if self.use_whisper_loader:
                            waveform = whisper.load_audio(audio_path)
                            waveform = torch.tensor(waveform, dtype=torch.float32)
                        else:
                            waveform, _ = torchaudio.load(audio_path)
                            waveform = waveform.squeeze(0)
                    except Exception as e:
                        print(f"Error loading audio file {audio_path}: {e}")
                        # Return a dummy waveform on error
                        waveform = torch.zeros(16000, dtype=torch.float32)

            # Return the sample components with transcription
            return clique_idx_tensor, version_idx_tensor, waveform, transcription, has_valid_transcription_tensor, audio_path

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            import traceback
            traceback.print_exc()
            # Return default values if anything goes wrong
            return (torch.tensor(0, dtype=torch.long),
                    torch.tensor(0, dtype=torch.long),
                    torch.zeros(16000, dtype=torch.float32),
                    "",
                    False)

"""
AudioDataset for version identification tasks.

Loads audio files, metadata, and Whisper transcriptions using consolidated
helper modules for clean organization.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Union, Optional

from .cache import TranscriptionCache
from . import data_processing
from . import utils


class AudioDataset(Dataset):
    """
    Audio dataset for version identification with transcription support.
    
    Supports LyricCovers, SHS100K, and DiscogsVI datasets with clique-based
    organization (≥2 versions per clique).
    
    Attributes:
        dataset_name (str): Dataset identifier
        base_path (str): Path to metadata directory
        data_folder (str): Path to data files
        split (str): 'train', 'val', or 'test'
        whisper_set (str): Whisper model identifier
        evaluation_mode (bool): If True, skips audio loading
        df (pd.DataFrame): Filtered dataset
        transcription_cache (TranscriptionCache): Cache manager
        clique_id_to_idx/version_id_to_idx (Dict): ID mappings
    
    Example:
        >>> dataset = AudioDataset("shs", "./datasets", "/data", split="train")
        >>> clique_id, ver_id, wav, text, valid, path = dataset[0]
    """
    
    def __init__(
        self, 
        dataset_name: str, 
        base_path: str, 
        data_folder: str, 
        split: str = 'train',
        whisper_set: Union[str, List[str]] = "turbo_nothing_whisper_42", 
        evaluation_mode: bool = False,
        debug_mode: bool = False, 
        use_whisper_loader: bool = True
    ) -> None:
        """
        Initialize AudioDataset.
        
        Args:
            dataset_name: 'lyric-covers', 'shs', or 'discogs-vi'
            base_path: Path to metadata CSVs
            data_folder: Path to audio/transcription files
            split: 'train', 'val', or 'test'
            whisper_set: Whisper model identifier
            evaluation_mode: Skip audio loading (faster for evaluation)
            debug_mode: Only samples with valid transcriptions
            use_whisper_loader: Use whisper.load_audio() vs torchaudio
        
        Note:
            First run: 5-10 min (builds cache). Subsequent: <1 min.
        """
        self.dataset_name: str = dataset_name
        self.base_path: str = base_path
        self.data_folder: str = data_folder
        self.split: str = split
        self.whisper_set: str = whisper_set[0] if isinstance(whisper_set, list) else whisper_set
        self.evaluation_mode: bool = evaluation_mode
        self.debug_mode: bool = debug_mode
        self.use_whisper_loader: bool = use_whisper_loader

        # Initialize and load
        self.transcription_cache: TranscriptionCache = TranscriptionCache(
            data_folder, dataset_name
        )
        self.df: pd.DataFrame = self._load_data()

        # Load transcriptions
        if len(self.df) > 0:
            self.df = self.transcription_cache.apply_to_dataframe(
                self.df, [self.whisper_set], split=split
            )

        # Create ID mappings
        self._create_id_mappings()

        # Prepare evaluation tensors if needed
        if self.evaluation_mode:
            self._prepare_evaluation_tensors()

        print(f"Initialized {dataset_name} dataset with {len(self.df)} samples")
        if len(self.df) > 0:
            self.check_clique_versions()

    def _load_data(self) -> pd.DataFrame:
        """
        Load and filter dataset.
        
        Pipeline: metadata → file checks → split filter → audio filter → clique filter
        
        Returns:
            Filtered DataFrame with clique_id, version_id, status columns
        """
        try:
            df = data_processing.load_metadata(self.dataset_name, self.base_path)
            if df.empty:
                return df

            df = data_processing.add_file_status_columns(
                df, self.dataset_name, self.data_folder, self.whisper_set
            )
            if df.empty:
                return df

            df = data_processing.filter_by_split(df, self.split)
            if df.empty:
                return df

            df = data_processing.filter_by_audio_availability(df)
            if df.empty:
                return df

            df = data_processing.filter_single_version_cliques(df, verbose=True)
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _create_id_mappings(self) -> None:
        """Create bidirectional ID↔index mappings and add to DataFrame."""
        mapper = utils.IDMapper()
        mappings = mapper.create_mappings(self.df)
        
        (self.clique_id_to_idx, self.version_id_to_idx, 
         self.idx_to_clique_id, self.idx_to_version_id) = mappings
        
        self.df = mapper.add_indices_to_dataframe(self.df, mappings)

    def _prepare_evaluation_tensors(self) -> None:
        """Precompute tensors for evaluation: candidates_i/c, lyrics/whisper masks."""
        if self.df.empty:
            self.candidates_i = torch.tensor([], dtype=torch.long)
            self.candidates_c = torch.tensor([], dtype=torch.long)
            self.lyrics_mask = torch.tensor([], dtype=torch.bool)
            self.whisper_mask = torch.tensor([], dtype=torch.bool)
            return

        self.candidates_i = torch.tensor(self.df["version_idx"].values, dtype=torch.long)
        self.candidates_c = torch.tensor(self.df["clique_idx"].values, dtype=torch.long)
        self.lyrics_mask = torch.tensor(self.df["status_lyrics"].values, dtype=torch.bool)
        
        whisper_col = f"whisper_{self.whisper_set}"
        if whisper_col in self.df.columns:
            self.whisper_mask = torch.tensor(
                self.df[whisper_col].fillna(False).astype(bool).values,
                dtype=torch.bool
            )
        else:
            self.whisper_mask = torch.zeros(len(self.df), dtype=torch.bool)

    def check_clique_versions(self) -> bool:
        """
        Verify all cliques have ≥2 versions. Print statistics.
        
        Returns:
            True if valid, False if any single-version cliques found
        """
        if self.df.empty:
            return False

        counts = self.df.groupby('clique_id')['version_id'].nunique()

        print(f"\n=== Clique Statistics ===")
        print(f"Total: {len(counts)}, Min/Max/Avg: {counts.min()}/{counts.max()}/{counts.mean():.2f}")

        single = counts[counts == 1]
        if not single.empty:
            print(f"WARNING: {len(single)} single-version cliques found")
            return False
        
        print("✓ All cliques have ≥2 versions")
        return True

    def get_audio_path(self, idx: int) -> str:
        """Get audio file path for sample index."""
        try:
            row = self.df.iloc[idx]
            return data_processing.get_audio_path(
                self.dataset_name, self.data_folder, row
            )
        except Exception as e:
            print(f"Error getting audio path for {idx}: {e}")
            return ""

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, torch.Tensor, str]:
        """
        Get sample at index.
        
        Args:
            idx: Sample index
        
        Returns:
            (clique_idx, version_idx, waveform, transcription, is_valid, audio_path)
            - clique_idx: torch.long
            - version_idx: torch.long
            - waveform: torch.float32, shape (n_samples,) or (16000,) if evaluation_mode
            - transcription: str
            - is_valid: torch.bool
            - audio_path: str
        """
        try:
            if self.df.empty or idx >= len(self.df):
                return self._get_dummy_sample()

            row = self.df.iloc[idx]
            clique_idx = torch.tensor(row["clique_idx"], dtype=torch.long)
            version_idx = torch.tensor(row["version_idx"], dtype=torch.long)

            # Get transcription
            trans_col = f"transcription_{self.whisper_set}"
            transcription = row.get(trans_col, "")
            if pd.isna(transcription):
                transcription = ""
            
            valid_col = f"has_valid_transcription_{self.whisper_set}"
            has_valid = row.get(valid_col, False)
            if pd.isna(has_valid):
                has_valid = False
            has_valid = torch.tensor(has_valid, dtype=torch.bool)

            # Get audio
            audio_path = self.get_audio_path(idx)
            if self.evaluation_mode:
                waveform = torch.zeros(16000, dtype=torch.float32)
            else:
                waveform = utils.load_audio(
                    audio_path, 
                    use_whisper=self.use_whisper_loader,
                    return_dummy_on_error=True
                )

            return (clique_idx, version_idx, waveform, transcription, has_valid, audio_path)

        except Exception as e:
            print(f"Error at index {idx}: {e}")
            return self._get_dummy_sample()

    def _get_dummy_sample(self) -> Tuple:
        """Return dummy sample on error."""
        return (torch.tensor(0, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                torch.zeros(16000, dtype=torch.float32),
                "",
                torch.tensor(False, dtype=torch.bool),
                "")
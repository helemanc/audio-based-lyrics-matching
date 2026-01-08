"""
Utility functions for audio dataset.

Includes ID mapping and audio loading helpers.
"""

import torch
import whisper
import torchaudio
import os
import pandas as pd
from typing import Dict, Tuple


# ============================================================================
# ID MAPPING
# ============================================================================

class IDMapper:
    """Manages ID to index mappings."""
    
    @staticmethod
    def create_mappings(df: pd.DataFrame) -> Tuple[Dict, Dict, Dict, Dict]:
        """Create bidirectional ID mappings."""
        try:
            if df.empty:
                return {}, {}, {}, {}

            unique_cliques = sorted(df["clique_id"].unique())
            unique_versions = sorted(df["version_id"].unique())

            clique_id_to_idx = {cid: idx for idx, cid in enumerate(unique_cliques)}
            version_id_to_idx = {vid: idx for idx, vid in enumerate(unique_versions)}
            idx_to_clique_id = {idx: cid for cid, idx in clique_id_to_idx.items()}
            idx_to_version_id = {idx: vid for vid, idx in version_id_to_idx.items()}

            return (clique_id_to_idx, version_id_to_idx, 
                   idx_to_clique_id, idx_to_version_id)
        except Exception as e:
            print(f"Error creating mappings: {e}")
            return {}, {}, {}, {}
    
    @staticmethod
    def add_indices_to_dataframe(df: pd.DataFrame, 
                                 mappings: Tuple[Dict, Dict, Dict, Dict]) -> pd.DataFrame:
        """Add index columns to dataframe."""
        try:
            if df.empty:
                return df
            clique_id_to_idx, version_id_to_idx, _, _ = mappings
            df["clique_idx"] = df["clique_id"].map(clique_id_to_idx)
            df["version_idx"] = df["version_id"].map(version_id_to_idx)
            return df
        except Exception as e:
            print(f"Error adding indices: {e}")
            return df


# ============================================================================
# AUDIO LOADING
# ============================================================================

def load_audio(audio_path: str, use_whisper: bool = True, 
              return_dummy_on_error: bool = True) -> torch.Tensor:
    """
    Load audio file.
    
    Args:
        audio_path: Path to audio file
        use_whisper: If True, uses Whisper's loader; else torchaudio
        return_dummy_on_error: If True, returns silence on error
    
    Returns:
        Audio waveform tensor, shape (n_samples,)
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio not found: {audio_path}")
        
        if use_whisper:
            waveform_np = whisper.load_audio(audio_path)
            return torch.tensor(waveform_np, dtype=torch.float32)
        else:
            waveform, _ = torchaudio.load(audio_path)
            return waveform.squeeze(0)
    
    except Exception as e:
        if return_dummy_on_error:
            if audio_path:
                print(f"Warning: Error loading {audio_path}: {e}")
            return torch.zeros(16000, dtype=torch.float32)
        else:
            raise
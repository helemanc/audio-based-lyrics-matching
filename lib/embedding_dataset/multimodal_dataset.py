"""
Multimodal embedding datasets for version identification.

Provides datasets combining multiple embedding types:
- WEALY+CLEWS: Concatenated lyrics embeddings + audio embeddings
- Whisper+CLEWS: Transcription embeddings + audio embeddings
"""

import torch
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import os

from .base_dataset import EmbeddingDataset
from .utils import create_deterministic_song_id
from . import data_processing


class MultimodalEmbeddingDataset_WEALYCLEWS(EmbeddingDataset):
    """
    Dataset combining WEALY concatenated embeddings and CLEWS audio embeddings.
    
    Returns for each version:
        - wealy_concat: Dict with embeddings, chunk_info
        - full_clews: Full CLEWS embedding (116, 2048)
        - avg_clews: Averaged CLEWS embedding (2048,)
        - clews_mask: Mask for valid CLEWS positions
    
    Example:
        >>> dataset = MultimodalEmbeddingDataset_WEALYCLEWS(conf, split='train')
        >>> sample = dataset[0]
        >>> clique_id, ver_id, multimodal_emb = sample[0], sample[1], sample[2]
        >>> wealy = multimodal_emb['wealy']
        >>> clews = multimodal_emb['full_clews']
    """
    
    def __init__(self, conf: Any, split: str, augment: bool = False, 
                 verbose: bool = False) -> None:
        """
        Initialize WEALY+CLEWS multimodal dataset.
        
        Args:
            conf: Configuration object
            split: 'train', 'val', or 'test'
            augment: Apply augmentation
            verbose: Print progress
        """
        super().__init__(
            conf=conf,
            split=split,
            augment=augment,
            embedding_type="multimodal_wealy_clews",
            embedding_format="all",
            verbose=verbose
        )
        self.ensure_version_alignment()
    
    def _get_required_embedding_filename(self) -> str:
        """Return special marker for multimodal verification."""
        return "MULTIMODAL_WEALY_CLEWS_CONCAT"
    
    def verify_embeddings_exist(self) -> bool:
        """
        Verify WEALY concat, full CLEWS, avg CLEWS, and masks exist.
        
        Returns:
            True if all required embeddings exist
        """
        if self.verbose:
            print("Verifying WEALY concat + CLEWS embeddings...")
        
        hidden_states_path = Path(self.conf.path.hidden_states)
        required_files = [
            "hs_wealy_concat.pt",
            "hs_clews.pt",
            "hs_clews_avg.pt",
            "hs_clews_mask.pt"
        ]
        
        all_good = True
        for split_name in ["train", "val", "test"]:
            missing = []
            for clique_id, versions in self.splitdict[split_name].items():
                for version in versions:
                    # Check all required files exist
                    all_exist = all(
                        data_processing.check_embedding_exists(
                            self.dataset_name, str(hidden_states_path), version, fname
                        )
                        for fname in required_files
                    )
                    
                    if not all_exist:
                        missing.append(version)
            
            if missing:
                all_good = False
                if self.verbose:
                    print(f"  {split_name}: {len(missing)} missing embeddings")
            else:
                total = sum(len(v) for v in self.splitdict[split_name].values())
                if self.verbose:
                    print(f"  {split_name}: ✓ All {total} versions have embeddings")
        
        return all_good
    
    def ensure_version_alignment(self) -> None:
        """Build version alignment with deterministic IDs."""
        aligned_data = []
        for version_key in self.versions:
            if version_key in self.info:
                clique_id = self.info[version_key]['clique']
                version_str = version_key.split('-', 1)[1] if '-' in version_key else version_key
                det_id = create_deterministic_song_id(str(clique_id), str(version_str))
                aligned_data.append((det_id, version_key))
        
        aligned_data.sort(key=lambda x: x[0])
        self.versions = [version_key for _, version_key in aligned_data]
        
        for det_id, version_key in aligned_data:
            self.info[version_key]['id'] = det_id
    
    def _get_version_folder(self, version: str) -> Path:
        """Get folder path for version's embeddings."""
        hidden_states_path = Path(self.conf.path.hidden_states)
        
        if self.dataset_name == 'shs':
            set_id, ver_id = version.split('-')
            set_id_int = int(set_id)
            if set_id_int <= 9:
                folder_name = f"{set_id}-"
            elif set_id_int <= 99:
                folder_name = set_id
            else:
                folder_name = set_id[:2]
            return hidden_states_path / folder_name / version
        
        elif self.dataset_name == 'lyric-covers':
            return hidden_states_path / version
        
        elif self.dataset_name == 'discogs-vi':
            return hidden_states_path / version.replace('/', os.sep)
        
        return hidden_states_path
    
    def load_multimodal_embeddings(
        self, version: str
    ) -> Tuple[Dict, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load WEALY concat + full CLEWS + avg CLEWS + mask.
        
        Args:
            version: Version identifier
        
        Returns:
            Tuple of (wealy_concat_dict, full_clews, avg_clews, clews_mask)
            Falls back to dummy embeddings if files missing
        """
        version_folder = self._get_version_folder(version)
        
        # Load WEALY concatenated
        wealy_path = version_folder / "hs_wealy_concat.pt"
        try:
            wealy_data = torch.load(wealy_path, map_location='cpu')
            
            if isinstance(wealy_data, dict) and 'embeddings' in wealy_data:
                wealy_concat = wealy_data
                if wealy_concat['embeddings'].dtype == torch.float16:
                    wealy_concat['embeddings'] = wealy_concat['embeddings'].float()
            else:
                # Legacy format
                if wealy_data.dtype == torch.float16:
                    wealy_data = wealy_data.float()
                
                wealy_concat = {
                    'embeddings': wealy_data,
                    'chunk_info': {'total_chunks': wealy_data.shape[0] if wealy_data.dim() > 1 else 1},
                    'extraction_method': 'legacy_format'
                }
        except Exception as e:
            wealy_concat = {
                'embeddings': torch.zeros(10, self.conf.model.zdim),
                'chunk_info': {'total_chunks': 10},
                'extraction_method': 'dummy'
            }
            if self.verbose:
                print(f"Using dummy WEALY for {version}: {e}")
        
        # Load full CLEWS
        clews_path = version_folder / "hs_clews.pt"
        try:
            full_clews = torch.load(clews_path, map_location='cpu')
            if full_clews.dtype == torch.float16:
                full_clews = full_clews.float()
        except:
            full_clews = torch.zeros(116, 2048)
            if self.verbose:
                print(f"Using dummy full CLEWS for {version}")
        
        # Load avg CLEWS
        avg_path = version_folder / "hs_clews_avg.pt"
        try:
            avg_clews = torch.load(avg_path, map_location='cpu')
            if avg_clews.dtype == torch.float16:
                avg_clews = avg_clews.float()
        except:
            avg_clews = torch.zeros(2048)
            if self.verbose:
                print(f"Using dummy avg CLEWS for {version}")
        
        # Load mask
        mask_path = version_folder / "hs_clews_mask.pt"
        try:
            clews_mask = torch.load(mask_path, map_location='cpu')
        except:
            clews_mask = torch.ones(116, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy CLEWS mask for {version}")
        
        return wealy_concat, full_clews, avg_clews, clews_mask
    
    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get multimodal sample.
        
        Args:
            idx: Sample index
        
        Returns:
            List: [clique_id, ver_id_1, multimodal_dict_1, ver_id_2, multimodal_dict_2, ...]
            Each multimodal_dict contains: wealy, full_clews, avg_clews, clews_mask
        """
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions
        otherversions = [
            v for v in self.clique[cl]
            if v != v1 or torch.rand(1).item() < getattr(self, 'p_samesong', 0.0)
        ]
        
        if getattr(self, 'augment', False):
            perm = torch.randperm(len(otherversions)).tolist()
            otherversions = [otherversions[k] for k in perm]
        
        # Sample versions
        n_per_class = getattr(self, 'n_per_class', 2)
        v_n = [v1]
        i_n = [i1]
        for k in range(n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i_n.append(self.info[v]["id"])
            v_n.append(v)
        
        # Load embeddings
        output = [icl]
        for i, v in zip(i_n, v_n):
            wealy, full_clews, avg_clews, clews_mask = self.load_multimodal_embeddings(v)
            
            multimodal_emb = {
                'wealy': wealy,
                'full_clews': full_clews,
                'avg_clews': avg_clews,
                'clews_mask': clews_mask,
                'song_id': v,
                'class_id': icl
            }
            output.extend([i, multimodal_emb])
        
        return output


class MultimodalEmbeddingDataset_WHISPERCLEWS(EmbeddingDataset):
    """
    Dataset combining Whisper transcription embeddings and CLEWS audio embeddings.
    
    Returns for each version:
        - whisper: Whisper hs_last_seq embedding (seq_len, 1280)
        - whisper_mask: Mask for Whisper sequence
        - full_clews: Full CLEWS embedding (16, 2048)
        - avg_clews: Averaged CLEWS embedding (2048,)
        - clews_mask: Mask for valid CLEWS positions
    
    Example:
        >>> dataset = MultimodalEmbeddingDataset_WHISPERCLEWS(conf, split='train')
        >>> sample = dataset[0]
        >>> multimodal_emb = sample[2]  # First version's embeddings
        >>> whisper = multimodal_emb['whisper']
        >>> clews = multimodal_emb['full_clews']
    """
    
    def __init__(self, conf: Any, split: str, augment: bool = False,
                 verbose: bool = False) -> None:
        """
        Initialize Whisper+CLEWS multimodal dataset.
        
        Args:
            conf: Configuration object
            split: 'train', 'val', or 'test'
            augment: Apply augmentation
            verbose: Print progress
        """
        super().__init__(
            conf=conf,
            split=split,
            augment=augment,
            embedding_type="multimodal_whisper_clews",
            embedding_format="all",
            verbose=verbose
        )
        self.ensure_version_alignment()
    
    def _get_required_embedding_filename(self) -> str:
        """Return special marker for multimodal verification."""
        return "MULTIMODAL_WHISPER_CLEWS_ALL"
    
    def verify_embeddings_exist(self) -> bool:
        """
        Verify Whisper, full CLEWS, avg CLEWS, and masks exist.
        
        Returns:
            True if all required embeddings exist
        """
        if self.verbose:
            print("Verifying Whisper + CLEWS embeddings...")
        
        hidden_states_path = Path(self.conf.path.hidden_states)
        required_files = [
            "hs_last_seq.pt",
            "hs_clews.pt",
            "hs_clews_avg.pt",
            "hs_clews_mask.pt"
        ]
        
        all_good = True
        for split_name in ["train", "val", "test"]:
            missing = []
            for clique_id, versions in self.splitdict[split_name].items():
                for version in versions:
                    all_exist = all(
                        data_processing.check_embedding_exists(
                            self.dataset_name, str(hidden_states_path), version, fname
                        )
                        for fname in required_files
                    )
                    
                    if not all_exist:
                        missing.append(version)
            
            if missing:
                all_good = False
                if self.verbose:
                    print(f"  {split_name}: {len(missing)} missing embeddings")
            else:
                total = sum(len(v) for v in self.splitdict[split_name].values())
                if self.verbose:
                    print(f"  {split_name}: ✓ All {total} versions have embeddings")
        
        return all_good
    
    def ensure_version_alignment(self) -> None:
        """Build version alignment with deterministic IDs."""
        aligned_data = []
        for version_key in self.versions:
            if version_key in self.info:
                clique_id = self.info[version_key]['clique']
                version_str = version_key.split('-', 1)[1] if '-' in version_key else version_key
                det_id = create_deterministic_song_id(str(clique_id), str(version_str))
                aligned_data.append((det_id, version_key))
        
        aligned_data.sort(key=lambda x: x[0])
        self.versions = [version_key for _, version_key in aligned_data]
        
        for det_id, version_key in aligned_data:
            self.info[version_key]['id'] = det_id
    
    def _get_version_folder(self, version: str) -> Path:
        """Get folder path for version's embeddings."""
        hidden_states_path = Path(self.conf.path.hidden_states)
        
        if self.dataset_name == 'shs':
            set_id, ver_id = version.split('-')
            set_id_int = int(set_id)
            if set_id_int <= 9:
                folder_name = f"{set_id}-"
            elif set_id_int <= 99:
                folder_name = set_id
            else:
                folder_name = set_id[:2]
            return hidden_states_path / folder_name / version
        
        elif self.dataset_name == 'lyric-covers':
            return hidden_states_path / version
        
        elif self.dataset_name == 'discogs-vi':
            return hidden_states_path / version.replace('/', os.sep)
        
        return hidden_states_path
    
    def load_multimodal_embeddings(
        self, version: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load Whisper + full CLEWS + avg CLEWS + masks.
        
        Args:
            version: Version identifier
        
        Returns:
            Tuple of (whisper, whisper_mask, full_clews, avg_clews, clews_mask)
            Falls back to dummy embeddings if files missing
        """
        version_folder = self._get_version_folder(version)
        
        # Load Whisper
        whisper_path = version_folder / "hs_last_seq.pt"
        try:
            whisper = torch.load(whisper_path, map_location='cpu')
            if whisper.dtype == torch.float16:
                whisper = whisper.float()
            whisper_mask = torch.ones(whisper.shape[0], dtype=torch.bool)
        except:
            whisper = torch.zeros(15, 1280)
            whisper_mask = torch.ones(15, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy Whisper for {version}")
        
        # Load full CLEWS
        clews_path = version_folder / "hs_clews.pt"
        try:
            full_clews = torch.load(clews_path, map_location='cpu')
            if full_clews.dtype == torch.float16:
                full_clews = full_clews.float()
        except:
            full_clews = torch.zeros(16, 2048)
            if self.verbose:
                print(f"Using dummy full CLEWS for {version}")
        
        # Load avg CLEWS
        avg_path = version_folder / "hs_clews_avg.pt"
        try:
            avg_clews = torch.load(avg_path, map_location='cpu')
            if avg_clews.dtype == torch.float16:
                avg_clews = avg_clews.float()
        except:
            avg_clews = torch.zeros(2048)
            if self.verbose:
                print(f"Using dummy avg CLEWS for {version}")
        
        # Load mask
        mask_path = version_folder / "hs_clews_mask.pt"
        try:
            clews_mask = torch.load(mask_path, map_location='cpu')
        except:
            clews_mask = torch.ones(16, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy CLEWS mask for {version}")
        
        return whisper, whisper_mask, full_clews, avg_clews, clews_mask
    
    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get multimodal sample.
        
        Args:
            idx: Sample index
        
        Returns:
            List: [clique_id, ver_id_1, multimodal_dict_1, ver_id_2, multimodal_dict_2, ...]
            Each multimodal_dict contains: whisper, whisper_mask, full_clews, avg_clews, clews_mask
        """
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions
        otherversions = [
            v for v in self.clique[cl]
            if v != v1 or torch.rand(1).item() < getattr(self, 'p_samesong', 0.0)
        ]
        
        if getattr(self, 'augment', False):
            perm = torch.randperm(len(otherversions)).tolist()
            otherversions = [otherversions[k] for k in perm]
        
        # Sample versions
        n_per_class = getattr(self, 'n_per_class', 2)
        v_n = [v1]
        i_n = [i1]
        for k in range(n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i_n.append(self.info[v]["id"])
            v_n.append(v)
        
        # Load embeddings
        output = [icl]
        for i, v in zip(i_n, v_n):
            whisper, whisper_mask, full_clews, avg_clews, clews_mask = self.load_multimodal_embeddings(v)
            
            multimodal_emb = {
                'whisper': whisper,
                'whisper_mask': whisper_mask,
                'full_clews': full_clews,
                'avg_clews': avg_clews,
                'clews_mask': clews_mask,
                'song_id': v,
                'class_id': icl
            }
            output.extend([i, multimodal_emb])
        
        return output
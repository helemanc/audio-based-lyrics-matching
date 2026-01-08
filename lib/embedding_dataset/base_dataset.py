"""
Base embedding dataset for single-modal embeddings.

Simplified design using consolidated helper modules for metadata loading,
filtering, and validation.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import random

from . import data_processing
from . import utils
from .cache_manager import CacheManager


class EmbeddingDataset(Dataset):
    """
    Dataset for loading pre-extracted embeddings.
    
    Supports single-modal embeddings (SBERT, CLEWS, Whisper, WEALY) for
    version identification tasks with clique-based organization.
    
    Pipeline:
        1. Try cache load
        2. Load metadata from CSV
        3. Filter by audio availability
        4. Filter single-version cliques
        5. Remove overlapping cliques
        6. Filter/verify embeddings
        7. Rebuild with deterministic IDs
        8. Save cache
    
    Attributes:
        dataset_name (str): Dataset identifier
        embedding_type (str): Type of embedding
        embedding_format (str): Format of embedding
        split (str): Current split ('train', 'val', 'test')
        info (Dict): Version metadata {version_id: {metadata}}
        splitdict (Dict): {split: {clique_id: [version_ids]}}
        clique (Dict): Current split's cliques
        versions (List): Current split's versions
        clique2id (Dict): Clique to integer mapping
    
    Example:
        >>> dataset = EmbeddingDataset(conf, split='train')
        >>> clique_id, versions, embeddings = dataset[0]
    """
    
    def __init__(
        self,
        conf: Any,
        split: str,
        embedding_type: Optional[str] = None,
        embedding_format: Optional[str] = None,
        augment: bool = False,
        fullsongs: bool = False,
        n_per_class: int = 2,
        p_samesong: float = 0.0,
        verbose: bool = True,
        debug: bool = False,
        return_paths: bool = False
    ) -> None:
        """
        Initialize embedding dataset.
        
        Args:
            conf: Configuration object with paths and settings
            split: 'train', 'val', or 'test'
            embedding_type: Type of embedding ('encoder', 'sbert', 'clews', etc.)
                If None, uses conf.data.embedding_type
            embedding_format: Format ('concat', 'all', etc.)
                If None, uses conf.data.embedding_format
            augment: Apply augmentation (random permutation of versions)
            fullsongs: Use full songs (unused, for compatibility)
            n_per_class: Number of versions to sample per clique
            p_samesong: Probability of sampling same song twice
            verbose: Print progress
            debug: Enable debug mode (filters to available embeddings)
            return_paths: Return file paths with samples
        
        Note:
            First run: ~10-30 minutes (builds cache)
            Cached runs: <1 minute
        """
        self.conf = conf
        self.split: str = split
        self.augment: bool = augment
        self.fullsongs: bool = fullsongs
        self.n_per_class: int = n_per_class
        self.p_samesong: float = p_samesong
        self.verbose: bool = verbose
        self.debug: bool = debug
        self.return_paths: bool = return_paths
        
        # Set embedding config
        self.embedding_type: str = embedding_type or getattr(conf.data, 'embedding_type', 'encoder')
        self.embedding_format: str = embedding_format or getattr(conf.data, 'embedding_format', 'concat')
        
        # Get dataset name and nickname
        self.dataset_name: str = getattr(conf.data, 'dataset_name', 'shs')
        self.dataset_nickname: str = self._get_dataset_nickname()
        
        if self.verbose:
            print(f"Dataset: {self.dataset_name} ({self.dataset_nickname})")
            print(f"Embedding: {self.embedding_type}, format: {self.embedding_format}")
        
        # Initialize data structures
        self.info: Dict[str, Dict] = {}
        self.splitdict: Dict[str, Dict[str, List[str]]] = {}
        self.clique2id: Dict[str, int] = {}
        
        # Initialize cache manager
        self.cache_manager = CacheManager(self, verbose)
        
        # Build dataset
        self.info, self.splitdict, self.clique2id = self._build_dataset()
        
        # Set current split's clique
        self.clique: Dict[str, List[str]] = self.splitdict[split]
        
        # Filter info to current split
        self._filter_info_to_current_split()
        
        # Ensure consistency for Discogs-VI
        if self.dataset_name == "discogs-vi":
            self._ensure_consistency()
        
        # Create split-specific clique mapping
        self._create_clique_id_mapping()
        
        # Build versions list
        self.versions: List[str] = []
        for vers in self.clique.values():
            self.versions.extend(vers)
        
        if self.verbose:
            print(f"\nFinal validation:")
            self._validate_data_structures()
    
    def _get_dataset_nickname(self) -> str:
        """Map dataset names to nicknames for file paths."""
        mapping = {
            'shs': 'shs',
            'lyric-covers': 'lyc',
            'discogs-vi': 'dvi'
        }
        return mapping.get(self.dataset_name, self.dataset_name)
    
    def _get_required_embedding_filename(self) -> Optional[str]:
        """
        Get required embedding filename based on type and format.
        
        Returns:
            Embedding filename or None if unknown combination
        """
        if self.embedding_type == "encoder":
            if self.embedding_format == "concat":
                return "x_concat.pt"
            elif self.embedding_format == "all":
                return "x_all.pt"
        elif self.embedding_type == "hidden_states":
            if self.embedding_format == "all":
                return "hs_all.pt"
        elif self.embedding_type == "last_hidden_states":
            if self.embedding_format == "concat":
                return "hs_last_seq.pt"
            elif self.embedding_format == "all":
                return "hs_last_all.pt"
        elif self.embedding_type == "last_hidden_states_en":
            if self.embedding_format == "concat":
                return "hs_last_seq_en.pt"
            elif self.embedding_format == "all":
                return "hs_last_all_en.pt"
        elif self.embedding_type == "sbert":
            return "hs_sbert.pt"
        elif self.embedding_type == "clews":
            return "hs_clews.pt"
        elif self.embedding_type == "multimodal":
            return "MULTIMODAL"
        
        return None
    
    def _build_dataset(self) -> Tuple[Dict, Dict, Dict]:
        """
        Build dataset through complete pipeline.
        
        Pipeline:
            1. Try cache
            2. Load metadata
            3. Filter by audio
            4. Filter single-version cliques
            5. Remove overlaps
            6. Filter/verify embeddings (if debug)
            7. Update info
            8. Rebuild with deterministic IDs
            9. Verify embeddings
            10. Save cache
        
        Returns:
            Tuple of (info, splitdict, clique2id)
        """
        # Try cache
        cached = self.cache_manager._load_from_cache()
        if cached:
            return self.info, self.splitdict, self.clique2id
        
        # Load metadata
        df = data_processing.load_metadata(self.dataset_name, self.conf, self.verbose)
        
        # Create ID mappings
        (clique_id_to_idx, version_id_to_idx, 
         idx_to_clique_id, idx_to_version_id) = utils.create_id_mappings(df, self.verbose)
        
        df = utils.add_indices_to_dataframe(df, clique_id_to_idx, version_id_to_idx)
        
        # Build initial info and splitdict
        self.info, self.splitdict = data_processing.build_info_and_splitdict(
            df, self.dataset_name, self.verbose
        )
        
        # Filter by audio
        self.splitdict = data_processing.filter_by_audio(
            self.splitdict, self.dataset_name, self.conf.path.data, self.verbose
        )
        
        # Filter single-version cliques
        self.splitdict = data_processing.filter_single_version_cliques(
            self.splitdict, self.verbose
        )
        
        # Remove overlaps
        self.splitdict = data_processing.remove_overlapping_cliques(
            self.splitdict, self.verbose
        )
        
        # Debug mode: filter to available embeddings
        if self.debug:
            required_filename = self._get_required_embedding_filename()
            if required_filename and required_filename != "MULTIMODAL":
                self.splitdict = data_processing.filter_by_embeddings(
                    self.splitdict,
                    self.dataset_name,
                    self.conf.path.hidden_states,
                    required_filename,
                    self.verbose
                )
        
        # Update info dict
        self._update_info_after_filtering()
        
        # Rebuild with deterministic IDs
        self.info = utils.rebuild_info_with_deterministic_ids(
            self.info, self.dataset_name, self.verbose
        )
        
        # Verify embeddings
        required_filename = self._get_required_embedding_filename()
        if required_filename and required_filename != "MULTIMODAL":
            embeddings_ok = data_processing.verify_embeddings(
                self.splitdict,
                self.dataset_name,
                self.conf.path.hidden_states,
                required_filename,
                self.verbose
            )
            
            # Save cache only if embeddings OK
            if embeddings_ok:
                self.cache_manager._save_to_cache()
            elif self.verbose:
                print("⚠ Not saving cache (missing embeddings)")
        
        # Create global clique mapping
        self._create_global_clique_mapping()
        
        # Print stats
        self._print_statistics()
        
        return self.info, self.splitdict, self.clique2id
    
    def _update_info_after_filtering(self) -> None:
        """Remove filtered versions from info dict."""
        if self.verbose:
            print("Updating info dict...")
        
        all_remaining = set()
        for split in ["train", "val", "test"]:
            for versions in self.splitdict[split].values():
                all_remaining.update(versions)
        
        original_count = len(self.info)
        self.info = {k: v for k, v in self.info.items() if k in all_remaining}
        
        if self.verbose:
            removed = original_count - len(self.info)
            print(f"  Info: {original_count} → {len(self.info)} (removed {removed})")
    
    def _create_global_clique_mapping(self) -> None:
        """Create global clique ID mapping across all splits."""
        if self.verbose:
            print("Creating global clique mapping...")
        
        self.clique2id = {}
        offset = 0
        
        for split_name in ["train", "val", "test"]:
            for i, clique_id in enumerate(self.splitdict[split_name].keys()):
                self.clique2id[clique_id] = offset + i
            offset += len(self.splitdict[split_name])
    
    def _filter_info_to_current_split(self) -> None:
        """Filter info to only versions in current split's cliques."""
        if self.verbose:
            print(f"Filtering info to {self.split} split...")
        
        current_cliques = set(self.clique.keys())
        original_count = len(self.info)
        
        self.info = {
            k: v for k, v in self.info.items()
            if v['clique'] in current_cliques
        }
        
        if self.verbose:
            print(f"  Info: {original_count} → {len(self.info)}")
    
    def _ensure_consistency(self) -> None:
        """Ensure perfect consistency for Discogs-VI."""
        if self.verbose:
            print("Ensuring consistency...")
        
        validator = utils.DataValidator(
            self.info, self.clique, self.versions, self.split, self.verbose
        )
        
        if not validator.validate():
            self.info, self.clique, self.versions = validator.ensure_consistency()
    
    def _create_clique_id_mapping(self) -> None:
        """Create clique ID mapping for current split with offset."""
        if self.split == "train":
            offset = 0
        elif self.split == "val":
            offset = len(self.splitdict["train"])
        else:
            offset = len(self.splitdict["train"]) + len(self.splitdict["val"])
        
        self.clique2id = {}
        for i, clique_id in enumerate(self.clique.keys()):
            self.clique2id[clique_id] = offset + i
    
    def _validate_data_structures(self) -> None:
        """Validate consistency between info, clique, and versions."""
        validator = utils.DataValidator(
            self.info, self.clique, self.versions, self.split, self.verbose
        )
        validator.validate()
    
    def _print_statistics(self) -> None:
        """Print dataset statistics for all splits."""
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        total_cliques = 0
        total_versions = 0
        
        for split_name in ["train", "val", "test"]:
            clique_count = len(self.splitdict[split_name])
            version_count = sum(len(v) for v in self.splitdict[split_name].values())
            
            print(f"{split_name.upper():>5}: {clique_count:>5} cliques, "
                  f"{version_count:>6} versions")
            
            total_cliques += clique_count
            total_versions += version_count
        
        print("-" * 50)
        print(f"TOTAL: {total_cliques:>5} cliques, {total_versions:>6} versions")
    
    def get_embedding_path(self, version: str) -> Optional[Path]:
        """
        Get path to embedding file for version.
        
        Args:
            version: Version identifier
        
        Returns:
            Path to embedding file or None if not found
        """
        required_filename = self._get_required_embedding_filename()
        if not required_filename or required_filename == "MULTIMODAL":
            return None
        
        return data_processing.get_embedding_path(
            self.dataset_name,
            self.conf.path.hidden_states,
            version,
            required_filename
        )
    
    def load_embedding(self, version: str) -> Optional[torch.Tensor]:
        """
        Load embedding for version.
        
        Args:
            version: Version identifier
        
        Returns:
            Embedding tensor or None if not found
        
        Note:
            Automatically converts float16 to float32
        """
        embedding_path = self.get_embedding_path(version)
        
        if not embedding_path:
            if self.verbose:
                print(f"Warning: Embedding not found for {version}")
            return None
        
        try:
            embedding = torch.load(embedding_path, map_location='cpu')
            
            # Convert to float32 if needed
            if isinstance(embedding, torch.Tensor) and embedding.dtype == torch.float16:
                embedding = embedding.float()
            elif isinstance(embedding, dict):
                embedding = {
                    k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.float16 else v)
                    for k, v in embedding.items()
                }
            
            # SBERT: ensure 2D
            if self.embedding_type == "sbert":
                if isinstance(embedding, torch.Tensor):
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)
                    return embedding
                else:
                    if self.verbose:
                        print(f"Warning: Expected tensor for SBERT, got {type(embedding)}")
                    return None
            
            return embedding
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading {embedding_path}: {e}")
            return None
    
    def __len__(self) -> int:
        """Return number of versions in current split."""
        return len(self.versions)
    
    def __getitem__(self, idx: int) -> List[Any]:
        """
        Get sample at index.
        
        Args:
            idx: Sample index
        
        Returns:
            List: [clique_id, version_id_1, embedding_1, version_id_2, embedding_2, ...]
            Length: 1 + (2 * n_per_class)
        
        Example:
            >>> sample = dataset[0]
            >>> clique_id = sample[0]
            >>> ver_id_1, emb_1 = sample[1], sample[2]
            >>> ver_id_2, emb_2 = sample[3], sample[4]
        """
        # Get anchor version
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions from same clique
        otherversions = []
        for v in self.clique[cl]:
            if v != v1 or random.random() < self.p_samesong:
                otherversions.append(v)
        
        # Apply augmentation (random permutation)
        if self.augment:
            perm = torch.randperm(len(otherversions)).tolist()
            otherversions = [otherversions[k] for k in perm]
        
        # Sample n_per_class versions
        v_n = [v1]
        i_n = [i1]
        for k in range(self.n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i = self.info[v]["id"]
            v_n.append(v)
            i_n.append(i)
        
        # Load embeddings
        output = [icl]
        for i, v in zip(i_n, v_n):
            embedding = self.load_embedding(v)
            output.extend([i, embedding])
        
        return output
"""
Utilities for embeddings datasets.

ID generation, mappings, and data validation.
"""

import hashlib
import pandas as pd
from typing import Dict, List, Set, Tuple


# ============================================================================
# DETERMINISTIC ID GENERATION
# ============================================================================

def create_deterministic_song_id(clique_str: str, version_str: str) -> int:
    """
    Create deterministic ID from clique and version strings.
    
    Uses MD5 hash to generate consistent integer ID across all sessions.
    
    Args:
        clique_str: Clique identifier
        version_str: Version identifier
    
    Returns:
        Positive 32-bit integer ID
    
    Example:
        >>> create_deterministic_song_id("123", "456")
        1234567890
    """
    combined = f"{clique_str}-{version_str}"
    hash_bytes = hashlib.md5(combined.encode('utf-8')).digest()
    return int.from_bytes(hash_bytes[:4], byteorder='big') & 0x7fffffff


# ============================================================================
# ID MAPPING
# ============================================================================

def create_id_mappings(df: pd.DataFrame, verbose: bool = True) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Create bidirectional ID mappings from dataframe.
    
    Args:
        df: DataFrame with clique_id and version_id columns
        verbose: Print progress
    
    Returns:
        Tuple of (clique_id_to_idx, version_id_to_idx, idx_to_clique_id, idx_to_version_id)
    """
    if df.empty:
        return {}, {}, {}, {}
    
    unique_cliques = sorted(df["clique_id"].unique())
    unique_versions = sorted(df["version_id"].unique())
    
    clique_id_to_idx = {cid: idx for idx, cid in enumerate(unique_cliques)}
    version_id_to_idx = {vid: idx for idx, vid in enumerate(unique_versions)}
    idx_to_clique_id = {idx: cid for cid, idx in clique_id_to_idx.items()}
    idx_to_version_id = {idx: vid for vid, idx in version_id_to_idx.items()}
    
    if verbose:
        print(f"Created mappings: {len(unique_cliques)} cliques, {len(unique_versions)} versions")
    
    return clique_id_to_idx, version_id_to_idx, idx_to_clique_id, idx_to_version_id


def add_indices_to_dataframe(
    df: pd.DataFrame,
    clique_id_to_idx: Dict,
    version_id_to_idx: Dict
) -> pd.DataFrame:
    """
    Add integer index columns to dataframe.
    
    Args:
        df: DataFrame with clique_id and version_id
        clique_id_to_idx: Clique ID to index mapping
        version_id_to_idx: Version ID to index mapping
    
    Returns:
        DataFrame with added clique_idx and version_idx columns
    """
    df["clique_idx"] = df["clique_id"].map(clique_id_to_idx)
    df["version_idx"] = df["version_id"].map(version_id_to_idx)
    return df


def rebuild_info_with_deterministic_ids(
    info: Dict[str, Dict],
    dataset_name: str,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Rebuild info dict with deterministic IDs.
    
    Args:
        info: Original info dict
        dataset_name: Dataset identifier
        verbose: Print progress
    
    Returns:
        Info dict with deterministic 'id' field
    """
    if verbose:
        print("Rebuilding info with deterministic IDs...")
    
    new_info = {}
    for version_key, meta in info.items():
        clique_str, version_str = extract_clique_version_for_hash(
            version_key, meta, dataset_name
        )
        det_id = create_deterministic_song_id(clique_str, version_str)
        
        new_meta = meta.copy()
        new_meta['id'] = det_id
        new_info[version_key] = new_meta
    
    if verbose:
        print(f"Rebuilt {len(new_info)} versions with deterministic IDs")
    
    return new_info


def extract_clique_version_for_hash(
    version_key: str,
    metadata: Dict,
    dataset_name: str
) -> Tuple[str, str]:
    """
    Extract (clique_str, version_str) for hash generation.
    
    Args:
        version_key: Version identifier
        metadata: Version metadata dict
        dataset_name: Dataset identifier
    
    Returns:
        Tuple of (clique_string, version_string)
    """
    if dataset_name == 'shs':
        if '-' not in version_key:
            raise ValueError(f"SHS version_key without '-': {version_key}")
        clique_str, version_str = version_key.split('-', 1)
        return str(clique_str), str(version_str)
    
    elif dataset_name == 'lyric-covers':
        clique_str = str(metadata.get('clique_id', metadata.get('clique')))
        version_str = str(metadata.get('version_id', metadata.get('version_key', version_key)))
        return clique_str, version_str
    
    elif dataset_name == 'discogs-vi':
        clique_str = str(metadata.get('clique_id', metadata.get('clique')))
        version_str = str(metadata.get('version_id', metadata.get('base_filename', 
                                      metadata.get('version_key', version_key))))
        version_str = version_str.replace('\\', '/')
        return clique_str, version_str
    
    clique_str = str(metadata.get('clique', ''))
    version_str = str(metadata.get('version_id', metadata.get('version_key', version_key)))
    return clique_str, version_str


# ============================================================================
# DATA VALIDATION
# ============================================================================

class DataValidator:
    """Validates consistency between info, splitdict, and versions."""
    
    def __init__(self, info: Dict, clique: Dict, versions: List, 
                 split: str = "", verbose: bool = False):
        self.info = info
        self.clique = clique
        self.versions = versions
        self.split = split
        self.verbose = verbose
    
    def validate(self) -> bool:
        """
        Validate data structures are consistent.
        
        Checks:
        1. All versions in clique are in info
        2. All versions in list are in info
        3. All cliques have ≥2 versions
        
        Returns:
            True if valid
        """
        if self.verbose:
            print(f"\n=== Validating {self.split} ===")
        
        # Check 1: Clique versions in info
        clique_versions = set()
        for vers in self.clique.values():
            clique_versions.update(vers)
        
        missing_in_info = clique_versions - set(self.info.keys())
        if missing_in_info:
            if self.verbose:
                print(f"ERROR: {len(missing_in_info)} versions in clique but not in info")
            return False
        
        # Check 2: List versions in info
        missing_versions = [v for v in self.versions if v not in self.info]
        if missing_versions:
            if self.verbose:
                print(f"ERROR: {len(missing_versions)} versions in list but not in info")
            return False
        
        # Check 3: Cliques have ≥2 versions
        single_cliques = {cid: vers for cid, vers in self.clique.items() if len(vers) < 2}
        if single_cliques:
            if self.verbose:
                print(f"ERROR: {len(single_cliques)} cliques with <2 versions")
            return False
        
        if self.verbose:
            print(f"✓ Valid: {len(self.clique)} cliques, {len(self.versions)} versions")
        
        return True
    
    def ensure_consistency(self) -> Tuple[Dict, Dict, List]:
        """
        Force consistency by removing invalid entries.
        
        Returns:
            Tuple of (cleaned_info, cleaned_clique, cleaned_versions)
        """
        # Remove invalid versions from cliques
        cleaned_clique = {}
        for clique_id, vers in self.clique.items():
            valid_vers = [v for v in vers if v in self.info]
            if len(valid_vers) >= 2:
                cleaned_clique[clique_id] = valid_vers
        
        # Rebuild versions list
        cleaned_versions = []
        for vers in cleaned_clique.values():
            cleaned_versions.extend(vers)
        
        # Filter info
        valid_version_set = set(cleaned_versions)
        cleaned_info = {v: data for v, data in self.info.items() 
                       if v in valid_version_set}
        
        if self.verbose:
            print(f"Cleaned: {len(cleaned_clique)} cliques, {len(cleaned_versions)} versions")
        
        return cleaned_info, cleaned_clique, cleaned_versions
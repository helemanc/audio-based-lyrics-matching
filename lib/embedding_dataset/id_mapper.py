"""
ID mapping methods extracted from original EmbeddingDataset.
"""
import os
from .utils import create_deterministic_song_id


class IDMapper:
    """Handles ID mapping creation and management"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def _create_id_mappings(self):
        """Create consistent integer mappings for clique and version IDs"""
        try:
            if self.dataset.df.empty:
                self.dataset.clique_id_to_idx = {}
                self.dataset.version_id_to_idx = {}
                self.dataset.idx_to_clique_id = {}
                self.dataset.idx_to_version_id = {}
                return
                
            unique_clique_ids = sorted(self.dataset.df["clique_id"].unique())
            unique_version_ids = sorted(self.dataset.df["version_id"].unique())
            
            self.dataset.clique_id_to_idx = {cid: idx for idx, cid in enumerate(unique_clique_ids)}
            self.dataset.version_id_to_idx = {vid: idx for idx, vid in enumerate(unique_version_ids)}
            
            self.dataset.idx_to_clique_id = {idx: cid for cid, idx in self.dataset.clique_id_to_idx.items()}
            self.dataset.idx_to_version_id = {idx: vid for vid, idx in self.dataset.version_id_to_idx.items()}
            
            self.dataset.df["clique_idx"] = self.dataset.df["clique_id"].map(self.dataset.clique_id_to_idx)
            self.dataset.df["version_idx"] = self.dataset.df["version_id"].map(self.dataset.version_id_to_idx)
            
            if self.verbose:
                print(f"Created mappings: {len(unique_clique_ids)} cliques, {len(unique_version_ids)} versions")
                
        except Exception as e:
            print(f"Error creating ID mappings: {e}")
            self.dataset.clique_id_to_idx = {}
            self.dataset.version_id_to_idx = {}
            self.dataset.idx_to_clique_id = {}
            self.dataset.idx_to_version_id = {}
    
    def _extract_clique_version_for_hash(self, version_key):
        """Return (clique_str, version_str) for create_deterministic_song_id"""
        md = self.dataset.info[version_key]

        if self.dataset.dataset_name == 'shs':
            if '-' not in version_key:
                raise ValueError(f"SHS version_key without '-': {version_key}")
            clique_str, version_str = version_key.split('-', 1)
            return str(clique_str), str(version_str)

        elif self.dataset.dataset_name == 'lyric-covers':
            clique_str = str(md.get('clique_id', md.get('clique')))
            version_str = str(md.get('version_id', md.get('version_key', version_key)))
            return clique_str, version_str

        elif self.dataset.dataset_name == 'discogs-vi':
            clique_str = str(md.get('clique_id', md.get('clique')))
            version_str = str(md.get('version_id', md.get('base_filename', md.get('version_key', version_key))))
            version_str = version_str.replace(os.sep, '/')
            return clique_str, version_str

        clique_str = str(md.get('clique', ''))
        version_str = str(md.get('version_id', md.get('version_key', version_key)))
        return clique_str, version_str
    
    def _rebuild_info_with_deterministic_ids(self):
        """Assign deterministic IDs using the same hashing strategy"""
        if self.verbose:
            print("Rebuilding info dict with deterministic IDs...")

        new_info = {}
        for version_key, meta in self.dataset.info.items():
            clique_str, version_str = self._extract_clique_version_for_hash(version_key)
            det_id = create_deterministic_song_id(clique_str, version_str)

            nm = meta.copy()
            nm['id'] = det_id
            new_info[version_key] = nm

            if self.verbose and len(new_info) <= 5:
                print(f"  {version_key}: {clique_str}-{version_str} -> {det_id}")

        self.dataset.info = new_info

        if self.verbose:
            print(f"Rebuilt info dict with {len(self.dataset.info)} versions using deterministic IDs")
    
    def create_global_clique_id_mapping(self):
        """Create global clique ID mapping across all splits (for caching)"""
        if self.verbose:
            print("Creating global clique ID mapping...")
        
        global_clique2id = {}
        offset = 0
        
        for split_name in ["train", "val", "test"]:
            for i, clique_id in enumerate(self.dataset.splitdict[split_name].keys()):
                global_clique2id[clique_id] = offset + i
            offset += len(self.dataset.splitdict[split_name])
        
        self.dataset.global_clique2id = global_clique2id
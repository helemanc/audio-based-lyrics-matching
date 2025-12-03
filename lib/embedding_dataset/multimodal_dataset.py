"""
Multimodal dataset classes for WEALY+CLEWS and Whisper+CLEWS.
"""
import torch
from pathlib import Path
from .base_dataset import EmbeddingDataset
from .utils import create_deterministic_song_id


class MultimodalEmbeddingDataset_WEALYCLEWS(EmbeddingDataset):
    """
    Dataset class to handle WEALY concatenated + CLEWS embeddings
    Returns: wealy_concat, full_clews, avg_clews, clews_mask
    """
    
    def __init__(self, conf, split, augment=False, verbose=False):
        super().__init__(
            conf=conf, 
            split=split, 
            augment=augment,
            embedding_type="multimodal_wealy_clews",
            embedding_format="all",
            verbose=verbose
        )
        self.ensure_version_alignment()
    
    def _get_required_embedding_filename(self):
        """Override to return special marker for verification"""
        return "MULTIMODAL_WEALY_CLEWS_CONCAT"
    
    def verify_embeddings_exist(self):
        """Verify that WEALY concat, full CLEWS, avg CLEWS, and masks exist"""
        if self.verbose:
            print("Verifying WEALY concat + full CLEWS + avg CLEWS + masks exist...")
        
        hidden_states_path = Path(self.conf.path.hidden_states)
        all_good = True
        missing_embeddings = []
        
        for split_name in ["train", "val", "test"]:
            missing = []
            for clique_id, versions in self.splitdict[split_name].items():
                for version in versions:
                    has_wealy_concat = self.verifier._embedding_exists(version, hidden_states_path, "hs_wealy_concat.pt")
                    has_full_clews = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews.pt")
                    has_avg_clews = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews_avg.pt")
                    has_clews_mask = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews_mask.pt")
                    
                    if not (has_wealy_concat and has_full_clews and has_avg_clews and has_clews_mask):
                        missing.append(version)
                        missing_embeddings.append((split_name, version))
            
            if missing:
                all_good = False
                if self.verbose:
                    print(f"  {split_name}: {len(missing)} versions missing embeddings")
            else:
                if self.verbose:
                    total = sum(len(v) for v in self.splitdict[split_name].values())
                    print(f"  {split_name}: ✓ All {total} versions have embeddings")
        
        return all_good
    
    def ensure_version_alignment(self):
        """Build alignment with deterministic IDs"""
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
    
    def load_multimodal_embeddings(self, version):
        """Load WEALY concat + full CLEWS + avg CLEWS + clews mask"""
        hidden_states_path = Path(self.conf.path.hidden_states)
        
        # Get version folder path
        if self.dataset_name == 'shs':
            set_id, ver_id = version.split('-')
            set_id_int = int(set_id)
            if set_id_int <= 9:
                folder_name = f"{set_id}-"
            elif set_id_int <= 99:
                folder_name = set_id
            else:
                folder_name = set_id[:2]
            version_folder = hidden_states_path / folder_name / version
        elif self.dataset_name == 'lyric-covers':
            version_folder = hidden_states_path / version
        elif self.dataset_name == 'discogs-vi':
            import os
            version_folder = hidden_states_path / version.replace('/', os.sep)
        else:
            return None, None, None, None
        
        # Load embedding files
        wealy_concat_path = version_folder / "hs_wealy_concat.pt"
        full_clews_path = version_folder / "hs_clews.pt"
        avg_clews_path = version_folder / "hs_clews_avg.pt"
        clews_mask_path = version_folder / "hs_clews_mask.pt"
        
        # Load WEALY concatenated format
        try:
            wealy_concat_data = torch.load(wealy_concat_path, map_location='cpu')
            
            if isinstance(wealy_concat_data, dict) and 'embeddings' in wealy_concat_data:
                wealy_concat = wealy_concat_data
                if wealy_concat['embeddings'].dtype == torch.float16:
                    wealy_concat['embeddings'] = wealy_concat['embeddings'].float()
            else:
                if wealy_concat_data.dtype == torch.float16:
                    wealy_concat_data = wealy_concat_data.float()
                
                wealy_concat = {
                    'embeddings': wealy_concat_data,
                    'chunk_info': {'total_chunks': wealy_concat_data.shape[0] if wealy_concat_data.dim() > 1 else 1},
                    'extraction_method': 'legacy_format'
                }
                
        except Exception as e:
            wealy_concat = {
                'embeddings': torch.zeros(10, self.conf.model.zdim),
                'chunk_info': {'total_chunks': 10},
                'extraction_method': 'dummy'
            }
            if self.verbose:
                print(f"Using dummy WEALY concat for {version}: {e}")
        
        # Load full CLEWS
        try:
            full_clews_emb = torch.load(full_clews_path, map_location='cpu')
            if full_clews_emb.dtype == torch.float16:
                full_clews_emb = full_clews_emb.float()
        except:
            full_clews_emb = torch.zeros(116, 2048)
            if self.verbose:
                print(f"Using dummy full CLEWS for {version}")
        
        # Load avg CLEWS
        try:
            avg_clews_emb = torch.load(avg_clews_path, map_location='cpu')
            if avg_clews_emb.dtype == torch.float16:
                avg_clews_emb = avg_clews_emb.float()
        except:
            avg_clews_emb = torch.zeros(2048)
            if self.verbose:
                print(f"Using dummy avg CLEWS for {version}")
        
        # Load CLEWS mask
        try:
            clews_mask = torch.load(clews_mask_path, map_location='cpu')
        except:
            clews_mask = torch.ones(116, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy CLEWS mask for {version}")
        
        return wealy_concat, full_clews_emb, avg_clews_emb, clews_mask
    
    def __getitem__(self, idx):
        """Get item with wealy_concat + full_clews + avg_clews + clews_mask"""
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions from same clique
        otherversions = [v for v in self.clique[cl] 
                        if v != v1 or torch.rand(1).item() < getattr(self, 'p_samesong', 0.0)]
        
        if getattr(self, 'augment', False):
            otherversions = [otherversions[k] for k in torch.randperm(len(otherversions)).tolist()]
        
        # Construct version array
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
            wealy_concat, full_clews_emb, avg_clews_emb, clews_mask = self.load_multimodal_embeddings(v)
            
            multimodal_embedding = {
                'wealy': wealy_concat,
                'full_clews': full_clews_emb,
                'avg_clews': avg_clews_emb,
                'clews_mask': clews_mask,
                'song_id': v,
                'class_id': icl
            }
            output += [i, multimodal_embedding]
        
        return output


class MultimodalEmbeddingDataset_WHISPERCLEWS(EmbeddingDataset):
    """
    Dataset class to handle Whisper + CLEWS embeddings 
    Returns: hs_last_seq, whisper_mask, full_clews, avg_clews, clews_mask
    """
    
    def __init__(self, conf, split, augment=False, verbose=False):
        super().__init__(
            conf=conf, 
            split=split, 
            augment=augment,
            embedding_type="multimodal_whisper_clews",
            embedding_format="all",
            verbose=verbose
        )
        self.ensure_version_alignment()
    
    def _get_required_embedding_filename(self):
        """Override to return special marker for verification"""
        return "MULTIMODAL_WHISPER_CLEWS_ALL"
    
    def verify_embeddings_exist(self):
        """Verify that Whisper hs_last_seq, full CLEWS, avg CLEWS, and masks exist"""
        if self.verbose:
            print("Verifying Whisper hs_last_seq + full CLEWS + avg CLEWS + masks exist...")
        
        hidden_states_path = Path(self.conf.path.hidden_states)
        all_good = True
        missing_embeddings = []
        
        for split_name in ["train", "val", "test"]:
            missing = []
            for clique_id, versions in self.splitdict[split_name].items():
                for version in versions:
                    has_whisper = self.verifier._embedding_exists(version, hidden_states_path, "hs_last_seq.pt")
                    has_full_clews = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews.pt")
                    has_avg_clews = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews_avg.pt")
                    has_clews_mask = self.verifier._embedding_exists(version, hidden_states_path, "hs_clews_mask.pt")
                    
                    if not (has_whisper and has_full_clews and has_avg_clews and has_clews_mask):
                        missing.append(version)
                        missing_embeddings.append((split_name, version))
            
            if missing:
                all_good = False
                if self.verbose:
                    print(f"  {split_name}: {len(missing)} versions missing embeddings")
            else:
                if self.verbose:
                    total = sum(len(v) for v in self.splitdict[split_name].values())
                    print(f"  {split_name}: ✓ All {total} versions have embeddings")
        
        return all_good
    
    def ensure_version_alignment(self):
        """Build alignment with deterministic IDs"""
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
    
    def load_multimodal_embeddings(self, version):
        """Load Whisper hs_last_seq + full CLEWS + avg CLEWS + clews mask"""
        hidden_states_path = Path(self.conf.path.hidden_states)
        
        # Get version folder path
        if self.dataset_name == 'shs':
            set_id, ver_id = version.split('-')
            set_id_int = int(set_id)
            if set_id_int <= 9:
                folder_name = f"{set_id}-"
            elif set_id_int <= 99:
                folder_name = set_id
            else:
                folder_name = set_id[:2]
            version_folder = hidden_states_path / folder_name / version
        elif self.dataset_name == 'lyric-covers':
            version_folder = hidden_states_path / version
        elif self.dataset_name == 'discogs-vi':
            import os
            version_folder = hidden_states_path / version.replace('/', os.sep)
        else:
            return None, None, None, None, None
        
        # Load embedding files
        whisper_path = version_folder / "hs_last_seq.pt"
        full_clews_path = version_folder / "hs_clews.pt"
        avg_clews_path = version_folder / "hs_clews_avg.pt"
        clews_mask_path = version_folder / "hs_clews_mask.pt"
        
        # Load Whisper hs_last_seq
        try:
            whisper_emb = torch.load(whisper_path, map_location='cpu')
            if whisper_emb.dtype == torch.float16:
                whisper_emb = whisper_emb.float()
            whisper_mask = torch.ones(whisper_emb.shape[0], dtype=torch.bool)
        except:
            whisper_emb = torch.zeros(15, 1280)
            whisper_mask = torch.ones(15, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy Whisper for {version}")
        
        # Load full CLEWS
        try:
            full_clews_emb = torch.load(full_clews_path, map_location='cpu')
            if full_clews_emb.dtype == torch.float16:
                full_clews_emb = full_clews_emb.float()
        except:
            full_clews_emb = torch.zeros(16, 2048)
            if self.verbose:
                print(f"Using dummy full CLEWS for {version}")
        
        # Load avg CLEWS
        try:
            avg_clews_emb = torch.load(avg_clews_path, map_location='cpu')
            if avg_clews_emb.dtype == torch.float16:
                avg_clews_emb = avg_clews_emb.float()
        except:
            avg_clews_emb = torch.zeros(2048)
            if self.verbose:
                print(f"Using dummy avg CLEWS for {version}")
        
        # Load CLEWS mask
        try:
            clews_mask = torch.load(clews_mask_path, map_location='cpu')
        except:
            clews_mask = torch.ones(16, dtype=torch.bool)
            if self.verbose:
                print(f"Using dummy CLEWS mask for {version}")
        
        return whisper_emb, whisper_mask, full_clews_emb, avg_clews_emb, clews_mask
    
    def __getitem__(self, idx):
        """Get item with whisper + whisper_mask + full_clews + avg_clews + clews_mask"""
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions from same clique
        otherversions = [v for v in self.clique[cl] 
                        if v != v1 or torch.rand(1).item() < getattr(self, 'p_samesong', 0.0)]
        
        if getattr(self, 'augment', False):
            otherversions = [otherversions[k] for k in torch.randperm(len(otherversions)).tolist()]
        
        # Construct version array
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
            whisper_emb, whisper_mask, full_clews_emb, avg_clews_emb, clews_mask = self.load_multimodal_embeddings(v)
            
            multimodal_embedding = {
                'whisper': whisper_emb,
                'whisper_mask': whisper_mask,
                'full_clews': full_clews_emb,
                'avg_clews': avg_clews_emb,
                'clews_mask': clews_mask,
                'song_id': v,
                'class_id': icl
            }
            output += [i, multimodal_embedding]
        
        return output
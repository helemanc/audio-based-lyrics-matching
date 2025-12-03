"""
Embedding verification methods extracted from original EmbeddingDataset.
"""
from pathlib import Path
import os

class EmbeddingVerifier:
    """Handles embedding verification"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def verify_embeddings_exist(self):
        """Verify that embeddings exist for all remaining versions"""
        if self.verbose:
            print("Verifying embeddings exist for all remaining versions...")
        
        hidden_states_path = Path(self.dataset.conf.path.hidden_states)
        required_filename = self.dataset._get_required_embedding_filename()
        
        if not required_filename:
            print(f"Error: Unknown embedding type/format combination")
            return False
        
        if self.dataset.embedding_type == "multimodal":
            return self._verify_multimodal_embeddings_exist(hidden_states_path)
        
        # Single-modal verification
        if self.verbose:
            print(f"Looking for embedding file: {required_filename}")
        
        all_good = True
        all_missing_embeddings = []
        
        for split_name in ["train", "val", "test"]:
            missing_embeddings = []
            
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                for version in versions:
                    if not self._embedding_exists(version, hidden_states_path, required_filename):
                        missing_embeddings.append(version)
                        all_missing_embeddings.append((split_name, version))
            
            if missing_embeddings:
                all_good = False
                print(f"  {split_name}: {len(missing_embeddings)} versions missing embeddings:")
                for version in missing_embeddings[:10]:
                    print(f"    {version}")
                if len(missing_embeddings) > 10:
                    print(f"    ... and {len(missing_embeddings) - 10} more")
            else:
                print(f"  {split_name}: ✓ All {sum(len(v) for v in self.dataset.splitdict[split_name].values())} versions have embeddings")
        
        if all_missing_embeddings:
            self._save_missing_embeddings_list(all_missing_embeddings)
        
        if all_good and self.verbose:
            print("✓ All versions have corresponding embeddings!")
        
        return all_good
    
    def _verify_multimodal_embeddings_exist(self, hidden_states_path):
        """Verify that both Whisper and CLEWS embeddings exist for multimodal mode"""
        whisper_filename = "hs_last_seq.pt"
        clews_filename = "hs_clews.pt"
        
        if self.verbose:
            print("Multimodal verification:")
            print(f"  Looking for Whisper embeddings: {whisper_filename}")
            print(f"  Looking for CLEWS embeddings: {clews_filename}")
        
        all_good = True
        all_missing_embeddings = []
        
        for split_name in ["train", "val", "test"]:
            missing_whisper = []
            missing_clews = []
            has_both = []
            
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                for version in versions:
                    has_whisper = self._embedding_exists(version, hidden_states_path, whisper_filename)
                    has_clews = self._embedding_exists(version, hidden_states_path, clews_filename)
                    
                    if has_whisper and has_clews:
                        has_both.append(version)
                    else:
                        if not has_whisper:
                            missing_whisper.append(version)
                            all_missing_embeddings.append((split_name, version, "whisper"))
                        if not has_clews:
                            missing_clews.append(version)
                            all_missing_embeddings.append((split_name, version, "clews"))
            
            total_versions = sum(len(v) for v in self.dataset.splitdict[split_name].values())
            
            print(f"  {split_name}:")
            print(f"    Both embeddings: {len(has_both)}/{total_versions}")
            
            if missing_whisper:
                print(f"    Missing Whisper: {len(missing_whisper)} versions")
                all_good = False
                
            if missing_clews:
                print(f"    Missing CLEWS: {len(missing_clews)} versions")
                
            if not missing_whisper and not missing_clews:
                print(f"    ✓ All versions have both embeddings")
        
        if all_missing_embeddings:
            self._save_missing_multimodal_embeddings_list(all_missing_embeddings)
        
        whisper_missing = [item for item in all_missing_embeddings if len(item) > 2 and item[2] == "whisper"]
        
        if whisper_missing:
            if self.verbose:
                print(f"❌ {len(whisper_missing)} versions missing critical Whisper embeddings")
            return False
        
        if self.verbose:
            clews_missing = [item for item in all_missing_embeddings if len(item) > 2 and item[2] == "clews"]
            if clews_missing:
                print(f"⚠️  {len(clews_missing)} versions missing CLEWS (will use dummy embeddings)")
            print("✓ All versions have at least Whisper embeddings!")
        
        return True
    
    def _embedding_exists(self, version, hidden_states_path, required_filename):
        """Check if specific embedding file exists for given version"""
        if self.dataset.dataset_name == 'shs':
            return self._embedding_exists_shs(version, hidden_states_path, required_filename)
        elif self.dataset.dataset_name == 'lyric-covers':
            return self._embedding_exists_lyric_covers(version, hidden_states_path, required_filename)
        elif self.dataset.dataset_name == 'discogs-vi':
            return self._embedding_exists_discogs_vi(version, hidden_states_path, required_filename)
        return False
    
    def _embedding_exists_shs(self, version, hidden_states_path, required_filename):
        """Check if SHS embedding file exists"""
        if '-' not in version:
            return False
            
        set_id, ver_id = version.split('-', 1)
        
        possible_folders = [
            set_id,
            f"{set_id}-" if set_id.isdigit() and int(set_id) < 10 else set_id,
            set_id[:2] if len(set_id) > 2 else set_id
        ]
        
        for folder_name in possible_folders:
            version_folder = hidden_states_path / folder_name / f"{set_id}-{ver_id}"
            embedding_file = version_folder / required_filename
            print
            if embedding_file.exists():
                return True
        
        return False
    
    def _embedding_exists_lyric_covers(self, version, hidden_states_path, required_filename):
        """Check if Lyric Covers embedding file exists"""
        version_folder = hidden_states_path / version
        embedding_file = version_folder / required_filename
        return embedding_file.exists()
    
    def _embedding_exists_discogs_vi(self, version, hidden_states_path, required_filename):
        """Check if Discogs-VI embedding file exists"""
        version_folder = hidden_states_path / version.replace('/', os.sep)
        embedding_file = version_folder / required_filename
        return embedding_file.exists()
    
    def _save_missing_embeddings_list(self, missing_embeddings_list):
        """Save list of missing embeddings to a file for re-extraction"""
        try:
            cache_dir = None
            if hasattr(self.dataset.conf.path, 'cache'):
                cache_dir = Path(self.dataset.conf.path.cache) / self.dataset.dataset_nickname
            elif hasattr(self.dataset.conf.path, 'working_dir'):
                cache_dir = Path(self.dataset.conf.path.working_dir) / 'cache' / self.dataset.dataset_nickname
            else:
                if self.verbose:
                    print("Warning: No cache directory configured")
                return
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            missing_file = cache_dir / f"missing_embeddings_{self.dataset.embedding_type}.txt"
            
            with open(missing_file, 'w') as f:
                version_ids = [version for split_name, version in missing_embeddings_list]
                unique_versions = sorted(set(version_ids))
                
                for version in unique_versions:
                    f.write(f"{version}\n")
            
            if self.verbose:
                print(f"✓ Saved {len(set(version_ids))} missing embeddings to: {missing_file}")
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not save missing embeddings list: {e}")
    
    def _save_missing_multimodal_embeddings_list(self, missing_embeddings_list):
        """Save list of missing multimodal embeddings to files for re-extraction"""
        try:
            cache_dir = None
            if hasattr(self.dataset.conf.path, 'cache'):
                cache_dir = Path(self.dataset.conf.path.cache) / self.dataset.dataset_nickname
            elif hasattr(self.dataset.conf.path, 'working_dir'):
                cache_dir = Path(self.dataset.conf.path.working_dir) / 'cache' / self.dataset.dataset_nickname
            else:
                if self.verbose:
                    print("Warning: No cache directory configured")
                return
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            missing_whisper = []
            missing_clews = []
            
            for item in missing_embeddings_list:
                if len(item) >= 3:
                    split_name, version, emb_type = item[:3]
                    if emb_type == "whisper":
                        missing_whisper.append(version)
                    elif emb_type == "clews":
                        missing_clews.append(version)
            
            if missing_whisper:
                whisper_file = cache_dir / "missing_embeddings_whisper.txt"
                with open(whisper_file, 'w') as f:
                    for version in sorted(set(missing_whisper)):
                        f.write(f"{version}\n")
                if self.verbose:
                    print(f"✓ Saved {len(set(missing_whisper))} missing Whisper embeddings to: {whisper_file}")
            
            if missing_clews:
                clews_file = cache_dir / "missing_embeddings_clews.txt"
                with open(clews_file, 'w') as f:
                    for version in sorted(set(missing_clews)):
                        f.write(f"{version}\n")
                if self.verbose:
                    print(f"✓ Saved {len(set(missing_clews))} missing CLEWS embeddings to: {clews_file}")
                    
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not save missing embeddings lists: {e}")
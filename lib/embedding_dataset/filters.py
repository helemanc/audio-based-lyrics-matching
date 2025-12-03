"""
Filtering methods extracted from original EmbeddingDataset.
"""
from pathlib import Path
from .embedding_verifier import EmbeddingVerifier

class DatasetFilter:
    """Handles all filtering operations"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def remove_versions_without_audio(self):
        """Remove versions that don't have corresponding audio files"""
        if self.verbose:
            print("Removing versions without audio files...")
        
        if self.dataset.dataset_name == 'shs':
            audio_base_path = Path(self.dataset.conf.path.data) / "SHS100K" / "audio"
        elif self.dataset.dataset_name == 'lyric-covers':
            audio_base_path = Path(self.dataset.conf.path.data) / "LyricCovers" / "audio"
        elif self.dataset.dataset_name == 'discogs-vi':
            audio_base_path = Path(self.dataset.conf.path.data) / "DiscogsVI" / "audio"
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset.dataset_name}")
        
        for split_name in ["train", "val", "test"]:
            original_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            filtered_cliques = {}
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                versions_with_audio = []
                for version in versions:
                    if self._audio_exists(version, audio_base_path):
                        versions_with_audio.append(version)
                
                if versions_with_audio:
                    filtered_cliques[clique_id] = versions_with_audio
            
            self.dataset.splitdict[split_name] = filtered_cliques
            new_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            if self.verbose:
                print(f"  {split_name}: {original_count} -> {new_count} versions (removed {original_count - new_count})")
    
    def _audio_exists(self, version, audio_base_path):
        """Check if audio file exists for given version"""
        if self.dataset.dataset_name == 'shs':
            return self._audio_exists_shs(version, audio_base_path)
        elif self.dataset.dataset_name == 'lyric-covers':
            return self._audio_exists_lyric_covers(version, audio_base_path)
        elif self.dataset.dataset_name == 'discogs-vi':
            return self._audio_exists_discogs_vi(version, audio_base_path)
        return False
    
    def _audio_exists_shs(self, version, audio_base_path):
        """Check if SHS audio file exists"""
        if '-' not in version:
            return False
            
        set_id, ver_id = version.split('-', 1)
        
        possible_folders = [
            set_id,
            f"{set_id}-" if set_id.isdigit() and int(set_id) < 10 else set_id,
            set_id[:2] if len(set_id) > 2 else set_id
        ]
        
        for folder_name in possible_folders:
            audio_file = audio_base_path / folder_name / f"{version}.mp3"
            if audio_file.exists():
                return True
        
        return False
    
    def _audio_exists_lyric_covers(self, version, audio_base_path):
        """Check if Lyric Covers audio file exists"""
        audio_file = audio_base_path / version / f"{version}_audio.mp3"
        return audio_file.exists()
    
    def _audio_exists_discogs_vi(self, version, audio_base_path):
        """Check if Discogs-VI audio file exists"""
        audio_file = audio_base_path / f"{version}.mp3"
        return audio_file.exists()
    
    def remove_single_version_cliques(self):
        """Remove cliques that have only 1 version remaining"""
        if self.verbose:
            print("Removing cliques with only 1 version...")
        
        for split_name in ["train", "val", "test"]:
            original_clique_count = len(self.dataset.splitdict[split_name])
            original_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            filtered_cliques = {}
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                if len(versions) >= 2:
                    filtered_cliques[clique_id] = versions
            
            self.dataset.splitdict[split_name] = filtered_cliques
            new_clique_count = len(self.dataset.splitdict[split_name])
            new_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            if self.verbose:
                removed_cliques = original_clique_count - new_clique_count
                removed_versions = original_version_count - new_version_count
                print(f"  {split_name}: {original_clique_count} -> {new_clique_count} cliques (removed {removed_cliques})")
                print(f"  {split_name}: {original_version_count} -> {new_version_count} versions (removed {removed_versions})")
    
    def remove_overlapping_cliques(self):
        """Remove overlapping cliques: train takes priority over val/test"""
        if self.verbose:
            print("Checking for overlapping cliques across splits...")
        
        train_cliques = set(self.dataset.splitdict["train"].keys())
        
        overlapping_val = set(self.dataset.splitdict["val"].keys()) & train_cliques
        for clique_id in overlapping_val:
            del self.dataset.splitdict["val"][clique_id]
        
        overlapping_test = set(self.dataset.splitdict["test"].keys()) & train_cliques
        for clique_id in overlapping_test:
            del self.dataset.splitdict["test"][clique_id]
        
        if self.verbose and (overlapping_val or overlapping_test):
            print(f"  Removed {len(overlapping_val)} overlapping cliques from val")
            print(f"  Removed {len(overlapping_test)} overlapping cliques from test")
        elif self.verbose:
            print("  No overlapping cliques found")
    
    def _filter_to_available_embeddings(self):
        """Filter dataset to only include versions with available embeddings (debug mode)"""
        if self.verbose:
            print("DEBUG MODE: Filtering to only versions with available embeddings...")
        
        hidden_states_path = Path(self.dataset.conf.path.hidden_states)
        required_filename = self.dataset._get_required_embedding_filename()
        
        if not required_filename:
            print(f"Error: Unknown embedding type/format combination")
            return
        
        if self.dataset.embedding_type == "multimodal":
            self._filter_to_available_multimodal_embeddings(hidden_states_path)
            return
        
        # Single-modal filtering

        verifier = EmbeddingVerifier(self.dataset, self.verbose)
        
        for split_name in ["train", "val", "test"]:
            original_clique_count = len(self.dataset.splitdict[split_name])
            original_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            filtered_cliques = {}
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                versions_with_embeddings = []
                for version in versions:
                    if verifier._embedding_exists(version, hidden_states_path, required_filename):
                        versions_with_embeddings.append(version)
                
                if len(versions_with_embeddings) >= 2:
                    filtered_cliques[clique_id] = versions_with_embeddings
            
            self.dataset.splitdict[split_name] = filtered_cliques
            new_clique_count = len(self.dataset.splitdict[split_name])
            new_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            if self.verbose:
                removed_cliques = original_clique_count - new_clique_count
                removed_versions = original_version_count - new_version_count
                print(f"  {split_name}: {original_clique_count} -> {new_clique_count} cliques (removed {removed_cliques})")
                print(f"  {split_name}: {original_version_count} -> {new_version_count} versions (removed {removed_versions})")
    
    def _filter_to_available_multimodal_embeddings(self, hidden_states_path):
        """Filter to versions that have at least Whisper embeddings (CLEWS is optional)"""
        verifier = EmbeddingVerifier(self.dataset, self.verbose)
        
        whisper_filename = "hs_last_seq.pt"
        
        if self.verbose:
            print("Filtering multimodal embeddings (requiring Whisper, CLEWS optional)...")
        
        for split_name in ["train", "val", "test"]:
            original_clique_count = len(self.dataset.splitdict[split_name])
            original_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            filtered_cliques = {}
            for clique_id, versions in self.dataset.splitdict[split_name].items():
                versions_with_whisper = []
                for version in versions:
                    if verifier._embedding_exists(version, hidden_states_path, whisper_filename):
                        versions_with_whisper.append(version)
                
                if len(versions_with_whisper) >= 2:
                    filtered_cliques[clique_id] = versions_with_whisper
            
            self.dataset.splitdict[split_name] = filtered_cliques
            new_clique_count = len(self.dataset.splitdict[split_name])
            new_version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
            
            if self.verbose:
                removed_cliques = original_clique_count - new_clique_count
                removed_versions = original_version_count - new_version_count
                print(f"  {split_name}: {original_clique_count} -> {new_clique_count} cliques (removed {removed_cliques})")
                print(f"  {split_name}: {original_version_count} -> {new_version_count} versions (removed {removed_versions})")
    
    def _update_info_after_filtering(self):
        """Remove versions from info that were filtered out"""
        if self.verbose:
            print("Updating info dict to remove filtered versions...")
        
        all_remaining_versions = set()
        for split_name in ["train", "val", "test"]:
            for versions in self.dataset.splitdict[split_name].values():
                all_remaining_versions.update(versions)
        
        original_count = len(self.dataset.info)
        self.dataset.info = {k: v for k, v in self.dataset.info.items() if k in all_remaining_versions}
        
        if self.verbose:
            print(f"Info dict: {original_count} -> {len(self.dataset.info)} versions (removed {original_count - len(self.dataset.info)})")
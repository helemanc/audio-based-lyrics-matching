"""
Base embedding dataset class - refactored to use manager classes.
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Import manager classes
from .metadata_loaders import MetadataLoader
from .filters import DatasetFilter
from .cache_manager import CacheManager
from .path_manager import PathManager
from .id_mapper import IDMapper
from .embedding_verifier import EmbeddingVerifier
from .validator import DataValidator
LIMIT_CLIQUES = None


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, embedding_type=None, embedding_format=None, 
                 augment=False, fullsongs=False, n_per_class=2, p_samesong=0.0, 
                 verbose=True, debug=False, return_paths=False):
        self.conf = conf
        self.split = split
        self.augment = augment
        self.fullsongs = fullsongs
        self.n_per_class = n_per_class
        self.p_samesong = p_samesong
        self.verbose = verbose
        self.debug = debug
        self.info = {}
        self.splitdict = {}
        self.clique2id = {}
        self.return_paths = return_paths
        
        # Set embedding type and format
        self.embedding_type = embedding_type or getattr(conf.data, 'embedding_type', 'encoder')
        self.embedding_format = embedding_format or getattr(conf.data, 'embedding_format', 'concat')
        
        # Get dataset name
        self.dataset_name = getattr(conf.data, 'dataset_name', 'shs')
        self.dataset_nickname = self._get_dataset_nickname()
        
        if self.verbose:
            print(f"Dataset: {self.dataset_name} ({self.dataset_nickname})")
            print(f"Using embedding_type: {self.embedding_type}, embedding_format: {self.embedding_format}")
        
        # Initialize manager classes
        self.metadata_loader = MetadataLoader(self, verbose)
        self.filter = DatasetFilter(self, verbose)
        self.cache_manager = CacheManager(self, verbose)
        self.path_manager = PathManager(self, verbose)
        self.id_mapper = IDMapper(self, verbose)
        self.verifier = EmbeddingVerifier(self, verbose)
        self.validator = DataValidator(self, verbose)
        
        # Build the dataset
        self.info, self.splitdict, self.clique2id = self.build_clean_dataset()
        
        # Apply clique limit if specified
        if LIMIT_CLIQUES is None:
            self.clique = self.splitdict[split]
        else:
            if self.verbose:
                print(f"[Limiting cliques to {LIMIT_CLIQUES}]")
            self.clique = {}
            for key, item in self.splitdict[split].items():
                self.clique[key] = item
                if len(self.clique) == LIMIT_CLIQUES:
                    break

        # Filter info to current split
        self._filter_info_to_current_split_only()

        if self.dataset_name == "discogs-vi":
            self._ensure_perfect_consistency()
    
        # Create clique ID mapping for current split
        self._create_clique_id_mapping()

        # Get versions list
        self.versions = []
        for vers in self.clique.values():
            self.versions += vers

        if self.verbose:
            print(f"\nFinal validation after dataset construction:")
            self._validate_data_structures()
    
    def _get_dataset_nickname(self):
        """Map dataset names to nicknames for file paths"""
        name_mapping = {
            'shs': 'shs',
            'lyric-covers': 'lyc', 
            'discogs-vi': 'dvi'
        }
        return name_mapping.get(self.dataset_name, self.dataset_name)
    
    def _get_required_embedding_filename(self):
        """Get the required embedding filename based on type and format"""
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
            return "MULTIMODAL_VERIFICATION"
        
        return None
    
    def build_clean_dataset(self):
        """Run the complete pipeline"""
        # Try to load from cache first
        if self.metadata_loader.build_metadata_from_filesystem():
            if hasattr(self, '_loaded_from_cache'):
                self.print_statistics()
                return self.info, self.splitdict, self.clique2id
        
        # Otherwise, run the full pipeline
        self.filter.remove_versions_without_audio()
        self.filter.remove_single_version_cliques()
        self.filter.remove_overlapping_cliques()

        if self.debug:
            self.filter._filter_to_available_embeddings()
        
        # Update info dict
        self.filter._update_info_after_filtering()
        
        # Rebuild info dict with deterministic IDs
        self.id_mapper._rebuild_info_with_deterministic_ids()
        
        # Verify embeddings exist
        embeddings_ok = self.verifier.verify_embeddings_exist()
        
        # Only save to cache if all embeddings are present
        if embeddings_ok:
            self.cache_manager._save_to_cache()
        else:
            if self.verbose:
                print("⚠ Not saving to cache due to missing embeddings")
        
        # Create global clique ID mapping
        self.id_mapper.create_global_clique_id_mapping()
        self.print_statistics()
        
        return self.info, self.splitdict, self.clique2id
    
    def _filter_info_to_current_split_only(self):
        """Filter info to only contain versions whose CLIQUES exist in current split"""
        self.validator.filter_info_to_current_split_only()
    
    def _ensure_perfect_consistency(self):
        """Ensure perfect consistency between info, clique, and versions for DVI"""
        self.validator.ensure_perfect_consistency()
    
    def _validate_data_structures(self):
        """Debug method to validate that info, clique, and versions are consistent"""
        self.validator.validate_data_structures()
        
    def _create_clique_id_mapping(self):
        """Create clique ID mapping for current split"""
        self.clique2id = {}
        if self.split == "train":
            offset = 0
        elif self.split == "val":
            offset = len(self.splitdict["train"])
        else:
            offset = len(self.splitdict["train"]) + len(self.splitdict["val"])
        
        for i, cl in enumerate(self.clique.keys()):
            self.clique2id[cl] = offset + i
    
    def print_statistics(self):
        """Print final statistics for each split"""
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        
        total_cliques = 0
        total_versions = 0
        
        for split_name in ["train", "val", "test"]:
            clique_count = len(self.splitdict[split_name])
            version_count = sum(len(versions) for versions in self.splitdict[split_name].values())
            
            print(f"{split_name.upper():>5}: {clique_count:>5} cliques, {version_count:>6} versions")
            
            total_cliques += clique_count
            total_versions += version_count
        
        print("-" * 50)
        print(f"TOTAL: {total_cliques:>5} cliques, {total_versions:>6} versions")
    
    def get_embedding_path(self, version):
        """Get the path to the embedding file for a given version"""
        return self.path_manager.get_embedding_path(version)
    
    def load_embedding(self, version):
        """Load embedding for a given version"""
        embedding_path = self.get_embedding_path(version)
        
        if embedding_path is None:
            if self.verbose:
                print(f"Warning: Embedding file not found for version {version}")
            return None
        
        try:
            embedding = torch.load(embedding_path, map_location='cpu')
            
            # Convert to float32 if needed
            if isinstance(embedding, torch.Tensor) and embedding.dtype == torch.float16:
                embedding = embedding.float()
            elif isinstance(embedding, dict):
                embedding = {k: (v.float() if v.dtype == torch.float16 else v) 
                            for k, v in embedding.items()}
            
            if self.embedding_type == "sbert":
                if isinstance(embedding, torch.Tensor):
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)
                    return embedding
                else:
                    if self.verbose:
                        print(f"Warning: Expected tensor for SBERT embedding, got {type(embedding)}")
                    return None
            
            return embedding
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading embedding from {embedding_path}: {e}")
            return None
    
    # Delegate cache methods to cache_manager
    def _load_from_cache(self):
        return self.cache_manager._load_from_cache()
    
    def _save_to_cache(self):
        return self.cache_manager._save_to_cache()
    
    def __len__(self):
        return len(self.versions)
    
    def __getitem__(self, idx):
        # Get v1 (anchor) and clique
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions from same clique
        otherversions = []
        for v in self.clique[cl]:
            if v != v1 or torch.rand(1).item() < getattr(self, 'p_samesong', 0.0):
                otherversions.append(v)
        
        if getattr(self, 'augment', False):
            new_vers = []
            for k in torch.randperm(len(otherversions)).tolist():
                new_vers.append(otherversions[k])
            otherversions = new_vers
            
        # Construct v1..vn array (n_per_class)
        n_per_class = getattr(self, 'n_per_class', 2)
        v_n = [v1]
        i_n = [i1]
        for k in range(n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i = self.info[v]["id"]
            v_n.append(v)
            i_n.append(i)
        
        # Load embeddings and create output
        output = [icl]
        for i, v in zip(i_n, v_n):
            embedding = self.load_embedding(v)
            output += [i, embedding]
        
        return output
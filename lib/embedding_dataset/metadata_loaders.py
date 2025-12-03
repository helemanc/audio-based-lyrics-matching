"""
Metadata loading methods extracted from original EmbeddingDataset.
"""
import pandas as pd
import os
from collections import defaultdict
import torch
from omegaconf import OmegaConf
from .id_mapper import IDMapper


class MetadataLoader:
    """Handles loading metadata from CSV files"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def build_metadata_from_filesystem(self):
        """Build metadata from filesystem structure, checking for existing metadata first"""
        
        # Priority 0: Check if final processed cache exists
        if self.dataset._load_from_cache():
            if self.verbose:
                print("✓ Loaded complete processed dataset from cache!")
            return True
        
        # Priority 1: Check if metadata file already exists
        metadata_path = OmegaConf.select(self.dataset.conf, 'path.meta')
        if metadata_path and os.path.exists(metadata_path):
            if self.verbose:
                print(f"Found existing metadata file at {metadata_path} - loading it...")
            return self._load_existing_metadata(metadata_path)
        
        # If no existing metadata, build from CSV files based on dataset
        if self.dataset.dataset_name == 'shs':
            return self._build_from_shs_csv()
        elif self.dataset.dataset_name == 'lyric-covers':
            return self._build_from_lyric_covers_csv()
        elif self.dataset.dataset_name == 'discogs-vi':
            return self._build_from_discogs_vi_csv()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset.dataset_name}")
    
    def _load_existing_metadata(self, metadata_path):
        """Load existing metadata file""" 
        try:
            if self.verbose:
                print(f"Loading metadata from {metadata_path}...")
            
            self.dataset.info, self.dataset.splitdict = torch.load(metadata_path, map_location='cpu')
            
            if self.verbose:
                total_versions = len(self.dataset.info)
                total_splits = sum(len(split_dict) for split_dict in self.dataset.splitdict.values())
                print(f"Loaded metadata with {total_versions} versions across {total_splits} split-cliques")
                for split_name, split_data in self.dataset.splitdict.items():
                    clique_count = len(split_data)
                    version_count = sum(len(versions) for versions in split_data.values())
                    print(f"  {split_name}: {clique_count} cliques, {version_count} versions")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"Error loading metadata file {metadata_path}: {e}")
                print("Will rebuild metadata from source files...")
            return self._build_from_shs_csv()
    
    def _build_from_shs_csv(self):
        """Build metadata from SHS100K CSV files using existing train/val/test splits"""
        if self.verbose:
            print("Building metadata from SHS100K CSV files...")
        
        # Load main SHS data
        shs_df = pd.read_csv(self.dataset.conf.path.shs_data)
        if self.verbose:
            print(f"Loaded {len(shs_df)} entries from CSV")
        
        # Load existing split files and merge
        split_files = {
            "train": os.path.join(self.dataset.conf.path.shs_splits, "SHS100K-TRAIN"),
            "val": os.path.join(self.dataset.conf.path.shs_splits, "SHS100K-VAL"), 
            "test": os.path.join(self.dataset.conf.path.shs_splits, "SHS100K-TEST")
        }
        
        split_dfs = []
        for split_name, split_file in split_files.items():
            if self.verbose:
                print(f"Loading {split_name} split from {split_file}")
            
            with open(split_file, 'r') as f:
                split_data = []
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            set_id, ver_id = parts[0].strip(), parts[1].strip()
                            split_data.append({'set_id': int(set_id), 'ver_id': int(ver_id), 'split': split_name})
                
                if split_data:
                    split_df = pd.DataFrame(split_data)
                    split_dfs.append(split_df)
        
        if split_dfs:
            all_splits_df = pd.concat(split_dfs, ignore_index=True)
            self.dataset.df = shs_df.merge(all_splits_df, on=['set_id', 'ver_id'], how='inner')
        else:
            self.dataset.df = shs_df.copy()
            self.dataset.df['split'] = 'train'
        
        # Add standardized columns
        self.dataset.df["clique_id"] = self.dataset.df["set_id"]                
        self.dataset.df["version_id"] = self.dataset.df["ver_id"]
        
        # Create ID mappings and build info/splitdict
        id_mapper = IDMapper(self.dataset, self.verbose)
        id_mapper._create_id_mappings()
        self._build_info_and_splitdict_from_df()
        
        if self.verbose:
            print("Built initial metadata from SHS files")
        
        self._save_metadata_if_configured()
        return True
    
    def _build_from_lyric_covers_csv(self):
        """Build metadata from Lyric Covers CSV files using existing train/val/test splits"""
        if self.verbose:
            print("Building metadata from Lyric Covers CSV files...")
        
        split_files = {
            "train": "train_no_dup.csv",
            "val": "val_no_dup.csv", 
            "test": "test_no_dup.csv"
        }
        
        split_dfs = []
        for split_name, split_file in split_files.items():
            split_path = os.path.join(self.dataset.conf.path.lyric_covers_data, split_file)
            if self.verbose:
                print(f"Loading {split_name} split from {split_path}")
            
            split_df = pd.read_csv(split_path)
            split_df['split'] = split_name
            split_dfs.append(split_df)
            if self.verbose:
                print(f"  Loaded {len(split_df)} entries from {split_file}")
        
        self.dataset.df = pd.concat(split_dfs, ignore_index=True)
        
        # Add standardized columns
        self.dataset.df["clique_id"] = self.dataset.df["label"]                
        self.dataset.df["version_id"] = self.dataset.df["id"]
        
        # Create ID mappings and build info/splitdict
        id_mapper = IDMapper(self.dataset, self.verbose)
        id_mapper._create_id_mappings()
        self._build_info_and_splitdict_from_df()
        
        if self.verbose:
            print("Built initial metadata from Lyric Covers files")
        
        self._save_metadata_if_configured()
        return True
    
    def _build_from_discogs_vi_csv(self):
        """Build metadata from Discogs-VI CSV files"""
        if self.verbose:
            print("Building metadata from Discogs-VI CSV files...")
        
        csv_path = os.path.join(self.dataset.conf.path.discogs_vi_data, "id-to-file-mapping.csv")
        self.dataset.df = pd.read_csv(csv_path, names=['split', 'clique_id', 'version_id', 'youtube_id', 'base_filename'])
        
        if self.verbose:
            print(f"Loaded {len(self.dataset.df)} entries from id-to-file-mapping.csv")
            print(f"Splits found: {self.dataset.df['split'].value_counts().to_dict()}")
        
        # Add standardized columns
        self.dataset.df["clique_id"] = self.dataset.df["clique_id"].astype(str)
        self.dataset.df["version_id"] = self.dataset.df["version_id"].astype(str)
        
        # Create ID mappings and build info/splitdict
        id_mapper = IDMapper(self.dataset, self.verbose)
        id_mapper._create_id_mappings()
        self._build_info_and_splitdict_from_df()
        
        if self.verbose:
            print("Built initial metadata from Discogs-VI files")
        
        self._save_metadata_if_configured()
        return True
    
    def _build_info_and_splitdict_from_df(self):
        """Build info dict and splitdict from the processed dataframe"""
        self.dataset.info = {}
        self.dataset.splitdict = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
        
        split_counter = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
        
        for idx, row in self.dataset.df.iterrows():
            # Create version key based on dataset
            if self.dataset.dataset_name == 'shs':
                version_key = f"{row['set_id']}-{row['ver_id']}"
                filename = f"{row['set_id']}-{row['ver_id']}.mp3"
            elif self.dataset.dataset_name == 'lyric-covers':
                version_key = str(row['id'])
                filename = f"{row['id']}_audio.mp3"
            elif self.dataset.dataset_name == 'discogs-vi':
                version_key = str(row['base_filename'])
                filename = f"{row['base_filename']}.mp3"
            else:
                continue
            
            # Build info dict entry
            self.dataset.info[version_key] = {
                'id': idx,
                'clique': str(row['clique_id']),
                'clique_idx': row['clique_idx'],
                'version_idx': row['version_idx'],
                'filename': filename,
                'version_key': version_key
            }
            
            # Add dataset-specific fields
            if self.dataset.dataset_name == 'shs':
                self.dataset.info[version_key].update({
                    'set_id': int(row['set_id']),
                    'ver_id': int(row['ver_id'])
                })
            elif self.dataset.dataset_name == 'lyric-covers':
                self.dataset.info[version_key].update({
                    'original_id': str(row['original_id']),
                    'is_cover': bool(row['is_cover']),
                    'song_text_type': str(row['song_text_type']),
                    'version_id': str(row['id'])
                })
            elif self.dataset.dataset_name == 'discogs-vi':
                self.dataset.info[version_key].update({
                    'base_filename': str(row['base_filename']),
                    'youtube_id': str(row['youtube_id']),
                    'version_id': str(row['version_id'])
                })
            
            # Add to split dict
            split_name = str(row['split']).lower()
            
            if split_name in self.dataset.splitdict:
                clique_key = str(row['clique_id'])
                self.dataset.splitdict[split_name][clique_key].append(version_key)
                split_counter[split_name] += 1
            else:
                if self.verbose:
                    print(f"Warning: Unknown split '{split_name}' found for version {version_key}")
                split_counter['unknown'] += 1
        
        # Convert defaultdict to regular dict
        for split_name in ["train", "val", "test"]:
            self.dataset.splitdict[split_name] = dict(self.dataset.splitdict[split_name])
        
        if self.verbose:
            total_versions = len(self.dataset.info)
            print(f"Built info dict with {total_versions} total versions")
            for split_name in ["train", "val", "test"]:
                clique_count = len(self.dataset.splitdict[split_name])
                version_count = sum(len(versions) for versions in self.dataset.splitdict[split_name].values())
                print(f"  {split_name}: {clique_count} cliques, {version_count} versions")
    
    def _save_metadata_if_configured(self):
        """Save metadata to file if a save path is configured"""        
        metadata_path = OmegaConf.select(self.dataset.conf, 'path.meta')
        
        if metadata_path:
            try:
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                metadata = (self.dataset.info, self.dataset.splitdict)
                torch.save(metadata, metadata_path)
                
                if self.verbose:
                    print(f"Saved metadata to {metadata_path} for future use")
                    
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not save metadata to {metadata_path}: {e}")
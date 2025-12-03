"""
Data validation and consistency checking for embedding datasets.
"""


class DataValidator:
    """Handles data structure validation and consistency checks"""
    
    def __init__(self, dataset, verbose=False):
        self.dataset = dataset
        self.verbose = verbose
    
    def filter_info_to_current_split_only(self):
        """Filter info to only contain versions whose CLIQUES exist in current split"""
        if self.verbose:
            print(f"Filtering info to only contain versions from current split: {self.dataset.split}")
        
        current_split_cliques = set(self.dataset.clique.keys())
        
        original_count = len(self.dataset.info)
        filtered_info = {}
        
        for version_key, version_data in self.dataset.info.items():
            clique_id = version_data['clique']
            if clique_id in current_split_cliques:
                filtered_info[version_key] = version_data
        
        self.dataset.info = filtered_info
        
        if self.verbose:
            print(f"Info dict: {original_count} -> {len(self.dataset.info)} versions (kept only versions from current split cliques)")
            
        # Rebuild versions list
        self.dataset.versions = []
        for vers in self.dataset.clique.values():
            self.dataset.versions += vers
            
        # Verify consistency
        missing_versions = [v for v in self.dataset.versions if v not in self.dataset.info]
        if missing_versions:
            if self.verbose:
                print(f"WARNING: {len(missing_versions)} versions in clique but not in info after filtering")
                for clique_id, versions in self.dataset.clique.items():
                    self.dataset.clique[clique_id] = [v for v in versions if v in self.dataset.info]
                
                self.dataset.versions = []
                for vers in self.dataset.clique.values():
                    self.dataset.versions += vers
                
                print(f"Cleaned up: final versions list has {len(self.dataset.versions)} versions")
    
    def ensure_perfect_consistency(self):
        """Ensure perfect consistency between info, clique, and versions for DVI"""
        if self.verbose:
            print(f"Ensuring perfect consistency for split: {self.dataset.split}")
        
        # Remove any cliques with versions not in self.dataset.info
        cleaned_cliques = {}
        total_removed_versions = 0
        
        for clique_id, versions in self.dataset.clique.items():
            valid_versions = [v for v in versions if v in self.dataset.info]
            if len(valid_versions) >= 2:
                cleaned_cliques[clique_id] = valid_versions
                removed = len(versions) - len(valid_versions)
                total_removed_versions += removed
            else:
                total_removed_versions += len(versions)
        
        self.dataset.clique = cleaned_cliques
        
        if self.verbose and total_removed_versions > 0:
            print(f"Cleaned cliques: removed {total_removed_versions} invalid version references")
        
        # Rebuild versions list
        old_version_count = len(self.dataset.versions) if hasattr(self.dataset, 'versions') else 0
        self.dataset.versions = []
        for vers in self.dataset.clique.values():
            self.dataset.versions += vers
        
        if self.verbose:
            print(f"Versions list: {old_version_count} -> {len(self.dataset.versions)} versions")
        
        # Remove info entries for cliques not in current split
        current_split_cliques = set(self.dataset.clique.keys())
        old_info_count = len(self.dataset.info)
        
        filtered_info = {}
        for version_key, version_data in self.dataset.info.items():
            clique_id = version_data['clique']
            if clique_id in current_split_cliques:
                filtered_info[version_key] = version_data
        
        self.dataset.info = filtered_info
        
        if self.verbose:
            print(f"Info dict: {old_info_count} -> {len(self.dataset.info)} versions (kept only current split)")
        
        # Final validation
        missing_versions = [v for v in self.dataset.versions if v not in self.dataset.info]
        if missing_versions:
            if self.verbose:
                print(f"ERROR: Found {len(missing_versions)} versions in clique but not in info!")
                print(f"Sample missing versions: {missing_versions[:5]}")
            raise ValueError(f"Inconsistency detected: {len(missing_versions)} versions in clique but not in info")
        
        # Verify all cliques in info match current split
        info_cliques = set()
        for version_data in self.dataset.info.values():
            info_cliques.add(version_data['clique'])
        
        clique_mismatch = info_cliques - current_split_cliques
        if clique_mismatch:
            if self.verbose:
                print(f"ERROR: Found {len(clique_mismatch)} cliques in info but not in current split!")
            raise ValueError(f"Inconsistency detected: cliques in info don't match current split")
        
        if self.verbose:
            print(f"✓ Perfect consistency achieved:")
            print(f"  - Cliques: {len(self.dataset.clique)}")
            print(f"  - Versions: {len(self.dataset.versions)}")
            print(f"  - Info entries: {len(self.dataset.info)}")
    
    def validate_data_structures(self):
        """Debug method to validate that info, clique, and versions are consistent"""
        print(f"\n=== VALIDATING DATA STRUCTURES FOR SPLIT: {self.dataset.split} ===")
        
        print(f"Info dict entries: {len(self.dataset.info)}")
        print(f"Current split cliques: {len(self.dataset.clique)}")
        print(f"Versions list: {len(self.dataset.versions)}")
        
        # Check consistency
        versions_in_info = 0
        versions_not_in_info = []
        for version in self.dataset.versions:
            if version in self.dataset.info:
                versions_in_info += 1
            else:
                versions_not_in_info.append(version)
        
        print(f"Versions found in info: {versions_in_info}/{len(self.dataset.versions)}")
        if versions_not_in_info:
            print(f"ERROR: {len(versions_not_in_info)} versions NOT found in info:")
            for v in versions_not_in_info[:5]:
                print(f"  Missing: {v}")
        else:
            print("✓ All versions found in info dict")
        
        # Check that all info entries belong to current split
        info_splits = {}
        for version, data in self.dataset.info.items():
            clique = data['clique']
            if clique in self.dataset.clique:
                info_splits['current_split'] = info_splits.get('current_split', 0) + 1
            else:
                info_splits['other_split'] = info_splits.get('other_split', 0) + 1
        
        print(f"Info entries by split:")
        print(f"  Current split ({self.dataset.split}): {info_splits.get('current_split', 0)}")
        print(f"  Other splits: {info_splits.get('other_split', 0)}")
        
        if info_splits.get('other_split', 0) > 0:
            print(f"ERROR: Info contains versions from other splits!")
        else:
            print(f"✓ All info entries belong to current split")
        
        print(f"=== END VALIDATION ===\n")
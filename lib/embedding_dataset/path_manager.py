"""
Path management methods extracted from original EmbeddingDataset.
"""
from pathlib import Path
import os


class PathManager:
    """Handles path construction for embeddings and audio"""
    
    def __init__(self, dataset, verbose=True):
        self.dataset = dataset
        self.verbose = verbose
    
    def get_embedding_path(self, version):
        """Get the path to the embedding file for a given version"""
        hidden_states_path = Path(self.dataset.conf.path.hidden_states)
        required_filename = self.dataset._get_required_embedding_filename()
        
        if self.dataset.dataset_name == 'shs':
            if '-' not in version:
                return None
            set_id, ver_id = version.split('-', 1)
            
            possible_folders = [
                set_id,
                f"{set_id}-" if set_id.isdigit() and int(set_id) < 10 else set_id,
                set_id[:2] if len(set_id) > 2 else set_id
            ]
            
            for folder_name in possible_folders:
                version_folder = hidden_states_path / folder_name / f"{set_id}-{ver_id}"
                embedding_file = version_folder / required_filename
                if embedding_file.exists():
                    return embedding_file
                    
        elif self.dataset.dataset_name == 'lyric-covers':
            version_folder = hidden_states_path / version
            embedding_file = version_folder / required_filename
            if embedding_file.exists():
                return embedding_file
                
        elif self.dataset.dataset_name == 'discogs-vi':
            version_folder = hidden_states_path / version.replace('/', os.sep)
            embedding_file = version_folder / required_filename
            if embedding_file.exists():
                return embedding_file
        
        return None
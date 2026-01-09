import torch 
from pathlib import Path 
import os 

def define_save_folders(dataset_name, data_folder):
    """
    Define folder names for saving transcriptions and hidden states.
    """
    # Define the folder names based on the dataset name
    if dataset_name == "shs":
        dataset_transcription_folder = "SHS100K-transcriptions"
        dataset_encoder_embeddings_folder = "SHS100K-encoder-embeddings"
        dataset_hidden_states_folder = "SHS100K-hidden-states"
    elif dataset_name == "lyric-covers":
        dataset_transcription_folder = "LyricCovers-transcriptions"
        dataset_encoder_embeddings_folder = "LyricCovers-encoder-embeddings"
        dataset_hidden_states_folder = "LyricCovers-hidden-states"
    elif dataset_name == "discogs-vi":
        dataset_transcription_folder = "DiscogsVI-transcriptions"
        dataset_encoder_embeddings_folder = "DiscogsVI-encoder-embeddings"
        dataset_hidden_states_folder = "DiscogsVI-hidden-states"
    
    # Create the full paths for the transcription and hidden states folders
    transcription_path = os.path.join(data_folder, dataset_transcription_folder)
    hidden_states_path = os.path.join(data_folder, dataset_hidden_states_folder)

    # Create the directories if they do not exist
    transcription_path = Path(transcription_path)
    hidden_states_path = Path(hidden_states_path)
    transcription_path.mkdir(parents=True, exist_ok=True)
    hidden_states_path.mkdir(parents=True, exist_ok=True)
    return transcription_path, hidden_states_path


# Modified feature extraction loop
def get_save_path_for_dataset(hidden_states_folder, dataset_name, clique_id, version_id, save_components):
    """
    Build the save path based on dataset structure
    """
    base_path = Path(hidden_states_folder)
    
    if dataset_name == "shs":
        # SHS: hidden_states/{set_folder}/{set_id-ver_id}/
        clique_folder, version_folder = save_components
        return base_path / clique_folder / version_folder
        
    elif dataset_name == "lyric-covers":  
        # Lyric Covers: hidden_states/{id}/
        version_folder = save_components[0]
        return base_path / version_folder
        
    elif dataset_name == "discogs-vi":
        # Discogs-VI: hidden_states/{base_filename_path}/
        # Handle potential subdirectories
        return base_path / Path(*save_components)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
def get_tensor_filename(embedding_type, embedding_format):
    """Get tensor filename based on embedding type and format"""
    if embedding_type == "encoder":
        if embedding_format == "concat":
            return "x_concat.pt"
        elif embedding_format == "all":
            return "x_all.pt"
    elif embedding_type == "hidden_states":
        if embedding_format == "all":
            return "hs_all.pt"
    elif embedding_type == "last_hidden_states":
        if embedding_format == "concat":
            return "hs_last_seq.pt"
        elif embedding_format == "all":
            return "hs_last_all.pt"
    elif embedding_type == "last_hidden_states_en":
        if embedding_format == "concat":
            return "hs_last_seq_en.pt"
        elif embedding_format == "all":
            return "hs_last_all_en.pt"
    elif embedding_type == "sbert":
        return "hs_sbert.pt"
    
    return None

def extract_path_info_for_dataset(audio_path, dataset_name):
    """
    Extract clique_id, version_id, and save_base_path based on dataset type
    
    Args:
        audio_path: Path to audio file
        dataset_name: Name of dataset ('shs', 'lyric-covers', 'discogs-vi')
    
    Returns:
        tuple: (clique_id, version_id, save_base_path_components)
    """
    audio_path = Path(audio_path)
    
    if dataset_name == "shs":
        # SHS structure: /path/to/SHS100K/audio/{set_folder}/{set_id-ver_id}.mp3
        # Example: /data/SHS100K/audio/0-/0-1.mp3
        clique_id = audio_path.parent.name  # "0-" 
        version_id = audio_path.stem        # "0-1"
        # For saving: use the clique_id (set_folder) and version_id
        return clique_id, version_id, (clique_id, version_id)
        
    elif dataset_name == "lyric-covers":
        # Lyric Covers structure: /path/to/LyricCovers/audio/{id}/{id}_audio.mp3  
        # Example: /data/LyricCovers/audio/12345/12345_audio.mp3
        version_id = audio_path.parent.name  # "12345"
        clique_id = version_id  # For lyric covers, we'll need to get clique from dataframe
        # For saving: use just the version_id as folder
        return clique_id, version_id, (version_id,)
        
    elif dataset_name == "discogs-vi":
        # Discogs-VI structure: /path/to/DiscogsVI/audio/something/something_1.mp3
        # We need the last 2 components: something/something_1.mp3
        version_id = audio_path.stem  # "something_1"
        clique_id = version_id  # For discogs-vi, we'll need to get clique from dataframe
        
        # Get the directory name and filename as the last 2 components
        dir_name = audio_path.parent.name  # "something"
        filename = audio_path.stem  # "something_1"
        save_components = (dir_name, filename)
    
        return clique_id, version_id, save_components
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# def save_sbert_embeddings(sbert_embedding, save_base_path, decoding_config_name=None):
#     """
#     Save SBERT embeddings to disk
    
#     Args:
#         sbert_embedding: SBERT embedding tensor
#         save_base_path: Base path for saving
#         decoding_config_name: Optional config name (for compatibility)
#     """
#     save_base_path.mkdir(parents=True, exist_ok=True)
    
#     sbert_path = save_base_path / "hs_sbert.pt"
    
#     pdb.set_trace()
#     torch.save(sbert_embedding, sbert_path)
    # if not sbert_path.exists():
    #     torch.save(sbert_embedding, sbert_path)
    #     torch.load(sbert_path)
def save_sbert_embeddings(sbert_embedding, save_base_path, decoding_config_name=None):
    """
    Save SBERT embeddings to disk
    """
    save_base_path.mkdir(parents=True, exist_ok=True)
    sbert_path = save_base_path / "hs_sbert.pt"

    # # Convert to numpy for easier inspection
    # emb_np = sbert_embedding.detach().cpu().numpy()

    # print(f"[save_sbert_embeddings] Saving to {sbert_path}, "
    #       f"shape={emb_np.shape}, "
    #       f"min={emb_np.min():.4f}, max={emb_np.max():.4f}, "
    #       f"mean={emb_np.mean():.4f}, std={emb_np.std():.4f}",
    #       flush=True)

    # # Check if it's all zeros
    # if np.allclose(emb_np, 0):
    #     print(f"[WARNING] Embedding is all zeros! {sbert_path}", flush=True)

    torch.save(sbert_embedding, sbert_path)
        
def save_transcription_and_latents(dataset_name, result, frames_audio_features, lyric_feature_seq, 
                 transcription_folder, hidden_states_folder, save_base_path, save_components,
                 decoding_config_name, save_transcription=False, 
                 save_encoder_embeddings=False, save_encoder_embeddings_seq=False,
                 save_hidden_states=False, save_last_hidden_states=False, 
                 save_last_hidden_states_seq=False, language=None):
    """
    Save various model outputs based on flags. Skips existing files.
    """
    
    save_base_path.mkdir(parents=True, exist_ok=True)
    
    if save_transcription:
        if dataset_name == "shs":
            transcription_save_path = Path(transcription_folder) / "transcriptions" / save_components[0] / save_components[1]
            detected_language_save_path = Path(transcription_folder) / "detected_language" / save_components[0] / save_components[1]
        elif dataset_name == "lyric-covers":
            transcription_save_path = Path(transcription_folder) / "transcriptions" / save_components[0] 
            detected_language_save_path = Path(transcription_folder) / "detected_language" / save_components[0] 
        elif dataset_name == "discogs-vi":
            print(save_components)
            transcription_save_path = Path(transcription_folder) / "transcriptions" / save_components[0] / save_components[1]
            detected_language_save_path = Path(transcription_folder) / "detected_language" / save_components[0] / save_components[1]
        
        transcription_save_path.mkdir(parents=True, exist_ok=True)
        txt_path = transcription_save_path / f"{decoding_config_name}.txt"
        detected_language_save_path.mkdir(parents=True, exist_ok=True)
        lang_path = detected_language_save_path / "detected_language.txt"

        if not txt_path.exists():
            with open(txt_path, 'w') as f:
                f.write(result['text'])
        
        if not lang_path.exists():
            with open(lang_path, 'w') as f:
                f.write(result['language'])
    
    if save_encoder_embeddings:
        # Option 1: Save as a single dictionary (faster I/O)
        if language == None:
            dict_path = save_base_path / "x_all.pt"
        else: 
             dict_path = save_base_path / f"x_all_{language}.pt"
        if not dict_path.exists():
            embeddings_dict = {f"x_{i}": emb.half() for i, emb in enumerate(frames_audio_features)}
            torch.save(embeddings_dict, dict_path)
        
        # Option 2: If you need individual files (with existence check)
        # for i, emb in enumerate(frames_audio_features):
        #     path = save_base_path / f"x_{i}.pt"
        #     if not path.exists():
        #         torch.save(emb.half(), path)
    
    if save_encoder_embeddings_seq:
        if language == None: 
            concat_path = save_base_path / "x_concat.pt"
        else: 
            concat_path = save_base_path / f"x_concat_{language}.pt"
        if not concat_path.exists():
            concat_embedding = torch.cat(frames_audio_features, dim=0)
            torch.save(concat_embedding.half(), concat_path)

    if save_hidden_states:
        # Option 1: Save all hidden states in one file (much faster)
        if language == None: 
            hs_path = save_base_path / "hs_all.pt"
        else: 
            hs_path = save_base_path / f"hs_all_{language}.pt"
        if not hs_path.exists():
            hs_dict = {}
            for chunk_idx, chunk in enumerate(result['frames_hidden_states']):
                for step_idx, step_hidden in enumerate(chunk):
                    hs = torch.stack(step_hidden, dim=0)
                    hs_dict[f"hs_{chunk_idx}_{step_idx}"] = hs.half()
            torch.save(hs_dict, hs_path)
        
        # Option 2: If you need individual files
        # for i, chunk in enumerate(result['frames_hidden_states']):
        #     for j, step_hidden in enumerate(chunk):
        #         path = save_base_path / f"hs_{i}_{j}.pt"
        #         if not path.exists():
        #             torch.save(torch.stack(step_hidden, dim=0).half(), path)

    if save_last_hidden_states:
        # Option 1: Save all in one file
        if language == None:     
            last_hs_path = save_base_path / "hs_last_all.pt"
        else: 
            last_hs_path = save_base_path / f"hs_last_all_{language}.pt"
        if not last_hs_path.exists():
            last_hs_dict = {
                f"hs_last_{i}_{j}": step_hidden.half()
                for i, chunk in enumerate(result['frames_last_hidden_states'])
                for j, step_hidden in enumerate(chunk)
            }
            torch.save(last_hs_dict, last_hs_path)
    
    if save_last_hidden_states_seq:
        if language == None: 
            seq_path = save_base_path / "hs_last_seq.pt"
        else: 
            seq_path = save_base_path / f"hs_last_seq_{language}.pt"
        if not seq_path.exists():
            torch.save(lyric_feature_seq.half(), seq_path)
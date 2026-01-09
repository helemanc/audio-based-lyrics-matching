"""
Extraction utilities for feature extraction scripts.

Provides helper functions for:
- Dataloader creation based on extraction type
- Evaluation tensor preparation
- Embedding extraction for evaluation
- Main extraction loop orchestration
"""

import os
from typing import Tuple, Optional, List
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import DictConfig
from lightning import Fabric

from lib import dataset
from lib.extractors import BaseExtractor
from utils import print_utils
from utils.latents_extraction_utils import (
    extract_path_info_for_dataset,
    get_save_path_for_dataset
)


# ============================================================================
# DATALOADER CREATION
# ============================================================================

def create_dataloader_for_extraction(
    conf: DictConfig,
    fabric: Fabric
) -> DataLoader:
    """
    Create appropriate dataloader based on extraction type.
    
    Automatically selects:
    - AudioDataset for Whisper/SBERT (raw audio + transcriptions)
    - EmbeddingDataset for WEALY/CLEWS (pre-extracted embeddings)
    
    Args:
        conf: Configuration object with extraction.type
        fabric: Fabric instance for distributed setup
    
    Returns:
        Configured dataloader
    
    Raises:
        ValueError: If extraction type not recognized
    
    Example:
        >>> dataloader = create_dataloader_for_extraction(conf, fabric)
    """
    extraction_type = conf.extraction.type
    
    if extraction_type in ['whisper', 'sbert']:
        # Audio dataset
        base_path = conf.path.base_path
        if "leonardo" not in base_path:
            base_path = os.path.join(os.getcwd(), base_path)
        
        dataloader = dataset.create_dataloader(
            dataset_name=conf.data.dataset_name,
            base_path=base_path,
            data_folder=conf.path.data,
            batch_size=conf.data.batch_size,
            whisper_set=conf.data.whisper_set,
            split=conf.data.split,
            evaluation_mode=conf.data.get('evaluation_mode', False),
            enforce_max_duration=conf.data.get('enforce_max_duration', False),
            num_workers=conf.data.get('nworkers', 4),
            debug_num_cliques=conf.data.get('debug_num_cliques', None),
            pin_memory=True,
            use_transcriptions=conf.data.use_transcriptions
        )
        
        return fabric.setup_dataloaders(dataloader)
        
    elif extraction_type == 'wealy':
        # Embedding dataset with dataset reference stored
        ds = dataset.EmbeddingDataset(
            conf,
            split=conf.data.split,
            embedding_type=conf.data.embedding_type,
            embedding_format=conf.data.embedding_format,
            verbose=fabric.is_global_zero
        )
        
        # Create collate function
        overlap_percentage = conf.extraction.get('overlap_percentage', 0.9)
        collate_fn = dataset.create_collate_fn(
            conf,
            deterministic=False,
            use_overlapping_chunks=True,
            overlap_percentage=overlap_percentage
        )
        
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=1,  # Process one song at a time
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Return both dataloader and dataset reference
        return fabric.setup_dataloaders(dataloader), ds
    
    elif extraction_type == 'clews':
        # CLEWS setup
        from utils.clews_utils import setup_clews_dataset, auto_setup_clews_paths
        from omegaconf import OmegaConf
        
        config_path, checkpoint_path = auto_setup_clews_paths(conf)
        clews_conf = OmegaConf.load(config_path)
        
        ds, dataloader = setup_clews_dataset(
            conf,
            clews_conf,
            conf.data.split,
            fabric,
            return_paths=True
        )
        
        return dataloader
    
    else:
        raise ValueError(
            f"Unknown extraction type: {extraction_type}. "
            f"Supported: whisper, sbert, wealy, clews"
        )

# ============================================================================
# EVALUATION HELPERS
# ============================================================================

def get_embedding_for_evaluation(
    features: dict,
    extraction_type: str
) -> Optional[torch.Tensor]:
    """
    Extract embedding tensor for evaluation from features dict.
    
    Note: Some extraction types (like WEALY concatenated) don't support
    evaluation during extraction and will return None.
    
    Args:
        features: Feature dictionary from extractor
        extraction_type: Type of extraction ('whisper', 'sbert', etc.)
    
    Returns:
        Embedding tensor or None if extraction type doesn't support evaluation
    """
    if extraction_type == 'whisper':
        # Use encoder embedding for Whisper
        return features.get('encoder_embedding')
    
    elif extraction_type == 'sbert':
        # SBERT embedding is already averaged
        return features.get('sbert_embedding')
    
    elif extraction_type == 'wealy':
        # WEALY concatenated mode does NOT support evaluation during extraction
        # Evaluation should be done separately at test time using saved chunks
        return None
    
    elif extraction_type == 'clews':
        # CLEWS embedding
        return features.get('clews_embedding')
    
    return None


def prepare_evaluation_tensors(
    all_c: List[torch.Tensor],
    all_i: List[torch.Tensor],
    all_z: List[torch.Tensor],
    fabric: Fabric
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare tensors for evaluation by gathering across GPUs.
    
    Concatenates batches and performs all_gather for distributed evaluation.
    
    Args:
        all_c: List of clique ID tensors
        all_i: List of version ID tensors
        all_z: List of embedding tensors
        fabric: Fabric instance for distributed operations
    
    Returns:
        Tuple of (query_c, query_i, query_z, cand_c, cand_i, cand_z)
        - query_*: Tensors for this GPU
        - cand_*: Tensors gathered from all GPUs (for comparison)
    
    Example:
        >>> query_c, query_i, query_z, cand_c, cand_i, cand_z = \
        ...     prepare_evaluation_tensors(all_c, all_i, all_z, fabric)
    """
    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    
    myprint("Preparing evaluation tensors...")
    
    # Concatenate batches
    all_c = torch.cat(all_c, dim=0)
    all_i = torch.cat(all_i, dim=0)
    all_z = torch.cat(all_z, dim=0)
    
    # Setup query tensors (this GPU)
    query_c = all_c
    query_i = all_i
    query_z = all_z.half()
    
    # Setup candidate tensors (all GPUs)
    cand_c = query_c.clone()
    cand_i = query_i.clone()
    cand_z = query_z.clone()
    
    # Gather from all GPUs
    fabric.barrier()
    cand_c = fabric.all_gather(cand_c)
    cand_i = fabric.all_gather(cand_i)
    cand_z = fabric.all_gather(cand_z)
    
    # Flatten gathered tensors
    cand_c = torch.cat(torch.unbind(cand_c, dim=0), dim=0)
    cand_i = torch.cat(torch.unbind(cand_i, dim=0), dim=0)
    cand_z = torch.cat(torch.unbind(cand_z, dim=0), dim=0)
    
    myprint(f"Query shape: {query_z.shape}, Candidate shape: {cand_z.shape}")
    
    return query_c, query_i, query_z, cand_c, cand_i, cand_z


# ============================================================================
# MAIN EXTRACTION LOOP
# ============================================================================

def run_extraction_loop(
    extractor: BaseExtractor,
    dataloader: DataLoader,
    model: any,
    conf: DictConfig,
    fabric: Fabric
) -> Tuple[Optional[torch.Tensor], ...]:
    """
    Run main extraction loop with progress tracking.
    
    Orchestrates:
    1. Feature extraction for each batch
    2. Saving features to disk
    3. Accumulating embeddings for evaluation
    4. Progress reporting
    
    Args:
        extractor: Feature extractor instance
        dataloader: Data loader
        model: Loaded model
        conf: Configuration object
        fabric: Fabric instance
    
    Returns:
        Tuple of (query_c, query_i, query_z, cand_c, cand_i, cand_z) if 
        evaluation enabled, else (None, None, None, None, None, None)
    
    Example:
        >>> extractor = create_extractor('whisper', conf, fabric)
        >>> model = extractor.load_model()
        >>> dataloader = create_dataloader_for_extraction(conf, fabric)
        >>> results = run_extraction_loop(extractor, dataloader, model, conf, fabric)
    """
    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    
    myprint("Starting feature extraction loop...")
    
    # For evaluation - separate encoder and decoder for Whisper
    all_c, all_i = [], []
    all_z_e, all_z_d = [], []  # Separate for Whisper
    all_z = []  # Single embedding for SBERT/CLEWS
    
    run_evaluation = conf.evaluation.get('run_evaluation', False)
    
    pbar = tqdm(dataloader, desc="Extract", disable=not fabric.is_global_zero)
    
    for batch in pbar:
        # Extract features for batch
        features_list = extractor.extract_features(batch, model)
        
        # Get batch IDs
        c, i = batch[0], batch[1]
        
        # Track batch IDs
        if run_evaluation:
            all_c.append(c)
            all_i.append(i)
        
        # Process each sample
        for j, features in enumerate(features_list):
            extractor.stats.total_files += 1
            
            # Save features
            try:
                save_path = features.get('save_base_path')
                if save_path:
                    extractor.save_features(
                        features,
                        save_path,
                        metadata={'save_components': features.get('save_components')}
                    )
                else:
                    # Fallback to using path from features
                    audio_path = features.get('audio_path')
                    if audio_path:
                        from utils.latents_extraction_utils import extract_path_info_for_dataset, get_save_path_for_dataset
                        
                        clique_id, version_id, save_components = extract_path_info_for_dataset(
                            audio_path, conf.data.dataset_name
                        )
                        
                        save_base_path = get_save_path_for_dataset(
                            extractor.hidden_states_folder,
                            conf.data.dataset_name,
                            clique_id,
                            version_id,
                            save_components
                        )
                        
                        extractor.save_features(
                            features,
                            save_base_path,
                            metadata={'save_components': save_components}
                        )
                
                if not features.get('skipped', False):
                    extractor.stats.extracted_files += 1
                else:
                    extractor.stats.skipped_files += 1
                    
            except Exception as e:
                myprint(f"Error saving features: {e}")
                extractor.stats.failed_files += 1
                continue
            
            # Accumulate for evaluation based on extractor type
            if run_evaluation:
                if conf.extraction.type == 'whisper':
                    # Whisper has separate encoder/decoder
                    encoder_emb = features.get('encoder_embedding')
                    decoder_emb = features.get('decoder_embedding')
                    
                    if encoder_emb is not None and decoder_emb is not None:
                        # Handle based on embedding_type
                        embedding_type = conf.data.embedding_type
                        
                        if embedding_type == "encoder":
                            all_z_e.append(encoder_emb.unsqueeze(0))
                            all_z_d.append(torch.zeros_like(encoder_emb).unsqueeze(0))
                        elif embedding_type in ["last_hidden_states", "last_hidden_states_en", "hidden_states"]:
                            all_z_d.append(decoder_emb.unsqueeze(0))
                            all_z_e.append(torch.zeros_like(decoder_emb).unsqueeze(0))
                        elif embedding_type == "evaluation_enc_dec":
                            all_z_e.append(encoder_emb.unsqueeze(0))
                            all_z_d.append(decoder_emb.unsqueeze(0))
                        else:
                            # Default: use both
                            all_z_e.append(encoder_emb.unsqueeze(0))
                            all_z_d.append(decoder_emb.unsqueeze(0))
                
                elif conf.extraction.type in ['sbert', 'clews']:
                    # SBERT/CLEWS have single embedding
                    embedding = features.get('sbert_embedding') or features.get('clews_embedding')
                    if embedding is not None:
                        all_z.append(embedding.unsqueeze(0))
                
                # WEALY doesn't support evaluation during extraction
        
        # Update progress
        pbar.set_postfix({
            'extracted': extractor.stats.extracted_files,
            'skipped': extractor.stats.skipped_files,
            'failed': extractor.stats.failed_files
        })
    
    myprint(f"\n✓ Extraction complete! {extractor.stats}")
    
    # Prepare for evaluation
    if run_evaluation:
        if conf.extraction.type == 'whisper' and all_z_e and all_z_d:
            # Whisper evaluation
            all_c = torch.cat(all_c, dim=0)
            all_i = torch.cat(all_i, dim=0)
            all_z_e = torch.cat(all_z_e, dim=0)
            all_z_d = torch.cat(all_z_d, dim=0)
            
            query_c, query_i = all_c, all_i
            query_z_e, query_z_d = all_z_e.half(), all_z_d.half()
            
            # Clone for candidates
            cand_c, cand_i = query_c.clone(), query_i.clone()
            cand_z_e, cand_z_d = query_z_e.clone(), query_z_d.clone()
            
            # Gather from all GPUs
            fabric.barrier()
            cand_c = fabric.all_gather(cand_c)
            cand_i = fabric.all_gather(cand_i)
            cand_z_e = fabric.all_gather(cand_z_e)
            cand_z_d = fabric.all_gather(cand_z_d)
            
            # Flatten
            cand_c = torch.cat(torch.unbind(cand_c, dim=0), dim=0)
            cand_i = torch.cat(torch.unbind(cand_i, dim=0), dim=0)
            cand_z_e = torch.cat(torch.unbind(cand_z_e, dim=0), dim=0)
            cand_z_d = torch.cat(torch.unbind(cand_z_d, dim=0), dim=0)
            
            myprint(f"Query shapes: z_e={query_z_e.shape}, z_d={query_z_d.shape}")
            myprint(f"Candidate shapes: c={cand_c.shape}, i={cand_i.shape}")
            
            # Return in format expected by run_evaluation
            # For Whisper: (query_z_e, query_z_d, cand_z_e, cand_z_d, ...)
            return query_c, query_i, (query_z_e, query_z_d), cand_c, cand_i, (cand_z_e, cand_z_d)
            
        elif conf.extraction.type in ['sbert', 'clews'] and all_z:
            # SBERT/CLEWS evaluation
            all_c = torch.cat(all_c, dim=0)
            all_i = torch.cat(all_i, dim=0)
            all_z = torch.cat(all_z, dim=0)
            
            query_c, query_i, query_z = all_c, all_i, all_z.half()
            cand_c, cand_i, cand_z = query_c.clone(), query_i.clone(), query_z.clone()
            
            # Gather from all GPUs
            fabric.barrier()
            cand_c = fabric.all_gather(cand_c)
            cand_i = fabric.all_gather(cand_i)
            cand_z = fabric.all_gather(cand_z)
            
            # Flatten
            cand_c = torch.cat(torch.unbind(cand_c, dim=0), dim=0)
            cand_i = torch.cat(torch.unbind(cand_i, dim=0), dim=0)
            cand_z = torch.cat(torch.unbind(cand_z, dim=0), dim=0)
            
            myprint(f"Query shape: {query_z.shape}, Candidates: {cand_z.shape}")
            
            return query_c, query_i, query_z, cand_c, cand_i, cand_z
    
    return (None, None, None, None, None, None)
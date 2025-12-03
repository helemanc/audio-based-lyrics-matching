"""
Collate functions for batching embeddings in DataLoader.
Handles single-modal and multimodal (WEALY+CLEWS, Whisper+CLEWS) embeddings.
"""
import torch
import random


def load_wealy_with_chunking(wealy_data, mode='random', deterministic_chunk_size=1500):
    """
    Load and chunk WEALY concatenated embeddings based on mode.
    
    Args:
        wealy_data: Dict with 'embeddings' (n_chunks, 512) and metadata
        mode: 'random', 'deterministic', or 'all'
        deterministic_chunk_size: For future use (currently ignored)
    
    Returns:
        WEALY embedding: (512,) for train/val, (n_chunks, 512) for test
    """
    # Handle new concatenated format
    if isinstance(wealy_data, dict) and 'embeddings' in wealy_data:
        wealy_embeddings = wealy_data['embeddings']  # Shape: (n_chunks, 512)
    else:
        # Legacy format fallback
        wealy_embeddings = wealy_data
        if wealy_embeddings.dim() == 1:
            wealy_embeddings = wealy_embeddings.unsqueeze(0)  # (1, 512)
        elif wealy_embeddings.dim() == 0:
            wealy_embeddings = wealy_embeddings.unsqueeze(0).unsqueeze(0)  # (1, 1) -> (1, 1)
    
    n_chunks = wealy_embeddings.shape[0]
    
    if mode == 'random':
        # Training: Random chunk selection
        if n_chunks == 1:
            return wealy_embeddings[0]  # (512,)
        else:
            random_idx = torch.randint(0, n_chunks, (1,)).item()
            return wealy_embeddings[random_idx]  # (512,)
    
    elif mode == 'deterministic':
        # Validation: Always first chunk
        return wealy_embeddings[0]  # (512,)
    
    elif mode == 'all':
        # Test: Return all chunks
        return wealy_embeddings  # (n_chunks, 512)
    
    else:
        raise ValueError(f"Unknown WEALY chunking mode: {mode}")


def handle_wealy_test_mode(batch, n_per_class):
    """
    Handle test mode where we return all WEALY chunks per song.
    Returns a different structure optimized for test evaluation.
    """
    all_song_data = []
    
    batch_size = len(batch)
    
    for i, item in enumerate(batch):
        clique_id = item[0]
        
        for j in range(n_per_class):
            version_id = item[1 + j * 2]
            multimodal_dict = item[2 + j * 2]
            
            # Get all WEALY chunks for this song
            wealy_all_chunks = load_wealy_with_chunking(
                multimodal_dict['wealy'], 
                mode='all'
            )  # (n_chunks, 512)
            
            song_data = {
                'clique_id': clique_id,
                'version_id': version_id,
                'wealy_all_chunks': wealy_all_chunks,  # All chunks for evaluation
                'full_clews': multimodal_dict['full_clews'],
                'avg_clews': multimodal_dict['avg_clews'],
                'clews_mask': multimodal_dict['clews_mask'],
                'batch_idx': i,
                'version_idx': j
            }
            all_song_data.append(song_data)
    
    return all_song_data  # Return list of song dictionaries for test evaluation


def collate_embeddings_fixed_length(batch, use_random_chunks=False, 
                                   chunk_size=1000, use_overlapping_chunks=False, overlap_percentage=0.9,
                                   use_avg_pooling=False, embedding_type="whisper"):
    """
    Collate embeddings with support for different chunking strategies:
    - Standard: Fixed length truncation (max_length-chunk_size)
    - Random chunks: Random chunk of chunk_size (for training)
    - Deterministic: First chunk of chunk_size (for validation)  
    - Overlapping chunks: Multiple overlapping chunks (for test)
    - Average pooling: Average over time dimension (BS, TIME, EMB_SIZE) -> (BS, EMB_SIZE)
    
    Args:
        batch: List of items where each item contains clique_id, version_ids, and embeddings
        use_random_chunks: If True, use random chunking (training)
        chunk_size: Size of chunks for random/deterministic/overlapping modes
        use_overlapping_chunks: If True, generate overlapping chunks (test)
        overlap_percentage: Percentage of overlap between chunks (0.0 to 0.99)
        use_avg_pooling: If True, average over time dimension instead of chunking
        embedding_type: Type of embedding ("sbert", "clews", etc.) - controls processing logic
    
    Returns:
        If use_overlapping_chunks=False: Original format [clique_ids, version_ids, embeddings, masks, ...]
        If use_overlapping_chunks=True: [clique_ids, version_ids, embeddings, masks, chunk_info]
        If use_avg_pooling=True: [clique_ids, version_ids, embeddings, masks, ...] where embeddings are (BS, EMB_SIZE)
    """
    
    batch_size = len(batch)
    n_per_class = (len(batch[0]) - 1) // 2
    
    # Get embedding dimension - ADD NULL CHECK FOR SBERT
    first_emb = batch[0][2]
    if first_emb is None:
        raise ValueError("First embedding in batch is None - check SBERT extraction")
    embed_dim = first_emb.shape[-1]  # Works for both (1, embed_dim) and (seq_len, embed_dim)
    
    # EXTENDED DETECTION LOGIC - DETECT IF WE HAVE FIXED-SHAPE EMBEDDINGS
    is_sbert_like = first_emb.shape[0] == 1
    is_clews_like = embedding_type == "clews"  # CLEWS has fixed shape (16, 2048)
    is_fixed_shape = is_sbert_like or is_clews_like
    
    if use_avg_pooling:
        # Average pooling mode - works for SBERT, CLEWS, and Whisper
        
        # Pre-allocate all tensors
        clique_ids = torch.empty(batch_size, dtype=torch.long)
        output = [clique_ids]
        
        # Pre-allocate for each version position
        for j in range(n_per_class):
            version_ids = torch.empty(batch_size, dtype=torch.long)
            embeddings = torch.zeros(batch_size, embed_dim)  # No time dimension after averaging
            masks = torch.ones(batch_size, dtype=torch.bool)  # Always valid after averaging
            output.extend([version_ids, embeddings, masks])
        
        # Fill tensors with direct indexing
        for i, item in enumerate(batch):
            clique_ids[i] = item[0]
            
            for j in range(n_per_class):
                version_idx = 1 + j * 3
                emb_idx = 2 + j * 3
                mask_idx = 3 + j * 3
                
                output[version_idx][i] = item[1 + j * 2]
                
                emb = item[2 + j * 2]
                
                # ADD NULL CHECK AND HANDLE DIFFERENT EMBEDDING TYPES
                if emb is None:
                    # Handle missing embeddings
                    output[emb_idx][i] = torch.zeros(embed_dim)
                    output[mask_idx][i] = False  # Mark as invalid
                elif emb.shape[0] == 1:
                    # For SBERT (converted to shape (1, embed_dim)), use the single vector
                    output[emb_idx][i] = emb[0]  # Remove the time dimension
                    output[mask_idx][i] = True
                else:
                    # For CLEWS and Whisper, average over time dimension
                    output[emb_idx][i] = emb.mean(dim=0)
                    output[mask_idx][i] = True  # Always valid after averaging
        
        return output
    
    elif not use_overlapping_chunks:
        # Standard modes: fixed/random/deterministic chunking
        
        # Determine the actual length to use
        if use_random_chunks:
            actual_length = chunk_size  # Random chunks use chunk_size
        else:
            # For deterministic mode, we want first chunk_size, not max_length
            # This ensures val always uses first chunk of consistent size
            actual_length = chunk_size
        
        # OVERRIDE FOR FIXED-SHAPE EMBEDDINGS
        if is_sbert_like:
            actual_length = 1  # SBERT always uses length 1
        elif is_clews_like:
            actual_length = first_emb.shape[0]  # CLEWS uses its fixed length (16)
        
        # Pre-allocate all tensors
        clique_ids = torch.empty(batch_size, dtype=torch.long)
        output = [clique_ids]
        
        # Pre-allocate for each version position
        for j in range(n_per_class):
            version_ids = torch.empty(batch_size, dtype=torch.long)
            embeddings = torch.zeros(batch_size, actual_length, embed_dim)
            masks = torch.zeros(batch_size, actual_length, dtype=torch.bool)
            output.extend([version_ids, embeddings, masks])
        
        # Fill tensors with direct indexing
        for i, item in enumerate(batch):
            clique_ids[i] = item[0]
            
            for j in range(n_per_class):
                version_idx = 1 + j * 3
                emb_idx = 2 + j * 3
                mask_idx = 3 + j * 3
                
                output[version_idx][i] = item[1 + j * 2]
                
                emb = item[2 + j * 2]
                
                # ADD NULL CHECK
                if emb is None:
                    # Handle missing embeddings - fill with zeros and mark invalid
                    # This preserves tensor shapes
                    output[mask_idx][i, :] = False  # All positions invalid
                    continue
                
                # HANDLE DIFFERENT EMBEDDING TYPES
                if emb.shape[0] == 1:
                    # SBERT CASE: Simple handling
                    output[emb_idx][i, 0] = emb[0]  # Put the single vector at position 0
                    output[mask_idx][i, 0] = True   # Only position 0 is valid
                elif is_clews_like:
                    # CLEWS CASE: Take as-is, no chunking (shape: (16, 2048))
                    output[emb_idx][i, :] = emb  # Copy all timesteps
                    output[mask_idx][i, :] = True  # All positions valid
                else:
                    # DEFAULT CASE (Whisper and others): Original logic preserved
                    seq_len = emb.shape[0]
                    
                    if use_random_chunks:
                        # TRAIN: Random chunk of chunk_size
                        if seq_len <= chunk_size:
                            output[emb_idx][i, :seq_len] = emb
                            output[mask_idx][i, :seq_len] = True
                        else:
                            start_idx = torch.randint(0, seq_len - chunk_size + 1, (1,)).item()
                            end_idx = start_idx + chunk_size
                            output[emb_idx][i, :] = emb[start_idx:end_idx]
                            output[mask_idx][i, :] = True
                    else:
                        # VAL: Deterministic first chunk of chunk_size (always starts at 0)
                        use_len = min(seq_len, chunk_size)  # Use chunk_size, not max_length
                        output[emb_idx][i, :use_len] = emb[:use_len]  # Always start from 0
                        output[mask_idx][i, :use_len] = True
        
        return output
    
    else:
        # TEST: Overlapping chunks mode - ADD SUPPORT FOR FIXED-SHAPE EMBEDDINGS
        
        if is_fixed_shape:
            # FIXED-SHAPE EMBEDDINGS (SBERT and CLEWS): Simple case - one chunk per embedding
            all_chunks = []
            chunk_info = []
            
            # DETERMINE CHUNK SIZE FOR FIXED SHAPES
            if is_sbert_like:
                fixed_chunk_size = 1
            elif is_clews_like:
                fixed_chunk_size = first_emb.shape[0]  # Should be 16 for CLEWS
            else:
                fixed_chunk_size = 1  # Fallback
            
            for i, item in enumerate(batch):
                clique_id = item[0]
                
                for j in range(n_per_class):
                    version_id = item[1 + j * 2]
                    emb = item[2 + j * 2]  # Shape: (1, embed_dim) for SBERT or (16, 2048) for CLEWS
                    
                    # ADD NULL CHECK
                    if emb is None:
                        # Create zero chunk for missing embedding
                        chunk = torch.zeros(fixed_chunk_size, embed_dim)
                        mask = torch.zeros(fixed_chunk_size, dtype=torch.bool)  # Invalid
                    else:
                        # For fixed-shape embeddings, just one chunk with the full embedding
                        chunk = emb  # Shape: (1, embed_dim) or (16, 2048)
                        mask = torch.ones(fixed_chunk_size, dtype=torch.bool)  # All positions valid
                    
                    all_chunks.append((clique_id, version_id, chunk, mask))
                    chunk_info.append((i, j, 0))  # (original_batch_idx, version_idx, chunk_idx)
            
            # Convert to batch format
            total_chunks = len(all_chunks)
            
            clique_ids = torch.empty(total_chunks, dtype=torch.long)
            version_ids = torch.empty(total_chunks, dtype=torch.long)
            embeddings = torch.zeros(total_chunks, fixed_chunk_size, embed_dim)
            masks = torch.zeros(total_chunks, fixed_chunk_size, dtype=torch.bool)
            
            for idx, (clique_id, version_id, chunk, mask) in enumerate(all_chunks):
                clique_ids[idx] = clique_id
                version_ids[idx] = version_id
                embeddings[idx] = chunk
                masks[idx] = mask
            
            return [clique_ids, version_ids, embeddings, masks, chunk_info]
        
        else:
            # VARIABLE-LENGTH EMBEDDINGS (Whisper and others): Keep existing logic unchanged
            overlap_size = int(chunk_size * overlap_percentage)  
            stride = chunk_size - overlap_size                    
            stride = max(1, stride)
            
            # Collect all chunks and metadata
            all_chunks = []
            chunk_info = []  # [(original_batch_idx, original_version_idx, chunk_idx), ...]
            
            for i, item in enumerate(batch):
                clique_id = item[0]
                
                for j in range(n_per_class):
                    version_id = item[1 + j * 2]
                    emb = item[2 + j * 2]  # Shape: (seq_len, embed_dim)
                    
                    # ADD NULL CHECK
                    if emb is None:
                        # Create zero chunk for missing embedding
                        chunk = torch.zeros(chunk_size, embed_dim)
                        mask = torch.zeros(chunk_size, dtype=torch.bool)
                        all_chunks.append((clique_id, version_id, chunk, mask))
                        chunk_info.append((i, j, 0))
                        continue
                    
                    # Original overlapping chunks logic for variable-length embeddings
                    seq_len = emb.shape[0]
                    
                    if seq_len <= chunk_size:
                        # If sequence is shorter than chunk size, use as single chunk
                        chunk = torch.zeros(chunk_size, embed_dim)
                        mask = torch.zeros(chunk_size, dtype=torch.bool)
                        chunk[:seq_len] = emb
                        mask[:seq_len] = True
                        
                        all_chunks.append((clique_id, version_id, chunk, mask))
                        chunk_info.append((i, j, 0))
                    
                    else:
                        # Generate overlapping chunks
                        chunk_idx = 0
                        for start_pos in range(0, seq_len - chunk_size + 1, stride):
                            end_pos = start_pos + chunk_size
                            
                            chunk = emb[start_pos:end_pos]
                            mask = torch.ones(chunk_size, dtype=torch.bool)
                            
                            all_chunks.append((clique_id, version_id, chunk, mask))
                            chunk_info.append((i, j, chunk_idx))
                            chunk_idx += 1
            
            # Convert to batch format
            total_chunks = len(all_chunks)
            
            # Create output tensors
            clique_ids = torch.empty(total_chunks, dtype=torch.long)
            version_ids = torch.empty(total_chunks, dtype=torch.long)
            embeddings = torch.zeros(total_chunks, chunk_size, embed_dim)
            masks = torch.zeros(total_chunks, chunk_size, dtype=torch.bool)
            
            for idx, (clique_id, version_id, chunk, mask) in enumerate(all_chunks):
                clique_ids[idx] = clique_id
                version_ids[idx] = version_id
                embeddings[idx] = chunk
                masks[idx] = mask
            
            # Return format: [clique_ids, version_ids, embeddings, masks, chunk_info]
            # chunk_info helps map chunks back to original songs during evaluation
            return [clique_ids, version_ids, embeddings, masks, chunk_info]


def create_collate_fn(conf, deterministic=False, use_overlapping_chunks=False, overlap_percentage=0.9, use_avg_pooling=None, apply_masks_with_padding=False):
    """
    Create collate function with CONSISTENT batch structure for single-modal and multimodal variants.
    
    This function dynamically creates a collate function based on the model type specified in the configuration.
    It handles different embedding combinations and produces consistent interleaved batch formats for training.
    
    Args:
        conf: Configuration object containing model and data settings
        deterministic: If True, uses deterministic chunking (for validation)
        use_overlapping_chunks: If True, creates overlapping chunks for test evaluation
        overlap_percentage: Overlap ratio between chunks (0.0 to 0.99)
        use_avg_pooling: If True, averages embeddings over time dimension
        apply_masks_with_padding: If True, applies masks to remove invalid positions and pads to max valid length in batch
    
    Returns:
        Callable collate function that processes batches according to the model type
    
    Supported model types and their batch formats:
    - Single-modal models: [class_ids, ver_ids_1, embeddings_1, masks_1, ver_ids_2, embeddings_2, masks_2, ...]
      Standard format with 3 items per version (ver_id + embedding + mask)
      
    - multimodal-cross-attention: [class_ids, ver_ids_1, wealy_emb_1, full_clews_emb_1, avg_clews_emb_1, clews_mask_1, ver_ids_2, ...]
      Uses MultimodalEmbeddingDataset_WEALYCLEWS with frozen WEALY encoder
      5 items per version (ver_id + wealy_emb + full_clews_emb + avg_clews_emb + clews_mask)
      Provides both full temporal CLEWS (116x2048) and pre-computed averaged CLEWS (2048-dim)
      
    - multimodal-two-stream: [class_ids, ver_ids_1, whisper_emb_1, whisper_mask_1, full_clews_emb_1, avg_clews_emb_1, clews_mask_1, ...]
      Uses MultimodalEmbeddingDataset_WHISPERCLEWS with Whisper hs_last_seq
      6 items per version (ver_id + whisper_emb + whisper_mask + full_clews_emb + avg_clews_emb + clews_mask)
      Handles variable-length Whisper sequences with masks and CLEWS embeddings (116x2048)
      
    When apply_masks_with_padding=True:
      - Finds max valid length across batch for each masked embedding type
      - Applies masks to remove invalid positions 
      - Pads to max valid length for consistent tensor sizes
      - Updates masks accordingly for padded positions
    """
    
    # Determine use_avg_pooling setting
    if use_avg_pooling is None:
        use_avg_pooling = conf.data.get('use_avg_pooling', False)
    
    # Check for avg-clews option
    use_avg_clews = conf.data.get('use_avg_clews', False)
    
    # Get apply_masks_with_padding from config if not provided
    if apply_masks_with_padding is None:
        apply_masks_with_padding = conf.data.get('apply_masks_with_padding', False)
    
    # Check model type
    model_name = conf.model.get('name', 'whisper')
    is_wealy_clews = model_name in ['wealy-clews', 'multimodal-cross-attention', 'multimodal-concatenation', 'multimodal-cross-attention-residual']
    is_whisper_clews = model_name in ['whisper-clews', 'multimodal-two-stream']
    
    if is_wealy_clews:
        # WEALY+CLEWS model collate function with concatenated WEALY handling
        def wealy_clews_collate_fn(batch):
            """
            Updated to handle WEALY concatenated embeddings
            Produces format: [class_ids, ver_ids_1, wealy_emb_1, full_clews_emb_1, avg_clews_emb_1, clews_mask_1, 
                                         ver_ids_2, wealy_emb_2, full_clews_emb_2, avg_clews_emb_2, clews_mask_2, ...]
            Items per version: 5 (ver_id + wealy_emb + full_clews_emb + avg_clews_emb + clews_mask)
            """
            batch_size = len(batch)
            n_per_class = (len(batch[0]) - 1) // 2
            
            # Determine WEALY chunking mode based on evaluation settings
            if use_overlapping_chunks:
                wealy_mode = 'all'  # Test: use all chunks
            elif deterministic:
                wealy_mode = 'deterministic'  # Val: first chunk
            else:
                wealy_mode = 'random'  # Train: random chunk
            
            # Pre-allocate tensors
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]
            
            # Handle test mode differently (all chunks)
            if wealy_mode == 'all':
                return handle_wealy_test_mode(batch, n_per_class)
            
            # Training/Validation mode (single chunk per song)
            first_multimodal = batch[0][2]
            
            # Get WEALY embedding dimensions using chunking function
            sample_wealy = load_wealy_with_chunking(
                first_multimodal['wealy'], 
                mode=wealy_mode
            )
            wealy_dim = sample_wealy.numel()  # Should be 512
            
            # Get CLEWS dimensions (unchanged)
            full_clews_shape = first_multimodal['full_clews'].shape    # Should be (116, 2048)
            avg_clews_dim = first_multimodal['avg_clews'].shape[-1]    # Should be 2048
            clews_mask_dim = first_multimodal['clews_mask'].numel()    # Should be 116
            
            # Handle mask application with padding for CLEWS (unchanged)
            if apply_masks_with_padding:
                # Find max valid CLEWS length across all samples in batch
                max_clews_len = 0
                all_clews_data = []
                
                for item in batch:
                    for j in range(n_per_class):
                        multimodal_dict = item[2 + j * 2]
                        clews_mask = multimodal_dict['clews_mask']
                        valid_len = (~clews_mask).sum().item()  # Count valid positions
                        max_clews_len = max(max_clews_len, valid_len)
                        all_clews_data.append((multimodal_dict, valid_len))
                
                # Use max valid length for tensor allocation
                actual_clews_len = max_clews_len
            else:
                # Use original full length
                actual_clews_len = clews_mask_dim
            
            # Pre-allocate for each version position (5 items per version)
            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)
                wealy_embeddings = torch.zeros(batch_size, wealy_dim)  # Now single vector per song
                full_clews_embeddings = torch.zeros(batch_size, actual_clews_len, full_clews_shape[1])  # (B, actual_len, 2048)
                avg_clews_embeddings = torch.zeros(batch_size, avg_clews_dim)          # (B, 2048)
                clews_masks = torch.zeros(batch_size, actual_clews_len, dtype=torch.bool) # (B, actual_len)
                output.extend([version_ids, wealy_embeddings, full_clews_embeddings, avg_clews_embeddings, clews_masks])
            
            # Fill tensors
            data_idx = 0
            for i, item in enumerate(batch):
                clique_ids[i] = item[0]
                
                for j in range(n_per_class):
                    version_idx = 1 + j * 5      # positions: 1, 6, 11, ... (stride 5)
                    wealy_idx = 2 + j * 5        # positions: 2, 7, 12, ...
                    full_clews_idx = 3 + j * 5   # positions: 3, 8, 13, ...
                    avg_clews_idx = 4 + j * 5    # positions: 4, 9, 14, ...
                    clews_mask_idx = 5 + j * 5   # positions: 5, 10, 15, ...
                    
                    output[version_idx][i] = item[1 + j * 2]  # version_id
                    
                    multimodal_dict = item[2 + j * 2]
                    
                    # Load WEALY with appropriate chunking strategy
                    wealy_emb = load_wealy_with_chunking(
                        multimodal_dict['wealy'], 
                        mode=wealy_mode
                    )
                    # Ensure it's 1D for storage
                    if wealy_emb.dim() == 0:
                        wealy_emb = wealy_emb.unsqueeze(0)
                    elif wealy_emb.dim() > 1:
                        wealy_emb = wealy_emb.flatten()
                    output[wealy_idx][i] = wealy_emb
                    
                    # Handle CLEWS with optional mask application (unchanged)
                    if apply_masks_with_padding:
                        multimodal_dict_data, valid_len = all_clews_data[data_idx]
                        data_idx += 1
                        
                        full_clews_emb = multimodal_dict_data['full_clews']
                        clews_mask = multimodal_dict_data['clews_mask']
                        
                        # Apply mask to CLEWS embedding
                        valid_positions = ~clews_mask
                        if valid_len > 0:
                            masked_clews = full_clews_emb[valid_positions]  # (valid_len, 2048)
                            # Pad to max length
                            padded_clews = torch.zeros(actual_clews_len, full_clews_emb.shape[1])
                            padded_clews[:valid_len] = masked_clews
                            # Create new mask (True for padded positions)
                            new_mask = torch.zeros(actual_clews_len, dtype=torch.bool)
                            new_mask[valid_len:] = True
                        else:
                            # No valid positions - all zeros and all masked
                            padded_clews = torch.zeros(actual_clews_len, full_clews_emb.shape[1])
                            new_mask = torch.ones(actual_clews_len, dtype=torch.bool)
                        
                        output[full_clews_idx][i] = padded_clews
                        output[clews_mask_idx][i] = new_mask
                        output[avg_clews_idx][i] = multimodal_dict['avg_clews']
                    else:
                        # Original behavior - no mask application
                        full_clews_emb = multimodal_dict['full_clews']
                        avg_clews_emb = multimodal_dict['avg_clews']
                        clews_mask = multimodal_dict['clews_mask']
                        
                        # Handle CLEWS embeddings and mask
                        output[full_clews_idx][i] = full_clews_emb
                        output[avg_clews_idx][i] = avg_clews_emb
                        output[clews_mask_idx][i] = clews_mask
            
            return output
        
        return wealy_clews_collate_fn
        
    elif is_whisper_clews:
        # Whisper+CLEWS model collate function (handles two-stream models)
        def whisper_clews_collate_fn(batch):
            """
            Produces format: [class_ids, ver_ids_1, whisper_emb_1, whisper_mask_1, full_clews_emb_1, avg_clews_emb_1, clews_mask_1,
                                         ver_ids_2, whisper_emb_2, whisper_mask_2, full_clews_emb_2, avg_clews_emb_2, clews_mask_2, ...]
            Items per version: 6 (ver_id + whisper_emb + whisper_mask + full_clews_emb + avg_clews_emb + clews_mask)
            """
            batch_size = len(batch)
            n_per_class = (len(batch[0]) - 1) // 2
            
            # Pre-allocate tensors
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]
            
            # Get dimensions from first batch item
            first_multimodal = batch[0][2]
            whisper_shape = first_multimodal['whisper'].shape              # (seq_len, 1280)
            whisper_mask_dim = first_multimodal['whisper_mask'].numel()    # seq_len
            full_clews_shape = first_multimodal['full_clews'].shape        # Should be (116, 2048)
            avg_clews_dim = first_multimodal['avg_clews'].shape[-1]        # Should be 2048
            clews_mask_dim = first_multimodal['clews_mask'].numel()        # Should be 116
            
            # Handle mask application with padding if enabled
            if apply_masks_with_padding:
                # Find max valid lengths for both Whisper and CLEWS across batch
                max_whisper_len = 0
                max_clews_len = 0
                all_multimodal_data = []
                
                for item in batch:
                    for j in range(n_per_class):
                        multimodal_dict = item[2 + j * 2]
                        whisper_mask = multimodal_dict['whisper_mask']
                        clews_mask = multimodal_dict['clews_mask']
                        
                        valid_whisper_len = (~whisper_mask).sum().item()
                        valid_clews_len = (~clews_mask).sum().item()
                        
                        max_whisper_len = max(max_whisper_len, valid_whisper_len)
                        max_clews_len = max(max_clews_len, valid_clews_len)
                        
                        all_multimodal_data.append((multimodal_dict, valid_whisper_len, valid_clews_len))
                
                actual_whisper_len = max_whisper_len
                actual_clews_len = max_clews_len
            else:
                actual_whisper_len = whisper_mask_dim
                actual_clews_len = clews_mask_dim
            
            # Pre-allocate for each version position (6 items per version)
            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)
                whisper_embeddings = torch.zeros(batch_size, actual_whisper_len, whisper_shape[1])  # (B, actual_whisper_len, 1280)
                whisper_masks = torch.zeros(batch_size, actual_whisper_len, dtype=torch.bool)       # (B, actual_whisper_len)
                full_clews_embeddings = torch.zeros(batch_size, actual_clews_len, full_clews_shape[1])  # (B, actual_clews_len, 2048)
                avg_clews_embeddings = torch.zeros(batch_size, avg_clews_dim)                       # (B, 2048)
                clews_masks = torch.zeros(batch_size, actual_clews_len, dtype=torch.bool)          # (B, actual_clews_len)
                output.extend([version_ids, whisper_embeddings, whisper_masks, 
                              full_clews_embeddings, avg_clews_embeddings, clews_masks])
            
            # Fill tensors
            data_idx = 0
            for i, item in enumerate(batch):
                clique_ids[i] = item[0]
                
                for j in range(n_per_class):
                    version_idx = 1 + j * 6          # positions: 1, 7, 13, ... (stride 6)
                    whisper_idx = 2 + j * 6          # positions: 2, 8, 14, ...
                    whisper_mask_idx = 3 + j * 6     # positions: 3, 9, 15, ...
                    full_clews_idx = 4 + j * 6       # positions: 4, 10, 16, ...
                    avg_clews_idx = 5 + j * 6        # positions: 5, 11, 17, ...
                    clews_mask_idx = 6 + j * 6       # positions: 6, 12, 18, ...
                    
                    output[version_idx][i] = item[1 + j * 2]  # version_id
                    
                    if apply_masks_with_padding:
                        # Apply masks ONLY to CLEWS, keep Whisper unchanged
                        multimodal_dict, valid_whisper_len, valid_clews_len = all_multimodal_data[data_idx]
                        data_idx += 1
                        
                        whisper_emb = multimodal_dict['whisper']
                        whisper_mask = multimodal_dict['whisper_mask']
                        full_clews_emb = multimodal_dict['full_clews']
                        clews_mask = multimodal_dict['clews_mask']
                        
                        # Keep Whisper unchanged (no mask application)
                        output[whisper_idx][i] = whisper_emb
                        output[whisper_mask_idx][i] = whisper_mask
                        
                        # Apply mask ONLY to CLEWS embedding
                        valid_clews_positions = ~clews_mask
                        if valid_clews_len > 0:
                            masked_clews = full_clews_emb[valid_clews_positions]  # (valid_clews_len, 2048)
                            padded_clews = torch.zeros(actual_clews_len, full_clews_emb.shape[1])
                            padded_clews[:valid_clews_len] = masked_clews
                            new_clews_mask = torch.zeros(actual_clews_len, dtype=torch.bool)
                            new_clews_mask[valid_clews_len:] = True
                        else:
                            padded_clews = torch.zeros(actual_clews_len, full_clews_emb.shape[1])
                            new_clews_mask = torch.ones(actual_clews_len, dtype=torch.bool)
                        
                        output[full_clews_idx][i] = padded_clews
                        output[clews_mask_idx][i] = new_clews_mask
                        output[avg_clews_idx][i] = multimodal_dict['avg_clews']
                    else:
                        # Original behavior - no mask application
                        multimodal_dict = item[2 + j * 2]
                        whisper_emb = multimodal_dict['whisper']
                        whisper_mask = multimodal_dict['whisper_mask']
                        full_clews_emb = multimodal_dict['full_clews']
                        avg_clews_emb = multimodal_dict['avg_clews']
                        clews_mask = multimodal_dict['clews_mask']
                        
                        # Handle all embeddings and masks
                        output[whisper_idx][i] = whisper_emb
                        output[whisper_mask_idx][i] = whisper_mask
                        output[full_clews_idx][i] = full_clews_emb
                        output[avg_clews_idx][i] = avg_clews_emb
                        output[clews_mask_idx][i] = clews_mask
            
            return output
        
        return whisper_clews_collate_fn
    
    else:
        # Single-modal collate function
        embedding_type = conf.data.get('embedding_type', 'whisper')
        if embedding_type == 'clews':
            embedding_type = 'clews'
        else:
            embedding_type = 'whisper'
            
        if use_avg_pooling:
            return lambda batch: collate_embeddings_fixed_length(
                batch, use_avg_pooling=True, embedding_type=embedding_type)
        elif use_overlapping_chunks:
            return lambda batch: collate_embeddings_fixed_length(
                batch, use_random_chunks=False,
                chunk_size=conf.data.get('chunk_size', 1000),
                use_overlapping_chunks=True, overlap_percentage=overlap_percentage,
                embedding_type=embedding_type)
        else:
            if deterministic:
                return lambda batch: collate_embeddings_fixed_length(
                    batch, use_random_chunks=False,
                    chunk_size=conf.data.get('chunk_size', 1000), embedding_type=embedding_type)
            else:
                return lambda batch: collate_embeddings_fixed_length(
                    batch, use_random_chunks=conf.data.get('use_random_chunks', False),
                    chunk_size=conf.data.get('chunk_size', 1000), embedding_type=embedding_type)
"""
Collate functions for batching embeddings in DataLoader.

Handles single-modal (SBERT, CLEWS, Whisper) and multimodal (WEALY+CLEWS,
Whisper+CLEWS) embeddings with multiple chunking strategies.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


def load_wealy_with_chunking(
    wealy_data: Union[Dict[str, torch.Tensor], torch.Tensor],
    mode: str = "random",
    deterministic_chunk_size: int = 1500,
) -> torch.Tensor:
    """
    Load and chunk WEALY concatenated embeddings.

    Args:
        wealy_data: Dict with 'embeddings' key or raw tensor
        mode: Chunking strategy:
            - 'random': Random chunk (training)
            - 'deterministic': First chunk (validation)
            - 'all': All chunks (test)
        deterministic_chunk_size: Reserved for future use

    Returns:
        WEALY embedding:
            - (512,) for random/deterministic modes
            - (n_chunks, 512) for 'all' mode

    Raises:
        ValueError: If mode not recognized
    """
    # Extract embeddings
    if isinstance(wealy_data, dict) and "embeddings" in wealy_data:
        wealy_embeddings = wealy_data["embeddings"]
    else:
        # Legacy format fallback
        wealy_embeddings = wealy_data
        if wealy_embeddings.dim() == 1:
            wealy_embeddings = wealy_embeddings.unsqueeze(0)
        elif wealy_embeddings.dim() == 0:
            wealy_embeddings = wealy_embeddings.unsqueeze(0).unsqueeze(0)

    n_chunks = wealy_embeddings.shape[0]

    if mode == "random":
        if n_chunks == 1:
            return wealy_embeddings[0]
        else:
            idx = torch.randint(0, n_chunks, (1,)).item()
            return wealy_embeddings[idx]

    elif mode == "deterministic":
        return wealy_embeddings[0]

    elif mode == "all":
        return wealy_embeddings

    else:
        raise ValueError(f"Unknown WEALY chunking mode: {mode}")


def handle_wealy_test_mode(batch: List[Any], n_per_class: int) -> List[Dict[str, Any]]:
    """
    Handle test mode returning all WEALY chunks per song.

    Args:
        batch: List of items with structure:
            [clique_id, ver_id_1, multimodal_dict_1, ver_id_2, multimodal_dict_2, ...]
        n_per_class: Number of versions per clique

    Returns:
        List of dicts containing:
            - clique_id: int
            - version_id: int
            - wealy_all_chunks: Tensor (n_chunks, 512)
            - full_clews: Tensor
            - avg_clews: Tensor
            - clews_mask: Tensor
            - batch_idx: int
            - version_idx: int
    """
    all_song_data = []

    for i, item in enumerate(batch):
        clique_id = item[0]

        for j in range(n_per_class):
            version_id = item[1 + j * 2]
            multimodal_dict = item[2 + j * 2]

            # Get all chunks
            wealy_all_chunks = load_wealy_with_chunking(
                multimodal_dict["wealy"], mode="all"
            )

            song_data = {
                "clique_id": clique_id,
                "version_id": version_id,
                "wealy_all_chunks": wealy_all_chunks,
                "full_clews": multimodal_dict["full_clews"],
                "avg_clews": multimodal_dict["avg_clews"],
                "clews_mask": multimodal_dict["clews_mask"],
                "batch_idx": i,
                "version_idx": j,
            }
            all_song_data.append(song_data)

    return all_song_data


def collate_embeddings_fixed_length(
    batch: List[Any],
    use_random_chunks: bool = False,
    chunk_size: int = 1000,
    use_overlapping_chunks: bool = False,
    overlap_percentage: float = 0.9,
    use_avg_pooling: bool = False,
    embedding_type: str = "whisper",
) -> List[torch.Tensor]:
    """
    Collate embeddings with multiple chunking strategies.

    Supports:
    - Standard: Fixed-length truncation
    - Random chunks: Random selection (training)
    - Deterministic: First chunk (validation)
    - Overlapping: Multiple overlapping chunks (test)
    - Average pooling: Average over time dimension

    Args:
        batch: List of items [clique_id, ver_id_1, emb_1, ver_id_2, emb_2, ...]
        use_random_chunks: Use random chunking (training)
        chunk_size: Size of chunks
        use_overlapping_chunks: Generate overlapping chunks (test)
        overlap_percentage: Overlap between chunks (0.0-0.99)
        use_avg_pooling: Average over time dimension
        embedding_type: 'sbert', 'clews', 'whisper', etc.

    Returns:
        Standard/avg: [clique_ids, ver_ids_1, embs_1, masks_1, ver_ids_2, embs_2, masks_2, ...]
        Overlapping: [clique_ids, ver_ids, embs, masks, chunk_info]
            - Stride: 3 items per version

    Raises:
        ValueError: If first embedding is None
    """
    batch_size = len(batch)
    n_per_class = (len(batch[0]) - 1) // 2

    # Get embedding dimension
    first_emb = batch[0][2]
    if first_emb is None:
        raise ValueError("First embedding is None - check extraction")
    embed_dim = first_emb.shape[-1]

    # Detect fixed-shape embeddings
    is_sbert_like = first_emb.shape[0] == 1
    is_clews_like = embedding_type == "clews"
    is_fixed_shape = is_sbert_like or is_clews_like

    if use_avg_pooling:
        # Average pooling mode
        clique_ids = torch.empty(batch_size, dtype=torch.long)
        output = [clique_ids]

        # Pre-allocate for each version
        for j in range(n_per_class):
            version_ids = torch.empty(batch_size, dtype=torch.long)
            embeddings = torch.zeros(batch_size, embed_dim)
            masks = torch.ones(batch_size, dtype=torch.bool)
            output.extend([version_ids, embeddings, masks])

        # Fill tensors
        for i, item in enumerate(batch):
            clique_ids[i] = item[0]

            for j in range(n_per_class):
                version_idx = 1 + j * 3
                emb_idx = 2 + j * 3
                mask_idx = 3 + j * 3

                output[version_idx][i] = item[1 + j * 2]

                emb = item[2 + j * 2]

                if emb is None:
                    output[emb_idx][i] = torch.zeros(embed_dim)
                    output[mask_idx][i] = False
                elif emb.shape[0] == 1:
                    # SBERT: (1, embed_dim)
                    output[emb_idx][i] = emb[0]
                    output[mask_idx][i] = True
                else:
                    # Average over time
                    output[emb_idx][i] = emb.mean(dim=0)
                    output[mask_idx][i] = True

        return output

    elif use_overlapping_chunks:
        # Overlapping chunks mode (test)
        if is_fixed_shape:
            # Fixed-shape: no chunking needed
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]

            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)

                if is_sbert_like:
                    embeddings = torch.zeros(batch_size, embed_dim)
                    masks = torch.ones(batch_size, dtype=torch.bool)
                elif is_clews_like:
                    embeddings = torch.zeros(batch_size, 16, embed_dim)
                    masks = torch.ones(batch_size, 16, dtype=torch.bool)

                output.extend([version_ids, embeddings, masks])

            for i, item in enumerate(batch):
                clique_ids[i] = item[0]

                for j in range(n_per_class):
                    version_idx = 1 + j * 3
                    emb_idx = 2 + j * 3
                    mask_idx = 3 + j * 3

                    output[version_idx][i] = item[1 + j * 2]
                    emb = item[2 + j * 2]

                    if emb is None:
                        output[mask_idx][i] = False
                    else:
                        if is_sbert_like:
                            output[emb_idx][i] = emb[0]
                        else:
                            output[emb_idx][i] = emb
                        output[mask_idx][i] = True

            chunk_info = {"num_chunks_per_song": 1}
            output.append(chunk_info)
            return output

        else:
            # Variable-length: create overlapping chunks
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
                    emb = item[2 + j * 2]

                    if emb is None:
                        # Create zero chunk for missing embedding
                        chunk = torch.zeros(chunk_size, embed_dim)
                        mask = torch.zeros(chunk_size, dtype=torch.bool)
                        all_chunks.append((clique_id, version_id, chunk, mask))
                        chunk_info.append((i, j, 0))
                        continue

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

            clique_ids = torch.empty(total_chunks, dtype=torch.long)
            version_ids = torch.empty(total_chunks, dtype=torch.long)
            embeddings = torch.zeros(total_chunks, chunk_size, embed_dim)
            masks = torch.zeros(total_chunks, chunk_size, dtype=torch.bool)

            for idx, (clique_id, version_id, chunk, mask) in enumerate(all_chunks):
                clique_ids[idx] = clique_id
                version_ids[idx] = version_id
                embeddings[idx] = chunk
                masks[idx] = mask

            return [clique_ids, version_ids, embeddings, masks, chunk_info]

    else:
        # Standard or random chunking
        if is_fixed_shape:
            # Fixed-shape embeddings (SBERT, CLEWS)
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]

            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)

                if is_sbert_like:
                    embeddings = torch.zeros(batch_size, embed_dim)
                    masks = torch.ones(batch_size, dtype=torch.bool)
                elif is_clews_like:
                    embeddings = torch.zeros(batch_size, 16, embed_dim)
                    masks = torch.ones(batch_size, 16, dtype=torch.bool)

                output.extend([version_ids, embeddings, masks])

            for i, item in enumerate(batch):
                clique_ids[i] = item[0]

                for j in range(n_per_class):
                    version_idx = 1 + j * 3
                    emb_idx = 2 + j * 3
                    mask_idx = 3 + j * 3

                    output[version_idx][i] = item[1 + j * 2]
                    emb = item[2 + j * 2]

                    if emb is None:
                        output[mask_idx][i] = False
                    else:
                        if is_sbert_like:
                            output[emb_idx][i] = emb[0]
                        else:
                            output[emb_idx][i] = emb
                        output[mask_idx][i] = True

            return output

        else:
            # Variable-length embeddings
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]

            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)
                embeddings = torch.zeros(batch_size, chunk_size, embed_dim)
                masks = torch.zeros(batch_size, chunk_size, dtype=torch.bool)
                output.extend([version_ids, embeddings, masks])

            for i, item in enumerate(batch):
                clique_ids[i] = item[0]

                for j in range(n_per_class):
                    version_idx = 1 + j * 3
                    emb_idx = 2 + j * 3
                    mask_idx = 3 + j * 3

                    output[version_idx][i] = item[1 + j * 2]
                    emb = item[2 + j * 2]

                    if emb is None:
                        output[mask_idx][i, :] = True
                    else:
                        seq_len = emb.shape[0]

                        if use_random_chunks and seq_len > chunk_size:
                            max_start = seq_len - chunk_size
                            start = random.randint(0, max_start)
                        else:
                            start = 0

                        end = min(start + chunk_size, seq_len)
                        actual_len = end - start

                        output[emb_idx][i, :actual_len] = emb[start:end]
                        output[mask_idx][i, actual_len:] = True

            return output


def create_collate_fn(
    conf: Any,
    deterministic: bool = False,
    use_avg_pooling: bool = False,
    use_overlapping_chunks: bool = False,
    overlap_percentage: float = 0.9,
    apply_masks_with_padding: bool = False,
    wealy_test_mode: bool = False,
) -> Callable:
    """
    Factory function to create appropriate collate function.

    Args:
        conf: Configuration object with data settings
        deterministic: Use deterministic chunking (validation)
        use_avg_pooling: Average over time dimension
        use_overlapping_chunks: Generate overlapping chunks (test)
        overlap_percentage: Overlap between chunks (0.0-0.99)
        apply_masks_with_padding: Apply masks and pad to valid lengths
        wealy_test_mode: Return list of song dicts (WEALY test)

    Returns:
        Collate function appropriate for the model type

    Examples:
        >>> # Single-modal
        >>> collate_fn = create_collate_fn(conf)
        >>>
        >>> # Multimodal WEALY+CLEWS
        >>> collate_fn = create_collate_fn(conf, wealy_test_mode=True)
        >>>
        >>> # Test with overlapping chunks
        >>> collate_fn = create_collate_fn(conf, use_overlapping_chunks=True)
    """
    is_wealy_clews = conf.data.get("embedding_type") == "multimodal_wealy_clews"
    is_whisper_clews = conf.data.get("embedding_type") == "multimodal_whisper_clews"

    if is_wealy_clews:
        # WEALY+CLEWS collate function
        def wealy_clews_collate_fn(
            batch: List[Any],
        ) -> Union[List[Dict], List[torch.Tensor]]:
            """
            Collate WEALY+CLEWS multimodal data.

            Format: [clique_ids, ver_id_1, wealy_1, full_clews_1, avg_clews_1, clews_mask_1,
                                  ver_id_2, wealy_2, full_clews_2, avg_clews_2, clews_mask_2, ...]
            Stride: 5 items per version

            Args:
                batch: List of multimodal items

            Returns:
                Test mode: List of song dicts
                Normal: List of tensors
            """
            if wealy_test_mode:
                n_per_class = (len(batch[0]) - 1) // 2
                return handle_wealy_test_mode(batch, n_per_class)

            batch_size = len(batch)
            n_per_class = (len(batch[0]) - 1) // 2

            # Determine WEALY chunking mode
            if use_avg_pooling:
                wealy_mode = "deterministic"  # Use first chunk then average
            elif deterministic:
                wealy_mode = "deterministic"
            else:
                wealy_mode = "random"

            # Pre-allocate
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]

            # Get dimensions
            first_multimodal = batch[0][2]
            wealy_dim = 512
            full_clews_shape = first_multimodal["full_clews"].shape
            avg_clews_dim = first_multimodal["avg_clews"].shape[-1]
            clews_mask_dim = first_multimodal["clews_mask"].numel()

            # Pre-allocate for each version (5 items per version)
            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)
                wealy_embeddings = torch.zeros(batch_size, wealy_dim)
                full_clews_embeddings = torch.zeros(
                    batch_size, full_clews_shape[0], full_clews_shape[1]
                )
                avg_clews_embeddings = torch.zeros(batch_size, avg_clews_dim)
                clews_masks = torch.zeros(batch_size, clews_mask_dim, dtype=torch.bool)
                output.extend(
                    [
                        version_ids,
                        wealy_embeddings,
                        full_clews_embeddings,
                        avg_clews_embeddings,
                        clews_masks,
                    ]
                )

            # Fill tensors
            for i, item in enumerate(batch):
                clique_ids[i] = item[0]

                for j in range(n_per_class):
                    version_idx = 1 + j * 5
                    wealy_idx = 2 + j * 5
                    full_clews_idx = 3 + j * 5
                    avg_clews_idx = 4 + j * 5
                    clews_mask_idx = 5 + j * 5

                    output[version_idx][i] = item[1 + j * 2]

                    multimodal_dict = item[2 + j * 2]

                    # Load WEALY with chunking
                    wealy_emb = load_wealy_with_chunking(
                        multimodal_dict["wealy"], mode=wealy_mode
                    )

                    output[wealy_idx][i] = wealy_emb
                    output[full_clews_idx][i] = multimodal_dict["full_clews"]
                    output[avg_clews_idx][i] = multimodal_dict["avg_clews"]
                    output[clews_mask_idx][i] = multimodal_dict["clews_mask"]

            return output

        return wealy_clews_collate_fn

    elif is_whisper_clews:
        # Whisper+CLEWS collate function
        def whisper_clews_collate_fn(batch: List[Any]) -> List[torch.Tensor]:
            """
            Collate Whisper+CLEWS multimodal data.

            Format: [clique_ids, ver_id_1, whisper_1, whisper_mask_1, full_clews_1, avg_clews_1, clews_mask_1,
                                  ver_id_2, whisper_2, whisper_mask_2, full_clews_2, avg_clews_2, clews_mask_2, ...]
            Stride: 6 items per version

            Args:
                batch: List of multimodal items

            Returns:
                List of tensors
            """
            batch_size = len(batch)
            n_per_class = (len(batch[0]) - 1) // 2

            # Pre-allocate
            clique_ids = torch.empty(batch_size, dtype=torch.long)
            output = [clique_ids]

            # Get dimensions
            first_multimodal = batch[0][2]
            whisper_shape = first_multimodal["whisper"].shape
            whisper_mask_dim = first_multimodal["whisper_mask"].numel()
            full_clews_shape = first_multimodal["full_clews"].shape
            avg_clews_dim = first_multimodal["avg_clews"].shape[-1]
            clews_mask_dim = first_multimodal["clews_mask"].numel()

            # Handle mask application if enabled
            if apply_masks_with_padding:
                # Find max valid lengths
                max_whisper_len = 0
                max_clews_len = 0
                all_multimodal_data = []

                for item in batch:
                    for j in range(n_per_class):
                        multimodal_dict = item[2 + j * 2]
                        whisper_mask = multimodal_dict["whisper_mask"]
                        clews_mask = multimodal_dict["clews_mask"]

                        valid_whisper = (~whisper_mask).sum().item()
                        valid_clews = (~clews_mask).sum().item()

                        max_whisper_len = max(max_whisper_len, valid_whisper)
                        max_clews_len = max(max_clews_len, valid_clews)

                        all_multimodal_data.append(
                            (multimodal_dict, valid_whisper, valid_clews)
                        )

                actual_whisper_len = max_whisper_len
                actual_clews_len = max_clews_len
            else:
                actual_whisper_len = whisper_mask_dim
                actual_clews_len = clews_mask_dim

            # Pre-allocate for each version (6 items per version)
            for j in range(n_per_class):
                version_ids = torch.empty(batch_size, dtype=torch.long)
                whisper_embeddings = torch.zeros(
                    batch_size, actual_whisper_len, whisper_shape[1]
                )
                whisper_masks = torch.zeros(
                    batch_size, actual_whisper_len, dtype=torch.bool
                )
                full_clews_embeddings = torch.zeros(
                    batch_size, actual_clews_len, full_clews_shape[1]
                )
                avg_clews_embeddings = torch.zeros(batch_size, avg_clews_dim)
                clews_masks = torch.zeros(
                    batch_size, actual_clews_len, dtype=torch.bool
                )
                output.extend(
                    [
                        version_ids,
                        whisper_embeddings,
                        whisper_masks,
                        full_clews_embeddings,
                        avg_clews_embeddings,
                        clews_masks,
                    ]
                )

            # Fill tensors
            data_idx = 0
            for i, item in enumerate(batch):
                clique_ids[i] = item[0]

                for j in range(n_per_class):
                    version_idx = 1 + j * 6
                    whisper_idx = 2 + j * 6
                    whisper_mask_idx = 3 + j * 6
                    full_clews_idx = 4 + j * 6
                    avg_clews_idx = 5 + j * 6
                    clews_mask_idx = 6 + j * 6

                    output[version_idx][i] = item[1 + j * 2]

                    if apply_masks_with_padding:
                        multimodal_dict, valid_whisper, valid_clews = (
                            all_multimodal_data[data_idx]
                        )
                        data_idx += 1

                        # Whisper: no masking
                        output[whisper_idx][i] = multimodal_dict["whisper"]
                        output[whisper_mask_idx][i] = multimodal_dict["whisper_mask"]

                        # CLEWS: apply masking
                        full_clews = multimodal_dict["full_clews"]
                        clews_mask = multimodal_dict["clews_mask"]
                        valid_positions = ~clews_mask

                        if valid_clews > 0:
                            masked_clews = full_clews[valid_positions]
                            padded_clews = torch.zeros(
                                actual_clews_len, full_clews.shape[1]
                            )
                            padded_clews[:valid_clews] = masked_clews
                            new_mask = torch.zeros(actual_clews_len, dtype=torch.bool)
                            new_mask[valid_clews:] = True
                        else:
                            padded_clews = torch.zeros(
                                actual_clews_len, full_clews.shape[1]
                            )
                            new_mask = torch.ones(actual_clews_len, dtype=torch.bool)

                        output[full_clews_idx][i] = padded_clews
                        output[clews_mask_idx][i] = new_mask
                        output[avg_clews_idx][i] = multimodal_dict["avg_clews"]
                    else:
                        # No mask application
                        multimodal_dict = item[2 + j * 2]
                        output[whisper_idx][i] = multimodal_dict["whisper"]
                        output[whisper_mask_idx][i] = multimodal_dict["whisper_mask"]
                        output[full_clews_idx][i] = multimodal_dict["full_clews"]
                        output[avg_clews_idx][i] = multimodal_dict["avg_clews"]
                        output[clews_mask_idx][i] = multimodal_dict["clews_mask"]

            return output

        return whisper_clews_collate_fn

    else:
        # Single-modal collate function
        embedding_type = conf.data.get("embedding_type", "whisper")
        if embedding_type == "clews":
            embedding_type = "clews"
        else:
            embedding_type = "whisper"

        if use_avg_pooling:
            return lambda batch: collate_embeddings_fixed_length(
                batch, use_avg_pooling=True, embedding_type=embedding_type
            )
        elif use_overlapping_chunks:
            return lambda batch: collate_embeddings_fixed_length(
                batch,
                use_random_chunks=False,
                chunk_size=conf.data.get("chunk_size", 1000),
                use_overlapping_chunks=True,
                overlap_percentage=overlap_percentage,
                embedding_type=embedding_type,
            )
        else:
            if deterministic:
                return lambda batch: collate_embeddings_fixed_length(
                    batch,
                    use_random_chunks=False,
                    chunk_size=conf.data.get("chunk_size", 1000),
                    embedding_type=embedding_type,
                )
            else:
                return lambda batch: collate_embeddings_fixed_length(
                    batch,
                    use_random_chunks=conf.data.get("use_random_chunks", False),
                    chunk_size=conf.data.get("chunk_size", 1000),
                    embedding_type=embedding_type,
                )

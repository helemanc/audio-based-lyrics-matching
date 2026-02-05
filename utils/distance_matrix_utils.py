#!/usr/bin/env python3
"""
Utilities for computing and saving distance matrices.

This module provides functions for:
- Extracting embeddings from pre-computed features
- Computing pairwise distances between songs
- Gathering results across multiple GPUs
- Saving distance matrices with metadata
"""

import gc
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from lib.evaluation import eval


@torch.inference_mode()
def extract_embeddings_from_precomputed(
    model, dloader, conf, args, myprogbar, desc="Extract embeddings"
):
    """
    Extract embeddings using pre-computed features (same approach as training).

    Args:
        model: Trained model
        dloader: DataLoader
        conf: Configuration
        args: Command-line arguments
        myprogbar: Progress bar function
        desc: Description for progress bar

    Returns:
        Depending on mode:
        - Standard: (cliques, versions, embeddings, masks, None)
        - Overlapping: (None, None, None, None, list of chunks)
    """
    if not args.use_overlapping_chunks:
        # Standard mode: single embedding per song
        all_c, all_i, all_z, all_m = [], [], [], []
        model.eval()

        for n, batch in enumerate(myprogbar(dloader, desc=desc, leave=True)):
            n_per_class = conf.data.n_per_class

            # Extract batch data
            cc = torch.cat([batch[0]] * n_per_class, dim=0)
            ii = torch.cat(batch[1::3], dim=0)
            xx = torch.cat(batch[2::3], dim=0)
            masks = torch.cat(batch[3::3], dim=0)

            # Process through model
            xx = model.prepare(xx)
            cc, ii, xx = cc.clone(), ii.clone(), xx.clone()
            zz, _ = model.embed(xx, masks)

            # Split back and take first sample (anchor)
            clist = torch.chunk(cc, n_per_class, dim=0)
            ilist = torch.chunk(ii, n_per_class, dim=0)
            zlist = torch.chunk(zz, n_per_class, dim=0)
            mlist = torch.chunk(masks, n_per_class, dim=0)

            all_c.append(clist[0])
            all_i.append(ilist[0])
            all_z.append(zlist[0])
            all_m.append(mlist[0])

            # Limit for debugging
            if args.limit_num is not None and len(all_z) >= args.limit_num / args.ngpus:
                break

        # Concatenate
        all_c = torch.cat(all_c, dim=0)
        all_i = torch.cat(all_i, dim=0)
        all_z = torch.cat(all_z, dim=0)
        all_m = torch.cat(all_m, dim=0)

        return all_c, all_i, all_z, all_m, None

    else:
        # Overlapping chunks mode
        all_chunks = []
        processed_songs = set()
        model.eval()

        for batch_idx, batch in enumerate(myprogbar(dloader, desc=desc, leave=True)):
            clique_ids = batch[0]
            version_ids = batch[1]
            embeddings = batch[2]
            masks = batch[3]
            chunk_info = batch[4]

            embeddings = model.prepare(embeddings)
            chunk_embeddings, _ = model.embed(embeddings, masks)

            # Process chunks
            batch_chunks = []
            batch_songs = set()

            for chunk_idx in range(len(clique_ids)):
                original_batch_idx, original_version_idx, local_chunk_idx = chunk_info[
                    chunk_idx
                ]

                clique_id = clique_ids[chunk_idx].item()
                version_id = version_ids[chunk_idx].item()
                song_id = int(f"{clique_id:06d}{version_id:06d}")

                batch_songs.add((clique_id, version_id))

                chunk_data = {
                    "clique_id": clique_id,
                    "version_id": version_id,
                    "embedding": chunk_embeddings[chunk_idx],
                    "mask": masks[chunk_idx],
                    "song_id": song_id,
                    "chunk_idx": local_chunk_idx,
                }
                batch_chunks.append(chunk_data)

            # Check song limit
            if args.limit_num is not None:
                songs_per_gpu = max(1, args.limit_num // args.ngpus)
                new_songs = batch_songs - processed_songs
                if len(processed_songs) + len(new_songs) > songs_per_gpu:
                    break

            all_chunks.extend(batch_chunks)
            processed_songs.update(batch_songs)

            # Memory cleanup
            if batch_idx % 5 == 0 and batch_idx > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return None, None, None, None, all_chunks


def gather_chunks(fabric, query_chunks):
    """
    Gather chunks from all GPUs in distributed setting.

    Args:
        fabric: Lightning Fabric instance
        query_chunks: List of chunk dictionaries from this GPU

    Returns:
        List of all chunks from all GPUs
    """
    if len(query_chunks) == 0:
        local_embeddings = torch.empty(0, 0, device=fabric.device)
        local_masks = torch.empty(0, 0, device=fabric.device)
        local_clique_ids = torch.empty(0, dtype=torch.long, device=fabric.device)
        local_version_ids = torch.empty(0, dtype=torch.long, device=fabric.device)
        local_song_ids = torch.empty(0, dtype=torch.long, device=fabric.device)
        local_chunk_indices = torch.empty(0, dtype=torch.long, device=fabric.device)
    else:
        local_embeddings = torch.stack([c["embedding"] for c in query_chunks])
        local_masks = torch.stack([c["mask"] for c in query_chunks])
        local_clique_ids = torch.tensor(
            [c["clique_id"] for c in query_chunks],
            dtype=torch.long,
            device=fabric.device,
        )
        local_version_ids = torch.tensor(
            [c["version_id"] for c in query_chunks],
            dtype=torch.long,
            device=fabric.device,
        )
        local_song_ids = torch.tensor(
            [c["song_id"] for c in query_chunks], dtype=torch.long, device=fabric.device
        )
        local_chunk_indices = torch.tensor(
            [c["chunk_idx"] for c in query_chunks],
            dtype=torch.long,
            device=fabric.device,
        )

    del query_chunks
    gc.collect()

    fabric.barrier()

    # Get sizes and max size
    local_size = torch.tensor(
        [len(local_embeddings)], dtype=torch.long, device=fabric.device
    )
    all_sizes = fabric.all_gather(local_size)
    max_size = max(size.item() for size in all_sizes) if len(all_sizes) > 0 else 0

    if max_size == 0:
        return []

    # Pad to max size
    if len(local_embeddings) == 0:
        embed_dim = 512  # Default, adjust based on model
        chunk_size = 1500
        local_embeddings = torch.zeros(
            max_size, embed_dim, device=fabric.device, dtype=torch.half
        )
        local_masks = torch.zeros(
            max_size, chunk_size, device=fabric.device, dtype=torch.bool
        )
        local_clique_ids = torch.zeros(max_size, device=fabric.device, dtype=torch.long)
        local_version_ids = torch.zeros(
            max_size, device=fabric.device, dtype=torch.long
        )
        local_song_ids = torch.zeros(max_size, device=fabric.device, dtype=torch.long)
        local_chunk_indices = torch.zeros(
            max_size, device=fabric.device, dtype=torch.long
        )
    elif len(local_embeddings) < max_size:
        pad_size = max_size - len(local_embeddings)
        embed_dim = local_embeddings.shape[1]
        chunk_size = local_masks.shape[1]

        local_embeddings = torch.cat(
            [
                local_embeddings,
                torch.zeros(
                    pad_size,
                    embed_dim,
                    device=local_embeddings.device,
                    dtype=local_embeddings.dtype,
                ),
            ],
            dim=0,
        )
        local_masks = torch.cat(
            [
                local_masks,
                torch.zeros(
                    pad_size,
                    chunk_size,
                    device=local_masks.device,
                    dtype=local_masks.dtype,
                ),
            ],
            dim=0,
        )
        local_clique_ids = torch.cat(
            [
                local_clique_ids,
                torch.zeros(
                    pad_size,
                    device=local_clique_ids.device,
                    dtype=local_clique_ids.dtype,
                ),
            ],
            dim=0,
        )
        local_version_ids = torch.cat(
            [
                local_version_ids,
                torch.zeros(
                    pad_size,
                    device=local_version_ids.device,
                    dtype=local_version_ids.dtype,
                ),
            ],
            dim=0,
        )
        local_song_ids = torch.cat(
            [
                local_song_ids,
                torch.zeros(
                    pad_size, device=local_song_ids.device, dtype=local_song_ids.dtype
                ),
            ],
            dim=0,
        )
        local_chunk_indices = torch.cat(
            [
                local_chunk_indices,
                torch.zeros(
                    pad_size,
                    device=local_chunk_indices.device,
                    dtype=local_chunk_indices.dtype,
                ),
            ],
            dim=0,
        )

    # Gather from all GPUs
    all_embeddings = fabric.all_gather(local_embeddings)
    all_masks = fabric.all_gather(local_masks)
    all_clique_ids = fabric.all_gather(local_clique_ids)
    all_version_ids = fabric.all_gather(local_version_ids)
    all_song_ids = fabric.all_gather(local_song_ids)
    all_chunk_indices = fabric.all_gather(local_chunk_indices)

    # Clean up
    del (
        local_embeddings,
        local_masks,
        local_clique_ids,
        local_version_ids,
        local_song_ids,
        local_chunk_indices,
    )

    # Reconstruct chunks
    all_chunks = []
    for gpu_idx in range(fabric.world_size):
        gpu_size = all_sizes[gpu_idx].item()
        if gpu_size > 0:
            for chunk_idx in range(gpu_size):
                chunk_data = {
                    "clique_id": all_clique_ids[gpu_idx, chunk_idx].item(),
                    "version_id": all_version_ids[gpu_idx, chunk_idx].item(),
                    "embedding": all_embeddings[gpu_idx, chunk_idx].clone().detach(),
                    "mask": all_masks[gpu_idx, chunk_idx].clone().detach(),
                    "song_id": all_song_ids[gpu_idx, chunk_idx].item(),
                    "chunk_idx": all_chunk_indices[gpu_idx, chunk_idx].item(),
                }
                all_chunks.append(chunk_data)

    del (
        all_embeddings,
        all_masks,
        all_clique_ids,
        all_version_ids,
        all_song_ids,
        all_chunk_indices,
        all_sizes,
    )
    gc.collect()

    return all_chunks


def compute_query_distances(query_song_id, query_embeddings, song_list, topk=1):
    """
    Compute distances from one query to all candidates using vectorized operations.

    Args:
        query_song_id: ID of query song
        query_embeddings: Query embeddings tensor (n_chunks, embed_dim)
        song_list: List of (song_id, song_data) tuples
        topk: Number of top distances to average

    Returns:
        Distance vector (n_candidates,)
    """
    distances = torch.full(
        (len(song_list),), float("inf"), device=query_embeddings.device
    )
    query_norm = F.normalize(query_embeddings, p=2, dim=-1)

    for j, (cand_song_id, cand_song_data) in enumerate(song_list):
        if query_song_id == cand_song_id:
            distances[j] = float("-inf")  # Will be set to 0 later
            continue

        cand_embeddings = torch.stack(cand_song_data["chunks"])
        cand_norm = F.normalize(cand_embeddings, p=2, dim=-1)

        # Cosine similarity → distance
        sim_matrix = torch.mm(query_norm, cand_norm.T)
        distance_matrix = 1 - sim_matrix

        # Apply topk averaging
        if topk == 1:
            distances[j] = distance_matrix.min()
        else:
            flat_distances = distance_matrix.flatten()
            if len(flat_distances) >= topk:
                topk_distances = torch.topk(flat_distances, k=topk, largest=False)[0]
                distances[j] = topk_distances.mean()
            else:
                distances[j] = flat_distances.mean()

    return distances


def evaluate_with_distance_saving(fabric, query_chunks, args, myprint):
    """
    Evaluate overlapping chunks and optionally save distance matrix.

    Args:
        fabric: Lightning Fabric instance
        query_chunks: List of query chunks
        args: Command-line arguments
        myprint: Print function

    Returns:
        Tuple of (aps, r1s, rpcs) for evaluation metrics
    """
    topk = getattr(args, "topk_distance", 1)

    # Gather all chunks from all GPUs
    all_chunks = gather_chunks(fabric, query_chunks)

    if len(all_chunks) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Group chunks by song
    songs = {}
    for chunk in all_chunks:
        song_id = chunk["song_id"]
        if song_id not in songs:
            songs[song_id] = {
                "clique_id": chunk["clique_id"],
                "version_id": chunk["version_id"],
                "chunks": [],
            }
        songs[song_id]["chunks"].append(chunk["embedding"])

    song_list = list(songs.items())
    total_songs = len(song_list)

    # Distribute work across GPUs
    songs_per_gpu = total_songs // fabric.world_size
    start_idx = fabric.global_rank * songs_per_gpu
    end_idx = (
        total_songs
        if fabric.global_rank == fabric.world_size - 1
        else start_idx + songs_per_gpu
    )

    local_song_list = song_list[start_idx:end_idx]
    local_queries = len(local_song_list)

    if local_queries == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Compute distance matrix
    distance_matrix = torch.full((local_queries, total_songs), float("inf"))
    raw_distances = [] if args.save_distance_matrix is not None else None

    for i, (query_song_id, query_song_data) in enumerate(local_song_list):
        query_embeddings = torch.stack(query_song_data["chunks"])

        distance_row = compute_query_distances(
            query_song_id, query_embeddings, song_list, topk
        )
        distance_matrix[i, :] = distance_row

        if raw_distances is not None:
            raw_distances.append(distance_row.cpu())

        del query_embeddings, distance_row

        if (i + 1) % 50 == 0:
            myprint(f"  Processed {i + 1}/{local_queries} queries")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Build metadata
    local_query_cliques = torch.tensor(
        [s["clique_id"] for _, s in local_song_list], dtype=torch.long
    )
    local_query_versions = torch.tensor(
        [s["version_id"] for _, s in local_song_list], dtype=torch.long
    )
    all_cand_cliques = torch.tensor(
        [s["clique_id"] for _, s in song_list], dtype=torch.long
    )
    all_cand_versions = torch.tensor(
        [s["version_id"] for _, s in song_list], dtype=torch.long
    )

    # Evaluate
    aps, r1s, rpcs = eval.compute_baseline(
        distance_matrix,
        local_query_cliques,
        local_query_versions,
        all_cand_cliques,
        all_cand_versions,
    )

    # Save distance matrix if requested
    if args.save_distance_matrix is not None and raw_distances is not None:
        save_distance_matrix(
            fabric,
            raw_distances,
            song_list,
            local_song_list,
            all_cand_cliques,
            all_cand_versions,
            args,
            myprint,
        )

    return aps, r1s, rpcs


def save_distance_matrix(
    fabric,
    raw_distances,
    song_list,
    local_song_list,
    all_cand_cliques,
    all_cand_versions,
    args,
    myprint,
):
    """
    Save complete distance matrix to disk (only on rank 0).

    Args:
        fabric: Lightning Fabric instance
        raw_distances: List of distance tensors from this GPU
        song_list: Complete list of songs
        local_song_list: Songs processed by this GPU
        all_cand_cliques: All candidate clique IDs
        all_cand_versions: All candidate version IDs
        args: Command-line arguments
        myprint: Print function
    """
    # Gather distances from all GPUs
    if len(raw_distances) > 0:
        local_distances = torch.stack(raw_distances)
        local_query_count = local_distances.shape[0]
    else:
        local_distances = torch.empty(0, len(song_list))
        local_query_count = 0

    fabric.barrier()
    local_count_tensor = torch.tensor(
        [local_query_count], dtype=torch.long, device=fabric.device
    )
    all_counts = fabric.all_gather(local_count_tensor)
    max_local_queries = max(c.item() for c in all_counts) if len(all_counts) > 0 else 0

    if max_local_queries > 0:
        if local_query_count < max_local_queries:
            pad_size = max_local_queries - local_query_count
            padding = torch.full((pad_size, len(song_list)), float("inf"))
            local_distances_padded = torch.cat([local_distances, padding], dim=0)
        else:
            local_distances_padded = local_distances

        all_gpu_distances = fabric.all_gather(local_distances_padded)

        # Reconstruct full matrix
        distance_parts = []
        for gpu_idx in range(fabric.world_size):
            gpu_count = all_counts[gpu_idx].item()
            if gpu_count > 0:
                distance_parts.append(all_gpu_distances[gpu_idx][:gpu_count])

        if distance_parts:
            full_distance_matrix = torch.cat(distance_parts, dim=0)
        else:
            full_distance_matrix = torch.empty(0, len(song_list))
    else:
        full_distance_matrix = torch.empty(0, len(song_list))

    # Only rank 0 saves
    if fabric.is_global_zero:
        myprint("Preparing distance matrix for saving...")

        # Create references
        query_references = []
        query_idx = 0

        for gpu_rank in range(fabric.world_size):
            gpu_songs_per_gpu = len(song_list) // fabric.world_size
            gpu_start = gpu_rank * gpu_songs_per_gpu
            gpu_end = (
                len(song_list)
                if gpu_rank == fabric.world_size - 1
                else gpu_start + gpu_songs_per_gpu
            )
            gpu_query_count = all_counts[gpu_rank].item()

            for local_idx in range(gpu_query_count):
                global_song_idx = gpu_start + local_idx
                if global_song_idx < len(song_list):
                    song_id, song_data = song_list[global_song_idx]
                    query_references.append(
                        {
                            "clique": song_data["clique_id"],
                            "version": song_data["version_id"],
                            "matrix_row": query_idx,
                            "original_index": query_idx,
                        }
                    )
                    query_idx += 1

        candidate_references = []
        for matrix_col_idx in range(len(all_cand_cliques)):
            candidate_references.append(
                {
                    "clique": all_cand_cliques[matrix_col_idx].item(),
                    "version": all_cand_versions[matrix_col_idx].item(),
                    "matrix_col": matrix_col_idx,
                    "original_index": matrix_col_idx,
                }
            )

        # Prepare data
        distance_data = {
            "distance_matrix": full_distance_matrix.cpu().numpy(),
            "query_references": query_references,
            "candidate_references": candidate_references,
            "metadata": {
                "n_queries": len(query_references),
                "n_candidates": len(candidate_references),
                "checkpoint": args.checkpoint,
                "partition": args.partition,
                "use_overlapping_chunks": args.use_overlapping_chunks,
                "overlap_percentage": args.overlap_percentage
                if args.use_overlapping_chunks
                else None,
                "chunk_size": args.chunk_size if args.use_overlapping_chunks else None,
                "topk_distance": args.topk_distance,
            },
        }

        myprint(f"Saving distance matrix to: {args.save_distance_matrix}")
        os.makedirs(
            os.path.dirname(os.path.abspath(args.save_distance_matrix)), exist_ok=True
        )
        with open(args.save_distance_matrix, "wb") as f:
            pickle.dump(distance_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        myprint(f"  Distance matrix shape: {full_distance_matrix.shape}")
        myprint(
            f"  Queries: {len(query_references)}, Candidates: {len(candidate_references)}"
        )


def gather_evaluation_results(fabric, local_aps, local_r1s, local_rpcs):
    """
    Gather evaluation results from all GPUs.

    Args:
        fabric: Lightning Fabric instance
        local_aps: Local average precision scores
        local_r1s: Local rank-1 scores
        local_rpcs: Local rank percentile scores

    Returns:
        Tuple of (aps, r1s, rpcs) gathered from all GPUs
    """
    fabric.barrier()

    if len(local_aps) == 0:
        local_aps = torch.tensor([], dtype=torch.float32, device=fabric.device)
        local_r1s = torch.tensor([], dtype=torch.float32, device=fabric.device)
        local_rpcs = torch.tensor([], dtype=torch.float32, device=fabric.device)
    else:
        local_aps = local_aps.to(fabric.device)
        local_r1s = local_r1s.to(fabric.device)
        local_rpcs = local_rpcs.to(fabric.device)

    local_size = torch.tensor([len(local_aps)], dtype=torch.long, device=fabric.device)
    all_sizes = fabric.all_gather(local_size)
    max_size = max(size.item() for size in all_sizes)

    if max_size == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

    # Pad
    if len(local_aps) < max_size:
        pad_size = max_size - len(local_aps)
        local_aps = torch.cat(
            [local_aps, torch.zeros(pad_size, device=local_aps.device)]
        )
        local_r1s = torch.cat(
            [local_r1s, torch.zeros(pad_size, device=local_r1s.device)]
        )
        local_rpcs = torch.cat(
            [local_rpcs, torch.zeros(pad_size, device=local_rpcs.device)]
        )

    # Gather
    all_aps = fabric.all_gather(local_aps)
    all_r1s = fabric.all_gather(local_r1s)
    all_rpcs = fabric.all_gather(local_rpcs)

    # Reconstruct
    final_aps, final_r1s, final_rpcs = [], [], []
    for gpu_idx in range(fabric.world_size):
        gpu_size = all_sizes[gpu_idx].item()
        if gpu_size > 0:
            final_aps.append(all_aps[gpu_idx][:gpu_size])
            final_r1s.append(all_r1s[gpu_idx][:gpu_size])
            final_rpcs.append(all_rpcs[gpu_idx][:gpu_size])

    if final_aps:
        return torch.cat(final_aps), torch.cat(final_r1s), torch.cat(final_rpcs)
    else:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])

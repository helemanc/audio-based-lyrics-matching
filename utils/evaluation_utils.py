"""
Evaluation utilities for retrieval metrics computation.

Provides functions for evaluating encoder/decoder latent representations using
retrieval metrics (MAP, MR1, ARP) with distributed computing support via Lightning Fabric.

Typical usage:
    >>> run_evaluation(
    ...     query_encoder, query_decoder, cand_encoder, cand_decoder,
    ...     distances=['cosine'], query_cliques, query_ids,
    ...     cand_cliques, cand_ids, fabric, 'shs', 'whisper_base'
    ... )
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from lib.evaluation.eval import compute


def evaluate_latents(
    query_z: torch.Tensor,
    cand_z: torch.Tensor,
    latent_type: str,
    distances: List[str],
    query_c: torch.Tensor,
    query_i: torch.Tensor,
    cand_c: torch.Tensor,
    cand_i: torch.Tensor,
    fabric: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate latent representations using retrieval metrics.

    Computes MAP (Mean Average Precision), MR1 (Mean Rank@1), and ARP
    (Average Rank Percentile) for each query-candidate pair using specified
    distance functions. Results are gathered across GPUs and confidence intervals
    are computed.

    Args:
        query_z: Query latent embeddings, shape (B, S, C) where B is batch size,
                S is sequence length, C is embedding dimension
        cand_z: Candidate latent embeddings, shape (B', S, C)
        latent_type: Latent identifier ('encoder' or 'decoder') for result naming
        distances: List of distance function names to evaluate (e.g., ['cosine'])
        query_c: Query clique IDs, shape (B,)
        query_i: Query version IDs, shape (B,)
        cand_c: Candidate clique IDs, shape (B',)
        cand_i: Candidate version IDs, shape (B',)
        fabric: Lightning Fabric instance for distributed computing

    Returns:
        Dictionary mapping "{latent_type}_latents_{distance}" to results containing:
            - mean: Dict with MAP, MR1, ARP mean values
            - confidence_interval_{distance}: Dict with 95% CI for each metric

    Example:
        >>> results = evaluate_latents(
        ...     query_encoder, cand_encoder, 'encoder', ['cosine'],
        ...     query_cliques, query_ids, cand_cliques, cand_ids, fabric
        ... )
        >>> print(f"Encoder MAP: {results['encoder_latents_cosine']['mean']['MAP']:.4f}")
    """
    results = {}

    for dist in distances:
        # Evaluate each query
        aps = []
        r1s = []
        rpcs = []

        for n in tqdm(
            range(len(query_z)), desc="Retrieve", disable=not fabric.is_global_zero
        ):
            ap, r1, rpc = compute(
                model=None,
                queries_c=query_c[n : n + 1],
                queries_i=query_i[n : n + 1],
                queries_z=query_z[n : n + 1],
                candidates_c=cand_c,
                candidates_i=cand_i,
                candidates_z=cand_z,
                redux_strategy=None,
                batch_size_candidates=2**15,
                distance_fn=dist,
            )
            aps.append(ap)
            r1s.append(r1)
            rpcs.append(rpc)

        # Stack metrics
        aps = torch.stack(aps)
        r1s = torch.stack(r1s)
        rpcs = torch.stack(rpcs)

        # Collect metrics from all GPUs
        fabric.barrier()
        aps = fabric.all_gather(aps)
        r1s = fabric.all_gather(r1s)
        rpcs = fabric.all_gather(rpcs)

        # Flatten the gathered metrics
        aps = torch.cat(torch.unbind(aps, dim=0), dim=0)
        r1s = torch.cat(torch.unbind(r1s, dim=0), dim=0)
        rpcs = torch.cat(torch.unbind(rpcs, dim=0), dim=0)

        if fabric.is_global_zero:
            logdict_mean = {
                "MAP": aps.mean().item(),
                "MR1": r1s.mean().item(),
                "ARP": rpcs.mean().item(),
            }

            logdict_ci = {
                "MAP": 1.96 * aps.std().item() / math.sqrt(len(aps)),
                "MR1": 1.96 * r1s.std().item() / math.sqrt(len(r1s)),
                "ARP": 1.96 * rpcs.std().item() / math.sqrt(len(rpcs)),
            }

            print("=" * 100)
            print(f"Result {latent_type.capitalize()} Latents - Distance {dist}:")
            print(
                f"  Avg --> MAP: {logdict_mean['MAP']:.4f}, MR1: {logdict_mean['MR1']:.4f}, ARP: {logdict_mean['ARP']:.4f}"
            )
            print(
                f"  c.i. -> MAP: {logdict_ci['MAP']:.4f}, MR1: {logdict_ci['MR1']:.4f}, ARP: {logdict_ci['ARP']:.4f}"
            )
            print("=" * 100)

            results[f"{latent_type}_latents_{dist}"] = {
                "mean": logdict_mean,
                f"confidence_interval_{dist}": logdict_ci,
            }

    return results


def run_evaluation(
    query_z_e: torch.Tensor,
    query_z_d: torch.Tensor,
    cand_z_e: torch.Tensor,
    cand_z_d: torch.Tensor,
    distances: List[str],
    query_c: torch.Tensor,
    query_i: torch.Tensor,
    cand_c: torch.Tensor,
    cand_i: torch.Tensor,
    fabric: Any,
    dataset_name: str,
    decoding_config_name: str,
) -> None:
    """
    Run comprehensive evaluation for both encoder and decoder latents.

    Evaluates both encoder and decoder representations, computes retrieval metrics
    for each, and saves combined results to JSON file. Only rank 0 performs file I/O.

    Args:
        query_z_e: Query encoder embeddings, shape (B, S, C)
        query_z_d: Query decoder embeddings, shape (B, S, C)
        cand_z_e: Candidate encoder embeddings, shape (B', S, C)
        cand_z_d: Candidate decoder embeddings, shape (B', S, C)
        distances: List of distance functions to evaluate (e.g., ['cosine', 'euclidean'])
        query_c: Query clique IDs, shape (B,)
        query_i: Query version IDs, shape (B,)
        cand_c: Candidate clique IDs, shape (B',)
        cand_i: Candidate version IDs, shape (B',)
        fabric: Lightning Fabric instance for distributed computing
        dataset_name: Dataset identifier (e.g., 'shs', 'lyric-covers')
        decoding_config_name: Configuration identifier for naming output file

    Side Effects:
        Creates evaluation_results_{decoding_config_name}.json with all metrics

    Example:
        >>> run_evaluation(
        ...     query_enc, query_dec, cand_enc, cand_dec,
        ...     ['cosine'], query_c, query_i, cand_c, cand_i,
        ...     fabric, 'shs', 'whisper_base_42'
        ... )
        Results saved to evaluation_results_whisper_base_42.json
    """

    # Evaluate encoder latents
    encoder_results = evaluate_latents(
        query_z_e,
        cand_z_e,
        "encoder",
        distances,
        query_c,
        query_i,
        cand_c,
        cand_i,
        fabric,
    )

    # Evaluate decoder latents
    decoder_results = evaluate_latents(
        query_z_d,
        cand_z_d,
        "decoder",
        distances,
        query_c,
        query_i,
        cand_c,
        cand_i,
        fabric,
    )

    # Combine results and save
    if fabric.is_global_zero:
        all_results = {**encoder_results, **decoder_results}

        results_file = f"evaluation_results_{decoding_config_name}.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=4)

        print(f"Results saved to {results_file}")

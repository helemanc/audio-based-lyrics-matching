#!/usr/bin/env python3
"""
Multimodal fusion utilities for combining distance matrices from different modalities.

This module provides functionality to:
- Load and align distance matrices from different sources (e.g., CLEWS + WEALY)
- Combine distance matrices with weighted fusion
- Evaluate fusion combinations using standard metrics
"""

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from . import eval


def load_distance_matrix(path: str) -> Dict[str, Any]:
    """
    Load a distance matrix pickle file.

    Args:
        path: Path to the pickle file

    Returns:
        Dictionary containing:
            - 'distance_matrix': numpy array of shape (n_queries, n_candidates)
            - 'query_references': list of dicts with 'clique', 'version', 'matrix_row'
            - 'candidate_references': list of dicts with 'clique', 'version', 'matrix_col'
            - 'metadata': dict with additional information
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def align_distance_matrices(
    matrix1_data: Dict[str, Any],
    matrix2_data: Dict[str, Any],
    match_on: str = "version",
) -> Tuple[
    np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """
    Align two distance matrices based on common queries and candidates.

    Args:
        matrix1_data: First distance matrix data (from load_distance_matrix)
        matrix2_data: Second distance matrix data (from load_distance_matrix)
        match_on: What to match on - "version" (default) or "clique+version"

    Returns:
        Tuple of:
            - matrix1_aligned: Aligned matrix 1 (n_common_queries, n_common_candidates)
            - matrix2_aligned: Aligned matrix 2 (n_common_queries, n_common_candidates)
            - query_cliques: Clique IDs for queries (n_common_queries,)
            - query_versions: Version IDs for queries (n_common_queries,)
            - cand_cliques: Clique IDs for candidates (n_common_candidates,)
            - cand_versions: Version IDs for candidates (n_common_candidates,)
    """
    # Create mappings from version/clique to matrix indices
    if match_on == "version":
        # Match only on version ID (assumes versions are unique)
        m1_q_map = {
            r["version"]: (r["matrix_row"], r["clique"])
            for r in matrix1_data["query_references"]
        }
        m1_c_map = {
            r["version"]: (r["matrix_col"], r["clique"])
            for r in matrix1_data["candidate_references"]
        }
        m2_q_map = {
            r["version"]: r["matrix_row"] for r in matrix2_data["query_references"]
        }
        m2_c_map = {
            r["version"]: r["matrix_col"] for r in matrix2_data["candidate_references"]
        }

        # Find common versions
        common_q = sorted(set(m1_q_map.keys()) & set(m2_q_map.keys()))
        common_c = sorted(set(m1_c_map.keys()) & set(m2_c_map.keys()))

        # Get alignment indices
        m1_q_idx = np.array([m1_q_map[v][0] for v in common_q], dtype=int)
        m1_c_idx = np.array([m1_c_map[v][0] for v in common_c], dtype=int)
        m2_q_idx = np.array([m2_q_map[v] for v in common_q], dtype=int)
        m2_c_idx = np.array([m2_c_map[v] for v in common_c], dtype=int)

        # Create metadata tensors (use clique from matrix1)
        query_cliques = torch.tensor(
            [m1_q_map[v][1] for v in common_q], dtype=torch.long
        )
        query_versions = torch.tensor(common_q, dtype=torch.long)
        cand_cliques = torch.tensor(
            [m1_c_map[v][1] for v in common_c], dtype=torch.long
        )
        cand_versions = torch.tensor(common_c, dtype=torch.long)

    else:  # match on clique+version
        # Match on both clique and version
        m1_q_map = {
            (r["clique"], r["version"]): r["matrix_row"]
            for r in matrix1_data["query_references"]
        }
        m1_c_map = {
            (r["clique"], r["version"]): r["matrix_col"]
            for r in matrix1_data["candidate_references"]
        }
        m2_q_map = {
            (r["clique"], r["version"]): r["matrix_row"]
            for r in matrix2_data["query_references"]
        }
        m2_c_map = {
            (r["clique"], r["version"]): r["matrix_col"]
            for r in matrix2_data["candidate_references"]
        }

        # Find common (clique, version) pairs
        common_q = sorted(set(m1_q_map.keys()) & set(m2_q_map.keys()))
        common_c = sorted(set(m1_c_map.keys()) & set(m2_c_map.keys()))

        # Get alignment indices
        m1_q_idx = np.array([m1_q_map[k] for k in common_q], dtype=int)
        m1_c_idx = np.array([m1_c_map[k] for k in common_c], dtype=int)
        m2_q_idx = np.array([m2_q_map[k] for k in common_q], dtype=int)
        m2_c_idx = np.array([m2_c_map[k] for k in common_c], dtype=int)

        # Create metadata tensors
        query_cliques = torch.tensor([k[0] for k in common_q], dtype=torch.long)
        query_versions = torch.tensor([k[1] for k in common_q], dtype=torch.long)
        cand_cliques = torch.tensor([k[0] for k in common_c], dtype=torch.long)
        cand_versions = torch.tensor([k[1] for k in common_c], dtype=torch.long)

    # Align matrices using fancy indexing
    m1_aligned = matrix1_data["distance_matrix"][np.ix_(m1_q_idx, m1_c_idx)]
    m2_aligned = matrix2_data["distance_matrix"][np.ix_(m2_q_idx, m2_c_idx)]

    print(f"Alignment complete:")
    print(f"  Matrix 1: {matrix1_data['distance_matrix'].shape} → {m1_aligned.shape}")
    print(f"  Matrix 2: {matrix2_data['distance_matrix'].shape} → {m2_aligned.shape}")
    print(f"  Common queries: {len(common_q)}, Common candidates: {len(common_c)}")

    return (
        m1_aligned,
        m2_aligned,
        query_cliques,
        query_versions,
        cand_cliques,
        cand_versions,
    )


def combine_distance_matrices(
    matrix1: np.ndarray, matrix2: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Combine two distance matrices with weighted fusion.

    Combined distance = matrix1 + alpha * matrix2

    Args:
        matrix1: First distance matrix (e.g., CLEWS)
        matrix2: Second distance matrix (e.g., WEALY)
        alpha: Weight for matrix2

    Returns:
        Combined distance matrix
    """
    return matrix1 + alpha * matrix2


def evaluate_fusion(
    distance_matrix: np.ndarray,
    query_cliques: torch.Tensor,
    query_versions: torch.Tensor,
    cand_cliques: torch.Tensor,
    cand_versions: torch.Tensor,
) -> Tuple[float, float, float, float, float, float]:
    """
    Evaluate a distance matrix using standard metrics.

    Args:
        distance_matrix: Distance matrix (n_queries, n_candidates)
        query_cliques: Clique IDs for queries
        query_versions: Version IDs for queries
        cand_cliques: Clique IDs for candidates
        cand_versions: Version IDs for candidates

    Returns:
        Tuple of (MAP_mean, MAP_ci, MR1_mean, MR1_ci, ARP_mean, ARP_ci)
    """
    distance_tensor = torch.from_numpy(distance_matrix).float()

    # Compute metrics using eval.compute_baseline
    aps, r1s, rpcs = eval.compute_baseline(
        distance_tensor, query_cliques, query_versions, cand_cliques, cand_versions
    )

    def compute_stats(vals: torch.Tensor) -> Tuple[float, float]:
        """Compute mean and 95% confidence interval."""
        if len(vals) == 0:
            return 0.0, 0.0
        mean_val = float(vals.mean())
        ci = float(
            1.96 * vals.std() / torch.sqrt(torch.tensor(len(vals), dtype=torch.float))
        )
        return mean_val, ci

    map_mean, map_ci = compute_stats(aps)
    mr1_mean, mr1_ci = compute_stats(r1s)
    arp_mean, arp_ci = compute_stats(rpcs)

    return map_mean, map_ci, mr1_mean, mr1_ci, arp_mean, arp_ci


def grid_search_fusion(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    query_cliques: torch.Tensor,
    query_versions: torch.Tensor,
    cand_cliques: torch.Tensor,
    cand_versions: torch.Tensor,
    alphas: List[float],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple alpha values for fusion and return results.

    Args:
        matrix1: First distance matrix
        matrix2: Second distance matrix
        query_cliques: Clique IDs for queries
        query_versions: Version IDs for queries
        cand_cliques: Clique IDs for candidates
        cand_versions: Version IDs for candidates
        alphas: List of alpha values to try
        verbose: Print progress

    Returns:
        DataFrame with columns: alpha, MAP, MAP_CI, MR1, MR1_CI, ARP, ARP_CI
    """
    results = []

    for i, alpha in enumerate(alphas):
        if verbose:
            print(f"Evaluating alpha={alpha:.2f} ({i + 1}/{len(alphas)})...", end="\r")

        combined = combine_distance_matrices(matrix1, matrix2, alpha)
        map_mean, map_ci, mr1_mean, mr1_ci, arp_mean, arp_ci = evaluate_fusion(
            combined, query_cliques, query_versions, cand_cliques, cand_versions
        )

        results.append(
            {
                "alpha": alpha,
                "MAP": map_mean,
                "MAP_CI": map_ci,
                "MR1": mr1_mean,
                "MR1_CI": mr1_ci,
                "ARP": arp_mean,
                "ARP_CI": arp_ci,
            }
        )

    if verbose:
        print()  # Clear progress line

    return pd.DataFrame(results)


def print_fusion_results(results_df: pd.DataFrame, top_n: int = 5):
    """
    Pretty-print fusion results.

    Args:
        results_df: Results DataFrame from grid_search_fusion
        top_n: Number of top results to highlight
    """
    print("=" * 70)
    print("FUSION RESULTS")
    print("=" * 70)
    print(f"{'Alpha':<8} {'MAP':<18} {'MR1':<18} {'ARP':<18}")
    print("-" * 70)

    # Sort by MAP descending
    sorted_df = results_df.sort_values("MAP", ascending=False)

    for idx, (_, row) in enumerate(sorted_df.iterrows()):
        marker = "★" if idx < top_n else " "
        print(
            f"{marker} {row['alpha']:<6.2f} "
            f"{row['MAP']:.4f}±{row['MAP_CI']:.4f}   "
            f"{row['MR1']:.4f}±{row['MR1_CI']:.4f}   "
            f"{row['ARP']:.4f}±{row['ARP_CI']:.4f}"
        )

    print("=" * 70)

    # Print best result
    best = sorted_df.iloc[0]
    print(f"\nBest alpha: {best['alpha']:.2f}")
    print(f"  MAP: {best['MAP']:.4f} ± {best['MAP_CI']:.4f}")
    print(f"  MR1: {best['MR1']:.4f} ± {best['MR1_CI']:.4f}")
    print(f"  ARP: {best['ARP']:.4f} ± {best['ARP_CI']:.4f}")

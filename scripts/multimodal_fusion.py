#!/usr/bin/env python3
"""
Multimodal Fusion for Version Identification

This script combines distance matrices from different modalities (e.g., CLEWS + WEALY)
and evaluates different fusion weights to find optimal combinations.

Usage:
    # Basic usage: Combine two distance matrices
    python scripts/multimodal_fusion.py \
        --matrix1 path/to/clews_distances.pkl \
        --matrix2 path/to/wealy_distances.pkl \
        --output results.csv

    # Custom alpha range
    python scripts/multimodal_fusion.py \
        --matrix1 path/to/clews_distances.pkl \
        --matrix2 path/to/wealy_distances.pkl \
        --alphas 0.0 0.5 1.0 1.5 2.0 \
        --output results.csv

    # Fine-grained search around a specific value
    python scripts/multimodal_fusion.py \
        --matrix1 path/to/clews_distances.pkl \
        --matrix2 path/to/wealy_distances.pkl \
        --alpha_min 1.0 \
        --alpha_max 2.0 \
        --alpha_step 0.1 \
        --output results.csv

The fusion formula is: combined_distance = matrix1 + alpha * matrix2

Expected input format (distance matrix pickle files):
    {
        'distance_matrix': np.ndarray,  # (n_queries, n_candidates)
        'query_references': [{'clique': int, 'version': int, 'matrix_row': int}, ...],
        'candidate_references': [{'clique': int, 'version': int, 'matrix_col': int}, ...],
        'metadata': dict  # optional
    }
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.evaluation import fusion


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Combine and evaluate multimodal distance matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input files
    parser.add_argument(
        "--matrix1", required=True, help="Path to first distance matrix (e.g., CLEWS)"
    )
    parser.add_argument(
        "--matrix2", required=True, help="Path to second distance matrix (e.g., WEALY)"
    )

    # Alpha specification (multiple options)
    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        help="List of alpha values to try (e.g., --alphas 0.0 0.5 1.0 2.0)",
    )
    alpha_group.add_argument(
        "--alpha_range",
        type=float,
        nargs=3,
        metavar=("MIN", "MAX", "STEP"),
        help="Alpha range as min max step (e.g., --alpha_range 0.0 2.0 0.1)",
    )

    # Output
    parser.add_argument(
        "--output",
        default="fusion_results.csv",
        help="Output CSV file for results (default: fusion_results.csv)",
    )

    # Options
    parser.add_argument(
        "--match_on",
        choices=["version", "clique+version"],
        default="version",
        help="What to match on when aligning matrices (default: version)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top results to highlight (default: 5)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Set default alphas if none specified
    if args.alphas is None and args.alpha_range is None:
        args.alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0]
    elif args.alpha_range is not None:
        min_val, max_val, step = args.alpha_range
        args.alphas = list(np.arange(min_val, max_val + step / 2, step))

    return args


def main():
    """Main fusion workflow."""
    args = parse_args()

    print("=" * 70)
    print("MULTIMODAL FUSION FOR VERSION IDENTIFICATION")
    print("=" * 70)
    print(f"Matrix 1: {args.matrix1}")
    print(f"Matrix 2: {args.matrix2}")
    print(
        f"Alpha values: {len(args.alphas)} values from {min(args.alphas):.2f} to {max(args.alphas):.2f}"
    )
    print(f"Output: {args.output}")
    print()

    # Load distance matrices
    print("Loading distance matrices...")
    matrix1_data = fusion.load_distance_matrix(args.matrix1)
    matrix2_data = fusion.load_distance_matrix(args.matrix2)

    print(f"  Matrix 1 shape: {matrix1_data['distance_matrix'].shape}")
    print(f"  Matrix 2 shape: {matrix2_data['distance_matrix'].shape}")
    print()

    # Align matrices
    print(f"Aligning matrices (matching on {args.match_on})...")
    m1_aligned, m2_aligned, q_cliques, q_versions, c_cliques, c_versions = (
        fusion.align_distance_matrices(
            matrix1_data, matrix2_data, match_on=args.match_on
        )
    )
    print()

    # Check for data loss during alignment
    m1_orig_size = (
        matrix1_data["distance_matrix"].shape[0]
        * matrix1_data["distance_matrix"].shape[1]
    )
    m1_aligned_size = m1_aligned.shape[0] * m1_aligned.shape[1]
    loss_pct = (1 - m1_aligned_size / m1_orig_size) * 100

    if loss_pct > 5:
        print(f"⚠ WARNING: Alignment dropped {loss_pct:.1f}% of matrix 1 data")
        print(f"  This may affect results. Consider checking version ID consistency.\n")

    # Grid search over alpha values
    print("Evaluating fusion combinations...")
    results = fusion.grid_search_fusion(
        m1_aligned,
        m2_aligned,
        q_cliques,
        q_versions,
        c_cliques,
        c_versions,
        args.alphas,
        verbose=not args.quiet,
    )

    # Save results
    print(f"\nSaving results to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    # Print results
    print()
    fusion.print_fusion_results(results, top_n=args.top_n)

    print("\n" + "=" * 70)
    print("FUSION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

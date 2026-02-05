"""
Utility functions for baseline evaluation and reporting.

Provides statistical analysis, confidence interval computation, and
formatted reporting for baseline evaluation results.
"""

import math
from typing import Dict, Tuple

import torch


def compute_confidence_interval(
    values: torch.Tensor, confidence_level: float = 0.95
) -> Tuple[float, float, float, float, float]:
    """
    Compute confidence interval for the mean.

    Args:
        values: Tensor of values
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, std, ci_margin, ci_lower, ci_upper)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mean_val = float(values.mean().item())
    std_val = float(values.std().item())
    n = len(values)

    # Calculate confidence interval margin using z-score
    if confidence_level == 0.95:
        z_score = 1.96
    elif confidence_level == 0.99:
        z_score = 2.576
    elif confidence_level == 0.90:
        z_score = 1.645
    else:
        from scipy import stats

        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

    ci_margin = float(z_score * std_val / math.sqrt(n))
    ci_lower = mean_val - ci_margin
    ci_upper = mean_val + ci_margin

    return mean_val, std_val, ci_margin, ci_lower, ci_upper


def print_detailed_results(
    baseline_name: str,
    aps: torch.Tensor,
    r1s: torch.Tensor,
    rpcs: torch.Tensor,
    confidence_level: float = 0.95,
) -> Dict:
    """
    Print detailed results with standard deviation and confidence intervals.

    Args:
        baseline_name: Name of the baseline method
        aps: Average Precision scores
        r1s: Recall@1 scores
        rpcs: RPC scores
        confidence_level: Confidence level for CI calculation

    Returns:
        Dictionary containing detailed statistics
    """
    print(f"\n{baseline_name.upper()} Baseline:")
    print(f"  Samples: {len(aps)}")

    # Mean Average Precision
    map_mean, map_std, map_ci, map_lower, map_upper = compute_confidence_interval(
        aps, confidence_level
    )
    print(
        f"  Mean AP:  {map_mean:.4f} ± {map_std:.4f} (std) | {map_mean:.4f} ± {map_ci:.4f} (CI) | [{map_lower:.4f}, {map_upper:.4f}]"
    )

    # Recall@1
    r1s_float = r1s.float()
    r1_mean, r1_std, r1_ci, r1_lower, r1_upper = compute_confidence_interval(
        r1s_float, confidence_level
    )
    print(
        f"  Mean R@1: {r1_mean:.4f} ± {r1_std:.4f} (std) | {r1_mean:.4f} ± {r1_ci:.4f} (CI) | [{r1_lower:.4f}, {r1_upper:.4f}]"
    )

    # RPC
    rpc_mean, rpc_std, rpc_ci, rpc_lower, rpc_upper = compute_confidence_interval(
        rpcs, confidence_level
    )
    print(
        f"  Mean RPC: {rpc_mean:.4f} ± {rpc_std:.4f} (std) | {rpc_mean:.4f} ± {rpc_ci:.4f} (CI) | [{rpc_lower:.4f}, {rpc_upper:.4f}]"
    )

    return {
        "map": {
            "mean": map_mean,
            "std": map_std,
            "ci": map_ci,
            "ci_lower": map_lower,
            "ci_upper": map_upper,
        },
        "r1": {
            "mean": r1_mean,
            "std": r1_std,
            "ci": r1_ci,
            "ci_lower": r1_lower,
            "ci_upper": r1_upper,
        },
        "rpc": {
            "mean": rpc_mean,
            "std": rpc_std,
            "ci": rpc_ci,
            "ci_lower": rpc_lower,
            "ci_upper": rpc_upper,
        },
    }


def print_comparison_summary(
    results_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    confidence_level: float = 0.95,
):
    """
    Print a summary comparison table of all methods.

    Args:
        results_dict: Dictionary mapping method names to (aps, r1s, rpcs) tuples
        confidence_level: Confidence level for CI calculation
    """
    print(f"\n{'=' * 100}")
    print(
        f"EVALUATION RESULTS SUMMARY ({confidence_level * 100:.0f}% Confidence Intervals)"
    )
    print(f"{'=' * 100}")
    print(f"{'Method':<18} {'MAP':<25} {'R@1':<25} {'RPC':<25}")
    print(f"{'-' * 18} {'-' * 25} {'-' * 25} {'-' * 25}")

    for method_name, (aps, r1s, rpcs) in results_dict.items():
        # Calculate statistics
        map_mean, map_std, map_ci, _, _ = compute_confidence_interval(
            aps, confidence_level
        )
        r1_mean, r1_std, r1_ci, _, _ = compute_confidence_interval(
            r1s.float(), confidence_level
        )
        rpc_mean, rpc_std, rpc_ci, _, _ = compute_confidence_interval(
            rpcs, confidence_level
        )

        # Format strings
        map_str = f"{map_mean:.4f} ± {map_ci:.4f}"
        r1_str = f"{r1_mean:.4f} ± {r1_ci:.4f}"
        rpc_str = f"{rpc_mean:.4f} ± {rpc_ci:.4f}"

        print(f"{method_name:<18} {map_str:<25} {r1_str:<25} {rpc_str:<25}")

    print(f"{'=' * 100}")
    print("Legend: Mean ± 95% Confidence Interval")
    print("Note: CI represents uncertainty about the true population mean")
    print("      Smaller CI = more precise estimate of the mean")


def print_tfidf_analysis(results: Dict, detailed_stats: Dict, tfidf_top_k: int):
    """
    Print detailed TF-IDF comparison including filtering analysis.

    Args:
        results: Dictionary of results
        detailed_stats: Dictionary of detailed statistics
        tfidf_top_k: Top-k filtering parameter
    """
    if "tfidf-cosine" not in results or "tfidf-lucene" not in results:
        return

    print(f"\n{'=' * 80}")
    print("TF-IDF DETAILED COMPARISON")
    print(f"{'=' * 80}")

    # Compare methods with filtering
    cosine_stats = detailed_stats["tfidf-cosine"]
    lucene_stats = detailed_stats["tfidf-lucene"]

    print(f"WITH TOP-{tfidf_top_k} FILTERING:")
    print(
        f"  Cosine MAP:   {cosine_stats['map']['mean']:.4f} ± {cosine_stats['map']['ci']:.4f}"
    )
    print(
        f"  Lucene MAP:   {lucene_stats['map']['mean']:.4f} ± {lucene_stats['map']['ci']:.4f}"
    )

    # Compare methods without filtering (if available)
    if "tfidf-cosine_all" in results and "tfidf-lucene_all" in results:
        cosine_all_stats = detailed_stats["tfidf-cosine_all"]
        lucene_all_stats = detailed_stats["tfidf-lucene_all"]

        print(f"\nWITHOUT FILTERING (ALL CANDIDATES):")
        print(
            f"  Cosine MAP:   {cosine_all_stats['map']['mean']:.4f} ± {cosine_all_stats['map']['ci']:.4f}"
        )
        print(
            f"  Lucene MAP:   {lucene_all_stats['map']['mean']:.4f} ± {lucene_all_stats['map']['ci']:.4f}"
        )

        # Analysis of filtering impact
        print(f"\nFILTERING IMPACT ANALYSIS:")
        cosine_improvement = (
            (
                (cosine_stats["map"]["mean"] - cosine_all_stats["map"]["mean"])
                / cosine_all_stats["map"]["mean"]
                * 100
            )
            if cosine_all_stats["map"]["mean"] > 0
            else 0
        )
        lucene_improvement = (
            (
                (lucene_stats["map"]["mean"] - lucene_all_stats["map"]["mean"])
                / lucene_all_stats["map"]["mean"]
                * 100
            )
            if lucene_all_stats["map"]["mean"] > 0
            else 0
        )

        print(f"  Cosine - Top-{tfidf_top_k} vs All: {cosine_improvement:+.1f}% change")
        print(f"  Lucene - Top-{tfidf_top_k} vs All: {lucene_improvement:+.1f}% change")

        if cosine_improvement > 0 and lucene_improvement > 0:
            print(
                f"  → Top-{tfidf_top_k} filtering improves performance for both methods"
            )
        elif cosine_improvement < 0 and lucene_improvement < 0:
            print(f"  → Top-{tfidf_top_k} filtering hurts performance for both methods")
        else:
            print(f"  → Top-{tfidf_top_k} filtering has mixed effects")

    # Method comparison
    method_improvement = (
        (
            (lucene_stats["map"]["mean"] - cosine_stats["map"]["mean"])
            / cosine_stats["map"]["mean"]
            * 100
        )
        if cosine_stats["map"]["mean"] > 0
        else 0
    )
    print(f"\nMETHOD COMPARISON (with filtering):")
    print(f"  Lucene vs Cosine: {method_improvement:+.1f}% difference")

    # Statistical significance check
    overlap = not (
        cosine_stats["map"]["ci_upper"] < lucene_stats["map"]["ci_lower"]
        or lucene_stats["map"]["ci_upper"] < cosine_stats["map"]["ci_lower"]
    )

    if overlap:
        print(
            f"  Note: Confidence intervals overlap - difference may not be statistically significant"
        )
    else:
        print(
            f"  Note: Confidence intervals do not overlap - difference is likely significant"
        )


def print_baseline_bounds_analysis(results: Dict, detailed_stats: Dict):
    """
    Print baseline bounds analysis including modified variants.

    Args:
        results: Dictionary of results
        detailed_stats: Dictionary of detailed statistics
    """
    if not all(k in results for k in ["ideal", "random", "modified-random"]):
        return

    print(f"\n{'=' * 80}")
    print("BASELINE BOUNDS ANALYSIS")
    print(f"{'=' * 80}")

    ideal_stats = detailed_stats["ideal"]
    random_stats = detailed_stats["random"]
    modified_random_stats = detailed_stats["modified-random"]

    print(f"UPPER BOUNDS:")
    print(
        f"  Ideal MAP:          {ideal_stats['map']['mean']:.4f} ± {ideal_stats['map']['ci']:.4f}"
    )
    if "modified_ideal" in detailed_stats:
        modified_ideal_stats = detailed_stats["modified_ideal"]
        print(
            f"  Modified Ideal MAP: {modified_ideal_stats['map']['mean']:.4f} ± {modified_ideal_stats['map']['ci']:.4f}"
        )
    print(
        f"  Modified Random MAP:{modified_random_stats['map']['mean']:.4f} ± {modified_random_stats['map']['ci']:.4f}"
    )

    print(f"\nLOWER BOUNDS:")
    print(
        f"  Random MAP:         {random_stats['map']['mean']:.4f} ± {random_stats['map']['ci']:.4f}"
    )

    # Performance envelope analysis
    max_upper_bound = max(
        ideal_stats["map"]["mean"], modified_random_stats["map"]["mean"]
    )
    if "modified_ideal" in detailed_stats:
        max_upper_bound = max(
            max_upper_bound, detailed_stats["modified_ideal"]["map"]["mean"]
        )

    performance_envelope = max_upper_bound - random_stats["map"]["mean"]
    print(f"\nPERFORMANCE ENVELOPE:")
    print(f"  Total range (best upper bound - random): {performance_envelope:.4f}")
    print(
        f"  Modified-Random provides upper bound by setting distance=0 for lyric matches"
    )

    # Analysis of upper bounds
    if modified_random_stats["map"]["mean"] > ideal_stats["map"]["mean"]:
        improvement = (
            (modified_random_stats["map"]["mean"] - ideal_stats["map"]["mean"])
            / ideal_stats["map"]["mean"]
            * 100
        )
        print(
            f"  → Modified Random is {improvement:.1f}% better than Ideal (highest upper bound)"
        )
    elif modified_random_stats["map"]["mean"] < ideal_stats["map"]["mean"]:
        difference = (
            (ideal_stats["map"]["mean"] - modified_random_stats["map"]["mean"])
            / ideal_stats["map"]["mean"]
            * 100
        )
        print(f"  → Ideal is {difference:.1f}% better than Modified Random")
    else:
        print(f"  → Modified Random and Ideal perform similarly")


def print_additional_statistics(results: Dict):
    """
    Print additional statistics like percentiles and ranges.

    Args:
        results: Dictionary of results
    """
    print(f"\nADDITIONAL STATISTICS:")
    for baseline_name, (aps, r1s, rpcs) in results.items():
        # Calculate percentiles
        ap_percentiles = [
            float(torch.quantile(aps, q).item()) for q in [0.25, 0.5, 0.75]
        ]
        print(
            f"{baseline_name} MAP percentiles: 25th={ap_percentiles[0]:.4f}, "
            f"50th={ap_percentiles[1]:.4f}, 75th={ap_percentiles[2]:.4f}"
        )
        print(
            f"{baseline_name} MAP range: [{float(aps.min().item()):.4f}, "
            f"{float(aps.max().item()):.4f}]"
        )

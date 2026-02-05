#!/usr/bin/env python3
"""
Compute Transcription-Based Baselines for Version Identification

Evaluates transcription-based baselines: SBERT, TF-IDF, and theoretical bounds.

Usage:
    python scripts/compute_baselines.py \
        jobname=shs_test_baselines \
        conf=configs/evaluation/baselines.yaml \
        data.dataset_name=shs \
        data.split=test
"""

import os
import sys
from pathlib import Path

import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.audio_dataset import create_dataloader
from lib.evaluation.baselines import BaselinesEvaluator
from utils import baselines_utils, print_utils


def load_configuration() -> DictConfig:
    """Load and merge configuration from CLI and file."""
    args = OmegaConf.from_cli()
    assert "jobname" in args, "Must provide jobname argument"
    assert "conf" in args, "Must provide conf argument"

    conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
    conf.jobname = args.jobname
    return conf


def setup_fabric(conf: DictConfig) -> Fabric:
    """Initialize Lightning Fabric for distributed computing."""
    fabric = Fabric(
        accelerator="cuda",
        devices=conf.fabric.ngpus,
        num_nodes=conf.fabric.nnodes,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision=conf.fabric.precision,
    )
    fabric.launch()
    fabric.barrier()
    fabric.seed_everything(conf.seed, workers=True)
    return fabric


def setup_pytorch(conf: DictConfig) -> None:
    """Configure PyTorch backend settings."""
    torch.backends.cudnn.benchmark = conf.pytorch.cudnn_benchmark
    torch.backends.cudnn.deterministic = conf.pytorch.cudnn_deterministic
    torch.set_float32_matmul_precision(conf.pytorch.float32_matmul_precision)
    torch.autograd.set_detect_anomaly(conf.pytorch.detect_anomaly)


def main():
    """Main baseline evaluation workflow."""

    # Setup
    conf = load_configuration()
    setup_pytorch(conf)
    fabric = setup_fabric(conf)

    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )

    # Print configuration
    if fabric.is_global_zero:
        print("=" * 65)
        print(OmegaConf.to_yaml(conf)[:-1])
        print("=" * 65)

    # ========================================================================
    # Data Setup
    # ========================================================================
    myprint("Setting up dataloader...")

    # Determine base_path: use specified path or construct from working directory
    base_path = conf.path.get("base_path", os.getcwd())

    dataloader = create_dataloader(
        dataset_name=conf.data.dataset_name,
        base_path=base_path,
        data_folder=conf.path.data,
        batch_size=conf.data.batch_size,
        whisper_set=conf.data.whisper_set,
        split=conf.data.split,
        evaluation_mode=conf.data.evaluation_mode,
        enforce_max_duration=conf.data.get("enforce_max_duration", False),
        num_workers=conf.data.nworkers,
        debug_num_cliques=conf.data.get("debug_num_cliques", None),
    )

    dataloader = fabric.setup_dataloaders(dataloader)

    # ========================================================================
    # Initialize Evaluator
    # ========================================================================
    myprint("Initializing Evaluator...")

    evaluator = BaselinesEvaluator(
        model_name=conf.baselines.sbert.model_name,
        batch_size=conf.baselines.sbert.batch_size,
    )

    # ========================================================================
    # Run Evaluation
    # ========================================================================
    myprint(f"Computing baselines: {conf.baselines.compute}")
    myprint(f"TF-IDF method: {conf.baselines.tfidf.method}")
    myprint(f"TF-IDF top-k: {conf.baselines.tfidf.top_k}")
    if conf.baselines.tfidf.compute_all_variants:
        myprint(
            "TF-IDF will compute both filtered (top-k) and unfiltered (all) variants"
        )

    results = evaluator.evaluate_dataset(
        dataloader,
        encode_chunk_size=conf.baselines.sbert.encode_chunk_size,
        similarity_chunk_size=conf.baselines.sbert.similarity_chunk_size,
        tfidf_method=conf.baselines.tfidf.method,
        tfidf_top_k=conf.baselines.tfidf.top_k,
        compute_baselines=conf.baselines.compute,
        compute_all_tfidf=conf.baselines.tfidf.compute_all_variants,
    )

    # Move results to CPU and synchronize
    for method_name in results:
        aps, r1s, rpcs = results[method_name]
        results[method_name] = (aps.cpu(), r1s.cpu(), rpcs.cpu())

    fabric.barrier()

    # ========================================================================
    # Results Analysis and Reporting
    # ========================================================================
    if fabric.is_global_zero:
        print("\n" + "=" * 100)
        print("DETAILED EVALUATION RESULTS")
        print("=" * 100)

        # Store detailed statistics
        detailed_stats = {}

        # Print detailed results for each baseline
        if conf.reporting.print_detailed_stats:
            for baseline_name, (aps, r1s, rpcs) in results.items():
                stats = baselines_utils.print_detailed_results(
                    baseline_name,
                    aps,
                    r1s,
                    rpcs,
                    confidence_level=conf.reporting.confidence_level,
                )
                detailed_stats[baseline_name] = stats

        # Print comparison summary
        if conf.reporting.print_comparison_table:
            baselines_utils.print_comparison_summary(
                results, confidence_level=conf.reporting.confidence_level
            )

        # Print TF-IDF analysis
        if conf.reporting.print_tfidf_analysis:
            baselines_utils.print_tfidf_analysis(
                results, detailed_stats, conf.baselines.tfidf.top_k
            )

        # Print baseline bounds analysis
        baselines_utils.print_baseline_bounds_analysis(results, detailed_stats)

        # Print additional statistics
        baselines_utils.print_additional_statistics(results)

        # ====================================================================
        # Save Results
        # ====================================================================
        if conf.reporting.save_detailed_results:
            save_data = {
                "raw_results": results,
                "detailed_stats": detailed_stats,
                "config": conf,
                "evaluation_info": {
                    "baselines_computed": conf.baselines.compute,
                    "tfidf_method": conf.baselines.tfidf.method,
                    "tfidf_top_k": conf.baselines.tfidf.top_k,
                    "compute_all_tfidf": conf.baselines.tfidf.compute_all_variants,
                    "confidence_level": conf.reporting.confidence_level,
                    "available_variants": list(results.keys()),
                },
            }

            os.makedirs(conf.path.save_results_path, exist_ok=True)
            save_path = os.path.join(
                conf.path.save_results_path, f"baselines_{conf.jobname}.pt"
            )

            torch.save(save_data, save_path)
            myprint(f"\nDetailed results saved to: {save_path}")

        # ====================================================================
        # Final Summary
        # ====================================================================
        print(f"\n{'=' * 100}")
        print(f"Dataset: {conf.data.dataset_name}")
        print(f"Split: {conf.data.split}")
        print(f"Total queries evaluated: {len(list(results.values())[0][0])}")
        if conf.baselines.tfidf.compute_all_variants:
            print(
                f"TF-IDF variants computed: top-{conf.baselines.tfidf.top_k} filtering + all candidates"
            )
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 100}")

    myprint("\nEvaluation completed!")


if __name__ == "__main__":
    main()

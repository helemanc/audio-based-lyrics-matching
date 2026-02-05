#!/usr/bin/env python3
"""
Feature Extraction Script

Extracts embeddings from audio files using configurable extractors.

Usage:
    python scripts/feature_extraction.py \
        jobname=my_extraction \
        conf=configs/extraction/whisper_base.yaml \
        extraction.type=whisper \
        extraction.skip_existing=true

Supported extractors: whisper, sbert, wealy, clews
"""

import os
import sys
from pathlib import Path

import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import print_utils first (no dependencies)
from utils import print_utils

# Check if this is CLEWS extraction BEFORE importing lib modules
# This avoids whisper import errors in CLEWS container
args_early = OmegaConf.from_cli()
if "conf" in args_early:
    conf_early = OmegaConf.load(args_early.conf)
    is_clews = conf_early.get("extraction", {}).get("type") == "clews"
else:
    is_clews = False

# Import appropriate modules based on extraction type
if is_clews:
    # CLEWS-specific imports
    from utils.clews_utils import run_clews_extraction
else:
    # Standard extraction imports
    from lib.extractors import create_extractor
    from utils.evaluation_utils import run_evaluation
    from utils.extraction_utils import (
        create_dataloader_for_extraction,
        run_extraction_loop,
    )


def main():
    """Main extraction pipeline."""

    # ========================================================================
    # Configuration
    # ========================================================================

    args = OmegaConf.from_cli()
    assert "jobname" in args and "conf" in args, "Must provide jobname and conf"

    conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
    conf.jobname = args.jobname

    # Set defaults
    if not hasattr(conf, "extraction"):
        conf.extraction = OmegaConf.create({})
    conf.extraction.setdefault("type", "whisper")
    conf.extraction.setdefault("skip_existing", True)

    # ========================================================================
    # CLEWS Special Handling
    # ========================================================================
    # CLEWS uses standalone extraction logic to avoid whisper dependencies

    if conf.extraction.type == "clews":
        run_clews_extraction(args, conf)
        return

    # ========================================================================
    # PyTorch Setup
    # ========================================================================

    torch.backends.cudnn.benchmark = conf.pytorch.cudnn_benchmark
    torch.backends.cudnn.deterministic = conf.pytorch.cudnn_deterministic
    torch.set_float32_matmul_precision(conf.pytorch.float32_matmul_precision)
    torch.autograd.set_detect_anomaly(conf.pytorch.detect_anomaly)

    # ========================================================================
    # Fabric Setup
    # ========================================================================

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

    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)

    # ========================================================================
    # Print Configuration
    # ========================================================================

    if fabric.is_global_zero:
        print("=" * 70)
        print(f"FEATURE EXTRACTION: {conf.extraction.type.upper()}")
        print("=" * 70)
        print(OmegaConf.to_yaml(conf)[:-1])
        print("=" * 70)

    # ========================================================================
    # Setup Extractor
    # ========================================================================

    myprint("Creating extractor...")
    extractor = create_extractor(conf.extraction.type, conf, fabric)

    myprint("Loading model...")
    model = extractor.load_model()

    # ========================================================================
    # Setup Dataloader
    # ========================================================================

    myprint("Setting up dataloader...")

    # Wealy returns both dataloader and dataset reference
    if conf.extraction.type == "wealy":
        dataloader, dataset_ref = create_dataloader_for_extraction(conf, fabric)
        extractor.dataset_ref = dataset_ref
    else:
        # Standard dataloader creation (whisper, sbert, etc.)
        dataloader = create_dataloader_for_extraction(conf, fabric)

    # ========================================================================
    # Run Extraction
    # ========================================================================

    query_c, query_i, query_z, cand_c, cand_i, cand_z = run_extraction_loop(
        extractor, dataloader, model, conf, fabric
    )

    # ========================================================================
    # Run Evaluation (Optional)
    # ========================================================================

    # Only run evaluation if:
    # 1. Evaluation is enabled in config
    # 2. Embeddings were actually accumulated (some extractors don't support it)

    if conf.evaluation.get("run_evaluation", False):
        if query_z is not None:
            myprint("Running evaluation...")

            # Handle different evaluation formats
            if conf.extraction.type == "whisper":
                # Unpack encoder/decoder embeddings
                query_z_e, query_z_d = query_z
                cand_z_e, cand_z_d = cand_z

                run_evaluation(
                    query_z_e,
                    query_z_d,
                    cand_z_e,
                    cand_z_d,
                    conf.evaluation.get("distances", ["cosine"]),
                    query_c,
                    query_i,
                    cand_c,
                    cand_i,
                    fabric,
                    conf.data.dataset_name,
                    extractor.decoding_config_name
                    if hasattr(extractor, "decoding_config_name")
                    else conf.jobname,
                )
            else:
                # SBERT/CLEWS - single embedding
                run_evaluation(
                    query_z,
                    query_z,  # Use same for both
                    cand_z,
                    cand_z,
                    conf.evaluation.get("distances", ["cosine"]),
                    query_c,
                    query_i,
                    cand_c,
                    cand_i,
                    fabric,
                    conf.data.dataset_name,
                    f"{conf.data.dataset_name}_{conf.extraction.type}",
                )
        else:
            myprint(
                f"Skipping evaluation - {conf.extraction.type} doesn't support "
                "evaluation during extraction"
            )

    myprint("✓ Feature extraction completed successfully!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute Distance Matrix for Version Identification

This script computes pairwise distance matrices for trained WEALY models.
It supports two modes:
  1. Standard mode: Single embedding per song
  2. Overlapping chunks mode: Multiple overlapping chunks per song

The output distance matrix can be used for:
  - Analysis and debugging
  - Multimodal fusion (combining with other modalities like CLEWS)
  - Custom retrieval experiments

Usage:
    # Standard mode (auto-saves to logs/<config_name>/distance_matrix/)
    python scripts/compute_distance_matrix.py \
        checkpoint=logs/wealy_shs/best.ckpt \
        partition=test

    # Overlapping chunks mode with topk averaging
    python scripts/compute_distance_matrix.py \
        checkpoint=logs/wealy_shs/best.ckpt \
        partition=test \
        use_overlapping_chunks=true \
        overlap_percentage=0.9 \
        chunk_size=1500 \
        topk_distance=3

    # Custom save location (optional)
    python scripts/compute_distance_matrix.py \
        checkpoint=logs/wealy_shs/best.ckpt \
        partition=test \
        save_distance_matrix=/custom/path/distances.pkl

Default save location: logs/<config_name>/distance_matrix/<partition>_<mode>_distances.pkl

Output format (pickle file):
    {
        'distance_matrix': np.ndarray,  # (n_queries, n_candidates)
        'query_references': [{'clique': int, 'version': int, 'matrix_row': int}, ...],
        'candidate_references': [{'clique': int, 'version': int, 'matrix_col': int}, ...],
        'metadata': {
            'checkpoint': str,
            'partition': str,
            'use_overlapping_chunks': bool,
            'topk_distance': int,
            ...
        }
    }
"""

import importlib
import math
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

from lib.embedding_dataset.base_dataset import EmbeddingDataset
from lib.embedding_dataset.collate_functions import create_collate_fn
from lib.evaluation import eval
from utils import distance_matrix_utils, print_utils, pytorch_utils


def main():
    """Main distance matrix computation workflow."""
    # Parse arguments
    args = OmegaConf.from_cli()
    assert "checkpoint" in args, "Must provide checkpoint=path/to/checkpoint.ckpt"

    log_path, _ = os.path.split(args.checkpoint)

    # Set defaults
    args.ngpus = getattr(args, "ngpus", 1)
    args.nnodes = getattr(args, "nnodes", 1)
    args.precision = "bf16-mixed"
    args.partition = getattr(args, "partition", "test")
    args.limit_num = getattr(args, "limit_num", None)
    args.use_overlapping_chunks = getattr(args, "use_overlapping_chunks", False)
    args.overlap_percentage = getattr(args, "overlap_percentage", 0.9)
    args.chunk_size = getattr(args, "chunk_size", 1500)
    args.topk_distance = getattr(args, "topk_distance", 1)

    # Auto-generate save path if not provided
    if "save_distance_matrix" not in args or args.save_distance_matrix is None:
        # Default: save in logs/<config_name>/distance_matrix/<partition>_distances.pkl
        distance_matrix_dir = os.path.join(log_path, "distance_matrix")
        mode_suffix = "_overlapping" if args.use_overlapping_chunks else "_standard"
        topk_suffix = f"_topk{args.topk_distance}" if args.topk_distance > 1 else ""
        filename = f"{args.partition}{mode_suffix}{topk_suffix}_distances.pkl"
        args.save_distance_matrix = os.path.join(distance_matrix_dir, filename)
    else:
        args.save_distance_matrix = getattr(args, "save_distance_matrix", None)

    # Init Fabric
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(False)

    fabric = Fabric(
        accelerator="cuda",
        devices=args.ngpus,
        num_nodes=args.nnodes,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision=args.precision,
    )
    fabric.launch()
    fabric.barrier()
    fabric.seed_everything(44 + fabric.global_rank, workers=True)

    # Init print utilities
    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )
    myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
        it, desc=desc, leave=leave, doit=fabric.is_global_zero
    )
    fabric.barrier()

    # Load config
    myprint("=" * 70)
    myprint("COMPUTE DISTANCE MATRIX")
    myprint("=" * 70)
    myprint(OmegaConf.to_yaml(args))
    myprint("Loading model configuration...")
    conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))

    # Init model
    myprint("Initializing model...")
    module = importlib.import_module("lib.models." + conf.model.name)
    with fabric.init_module():
        model = module.Model(
            conf.model,
            use_avg_pooling=conf.data.use_avg_pooling,
            embedding_type=conf.data.embedding_type,
            sr=conf.data.samplerate,
        )
    model = fabric.setup(model)
    model.mark_forward_method("embed")

    # Load checkpoint
    myprint("Loading checkpoint...")
    state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
    fabric.load(args.checkpoint, state)
    model, _, _, conf, epoch, _, best = pytorch_utils.set_state(state)
    myprint(f"  Loaded epoch {epoch}, best metric: {best:.3f}")
    model.eval()

    # Setup dataset
    myprint("Setting up dataset...")
    dset = EmbeddingDataset(
        conf,
        split=args.partition,
        augment=False,
        embedding_type=conf.data.embedding_type,
        embedding_format=conf.data.embedding_format,
        verbose=fabric.is_global_zero,
        return_paths=True,
    )

    collate_fn = create_collate_fn(
        conf,
        deterministic=not args.use_overlapping_chunks,
        use_overlapping_chunks=args.use_overlapping_chunks,
        overlap_percentage=args.overlap_percentage,
    )

    dloader = torch.utils.data.DataLoader(
        dset,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    dloader = fabric.setup_dataloaders(dloader)

    # Main evaluation
    mode_str = "overlapping chunks" if args.use_overlapping_chunks else "standard"
    topk_str = f" (topk={args.topk_distance})" if args.topk_distance > 1 else ""
    myprint(f"\nMode: {mode_str}{topk_str}")
    myprint(f"Partition: {args.partition}")
    if args.save_distance_matrix:
        myprint(f"Will save distance matrix to: {args.save_distance_matrix}")
    myprint("")

    with torch.inference_mode():
        myprint("Extracting embeddings...")

        if not args.use_overlapping_chunks:
            # Standard mode
            query_c, query_i, query_z, query_m, _ = (
                distance_matrix_utils.extract_embeddings_from_precomputed(
                    model, dloader, conf, args, myprogbar, desc="Extracting"
                )
            )

            query_c = query_c.int()
            query_i = query_i.int()
            query_z = query_z.half()

            cand_c, cand_i, cand_z, cand_m = (
                query_c.clone(),
                query_i.clone(),
                query_z.clone(),
                query_m.clone(),
            )

            fabric.barrier()
            cand_c = torch.cat(torch.unbind(fabric.all_gather(cand_c), dim=0), dim=0)
            cand_i = torch.cat(torch.unbind(fabric.all_gather(cand_i), dim=0), dim=0)
            cand_z = torch.cat(torch.unbind(fabric.all_gather(cand_z), dim=0), dim=0)
            cand_m = torch.cat(torch.unbind(fabric.all_gather(cand_m), dim=0), dim=0)

            # Evaluate
            aps, r1s, rpcs = [], [], []
            for n in myprogbar(
                range(len(query_z)), desc="Computing distances", leave=True
            ):
                ap, r1, rpc = eval.compute(
                    model,
                    query_c[n : n + 1],
                    query_i[n : n + 1],
                    query_z[n : n + 1],
                    cand_c,
                    cand_i,
                    cand_z,
                    batch_size_candidates=2**15,
                )
                aps.append(ap)
                r1s.append(r1)
                rpcs.append(rpc)

            aps = torch.stack(aps)
            r1s = torch.stack(r1s)
            rpcs = torch.stack(rpcs)

            fabric.barrier()
            aps = torch.cat(torch.unbind(fabric.all_gather(aps), dim=0), dim=0)
            r1s = torch.cat(torch.unbind(fabric.all_gather(r1s), dim=0), dim=0)
            rpcs = torch.cat(torch.unbind(fabric.all_gather(rpcs), dim=0), dim=0)

        else:
            # Overlapping chunks mode
            _, _, _, _, query_chunks = (
                distance_matrix_utils.extract_embeddings_from_precomputed(
                    model, dloader, conf, args, myprogbar, desc="Extracting"
                )
            )

            aps, r1s, rpcs = distance_matrix_utils.evaluate_with_distance_saving(
                fabric, query_chunks, args, myprint
            )
            fabric.barrier()
            aps, r1s, rpcs = distance_matrix_utils.gather_evaluation_results(
                fabric, aps, r1s, rpcs
            )

    # Print results
    aps = aps.cpu()
    r1s = r1s.cpu()
    rpcs = rpcs.cpu()

    fabric.barrier()

    if fabric.is_global_zero:
        map_val = float(aps.mean().item())
        mr1_val = float(r1s.mean().item())
        arp_val = float(rpcs.mean().item())

        map_ci = float((1.96 * aps.std() / math.sqrt(len(aps))).item())
        mr1_ci = float((1.96 * r1s.std() / math.sqrt(len(r1s))).item())
        arp_ci = float((1.96 * rpcs.std() / math.sqrt(len(rpcs))).item())

        print()
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"MAP: {map_val:.4f} ± {map_ci:.4f}")
        print(f"MR1: {mr1_val:.4f} ± {mr1_ci:.4f}")
        print(f"ARP: {arp_val:.4f} ± {arp_ci:.4f}")
        print(f"Queries: {len(aps)}")
        print("=" * 70)

        if args.save_distance_matrix:
            print(f"\n✓ Distance matrix saved to: {args.save_distance_matrix}")

    fabric.barrier()


if __name__ == "__main__":
    main()

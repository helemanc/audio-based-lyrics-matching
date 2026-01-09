#!/usr/bin/env python3
"""
Evaluation script for unimodal version identification models.

Supports:
- Standard evaluation: Single embedding per version
- Overlapping chunks: Multiple overlapping chunk embeddings per version
- Distributed evaluation with checkpointing
- Resumable evaluation from checkpoints

Usage:
    python scripts/evaluate.py \
        checkpoint=path/to/checkpoint.ckpt \
        partition=test \
        use_overlapping_chunks=true \
        ngpus=4
"""

import sys
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import inference_utils


#########################################################################################
# Main
#########################################################################################

def main() -> None:
    """Main evaluation entry point."""
    
    print("Starting distributed evaluation")
    
    # Setup configuration
    args = inference_utils.setup_configuration()
    
    # Setup PyTorch and Fabric
    inference_utils.setup_pytorch(args)
    fabric = inference_utils.setup_fabric(args)
    
    # Print configuration
    inference_utils.print_configuration(args, fabric)
    
    # Setup checkpointing
    checkpointing_enabled = not args.disable_checkpointing
    verbose_memory = not args.disable_memory_logging
    
    checkpoint_manager = inference_utils.EvaluationCheckpoint(
        args.checkpoint_dir, fabric, enabled=checkpointing_enabled
    )
    
    try:
        # Load model and data
        model, conf = inference_utils.load_model_and_config(args, fabric, verbose_memory)
        dloader = inference_utils.setup_dataloader(conf, args, fabric, verbose_memory)
        
        # Evaluate
        with torch.inference_mode():
            if args.use_overlapping_chunks:
                print("Evaluating with overlapping chunks...")
                
                # Extract embeddings
                _, _, _, _, chunks = inference_utils.extract_embeddings_with_checkpointing(
                    model, dloader, args, fabric, checkpoint_manager,
                    verbose_memory=verbose_memory, conf=conf
                )
                
                inference_utils.log_memory_usage(
                    fabric, "After embedding extraction", verbose=verbose_memory
                )
                
                # Evaluate chunks
                aps, r1s, rpcs = inference_utils.evaluate_overlapping_chunks_fast(
                    fabric, chunks, args, checkpoint_manager, verbose_memory=verbose_memory
                )
                
                # Gather results
                print("Gathering final results...")
                aps, r1s, rpcs = inference_utils.gather_results_safely(fabric, aps, r1s, rpcs)
                
            else:
                print("Standard evaluation mode...")
                
                # Extract embeddings
                q_c, q_i, q_z, q_m, _ = inference_utils.extract_embeddings_with_checkpointing(
                    model, dloader, args, fabric, checkpoint_manager,
                    verbose_memory=verbose_memory, conf=conf
                )
                
                inference_utils.log_memory_usage(
                    fabric, "After embedding extraction", verbose=verbose_memory
                )
                
                # Evaluate
                aps, r1s, rpcs = inference_utils.evaluate_standard_mode(
                    model, q_c, q_i, q_z, q_m, fabric, checkpoint_manager, args
                )
                
                # Gather results
                aps, r1s, rpcs = inference_utils.gather_results_safely(fabric, aps, r1s, rpcs)
        
        # Report results (only on rank 0)
        if len(aps) > 0:
            inference_utils.print_results(aps, r1s, rpcs, args, fabric)
            inference_utils.save_results(aps, r1s, rpcs, args, fabric)
        
        print("✓ Evaluation completed successfully!")
    
    except Exception as e:
        print(f"[GPU {fabric.global_rank}] Fatal error: {e}")
        import traceback
        import time
        import os
        traceback.print_exc()
        
        # Save crash report
        crash_report_path = os.path.join(
            args.checkpoint_dir,
            f"crash_report_rank_{fabric.global_rank}.txt"
        )
        with open(crash_report_path, 'w') as f:
            f.write(f"Crash at {time.ctime()}\n")
            f.write(f"Error: {e}\n")
            f.write(traceback.format_exc())
        
        print(f"✗ Crash report saved to {crash_report_path}")


if __name__ == "__main__":
    main()
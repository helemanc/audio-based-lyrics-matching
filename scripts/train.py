#!/usr/bin/env python3
"""
Training script for unimodal version identification models.

Supports single-modal models:
- Whisper (encoder/decoder embeddings)
- SBERT (sentence embeddings)
- WEALY (learned embeddings)

Usage:
    python scripts/train.py \
        jobname=my_experiment \
        conf=configs/training/whisper_base.yaml
"""

import sys
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf, DictConfig
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import print_utils, pytorch_utils, training_utils


def load_configuration() -> DictConfig:
    """
    Load and merge configuration from CLI and file.
    
    Returns:
        Merged configuration
    
    Raises:
        AssertionError: If required arguments are missing
    """
    args = OmegaConf.from_cli()
    assert "jobname" in args, "Must provide jobname argument"
    assert "conf" in args, "Must provide conf argument"
    
    conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
    conf.jobname = args.jobname
    conf.data.path = conf.path
    conf.path.logs = os.path.join(conf.path.logs, conf.jobname)
    
    return conf


def setup_fabric(conf: DictConfig) -> Fabric:
    """
    Initialize Lightning Fabric for distributed training.
    
    Args:
        conf: Configuration object
    
    Returns:
        Initialized Fabric instance
    """
    fabric = Fabric(
        accelerator="cuda",
        devices=conf.fabric.ngpus,
        num_nodes=conf.fabric.nnodes,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision=conf.fabric.precision,
        loggers=pytorch_utils.get_logger(conf.path.logs),
    )
    fabric.launch()
    fabric.barrier()
    fabric.seed_everything(conf.seed, workers=True)
    
    return fabric


def setup_pytorch(conf: DictConfig) -> None:
    """
    Configure PyTorch backend settings.
    
    Args:
        conf: Configuration object
    """
    torch.backends.cudnn.benchmark = conf.pytorch.cudnn_benchmark
    torch.backends.cudnn.deterministic = conf.pytorch.cudnn_deterministic
    torch.set_float32_matmul_precision(conf.pytorch.float32_matmul_precision)
    torch.autograd.set_detect_anomaly(conf.pytorch.detect_anomaly)


def initialize_training_components(
    conf: DictConfig,
    fabric: Fabric
) -> tuple:
    """
    Initialize model, optimizer, scheduler, and early stopping.
    
    Args:
        conf: Configuration object
        fabric: Fabric instance
    
    Returns:
        Tuple of (model, optimizer, scheduler, sched_on_epoch, early_stopping)
    """
    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    
    # Initialize model
    myprint("Initializing model...")
    model = training_utils.initialize_model(conf, fabric, verbose=True)
    
    # Initialize optimizer
    myprint("Initializing optimizer...")
    optim = pytorch_utils.get_optimizer(conf.training.optim, model)
    optim = fabric.setup_optimizers(optim)
    
    # Initialize scheduler
    sched, sched_on_epoch = pytorch_utils.get_scheduler(
        conf.training.optim,
        optim,
        epochs=conf.training.numepochs,
        mode=conf.training.monitor.mode,
    )
    
    # Initialize early stopping
    myprint("Initializing early stopping...")
    early_stopping_config = conf.get('early_stopping', {})
    early_stopping = pytorch_utils.EarlyStopping(
        patience=early_stopping_config.get('patience', 10),
        mode=early_stopping_config.get('mode', 'max'),
        min_delta=early_stopping_config.get('min_delta', 0.0),
        enabled=early_stopping_config.get('enabled', False)
    )
    
    if early_stopping.enabled:
        myprint(
            f"  Early stopping: patience={early_stopping.patience}, "
            f"mode={early_stopping.mode}"
        )
    else:
        myprint("  Early stopping disabled")
    
    return model, optim, sched, sched_on_epoch, early_stopping


def load_checkpoint_if_exists(
    conf: DictConfig,
    model, optim, sched,
    fabric: Fabric
) -> tuple:
    """
    Load checkpoint if it exists.
    
    Args:
        conf: Configuration object
        model: Model instance
        optim: Optimizer
        sched: Scheduler
        fabric: Fabric instance
    
    Returns:
        Tuple of (model, optim, sched, conf, epoch, lr, cost_best)
    """
    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    
    epoch = 0
    cost_best = torch.inf if conf.training.monitor.mode == "min" else -torch.inf
    
    # Initialize scheduler with best cost
    if conf.training.optim.sched.startswith("plateau"):
        sched.step(cost_best)
    
    lr = sched.get_last_lr()[0]
    
    # Check for checkpoint
    fn_ckpt_last, _, _ = training_utils.get_checkpoint_paths(conf.path.logs)
    
    fn_ckpt = None
    if conf.checkpoint is not None:
        fn_ckpt = conf.checkpoint
    elif os.path.exists(fn_ckpt_last):
        fn_ckpt = fn_ckpt_last
    
    if fn_ckpt is not None:
        myprint(f"Loading checkpoint: {fn_ckpt}")
        state = pytorch_utils.get_state(model, optim, sched, conf, epoch, lr, cost_best)
        fabric.load(fn_ckpt, state)
        model, optim, sched, conf, epoch, lr, cost_best = pytorch_utils.set_state(state)
        myprint(f"  ✓ Resumed from epoch {epoch}")
    
    return model, optim, sched, conf, epoch, lr, cost_best


def main() -> None:
    """Main training entry point."""
    
    # Load configuration
    conf = load_configuration()
    
    # Setup PyTorch
    setup_pytorch(conf)
    
    # Setup Fabric
    fabric = setup_fabric(conf)
    
    myprint = lambda s: print_utils.myprint(s, doit=fabric.is_global_zero)
    
    # Print config
    myprint("-" * 70)
    myprint(OmegaConf.to_yaml(conf)[:-1])
    myprint("-" * 70)
    

    #Save configuration to log directory for inference
    log_dir = training_utils.ensure_checkpoint_directory(conf, fabric)
    training_utils.save_configuration(conf, log_dir, fabric)
    
    # Initialize training components
    model, optim, sched, sched_on_epoch, early_stopping = (
        initialize_training_components(conf, fabric)
    )
    
    # Load checkpoint if exists
    model, optim, sched, conf, epoch, lr, cost_best = (
        load_checkpoint_if_exists(conf, model, optim, sched, fabric)
    )
    
    # Re-seed for different augmentations per GPU
    myprint("Re-seeding for distributed training...")
    fabric.barrier()
    fabric.seed_everything(
        (epoch + 1) * (conf.seed + fabric.global_rank), workers=True
    )
    
    # Load data
    myprint("Loading datasets...")
    ds_train, ds_valid = training_utils.create_datasets(conf, fabric)
    dl_train, dl_valid = training_utils.create_dataloaders(
        ds_train, ds_valid, conf, fabric
    )
    
    # Train
    training_utils.train(
        model, optim, sched, sched_on_epoch,
        dl_train, dl_valid, early_stopping,
        conf, fabric,
        start_epoch=epoch,
        cost_best=cost_best
    )
    
    myprint("✓ Training completed!")


if __name__ == "__main__":
    main()
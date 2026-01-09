"""
Training utilities for unimodal version identification models.

Provides functions for:
- Model initialization and setup
- Dataset and dataloader creation
- Batch processing and loss computation
- Training and validation loops

Supports single-modal models: Whisper, SBERT, WEALY
"""

import math
import warnings
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from lightning import Fabric
import torchinfo

from lib import dataset
from utils import print_utils, pytorch_utils
from lib.evaluation import eval

import os 


#########################################################################################
# Type Definitions
#########################################################################################

LogDict = pytorch_utils.LogDict
BatchType = List[torch.Tensor]
OutputsType = List[Union[torch.Tensor, None]]


#########################################################################################
# Model Initialization
#########################################################################################

def initialize_model(
    conf: DictConfig,
    fabric: Fabric,
    verbose: bool = True
) -> nn.Module:
    """
    Initialize unimodal model based on configuration.
    
    Supports single-modal architectures for embeddings from:
    - Whisper (encoder/decoder)
    - SBERT (sentence embeddings)
    - WEALY (learned embeddings)
    
    Args:
        conf: Configuration object with model specifications
        fabric: Fabric instance for distributed training
        verbose: Whether to print model summary
    
    Returns:
        Initialized and Fabric-wrapped model with marked forward methods
    
    Example:
        >>> model = initialize_model(conf, fabric, verbose=True)
        >>> # Model ready for training with prepare/embed/loss methods
    """
    # Import model module
    module = importlib.import_module("lib.models." + conf.model.name)
    
    # Initialize model
    with fabric.init_module():
        model = module.Model(
            conf.model,
            use_avg_pooling=conf.data.get('use_avg_pooling', False),
            embedding_type=conf.data.get('embedding_type', 'last_hidden_states'),
            sr=conf.data.samplerate
        )
    
    # Print summary
    if verbose and fabric.is_global_zero:
        torchinfo.summary(model, depth=2)
    
    # Setup with Fabric
    model = fabric.setup(model)
    model.mark_forward_method("prepare")
    model.mark_forward_method("embed")
    model.mark_forward_method("loss")
    
    return model


#########################################################################################
# Data Loading
#########################################################################################

def create_datasets(
    conf: DictConfig,
    fabric: Fabric
) -> Tuple[dataset.EmbeddingDataset, dataset.EmbeddingDataset]:
    """
    Create train and validation embedding datasets.
    
    Uses EmbeddingDataset for pre-extracted embeddings from:
    - Whisper (encoder/decoder embeddings)
    - SBERT (sentence embeddings)
    - WEALY (learned embeddings)
    
    Args:
        conf: Configuration object
        fabric: Fabric instance
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    
    Example:
        >>> ds_train, ds_valid = create_datasets(conf, fabric)
        >>> print(f"Train: {len(ds_train)}, Valid: {len(ds_valid)}")
    """
    verbose = fabric.is_global_zero
    
    if verbose:
        print(f"Loading {conf.data.embedding_type} embedding dataset...")
    
    ds_train = dataset.EmbeddingDataset(
        conf,
        split="train",
        augment=False,
        embedding_type=conf.data.get('embedding_type', 'last_hidden_states'),
        embedding_format=conf.data.get('embedding_format', 'concat'),
        verbose=verbose
    )
    
    ds_valid = dataset.EmbeddingDataset(
        conf,
        split="val",
        augment=False,
        embedding_type=conf.data.get('embedding_type', 'last_hidden_states'),
        embedding_format=conf.data.get('embedding_format', 'concat'),
        verbose=verbose
    )
    
    return ds_train, ds_valid


def create_dataloaders(
    ds_train: dataset.EmbeddingDataset,
    ds_valid: dataset.EmbeddingDataset,
    conf: DictConfig,
    fabric: Fabric
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with appropriate collate functions.
    
    Train dataloader uses non-deterministic chunk selection for augmentation.
    Validation dataloader uses deterministic (first chunk) selection.
    
    Args:
        ds_train: Training dataset
        ds_valid: Validation dataset
        conf: Configuration object
        fabric: Fabric instance
    
    Returns:
        Tuple of (train_dataloader, validation_dataloader)
    
    Raises:
        AssertionError: If batch size is not > 1
    
    Example:
        >>> dl_train, dl_valid = create_dataloaders(ds_train, ds_valid, conf, fabric)
        >>> for batch in dl_train:
        ...     # Training batch
        ...     break
    """
    assert conf.training.batchsize > 1, "Batch size must be > 1"
    
    # Create collate functions
    collate_fn_train = dataset.create_collate_fn(
        conf,
        deterministic=False,
        use_overlapping_chunks=False,
        apply_masks_with_padding=conf.data.get('apply_masks_with_padding', False)
    )
    
    collate_fn_val = dataset.create_collate_fn(
        conf,
        deterministic=True,  # Always take first chunk for validation
        use_overlapping_chunks=False,
        apply_masks_with_padding=conf.data.get('apply_masks_with_padding', False)
    )
    
    # Create dataloaders
    dl_train = DataLoader(
        ds_train,
        num_workers=conf.data.nworkers,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=collate_fn_train,
        batch_size=conf.training.batchsize,
        shuffle=True,
        drop_last=False
    )
    
    dl_valid = DataLoader(
        ds_valid,
        num_workers=conf.data.nworkers,
        persistent_workers=False,
        pin_memory=True,
        collate_fn=collate_fn_val,
        batch_size=conf.training.batchsize,
        shuffle=False,
        drop_last=False
    )
    
    # Setup with Fabric
    dl_train, dl_valid = fabric.setup_dataloaders(dl_train, dl_valid)
    
    return dl_train, dl_valid


#########################################################################################
# Batch Processing
#########################################################################################

def extract_batch_components(
    batch: BatchType,
    conf: DictConfig
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract components from unimodal batch.
    
    Batch structure:
    [class_ids, ver_ids_1, emb_1, mask_1, ver_ids_2, emb_2, mask_2, ...]
    
    Each version has 3 items:
    - version_id: Unique identifier
    - embedding: Pre-extracted embedding tensor
    - mask: Padding/validity mask
    
    Args:
        batch: Input batch from dataloader
        conf: Configuration object
    
    Returns:
        Tuple of (class_ids, version_ids, embeddings, masks)
        - class_ids: Tensor of shape (batch_size * n_per_class,)
        - version_ids: Tensor of shape (batch_size * n_per_class,)
        - embeddings: Tensor of shape (batch_size * n_per_class, seq_len, emb_dim)
        - masks: Tensor of shape (batch_size * n_per_class, seq_len)
    
    Example:
        >>> cc, ii, xx, masks = extract_batch_components(batch, conf)
        >>> print(f"Classes: {cc.shape}, Embeddings: {xx.shape}")
    """
    n_per_class = conf.data.n_per_class
    items_per_version = 3  # ver_id, embedding, mask
    
    # Extract class labels (repeated for each version)
    cc = torch.cat([batch[0]] * n_per_class, dim=0)
    
    # Extract version IDs: positions 1, 4, 7, ... (stride = 3)
    ii = torch.cat(batch[1::items_per_version], dim=0)
    
    # Extract embeddings: positions 2, 5, 8, ... (stride = 3)
    xx = torch.cat(batch[2::items_per_version], dim=0)
    
    # Extract masks: positions 3, 6, 9, ... (stride = 3)
    masks = torch.cat(batch[3::items_per_version], dim=0)
    
    return cc, ii, xx, masks


#########################################################################################
# Loss Computation
#########################################################################################

def compute_loss(
    batch: BatchType,
    logdict: LogDict,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    sched_on_epoch: bool,
    conf: DictConfig,
    fabric: Fabric,
    training: bool = False
) -> Tuple[OutputsType, LogDict]:
    """
    Compute loss for unimodal model.
    
    Process:
    1. Extract batch components (class_ids, version_ids, embeddings, masks)
    2. Clone tensors for gradient computation
    3. Forward pass: prepare -> embed -> loss
    4. Backward pass if training
    5. Prepare outputs for evaluation
    
    Args:
        batch: Input batch from dataloader
        logdict: Dictionary for logging metrics
        model: Model instance
        optim: Optimizer
        sched: Learning rate scheduler
        sched_on_epoch: Whether scheduler steps per epoch (vs per step)
        conf: Configuration object
        fabric: Fabric instance
        training: Whether in training mode (enables gradients)
    
    Returns:
        Tuple of (outputs_for_evaluation, updated_logdict)
        - outputs: List with [class_ids, ver_ids_1, embeddings_1, ver_ids_2, ...]
        - logdict: Updated with loss values
    
    Example:
        >>> outputs, logdict = compute_loss(
        ...     batch, logdict, model, optim, sched, False,
        ...     conf, fabric, training=True
        ... )
        >>> print(f"Loss: {logdict.get('l_main')[-1]:.4f}")
    """
    n_per_class = conf.data.n_per_class
    
    # Extract batch components
    with torch.inference_mode():
        cc, ii, xx, masks = extract_batch_components(batch, conf)
    
    # Clone for gradient computation
    cc = cc.clone()
    ii = ii.clone()
    xx = xx.clone()
    masks = masks.clone()
    
    # Training step
    if training:
        optim.zero_grad(set_to_none=True)
    
    # Forward pass
    xx_prepared = model.prepare(xx)
    zz, extra = model.embed(xx_prepared, masks)
    loss, logdct = model.loss(cc, ii, zz, extra=extra)
    
    # Backward pass
    if training:
        fabric.backward(loss)
        optim.step()
        if not sched_on_epoch:
            sched.step()
    
    # Prepare outputs for evaluation
    with torch.inference_mode():
        clist = torch.chunk(cc, n_per_class, dim=0)
        ilist = torch.chunk(ii, n_per_class, dim=0)
        zlist = torch.chunk(zz, n_per_class, dim=0)
        
        outputs = [clist[0]] + [None] * (2 * n_per_class)
        outputs[1::2] = ilist
        outputs[2::2] = zlist
        
        logdict.append(logdct)
    
    return outputs, logdict


#########################################################################################
# Training Loop
#########################################################################################

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    sched_on_epoch: bool,
    conf: DictConfig,
    fabric: Fabric,
    desc: Optional[str] = None
) -> LogDict:
    """
    Train for one epoch.
    
    Args:
        model: Model instance
        dataloader: Training dataloader
        optim: Optimizer
        sched: Learning rate scheduler
        sched_on_epoch: Whether scheduler steps per epoch (vs per step)
        conf: Configuration object
        fabric: Fabric instance
        desc: Description for progress bar
    
    Returns:
        LogDict with training metrics (losses)
    
    Example:
        >>> logdict = train_epoch(model, dl_train, optim, sched, False, conf, fabric)
        >>> print(f"Avg loss: {logdict.get('l_main'):.4f}")
    """
    model.train()
    logdict = pytorch_utils.LogDict()
    
    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )
    myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
        it, desc=desc, leave=leave, doit=fabric.is_global_zero
    )
    
    fabric.barrier()
    
    for n, batch in enumerate(myprogbar(dataloader, desc=desc)):
        if conf.limit_batches is not None and n >= conf.limit_batches:
            break
        
        _, logdict = compute_loss(
            batch, logdict, model, optim, sched, sched_on_epoch,
            conf, fabric, training=True
        )
        
        losses = logdict.get("l_main")
        myprint(f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}]", end="")
    
    return logdict


#########################################################################################
# Validation Loop
#########################################################################################

@torch.inference_mode()
def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    sched_on_epoch: bool,
    conf: DictConfig,
    fabric: Fabric,
    desc: Optional[str] = None
) -> LogDict:
    """
    Validate for one epoch with retrieval evaluation.
    
    Computes:
    - Loss on validation set
    - Retrieval metrics (MAP, MR1, ARP) using cosine distance
    
    Args:
        model: Model instance
        dataloader: Validation dataloader
        optim: Optimizer (not used, for signature compatibility)
        sched: Scheduler (not used, for signature compatibility)
        sched_on_epoch: Whether scheduler steps per epoch
        conf: Configuration object
        fabric: Fabric instance
        desc: Description for progress bar
    
    Returns:
        LogDict with validation metrics (loss, MAP, MR1, ARP, COMP)
    
    Example:
        >>> logdict = validate_epoch(model, dl_valid, optim, sched, False, conf, fabric)
        >>> print(f"MAP: {logdict.get('m_MAP'):.4f}")
    """
    model.eval()
    logdict = pytorch_utils.LogDict()
    
    queries_c: List[torch.Tensor] = []
    queries_i: List[torch.Tensor] = []
    queries_z: List[torch.Tensor] = []
    
    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )
    myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(
        it, desc=desc, leave=leave, doit=fabric.is_global_zero
    )
    
    fabric.barrier()
    
    # Compute loss and accumulate embeddings
    for n, batch in enumerate(myprogbar(dataloader, desc=desc)):
        outputs, logdict = compute_loss(
            batch, logdict, model, optim, sched, sched_on_epoch,
            conf, fabric, training=False
        )
        
        losses = logdict.get("l_main")
        myprint(f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}]", end="")
        
        # Accumulate for MAP evaluation
        cl, i1, z1 = outputs[:3]
        queries_c.append(cl)
        queries_i.append(i1)
        queries_z.append(z1)
    
    # Concatenate queries
    queries_c = torch.cat(queries_c, dim=0)
    queries_i = torch.cat(queries_i, dim=0)
    queries_z = torch.cat(queries_z, dim=0)
    
    # Gather from all GPUs
    fabric.barrier()
    all_c = fabric.all_gather(queries_c)
    all_i = fabric.all_gather(queries_i)
    all_z = fabric.all_gather(queries_z)
    
    all_c = torch.cat(torch.unbind(all_c, dim=0), dim=0)
    all_i = torch.cat(torch.unbind(all_i, dim=0), dim=0)
    all_z = torch.cat(torch.unbind(all_z, dim=0), dim=0)
    
    # Evaluate retrieval
    myprint("Eval... ", end="")
    aps, r1s, rpcs = eval.compute(
        model,
        queries_c,
        queries_i,
        queries_z,
        all_c,
        all_i,
        all_z,
        distance_fn="cosine",
    )
    
    # Compute composite metric
    comp = (rpcs * (1 - aps)) ** 0.5
    
    logdict.append({
        "m_MAP": aps,
        "m_MR1": r1s,
        "m_ARP": rpcs,
        "m_COMP": comp
    })
    
    return logdict


#########################################################################################
# Checkpoint Management
#########################################################################################

def get_checkpoint_paths(log_dir: str) -> Tuple[str, str, str]:
    """
    Get paths for checkpoint files.
    
    Args:
        log_dir: Directory for logs and checkpoints
    
    Returns:
        Tuple of (last_checkpoint, best_checkpoint, epoch_checkpoint_template)
    
    Example:
        >>> last, best, epoch = get_checkpoint_paths('/logs/exp1')
        >>> print(last)
        /logs/exp1/checkpoint_last.ckpt
    """
    fn_ckpt_last = os.path.join(log_dir, "checkpoint_last.ckpt")
    fn_ckpt_best = os.path.join(log_dir, "checkpoint_best.ckpt")
    fn_ckpt_epoch = os.path.join(log_dir, "checkpoint_$epoch$.ckpt")
    
    return fn_ckpt_last, fn_ckpt_best, fn_ckpt_epoch


def should_save_checkpoint(
    epoch: int,
    save_freq: Optional[int]
) -> bool:
    """
    Determine if checkpoint should be saved this epoch.
    
    Args:
        epoch: Current epoch (0-indexed)
        save_freq: Save frequency in epochs (None = no periodic saving)
    
    Returns:
        True if checkpoint should be saved
    
    Example:
        >>> should_save_checkpoint(49, save_freq=10)
        False
        >>> should_save_checkpoint(50, save_freq=10)
        True
    """
    if save_freq is None:
        return False
    return (epoch + 1) % save_freq == 0


def is_best_model(
    cost_current: float,
    cost_best: float,
    mode: str
) -> bool:
    """
    Check if current model is best so far.
    
    Args:
        cost_current: Current metric value
        cost_best: Best metric value so far
        mode: 'max' or 'min' for metric optimization
    
    Returns:
        True if current model is best
    
    Example:
        >>> is_best_model(0.85, 0.80, mode='max')
        True
        >>> is_best_model(0.75, 0.80, mode='max')
        False
    """
    if mode == "max":
        return cost_current > cost_best
    else:  # mode == "min"
        return cost_current < cost_best


#########################################################################################
# Main Training Loop
#########################################################################################

def train(
    model: nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler._LRScheduler,
    sched_on_epoch: bool,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    early_stopping: pytorch_utils.EarlyStopping,
    conf: DictConfig,
    fabric: Fabric,
    start_epoch: int = 0,
    cost_best: float = float('inf')
) -> None:
    """
    Main training loop across epochs.
    
    For each epoch:
    1. Train on training set
    2. Validate on validation set with retrieval metrics
    3. Update learning rate scheduler
    4. Check early stopping
    5. Save checkpoints (last, best, periodic)
    
    Args:
        model: Model instance
        optim: Optimizer
        sched: Learning rate scheduler
        sched_on_epoch: Whether scheduler steps per epoch (vs per step)
        dl_train: Training dataloader
        dl_valid: Validation dataloader
        early_stopping: Early stopping handler
        conf: Configuration object
        fabric: Fabric instance
        start_epoch: Starting epoch number (for resuming)
        cost_best: Best cost seen so far (for resuming)
    
    Example:
        >>> train(model, optim, sched, True, dl_train, dl_valid,
        ...       early_stopping, conf, fabric, start_epoch=0)
    """
    import os
    
    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )
    
    timer = print_utils.Timer()
    
    fn_ckpt_last, fn_ckpt_best, fn_ckpt_epoch = get_checkpoint_paths(conf.path.logs)
    
    lr = sched.get_last_lr()[0]
    stop = None
    
    myprint("Training...")
    
    for epoch in range(start_epoch, conf.training.numepochs):
        desc = f"{epoch+1:{len(str(conf.training.numepochs))}d}/{conf.training.numepochs}"
        fabric.log("hpar/epoch", epoch + 1, step=epoch + 1)
        
        # Train
        logdict_train = train_epoch(
            model, dl_train, optim, sched, sched_on_epoch,
            conf, fabric, desc="Train " + desc
        )
        logdict_train.sync_and_mean(fabric)
        fabric.log_dict(logdict_train.get(prefix="train/"), step=epoch + 1)
        
        # Validate
        logdict_valid = validate_epoch(
            model, dl_valid, optim, sched, sched_on_epoch,
            conf, fabric, desc="Valid " + desc
        )
        logdict_valid.sync_and_mean(fabric)
        fabric.log_dict(logdict_valid.get(prefix="valid/"), step=epoch + 1)
        
        # Report
        tmp = logdict_valid.get(keys=["l_main", "m_MAP", "m_ARP", "m_COMP"])
        tmp["l_main_t"] = logdict_train.get("l_main")
        report = print_utils.report(tmp, desc=f"[{timer.time()}] Epoch {desc}")
        
        # Check for NaN/inf
        for aux in tmp.values():
            if math.isnan(aux) or math.isinf(aux):
                stop = "NaN or inf reached!"
                break
        
        # Get current cost
        cost_current = logdict_valid.get(conf.training.monitor.quantity)
        
        # Early stopping
        early_stopping_config = conf.get('early_stopping', {})
        early_stopping_metric = early_stopping_config.get(
            'metric', conf.training.monitor.quantity
        )
        early_stopping_score = logdict_valid.get(early_stopping_metric)
        
        if early_stopping(early_stopping_score):
            stop = (
                f"Early stopping triggered after {early_stopping.counter} "
                f"epochs without improvement in {early_stopping_metric}"
            )
        
        if early_stopping.enabled:
            report += f"  (ES: {early_stopping.counter}/{early_stopping.patience})"
        
        # Learning rate schedule
        fabric.log("hpar/lr", lr, step=epoch + 1)
        if sched_on_epoch:
            if conf.training.optim.sched.startswith("plateau"):
                sched.step(cost_current)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sched.step()
            
            new_lr = sched.get_last_lr()[0]
            if new_lr != lr:
                if conf.training.optim.sched.startswith("plateau"):
                    report += f"  (lr={new_lr:.1e})"
                lr = new_lr
        
        if "min_lr" in conf.training.optim and lr < conf.training.optim.min_lr:
            stop = "Min lr reached."
        
        # Periodic checkpoint
        if should_save_checkpoint(epoch, conf.training.save_freq):
            fn = fn_ckpt_epoch.replace("$epoch$", f"epoch{epoch + 1}")
            state = pytorch_utils.get_state(
                model, optim, sched, conf, epoch + 1, lr, cost_best
            )
            fabric.save(fn, state)
        
        # Save best
        if is_best_model(cost_current, cost_best, conf.training.monitor.mode):
            cost_best = cost_current
            state = pytorch_utils.get_state(
                model, optim, sched, conf, epoch + 1, lr, cost_best
            )
            fabric.save(fn_ckpt_best, state)
            report += "  *"
        
        # Save last
        state = pytorch_utils.get_state(
            model, optim, sched, conf, epoch + 1, lr, cost_best
        )
        fabric.save(fn_ckpt_last, state)
        
        # Print and check stop
        myprint(report)
        if stop is not None:
            myprint(stop + " Stop.")
            break
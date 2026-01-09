"""
Inference utilities for distributed evaluation of version identification models.

Provides functions for:
- Configuration and Fabric setup
- Model and data loading
- Checkpointing system for resumable evaluation
- Memory management and logging
- Safe distributed gathering with timeout protection
- Embedding extraction with chunking support
- Chunk-based evaluation for overlapping embeddings
- Result gathering and reporting

Mirrors the structure of training_utils.py for consistency.
"""

import os
import sys
import time
import signal
import pickle
import gc
import math
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from omegaconf import OmegaConf, DictConfig
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from tqdm import tqdm

from lib import dataset
from lib.evaluation import eval
from utils import pytorch_utils


#########################################################################################
# Type Definitions
#########################################################################################

TensorDict = Dict[str, torch.Tensor]
ChunkType = Dict[str, Union[int, torch.Tensor]]
CheckpointData = Dict[str, Any]


#########################################################################################
# Configuration Setup
#########################################################################################

def setup_configuration() -> Any:
    """
    Setup evaluation configuration from command line arguments.
    
    Returns:
        Configuration object with all parameters
    
    Example:
        >>> args = setup_configuration()
        >>> print(args.ngpus, args.partition)
    """
    args = OmegaConf.from_cli()
    
    # Default values
    defaults = {
        'ngpus': 1,
        'nnodes': 1,
        'precision': 'bf16-mixed',
        'partition': 'test',
        'maxlen': 600,
        'qshop': 5,
        'cshop': 5,
        'use_overlapping_chunks': False,
        'overlap_percentage': 0.9,
        'chunk_size': 1500,
        'topk_distance': 1,
        'limit_num': None,
        'checkpoint_dir': 'eval_checkpoints',
        'resume_from_checkpoint': True,
        'disable_checkpointing': False,
        'disable_memory_logging': False
    }
    
    for k, v in defaults.items():
        setattr(args, k, getattr(args, k, v))
    
    return args


def print_configuration(args: Any, fabric: Fabric) -> None:
    """
    Print evaluation configuration summary (only on rank 0).
    
    Args:
        args: Configuration object
        fabric: Fabric instance for rank checking
    """
    if not fabric.is_global_zero:
        return
    
    print("=" * 70)
    print("EVALUATION CONFIGURATION")
    print("=" * 70)
    print(f"Mode: {'Overlapping chunks' if args.use_overlapping_chunks else 'Standard'}")
    print(f"GPUs: {args.ngpus} × {args.nnodes} nodes")
    print(f"Precision: {args.precision}")
    print(f"Partition: {args.partition}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Checkpointing: {'Enabled' if not args.disable_checkpointing else 'Disabled'}")
    print(f"Memory logging: {'Enabled' if not args.disable_memory_logging else 'Disabled'}")
    if args.use_overlapping_chunks:
        print(f"Chunk size: {args.chunk_size}")
        print(f"Overlap: {args.overlap_percentage * 100:.0f}%")
        print(f"Top-k distance: {args.topk_distance}")
    if args.limit_num:
        print(f"Limit: {args.limit_num} samples")
    print("=" * 70)


#########################################################################################
# Fabric Setup
#########################################################################################

def setup_pytorch(args: Any) -> None:
    """
    Configure PyTorch backend settings.
    
    Args:
        args: Configuration object
    """
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    torch.set_float32_matmul_precision("medium")


def setup_fabric(args: Any) -> Fabric:
    """
    Initialize Lightning Fabric for distributed evaluation.
    
    Args:
        args: Configuration object
    
    Returns:
        Initialized Fabric instance
    
    Example:
        >>> fabric = setup_fabric(args)
        >>> print(f"World size: {fabric.world_size}")
    """
    fabric = Fabric(
        accelerator="cuda",
        devices=args.ngpus,
        num_nodes=args.nnodes,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision=args.precision
    )
    fabric.launch()
    fabric.seed_everything(44 + fabric.global_rank, workers=True)
    
    return fabric


#########################################################################################
# Checkpointing System
#########################################################################################

class EvaluationCheckpoint:
    """
    Comprehensive checkpointing system for distributed evaluation.
    
    Enables resumable evaluation by saving intermediate results during:
    - Embedding extraction (per batch)
    - Evaluation computation (per query)
    
    Attributes:
        checkpoint_dir: Directory for saving checkpoints
        fabric: Fabric instance for distributed training
        enabled: Whether checkpointing is enabled
    
    Example:
        >>> checkpoint_mgr = EvaluationCheckpoint('checkpoints', fabric)
        >>> checkpoint_mgr.save_extraction_checkpoint(all_data, batch_idx, args)
        >>> data = checkpoint_mgr.load_extraction_checkpoint()
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "eval_checkpoints",
        fabric: Optional[Fabric] = None,
        enabled: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint files
            fabric: Fabric instance (required if enabled=True)
            enabled: Enable/disable checkpointing
        """
        self.checkpoint_dir = checkpoint_dir
        self.fabric = fabric
        self.enabled = enabled
        
        if self.enabled:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_extraction_checkpoint(
        self,
        all_data: Dict[str, List],
        batch_idx: int,
        args: Any
    ) -> None:
        """
        Save embedding extraction progress.
        
        Args:
            all_data: Dictionary with accumulated data
            batch_idx: Current batch index
            args: Arguments/configuration object
        """
        if not self.enabled:
            return
        
        checkpoint_data = {
            'all_data': all_data,
            'batch_idx': batch_idx,
            'rank': self.fabric.global_rank,
            'world_size': self.fabric.world_size,
            'args': args,
            'timestamp': time.time()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"extraction_checkpoint_rank_{self.fabric.global_rank}_batch_{batch_idx}.pkl"
        )
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"[GPU {self.fabric.global_rank}] Saved extraction checkpoint: batch {batch_idx}")
        except Exception as e:
            print(f"[GPU {self.fabric.global_rank}] Failed to save extraction checkpoint: {e}")
    
    def load_extraction_checkpoint(self) -> Optional[CheckpointData]:
        """
        Load the latest extraction checkpoint for current rank.
        
        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
        """
        if not self.enabled:
            return None
        
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(f"extraction_checkpoint_rank_{self.fabric.global_rank}")
        ]
        
        if not checkpoint_files:
            return None
        
        # Get latest checkpoint by batch index
        latest_file = max(
            checkpoint_files,
            key=lambda x: int(x.split('_batch_')[1].split('.pkl')[0])
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"[GPU {self.fabric.global_rank}] Loaded extraction checkpoint: {latest_file}")
            return checkpoint_data
        except Exception as e:
            print(f"[GPU {self.fabric.global_rank}] Failed to load extraction checkpoint: {e}")
            return None
    
    def save_evaluation_checkpoint(
        self,
        query_idx: int,
        partial_results: List[Tuple[torch.Tensor, ...]],
        local_queries: List[Tuple[int, Dict]],
        args: Any
    ) -> None:
        """
        Save evaluation progress.
        
        Args:
            query_idx: Current query index
            partial_results: List of partial evaluation results
            local_queries: List of query data for this rank
            args: Arguments/configuration object
        """
        if not self.enabled:
            return
        
        checkpoint_data = {
            'query_idx': query_idx,
            'partial_results': partial_results,
            'local_queries': local_queries,
            'rank': self.fabric.global_rank,
            'world_size': self.fabric.world_size,
            'args': args,
            'timestamp': time.time()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"evaluation_checkpoint_rank_{self.fabric.global_rank}_query_{query_idx}.pkl"
        )
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"[GPU {self.fabric.global_rank}] Saved evaluation checkpoint: query {query_idx}")
        except Exception as e:
            print(f"[GPU {self.fabric.global_rank}] Failed to save evaluation checkpoint: {e}")
    
    def load_evaluation_checkpoint(self) -> Optional[CheckpointData]:
        """
        Load the latest evaluation checkpoint for current rank.
        
        Returns:
            Checkpoint data dictionary or None if no checkpoint exists
        """
        if not self.enabled:
            return None
        
        checkpoint_files = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(f"evaluation_checkpoint_rank_{self.fabric.global_rank}")
        ]
        
        if not checkpoint_files:
            return None
        
        # Get latest checkpoint by query index
        latest_file = max(
            checkpoint_files,
            key=lambda x: int(x.split('_query_')[1].split('.pkl')[0])
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_file)
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            print(f"[GPU {self.fabric.global_rank}] Loaded evaluation checkpoint: {latest_file}")
            return checkpoint_data
        except Exception as e:
            print(f"[GPU {self.fabric.global_rank}] Failed to load evaluation checkpoint: {e}")
            return None


#########################################################################################
# Memory Management
#########################################################################################

def log_memory_usage(
    fabric: Fabric,
    context: str = "",
    verbose: bool = True
) -> None:
    """
    Log current GPU memory usage.
    
    Args:
        fabric: Fabric instance
        context: Context description for log message
        verbose: Whether to actually log (for conditional logging)
    
    Example:
        >>> log_memory_usage(fabric, "After model loading")
        [GPU 0] After model loading Memory - Allocated: 2.34GB, Reserved: 3.12GB, Max: 4.56GB
    """
    if not verbose or not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    
    print(
        f"[GPU {fabric.global_rank}] {context} Memory - "
        f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, "
        f"Max: {max_allocated:.2f}GB"
    )


#########################################################################################
# Safe Distributed Gathering
#########################################################################################

def safe_all_gather_with_timeout(
    fabric: Fabric,
    tensor: torch.Tensor,
    timeout_seconds: int = 300
) -> Optional[torch.Tensor]:
    """
    All-gather with timeout protection to prevent hanging.
    
    Uses signal.alarm to implement timeout. Returns None if operation
    times out or fails.
    
    Args:
        fabric: Fabric instance
        tensor: Tensor to gather across all ranks
        timeout_seconds: Timeout in seconds (default: 5 minutes)
    
    Returns:
        Gathered tensor of shape (world_size, *tensor.shape) or None if failed
    
    Example:
        >>> gathered = safe_all_gather_with_timeout(fabric, local_tensor, timeout_seconds=60)
        >>> if gathered is not None:
        ...     all_data = torch.cat(gathered.unbind(), dim=0)
    """
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("All-gather operation timed out")
    
    if tensor.numel() == 0:
        return torch.stack([torch.tensor([]) for _ in range(fabric.world_size)])
    
    # Set timeout alarm
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = fabric.all_gather(tensor)
        signal.alarm(0)
        return result
    except TimeoutError:
        print(f"[GPU {fabric.global_rank}] All-gather timed out after {timeout_seconds}s")
        signal.alarm(0)
        return None
    except Exception as e:
        print(f"[GPU {fabric.global_rank}] All-gather failed: {e}")
        signal.alarm(0)
        return None
    finally:
        signal.signal(signal.SIGALRM, old_handler)


#########################################################################################
# Model and Data Loading
#########################################################################################

def load_model_and_config(
    args: Any,
    fabric: Fabric,
    verbose_memory: bool = True
) -> Tuple[nn.Module, DictConfig]:
    """
    Load model and configuration from checkpoint.
    
    Args:
        args: Configuration with checkpoint path
        fabric: Fabric instance
        verbose_memory: Whether to log memory usage
    
    Returns:
        Tuple of (model, configuration)
    
    Example:
        >>> model, conf = load_model_and_config(args, fabric)
        >>> print(conf.model.name)
    """
    print("Loading model and configuration...")
    log_memory_usage(fabric, "Before model loading", verbose=verbose_memory)
    
    # Load configuration
    log_path = os.path.dirname(args.checkpoint)
    conf = OmegaConf.load(os.path.join(log_path, "configuration.yaml"))
    conf.data.chunk_size = args.chunk_size
    
    # Initialize model
    module = importlib.import_module(f"lib.models.{conf.model.name}")
    
    with fabric.init_module():
        model = module.Model(
            conf.model,
            use_avg_pooling=conf.data.use_avg_pooling,
            embedding_type=conf.data.embedding_type,
            sr=conf.data.samplerate
        )
    
    model = fabric.setup(model)
    model.mark_forward_method('embed')
    
    # Load checkpoint
    state = pytorch_utils.get_state(model, None, None, conf, None, None, None)
    fabric.load(args.checkpoint, state)
    model = pytorch_utils.set_state(state)[0].eval()
    
    log_memory_usage(fabric, "After model loading", verbose=verbose_memory)
    
    return model, conf


def setup_dataloader(
    conf: DictConfig,
    args: Any,
    fabric: Fabric,
    verbose_memory: bool = True
) -> DataLoader:
    """
    Setup dataset and dataloader for evaluation.
    
    Args:
        conf: Model configuration
        args: Evaluation arguments
        fabric: Fabric instance
        verbose_memory: Whether to log memory usage
    
    Returns:
        Configured dataloader
    
    Example:
        >>> dloader = setup_dataloader(conf, args, fabric)
        >>> print(f"Dataset size: {len(dloader.dataset)}")
    """
    print("Setting up dataset...")
    
    # Create dataset
    dset = dataset.EmbeddingDataset(
        conf,
        split=args.partition,
        augment=False,
        embedding_type=conf.data.embedding_type,
        embedding_format=conf.data.embedding_format,
        verbose=fabric.is_global_zero,
        return_paths=True
    )
    
    # Create collate function
    collate_fn = dataset.create_collate_fn(
        conf,
        deterministic=not args.use_overlapping_chunks,
        use_overlapping_chunks=args.use_overlapping_chunks,
        overlap_percentage=args.overlap_percentage
    )
    
    # Create dataloader (batch_size=1 recommended for evaluation)
    batch_size = 1
    dloader = fabric.setup_dataloaders(
        DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            persistent_workers=False,
            pin_memory=False,
            num_workers=0
        )
    )
    
    print(f"Dataset: {len(dset)} samples, batch size: {batch_size}")
    log_memory_usage(fabric, "After dataset setup", verbose=verbose_memory)
    
    return dloader


#########################################################################################
# Embedding Extraction
#########################################################################################

@torch.inference_mode()
def extract_embeddings_with_checkpointing(
    model: nn.Module,
    dloader: DataLoader,
    args: Any,
    fabric: Fabric,
    checkpoint_manager: EvaluationCheckpoint,
    verbose_memory: bool = True,
    conf: Optional[DictConfig] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[ChunkType]]]:
    """
    Extract embeddings with checkpointing and memory management.
    
    Supports two modes:
    1. Standard: Extract single embedding per version
    2. Overlapping chunks: Extract multiple overlapping chunk embeddings per version
    
    Args:
        model: Model for embedding extraction
        dloader: DataLoader with batch_size=1 recommended
        args: Arguments with use_overlapping_chunks, limit_num, ngpus
        fabric: Fabric instance for distributed training
        checkpoint_manager: Checkpoint manager for resumable extraction
        verbose_memory: Whether to log memory usage
        conf: Configuration object (optional, for use_avg_pooling)
    
    Returns:
        Tuple of:
        - class_ids: Clique IDs (empty if overlapping chunks)
        - version_ids: Version IDs (empty if overlapping chunks)
        - embeddings: Extracted embeddings (empty if overlapping chunks)
        - masks: Attention masks (empty if overlapping chunks)
        - chunks: List of chunk dictionaries (None if standard mode)
    
    Example:
        >>> q_c, q_i, q_z, q_m, chunks = extract_embeddings_with_checkpointing(
        ...     model, dloader, args, fabric, checkpoint_mgr
        ... )
    """
    model.eval()
    
    # Determine if using average pooling
    use_avg = bool(
        getattr(args, "use_avg_pooling", False) or
        (conf is not None and getattr(conf.data, "use_avg_pooling", False))
    )
    
    # Load or initialize data accumulator
    checkpoint_data = checkpoint_manager.load_extraction_checkpoint() if checkpoint_manager.enabled else None
    
    if checkpoint_data:
        all_data = checkpoint_data['all_data']
        # Ensure all keys exist (for backward compatibility)
        for k in ['c', 'i', 'z', 'm', 'chunks']:
            all_data.setdefault(k, [])
        start_batch = checkpoint_data['batch_idx'] + 1
        print(f"[GPU {fabric.global_rank}] Resuming from batch {start_batch}")
    else:
        all_data = {'c': [], 'i': [], 'z': [], 'm': [], 'chunks': []}
        start_batch = 0
    
    pbar = tqdm(
        enumerate(dloader),
        desc=f"GPU {fabric.global_rank}: Extract embeddings",
        disable=not fabric.is_global_zero,
        leave=True,
        total=len(dloader)
    )
    
    for batch_idx, batch in pbar:
        if batch_idx < start_batch:
            continue
        try:
            log_memory_usage(fabric, f"Batch {batch_idx}", verbose=verbose_memory)
            
            if args.use_overlapping_chunks and not use_avg:
                clique_ids, version_ids, embeddings, masks, chunk_info = batch
                
                # Flatten batch and n_per_class dimensions
                if embeddings.dim() == 4:
                    b, n, t, d = embeddings.shape
                    embeddings = embeddings.reshape(b * n, t, d)
                    clique_ids = clique_ids.unsqueeze(1).expand(-1, n).reshape(-1)
                    version_ids = version_ids.unsqueeze(1).expand(-1, n).reshape(-1)
                    
                    # CRITICAL: Also expand masks to match embeddings batch!
                    if masks.dim() == 4:
                        masks = masks.reshape(b * n, t, masks.size(-1))
                    elif masks.dim() == 3:
                        masks = masks.reshape(b * n, t)
                    elif masks.dim() == 2:
                        # Expand: [B, T] → [B*N, T]
                        masks = masks.unsqueeze(1).expand(-1, n, -1).reshape(b * n, t)
                
                # Ensure masks are 2D
                if masks.dim() == 3:
                    masks = masks[:, :, 0] if masks.size(-1) > 1 else masks.squeeze(-1)
                
                embeddings = model.prepare(embeddings)
                chunk_embeddings, _ = model.embed(embeddings, masks)

                
                # Create chunk data
                song_ids = (clique_ids * 1000000 + version_ids).cpu()
                chunks = [
                    {
                        'clique_id': c.item(),
                        'version_id': v.item(),
                        'embedding': e.cpu(),
                        'mask': m.cpu(),
                        'song_id': s.item(),
                        'chunk_idx': info[2]
                    }
                    for c, v, e, m, s, info in zip(
                        clique_ids, version_ids, chunk_embeddings,
                        masks, song_ids, chunk_info
                    )
                ]
                all_data['chunks'].extend(chunks)
        
                
                unique_songs = len(set(c['song_id'] for c in all_data['chunks']))
                pbar.set_postfix({'chunks': len(all_data['chunks']), 'songs': unique_songs})
                
                if args.limit_num and unique_songs >= args.limit_num // args.ngpus:
                    break
            else:
                # Standard mode
                n_per_class = conf.data.n_per_class
                cc = torch.cat([batch[0]] * n_per_class)
                ii = torch.cat(batch[1::3])
                xx = torch.cat(batch[2::3])
                masks = torch.cat(batch[3::3])
                
                xx = model.prepare(xx)
                zz, _ = model.embed(xx, masks)
                
                # Take first sample from each chunk
                chunk_size = len(zz) // n_per_class
                all_data['c'].append(cc[:chunk_size].cpu())
                all_data['i'].append(ii[:chunk_size].cpu())
                all_data['z'].append(zz[:chunk_size].cpu())
                all_data['m'].append(masks[:chunk_size].cpu())
                
                total_embeddings = sum(len(z) for z in all_data['z'])
                pbar.set_postfix({'embeddings': total_embeddings})
                
                if args.limit_num and total_embeddings >= args.limit_num // args.ngpus:
                    break
            
            # Memory cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save checkpoint every 100 batches
            if checkpoint_manager.enabled and (batch_idx + 1) % 100 == 0:
                checkpoint_manager.save_extraction_checkpoint(all_data, batch_idx, args)
        
        except Exception as e:
            print(f"[GPU {fabric.global_rank}] Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final checkpoint
    if checkpoint_manager.enabled:
        checkpoint_manager.save_extraction_checkpoint(all_data, batch_idx, args)
    
    # Return appropriate format
    if args.use_overlapping_chunks and not use_avg:
        return None, None, None, None, all_data['chunks']
    else:
        return (
            torch.cat(all_data['c']) if all_data['c'] else torch.tensor([]),
            torch.cat(all_data['i']) if all_data['i'] else torch.tensor([]),
            torch.cat(all_data['z']) if all_data['z'] else torch.tensor([]),
            torch.cat(all_data['m']) if all_data['m'] else torch.tensor([]),
            None
        )


#########################################################################################
# Chunk Gathering
#########################################################################################

def gather_chunks_safely(
    fabric: Fabric,
    local_chunks: List[ChunkType],
    verbose_memory: bool = True
) -> List[ChunkType]:
    """
    Robust gathering of variable-length chunk lists across ranks.
    
    Strategy:
    1. Serialize local chunks with pickle (CPU tensors)
    2. All-gather sizes
    3. Pad byte buffers to max size
    4. All-gather padded buffers
    5. Deserialize per-rank payloads
    
    Safe when different ranks have different numbers of chunks.
    
    Args:
        fabric: Fabric instance
        local_chunks: List of chunk dictionaries on this rank
        verbose_memory: Whether to log memory usage
    
    Returns:
        Combined list of all chunks from all ranks
    
    Example:
        >>> all_chunks = gather_chunks_safely(fabric, local_chunks)
        >>> print(f"Total chunks: {len(all_chunks)}")
    """
    import pickle
    import numpy as np
    
    if verbose_memory:
        log_memory_usage(fabric, "Before chunk gathering", verbose=True)
    
    try:
        # Move tensors to CPU for safe serialization
        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return x
        
        def _cpu_copy_chunk(ch):
            return {k: _to_cpu(v) for k, v in ch.items()}
        
        safe_local = [_cpu_copy_chunk(ch) for ch in (local_chunks or [])]
        
        # Serialize to bytes
        local_bytes = pickle.dumps(safe_local, protocol=pickle.HIGHEST_PROTOCOL)
        local_size = len(local_bytes)
        
        size_tensor = torch.tensor([local_size], device=fabric.device, dtype=torch.long)
        
        # All-gather sizes
        all_sizes = safe_all_gather_with_timeout(fabric, size_tensor, timeout_seconds=120)
        if all_sizes is None:
            print(f"[GPU {fabric.global_rank}] Size gather failed; returning LOCAL chunks only")
            return local_chunks or []
        
        # Flatten sizes
        if all_sizes.dim() > 1:
            all_sizes = all_sizes.view(-1)
        max_size = int(all_sizes.max().item())
        world_size = fabric.world_size
        
        # Edge case: no chunks anywhere
        if max_size == 0 and int(all_sizes.sum().item()) == 0:
            print(f"[GPU {fabric.global_rank}] No chunks on any rank")
            return []
        
        # Build padded byte buffer
        if local_size > 0:
            local_buf_cpu = np.frombuffer(local_bytes, dtype=np.uint8)
            local_buf = torch.from_numpy(local_buf_cpu).to(fabric.device, non_blocking=True)
        else:
            local_buf = torch.empty(0, dtype=torch.uint8, device=fabric.device)
        
        if local_buf.numel() < max_size:
            pad = torch.zeros(max_size - local_buf.numel(), dtype=torch.uint8, device=fabric.device)
            local_buf = torch.cat([local_buf, pad], dim=0)
        elif local_buf.numel() > max_size:
            local_buf = local_buf[:max_size]
        
        # All-gather padded buffers
        all_padded = safe_all_gather_with_timeout(fabric, local_buf, timeout_seconds=600)
        if all_padded is None:
            print(f"[GPU {fabric.global_rank}] Byte-buffer gather failed; returning LOCAL chunks only")
            return local_chunks or []
        
        # Deserialize each rank's payload
        all_chunks = []
        for r in range(world_size):
            n = int(all_sizes[r].item())
            if n <= 0:
                continue
            
            r_bytes = all_padded[r][:n].detach().cpu().numpy().tobytes()
            try:
                r_chunks = pickle.loads(r_bytes)
                all_chunks.extend(r_chunks)
            except Exception as e:
                print(f"[GPU {fabric.global_rank}] Warning: failed to unpickle rank {r} payload: {e}")
        
        if verbose_memory:
            print(f"[GPU {fabric.global_rank}] Successfully gathered {len(all_chunks)} total chunks")
            log_memory_usage(fabric, "After chunk gathering", verbose=True)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        return all_chunks
    
    except Exception as e:
        print(f"[GPU {fabric.global_rank}] Chunk gathering failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"[GPU {fabric.global_rank}] Falling back to LOCAL chunks only")
        return local_chunks or []


#########################################################################################
# Chunk-Based Evaluation
#########################################################################################

def evaluate_overlapping_chunks_fast(
    fabric: Fabric,
    chunks: List[ChunkType],
    args: Any,
    checkpoint_manager: EvaluationCheckpoint,
    verbose_memory: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast evaluation for overlapping chunks.
    
    Strategy:
    1. Gather chunks once
    2. Group by song and stack embeddings
    3. Move ALL candidates to GPU and pre-normalize
    4. Query loop only does matrix multiplications
    
    Args:
        fabric: Fabric instance
        chunks: List of chunk dictionaries from this rank
        args: Arguments with topk_distance
        checkpoint_manager: Checkpoint manager
        verbose_memory: Whether to log memory usage
    
    Returns:
        Tuple of (aps, r1s, rpcs) - 1D tensors of retrieval metrics
    
    Example:
        >>> aps, r1s, rpcs = evaluate_overlapping_chunks_fast(
        ...     fabric, chunks, args, checkpoint_mgr
        ... )
    """
    if not chunks:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    print(f"[GPU {fabric.global_rank}] Starting FAST overlapping evaluation")
    log_memory_usage(fabric, "Before gather", verbose=verbose_memory)
    
    # Gather all chunks
    all_chunks = gather_chunks_safely(fabric, chunks, verbose_memory=verbose_memory)
    if not all_chunks:
        print(f"[GPU {fabric.global_rank}] No chunks available for evaluation")
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # Group by song
    songs = {}
    for ch in all_chunks:
        sid = ch['song_id']
        s = songs.setdefault(sid, {
            'clique_id': ch['clique_id'],
            'version_id': ch['version_id'],
            'emb_list': []
        })
        s['emb_list'].append(ch['embedding'])
    
    for sid in songs:
        songs[sid]['emb'] = torch.stack(songs[sid]['emb_list'])  # (n_chunks, d)
        del songs[sid]['emb_list']
    
    # Move ALL candidates to GPU and pre-normalize
    print(f"[GPU {fabric.global_rank}] Moving {len(songs)} songs to GPU and pre-normalizing...")
    for sid in songs:
        songs[sid]['emb'] = torch.nn.functional.normalize(
            songs[sid]['emb'].to(fabric.device), dim=-1
        )
    
    # Candidate metadata
    song_list = list(songs.items())
    total_songs = len(song_list)
    cand_cliques = torch.tensor([d['clique_id'] for _, d in song_list], device=fabric.device)
    cand_versions = torch.tensor([d['version_id'] for _, d in song_list], device=fabric.device)
    
    # Shard queries across ranks
    per_gpu = total_songs // fabric.world_size
    start_idx = fabric.global_rank * per_gpu
    end_idx = total_songs if fabric.global_rank == fabric.world_size - 1 else start_idx + per_gpu
    local_queries = song_list[start_idx:end_idx]
    
    print(f"[GPU {fabric.global_rank}] Processing {len(local_queries)} queries vs {total_songs} candidates")
    
    # Progress bar
    pbar = tqdm(
        range(len(local_queries)),
        desc=f"GPU {fabric.global_rank}: Queries",
        disable=False if fabric.world_size == 1 else not fabric.is_global_zero,
        dynamic_ncols=True,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                   '[{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    topk = int(getattr(args, 'topk_distance', 1))
    partial = []
    recent_times = []
    
    for i in pbar:
        t0 = time.time()
        q_sid, q_data = local_queries[i]
        q_emb = songs[q_sid]['emb']  # Already normalized & on GPU
        
        # Compute distances to all candidates
        dists = torch.zeros(total_songs, device=fabric.device)
        
        for j, (c_sid, c_data) in enumerate(song_list):
            if q_sid == c_sid:
                dists[j] = 0.0
                continue
            
            sim = q_emb @ c_data['emb'].T  # (q_chunks, c_chunks)
            
            if topk == 1:
                dists[j] = 1.0 - sim.max()
            else:
                dist_flat = (1.0 - sim).flatten()
                k = min(topk, dist_flat.numel())
                dists[j] = torch.topk(dist_flat, k, largest=False)[0].mean() if k > 0 else dist_flat.mean()
        
        # Compute baseline metrics
        res = eval.compute_baseline(
            dists.unsqueeze(0),
            torch.tensor([q_data['clique_id']], device=fabric.device),
            torch.tensor([q_data['version_id']], device=fabric.device),
            cand_cliques, cand_versions
        )
        partial.append(res)
        
        # Update progress bar
        dt = time.time() - t0
        recent_times.append(dt)
        if len(recent_times) > 50:
            recent_times = recent_times[-50:]
        
        avg_t = sum(recent_times) / len(recent_times)
        cur_ap = partial[-1][0].item()
        mean_ap = sum(x[0].item() for x in partial) / len(partial)
        pbar.set_postfix({'AP': f'{mean_ap:.4f}', 'last': f'{cur_ap:.4f}', 's/q': f'{avg_t:.2f}'})
        
        # Periodic cleanup
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()
        
        if checkpoint_manager.enabled and (i + 1) % 100 == 0:
            checkpoint_manager.save_evaluation_checkpoint(i, partial, local_queries, args)
    
    pbar.close()
    
    # Return 1D tensors
    if partial:
        aps, r1s, rpcs = zip(*partial)
        aps = torch.stack(aps).reshape(-1).contiguous()
        r1s = torch.stack(r1s).reshape(-1).contiguous()
        rpcs = torch.stack(rpcs).reshape(-1).contiguous()
        
        if fabric.is_global_zero:
            print(f"[GPU {fabric.global_rank}] Local results: "
                  f"aps={tuple(aps.shape)}, r1s={tuple(r1s.shape)}, rpcs={tuple(rpcs.shape)}")
        
        return aps, r1s, rpcs
    
    return torch.tensor([]), torch.tensor([]), torch.tensor([])


#########################################################################################
# Standard Evaluation
#########################################################################################

def evaluate_standard_mode(
    model: nn.Module,
    q_c: torch.Tensor,
    q_i: torch.Tensor,
    q_z: torch.Tensor,
    q_m: torch.Tensor,
    fabric: Fabric,
    checkpoint_manager: EvaluationCheckpoint,
    args: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate model with standard single-embedding approach.
    
    Args:
        model: Model for evaluation
        q_c: Query clique IDs
        q_i: Query version IDs
        q_z: Query embeddings
        q_m: Query masks
        fabric: Fabric instance
        checkpoint_manager: Checkpoint manager
        args: Evaluation arguments
    
    Returns:
        Tuple of (aps, r1s, rpcs) tensors
    """
    print("Gathering candidate embeddings...")
    
    # Gather candidates
    cands = []
    for i, tensor in enumerate([q_c.to(fabric.device), q_i.to(fabric.device),
                                q_z.to(fabric.device), q_m.to(fabric.device)]):
        print(f"[GPU {fabric.global_rank}] Gathering tensor {i+1}/4")
        gathered = safe_all_gather_with_timeout(fabric, tensor, timeout_seconds=300)
        
        if gathered is None:
            print(f"[GPU {fabric.global_rank}] Failed to gather tensor {i}, using local only")
            cands.append(tensor)
        else:
            cands.append(torch.cat(gathered.unbind()))
    
    # Evaluate queries
    print("Computing retrieval metrics...")
    results = []
    
    # Load checkpoint if available
    eval_checkpoint = checkpoint_manager.load_evaluation_checkpoint() if checkpoint_manager.enabled else None
    start_idx = eval_checkpoint['query_idx'] + 1 if eval_checkpoint else 0
    if eval_checkpoint:
        results = eval_checkpoint['partial_results']
    
    pbar = tqdm(
        range(start_idx, len(q_z)),
        desc="Evaluating queries",
        disable=not fabric.is_global_zero
    )
    
    for n in pbar:
        try:
            result = eval.compute(
                model,
                q_c[n:n+1].to(fabric.device),
                q_i[n:n+1].to(fabric.device),
                q_z[n:n+1].to(fabric.device),
                *cands,
                batch_size_candidates=1024
            )
            results.append(result)
            
            # Memory cleanup
            if n % 10 == 0:
                torch.cuda.empty_cache()
            
            # Checkpoint periodically
            if checkpoint_manager.enabled and (n + 1) % 100 == 0:
                checkpoint_manager.save_evaluation_checkpoint(n, results, None, args)
        
        except Exception as e:
            print(f"[GPU {fabric.global_rank}] Error in query {n}: {e}")
            continue
    
    # Stack results
    aps, r1s, rpcs = map(torch.stack, zip(*results))
    
    return aps, r1s, rpcs


#########################################################################################
# Result Gathering
#########################################################################################

def gather_results_safely(
    fabric: Fabric,
    *results: torch.Tensor
) -> List[torch.Tensor]:
    """
    Gather 1D result tensors of possibly different lengths across ranks.
    
    Strategy:
    1. Flatten each local result to 1D float32 on device
    2. All-gather sizes
    3. Pad each local tensor to max size
    4. All-gather padded tensors
    5. Slice per-rank using sizes and concatenate
    
    Args:
        fabric: Fabric instance
        *results: Variable number of result tensors to gather
    
    Returns:
        List of gathered tensors (one per input)
    
    Example:
        >>> aps, r1s, rpcs = gather_results_safely(fabric, local_aps, local_r1s, local_rpcs)
    """
    print(f"[GPU {fabric.global_rank}] Gathering final results...")
    
    if not results:
        return [torch.tensor([]) for _ in range(3)]
    
    # Normalize each metric to 1D float32 on device
    local_tensors = []
    for res in results:
        if not isinstance(res, torch.Tensor):
            res = torch.tensor(res)
        res = res.detach()
        if res.dim() == 0:
            res = res.unsqueeze(0)
        elif res.dim() > 1:
            res = res.reshape(-1)
        res = res.to(device=fabric.device, dtype=torch.float32)
        local_tensors.append(res)
    
    # Gather sizes using first metric
    size0 = torch.tensor([local_tensors[0].numel()], device=fabric.device, dtype=torch.long)
    all_sizes = safe_all_gather_with_timeout(fabric, size0, timeout_seconds=60)
    
    if all_sizes is None:
        print(f"[GPU {fabric.global_rank}] Size gather failed, returning local results")
        return [t.cpu() for t in local_tensors]
    
    max_len = int(all_sizes.max().item())
    total_counts = int(all_sizes.sum().item())
    
    if max_len == 0 or total_counts == 0:
        print(f"[GPU {fabric.global_rank}] No results from any rank")
        return [torch.tensor([]) for _ in results]
    
    gathered_results = []
    for idx, res in enumerate(local_tensors):
        # Pad to max_len
        if res.numel() < max_len:
            pad = torch.zeros(max_len - res.numel(), dtype=res.dtype, device=res.device)
            res_padded = torch.cat([res, pad], dim=0)
        else:
            res_padded = res
        
        # All-gather padded tensors
        all_padded = safe_all_gather_with_timeout(fabric, res_padded, timeout_seconds=300)
        
        if all_padded is None:
            print(f"[GPU {fabric.global_rank}] Gather failed for result {idx}, returning local only")
            gathered_results.append(res.detach().cpu())
            continue
        
        # Trim per-rank by true size and concatenate
        pieces = []
        for r in range(fabric.world_size):
            n = int(all_sizes[r].item())
            if n > 0:
                pieces.append(all_padded[r][:n])
        
        gathered = torch.cat(pieces, dim=0).detach().cpu() if pieces else torch.tensor([])
        gathered_results.append(gathered)
    
    return gathered_results


#########################################################################################
# Results Reporting
#########################################################################################

def print_results(
    aps: torch.Tensor,
    r1s: torch.Tensor,
    rpcs: torch.Tensor,
    args: Any,
    fabric: Fabric
) -> None:
    """
    Print final evaluation results with statistics (only on rank 0).
    
    Args:
        aps: Average Precision scores
        r1s: Rank-1 scores
        rpcs: Average Rank Percentile scores
        args: Evaluation arguments
        fabric: Fabric instance for rank checking
    """
    if not fabric.is_global_zero:
        return
    
    if len(aps) == 0:
        print("No results to report")
        return
    
    stats = [
        (aps.mean(), aps.std()),
        (r1s.mean(), r1s.std()),
        (rpcs.mean(), rpcs.std())
    ]
    cis = [1.96 * std / math.sqrt(len(aps)) for _, std in stats]
    
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"MAP: {stats[0][0]:.4f} ± {cis[0]:.4f}")
    print(f"MR1: {stats[1][0]:.4f} ± {cis[1]:.4f}")
    print(f"ARP: {stats[2][0]:.4f} ± {cis[2]:.4f}")
    print(f"")
    print(f"Mode: {'Overlapping chunks' if args.use_overlapping_chunks else 'Standard'}")
    print(f"Queries: {len(aps)}")
    print(f"Partition: {args.partition}")
    print("=" * 70)


def save_results(
    aps: torch.Tensor,
    r1s: torch.Tensor,
    rpcs: torch.Tensor,
    args: Any,
    fabric: Fabric
) -> None:
    """
    Save evaluation results to disk (only on rank 0).
    
    Args:
        aps: Average Precision scores
        r1s: Rank-1 scores
        rpcs: Average Rank Percentile scores
        args: Evaluation arguments
        fabric: Fabric instance for rank checking
    """
    if not fabric.is_global_zero:
        return
    
    stats = [
        (aps.mean(), aps.std()),
        (r1s.mean(), r1s.std()),
        (rpcs.mean(), rpcs.std())
    ]
    
    results_path = os.path.join(args.checkpoint_dir, "final_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump({
            'aps': aps.cpu(),
            'r1s': r1s.cpu(),
            'rpcs': rpcs.cpu(),
            'stats': stats,
            'args': args
        }, f)
    
    print(f"✓ Results saved to {results_path}")
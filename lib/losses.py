"""
Loss functions for contrastive learning in version identification.

This module provides three main loss functions:
- NTXentLoss: Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
- TripletLoss: Triplet margin loss for metric learning
- CLEWSLoss: CLEWS contrastive loss with alignment and uniformity

All losses follow a unified interface:
    loss, logdict = loss_fn(z_label, z_idx, z, extra)

where:
    - z_label: Clique labels (same song/cover group)
    - z_idx: Version indices (individual recordings)
    - z: Embedding vectors
    - extra: Optional auxiliary information
"""

from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from . import tensor_ops as tops


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    Also known as SimCLR loss. Computes contrastive loss using cosine similarity
    with temperature scaling. Treats samples with the same label but different
    indices as positives, and samples with different labels as negatives.
    
    The loss encourages embeddings of the same song (same label, different versions)
    to be similar while pushing different songs apart.
    
    Mathematical formulation:
        For anchor i, let P(i) be the set of positives and N(i) be negatives.
        
        sim(i,j) = z_i · z_j / (||z_i|| ||z_j||)
        
        loss = -log( Σ_{j∈P(i)} exp(sim(i,j)/τ) / Σ_{k≠i} exp(sim(i,k)/τ) )
    
    Attributes:
        tau: Temperature parameter for scaling (lower = harder negatives)
    
    Example:
        >>> loss_fn = NTXentLoss(temperature=0.1)
        >>> z_label = torch.tensor([0, 0, 1, 1])  # Two songs, 2 versions each
        >>> z_idx = torch.tensor([0, 1, 0, 1])     # Version indices
        >>> z = torch.randn(4, 128)                # Embeddings
        >>> loss, logdict = loss_fn(z_label, z_idx, z)
        >>> loss.backward()
    """
    
    def __init__(self, temperature: float = 0.1) -> None:
        """
        Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter τ for scaling similarities.
                Lower values (0.05-0.1) make the loss focus on harder negatives.
                Higher values (0.5-1.0) soften the distribution.
                Default: 0.1 (typical for contrastive learning)
        """
        super().__init__()
        self.tau = temperature
        
    def forward(
        self,
        z_label: torch.Tensor,
        z_idx: torch.Tensor,
        z: torch.Tensor,
        extra: Optional[Any] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute NT-Xent contrastive loss.
        
        Args:
            z_label: Clique labels of shape (batch_size,).
                Samples with the same label belong to the same cover group.
            z_idx: Version indices of shape (batch_size,).
                Different recordings of the same song have different indices.
            z: Embeddings of shape (batch_size, embedding_dim).
                L2-normalized embeddings are recommended for best results.
            extra: Optional auxiliary information (unused, for interface compatibility)
        
        Returns:
            Tuple containing:
                - loss: Scalar loss tensor
                - logdict: Dictionary with loss components and diagnostics:
                    - l_main: Main loss value
                    - v_zmax: Maximum absolute value in embeddings
                    - v_zmean: Mean of embeddings
                    - v_zstd: Standard deviation of embeddings
        
        Note:
            If all samples have the same label (no negatives), a small fraction
            is reassigned to label -1 for numerical stability.
        
        Example:
            >>> loss_fn = NTXentLoss(temperature=0.07)
            >>> # Batch with 2 songs, 2 versions each
            >>> z_label = torch.tensor([0, 0, 1, 1])
            >>> z_idx = torch.tensor([0, 1, 0, 1])
            >>> z = F.normalize(torch.randn(4, 256), dim=1)
            >>> loss, logdict = loss_fn(z_label, z_idx, z)
            >>> print(f"Loss: {loss.item():.4f}")
            >>> print(f"Embedding stats: max={logdict['v_zmax']:.4f}")
        """
        assert len(z_label) == len(z_idx) and len(z_label) == len(z)
        
        # If no negatives, add label noise for loss stability
        # (we assume positives exist due to batch construction)
        if len(z_label.unique()) == 1:
            z_label[: max(2, int(len(z_label) * 0.01))] = -1

        # Create positive/negative masks
        same_label = z_label.view(-1, 1) == z_label.view(1, -1)
        same_idx = z_idx.view(-1, 1) == z_idx.view(1, -1)
        positives = same_label & (~same_idx)

        # Compute pairwise cosine similarities
        sim = tops.pairwise_distance_matrix(z, z, mode="cossim")

        # Temperature-scaled logits
        logits = sim / self.tau
        pos_mask = positives.float()

        # Mask out diagonal (self-similarity) to avoid numerical issues
        mask_diag = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask_diag, -1e9)

        # Numerical stability: subtract max before exp
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)

        # NT-Xent loss: -log(sum(pos_exp) / sum(all_exp))
        pos_exp_sum = (exp_logits * pos_mask).sum(dim=1)
        all_exp_sum = exp_logits.sum(dim=1)

        # Add small epsilon for numerical stability
        eps = 1e-8
        loss = -torch.log(pos_exp_sum / (all_exp_sum + eps) + eps).mean()
        
        logdict = {
            "l_main": loss,
            "v_zmax": z.abs().max(),
            "v_zmean": z.mean(),
            "v_zstd": z.std(),
        }
        return loss, logdict



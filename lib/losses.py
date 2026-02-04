"""
Loss functions for contrastive learning in version identification.

This module provides three main loss functions:
- NTXentLoss: Normalized Temperature-scaled Cross Entropy Loss (SimCLR)
- TripletLoss: Triplet margin loss for metric learning [WIP]
- CLEWSLoss: CLEWS contrastive loss with alignment and uniformity [WIP]

All losses follow a unified interface:
    loss, logdict = loss_fn(z_label, z_idx, z, extra)

where:
    - z_label: Clique labels (same song/cover group)
    - z_idx: Version indices (individual recordings)
    - z: Embedding vectors
    - extra: Optional auxiliary information
"""

from typing import Any, Dict, Optional, Tuple

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
        extra: Optional[Any] = None,
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


class TripletLoss(nn.Module):
    """
    Triplet margin loss for metric learning.

    Optimizes embeddings so that anchors are closer to positives than to
    negatives by at least a margin. For each anchor, finds one positive
    (same label, different index) and one negative (different label).

    Mathematical formulation:
        loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    where d is typically L2 distance or cosine distance.

    Attributes:
        triplet_loss: PyTorch's TripletMarginLoss module

    Example:
        >>> loss_fn = TripletLoss(margin=0.2, p=2)
        >>> z_label = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> z_idx = torch.tensor([0, 1, 0, 1, 0, 1])
        >>> z = torch.randn(6, 128)
        >>> loss, logdict = loss_fn(z_label, z_idx, z)
        >>> print(f"Triplet loss: {loss.item():.4f}")
    """

    def __init__(
        self,
        margin: float = 0.2,
        p: int = 2,
        eps: float = 1e-6,
        swap: bool = False,
        reduction: str = "mean",
    ) -> None:
        """
        Initialize triplet loss.

        Args:
            margin: Margin value for the triplet loss. The model tries to ensure
                d(anchor, positive) + margin < d(anchor, negative).
                Typical values: 0.1-0.5
            p: Norm degree for distance calculation (1 for L1, 2 for L2).
                Default: 2 (Euclidean distance)
            eps: Small epsilon for numerical stability in distance calculation
            swap: If True, uses the hardest negative in the triplet.
                Can improve learning but may be less stable.
            reduction: Specifies the reduction to apply to the output:
                'mean' | 'sum' | 'none'. Default: 'mean'
        """
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(
            margin=margin, p=p, eps=eps, swap=swap, reduction=reduction
        )

    def forward(
        self,
        z_label: torch.Tensor,
        z_idx: torch.Tensor,
        z: torch.Tensor,
        extra: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute triplet loss.

        Args:
            z_label: Clique labels of shape (batch_size,)
            z_idx: Version indices of shape (batch_size,)
            z: Embeddings of shape (batch_size, embedding_dim)
            extra: Optional auxiliary information (unused)

        Returns:
            Tuple containing:
                - loss: Scalar loss tensor (0.0 if no valid triplets found)
                - logdict: Dictionary with diagnostics:
                    - l_main: Main loss value
                    - v_zmax: Maximum absolute value in embeddings
                    - v_zmean: Mean of embeddings
                    - v_zstd: Standard deviation of embeddings
                    - n_triplets: Number of triplets formed (if applicable)

        Note:
            For each anchor, finds one positive (same label, different index)
            and one negative (different label). If no valid triplet can be
            formed for an anchor, it is skipped.

            Future work: Consider implementing hard negative mining (select the
            closest negative to the anchor) or semi-hard mining (negatives that
            are farther than positive but within margin) for improved training.
            Random sampling of positives/negatives could also help.

        Example:
            >>> loss_fn = TripletLoss(margin=0.3)
            >>> z_label = torch.tensor([0, 0, 0, 1, 1, 2])
            >>> z_idx = torch.tensor([0, 1, 2, 0, 1, 0])
            >>> z = torch.randn(6, 256)
            >>> loss, logdict = loss_fn(z_label, z_idx, z)
        """
        assert len(z_label) == len(z_idx) and len(z_label) == len(z)

        # If no negatives, add label noise for loss stability
        if len(z_label.unique()) == 1:
            z_label[: max(2, int(len(z_label) * 0.01))] = -1

        # Create simple triplets: for each sample, find one positive and one negative
        anchors, positives, negatives = self._create_triplets(z_label, z_idx)

        if len(anchors) == 0:
            # No valid triplets found, return zero loss
            loss = torch.tensor(0.0, device=z.device, requires_grad=True)
            logdict = {
                "l_main": loss,
                "v_zmax": z.abs().max(),
                "v_zmean": z.mean(),
                "v_zstd": z.std(),
                "n_triplets": 0,
            }
            return loss, logdict

        # Extract embeddings for triplets
        anchor_embeddings = z[anchors]
        positive_embeddings = z[positives]
        negative_embeddings = z[negatives]

        # Compute triplet loss using PyTorch's implementation
        loss = self.triplet_loss(
            anchor_embeddings, positive_embeddings, negative_embeddings
        )

        logdict = {
            "l_main": loss,
            "v_zmax": z.abs().max(),
            "v_zmean": z.mean(),
            "v_zstd": z.std(),
        }

        return loss, logdict

    def _create_triplets(
        self, z_label: torch.Tensor, z_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create triplets by finding one positive and one negative for each anchor.

        For each sample (anchor):
        1. Find positives: samples with same label but different index
        2. Find negatives: samples with different label
        3. Select first available positive and negative

        Args:
            z_label: Clique labels of shape (batch_size,)
            z_idx: Version indices of shape (batch_size,)

        Returns:
            Tuple containing:
                - anchors: Indices of anchor samples
                - positives: Indices of positive samples (one per anchor)
                - negatives: Indices of negative samples (one per anchor)

            All tensors have shape (num_valid_triplets,)

        Example:
            >>> # Internal method, typically not called directly
            >>> loss_fn = TripletLoss()
            >>> z_label = torch.tensor([0, 0, 1])
            >>> z_idx = torch.tensor([0, 1, 0])
            >>> anchors, pos, neg = loss_fn._create_triplets(z_label, z_idx)
            >>> # anchors[i] has positive at pos[i] and negative at neg[i]
        """
        device = z_label.device
        anchors = []
        positives = []
        negatives = []

        batch_size = len(z_label)

        for i in range(batch_size):
            # Find positives: same label, different idx
            pos_mask = (z_label == z_label[i]) & (z_idx != z_idx[i])
            valid_pos = torch.where(pos_mask)[0]

            # Find negatives: different label
            neg_mask = z_label != z_label[i]
            valid_neg = torch.where(neg_mask)[0]

            if len(valid_pos) == 0 or len(valid_neg) == 0:
                continue

            # Take the first available positive and negative
            # (could be randomized: valid_pos[torch.randint(len(valid_pos), (1,))])
            pos_idx = valid_pos[0]
            neg_idx = valid_neg[0]

            anchors.append(i)
            positives.append(pos_idx.item())
            negatives.append(neg_idx.item())

        return (
            torch.tensor(anchors, device=device),
            torch.tensor(positives, device=device),
            torch.tensor(negatives, device=device),
        )


class CLEWSLoss(nn.Module):
    """
    CLEWS (Contrastive Learning with Exponential Weighting and Scaling) Loss.

    Combines alignment (pulling positives together) and uniformity (spreading
    embeddings uniformly on the unit hypersphere) for cosine-based embeddings.

    Unlike InfoNCE/NT-Xent which uses softmax, CLEWS directly optimizes:
    - Alignment: Mean distance to positives (pull together)
    - Uniformity: Penalizes clustering via exponentially-weighted negative distances

    Mathematical formulation:
        L_align = E_{(i,j)∈P} [d(z_i, z_j)]
        L_uniform = E_i [log(1 + E_{k∈N(i)} [exp(b - γ·d(z_i, z_k))])]
        L = L_align + λ·L_uniform

    where d is cosine distance (1 - cosine_similarity) ∈ [0, 2].

    Attributes:
        gamma: Exponential scaling factor for uniformity (controls hardness)
        b: Bias term in uniformity exponential
        eps: Small epsilon for numerical stability
        epsilon: Alternative epsilon (legacy, for compatibility)
        uniformity_weight: Weight λ for uniformity term
        warmup_steps: Number of steps to linearly ramp up uniformity weight

    Example:
        >>> loss_fn = CLEWSLoss(gamma=8.0, uniformity_weight=0.5, warmup_steps=1000)
        >>> z_label = torch.tensor([0, 0, 1, 1])
        >>> z_idx = torch.tensor([0, 1, 0, 1])
        >>> z = F.normalize(torch.randn(4, 256), dim=1)  # L2-normalized
        >>> loss, logdict = loss_fn(z_label, z_idx, z, extra={'global_step': 500})
        >>> print(f"Alignment: {logdict['l_cent']:.4f}, "
        ...       f"Uniformity: {logdict['l_cont']:.4f}")
    """

    def __init__(
        self,
        gamma: float = 8.0,
        b: float = 1.0,
        eps: float = 1e-8,
        epsilon: float = 1e-6,
        uniformity_weight: float = 0.5,
        warmup_steps: int = 1000,
    ) -> None:
        """
        Initialize CLEWS loss.

        Args:
            gamma: Exponential scaling factor for uniformity.
                Controls how much to penalize nearby negatives.
                Typical range for cosine distance: 6-12
                Higher = harder negatives emphasized more
            b: Bias term in uniformity exponential.
                Keep small (0.5-2.0) for cosine distance to avoid saturation
            eps: Small epsilon for numerical stability in mean calculations
            epsilon: Alternative epsilon for log stability (legacy parameter)
            uniformity_weight: Weight λ for uniformity term.
                Balances alignment vs uniformity.
                Typical range: 0.3-0.7
            warmup_steps: Number of training steps to linearly ramp uniformity
                from 0 to uniformity_weight. Helps training stability.
                Set to 0 to disable warmup.
        """
        super().__init__()
        self.gamma = float(gamma)
        self.b = float(b)
        self.eps = float(eps)
        self.epsilon = float(epsilon)
        self.uniformity_weight = float(uniformity_weight)
        self.warmup_steps = int(warmup_steps)

    def _per_anchor_mean(
        self, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute per-anchor mean over masked entries.

        For each anchor (row), computes the mean of x over columns where
        mask is True, ignoring masked-out entries.

        Args:
            x: Values of shape (batch_size, batch_size)
            mask: Boolean mask of shape (batch_size, batch_size)
                True = include, False = exclude
            eps: Small epsilon to avoid division by zero

        Returns:
            Per-anchor means of shape (batch_size,)

        Example:
            >>> x = torch.tensor([[1.0, 2.0, 3.0],
            ...                   [4.0, 5.0, 6.0],
            ...                   [7.0, 8.0, 9.0]])
            >>> mask = torch.tensor([[True, True, False],
            ...                      [True, False, True],
            ...                      [False, True, True]])
            >>> loss_fn = CLEWSLoss()
            >>> means = loss_fn._per_anchor_mean(x, mask)
            >>> # means = [(1+2)/2, (4+6)/2, (8+9)/2] = [1.5, 5.0, 8.5]
        """
        w = mask.float()
        num = (x * w).sum(dim=1)
        den = w.sum(dim=1).clamp_min(eps)
        return num / den  # (B,)

    def forward(
        self,
        z_label: torch.Tensor,
        z_idx: torch.Tensor,
        z: torch.Tensor,
        extra: Optional[Dict[str, Any]] = None,
        numerically_friendly: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute CLEWS loss.

        Args:
            z_label: Clique labels of shape (batch_size,)
            z_idx: Version indices of shape (batch_size,)
            z: Embeddings of shape (batch_size, embedding_dim) or
                (batch_size, 1, embedding_dim).
                Should be L2-normalized for best results with cosine distance.
            extra: Optional dictionary with auxiliary information:
                - 'global_step': Current training step for uniformity warmup
            numerically_friendly: If True, use log1p instead of log for
                numerical stability. Recommended: True

        Returns:
            Tuple containing:
                - loss: Scalar loss tensor (alignment + weighted uniformity)
                - logdict: Dictionary with detailed diagnostics:
                    - l_main: Total loss
                    - l_cent: Alignment loss (centripetal, pulling together)
                    - l_cont: Uniformity loss (contrastive, pushing apart)
                    - cnt_pos_pairs: Number of positive pairs
                    - cnt_neg_pairs: Number of negative pairs
                    - anchors_with_pos: Fraction of anchors with ≥1 positive
                    - v_dpos: Mean distance to positives
                    - v_dneg: Mean distance to negatives
                    - uniformity_weight: Current uniformity weight (after warmup)
                    - z_max: Max absolute value in embeddings
                    - z_mean: Mean of embeddings
                    - z_std: Standard deviation of embeddings

        Raises:
            AssertionError: If batch size < 4 (need enough samples for pos/neg pairs)
            AssertionError: If tensor shapes are inconsistent

        Example:
            >>> # Initialize with warmup
            >>> loss_fn = CLEWSLoss(gamma=10.0, uniformity_weight=0.6, warmup_steps=2000)
            >>>
            >>> # Training loop
            >>> for step, batch in enumerate(dataloader):
            ...     z_label, z_idx, z = batch
            ...     z = F.normalize(z, dim=1)  # L2 normalize
            ...     loss, logdict = loss_fn(z_label, z_idx, z,
            ...                              extra={'global_step': step})
            ...
            ...     # Check if positives exist
            ...     if logdict['anchors_with_pos'] < 0.5:
            ...         print("Warning: Many anchors lack positives!")
            ...
            ...     loss.backward()
        """
        # ---- Shape validation ----
        if z.dim() == 3:
            # Allow (B, 1, C) if it leaks from a temporal pipeline
            assert z.size(1) == 1, f"CLEWS (vector) expects S=1, got S={z.size(1)}"
            z = z.squeeze(1)
        assert z.dim() == 2
        B = z.size(0)
        assert len(z_label) == len(z_idx) == B and B >= 4, (
            f"CLEWS requires batch_size ≥ 4, got {B}"
        )

        # If no negatives at all, inject tiny noise (rare)
        if z_label.unique().numel() == 1:
            z_label[: max(2, int(0.01 * B))] = -1

        # ---- Create positive/negative masks ----
        same_label = z_label.view(-1, 1) == z_label.view(1, -1)  # same clique
        same_idx = z_idx.view(-1, 1) == z_idx.view(1, -1)  # same sample/aug
        pos_mask = same_label & (~same_idx)  # positives
        neg_mask = ~same_label  # negatives

        # ---- Cosine distance (matches retrieval metric) ----
        z = F.normalize(z, p=2, dim=-1)
        sim = z @ z.t()  # (B, B) cosine similarity
        d = 1.0 - sim  # (B, B) cosine distance ∈ [0, 2]

        # ---- Per-anchor alignment ----
        # Mean positive distance for each anchor; skip anchors with no positives
        align_i = self._per_anchor_mean(d, pos_mask, eps=self.eps)  # (B,)
        has_pos = pos_mask.any(dim=1)
        loss_align = align_i[has_pos].mean() if has_pos.any() else (z.sum() * 0.0)

        # ---- Per-anchor uniformity ----
        # Mean over negatives of exp(b - gamma * d), then log1p
        exp_term = (self.b - self.gamma * d).exp()
        uni_i_core = self._per_anchor_mean(exp_term, neg_mask, eps=self.eps)  # (B,)

        if numerically_friendly:
            loss_uniform = uni_i_core.log1p().mean()
        else:
            loss_uniform = (uni_i_core + self.epsilon).log().mean()

        # ---- Uniformity warmup schedule ----
        uw_target = self.uniformity_weight
        uw = uw_target
        if self.warmup_steps > 0:
            # Accept 'global_step' either in 'extra' dict or as module attribute
            step = None
            if isinstance(extra, dict) and "global_step" in extra:
                step = int(extra["global_step"])
            elif hasattr(self, "global_step"):
                step = int(self.global_step)

            if step is not None:
                # Linear ramp from 0 to uw_target over warmup_steps
                uw = float(min(uw_target, uw_target * (step + 1) / self.warmup_steps))

        # ---- Total loss ----
        loss = loss_align + uw * loss_uniform

        # ---- Diagnostics (compute without gradients) ----
        with torch.no_grad():
            n_pos_pairs = pos_mask.float().sum()
            n_neg_pairs = neg_mask.float().sum()
            anchors_with_pos = (
                has_pos.float().mean()
            )  # fraction of anchors with ≥1 positive

            # Mean distances for positives and negatives
            # Note: tops.mmean treats mask=True as EXCLUDED, so we invert the masks
            v_dpos = (
                tops.mmean(d, mask=~pos_mask)
                if n_pos_pairs > 0
                else torch.tensor(0.0, device=z.device)
            )
            v_dneg = (
                tops.mmean(d, mask=~neg_mask)
                if n_neg_pairs > 0
                else torch.tensor(0.0, device=z.device)
            )

            uw_t = torch.tensor(uw, device=z.device)

        logdict = {
            "l_main": loss,
            "l_cent": loss_align,
            "l_cont": loss_uniform,
            "cnt_pos_pairs": n_pos_pairs,
            "cnt_neg_pairs": n_neg_pairs,
            "anchors_with_pos": anchors_with_pos,
            "v_dpos": v_dpos,
            "v_dneg": v_dneg,
            "uniformity_weight": uw_t,
            "z_max": z.abs().max(),
            "z_mean": z.mean(),
            "z_std": z.std(),
        }
        return loss, logdict

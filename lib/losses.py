import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners
from . import tensor_ops as tops 
from einops import rearrange
import math 

class NTXentLoss(nn.Module):
    """
    Wrapper around pytorch-metric-learning's NTXentLoss
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.tau = temperature
        
    def forward(self, z_label, z_idx, z, extra=None):
        """
        Args:
            z_label: (B,) labels for each sample
            z_idx: (B,) indices for each sample  
            z: (B, C) embeddings
            extra: additional info (unused)
        
        Returns:
            loss: scalar loss value
            logdict: dictionary with loss components and metrics
        """
        assert len(z_label) == len(z_idx) and len(z_label) == len(z)
        # If no negatives, add label noise for loss stability
        # (we assume positives exist due to batch construction)
        if len(z_label.unique()) == 1:
            z_label[: max(2, int(len(z_label) * 0.01))] = -1

        # Prepare
        sz = z.size(1)
        #z = rearrange(z, "b s c -> (b s) c")
        same_label = z_label.view(-1, 1) == z_label.view(1, -1)
        same_idx = z_idx.view(-1, 1) == z_idx.view(1, -1)
        positives = same_label & (~same_idx)

        # Distances
        sim = tops.pairwise_distance_matrix(z, z, mode="cossim")

        # Output
        logits = sim / self.tau
        pos_mask = positives.float()

        # Mask out diagonal (self-similarity) to avoid numerical issues
        mask_diag = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
        logits = logits.masked_fill(mask_diag, -1e9)

        # Standard NT-Xent: -log(sum(pos_exp) / sum(all_exp))
        #exp_logits = torch.exp(logits)
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)

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
    Triplet loss implementation using torch.nn.TripletMarginLoss
    """
    
    def __init__(self, margin=0.2, p=2, eps=1e-6, swap=False, reduction='mean'):
        super().__init__()
        self.triplet_loss = nn.TripletMarginLoss(
            margin=margin, 
            p=p, 
            eps=eps, 
            swap=swap, 
            reduction=reduction
        )
    def forward(self, z_label, z_idx, z, extra=None):
        """
        Args:
            z_label: (B,) labels for each sample
            z_idx: (B,) indices for each sample  
            z: (B, C) embeddings
            extra: additional info (unused)
        
        Returns:
            loss: scalar loss value
            logdict: dictionary with loss components and metrics
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
                "n_triplets": 0
            }
            return loss, logdict
        
        # Extract embeddings for triplets
        anchor_embeddings = z[anchors]
        positive_embeddings = z[positives] 
        negative_embeddings = z[negatives]
        
        # Compute triplet loss using PyTorch's implementation
        loss = self.triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        logdict = {
            "l_main": loss,
            "v_zmax": z.abs().max(),
            "v_zmean": z.mean(),
            "v_zstd": z.std(),
        }
        
        return loss, logdict
        
    def _create_triplets(self, z_label, z_idx):
        """Create simple triplets by finding one positive and one negative for each anchor"""
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
            neg_mask = (z_label != z_label[i])
            valid_neg = torch.where(neg_mask)[0]
            
            if len(valid_pos) == 0 or len(valid_neg) == 0:
                continue
                
            # Just take the first available positive and negative (or random)
            pos_idx = valid_pos[0]  # or: valid_pos[torch.randint(len(valid_pos), (1,))]
            neg_idx = valid_neg[0]  # or: valid_neg[torch.randint(len(valid_neg), (1,))]
            
            anchors.append(i)
            positives.append(pos_idx.item())
            negatives.append(neg_idx.item())
            
        return torch.tensor(anchors, device=device), \
               torch.tensor(positives, device=device), \
               torch.tensor(negatives, device=device)




class CLEWSLoss(nn.Module):
    """
    CLEWS for vector embeddings (B, C) with COSINE geometry.
    - Alignment: per-anchor mean positive distance.
    - Uniformity: per-anchor log(1 + mean(exp(b - gamma * d_neg))).
    This is still CLEWS (no softmax / no InfoNCE); we just average per anchor
    instead of a single batch-global average for stability.
    """

    def __init__(
        self,
        gamma: float = 8.0,           # cosine distance lives in [0,2]; 6–12 is typical
        b: float = 1.0,               # keep small for cosine to avoid saturation
        eps: float = 1e-8,
        epsilon: float = 1e-6,
        uniformity_weight: float = 0.5,
        warmup_steps: int = 1000      # linearly ramp uniformity from 0 -> weight
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.b = float(b)
        self.eps = float(eps)
        self.epsilon = float(epsilon)
        self.uniformity_weight = float(uniformity_weight)
        self.warmup_steps = int(warmup_steps)

    def _per_anchor_mean(self, x, mask, eps=1e-8):
        # x, mask : (B, B)
        # returns per-anchor mean over the second dim, ignoring masked-out entries
        w = mask.float()
        num = (x * w).sum(dim=1)
        den = w.sum(dim=1).clamp_min(eps)
        return num / den  # (B,)

    def forward(self, z_label, z_idx, z, extra=None, numerically_friendly=True):
        # ---- shapes ----
        if z.dim() == 3:
            # allow (B,1,C) if it leaks from a temporal pipe
            assert z.size(1) == 1, f"CLEWS (vector) expects S=1, got S={z.size(1)}"
            z = z.squeeze(1)
        assert z.dim() == 2
        B = z.size(0)
        assert len(z_label) == len(z_idx) == B and B >= 4

        # If no negatives at all, inject tiny noise (rare)
        if z_label.unique().numel() == 1:
            z_label[: max(2, int(0.01 * B))] = -1

        # ---- masks (CRITICAL) ----
        same_label = z_label.view(-1, 1) == z_label.view(1, -1)  # same clique
        same_idx   = z_idx.view(-1, 1)   == z_idx.view(1, -1)    # same sample/aug
        pos_mask   = (same_label & (~same_idx))                  # positives
        neg_mask   = (~same_label)                                # negatives

        # ---- cosine distance (matches retrieval) ----
        z = F.normalize(z, p=2, dim=-1)
        sim = z @ z.t()            # (B,B)
        d   = 1.0 - sim            # (B,B) in [0,2]

        # ---- per-anchor alignment ----
        # (mean positive distance for each anchor; anchors with no positives are skipped)
        align_i = self._per_anchor_mean(d, pos_mask, eps=self.eps)               # (B,)
        has_pos = pos_mask.any(dim=1)
        loss_align = align_i[has_pos].mean() if has_pos.any() else (z.sum() * 0.0)

        # ---- per-anchor uniformity ----
        # mean over negatives of exp(b - gamma * d), then log1p
        exp_term = (self.b - self.gamma * d).exp()
        uni_i_core = self._per_anchor_mean(exp_term, neg_mask, eps=self.eps)     # (B,)
        loss_uniform = uni_i_core.log1p().mean() if numerically_friendly else (uni_i_core + self.epsilon).log().mean()

        # ---- warmup uniformity ----
        uw_target = self.uniformity_weight
        uw = uw_target
        if self.warmup_steps > 0:
            # accept 'global_step' either in 'extra' or as attribute on the module
            step = None
            if isinstance(extra, dict) and "global_step" in extra:
                step = int(extra["global_step"])
            elif hasattr(self, "global_step"):
                step = int(self.global_step)
            if step is not None:
                uw = float(min(uw_target, uw_target * (step + 1) / self.warmup_steps))

        loss = loss_align + uw * loss_uniform

        # ---- diagnostics (tensors only) ----
        with torch.no_grad():
            n_pos_pairs = pos_mask.float().sum()
            n_neg_pairs = neg_mask.float().sum()
            anchors_with_pos = has_pos.float().mean()  # fraction of anchors that have at least one positive
            v_dpos = tops.mmean(d, mask=pos_mask) if n_pos_pairs > 0 else torch.tensor(0.0, device=z.device)
            v_dneg = tops.mmean(d, mask=neg_mask) if n_neg_pairs > 0 else torch.tensor(0.0, device=z.device)
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
            "z_std":  z.std(),
        }
        return loss, logdict



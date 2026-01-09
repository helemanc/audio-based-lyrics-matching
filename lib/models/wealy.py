"""
Version Identification Model for the training framework.

This module provides a universal model architecture that works with different
embedding types (Whisper, SBERT) using a transformer-based approach.

models/wealy.py

Interface:
    - Model(conf, sr) constructor
    - prepare() method for preprocessing
    - embed() method for feature extraction  
    - loss() method for contrastive learning
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from lib import layers
from lib import losses


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding from 'Attention Is All You Need'.
    
    Adds position information to embeddings using sine and cosine functions
    of different frequencies. This allows the model to learn to attend by
    relative positions.
    
    Attributes:
        dropout: Dropout layer applied after adding positional encoding
        pe: Positional encoding buffer of shape (1, max_length, hidden_dim)
    
    Example:
        >>> pos_enc = SinusoidalPositionalEncoding(hidden_dim=512, max_length=1000)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq_len, hidden_dim)
        >>> x_with_pos = pos_enc(x)
        >>> x_with_pos.shape
        torch.Size([32, 100, 512])
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        max_length: int = 5000, 
        dropout: float = 0.1
    ) -> None:
        """
        Initialize sinusoidal positional encoding.
        
        Args:
            hidden_dim: Dimension of the embeddings
            max_length: Maximum sequence length to support
            dropout: Dropout probability applied after adding position encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, hidden_dim)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * 
            -(math.log(10000.0) / hidden_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_length, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Embeddings with positional encoding added, same shape as input
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class Model(nn.Module):
    """
    Universal Version Identification Model.
    
    Uses transformer architecture for all embedding types:
    1. SBERT embeddings (B, 1, 384): Treated like temporal sequence
    2. Whisper embeddings (B, T, 1280): Full temporal pipeline  
    3. Averaged embeddings (B, 384/1280): Expand dims and process
    
    The model supports two modes:
    - Temporal mode (use_avg_pooling=False): Full transformer pipeline with
      optional convolutional downsampling, positional encoding, and pooling
    - Averaged mode (use_avg_pooling=True): Simple MLP for pre-averaged embeddings
    
    Attributes:
        sr: Sample rate (for compatibility)
        eps: Small epsilon value for numerical stability
        embedding_type: Type of embeddings ('sbert', 'last_hidden_states', etc.)
        conf: Configuration object
        use_avg_pooling: Whether to use averaged embedding mode
        working_dim: Internal working dimension
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> conf = OmegaConf.create({
        ...     'input_channels': 1280,
        ...     'hidden_dim': 512,
        ...     'zdim': 256,
        ...     'conv_blocks': 2,
        ...     'num_transformer_blocks': 4,
        ...     'num_heads': 8
        ... })
        >>> model = Model(conf, use_avg_pooling=False, embedding_type='last_hidden_states')
        >>> x = torch.randn(4, 100, 1280)  # (batch, time, features)
        >>> z, _ = model.embed(x)
        >>> z.shape
        torch.Size([4, 256])
    """
    
    def __init__(
        self, 
        conf: DictConfig, 
        use_avg_pooling: bool, 
        embedding_type: str, 
        sr: int = 16000, 
        eps: float = 1e-6
    ) -> None:
        """
        Initialize the model.
        
        Args:
            conf: Configuration object containing model hyperparameters:
                - input_channels: Input dimension (e.g., 1280 for Whisper, 384 for SBERT)
                - hidden_dim: Working dimension for transformer
                - zdim: Output embedding dimension
                - conv_blocks: Number of convolutional downsampling blocks
                - conv_kernel_size: Kernel size for conv layers
                - conv_stride: Stride for conv layers
                - num_transformer_blocks: Number of transformer encoder layers
                - num_heads: Number of attention heads
                - ff_dim: Feedforward dimension in transformer
                - dropout: Dropout probability
                - pool_type: Pooling type ('gempool' or 'meanpool')
                - positional_encoding: Config for positional encoding
                - loss: Loss function configuration
            use_avg_pooling: If True, use simple MLP for averaged embeddings.
                If False, use full temporal transformer pipeline
            embedding_type: Type of embeddings ('sbert', 'last_hidden_states', 
                'encoder', 'evaluation_enc_dec')
            sr: Sample rate (for compatibility with audio processing)
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.sr = sr
        self.eps = eps
        self.embedding_type = embedding_type
        self.conf = conf
        self.use_avg_pooling = use_avg_pooling
        
        # Working dimension for all modes
        self.working_dim: int = conf.hidden_dim
        
        # Build architecture based on mode
        if not self.use_avg_pooling:
            # Full temporal pipeline for all embedding types
            self.conv_layers = self._build_conv_layers(conf)
            self.pre_transformer_proj = nn.Linear(conf.input_channels, self.working_dim)
            self.positional_encoding = self._build_positional_encoding(conf)
            self.transformer = self._build_transformer()
            self.pool = self._build_pooling_layer(conf)
            self.proj = nn.Linear(self.working_dim, conf.zdim, bias=False)
        else:
            # Simple MLP for averaged embeddings (any type)
            self.avg_mlp = self._build_avg_mlp(conf)
        
        # Loss function setup
        self._loss_fn: Optional[nn.Module] = None
        self._setup_loss_function()

    def _build_avg_mlp(self, conf: DictConfig) -> nn.Sequential:
        """
        Build MLP for averaged embeddings (any type).
        
        Creates a feedforward network with:
        - Input projection from embedding dimension to working dimension
        - Optional hidden layers with ReLU and dropout
        - Output projection to final embedding dimension
        
        Args:
            conf: Configuration object with:
                - input_channels: Input dimension
                - hidden_dim: Working dimension
                - zdim: Output dimension
                - dropout: Dropout probability
                - avg_mlp_hidden_layers: Number of hidden layers (default: 2)
        
        Returns:
            Sequential module containing the MLP layers
        """
        layers_list = []
        
        # Input projection
        layers_list.append(nn.Linear(conf.input_channels, self.working_dim))
        layers_list.append(nn.ReLU())
        layers_list.append(nn.Dropout(conf.dropout))
        
        # Hidden layers
        num_hidden = getattr(conf, 'avg_mlp_hidden_layers', 2)
        for _ in range(num_hidden):
            layers_list.append(nn.Linear(self.working_dim, self.working_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(conf.dropout))
        
        # Output projection
        layers_list.append(nn.Linear(self.working_dim, conf.zdim))
        
        return nn.Sequential(*layers_list)
    
    def _build_pooling_layer(self, conf: DictConfig) -> nn.Module:
        """
        Build pooling layer based on configuration.
        
        Supports:
        - GeMPool (Generalized Mean Pooling): Learnable pooling with parameter p
        - MeanPool: Simple average pooling across time dimension
        
        Args:
            conf: Configuration object with pool_type attribute
        
        Returns:
            Pooling layer module
        
        Raises:
            ValueError: If pool_type is not recognized
        """
        pool_type = getattr(conf, 'pool_type', 'gempool').lower()
        
        if pool_type == 'gempool':
            return layers.GeMPool(ncha=self.working_dim)
        elif pool_type == 'meanpool':
            return layers.MeanPool()
        else:
            raise ValueError(
                f"Unknown pooling type: {pool_type}. "
                f"Supported: 'gempool', 'meanpool'"
            )
    
    def _build_positional_encoding(self, conf: DictConfig) -> nn.Module:
        """
        Build positional encoding based on configuration.
        
        Args:
            conf: Configuration object with positional_encoding settings:
                - enabled: Whether to use positional encoding
                - type: Type of encoding ('sinusoidal' or 'none')
                - max_length: Maximum sequence length to support
                - dropout: Dropout probability
        
        Returns:
            Positional encoding module or Identity if disabled
        """
        pos_config = getattr(conf, 'positional_encoding', None)
        
        if not pos_config or not getattr(pos_config, 'enabled', False):
            return nn.Identity()
        
        pos_type = getattr(pos_config, 'type', 'none').lower()
        max_length = getattr(pos_config, 'max_length', 5000)
        dropout = getattr(pos_config, 'dropout', 0.1)
        
        if pos_type == 'sinusoidal':
            return SinusoidalPositionalEncoding(
                hidden_dim=self.working_dim,
                max_length=max_length,
                dropout=dropout
            )
        else:
            return nn.Identity()
    
    def _build_conv_layers(self, conf: DictConfig) -> nn.Module:
        """
        Build convolutional layers for temporal downsampling.
        
        Creates a sequence of ConvBlocks that progressively downsample
        the temporal dimension while preserving feature dimension.
        Useful for reducing computational cost in long sequences.
        
        Args:
            conf: Configuration object with:
                - conv_blocks: Number of convolutional blocks (0 to disable)
                - input_channels: Feature dimension
                - conv_kernel_size: Kernel size for convolutions
                - conv_stride: Stride for downsampling
        
        Returns:
            Sequential module of ConvBlocks or Identity if disabled
        """
        if not conf.conv_blocks:
            return nn.Identity()
            
        layers_list = []
        channels = conf.input_channels
        
        for _ in range(conf.conv_blocks):
            layers_list.append(
                layers.ConvBlock(
                    in_channels=channels, 
                    out_channels=channels, 
                    kernel_size=conf.conv_kernel_size, 
                    stride=conf.conv_stride
                )
            )
            
        return nn.Sequential(*layers_list)
    
    def _build_transformer(self) -> nn.TransformerEncoder:
        """
        Build the transformer encoder.
        
        Creates a stack of transformer encoder layers with:
        - Multi-head self-attention
        - Feedforward network
        - Layer normalization (pre-norm architecture)
        - Residual connections
        
        Returns:
            TransformerEncoder module with configured number of layers
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.working_dim,
            nhead=self.conf.num_heads,
            dim_feedforward=self.conf.ff_dim,
            dropout=self.conf.dropout,
            batch_first=True,
            norm_first=True
        )
        return nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.conf.num_transformer_blocks
        )
    
    def _setup_loss_function(self) -> None:
        """
        Setup the contrastive loss function based on configuration.
        
        Supported loss types:
        - ntxent: Normalized Temperature-scaled Cross Entropy (NT-Xent)
        - triplet: Triplet loss with margin
        - clews: CLEWS contrastive loss
        
        Raises:
            ValueError: If loss type is not recognized
        """
        loss_type = getattr(self.conf.loss, 'type', 'ntxent')
        
        if loss_type == 'ntxent':
            self._loss_fn = losses.NTXentLoss(
                temperature=getattr(self.conf.loss, 'temperature', 0.1)
            )
        elif loss_type == 'triplet':
            self._loss_fn = losses.TripletLoss(
                margin=getattr(self.conf.loss, 'margin', 0.1),
                eps=self.eps
            )
        elif loss_type == 'clews':
            self._loss_fn = losses.CLEWSLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_shingle_params(self) -> Tuple[float, float]:
        """
        Return shingling parameters for compatibility.
        
        Returns:
            Tuple of (shingle_length, shingle_hop) in seconds
        """
        return 20.0, 20.0
    
    def prepare(
        self, 
        h: torch.Tensor, 
        shingle_len: Optional[float] = None, 
        shingle_hop: Optional[float] = None
    ) -> torch.Tensor:
        """
        Prepare input data (pass-through since data is pre-processed).
        
        Args:
            h: Input embeddings, shape (batch_size, time, features) or 
                (batch_size, features)
            shingle_len: Shingle length (unused, for compatibility)
            shingle_hop: Shingle hop (unused, for compatibility)
        
        Returns:
            Input tensor unchanged
        
        Raises:
            AssertionError: If input is not 2D or 3D
        """
        assert h.ndim in [2, 3], f"Expected 2D or 3D input, got {h.shape}"
        return h
    
    def embed(
        self, 
        h: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, None]:
        """
        Extract embeddings using transformer pipeline.
        
        Processes input through either:
        1. Averaged mode (use_avg_pooling=True): Simple MLP
        2. Temporal mode (use_avg_pooling=False): Full transformer pipeline
            - Optional convolutional downsampling
            - Linear projection to working dimension
            - Positional encoding
            - Transformer encoder
            - Pooling
            - Final projection to embedding space
        
        Args:
            h: Input embeddings
                - 3D: (batch_size, time, features) for temporal processing
                - 2D: (batch_size, features) for averaged embeddings
            attention_mask: Boolean mask of shape (batch_size, time)
                - True: Valid position, False: Padding
                - Ignored for 2D input
                - Optional for 3D input
        
        Returns:
            Tuple containing:
                - z: Output embeddings of shape (batch_size, zdim)
                - extra: None (for compatibility with training framework)
        
        Example:
            >>> # Temporal embeddings
            >>> h = torch.randn(4, 100, 1280)
            >>> mask = torch.ones(4, 100, dtype=torch.bool)
            >>> z, _ = model.embed(h, mask)
            >>> z.shape
            torch.Size([4, 256])
            
            >>> # Averaged embeddings
            >>> h_avg = torch.randn(4, 1280)
            >>> z_avg, _ = model.embed(h_avg)
            >>> z_avg.shape
            torch.Size([4, 256])
        """
        
        if self.use_avg_pooling:
            # AVERAGED EMBEDDINGS MODE: Use simple MLP
            if h.ndim == 3:
                # If 3D, average across time dimension first
                h = h.mean(dim=1)  # (B, T, D) -> (B, D)
            
            z = self.avg_mlp(h)
            return z, None
        
        else:
            # TEMPORAL PROCESSING MODE: Use full transformer pipeline
            
            # Ensure 3D input
            if h.ndim == 2:
                # (B, D) -> (B, 1, D) - treat as single timestep
                h = h.unsqueeze(1)
                if attention_mask is not None:
                    attention_mask = torch.ones(
                        h.shape[0], 1, 
                        dtype=torch.bool, 
                        device=h.device
                    )
            
            # Apply ConvBlocks for temporal downsampling (if configured)
            if self.conf.conv_blocks:
                x = h.transpose(1, 2)  # (B, D, T)
                x = self.conv_layers(x)  # (B, D, T_downsampled)
                x = x.transpose(1, 2)  # (B, T_downsampled, D)
                
                # Downsample attention mask to match
                if attention_mask is not None:
                    mask_x = attention_mask.float()
                    for _ in range(self.conf.conv_blocks):
                        mask_x = F.max_pool1d(
                            mask_x.unsqueeze(1), 
                            kernel_size=self.conf.conv_kernel_size, 
                            stride=self.conf.conv_stride,
                            padding=self.conf.conv_kernel_size // 2
                        ).squeeze(1)
                    attention_mask_down = mask_x > 0.5
                else:
                    attention_mask_down = None
            else:
                x = h
                attention_mask_down = attention_mask
            
            # Linear projection: input_dim → working_dim
            x = self.pre_transformer_proj(x)
            
            # Add positional encoding
            x = self.positional_encoding(x)
            
            # Transformer encoding with attention mask
            if attention_mask_down is not None:
                # PyTorch transformer uses inverted mask (True = ignore)
                src_key_padding_mask = ~attention_mask_down
            else:
                src_key_padding_mask = None
                
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            
            # Pool and project to final embedding space
            x = x.transpose(1, 2)  # (B, working_dim, T_downsampled)
            x = self.pool(x)  # (B, working_dim)
            z = self.proj(x)  # (B, zdim)
            
            return z, None

    def loss(
        self, 
        z_label: torch.Tensor, 
        z_idx: torch.Tensor, 
        z: torch.Tensor, 
        extra: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            z_label: Clique labels of shape (batch_size,)
            z_idx: Version indices of shape (batch_size,)
            z: Embeddings of shape (batch_size, zdim)
            extra: Optional extra information (unused)
        
        Returns:
            Loss scalar tensor
        """
        return self._loss_fn(z_label, z_idx, z, extra)
    
    def distances(
        self, 
        q: torch.Tensor, 
        c: torch.Tensor, 
        qmask: Optional[torch.Tensor] = None, 
        cmask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pairwise cosine distances between queries and candidates.
        
        Distance is computed as: dist = 1 - cosine_similarity
        
        Args:
            q: Query embeddings of shape (num_queries, zdim)
            c: Candidate embeddings of shape (num_candidates, zdim)
            qmask: Optional mask for queries (unused, for compatibility)
            cmask: Optional mask for candidates (unused, for compatibility)
        
        Returns:
            Distance matrix of shape (num_queries, num_candidates)
            where dist[i, j] = 1 - cosine_similarity(q[i], c[j])
        
        Example:
            >>> q = torch.randn(10, 256)
            >>> c = torch.randn(100, 256)
            >>> dist = model.distances(q, c)
            >>> dist.shape
            torch.Size([10, 100])
        """
        q_norm = F.normalize(q, p=2, dim=1)
        c_norm = F.normalize(c, p=2, dim=1)
        sim = torch.mm(q_norm, c_norm.t())
        dist = 1 - sim
        return dist
    
    def forward(
        self, 
        h: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Full forward pass for inference.
        
        Combines prepare() and embed() for end-to-end processing.
        
        Args:
            h: Input embeddings
                - 3D: (batch_size, time, features)
                - 2D: (batch_size, features)
            attention_mask: Optional boolean mask of shape (batch_size, time)
        
        Returns:
            Output embeddings of shape (batch_size, zdim)
        
        Example:
            >>> x = torch.randn(4, 100, 1280)
            >>> z = model(x)
            >>> z.shape
            torch.Size([4, 256])
        """
        h = self.prepare(h)
        z, _ = self.embed(h, attention_mask)
        return z
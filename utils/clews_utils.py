"""
CLEWS-specific utilities for feature extraction.

Provides complete CLEWS functionality including:
- Downloading CLEWS models from Zenodo
- Loading CLEWS configurations and models
- Shingle-based extraction with masking
- Averaged embeddings with mask pooling
- CLEWS dataset setup
"""

import os
import sys
import requests
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from tqdm import tqdm
import torch
from omegaconf import OmegaConf, DictConfig
import yaml
import importlib

from lib import tensor_ops as tops


# ============================================================================
# CLEWS ZENODO LINKS
# ============================================================================

ZENODO_URLS = {
    "clews": "https://zenodo.org/records/15045900/files/clews.zip"
}


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

def download_file(url: str, destination: Path, desc: Optional[str] = None) -> None:
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save file
        desc: Description for progress bar
    
    Raises:
        requests.RequestException: If download fails
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_clews_checkpoint(
    dataset_name: str,
    checkpoint_dir: str = "./clews_checkpoints"
) -> Tuple[str, str]:
    """
    Download CLEWS checkpoint from Zenodo if it doesn't exist.
    
    Args:
        dataset_name: "shs" or "dvi"
        checkpoint_dir: Directory to store checkpoints
    
    Returns:
        Tuple of (config_path, checkpoint_path)
    
    Raises:
        ValueError: If dataset not supported
        FileNotFoundError: If extraction fails
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected paths after extraction
    if dataset_name == "shs":
        config_path = checkpoint_dir / "shs-clews" / "config" / "shs-clews.yaml"
        checkpoint_path = checkpoint_dir / "shs-clews" / "checkpoint_best.ckpt"
        zip_name = "shs-clews"
    elif dataset_name == "dvi":
        config_path = checkpoint_dir / "dvi-clews" / "config" / "dvi-clews.yaml"
        checkpoint_path = checkpoint_dir / "dvi-clews" / "checkpoint_best.ckpt"
        zip_name = "dvi-clews"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Check if already downloaded
    if config_path.exists() and checkpoint_path.exists():
        print(f"✓ CLEWS {dataset_name} checkpoint already exists at {checkpoint_path}")
        return str(config_path), str(checkpoint_path)
    
    # Download from Zenodo
    print(f"Downloading CLEWS {dataset_name} checkpoint from Zenodo...")
    url = ZENODO_URLS["clews"]
    
    # Download to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {zip_name}") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                pbar.update(len(chunk))
        
        tmp_file_path = tmp_file.name
    
    # Extract zip file
    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
        zip_ref.extractall(checkpoint_dir)
    
    # Clean up temporary file
    os.unlink(tmp_file_path)
    
    # Verify extraction
    if not (config_path.exists() and checkpoint_path.exists()):
        raise FileNotFoundError("Failed to extract CLEWS checkpoint properly")
    
    print(f"✓ Successfully downloaded CLEWS {dataset_name} checkpoint!")
    print(f"  Config: {config_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    
    return str(config_path), str(checkpoint_path)


def auto_setup_clews_paths(conf: DictConfig) -> Tuple[str, str]:
    """
    Automatically setup CLEWS config and checkpoint paths.
    Downloads from Zenodo if needed.
    
    Args:
        conf: Configuration object
    
    Returns:
        Tuple of (config_path, checkpoint_path)
    """
    # Determine dataset name for CLEWS model selection
    dataset_name = conf.data.dataset_name
    
    # Map dataset names to CLEWS model types
    if dataset_name in ["shs"]:
        clews_dataset = "shs"
    elif dataset_name in ["discogs-vi", "dvi"]:
        clews_dataset = "dvi"
    else:
        # Default to SHS model
        print(f"Warning: Unknown dataset {dataset_name}, using SHS CLEWS model")
        clews_dataset = "shs"
    
    # Check if paths are already specified in config
    if hasattr(conf, 'clews'):
        if hasattr(conf.clews, 'config_path') and hasattr(conf.clews, 'checkpoint_path'):
            if conf.clews.config_path and conf.clews.checkpoint_path:
                config_path = Path(conf.clews.config_path)
                checkpoint_base = Path(conf.clews.checkpoint_path)
                
                # Handle checkpoint path construction
                if dataset_name == "shs":
                    checkpoint_path = checkpoint_base / "shs-clews" / "checkpoint_best.ckpt"
                else:
                    checkpoint_path = checkpoint_base / "dvi-clews" / "checkpoint_best.ckpt"
                
                if config_path.exists() and checkpoint_path.exists():
                    print("✓ Using CLEWS paths from config")
                    return str(config_path), str(checkpoint_path)
    
    # Auto-download and setup
    print(f"Auto-setting up CLEWS {clews_dataset} model...")
    checkpoint_dir = conf.clews.checkpoint_path if hasattr(conf, 'clews') else "./clews_checkpoints"
    config_path, checkpoint_path = download_clews_checkpoint(clews_dataset, checkpoint_dir)
    
    return config_path, checkpoint_path


# ============================================================================
# CLEWS MODEL LOADING
# ============================================================================

def load_clews_model(
    conf: DictConfig,
    clews_config_path: str,
    checkpoint_path: str,
    fabric: Any
) -> Tuple[Any, DictConfig]:
    """
    Load CLEWS model from config and checkpoint.
    
    Args:
        conf: Main configuration
        clews_config_path: Path to CLEWS config
        checkpoint_path: Path to CLEWS checkpoint
        fabric: Fabric instance
    
    Returns:
        Tuple of (model, clews_config)
    """
    # Load CLEWS config
    clews_conf = OmegaConf.load(clews_config_path)
    
    # Add CLEWS project to path
    clews_project_dir = conf.clews.project_dir
    if clews_project_dir not in sys.path:
        sys.path.insert(0, clews_project_dir)
    
    # Import CLEWS model
    module = importlib.import_module("models." + clews_conf.model.name)
    with fabric.init_module():
        model = module.Model(clews_conf.model, sr=clews_conf.data.samplerate)
    
    model = fabric.setup(model)
    model.mark_forward_method("prepare")
    model.mark_forward_method("embed")
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading CLEWS checkpoint: {checkpoint_path}")
        state = {'model': model}
        fabric.load(checkpoint_path, state)
        model = state['model']
        print(f"✓ Loaded checkpoint")
    else:
        print("Warning: No checkpoint provided or not found!")
    
    model.eval()
    return model, clews_conf


# ============================================================================
# CLEWS DATASET SETUP
# ============================================================================

def setup_clews_dataset(
    conf: DictConfig,
    clews_conf: DictConfig,
    split: str,
    fabric: Any,
    return_paths: bool = True
) -> Tuple[Any, Any]:
    """
    Setup CLEWS dataset and dataloader.
    
    Args:
        conf: Main configuration
        clews_conf: CLEWS configuration
        split: Dataset split ('train', 'val', 'test')
        fabric: Fabric instance
        return_paths: Whether dataset should return audio paths
    
    Returns:
        Tuple of (dataset, dataloader)
    
    Raises:
        FileNotFoundError: If metadata not found
        ImportError: If CLEWS dataset cannot be imported
    """
    # Change to CLEWS directory for imports
    clews_project_dir = conf.clews.project_dir
    original_cwd = os.getcwd()
    os.chdir(clews_project_dir)
    
    try:
        # Import CLEWS dataset
        dataset_module = importlib.import_module("lib.dataset")
        CLEWSDataset = dataset_module.Dataset
    finally:
        # Restore original directory
        os.chdir(original_cwd)
    
    # Check metadata
    metadata_path = os.path.join(conf.path.clews_cache_dir, "metadata-shs.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"CLEWS metadata not found at {metadata_path}. "
            "Run CLEWS preprocessing first!"
        )
    
    # Create CLEWS dataset config
    clews_data_conf = OmegaConf.create({
        'nworkers': conf.data.nworkers,
        'samplerate': clews_conf.data.samplerate,
        'audiolen': clews_conf.data.audiolen,
        'maxlen': clews_conf.data.get('maxlen', None),
        'pad_mode': clews_conf.data.pad_mode,
        'n_per_class': 1,  # For extraction
        'p_samesong': 0,
        'path': {
            'meta': metadata_path,
            'audio': conf.path.clews_audio_dir
        }
    })
    
    # Create dataset
    ds = CLEWSDataset(
        clews_data_conf,
        split,
        augment=False,
        verbose=fabric.is_global_zero,
        return_paths=return_paths
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=conf.data.batch_size,
        shuffle=False,
        num_workers=conf.data.nworkers,
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )
    
    # Setup with fabric (handles distributed sampling)
    dataloader = fabric.setup_dataloaders(dataloader)
    
    return ds, dataloader


# ============================================================================
# CLEWS EXTRACTION UTILITIES
# ============================================================================

def extract_clews_features_with_shingles(
    model: Any,
    audio_tensor: torch.Tensor,
    maxlen: int,
    shingle_len: Optional[float] = None,
    shingle_hop: Optional[float] = None,
    eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract CLEWS features using shingle-based method.
    
    This replicates the exact extraction method used in distance matrix computation.
    
    Args:
        model: CLEWS model
        audio_tensor: Audio waveform (1, num_samples) or (num_samples,)
        maxlen: Maximum audio length in samples
        shingle_len: Shingle length in seconds (None to use model default)
        shingle_hop: Shingle hop in seconds (None to use model default)
        eps: Epsilon for mask computation
    
    Returns:
        Tuple of (embeddings, mask)
            - embeddings: (num_shingles, embedding_dim) - fixed length
            - mask: (num_shingles,) - True for invalid positions
    
    Example:
        >>> audio = torch.randn(1, 160000)
        >>> embeddings, mask = extract_clews_features_with_shingles(
        ...     model, audio, maxlen=600*16000, shingle_len=None, shingle_hop=5
        ... )
    """
    # Ensure correct dimensions
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Truncate if too long (same as distance matrix script)
    if audio_tensor.size(1) > maxlen:
        audio_tensor = audio_tensor[:, :maxlen]
    
    # Get shingle parameters
    if shingle_len is None or shingle_hop is None:
        model_shingle_len, model_shingle_hop = model.get_shingle_params()
        if shingle_len is None:
            shingle_len = model_shingle_len
        if shingle_hop is None:
            shingle_hop = model_shingle_hop
    
    # Calculate number of shingles
    sr = model.sr
    num_shingles = int((maxlen - int(shingle_len * sr)) / int(shingle_hop * sr))
    
    # Extract embeddings
    with torch.no_grad():
        z = model(
            audio_tensor,
            shingle_len=int(audio_tensor.size(1) / sr) if shingle_len <= 0 else shingle_len,
            shingle_hop=int(0.99 * audio_tensor.size(1) / sr) if shingle_hop <= 0 else shingle_hop,
        )
        
        # Force to fixed length (same as distance matrix script)
        z = tops.force_length(
            z,
            1 if shingle_len <= 0 else num_shingles,
            dim=1,
            pad_mode="zeros",
            cut_mode="start",
        )
        
        # Create mask for zero embeddings
        m = z.abs().max(-1)[0] < eps
        
        # Remove batch dimension
        z = z.squeeze(0)  # (num_shingles, embedding_dim)
        m = m.squeeze(0)  # (num_shingles,)
    
    return z, m


def save_clews_embeddings(
    embeddings: torch.Tensor,
    save_path: Path,
    language: Optional[str] = None
) -> None:
    """
    Save full CLEWS shingle embeddings.
    
    Args:
        embeddings: (num_shingles, embedding_dim) tensor
        save_path: Base directory to save to
        language: Optional language tag
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"hs_clews{'_' + language if language else ''}.pt"
    filepath = save_path / filename
    
    if not filepath.exists():
        # Save as half precision
        torch.save(embeddings.half(), filepath)


def save_clews_mask(
    mask: torch.Tensor,
    save_path: Path,
    language: Optional[str] = None
) -> None:
    """
    Save CLEWS mask (indicates invalid shingles).
    
    Args:
        mask: (num_shingles,) boolean tensor - True for invalid
        save_path: Base directory to save to
        language: Optional language tag
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"hs_clews_mask{'_' + language if language else ''}.pt"
    filepath = save_path / filename
    
    if not filepath.exists():
        torch.save(mask, filepath)


def save_clews_averaged_embedding(
    embeddings: torch.Tensor,
    masks: torch.Tensor,
    save_path: Path,
    language: Optional[str] = None
) -> None:
    """
    Save averaged CLEWS embedding using masked mean pooling.
    
    Args:
        embeddings: (num_shingles, embedding_dim) tensor (half precision)
        masks: (num_shingles,) boolean tensor - True for invalid positions
        save_path: Base directory to save to
        language: Optional language tag
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Compute masked average
    valid_mask = ~masks  # Invert: True for valid positions
    valid_mask = valid_mask.float()
    
    if valid_mask.sum() > 0:
        # Mask out invalid positions
        masked_embeddings = embeddings * valid_mask.unsqueeze(-1)
        
        # Average over valid positions
        sum_embeddings = masked_embeddings.sum(dim=0)
        valid_count = valid_mask.sum()
        averaged_embedding = sum_embeddings / valid_count
    else:
        # Fallback: average everything (shouldn't happen)
        averaged_embedding = embeddings.mean(dim=0)
    
    # Save averaged embedding
    filename = f"hs_clews_avg{'_' + language if language else ''}.pt"
    filepath = save_path / filename
    
    if not filepath.exists():
        torch.save(averaged_embedding, filepath)


def get_clews_extraction_params(
    model: Any,
    maxlen_seconds: float = 600,
    shingle_len: Optional[float] = None,
    shingle_hop: Optional[float] = 5
) -> Dict[str, Any]:
    """
    Get CLEWS extraction parameters.
    
    Args:
        model: CLEWS model
        maxlen_seconds: Maximum audio length in seconds
        shingle_len: Shingle length in seconds (None for model default)
        shingle_hop: Shingle hop in seconds
    
    Returns:
        Dict with extraction parameters
    """
    sr = model.sr
    maxlen_samples = int(maxlen_seconds * sr)
    
    # Get shingle parameters
    if shingle_len is None or shingle_hop is None:
        model_shingle_len, model_shingle_hop = model.get_shingle_params()
        if shingle_len is None:
            shingle_len = model_shingle_len
        if shingle_hop is None:
            shingle_hop = model_shingle_hop
    
    # Calculate number of shingles
    num_shingles = int((maxlen_samples - int(shingle_len * sr)) / int(shingle_hop * sr))
    
    return {
        'maxlen_seconds': maxlen_seconds,
        'maxlen_samples': maxlen_samples,
        'shingle_len': shingle_len,
        'shingle_hop': shingle_hop,
        'num_shingles': num_shingles,
        'sample_rate': sr
    }


# ============================================================================
# PROJECT SETUP
# ============================================================================

def setup_clews_project(project_dir: Path) -> bool:
    """
    Ensure CLEWS project is set up and in Python path.
    
    Args:
        project_dir: Path to CLEWS project directory
    
    Returns:
        True if setup successful
    
    Raises:
        RuntimeError: If CLEWS cannot be imported
    """
    project_dir = Path(project_dir)
    
    # Add to Python path
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    
    return True
"""
CLEWS-specific utilities for feature extraction.

Provides complete CLEWS functionality including:
- Downloading CLEWS models from Zenodo
- Loading CLEWS configurations and models
- Shingle-based extraction with masking
- Averaged embeddings with mask pooling
- CLEWS dataset setup
"""

import importlib
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# ============================================================================
# CLEWS ZENODO LINKS
# ============================================================================

ZENODO_URLS = {"clews": "https://zenodo.org/records/15045900/files/clews.zip"}


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

    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_clews_checkpoint(
    dataset_name: str, checkpoint_dir: str = "./clews_checkpoints"
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=f"Downloading {zip_name}"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
                pbar.update(len(chunk))

        tmp_file_path = tmp_file.name

    # Extract zip file
    print(f"Extracting {zip_name}...")
    with zipfile.ZipFile(tmp_file_path, "r") as zip_ref:
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
    if hasattr(conf, "clews"):
        if hasattr(conf.clews, "config_path") and hasattr(
            conf.clews, "checkpoint_path"
        ):
            if conf.clews.config_path and conf.clews.checkpoint_path:
                config_path = Path(conf.clews.config_path)
                checkpoint_base = Path(conf.clews.checkpoint_path)

                # Handle checkpoint path construction
                if dataset_name == "shs":
                    checkpoint_path = (
                        checkpoint_base / "shs-clews" / "checkpoint_best.ckpt"
                    )
                else:
                    checkpoint_path = (
                        checkpoint_base / "dvi-clews" / "checkpoint_best.ckpt"
                    )

                if config_path.exists() and checkpoint_path.exists():
                    print("✓ Using CLEWS paths from config")
                    return str(config_path), str(checkpoint_path)

    # Auto-download and setup
    print(f"Auto-setting up CLEWS {clews_dataset} model...")
    checkpoint_dir = (
        conf.clews.checkpoint_path if hasattr(conf, "clews") else "./clews_checkpoints"
    )
    config_path, checkpoint_path = download_clews_checkpoint(
        clews_dataset, checkpoint_dir
    )

    return config_path, checkpoint_path


# ============================================================================
# CLEWS MODEL LOADING
# ============================================================================


def load_clews_model(
    conf: DictConfig, clews_config_path: str, checkpoint_path: str, fabric: Any
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
        if fabric.is_global_zero:
            print(f"Loading CLEWS checkpoint: {checkpoint_path}")
        state = {"model": model}
        fabric.load(checkpoint_path, state)
        model = state["model"]
        if fabric.is_global_zero:
            print(f"✓ Loaded checkpoint")
    else:
        if fabric.is_global_zero:
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
    return_paths: bool = True,
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
    clews_data_conf = OmegaConf.create(
        {
            "nworkers": conf.data.nworkers,
            "samplerate": clews_conf.data.samplerate,
            "audiolen": clews_conf.data.audiolen,
            "maxlen": clews_conf.data.get("maxlen", None),
            "pad_mode": clews_conf.data.pad_mode,
            "n_per_class": 1,  # For extraction
            "p_samesong": 0,
            "path": {"meta": metadata_path, "audio": conf.path.clews_audio_dir},
        }
    )

    # Create dataset
    ds = CLEWSDataset(
        clews_data_conf,
        split,
        augment=False,
        verbose=fabric.is_global_zero,
        return_paths=return_paths,
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
    eps: float = 1e-6,
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
    # Import CLEWS tensor ops (local import to avoid triggering whisper dependencies)
    from lib import tensor_ops as tops

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
            shingle_len=int(audio_tensor.size(1) / sr)
            if shingle_len <= 0
            else shingle_len,
            shingle_hop=int(0.99 * audio_tensor.size(1) / sr)
            if shingle_hop <= 0
            else shingle_hop,
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
    embeddings: torch.Tensor, save_path: Path, language: Optional[str] = None
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
    mask: torch.Tensor, save_path: Path, language: Optional[str] = None
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
    language: Optional[str] = None,
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
    shingle_hop: Optional[float] = 5,
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
        "maxlen_seconds": maxlen_seconds,
        "maxlen_samples": maxlen_samples,
        "shingle_len": shingle_len,
        "shingle_hop": shingle_hop,
        "num_shingles": num_shingles,
        "sample_rate": sr,
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


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================


def run_clews_extraction(args, conf):
    """
    Complete CLEWS extraction pipeline.

    Handles model loading, dataset setup, feature extraction, and saving.
    Avoids Whisper dependencies by using standalone CLEWS logic.

    Args:
        args: Command-line arguments
        conf: Configuration object
    """
    from lightning import Fabric
    from lightning.fabric.strategies import DDPStrategy
    from tqdm import tqdm

    from utils import print_utils

    # Set CLEWS-specific defaults
    args.ngpus = getattr(args, "ngpus", conf.fabric.ngpus)
    args.nnodes = getattr(args, "nnodes", conf.fabric.nnodes)
    args.precision = "32"  # CLEWS uses fp32
    args.partition = getattr(args, "partition", conf.data.split)
    args.skip_existing = getattr(args, "skip_existing", conf.extraction.skip_existing)
    args.limit_num = getattr(args, "limit_num", None)
    args.maxlen = getattr(args, "maxlen", 600)
    args.qslen = getattr(args, "qslen", None)
    args.qshop = getattr(args, "qshop", 5)

    # Setup PyTorch
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("medium")
    torch.autograd.set_detect_anomaly(False)

    # Setup Fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=args.ngpus,
        num_nodes=args.nnodes,
        strategy=DDPStrategy(broadcast_buffers=False),
        precision=args.precision,
    )

    fabric.launch()
    fabric.barrier()
    fabric.seed_everything(conf.seed + fabric.global_rank, workers=True)

    myprint = lambda s, end="\n": print_utils.myprint(
        s, end=end, doit=fabric.is_global_zero
    )

    if fabric.is_global_zero:
        print("=" * 70)
        print("CLEWS FEATURE EXTRACTION")
        print("=" * 70)
        print(OmegaConf.to_yaml(conf))
        print("=" * 70)

    # Setup checkpoint paths (auto-download if needed)
    myprint("Setting up CLEWS model...")
    config_path, checkpoint_path = auto_setup_clews_paths(conf)

    # Load CLEWS model using utility function
    myprint("Loading CLEWS model...")

    # CRITICAL: Change to CLEWS directory and clear cached modules
    clews_project_dir = conf.clews.project_dir
    original_cwd = os.getcwd()
    os.chdir(clews_project_dir)

    # Clear conflicting modules
    modules_to_clear = [
        k for k in sys.modules.keys() if k.startswith("utils") or k.startswith("lib")
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]

    # Add CLEWS to path
    if clews_project_dir not in sys.path:
        sys.path.insert(0, clews_project_dir)

    try:
        model, clews_conf = load_clews_model(conf, config_path, checkpoint_path, fabric)

        # Import CLEWS dataset
        dataset_module = importlib.import_module("lib.dataset")
        CLEWSDataset = dataset_module.Dataset
    finally:
        os.chdir(original_cwd)

    # Setup dataset and dataloader
    myprint("Setting up CLEWS dataloader...")

    metadata_filename = f"metadata-{conf.data.dataset_name}.pt"
    metadata_path = os.path.join(conf.path.clews_cache_dir, metadata_filename)

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"CLEWS metadata not found at {metadata_path}. "
            f"Run CLEWS preprocessing first (data_preproc.py)"
        )

    clews_data_conf = OmegaConf.create(
        {
            "nworkers": conf.data.nworkers,
            "samplerate": clews_conf.data.samplerate,
            "audiolen": clews_conf.data.audiolen,
            "maxlen": clews_conf.data.get("maxlen", None),
            "pad_mode": clews_conf.data.pad_mode,
            "n_per_class": 1,
            "p_samesong": 0,
            "path": {"meta": metadata_path, "audio": conf.path.clews_audio_dir},
        }
    )

    ds = CLEWSDataset(
        clews_data_conf,
        args.partition,
        augment=False,
        verbose=fabric.is_global_zero,
        return_paths=True,
    )

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=conf.data.batch_size,
        shuffle=False,
        num_workers=conf.data.nworkers,
        drop_last=False,
        persistent_workers=False,
        pin_memory=True,
    )

    dataloader = fabric.setup_dataloaders(dataloader)

    # Get extraction parameters
    myprint("Starting CLEWS feature extraction...")
    params = get_clews_extraction_params(
        model,
        maxlen_seconds=args.maxlen,
        shingle_len=args.qslen,
        shingle_hop=args.qshop,
    )

    myprint(f"  Max length: {params['maxlen_seconds']}s")
    myprint(f"  Shingle len: {params['shingle_len']}s")
    myprint(f"  Shingle hop: {params['shingle_hop']}s")
    myprint(f"  Num shingles: {params['num_shingles']}")

    # Import CLEWS tensor ops for extraction
    from lib import tensor_ops as tops

    # Path extraction utilities (inline to avoid import issues in container)
    def extract_path_info(audio_path, dataset_name):
        audio_path = Path(audio_path)
        if dataset_name == "shs":
            clique_id = audio_path.parent.name
            version_id = audio_path.stem
            return clique_id, version_id, (clique_id, version_id)
        elif dataset_name in ["discogs-vi", "dvi"]:
            version_id = audio_path.stem
            clique_id = version_id
            dir_name = audio_path.parent.name
            filename = audio_path.stem
            return clique_id, version_id, (dir_name, filename)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def get_save_path(
        hidden_states_folder, dataset_name, clique_id, version_id, save_components
    ):
        base_path = Path(hidden_states_folder)
        if dataset_name == "shs":
            clique_folder, version_folder = save_components
            return base_path / clique_folder / version_folder
        elif dataset_name in ["discogs-vi", "dvi"]:
            return base_path / Path(*save_components)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    hidden_states_folder = conf.path.save_data_path
    total_files = 0
    extracted_files = 0
    skipped_files = 0

    # Main extraction loop
    for batch in tqdm(
        dataloader, desc="Extract CLEWS", disable=not fabric.is_global_zero
    ):
        batch_size = len(batch[0]) if isinstance(batch[0], torch.Tensor) else 1

        for batch_idx in range(batch_size):
            # Extract audio tensors from batch
            item_ids = []
            audio_tensors = []
            for i in range(1, len(batch) - 1, 2):
                if isinstance(batch[i], torch.Tensor):
                    item_ids.append(batch[i][batch_idx].item())
                    audio_tensors.append(batch[i + 1][batch_idx])

            # Extract audio paths
            audio_paths = batch[-1]
            if isinstance(audio_paths, (list, tuple)):
                audio_paths = (
                    audio_paths[batch_idx]
                    if len(audio_paths) > batch_idx
                    else audio_paths[0]
                )

            # Process each audio
            for j, audio_tensor in enumerate(audio_tensors):
                total_files += 1
                audio_path = (
                    audio_paths[j]
                    if isinstance(audio_paths, (list, tuple))
                    else audio_paths
                )

                # Get save path
                clique_id, version_id, save_components = extract_path_info(
                    audio_path, conf.data.dataset_name
                )
                save_base_path = get_save_path(
                    hidden_states_folder,
                    conf.data.dataset_name,
                    clique_id,
                    version_id,
                    save_components,
                )

                # Check if should skip (all 3 files must exist with correct shape)
                clews_path = save_base_path / "hs_clews.pt"
                mask_path = save_base_path / "hs_clews_mask.pt"
                avg_path = save_base_path / "hs_clews_avg.pt"

                should_skip = False
                if (
                    args.skip_existing
                    and clews_path.exists()
                    and mask_path.exists()
                    and avg_path.exists()
                ):
                    try:
                        existing = torch.load(clews_path)
                        if existing.shape[0] == params["num_shingles"]:
                            should_skip = True
                            skipped_files += 1
                            if fabric.is_global_zero:
                                myprint(
                                    f"  Skipping {clique_id}/{version_id} (already extracted)"
                                )
                        else:
                            myprint(
                                f"  Overwriting {clique_id}/{version_id} (shape mismatch)"
                            )
                    except Exception as e:
                        myprint(f"  Overwriting {clique_id}/{version_id} (error: {e})")
                elif args.skip_existing and (
                    clews_path.exists() or mask_path.exists() or avg_path.exists()
                ):
                    myprint(
                        f"  Overwriting {clique_id}/{version_id} (incomplete files)"
                    )

                if should_skip:
                    continue

                extracted_files += 1

                if fabric.is_global_zero:
                    myprint(f"  Extracting CLEWS features for {clique_id}/{version_id}")

                # Extract features using utility function
                z, m = extract_clews_features_with_shingles(
                    model,
                    audio_tensor,
                    maxlen=params["maxlen_samples"],
                    shingle_len=params["shingle_len"],
                    shingle_hop=params["shingle_hop"],
                )

                # Move to CPU and convert to half precision
                z_cpu = z.cpu().half()
                m_cpu = m.cpu()

                if fabric.is_global_zero:
                    non_zero = (~m_cpu).sum().item()
                    myprint(
                        f"    Embeddings: {z_cpu.shape}, Non-zero: {non_zero}/{m_cpu.size(0)}"
                    )

                # Save using utility functions
                save_base_path.mkdir(parents=True, exist_ok=True)
                torch.save(z_cpu, save_base_path / "hs_clews.pt")
                torch.save(m_cpu, save_base_path / "hs_clews_mask.pt")

                # Save averaged embedding
                valid_mask = (~m_cpu).float()
                if valid_mask.sum() > 0:
                    avg = (z_cpu * valid_mask.unsqueeze(-1)).sum(0) / valid_mask.sum()
                else:
                    avg = z_cpu.mean(0)
                torch.save(avg, save_base_path / "hs_clews_avg.pt")

                if fabric.is_global_zero:
                    myprint(f"    Averaged: {avg.shape}")

                if args.limit_num and extracted_files >= args.limit_num:
                    break

            if args.limit_num and extracted_files >= args.limit_num:
                break

        if args.limit_num and extracted_files >= args.limit_num:
            break

    myprint(
        f"CLEWS extraction complete: {extracted_files} extracted, {skipped_files} skipped"
    )

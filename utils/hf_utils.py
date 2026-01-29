"""
Hugging Face Hub utilities for model checkpoint management.

Provides functions for:
- Downloading pre-trained models from Hugging Face Hub
- Sanitizing configurations for upload (removing personal paths)
- Applying path overrides from CLI arguments

Example usage:
    # Download a model
    checkpoint_path = download_model_from_hf(
        "username/wealy-shs-full",
        save_dir="logs"
    )

    # Apply path overrides
    conf = apply_path_overrides(conf, args)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import DictConfig, OmegaConf

# Hugging Face Hub is optional - only needed for download/upload
try:
    from huggingface_hub import HfApi, snapshot_download

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


#########################################################################################
# Constants
#########################################################################################

# Paths that should be overridable by users
OVERRIDABLE_PATHS = [
    "hidden_states",
    "meta",
    "shs_data",
    "shs_splits",
    "lyric_covers_data",
    "discogs_vi_data",
    "data",
    "cache",
]

# Default HF organization/username
DEFAULT_HF_ORG = "audio-based-lyrics-matching"

# Model registry: maps model names to HF repo IDs
# Users can use short names (e.g., "wealy-whisper-shs") or full repo IDs
MODEL_REGISTRY = {
    # SHS100K models
    "wealy-whisper-shs": f"{DEFAULT_HF_ORG}/wealy-whisper-shs",
    "wealy-sbert-shs": f"{DEFAULT_HF_ORG}/wealy-sbert-shs",
    "wealy-whisper-en-shs": f"{DEFAULT_HF_ORG}/wealy-whisper-en-shs",
    "wealy-avgembmlp-shs": f"{DEFAULT_HF_ORG}/wealy-avgembmlp-shs",
    "wealy-cls-shs": f"{DEFAULT_HF_ORG}/wealy-cls-shs",
    # Lyric Covers models
    "wealy-whisper-lyc": f"{DEFAULT_HF_ORG}/wealy-whisper-lyc",
    "wealy-sbert-lyc": f"{DEFAULT_HF_ORG}/wealy-sbert-lyc",
    # Discogs-VI models
    "wealy-whisper-dvi": f"{DEFAULT_HF_ORG}/wealy-whisper-dvi",
}

# Maps local directory names (with underscores) to HF model names (with dashes)
LOCAL_TO_HF_NAME = {
    "wealy_whisper_shs": "wealy-whisper-shs",
    "wealy_sbert_shs": "wealy-sbert-shs",
    "wealy_whisper_en_shs": "wealy-whisper-en-shs",
    "wealy_avgembmlp_shs": "wealy-avgembmlp-shs",
    "wealy_cls_shs": "wealy-cls-shs",
    "wealy_whisper_lyc": "wealy-whisper-lyc",
    "wealy_sbert_lyc": "wealy-sbert-lyc",
    "wealy_whisper_dvi": "wealy-whisper-dvi",
}


def list_available_models() -> Dict[str, str]:
    """
    List all available pre-trained models.

    Returns:
        Dictionary mapping short names to full HF repo IDs
    """
    return MODEL_REGISTRY.copy()


def print_available_models() -> None:
    """Print all available models in a formatted table."""
    print("\nAvailable pre-trained models:")
    print("-" * 60)
    print(f"{'Short Name':<25} {'HF Repository':<35}")
    print("-" * 60)
    for name, repo in MODEL_REGISTRY.items():
        print(f"{name:<25} {repo:<35}")
    print("-" * 60)
    print("\nUsage: model_name=<short-name> or model_name=<full-repo-id>")


#########################################################################################
# Download Functions
#########################################################################################


def check_hf_available() -> bool:
    """Check if huggingface_hub is installed."""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub is not installed.")
        print("Install it with: pip install huggingface_hub")
        return False
    return True


def get_local_model_path(model_name: str, base_dir: str = "logs") -> Path:
    """
    Get the local path where a model should be stored.

    Args:
        model_name: Model name (e.g., "wealy-shs-full")
        base_dir: Base directory for models (default: "logs")

    Returns:
        Path to local model directory
    """
    # Extract just the model name from repo_id if full path provided
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    return Path(base_dir) / model_name


def is_model_downloaded(model_name: str, base_dir: str = "logs") -> bool:
    """
    Check if a model is already downloaded locally.

    Args:
        model_name: Model name or HF repo ID
        base_dir: Base directory for models

    Returns:
        True if both checkpoint and config exist locally
    """
    local_path = get_local_model_path(model_name, base_dir)
    checkpoint_exists = any(local_path.glob("*.ckpt"))
    config_exists = (local_path / "configuration.yaml").exists()

    return checkpoint_exists and config_exists


def download_model_from_hf(
    repo_id: str, save_dir: str = "logs", revision: str = "main", force: bool = False
) -> Tuple[str, str]:
    """
    Download a model checkpoint and configuration from Hugging Face Hub.

    Args:
        repo_id: HF repository ID (e.g., "username/wealy-shs-full")
        save_dir: Base directory to save the model (default: "logs")
        revision: Git revision to download (default: "main")
        force: Force re-download even if exists locally

    Returns:
        Tuple of (checkpoint_path, config_path)

    Raises:
        ImportError: If huggingface_hub is not installed
        ValueError: If download fails

    Example:
        >>> ckpt_path, conf_path = download_model_from_hf("username/wealy-shs-full")
        >>> print(f"Checkpoint: {ckpt_path}")
    """
    if not check_hf_available():
        raise ImportError("huggingface_hub is required for downloading models")

    # Determine local save path
    model_name = repo_id.split("/")[-1]
    local_dir = Path(save_dir) / model_name

    # Check if already downloaded
    if not force and is_model_downloaded(model_name, save_dir):
        print(f"Model already exists at {local_dir}")
        checkpoint_path = str(next(local_dir.glob("*.ckpt")))
        config_path = str(local_dir / "configuration.yaml")
        return checkpoint_path, config_path

    print(f"Downloading model from {repo_id}...")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download all files from the repository
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )

        # Find the checkpoint file
        ckpt_files = list(local_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise ValueError(f"No .ckpt file found in {repo_id}")

        checkpoint_path = str(ckpt_files[0])
        config_path = str(local_dir / "configuration.yaml")

        if not Path(config_path).exists():
            raise ValueError(f"No configuration.yaml found in {repo_id}")

        print(f"Downloaded model to {local_dir}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Config: {config_path}")

        return checkpoint_path, config_path

    except Exception as e:
        raise ValueError(f"Failed to download model from {repo_id}: {e}")


def resolve_model_name(model_name: str) -> str:
    """
    Resolve a short model name to full HF repo ID.

    Args:
        model_name: Short name (e.g., "wealy-shs-full") or full repo ID

    Returns:
        Full HF repo ID
    """
    # If already a full repo ID (contains /), return as-is
    if "/" in model_name:
        return model_name

    # Look up in registry
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Assume it's under the default org
    return f"{DEFAULT_HF_ORG}/{model_name}"


#########################################################################################
# Configuration Management
#########################################################################################


def sanitize_config_for_upload(conf: DictConfig) -> DictConfig:
    """
    Sanitize configuration by removing personal paths.

    Replaces absolute paths with placeholder values that users
    must override at runtime.

    Args:
        conf: Original configuration

    Returns:
        Sanitized configuration safe for public upload

    Example:
        >>> clean_conf = sanitize_config_for_upload(conf)
        >>> OmegaConf.save(clean_conf, "configuration.yaml")
    """
    # Deep copy to avoid modifying original
    conf = OmegaConf.to_container(conf, resolve=True)
    conf = OmegaConf.create(conf)

    # Paths to sanitize - key name -> placeholder value
    path_placeholders = {
        "cache": "/path/to/cache",
        "logs": "/path/to/logs",
        "working_dir": "/path/to/project",
        "data": "/path/to/data",
        "save_data_path": "/path/to/data",
        "hidden_states": "/path/to/hidden-states",
        "meta": "/path/to/metadata.pt",
        "shs_data": "/path/to/shs_data.csv",
        "shs_splits": "/path/to/shs/splits",
        "lyric_covers_data": "/path/to/lyric-covers",
        "discogs_vi_data": "/path/to/discogs-vi",
    }

    # Prefixes that indicate personal/machine-specific paths
    personal_path_prefixes = (
        "/scratch",
        "/home/",
        "/Users/",
        "/tmp/",
        "/data/",
    )

    def is_personal_path(value: Any) -> bool:
        """Check if a value looks like a personal path."""
        if not isinstance(value, str):
            return False
        return value.startswith(personal_path_prefixes)

    def sanitize_dict(obj: Any) -> Any:
        """Recursively sanitize all paths in a dict/DictConfig."""
        if isinstance(obj, (DictConfig, dict)):
            for key in list(obj.keys()):
                value = obj[key]
                # If key is in our known path keys, use placeholder
                if key in path_placeholders:
                    obj[key] = path_placeholders[key]
                # If value looks like a personal path, sanitize it
                elif is_personal_path(value):
                    obj[key] = f"/path/to/{key}"
                # Recurse into nested dicts
                elif isinstance(value, (dict, DictConfig)):
                    sanitize_dict(value)
        return obj

    # Sanitize top-level path section
    if "path" in conf:
        sanitize_dict(conf.path)

    # Sanitize data.path section if it exists
    if "data" in conf and "path" in conf.data:
        sanitize_dict(conf.data.path)

    # Remove checkpoint path (will be set at runtime)
    if "checkpoint" in conf:
        conf.checkpoint = None

    # Sanitize jobname to remove personal identifiers
    if "jobname" in conf:
        conf.jobname = "wealy_model"

    # Fix model name: whisper-ft -> wealy (they're the same architecture)
    if "model" in conf and "name" in conf.model:
        if conf.model.name in ("whisper-ft", "whisper_ft"):
            conf.model.name = "wealy"

    # Ensure data.path section exists and is sanitized
    if "data" in conf:
        if "path" not in conf.data:
            # Create data.path mirroring the top-level path section
            conf.data.path = OmegaConf.create({})
        # Copy and sanitize path placeholders to data.path
        for key, placeholder in path_placeholders.items():
            if key not in conf.data.path:
                conf.data.path[key] = placeholder
            else:
                conf.data.path[key] = placeholder

    # Add pytorch section if missing (with safe defaults)
    if "pytorch" not in conf:
        conf.pytorch = OmegaConf.create(
            {
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "float32_matmul_precision": "medium",
                "detect_anomaly": False,
            }
        )

    # Add early_stopping section if missing
    if "early_stopping" not in conf:
        conf.early_stopping = OmegaConf.create(
            {
                "enabled": False,
                "patience": 10,
                "mode": "max",
                "min_delta": 0.0,
                "metric": "m_MAP",
            }
        )

    # Remove conf path (training config reference)
    if "conf" in conf:
        del conf["conf"]

    return conf


def apply_path_overrides(conf: DictConfig, args: Any) -> DictConfig:
    """
    Apply path overrides from CLI arguments to configuration.

    CLI arguments take precedence over config file values.

    Args:
        conf: Configuration object
        args: CLI arguments with path overrides

    Returns:
        Configuration with paths overridden

    Example:
        # CLI: hidden_states=/my/path/hidden-states
        >>> conf = apply_path_overrides(conf, args)
        >>> print(conf.path.hidden_states)  # /my/path/hidden-states
    """
    # List of path keys that can be overridden
    path_keys = [
        "hidden_states",
        "meta",
        "shs_data",
        "shs_splits",
        "lyric_covers_data",
        "discogs_vi_data",
        "data",
        "cache",
        "working_dir",
    ]

    for key in path_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            value = getattr(args, key)

            # Update in conf.path if it exists
            if "path" in conf and hasattr(conf.path, key):
                conf.path[key] = value

            # Also update in conf.data.path if it exists
            if "data" in conf and "path" in conf.data and hasattr(conf.data.path, key):
                conf.data.path[key] = value

    return conf


def validate_required_paths(conf: DictConfig, args: Any) -> List[str]:
    """
    Validate that required paths are set and exist.

    Args:
        conf: Configuration object
        args: CLI arguments

    Returns:
        List of error messages (empty if all paths are valid)
    """
    errors = []

    # Required paths for inference
    required = ["hidden_states"]

    for key in required:
        value = None

        # Check args first
        if hasattr(args, key) and getattr(args, key):
            value = getattr(args, key)
        # Then check conf.path
        elif "path" in conf and hasattr(conf.path, key):
            value = conf.path[key]
        # Then check conf.data.path
        elif "data" in conf and "path" in conf.data and hasattr(conf.data.path, key):
            value = conf.data.path[key]

        if not value:
            errors.append(
                f"Required path '{key}' is not set. Use CLI arg: {key}=/path/to/{key}"
            )
        elif value.startswith("/path/to/"):
            errors.append(
                f"Path '{key}' has placeholder value. Use CLI arg: {key}=/your/actual/path"
            )
        elif not os.path.exists(value):
            errors.append(f"Path '{key}' does not exist: {value}")

    return errors


#########################################################################################
# Upload Functions
#########################################################################################


def upload_model_to_hf(
    checkpoint_path: str,
    config_path: str,
    repo_id: str,
    commit_message: str = "Upload model checkpoint",
    private: bool = False,
) -> str:
    """
    Upload a model checkpoint and configuration to Hugging Face Hub.

    The configuration is automatically sanitized to remove personal paths.

    Args:
        checkpoint_path: Path to .ckpt file
        config_path: Path to configuration.yaml
        repo_id: HF repository ID (e.g., "username/wealy-shs-full")
        commit_message: Commit message for the upload
        private: Whether to make the repo private

    Returns:
        URL of the uploaded model

    Example:
        >>> url = upload_model_to_hf(
        ...     "logs/wealy_shs_full/best.ckpt",
        ...     "logs/wealy_shs_full/configuration.yaml",
        ...     "username/wealy-shs-full"
        ... )
    """
    if not check_hf_available():
        raise ImportError("huggingface_hub is required for uploading models")

    api = HfApi()

    # Create or get repo
    print(f"Creating/accessing repository: {repo_id}")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    # Load and sanitize config
    print("Sanitizing configuration...")
    conf = OmegaConf.load(config_path)
    clean_conf = sanitize_config_for_upload(conf)

    # Save sanitized config to temp file
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        clean_config_path = os.path.join(tmpdir, "configuration.yaml")
        OmegaConf.save(clean_conf, clean_config_path)

        # Upload checkpoint
        print(f"Uploading checkpoint: {checkpoint_path}")
        ckpt_name = os.path.basename(checkpoint_path)
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=ckpt_name,
            repo_id=repo_id,
            commit_message=f"{commit_message} - checkpoint",
        )

        # Upload sanitized config
        print("Uploading configuration...")
        api.upload_file(
            path_or_fileobj=clean_config_path,
            path_in_repo="configuration.yaml",
            repo_id=repo_id,
            commit_message=f"{commit_message} - config",
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"Model uploaded successfully: {url}")

    return url


def create_model_card(
    repo_id: str, model_name: str, description: str, usage_example: str
) -> str:
    """
    Create a README.md model card for the HF repository.

    Args:
        repo_id: HF repository ID
        model_name: Human-readable model name
        description: Model description
        usage_example: Example usage code

    Returns:
        README content as string
    """
    readme = f"""---
license: apache-2.0
tags:
- music
- version-identification
- audio
- lyrics
- wealy
---

# {model_name}

{description}

This is a WEALY (WEakly-supervised Audio-LYrics) model for music version identification.

## Usage

```bash
# Download and run inference
python scripts/inference.py \\
    model_name={repo_id} \\
    hidden_states=/path/to/your/hidden-states \\
    partition=test \\
    use_overlapping_chunks=true \\
    ngpus=1
```

### Required Path Arguments

When running inference, you must provide:

- `hidden_states`: Path to pre-extracted hidden states directory

### Optional Arguments

- `partition`: Dataset partition to evaluate (default: "test")
- `use_overlapping_chunks`: Enable overlapping chunk evaluation (default: false)
- `chunk_size`: Size of overlapping chunks (default: 1500)
- `overlap_percentage`: Overlap between chunks (default: 0.9)
- `ngpus`: Number of GPUs to use (default: 1)

## Model Details

{usage_example}

## Citation

If you use this model, please cite:

```bibtex
@article{{mancini2025wealy,
    title={{Leveraging Whisper Embeddings for Audio-based Lyrics Matching}},
    author={{Mancini, Eleonora and Serrà, Joan and Torroni, Paolo and Mitsufuji, Yuki}},
    journal={{arXiv preprint arXiv:2510.08176}},
    year={{2025}},
    url={{https://github.com/helemanc/audio-based-lyrics-matching}}
}}
```
"""
    return readme

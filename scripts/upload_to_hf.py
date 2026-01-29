#!/usr/bin/env python3
"""
Upload trained model checkpoints to Hugging Face Hub.

This script sanitizes the configuration (removes personal paths) before uploading.

Usage (single model):
    python scripts/upload_to_hf.py \
        checkpoint=logs/wealy_whisper_shs/best.ckpt \
        repo_id=username/wealy-whisper-shs

Usage (with auto-detected repo name from local dir):
    python scripts/upload_to_hf.py \
        checkpoint=logs/wealy_whisper_shs/best.ckpt

Usage (batch upload all models):
    python scripts/upload_to_hf.py --batch logs_dir=logs

Arguments:
    checkpoint: Path to the .ckpt file to upload
    repo_id: HF repository ID (optional - auto-detected from checkpoint path)
    logs_dir: Directory containing model folders (for batch mode)
    description: Optional model description for the model card
    private: Set to true for private repository (default: false)
    create_card: Create a README model card (default: true)
    --batch: Enable batch upload mode
    --list: List available model mappings
"""

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import hf_utils


def get_repo_id_from_path(checkpoint_path: str) -> str:
    """
    Derive HF repo ID from checkpoint path using LOCAL_TO_HF_NAME mapping.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Full HF repo ID
    """
    # Get the directory name (e.g., "wealy_whisper_shs")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    local_name = os.path.basename(checkpoint_dir)

    # Look up in mapping
    if local_name in hf_utils.LOCAL_TO_HF_NAME:
        hf_name = hf_utils.LOCAL_TO_HF_NAME[local_name]
        return f"{hf_utils.DEFAULT_HF_ORG}/{hf_name}"

    # Fallback: convert underscores to dashes
    hf_name = local_name.replace("_", "-")
    return f"{hf_utils.DEFAULT_HF_ORG}/{hf_name}"


def find_best_checkpoint(model_dir: Path) -> Path:
    """
    Find the best checkpoint in a model directory.

    Looks for 'checkpoint_best.ckpt', then any .ckpt file.

    Args:
        model_dir: Path to model directory

    Returns:
        Path to checkpoint file
    """
    best_ckpt = model_dir / "checkpoint_best.ckpt"
    if best_ckpt.exists():
        return best_ckpt

    # Find any .ckpt file
    ckpt_files = list(model_dir.glob("*.ckpt"))
    if ckpt_files:
        return ckpt_files[0]

    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def upload_single_model(
    checkpoint_path: str,
    repo_id: str,
    description: str,
    private: bool,
    create_card: bool,
) -> bool:
    """Upload a single model to HF Hub."""
    # Find configuration file
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "configuration.yaml")

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False

    if not os.path.exists(config_path):
        print(f"Error: Configuration not found: {config_path}")
        return False

    print("=" * 70)
    print(f"UPLOADING: {repo_id}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Private: {private}")

    try:
        # Upload model
        url = hf_utils.upload_model_to_hf(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            repo_id=repo_id,
            commit_message="Upload model checkpoint and configuration",
            private=private,
        )

        # Create model card if requested
        if create_card:
            print("Creating model card...")
            from huggingface_hub import HfApi

            # Load config to get model info
            conf = OmegaConf.load(config_path)
            # Normalize model name: whisper-ft -> wealy
            raw_model_name = conf.model.name if hasattr(conf, "model") else "model"
            if raw_model_name in ("whisper-ft", "whisper_ft"):
                model_name = "WEALY"
            else:
                model_name = raw_model_name.upper()

            readme_content = hf_utils.create_model_card(
                repo_id=repo_id,
                model_name=f"{model_name} - {repo_id.split('/')[-1]}",
                description=description,
                usage_example=f"""
This model was trained for version identification using the {model_name} architecture.

### Training Configuration
- Dataset: {getattr(conf.data, "dataset_name", "unknown")}
- Embedding type: {getattr(conf.data, "embedding_type", "unknown")}
- Embedding dimension: {getattr(conf.model, "zdim", "unknown")}
""",
            )

            api = HfApi()
            api.upload_file(
                path_or_fileobj=readme_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message="Add model card",
            )

        print(f"Uploaded: {url}")
        return True

    except Exception as e:
        print(f"Error uploading {repo_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


def batch_upload(logs_dir: str, private: bool, create_card: bool) -> None:
    """
    Upload all models found in logs directory.

    Args:
        logs_dir: Directory containing model folders
        private: Whether repos should be private
        create_card: Whether to create model cards
    """
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        sys.exit(1)

    # Find all model directories that have checkpoints
    model_dirs = []
    for local_name in hf_utils.LOCAL_TO_HF_NAME.keys():
        model_dir = logs_path / local_name
        if model_dir.exists() and (model_dir / "configuration.yaml").exists():
            try:
                ckpt = find_best_checkpoint(model_dir)
                model_dirs.append((local_name, model_dir, ckpt))
            except FileNotFoundError:
                print(f"Skipping {local_name}: no checkpoint found")

    if not model_dirs:
        print("No models found to upload.")
        print("\nExpected directory structure:")
        print(f"  {logs_dir}/")
        for name in hf_utils.LOCAL_TO_HF_NAME.keys():
            print(f"    {name}/")
            print(f"      best.ckpt")
            print(f"      configuration.yaml")
        sys.exit(1)

    print("=" * 70)
    print("BATCH UPLOAD")
    print("=" * 70)
    print(f"Found {len(model_dirs)} models to upload:")
    for local_name, _, ckpt in model_dirs:
        hf_name = hf_utils.LOCAL_TO_HF_NAME[local_name]
        print(f"  - {local_name} -> {hf_utils.DEFAULT_HF_ORG}/{hf_name}")
    print("=" * 70)

    response = input("\nProceed with batch upload? [y/N]: ").strip().lower()
    if response != "y":
        print("Upload cancelled.")
        sys.exit(0)

    # Upload each model
    results = []
    for local_name, model_dir, ckpt in model_dirs:
        hf_name = hf_utils.LOCAL_TO_HF_NAME[local_name]
        repo_id = f"{hf_utils.DEFAULT_HF_ORG}/{hf_name}"

        success = upload_single_model(
            checkpoint_path=str(ckpt),
            repo_id=repo_id,
            description=f"WEALY model: {hf_name}",
            private=private,
            create_card=create_card,
        )
        results.append((local_name, repo_id, success))
        print()

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH UPLOAD SUMMARY")
    print("=" * 70)
    succeeded = sum(1 for _, _, s in results if s)
    failed = len(results) - succeeded
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    print()
    for local_name, repo_id, success in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {local_name} -> {repo_id}")
    print("=" * 70)


def main():
    """Main upload entry point."""
    # Check for special flags
    if "--list" in sys.argv:
        hf_utils.print_available_models()
        print("\nLocal directory -> HF name mapping:")
        print("-" * 50)
        for local, hf in hf_utils.LOCAL_TO_HF_NAME.items():
            print(f"  {local} -> {hf}")
        sys.exit(0)

    args = OmegaConf.from_cli()

    # Check huggingface_hub is available
    if not hf_utils.check_hf_available():
        sys.exit(1)

    # Check HF username is configured
    if hf_utils.DEFAULT_HF_ORG == "YOUR_HF_USERNAME":
        print("Error: Please set your HF username in utils/hf_utils.py")
        print("  Change DEFAULT_HF_ORG = 'YOUR_HF_USERNAME' to your actual username")
        sys.exit(1)

    # Set defaults
    private = getattr(args, "private", False)
    create_card = getattr(args, "create_card", True)

    # Batch mode
    if "--batch" in sys.argv:
        logs_dir = getattr(args, "logs_dir", "logs")
        batch_upload(logs_dir, private, create_card)
        return

    # Single model mode
    if "checkpoint" not in args:
        print("Error: Must provide checkpoint=path/to/checkpoint.ckpt")
        print("  Or use --batch logs_dir=logs for batch upload")
        sys.exit(1)

    checkpoint_path = args.checkpoint
    description = getattr(args, "description", "Audio-based lyrics matching model")

    # Auto-detect repo_id if not provided
    if "repo_id" in args:
        repo_id = args.repo_id
    else:
        repo_id = get_repo_id_from_path(checkpoint_path)
        print(f"Auto-detected repo_id: {repo_id}")

    # Confirm before upload
    print("\nThe configuration will be sanitized (personal paths removed).")
    response = input(f"Upload to {repo_id}? [y/N]: ").strip().lower()
    if response != "y":
        print("Upload cancelled.")
        sys.exit(0)

    success = upload_single_model(
        checkpoint_path=checkpoint_path,
        repo_id=repo_id,
        description=description,
        private=private,
        create_card=create_card,
    )

    if success:
        print("\n" + "=" * 70)
        print("UPLOAD COMPLETE")
        print("=" * 70)
        print(f"\nUsers can download and use this model with:")
        print(f"  python scripts/inference.py \\")
        print(f"      model_name={repo_id} \\")
        print(f"      hidden_states=/path/to/hidden-states \\")
        print(f"      partition=test")
        print("=" * 70)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

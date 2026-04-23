"""
Feature extraction classes for different embedding types.

Provides extractors for:
- Whisper: Encoder and decoder embeddings
- SBERT: Sentence embeddings from transcriptions
- CLEWS: Audio embeddings
- WEALY: Learned lyrics embeddings
"""

import importlib
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from lightning import Fabric
from omegaconf import DictConfig

from utils import print_utils
from utils.clews_utils import (
    auto_setup_clews_paths,
    extract_clews_features_with_shingles,
    get_clews_extraction_params,
    load_clews_model,
    save_clews_averaged_embedding,
    save_clews_embeddings,
)
from utils.latents_extraction_utils import get_save_path_for_dataset, define_save_folders
from utils.clews_utils import (
    auto_setup_clews_paths,
    load_clews_model,
    extract_clews_features_with_shingles,
    save_clews_embeddings,
    save_clews_mask,
    save_clews_averaged_embedding,
    get_clews_extraction_params,
)


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ExtractionStats:
    """Track extraction statistics."""

    total_files: int = 0
    extracted_files: int = 0
    skipped_files: int = 0
    failed_files: int = 0

    def __str__(self) -> str:
        return (
            f"Total: {self.total_files}, "
            f"Extracted: {self.extracted_files}, "
            f"Skipped: {self.skipped_files}, "
            f"Failed: {self.failed_files}"
        )


# ============================================================================
# BASE EXTRACTOR
# ============================================================================


class BaseExtractor(ABC):
    """
    Base class for feature extractors.

    All extractors must implement:
    - load_model(): Load the extraction model
    - extract_features(): Extract features from a batch
    - save_features(): Save features to disk
    """

    def __init__(self, conf: DictConfig, fabric: Fabric, verbose: bool = True):
        """
        Initialize extractor.

        Args:
            conf: Configuration object
            fabric: Lightning Fabric instance
            verbose: Print progress messages
        """
        self.conf = conf
        self.fabric = fabric
        self.verbose = verbose
        self.myprint = lambda s, end="\n": print_utils.myprint(
            s, end=end, doit=fabric.is_global_zero
        )

        # Setup save folders
        if (
            conf.path.transcription_folder is None
            or conf.path.hidden_states_folder is None
        ):
            self.transcription_folder, self.hidden_states_folder = define_save_folders(
                conf.data.dataset_name, conf.path.save_data_path
            )
        else:
            self.transcription_folder = conf.path.transcription_folder
            self.hidden_states_folder = conf.path.hidden_states_folder

        self.stats = ExtractionStats()

    @abstractmethod
    def load_model(self) -> Any:
        """Load extraction model. Returns model object."""
        pass

    @abstractmethod
    def extract_features(
        self, batch: Tuple, model: Any
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract features from batch.

        Args:
            batch: Data batch from dataloader
            model: Loaded model

        Returns:
            List of dicts containing extracted features per sample
        """
        pass

    @abstractmethod
    def save_features(
        self,
        features: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save features to disk.

        Args:
            features: Dict of feature tensors
            save_path: Base path for saving
            metadata: Optional metadata to save with features
        """
        pass

    def should_skip(self, save_path: Path, filename: str) -> bool:
        """
        Check if extraction should be skipped.

        Args:
            save_path: Base save directory
            filename: Feature filename

        Returns:
            True if file exists and skip_existing is enabled
        """
        if not self.conf.extraction.get("skip_existing", True):
            return False

        feature_path = save_path / filename
        return feature_path.exists()


# ============================================================================
# WHISPER EXTRACTOR
# ============================================================================


class WhisperExtractor(BaseExtractor):
    """Extract encoder and decoder embeddings using Whisper."""

    def __init__(self, conf: DictConfig, fabric: Fabric, verbose: bool = True):
        """Initialize Whisper extractor."""
        super().__init__(conf, fabric, verbose)

        # Generate decoding config name (original functionality)
        if (
            hasattr(conf.whisper, "initial_prompt")
            and conf.whisper.initial_prompt is not None
        ):
            config_prompt = "prompt"
        else:
            config_prompt = "nothing"

        if hasattr(conf.whisper, "language") and conf.whisper.language is not None:
            self.decoding_config_name = f"{conf.data.dataset_name}_{config_prompt}_whisper_{conf.seed}_transcribe_{conf.whisper.language}"
        else:
            self.decoding_config_name = f"{conf.data.dataset_name}_{config_prompt}_whisper_{conf.seed}_transcribe"

        self.myprint(f"Decoding config name: {self.decoding_config_name}")

    def load_model(self) -> Any:
        """Load Whisper model."""
        import whisper

        self.myprint(f"Loading Whisper model: {self.conf.whisper.model_name}")
        model = whisper.load_model(self.conf.whisper.model_name)

        # Store device for loading existing tensors
        self.extraction_device = model.device

        self.unwrapped_model = model

        # Setup with Fabric
        model = self.fabric.setup(model)
        model.mark_forward_method("transcribe")

        self.myprint("✓ Whisper model loaded")
        return model

    def _get_tensor_filename(
        self, embedding_type: str, embedding_format: str
    ) -> Optional[str]:
        """Get tensor filename based on embedding type and format."""
        from utils.latents_extraction_utils import get_tensor_filename

        return get_tensor_filename(embedding_type, embedding_format)

    def _should_skip_extraction(
        self, save_path: Path, audio_path: str
    ) -> Tuple[bool, Optional[Path]]:
        """
        Check if extraction should be skipped and return tensor path.

        Returns:
            Tuple of (should_skip, tensor_path)
        """
        if not self.conf.extraction.get("skip_existing", True):
            return False, None

        tensor_filename = self._get_tensor_filename(
            self.conf.data.embedding_type, self.conf.data.embedding_format
        )

        if not tensor_filename:
            return False, None

        tensor_path = save_path / tensor_filename

        if tensor_path.exists():
            if self.fabric.is_global_zero:
                self.myprint(f"Skipping extraction - tensor exists: {tensor_path.name}")
            return True, tensor_path

        return False, None

    def _load_existing_embeddings(
        self, tensor_path: Path, embedding_type: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load existing embeddings for evaluation when skipping extraction.

        Returns:
            Tuple of (encoder_embedding, decoder_embedding)
        """
        try:
            existing_tensor = torch.load(
                tensor_path, map_location=self.extraction_device
            )

            if embedding_type == "encoder":
                # Only encoder available
                lyric_feature_e = (
                    existing_tensor.mean(dim=0)
                    if existing_tensor.dim() > 1
                    else existing_tensor
                )
                lyric_feature_d = torch.zeros_like(lyric_feature_e)

            elif embedding_type in [
                "last_hidden_states",
                "last_hidden_states_en",
                "hidden_states",
            ]:
                # Only decoder available
                lyric_feature_d = (
                    existing_tensor.mean(dim=0)
                    if existing_tensor.dim() > 1
                    else existing_tensor
                )
                lyric_feature_e = torch.zeros_like(lyric_feature_d)

            elif embedding_type == "evaluation_enc_dec":
                # Both encoder and decoder needed
                lyric_feature_d = (
                    existing_tensor.mean(dim=0)
                    if existing_tensor.dim() > 1
                    else existing_tensor
                )

                # Try to load encoder embeddings too
                encoder_filename = self._get_tensor_filename(
                    "encoder", self.conf.data.embedding_format
                )
                encoder_path = (
                    tensor_path.parent / encoder_filename if encoder_filename else None
                )

                if encoder_path and encoder_path.exists():
                    encoder_tensor = torch.load(
                        encoder_path, map_location=self.extraction_device
                    )
                    lyric_feature_e = (
                        encoder_tensor.mean(dim=0)
                        if encoder_tensor.dim() > 1
                        else encoder_tensor
                    )
                else:
                    lyric_feature_e = torch.zeros_like(lyric_feature_d)
            else:
                # Unknown type
                return None, None

            return lyric_feature_e, lyric_feature_d

        except Exception as e:
            if self.fabric.is_global_zero:
                self.myprint(f"Failed to load existing tensor: {e}")
            return None, None

    def extract_features(
        self, batch: Tuple, model: Any
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract Whisper embeddings.

        Handles:
        - Skipping existing extractions
        - Loading existing embeddings for evaluation
        - Different embedding types (encoder, decoder, evaluation_enc_dec)
        """
        c, i, waveforms, _, _, _, _, audio_paths = batch

        features_list = []

        for j, (waveform, audio_path) in enumerate(zip(waveforms, audio_paths)):
            # Get save path
            from utils.latents_extraction_utils import (
                extract_path_info_for_dataset,
                get_save_path_for_dataset,
            )

            clique_id, version_id, save_components = extract_path_info_for_dataset(
                audio_path, self.conf.data.dataset_name
            )

            save_base_path = get_save_path_for_dataset(
                self.hidden_states_folder,
                self.conf.data.dataset_name,
                clique_id,
                version_id,
                save_components,
            )

            should_skip, tensor_path = self._should_skip_extraction(
                save_base_path, audio_path
            )

            if should_skip:
                # Always increment skipped counter
                self.stats.skipped_files += 1

                if self.fabric.is_global_zero:
                    self.myprint(
                        f"Skipping extraction - tensor exists: {tensor_path.name if tensor_path else 'N/A'}"
                    )

                # Optionally load existing embeddings for evaluation
                if self.conf.evaluation.get("run_evaluation", False) and tensor_path:
                    lyric_feature_e, lyric_feature_d = self._load_existing_embeddings(
                        tensor_path, self.conf.data.embedding_type
                    )

                    if lyric_feature_e is not None and lyric_feature_d is not None:
                        features_list.append(
                            {
                                "encoder_embedding": lyric_feature_e,
                                "decoder_embedding": lyric_feature_d,
                                "audio_path": audio_path,
                                "skipped": True,
                            }
                        )

                # FIX: Continue here, OUTSIDE the run_evaluation check
                # This ensures we skip regardless of evaluation mode
                continue

            # ====================================================================
            # Extraction code (only reached if should_skip is False)
            # ====================================================================

            self.stats.extracted_files += 1

            if self.fabric.is_global_zero:
                self.myprint(f"Extracting features for {save_components}")

            with torch.no_grad():
                # Configure transcription
                transcribe_kwargs = {"initial_prompt": self.conf.whisper.initial_prompt}

                if (
                    hasattr(self.conf.whisper, "language")
                    and self.conf.whisper.language
                ):
                    transcribe_kwargs["language"] = self.conf.whisper.language
                if hasattr(self.conf.whisper, "carry_initial_prompt"):
                    transcribe_kwargs["carry_initial_prompt"] = (
                        self.conf.whisper.carry_initial_prompt
                    )
                if hasattr(self.conf.whisper, "condition_on_previous_text"):
                    transcribe_kwargs["condition_on_previous_text"] = (
                        self.conf.whisper.condition_on_previous_text
                    )

                # Properly format audio for Whisper
                waveform_clean = waveform.squeeze()
                if waveform_clean.dim() > 1:
                    waveform_clean = waveform_clean[0]
                waveform_np = waveform_clean.cpu().numpy().astype("float32")

                # Use unwrapped model to avoid DDP wrapper issues
                transcribe_model = getattr(self, "unwrapped_model", model)
                result = transcribe_model.transcribe(waveform_np, **transcribe_kwargs)

            # Process encoder embeddings
            frames_audio_features = result["frames_audio_features"]
            encoder_embeddings = torch.cat(frames_audio_features, dim=0)
            lyric_feature_e = encoder_embeddings.mean(dim=0)

            # Process decoder embeddings
            frames_hidden_states = result["frames_last_hidden_states"]
            all_final_states = []
            for chunk_states in frames_hidden_states:
                all_final_states.extend(chunk_states)

            lyric_feature_seq = torch.cat(all_final_states)
            lyric_feature_d = lyric_feature_seq.mean(dim=0)

            features_list.append(
                {
                    "encoder_embedding": lyric_feature_e,
                    "decoder_embedding": lyric_feature_d,
                    "encoder_sequence": encoder_embeddings,
                    "decoder_sequence": lyric_feature_seq,
                    "transcription": result.get("text", ""),
                    "audio_path": audio_path,
                    "result": result,
                    "frames_audio_features": frames_audio_features,
                    "save_base_path": save_base_path,
                    "save_components": save_components,
                    "skipped": False,
                }
            )

        return features_list

    def save_features(
        self,
        features: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save Whisper embeddings with decoding config name."""
        if features.get("skipped", False):
            # Already exists, skip saving
            return

        save_path.mkdir(parents=True, exist_ok=True)

        from utils.latents_extraction_utils import save_transcription_and_latents

        language = self.conf.whisper.get("language")

        # Use original save function with decoding_config_name
        save_transcription_and_latents(
            self.conf.data.dataset_name,
            features["result"],
            features["frames_audio_features"],
            features["decoder_sequence"],
            self.transcription_folder,
            self.hidden_states_folder,
            save_path,
            metadata.get("save_components")
            if metadata
            else features.get("save_components"),
            self.decoding_config_name,  # IMPORTANT: Pass decoding config name
            save_transcription=self.conf.extraction.get("save_transcription", True),
            save_encoder_embeddings=self.conf.extraction.get(
                "save_encoder_embeddings", True
            ),
            save_encoder_embeddings_seq=self.conf.extraction.get(
                "save_encoder_embeddings_seq", False
            ),
            save_hidden_states=self.conf.extraction.get("save_hidden_states", False),
            save_last_hidden_states=self.conf.extraction.get(
                "save_last_hidden_states", True
            ),
            save_last_hidden_states_seq=self.conf.extraction.get(
                "save_last_hidden_states_seq", False
            ),
            language=language,
        )


# ============================================================================
# SBERT EXTRACTOR
# ============================================================================


class SBERTExtractor(BaseExtractor):
    """Extract sentence embeddings using SBERT."""

    def __init__(self, conf: DictConfig, fabric: Fabric, verbose: bool = True):
        """Initialize SBERT extractor."""
        super().__init__(conf, fabric, verbose)

        # Generate decoding config name (for finding transcriptions)
        if hasattr(conf, "whisper"):
            if (
                hasattr(conf.whisper, "initial_prompt")
                and conf.whisper.initial_prompt is not None
            ):
                config_prompt = "prompt"
            else:
                config_prompt = "nothing"

            if hasattr(conf.whisper, "language") and conf.whisper.language is not None:
                self.decoding_config_name = f"{conf.data.dataset_name}_{config_prompt}_whisper_{conf.seed}_transcribe_{conf.whisper.language}"
            else:
                self.decoding_config_name = f"{conf.data.dataset_name}_{config_prompt}_whisper_{conf.seed}_transcribe"
        else:
            # Default if whisper config not present
            self.decoding_config_name = (
                f"{conf.data.dataset_name}_prompt_whisper_{conf.seed}_transcribe"
            )

        self.myprint(f"Using decoding config name: {self.decoding_config_name}")

    def load_model(self) -> Any:
        """Load SBERT model."""
        from sentence_transformers import SentenceTransformer

        model_name = self.conf.sbert.get("model_name", "all-MiniLM-L6-v2")
        self.myprint(f"Loading SBERT model: {model_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SentenceTransformer(model_name, device=device)

        # Store for later use
        self.extraction_device = device

        self.myprint("✓ SBERT model loaded")
        return model

    def _preprocess_transcription(self, transcription: str) -> str:
        """
        Preprocess transcription for SBERT encoding.

        Removes:
        - Timestamps [mm:ss]
        - Annotations (parenthetical and bracketed)
        - Excessive filler words (um, uh, ah, etc.)
        - Extra punctuation
        """
        if not transcription or not isinstance(transcription, str):
            return ""

        import re

        # Convert to lowercase
        text = transcription.lower()

        # Remove common transcription artifacts
        text = re.sub(r"\[\d+:\d+\]", "", text)  # Remove [mm:ss] timestamps
        text = re.sub(r"\(.*?\)", "", text)  # Remove parenthetical annotations
        text = re.sub(r"\[.*?\]", "", text)  # Remove bracketed annotations

        # Remove excessive filler words
        excessive_fillers = r"\b(um|uh|ah|hmm|er|eh)\b"
        text = re.sub(excessive_fillers, " ", text)

        # Clean up punctuation but preserve apostrophes
        text = re.sub(r"[^\w\s']", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def _find_transcription_file(
        self, version_id: str, dataset_name: str, transcription_folder: Path
    ) -> Optional[Path]:
        """
        Find transcription file following Whisper naming convention.

        Args:
            version_id: Version ID
            dataset_name: Dataset name
            transcription_folder: Base transcription folder

        Returns:
            Path to transcription file or None
        """
        if dataset_name == "shs":
            # For SHS: transcriptions/{set_folder}/{set_id-ver_id}/{decoding_config_name}.txt
            if "-" in version_id:
                set_id, ver_id = version_id.split("-", 1)

                # Try different folder structures
                set_id_int = int(set_id) if set_id.isdigit() else 0

                if set_id_int <= 9:
                    possible_set_folders = [f"{set_id}-", set_id]
                elif set_id_int <= 99:
                    possible_set_folders = [set_id]
                else:
                    folder_prefix = set_id[:2]
                    possible_set_folders = [folder_prefix, set_id]

                possible_paths = []
                for set_folder in possible_set_folders:
                    possible_paths.append(
                        transcription_folder
                        / "transcriptions"
                        / set_folder
                        / version_id
                        / f"{self.decoding_config_name}.txt"
                    )
            else:
                return None

        elif dataset_name == "lyric-covers":
            # For Lyric Covers: transcriptions/{version_id}/{decoding_config_name}.txt
            possible_paths = [
                transcription_folder
                / "transcriptions"
                / version_id
                / f"{self.decoding_config_name}.txt"
            ]

        elif dataset_name == "discogs-vi":
            # For Discogs-VI: transcriptions/{dir_name}/{filename}/{decoding_config_name}.txt
            if "/" in version_id:
                parts = version_id.split("/")
                possible_paths = [
                    transcription_folder
                    / "transcriptions"
                    / parts[0]
                    / parts[1]
                    / f"{self.decoding_config_name}.txt"
                ]
            else:
                possible_paths = [
                    transcription_folder
                    / "transcriptions"
                    / version_id
                    / f"{self.decoding_config_name}.txt"
                ]
        else:
            return None

        # Check which path exists
        for path in possible_paths:
            if path.exists():
                return path

        return None

    def _load_transcription(self, transcription_path: Path) -> str:
        """Load transcription from file."""
        try:
            with open(transcription_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            if self.fabric.is_global_zero:
                self.myprint(
                    f"Error loading transcription from {transcription_path}: {e}"
                )
            return ""

    def extract_features(
        self, batch: Tuple, model: Any
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract SBERT embeddings from transcriptions.

        Process:
        1. Find transcription file
        2. Load and preprocess transcription
        3. Encode with SBERT
        4. Handle empty transcriptions with zero embeddings
        """
        c, i, waveforms, texts, valid_flags, _, _, audio_paths = batch

        features_list = []

        # Get embedding dimension for zero embeddings
        embedding_dim = model.get_sentence_embedding_dimension()

        for j, audio_path in enumerate(audio_paths):
            # Get save path info
            from utils.latents_extraction_utils import (
                extract_path_info_for_dataset,
                get_save_path_for_dataset,
            )

            clique_id, version_id, save_components = extract_path_info_for_dataset(
                audio_path, self.conf.data.dataset_name
            )

            save_base_path = get_save_path_for_dataset(
                self.hidden_states_folder,
                self.conf.data.dataset_name,
                clique_id,
                version_id,
                save_components,
            )

            # Check if SBERT embedding already exists
            sbert_path = save_base_path / "hs_sbert.pt"
            if self.conf.extraction.get("skip_existing", True) and sbert_path.exists():
                if self.fabric.is_global_zero:
                    self.myprint(f"Skipping - SBERT embedding exists: {version_id}")

                # Load for evaluation if needed
                if self.conf.evaluation.get("run_evaluation", False):
                    try:
                        sbert_embedding = torch.load(
                            sbert_path, map_location=self.extraction_device
                        )
                        features_list.append(
                            {
                                "sbert_embedding": sbert_embedding,
                                "audio_path": audio_path,
                                "version_id": version_id,
                                "skipped": True,
                            }
                        )
                        self.stats.skipped_files += 1
                    except Exception as e:
                        if self.fabric.is_global_zero:
                            self.myprint(
                                f"Failed to load existing SBERT embedding: {e}"
                            )
                continue

            # Find transcription file
            transcription_path = self._find_transcription_file(
                version_id, self.conf.data.dataset_name, self.transcription_folder
            )

            if not transcription_path:
                if self.fabric.is_global_zero:
                    self.myprint(f"✗ Transcription not found for {version_id}")
                self.stats.failed_files += 1
                continue

            # Load transcription
            transcription = self._load_transcription(transcription_path)

            # Preprocess transcription
            processed_transcription = self._preprocess_transcription(transcription)

            # Encode transcription
            if processed_transcription.strip():
                with torch.no_grad():
                    sbert_embedding = model.encode(
                        [processed_transcription],
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        device=self.extraction_device,
                    ).squeeze(0)  # Remove batch dimension
            else:
                # Handle empty transcription - create zero embedding
                sbert_embedding = torch.zeros(
                    embedding_dim, device=self.extraction_device
                )
                if self.fabric.is_global_zero:
                    self.myprint(
                        f"Warning: Empty transcription for {version_id}, using zero embedding"
                    )

            self.stats.extracted_files += 1

            features_list.append(
                {
                    "sbert_embedding": sbert_embedding,
                    "audio_path": audio_path,
                    "version_id": version_id,
                    "save_base_path": save_base_path,
                    "skipped": False,
                }
            )

        return features_list

    def save_features(
        self,
        features: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save SBERT embeddings with decoding config name."""
        if features.get("skipped", False):
            return

        from utils.latents_extraction_utils import save_sbert_embeddings

        save_sbert_embeddings(
            features["sbert_embedding"],
            features.get("save_base_path", save_path),
            self.decoding_config_name,  # IMPORTANT: Pass decoding config name
        )


# ============================================================================
# WEALY EXTRACTOR
# ============================================================================


class WEALYExtractor(BaseExtractor):
    """Extract WEALY concatenated embeddings using overlapping chunks."""

    def __init__(self, conf: DictConfig, fabric: Fabric, verbose: bool = True):
        """Initialize WEALY extractor."""
        super().__init__(conf, fabric, verbose)
        self.dataset_ref = None  # Store dataset for version key mapping
        self.debug_count = 0

    def load_model(self) -> Any:
        """Load WEALY model from checkpoint."""
        import importlib

        self.myprint("Loading WEALY model...")

        # Load model architecture
        module = importlib.import_module("models." + self.conf.model.name)
        with self.fabric.init_module():
            model = module.Model(
                self.conf.model,
                use_avg_pooling=self.conf.data.get("use_avg_pooling", False),
                embedding_type=self.conf.data.embedding_type,
                sr=self.conf.data.samplerate,
            )

        model = self.fabric.setup(model)
        model.mark_forward_method("prepare")
        model.mark_forward_method("embed")

        # Load checkpoint
        checkpoint_path = self.conf.wealy.checkpoint_path
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"WEALY checkpoint not found: {checkpoint_path}")

        self.myprint(f"Loading checkpoint: {checkpoint_path}")

        try:
            state = {}
            self.fabric.load(checkpoint_path, state, strict=False)

            if "model" in state:
                model = state["model"]
            else:
                state = {"model": model}
                self.fabric.load(checkpoint_path, state, strict=False)
                model = state["model"]

        except Exception as e:
            self.myprint(f"Loading failed: {e}, trying direct torch.load...")

            if self.fabric.is_global_zero:
                raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
                model_state = raw_checkpoint.get("model", raw_checkpoint)

                if hasattr(model, "_forward_module"):
                    model._forward_module.load_state_dict(model_state, strict=False)
                else:
                    model.load_state_dict(model_state, strict=False)

            self.fabric.barrier()

        model.eval()
        self.myprint("✓ WEALY model loaded")
        return model

    def _get_version_save_path(
        self, version_key: str, dataset_name: str, hidden_states_folder: Path
    ) -> Optional[Path]:
        """
        Get save path for a version key using SHS folder logic.

        This follows the same folder structure as audio verification.
        """
        from utils.latents_extraction_utils import get_save_path_for_dataset

        if dataset_name == "shs":
            if "-" not in version_key:
                return None

            set_id, ver_id = version_key.split("-", 1)
            clique_id_extracted = set_id
            version_id_extracted = version_key

            # Use same folder logic as _audio_exists_shs
            set_id_int = int(set_id) if set_id.isdigit() else 0
            if set_id_int <= 9:
                save_components = (f"{set_id}-", version_key)
            elif set_id_int <= 99:
                save_components = (set_id, version_key)
            else:
                folder_prefix = set_id[:2]
                save_components = (folder_prefix, version_key)

        elif dataset_name == "lyric-covers":
            clique_id_extracted = (
                version_key.split("-")[0] if "-" in version_key else version_key
            )
            version_id_extracted = version_key
            save_components = version_key

        elif dataset_name == "discogs-vi":
            clique_id_extracted = (
                version_key.split("-")[0] if "-" in version_key else version_key
            )
            version_id_extracted = version_key
            save_components = version_key.replace("/", os.sep)

        else:
            # Generic fallback
            clique_id_extracted = (
                version_key.split("-")[0] if "-" in version_key else version_key
            )
            version_id_extracted = version_key
            save_components = version_key

        return get_save_path_for_dataset(
            hidden_states_folder,
            dataset_name,
            clique_id_extracted,
            version_id_extracted,
            save_components,
        )

    def extract_features(
        self, batch: Tuple, model: Any
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract WEALY embeddings from overlapping chunks.

        Process:
        1. Group chunks by song using version key mapping
        2. Process through WEALY model
        3. Concatenate chunks for each song
        4. Save with detailed metadata
        """
        clique_ids, version_ids, embeddings, masks, chunk_info = batch

        features_list = []

        # Group chunks by song using dataset reference for version key lookup
        songs_data = {}

        for i in range(len(clique_ids)):
            # Map deterministic ID back to actual version key using dataset
            deterministic_id = version_ids[i].item()
            version_key = None

            # Find the actual version key from dataset info
            if self.dataset_ref:
                for vkey, vdata in self.dataset_ref.info.items():
                    if vdata["id"] == deterministic_id:
                        version_key = vkey
                        break

            if not version_key:
                if self.fabric.is_global_zero:
                    self.myprint(
                        f"Warning: Could not find version key for ID {deterministic_id}"
                    )
                continue

            # Initialize song data if needed
            if version_key not in songs_data:
                songs_data[version_key] = {
                    "clique_id": clique_ids[i].item(),
                    "deterministic_id": deterministic_id,
                    "chunks": [],
                    "masks": [],
                    "chunk_indices": [],
                }

            songs_data[version_key]["chunks"].append(embeddings[i])
            songs_data[version_key]["masks"].append(masks[i])
            songs_data[version_key]["chunk_indices"].append(i)

        # Process each song separately
        for version_key, song_data in songs_data.items():
            # Get save path
            save_base_path = self._get_version_save_path(
                version_key, self.conf.data.dataset_name, self.hidden_states_folder
            )

            if not save_base_path:
                if self.fabric.is_global_zero:
                    self.myprint(
                        f"Warning: Could not determine save path for {version_key}"
                    )
                continue

            # Check if already exists
            wealy_path = save_base_path / "hs_wealy_concat.pt"
            if self.conf.extraction.get("skip_existing", True) and wealy_path.exists():
                self.stats.skipped_files += 1
                continue

            self.stats.extracted_files += 1

            # Stack all chunks for this song
            song_embeddings = torch.stack(
                song_data["chunks"]
            )  # (num_chunks, chunk_size, embed_dim)
            song_masks = torch.stack(song_data["masks"])  # (num_chunks, chunk_size)

            # DEBUG: Print dimensions for first 5 songs
            if self.debug_count < 5 and self.fabric.is_global_zero:
                self.myprint(
                    f"\n=== DEBUG Song {self.debug_count + 1}: {version_key} ==="
                )
                self.myprint(f"  Input chunks shape: {song_embeddings.shape}")
                self.myprint(f"  Input masks shape: {song_masks.shape}")
                self.myprint(f"  Number of chunks: {len(song_embeddings)}")
                self.myprint(
                    f"  Version key mapping: ID {song_data['deterministic_id']} -> {version_key}"
                )
                self.myprint(f"  Save path: {save_base_path}")
                self.debug_count += 1

            with torch.no_grad():
                # Process through WEALY
                song_embeddings = song_embeddings.to(self.fabric.device)
                song_masks = song_masks.to(self.fabric.device)

                prepared_embeddings = model.prepare(song_embeddings)
                wealy_chunks, _ = model.embed(prepared_embeddings, song_masks)

                # Move to CPU and concatenate
                concatenated_wealy = wealy_chunks.cpu()  # (num_chunks, wealy_dim)

                # DEBUG: Print output dimensions
                if self.debug_count <= 5 and self.fabric.is_global_zero:
                    self.myprint(f"  WEALY chunks shape: {wealy_chunks.shape}")
                    self.myprint(f"  Concatenated shape: {concatenated_wealy.shape}")
                    self.myprint(
                        f"  WEALY dimension per chunk: {concatenated_wealy.shape[1]}"
                    )
                    self.myprint(
                        f"  Total chunks concatenated: {concatenated_wealy.shape[0]}"
                    )

                # Prepare detailed metadata
                chunk_metadata = {
                    "chunk_size": self.conf.data.get("chunk_size", 1500),
                    "overlap_percentage": self.conf.extraction.get(
                        "overlap_percentage", 0.9
                    ),
                    "total_chunks": len(concatenated_wealy),
                    "wealy_dim": concatenated_wealy.shape[1],
                    "version_key": version_key,
                    "deterministic_id": song_data["deterministic_id"],
                    "clique_id": song_data["clique_id"],
                }

                features_list.append(
                    {
                        "wealy_embedding": concatenated_wealy,
                        "version_key": version_key,
                        "save_base_path": save_base_path,
                        "chunk_metadata": chunk_metadata,
                    }
                )

        return features_list

    def save_features(
        self,
        features: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save WEALY concatenated embeddings with detailed metadata."""
        save_base_path = features.get("save_base_path", save_path)
        save_base_path.mkdir(parents=True, exist_ok=True)

        language = (
            self.conf.wealy.get("language") if hasattr(self.conf, "wealy") else None
        )

        if language is None:
            wealy_path = save_base_path / "hs_wealy_concat.pt"
        else:
            wealy_path = save_base_path / f"hs_wealy_concat_{language}.pt"

        if not wealy_path.exists():
            save_data = {
                "embeddings": features["wealy_embedding"].half(),
                "chunk_info": features.get("chunk_metadata", {}),
                "extraction_method": "overlapping_chunks_via_collate_with_dataset_ref",
            }
            torch.save(save_data, wealy_path)


# ============================================================================
# CLEWS EXTRACTOR
# ============================================================================


class CLEWSExtractor(BaseExtractor):
    """
    Extract CLEWS audio embeddings using shingle-based extraction.

    CLEWS (Contrastive Learning of Musical Embeddings with Weak Supervision)
    extracts audio embeddings using a CNN-based architecture with shingle
    (overlapping window) processing.

    Requirements:
        - CLEWS project must be installed separately (see README)
        - Requires separate Python environment due to dependency conflicts
        - CLEWS checkpoints are auto-downloaded from Zenodo

    Outputs (per audio file):
        - hs_clews.pt: Full shingle embeddings (num_shingles, embedding_dim)
        - hs_clews_mask.pt: Mask for invalid shingles (num_shingles,)
        - hs_clews_avg.pt: Averaged embedding with mask pooling (embedding_dim,)
    """

    def __init__(self, conf: DictConfig, fabric: Fabric, verbose: bool = True):
        """Initialize CLEWS extractor."""
        super().__init__(conf, fabric, verbose)

        self.clews_conf = None
        self.extraction_params = None

        # Validate CLEWS configuration
        if not hasattr(conf, "clews"):
            raise ValueError(
                "CLEWS configuration not found. Add 'clews' section to config with:\n"
                "  - project_dir: Path to CLEWS repository\n"
                "  - config_path: Path to CLEWS config (optional, auto-download)\n"
                "  - checkpoint_path: Path to CLEWS checkpoint (optional, auto-download)"
            )

        if not hasattr(conf.clews, "project_dir") or not conf.clews.project_dir:
            raise ValueError(
                "CLEWS project_dir not specified. Clone CLEWS repository:\n"
                "  git clone https://github.com/helemanc/clews.git\n"
                "And set clews.project_dir in your config."
            )

    def load_model(self) -> Any:
        """
        Load CLEWS model from checkpoint.

        Auto-downloads checkpoint from Zenodo if not found locally.
        Adds CLEWS project to Python path for model imports.

        Returns:
            Loaded CLEWS model ready for inference
        """
        self.myprint("Loading CLEWS model...")

        # Add CLEWS project to path
        clews_project_dir = self.conf.clews.project_dir
        if clews_project_dir not in sys.path:
            sys.path.insert(0, clews_project_dir)

        # Auto-setup paths (download from Zenodo if needed)
        config_path, checkpoint_path = auto_setup_clews_paths(self.conf)

        self.myprint(f"  Config: {config_path}")
        self.myprint(f"  Checkpoint: {checkpoint_path}")

        # Load model
        model, self.clews_conf = load_clews_model(
            self.conf, config_path, checkpoint_path, self.fabric
        )

        # Get extraction parameters
        maxlen_seconds = self.conf.get("maxlen", 600)  # 10 minutes default
        shingle_hop = self.conf.get("qshop", 5)  # 5 second hop default

        self.extraction_params = get_clews_extraction_params(
            model,
            maxlen_seconds=maxlen_seconds,
            shingle_len=self.conf.get("qslen", None),
            shingle_hop=shingle_hop,
        )

        self.myprint(f"  Max length: {self.extraction_params['maxlen_seconds']}s")
        self.myprint(f"  Shingle length: {self.extraction_params['shingle_len']}s")
        self.myprint(f"  Shingle hop: {self.extraction_params['shingle_hop']}s")
        self.myprint(f"  Num shingles: {self.extraction_params['num_shingles']}")
        self.myprint("✓ CLEWS model loaded")

        return model

    def extract_features(
        self, batch: Tuple, model: Any
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Extract CLEWS embeddings from audio batch.

        Uses shingle-based extraction matching the distance matrix computation
        method from the original CLEWS codebase.

        Args:
            batch: Tuple containing (clique_ids, item_ids, audio_tensors, ..., audio_paths)
                   Format depends on CLEWS dataloader
            model: Loaded CLEWS model

        Returns:
            List of feature dicts, each containing:
                - clews_embeddings: (num_shingles, embedding_dim) tensor
                - clews_mask: (num_shingles,) boolean tensor
                - clews_avg: (embedding_dim,) averaged embedding
                - audio_path: Source audio path
                - save_base_path: Destination save path
        """
        features_list = []

        # Handle CLEWS batch format: (clique_id, item_id1, audio1, item_id2, audio2, ..., audio_paths)
        # The batch structure from CLEWS dataset is specific
        batch_size = len(batch[0]) if isinstance(batch[0], torch.Tensor) else 1

        for batch_idx in range(batch_size):
            # Extract clique_id
            if isinstance(batch[0], torch.Tensor):
                clique_id = batch[0][batch_idx].item()
            else:
                clique_id = batch[0]

            # Extract audio tensors and item_ids
            # CLEWS batch format: (clique_id, id1, audio1, id2, audio2, ..., audio_paths)
            audio_tensors = []
            item_ids = []
            for i in range(1, len(batch) - 1, 2):
                if isinstance(batch[i], torch.Tensor):
                    item_ids.append(batch[i][batch_idx].item())
                    audio_tensors.append(batch[i + 1][batch_idx])
                else:
                    item_ids.append(batch[i])
                    audio_tensors.append(batch[i + 1])

            # Extract audio paths
            if isinstance(batch[-1], (list, tuple)):
                audio_paths = (
                    batch[-1][batch_idx] if len(batch[-1]) > batch_idx else batch[-1][0]
                )
            else:
                audio_paths = batch[-1]

            # Process each audio in the clique
            for j, (item_id, audio_tensor) in enumerate(zip(item_ids, audio_tensors)):
                self.stats.total_files += 1

                # Get audio path
                if isinstance(audio_paths, (list, tuple)):
                    audio_path = (
                        audio_paths[j] if j < len(audio_paths) else audio_paths[0]
                    )
                else:
                    audio_path = audio_paths

                # Get save path using existing utilities
                clique_id_extracted, version_id, save_components = (
                    extract_path_info_for_dataset(
                        audio_path, self.conf.data.dataset_name
                    )
                )

                save_base_path = get_save_path_for_dataset(
                    self.hidden_states_folder,
                    self.conf.data.dataset_name,
                    clique_id_extracted,
                    version_id,
                    save_components,
                )

                # Check if should skip
                clews_path = save_base_path / "hs_clews.pt"
                clews_avg_path = save_base_path / "hs_clews_avg.pt"
                clews_mask_path = save_base_path / "hs_clews_mask.pt"

                if self.conf.extraction.get("skip_existing", True):
                    if (
                        clews_path.exists()
                        and clews_avg_path.exists()
                        and clews_mask_path.exists()
                    ):
                        self.stats.skipped_files += 1
                        if self.fabric.is_global_zero:
                            self.myprint(
                                f"Skipping - CLEWS embeddings exist: {version_id}"
                            )
                        continue

                # Extract features
                self.stats.extracted_files += 1

                if self.fabric.is_global_zero:
                    self.myprint(f"Extracting CLEWS features for {save_components}")

                # Move audio to device
                audio_tensor = audio_tensor.to(self.fabric.device)

                # Extract using shingle-based method
                embeddings, mask = extract_clews_features_with_shingles(
                    model,
                    audio_tensor,
                    maxlen=self.extraction_params["maxlen_samples"],
                    shingle_len=self.extraction_params["shingle_len"],
                    shingle_hop=self.extraction_params["shingle_hop"],
                )

                # Move to CPU
                embeddings = embeddings.cpu()
                mask = mask.cpu()

                # Compute averaged embedding with mask
                valid_mask = ~mask
                if valid_mask.sum() > 0:
                    masked_embeddings = embeddings * valid_mask.float().unsqueeze(-1)
                    avg_embedding = (
                        masked_embeddings.sum(dim=0) / valid_mask.float().sum()
                    )
                else:
                    avg_embedding = embeddings.mean(dim=0)

                if self.fabric.is_global_zero:
                    self.myprint(
                        f"  Shape: {embeddings.shape}, Valid shingles: {valid_mask.sum().item()}/{mask.size(0)}"
                    )

                features_list.append(
                    {
                        "clews_embeddings": embeddings,
                        "clews_mask": mask,
                        "clews_avg": avg_embedding,
                        "audio_path": audio_path,
                        "version_id": version_id,
                        "save_base_path": save_base_path,
                        "skipped": False,
                    }
                )

        return features_list

    def save_features(
        self,
        features: Dict[str, torch.Tensor],
        save_path: Path,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save CLEWS embeddings to disk.

        Saves three files:
            - hs_clews.pt: Full shingle embeddings (half precision)
            - hs_clews_mask.pt: Boolean mask for invalid shingles
            - hs_clews_avg.pt: Averaged embedding (half precision)

        Args:
            features: Dict with 'clews_embeddings', 'clews_mask', 'clews_avg'
            save_path: Base directory (uses features['save_base_path'] if available)
            metadata: Optional metadata (unused)
        """
        if features.get("skipped", False):
            return

        save_base_path = features.get("save_base_path", save_path)
        language = (
            self.conf.clews.get("language")
            if hasattr(self.conf.clews, "language")
            else None
        )

        # Save full embeddings
        save_clews_embeddings(
            features["clews_embeddings"], save_base_path, language=language
        )

        # Save mask
        save_clews_mask(features["clews_mask"], save_base_path, language=language)

        # Save averaged embedding
        save_clews_averaged_embedding(
            features["clews_embeddings"].half(),
            features["clews_mask"],
            save_base_path,
            language=language,
        )


# ============================================================================
# EXTRACTOR FACTORY
# ============================================================================


def create_extractor(
    extraction_type: str, conf: DictConfig, fabric: Fabric
) -> BaseExtractor:
    """
    Create appropriate extractor based on type.

    Args:
        extraction_type: 'whisper', 'sbert', 'clews', or 'wealy'
        conf: Configuration object
        fabric: Lightning Fabric instance

    Returns:
        Extractor instance

    Raises:
        ValueError: If extraction_type not recognized

    Example:
        >>> extractor = create_extractor('whisper', conf, fabric)
        >>> model = extractor.load_model()
        >>> features = extractor.extract_features(batch, model)
    """
    extractors = {
        "whisper": WhisperExtractor,
        "sbert": SBERTExtractor,
        "wealy": WEALYExtractor,
        "clews": CLEWSExtractor,
        'sbert': SBERTExtractor,
        'wealy': WEALYExtractor,
        'clews': CLEWSExtractor,
    }

    if extraction_type not in extractors:
        raise ValueError(
            f"Unknown extraction type: {extraction_type}. "
            f"Available: {list(extractors.keys())}"
        )

    return extractors[extraction_type](conf, fabric)

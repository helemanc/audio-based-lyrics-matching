# Leveraging Whisper Embeddings for Audio-based Lyrics Matching  
[![Status](https://img.shields.io/badge/status-preparing%20release-orange)](https://arxiv.org/abs/2510.08176v1)

This repository accompanies the paper:  
**_Leveraging Whisper Embeddings for Audio-based Lyrics Matching_**  
by *Eleonora Mancini, Joan Serrà, Paolo Torroni, and Yuki Mitsufuji*  
[[📄 Read the paper on arXiv](https://arxiv.org/abs/2510.08176v1)]

---

<div align="center" style="border: 2px solid #f39c12; border-radius: 10px; padding: 15px; background-color: #fff8e6;">

🎵  
<h3><b>Repository Under Active Development</b></h3>  
<p>This repository is being actively developed and tested.<br>
Core functionality for <b>WEALY</b> (feature extraction and unimodal training) is now available.<br>
Additional features are being added incrementally.</p>
<p>See <a href="#work-in-progress">Work in Progress</a> section for upcoming features.</p>

</div>

---

## 🧠 About the Project  

This project introduces **WEALY** — **W**hisper **E**mbeddings for **A**udio-based **LY**rics matching — a fully reproducible pipeline that leverages Whisper decoder embeddings for **audio-based lyrics matching**.

**WEALY** establishes transparent and reproducible baselines for version identification using:
- Pre-extracted Whisper decoder embeddings (hidden states)
- Learned transformer based model with contrastive learning
- Support for multiple datasets (SHS100K, Discogs-VI, Lyric Covers)

### ⚡ Built for Scale

This codebase is designed for **large-scale experiments** on **high-performance computing (HPC)** systems:
- **Multi-GPU training** via Lightning Fabric (distributed data parallel)
- **Efficient data loading** with caching and parallel workers
- **Large datasets**: ~100K-500K audio tracks per dataset
- **Computationally intensive**: Feature extraction and training require significant GPU resources

All scripts support distributed execution across multiple GPUs, making them suitable for both local multi-GPU setups and HPC cluster environments.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Dataset Organization](#dataset-organization)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
- [Configuration](#configuration)
- [Code Organization](#code-organization)
- [Work in Progress](#work-in-progress)
- [Citation](#citation)

---

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU(s) - **Recommended: 4+ GPUs for training**
- FFmpeg (for audio processing)
- ~1TB disk space (for datasets and embeddings)
- HPC environment (optional, but recommended for large-scale experiments)

### Step 1: Create Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate 
```
### Step 2: Install Modified Whisper (Required)

⚠️ **Important**: This project uses a modified version of Whisper to extract hidden states.

```bash
# Install from GitHub
pip install git+https://github.com/helemanc/whisper.git

# Or force reinstall if already present
pip install --force-reinstall git+https://github.com/helemanc/whisper.git
```

For details about this fork, see: https://github.com/helemanc/whisper

### Step 3: Install Project Dependencies
```bash
# Clone repository
git clone https://github.com/yourusername/audio-based-lyrics-matching.git
cd audio-based-lyrics-matching

# Install dependencies
pip install -r requirements.txt
```

### ⚠️ CLEWS Extraction (Separate Environment Required)

If you plan to extract CLEWS audio embeddings (currently in development), you will need a **separate Python environment** due to dependency conflicts with WEALY:

```bash
# Create separate environment for CLEWS
conda create -n clews-extraction python=3.11
conda activate clews-extraction

# Install CLEWS dependencies
pip install -r requirements_clews.txt
```

**Note**: CLEWS feature extraction is currently under development. Use the main environment for all WEALY-related tasks.

---

## Data

### Datasets

We support three datasets for version identification research:

| Dataset | Cliques | Versions | Source | Collection Rate |
|---------|---------|----------|--------|-----------------|
| **SHS100K** | ~10K | ~100K | Standard | 82% (YouTube) |
| **Discogs-VI-YT** | ~98K | ~493K | Standard | Full |
| **Lyric Covers 2.0** | ~24K | ~54K | Custom | Full |

**Dataset Properties**:
- All audio processed at **16 kHz mono** with **5-minute maximum length**
- **SHS100K-v2**: Established benchmark; YouTube dependencies limited collection to 82%
- **Discogs-VI-YT**: YouTube-available subset (~493K versions, ~98K cliques); addresses SHS limitations
- **Lyric Covers 2.0**: Deduplicated version (54,301 covers, 24,561 originals, 80 languages)

### Directory Structure

Place all datasets in your data directory:
```
/path/to/data/
├── SHS100K/
│   ├── audio/
│   │   └── <clique_id>-<version_id>.mp3
│   └── metadata/
├── DiscogsVI/
│   ├── audio/
│   │   └── <artist>/<track>.mp3
│   └── metadata/
└── LyricCovers/
    ├── <version_id>/
    │   ├── <version_id>_audio.mp3
    │   └── <version_id>_lyrics.txt
    └── metadata/
```

### Dataset Metadata

Required metadata files are in `datasets/`.


### Caching

⚠️ **First Run**: Dataset processing takes ~10-30 minutes. Subsequent runs load from cache in <1 minute.

**Cache location**: `cache/{dataset_name}/`

**What's cached**:
- Audio metadata and file paths
- Clique/version ID mappings
- Split assignments
- Embedding path mappings (for training)

**To regenerate cache**: Delete `cache/{dataset_name}/` directory

---

## Dataset Organization

This codebase uses two complementary dataset classes:

### 🎵 **AudioDataset** (Feature Extraction)
- **Purpose**: Load raw audio files for embedding extraction
- **Used in**: `scripts/feature_extraction.py`
- **Returns**: Audio waveforms + metadata
- **When to use**: When you need to extract features from audio

### 📊 **EmbeddingDataset** (Training & Validation)
- **Purpose**: Load pre-extracted embeddings for model training
- **Used in**: `scripts/train.py`, `scripts/inference.py`
- **Returns**: Pre-computed embeddings + metadata
- **When to use**: When training or evaluating models

**Workflow**:
```
Audio Files → [AudioDataset] → Feature Extraction → Embeddings
                                                        ↓
                                    [EmbeddingDataset] → Training/Evaluation
```

---

## Usage

### 1. Feature Extraction

#### Whisper Embeddings

Extract Whisper decoder hidden states from audio using **AudioDataset**.

⚠️ **Performance Note**: Whisper extraction is **inherently slow** due to the autoregressive nature of the model. The process runs sequentially through audio and cannot be easily parallelized within a single sample. Using multiple GPUs helps by distributing samples across GPUs, but expect long runtimes for large datasets.

**Command Template:**
```bash
python scripts/extract_whisper.py \
    jobname=<JOB_NAME> \
    conf=configs/extraction/whisper_base.yaml \
    data.dataset_name=<DATASET_NAME> \
    path.data=<PATH_TO_AUDIO_DATA> \
    path.save_data_path=<PATH_TO_SAVE_EMBEDDINGS> \
    path.working_dir=<PATH_TO_PROJECT> \
    path.cache=<PATH_TO_CACHE> \
    fabric.ngpus=<NUM_GPUS> \
    fabric.precision=<PRECISION>
```

**Dataset-Specific Parameters:**

For **SHS100K**, add:
```bash
    path.shs_data=<PATH_TO_DATASETS>/shs/shs_data.csv \
    path.shs_splits=<PATH_TO_DATASETS>/shs
```

For **Lyric-Covers**, add:
```bash
    path.lyric_covers_data=<PATH_TO_DATASETS>/lyric-covers
```

For **Discogs-VI**, add:
```bash
    path.discogs_vi_data=<PATH_TO_DATASETS>/discogs-vi
```

**Common Parameters:**
- `jobname`: Descriptive name for this extraction job
- `data.dataset_name`: Dataset identifier (`shs`, `lyric-covers`, `discogs-vi`)
- `data.embedding_type`: Embedding type to extract (default: `last_hidden_states`)
  - `last_hidden_states`: Auto-detect language
  - `last_hidden_states_en`: Force English
  - `encoder`: Whisper encoder embeddings
- `path.data`: Root directory with audio files
- `path.save_data_path`: Where to save extracted embeddings
- `fabric.ngpus`: Number of GPUs (recommended: 4-8 for faster extraction)
- `fabric.precision`: Computation precision (`bf16-mixed` for speed, `32` for accuracy)

**Output Structure:**
```
<PATH_TO_SAVE_EMBEDDINGS>/{Dataset}-hidden-states/
├── <clique_id>/
│   ├── <version_id>/
│   │   ├── hs_last_seq.pt        # Hidden states embeddings
```

**Example Output:**
```
SHS100K-hidden-states/
├── 0/
│   ├── 0/
│   │   ├── hs_last_seq.pt       # Shape: (seq_len, 1280)
```

#### SBERT Embeddings

Extract SBERT text embeddings from Whisper transcriptions:
```bash
python scripts/extract_sbert.py \
    jobname=<JOB_NAME> \
    conf=configs/extraction/sbert_base.yaml \
    data.dataset_name=<DATASET_NAME> \
    path.data=<PATH_TO_AUDIO_DATA> \
    path.save_data_path=<PATH_TO_SAVE_EMBEDDINGS> \
    path.transcriptions=<PATH_TO_WHISPER_TRANSCRIPTIONS> \
    fabric.ngpus=1
```

**Notes:**
- Requires pre-extracted Whisper transcriptions
- SBERT processing is fast (single GPU sufficient)
- Creates `hs_sbert.pt` files alongside Whisper embeddings

---

### 2. Training

Train WEALY models on pre-extracted embeddings using **EmbeddingDataset**.

**Command Template:**
```bash
python scripts/train.py \
    jobname=<EXPERIMENT_NAME> \
    conf=configs/training/wealy.yaml \
    data.dataset_name=<DATASET_NAME> \
    path.cache=<PATH_TO_CACHE> \
    path.logs=<PATH_TO_LOGS> \
    path.working_dir=<PATH_TO_PROJECT> \
    path.data=<PATH_TO_AUDIO_DATA> \
    path.save_data_path=<PATH_TO_SAVE_DATA> \
    path.hidden_states=<PATH_TO_EMBEDDINGS> \
    path.meta=<PATH_TO_CACHED_METADATA> \
    fabric.ngpus=<NUM_GPUS> \
    fabric.precision=<PRECISION>
```

**Dataset-Specific Parameters:**

For **SHS100K**, add:
```bash
    path.shs_data=<PATH_TO_DATASETS>/shs/shs_data.csv \
    path.shs_splits=<PATH_TO_DATASETS>/shs
```

For **Lyric-Covers**, add:
```bash
    path.lyric_covers_data=<PATH_TO_DATASETS>/lyric-covers
```

For **Discogs-VI**, add:
```bash
    path.discogs_vi_data=<PATH_TO_DATASETS>/discogs-vi
```

**Common Training Parameters:**
- `jobname`: Experiment name (creates `logs/<jobname>/` directory)
- `data.dataset_name`: Dataset to train on
- `path.hidden_states`: Pre-extracted embeddings directory
- `path.logs`: Base directory for checkpoints (will create `<logs>/<jobname>/`)
- `path.meta`: Cached metadata (auto-generated on first run)
- `fabric.ngpus`: Number of GPUs (recommended: 4 for optimal training speed)
- `fabric.precision`: `bf16-mixed` (faster) or `32` (more accurate)
- `training.batchsize`: Batch size per GPU (default: 64)
- `training.numepochs`: Maximum epochs (default: 1000)

**📖 For all available parameters, see `configs/training/wealy.yaml`**

**Training Output:**
```
<PATH_TO_LOGS>/<EXPERIMENT_NAME>/
├── configuration.yaml           # Auto-saved config
├── checkpoint_last.ckpt         # Latest epoch
├── checkpoint_best.ckpt         # Best model (based on validation MAP)
└── checkpoint_epoch_N.ckpt      # Periodic checkpoints (if enabled)
```

**Example - Training on SHS100K:**
```bash
python scripts/train.py \
    jobname=wealy_shs_baseline \
    conf=configs/training/wealy.yaml \
    data.dataset_name=shs \
    path.cache=/scratch/cache \
    path.logs=/scratch/logs \
    path.hidden_states=/scratch/embeddings/SHS100K-hidden-states \
    path.meta=/scratch/cache/shs/metadata-shs.pt \
    path.shs_data=/project/datasets/shs/shs_data.csv \
    path.shs_splits=/project/datasets/shs \
    fabric.ngpus=4
```

**Expected Training Time** (4 GPUs):
- SHS100K: ~24-48 hours
- Lyric-Covers: ~12-24 hours
- Discogs-VI: ~48-72 hours

---

### 3. Evaluation

Evaluate trained models on test sets using **EmbeddingDataset**.

#### Standard Evaluation
```bash
python scripts/INFERENCE.py \
    checkpoint=<PATH_TO_CHECKPOINT> \
    partition=test \
    use_overlapping_chunks=false \
    ngpus=<NUM_GPUS> \
    precision=<PRECISION>
```

**Use for**: Fast evaluation with single embedding per version

#### Overlapping Chunks Evaluation (Recommended)
```bash
python scripts/inference.py \
    checkpoint=<PATH_TO_CHECKPOINT> \
    partition=test \
    use_overlapping_chunks=true \
    chunk_size=1500 \
    overlap_percentage=0.9 \
    topk_distance=1 \
    ngpus=<NUM_GPUS> \
    precision=<PRECISION> \
    checkpoint_dir=<EVAL_RESULTS_DIR>
```

**Use for**: More robust evaluation, better handling of variable-length audio

**Common Parameters:**
- `checkpoint`: Path to trained model (e.g., `logs/wealy_shs_exp1/checkpoint_best.ckpt`)
- `partition`: Data split to evaluate (`test`, `val`, or `train`)
- `use_overlapping_chunks`: Use overlapping chunks for robust evaluation
- `ngpus`: Number of GPUs for distributed evaluation
- `checkpoint_dir`: Where to save evaluation results and checkpoints

**📖 Configuration is automatically loaded from `<checkpoint_dir>/configuration.yaml`**

**Evaluation Metrics:**
- **MAP** (Mean Average Precision): Primary metric
- **MR1** (Mean Rank-1): Percentage of queries with correct match at rank 1
- **ARP** (Average Rank Percentile): Average rank position as percentile

**Saved Results**: `<checkpoint_dir>/final_results.pkl`

---

## Configuration

### Configuration Files

All configurations are in `configs/`.


### Key Configuration: `configs/training/wealy.yaml`

This file contains **all training parameters** with detailed documentation:
```yaml
# See configs/training/wealy.yaml for:
# - Path configurations
# - Dataset settings (chunk size, augmentation, etc.)
# - Model architecture (layers, dimensions, attention heads)
# - Training hyperparameters (learning rate, batch size, scheduler)
# - Monitoring and early stopping
# - Distributed training setup
```

**To customize training**, either:
1. **Edit the config file** directly, or
2. **Override via command line**:
```bash
   python scripts/train.py \
       conf=configs/training/wealy.yaml \
       training.batchsize=128 \
       training.optim.lr=5e-4 \
       model.num_transformer_blocks=6
```

### Quick Test Configuration

Test the pipeline with minimal data:
```bash
python scripts/train.py \
    jobname=test_run \
    conf=configs/training/test_5_cliques.yaml \
    data.dataset_name=shs \
    path.hidden_states=<PATH_TO_EMBEDDINGS> \
    fabric.ngpus=1
```

This uses only 5 cliques and runs for 2 epochs (~2-5 minutes).

---

## Code Organization
```
audio-based-lyrics-matching/
├── configs/                        # Configuration files
│   ├── extraction/
│   │   ├── whisper.yaml            # Whisper extraction config
│   │   └── sbert.yaml              # SBERT extraction config
│   └── training/
│       ├── wealy.yaml              # Complete WEALY 
│
├── datasets/                       # Dataset metadata
│   ├── shs/
│   ├── discogs-vi/
│   └── lyric-covers/
│
├── lib/                            # Core library
│   ├── dataset/
│   │   ├── audio_dataset.py        # AudioDataset (feature extraction)
│   │   ├── embedding_dataset.py    # EmbeddingDataset (training/eval)
│   │   └── collate_functions.py    # Batch collation functions
│   ├── models/
│   │   ├── wealy.py                # WEALY model architecture
│   │   └── ...                     # Other model architectures
│   └── evaluation/
│       └── eval.py                 # Evaluation metrics (MAP, MR1, ARP)
│       └── distances.py            # Distances (cosine, euclidean)
│       └── baselines.py            # Baselines implementation (WIP)
│   ├── extractors.py               # Script for feature extraction
│   ├── layers.py                   # Models' layers
│   ├── losses.py                   # Losses definition
│   ├── tensor_ops.py               # Tensor operations
│
├── utils/                          # Utility modules
│   ├── training_utils.py           # Training logic and loops
│   ├── inference_utils.py          # Evaluation logic
│   ├── extraction_utils.py         # Feature extraction helpers
│   ├── evaluation_utils.py         # Metric computation
│   ├── print_utils.py              # Logging utilities
│   └── pytorch_utils.py            # PyTorch helpers
│
├── scripts/                        # Executable scripts
│   ├── feature_extraction.py       # Extract Whisper/SBERT embeddings
│   ├── train.py                    # Train models
│   └── inference.py                # Evaluate models
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---


## ✅ To-Do List

- [ ] **Fix inference script**

- [ ] **Fix Triplet Loss and CLEWS loss**

- [ ] **Integrate CLEWS audio embeddings**
  - [ ] Set up separate Python environment (dependency conflicts)

- [ ] **Integrate transcription-based baselines**
  - [ ] TF-IDF similarity on Whisper transcriptions  
  - [ ] SBERT cosine similarity baseline  
  - [ ] Edit distance (Levenshtein) baseline

- [ ] **Refactor and enable multimodality**
  - [ ] Late fusion strategies

- [ ] **Compute and cache distance matrices**
  - [ ] Pre-compute distance matrices for efficient evaluation

- [ ] **Implement multimodal evaluation pipeline**
  - [ ] Use cached distance matrices for evaluation


---

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{mancini2024wealy,
  title={Leveraging Whisper Embeddings for Audio-based Lyrics Matching},
  author={Mancini, Eleonora and Serrà, Joan and Torroni, Paolo and Mitsufuji, Yuki},
  journal={arXiv preprint arXiv:2510.08176},
  year={2025}
}
```

---

## License

[LICENSE TYPE] - See LICENSE file for details

---

## Contact

For questions or issues:
- **Open an issue**: [GitHub Issues](https://github.com/yourusername/audio-based-lyrics-matching/issues)
- **Email**: e.mancini@unibo.it


---

<div align="center">
<p><b>⭐ Star this repository if you find it useful!</b></p>
<p>Watch for updates as we continue adding features and improvements.</p>
</div>
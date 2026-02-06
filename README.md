# Leveraging Whisper Embeddings for Audio-based Lyrics Matching
[![Status](https://img.shields.io/badge/status-released-brightgreen)](https://arxiv.org/abs/2510.08176v1)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository accompanies the paper:
**_Leveraging Whisper Embeddings for Audio-based Lyrics Matching_**
by *Eleonora Mancini, Joan Serrà, Paolo Torroni, and Yuki Mitsufuji*
[[📄 Read the paper on arXiv](https://arxiv.org/abs/2510.08176v1)]

## 🧠 About the Project  

This project introduces **WEALY** — **W**hisper **E**mbeddings for **A**udio-based **LY**rics matching — a fully reproducible pipeline that leverages Whisper decoder embeddings for **audio-based lyrics matching**.

**WEALY** establishes transparent and reproducible baselines for version identification using:
- Pre-extracted Whisper decoder embeddings (hidden states)
- Learned transformer based model with contrastive learning
- Support for multiple datasets (SHS100K, Discogs-VI, LyricCovers)

### ✨ Key Features

<table>
<tr>
<td width="50%">

**🎤 Multi-Modal Feature Extraction**
- **Whisper** decoder embeddings (auto-language & English)
- **SBERT** text embeddings from transcriptions
- **CLEWS** audio embeddings
- Efficient caching and distributed processing

</td>
<td width="50%">

**🚀 Training & Inference**
- Transformer-based contrastive learning
- Multi-GPU distributed training (Lightning Fabric)
- Overlapping chunks for robust evaluation
- Automatic checkpointing and recovery

</td>
</tr>
<tr>
<td width="50%">

**📊 Comprehensive Evaluation**
- Standard metrics: MAP, MR1, ARP
- Transcription baselines: SBERT, TF-IDF
- Theoretical bounds: Ideal, Random, Modified variants
- Multimodal fusion with grid search

</td>
<td width="50%">

**🤗 Pre-trained Models**
- **8 models** on [Hugging Face Hub](https://huggingface.co/audio-based-lyrics-matching)
- SHS100K, Discogs-VI, Lyric Covers datasets
- Auto-download with cached inference
- Ready for immediate evaluation

</td>
</tr>
</table>

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
- [Usage](#usage)
  - [1. Feature Extraction](#1-feature-extraction)
  - [2. Training](#2-training)
  - [3. Inference](#3-inference)
  - [4. Distance Matrix Computation](#4-distance-matrix-computation)
  - [5. Multimodal Fusion](#5-multimodal-fusion)
  - [6. Baseline Evaluation](#6-baseline-evaluation)
- [Pre-trained Models](#pre-trained-models)
- [Configuration](#configuration)
- [Code Organization](#code-organization)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

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

CLEWS (Contrastive Learning of Musical Embeddings with Weak Supervision) audio embeddings require a **separate Python environment** due to dependency conflicts with the main WEALY environment.

#### Step 1: Clone CLEWS Repository

Clone our CLEWS fork at the **same parent level** as this repository:

```bash
# From the parent directory of audio-based-lyrics-matching
cd ..
git clone https://github.com/helemanc/clews.git

# Verify folder structure:
# Projects/
# ├── audio-based-lyrics-matching/
# └── clews/
```

#### Step 2: Create CLEWS Environment

```bash
# Create separate conda environment
conda create -n clews python=3.11
conda activate clews

# Install CLEWS dependencies
cd clews
pip install -r requirements.txt

# Install additional dependencies for feature extraction
pip install lightning omegaconf tqdm requests
```

#### Step 3: Configure CLEWS Paths

Update `configs/extraction/clews.yaml` with your paths:

```yaml
clews:
  project_dir: "/path/to/clews"  # Path to cloned CLEWS repository
  config_path: null              # Auto-download from Zenodo
  checkpoint_path: "/path/to/clews_checkpoints"  # Where to store checkpoints
```

**Note**: CLEWS checkpoints are automatically downloaded from Zenodo on first use.

#### Available CLEWS Models

| Model | Dataset | Zenodo |
|-------|---------|--------|
| `shs-clews` | SHS100K | [zenodo.org/records/15045900](https://zenodo.org/records/15045900) |
| `dvi-clews` | Discogs-VI | [zenodo.org/records/15045900](https://zenodo.org/records/15045900) |

---

## Data

### Datasets

We support three datasets for version identification research:

| Dataset | Cliques | Versions | Source | Collection Rate |
|---------|---------|----------|--------|-----------------|
| **SHS100K** | ~10K | ~100K | Standard | 82% (YouTube) |
| **Discogs-VI-YT** | ~98K | ~493K | Standard | Full |
| **LyricCovers 2.0** | ~24K | ~54K | Custom | Full |

**Dataset Properties**:
- All audio processed at **16 kHz mono** with **5-minute maximum length**
- **SHS100K-v2**: Established benchmark; YouTube dependencies limited collection to 82%
- **Discogs-VI-YT**: YouTube-available subset (~493K versions, ~98K cliques); addresses SHS limitations
- **LyricCovers 2.0**: Deduplicated version (54,301 covers, 24,561 originals, 80 languages)

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

### Lyric Covers Dataset Preparation

⚠️ **Note**: The Lyric Covers dataset requires manual preparation. Metadata is provided in `datasets/lyric-covers/data.csv`.

**Expected Directory Structure**:
```
/path/to/data/LyricCovers/
├── 1001/
│   ├── 1001_lyrics.txt
│   └── 1001_audio.mp3
├── 1002/
│   ├── 1002_lyrics.txt
│   └── 1002_audio.mp3
└── ...
```

**Audio Format Requirements**:
- **Format**: MP3
- **Channels**: Mono
- **Sample rate**: 16 kHz
- **Naming**: `{song_id}_audio.mp3`

**Lyrics Format Requirements**:
- **Format**: Plain text (.txt)
- **Encoding**: UTF-8
- **Naming**: `{song_id}_lyrics.txt`
- **Content**: Full lyrics text

**Data Preparation**:
After obtaining the audio files, resample them to **16 kHz mono MP3** format and organize them following the directory structure above.


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
python scripts/feature_extraction.py \
    jobname=<JOB_NAME> \
    conf=configs/extraction/whisper.yaml \
    data.dataset_name=<DATASET_NAME> \
    data.split=<SPLIT> \
    path.data=<PATH_TO_AUDIO_DATA> \
    path.base_path=<PATH_TO_BASE> \ 
    path.save_data_path=<PATH_TO_SAVE_EMBEDDINGS> \
    path.working_dir=<PATH_TO_PROJECT> \
    fabric.ngpus=<NUM_GPUS> \
    fabric.precision=<PRECISION>
```

**Common Parameters:**
- `jobname`: Descriptive name for this extraction job
- `data.dataset_name`: Dataset identifier (`shs`, `lyric-covers`, `discogs-vi`)
- `data.embedding_type`: Embedding type to extract (default: `last_hidden_states`)
  - `last_hidden_states`: Auto-detect language
  - `last_hidden_states_en`: Force English
  - `encoder`: Whisper encoder embeddings
- `path.data`: Root directory with audio files
- `path.base_path`: Base path for dataset-specific parameters (`<PATH_TO_WORKING_DIR>/datasets`)
- `path.save_data_path`: Where to save extracted embeddings
- `path.working_dir`: Working directory (`<PATH_TO_WORKING_DIR>/audio-based-lyrics-matching`)
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
python scripts/feature_extraction.py \
    jobname=<JOB_NAME> \
    conf=configs/extraction/sbert.yaml \
    data.dataset_name=<DATASET_NAME> \
    data.split=<SPLIT_NAME> \
    path.base_path=<PATH_TO_WORKING_DIR>/datasets \
    path.working_dir=<PATH_TO_WORKING_DIR>/audio-based-lyrics-matching \
    path.data=<PATH_TO_AUDIO_DATA> \
    path.save_data_path=<PATH_TO_SAVE_EMBEDDINGS> \
    path.transcriptions=<PATH_TO_WHISPER_TRANSCRIPTIONS> \
    fabric.ngpus=1
```

**Notes:**
- Requires pre-extracted Whisper transcriptions
- SBERT processing is fast (single GPU sufficient)
- Creates `hs_sbert.pt` files alongside Whisper embeddings

#### CLEWS Audio Embeddings - Complete Workflow

⚠️ **Requires separate Python environment** - See [CLEWS Installation](#%EF%B8%8F-clews-extraction-separate-environment-required) above.

CLEWS extracts audio embeddings using a CNN-based architecture with shingle (overlapping window) processing. Follow this complete workflow for CLEWS feature extraction and distance matrix computation.

##### Step 1: Data Preprocessing (Required Once Per Dataset)

Before extracting CLEWS features, preprocess the dataset to create metadata files using `data_preproc.py` from the CLEWS repository.

**Activate CLEWS environment and navigate to CLEWS directory:**
```bash
conda activate clews
cd /path/to/clews
```

**For SHS100K:**
```bash
python data_preproc.py \
    --njobs=16 \
    --dataset=SHS100K \
    --path_meta=/path/to/audio-based-lyrics-matching/datasets/shs \
    --path_audio=/path/to/data/SHS100K/audio/ \
    --ext_in=mp3 \
    --fn_out=/path/to/audio-based-lyrics-matching/cache/clews/metadata-shs.pt
```

**For Discogs-VI:**
```bash
python data_preproc.py \
    --njobs=16 \
    --dataset=DiscogsVI \
    --path_meta=/path/to/audio-based-lyrics-matching/datasets/discogs-vi \
    --path_audio=/path/to/data/DiscogsVI/audio/ \
    --ext_in=mp3 \
    --fn_out=/path/to/audio-based-lyrics-matching/cache/clews/metadata-dvi.pt
```

##### Step 2: CLEWS Feature Extraction

Switch to the audio-based-lyrics-matching repository while keeping the CLEWS environment active:

```bash
conda activate clews
cd /path/to/audio-based-lyrics-matching

python scripts/feature_extraction.py \
    conf=configs/extraction/clews.yaml \
    jobname=shs_clews_extraction \
    data.dataset_name=shs \
    data.split=test \
    path.save_data_path=/path/to/data/SHS100K-hidden-states \
    path.clews_cache_dir=/path/to/audio-based-lyrics-matching/cache/clews \
    path.clews_audio_dir=/path/to/data/SHS100K/audio/ \
    clews.project_dir=/path/to/clews \
    clews.checkpoint_path=/path/to/clews/checkpoints \
    fabric.ngpus=4
```

##### Step 3: CLEWS Distance Matrix Computation

Switch back to the CLEWS repository to compute distance matrices using CLEWS scripts:

```bash
conda activate clews
cd /path/to/clews

python compute_distance_matrix.py \
    checkpoint_dir=/path/to/clews/checkpoints \
    dataset=shs \
    output_dir=/path/to/audio-based-lyrics-matching/logs/clews-shs/distance_matrix \
    path_meta=/path/to/cache/clews/metadata-shs.pt \
    path_audio=/path/to/data/SHS100K/audio/ \
    partition=test \
    ngpus=4
```

**CLEWS Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `maxlen` | 600 | Maximum audio length in seconds (10 min) |
| `qshop` | 5 | Shingle hop in seconds |
| `qslen` | null | Shingle length (null = use model default) |
| `clews.project_dir` | required | Path to cloned CLEWS repository |
| `clews.checkpoint_path` | required | Path to store/load CLEWS checkpoints |
| `checkpoint_dir` | required | Path to CLEWS checkpoints for distance computation |
| `output_dir` | required | Where to save distance matrices |

**Output Structure:**
```
logs/clews-shs/distance_matrix/
└── shs_clews_distance_matrix_test.pkl
```

**Notes:**
- CLEWS checkpoints are auto-downloaded from Zenodo on first use
- Extraction uses half-precision for storage efficiency
- Distance matrix includes metadata for fusion with other modalities

---

### 2. Training

Train WEALY models on pre-extracted embeddings using **EmbeddingDataset**.

**Command Template:**
```bash
python scripts/train.py \
    jobname=<EXPERIMENT_NAME> \
    conf=configs/training/wealy.yaml \
    data.dataset_name=<DATASET_NAME> \
    training.batch_size=<BATCH_SIZE> \
    training.numepochs=<NUM_EPOCHS> \
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

For **LyricCovers**, add:
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
- LyricCovers: ~12-24 hours
- Discogs-VI: ~48-72 hours

---

### 3. Inference

Evaluate trained models on test sets using **EmbeddingDataset**.

The inference script supports **two ways to load models**:
1. **Local checkpoint**: Use a checkpoint file from your filesystem
2. **Hugging Face Hub**: Automatically download pre-trained models from HF

#### Option A: Using Local Checkpoint

```bash
python scripts/inference.py \
    checkpoint=logs/wealy-whisper-shs/checkpoint_best.ckpt \
    partition=test \
    chunk_size=1500 \
    overlap_percentage=0.9 \
    topk_distance=1 \
    use_overlapping_chunks=true \
    hidden_states=/path/to/hidden-states \
    ngpus=4 \
    disable_memory_logging=true
```

#### Option B: Using Hugging Face Models (Recommended)

```bash
python scripts/inference.py \
    model_name=wealy-whisper-shs \
    hidden_states=/path/to/hidden-states \
    partition=test \
    use_overlapping_chunks=true \
    ngpus=4
```

The model will be automatically downloaded to `logs/<model-name>/` on first use and cached for subsequent runs.

#### Distributed Inference with `torchrun`

For multi-GPU inference, use `torchrun` to spawn processes:

```bash
torchrun --nproc_per_node=4 --standalone scripts/inference.py \
    model_name=wealy-whisper-shs \
    hidden_states=/path/to/SHS100K-hidden-states \
    shs_data=/path/to/datasets/shs/shs_data.csv \
    shs_splits=/path/to/datasets/shs \
    cache=/path/to/cache \
    data=/path/to/data \
    partition=test \
    use_overlapping_chunks=true \
    ngpus=4
```

#### Dataset-Specific Path Overrides

Since models downloaded from HuggingFace have sanitized configurations (personal paths removed), you **must** provide the required paths via CLI arguments.

**For SHS100K models** (`wealy-*-shs`):
```bash
    hidden_states=/path/to/SHS100K-hidden-states \
    shs_data=/path/to/datasets/shs/shs_data.csv \
    shs_splits=/path/to/datasets/shs \
    cache=/path/to/cache \
    data=/path/to/data
```

**For LyricCovers models** (`wealy-*-lyc`):
```bash
    hidden_states=/path/to/LyricCovers-hidden-states \
    lyric_covers_data=/path/to/datasets/lyric-covers \
    cache=/path/to/cache \
    data=/path/to/data
```

**For Discogs-VI models** (`wealy-*-dvi`):
```bash
    hidden_states=/path/to/DiscogsVI-hidden-states \
    discogs_vi_data=/path/to/datasets/discogs-vi \
    cache=/path/to/cache \
    data=/path/to/data
```

#### Output Directory Structure

When running inference, the following directory structure is created:

```
logs/<model-name>/                    # Model directory (HF models stored here)
├── best.ckpt                         # Model checkpoint
├── configuration.yaml                # Model configuration
└── eval_checkpoints/                 # Created during inference
    ├── extraction_checkpoint_*.pkl   # Intermediate extraction checkpoints
    ├── evaluation_checkpoint_*.pkl   # Intermediate evaluation checkpoints
    ├── final_results.pkl             # Final evaluation metrics
    └── crash_report_rank_*.txt       # Error logs (if any)
```

The `eval_checkpoints/` folder enables **resumable inference** - if evaluation is interrupted, it will resume from the last checkpoint automatically.

#### Evaluation Modes

**Standard Evaluation** (faster):
```bash
use_overlapping_chunks=false
```
Uses single embedding per version. Suitable for quick testing.

**Overlapping Chunks Evaluation** (recommended for final results):
```bash
use_overlapping_chunks=true \
chunk_size=1500 \
overlap_percentage=0.9 \
topk_distance=1
```
Generates multiple overlapping chunks per audio file for more robust evaluation.

#### All Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | None | HF model name (e.g., `wealy-whisper-shs`) |
| `checkpoint` | None | Local checkpoint path (alternative to `model_name`) |
| `partition` | `test` | Dataset split (`test`, `val`, `train`) |
| `ngpus` | 1 | Number of GPUs |
| `precision` | `bf16-mixed` | Computation precision |
| `use_overlapping_chunks` | `false` | Enable overlapping chunk evaluation |
| `chunk_size` | 1500 | Size of overlapping chunks |
| `overlap_percentage` | 0.9 | Overlap between chunks (0.0-0.99) |
| `topk_distance` | 1 | Top-k distance aggregation |
| `disable_checkpointing` | `false` | Disable intermediate checkpoints |
| `disable_memory_logging` | `true` | Disable GPU memory logging |
| `force_download` | `false` | Force re-download from HF |

#### Evaluation Metrics

- **MAP** (Mean Average Precision): Primary metric for retrieval quality
- **MR1** (Mean Rank-1): Percentage of queries with correct match at rank 1
- **ARP** (Average Rank Percentile): Average rank position as percentile

Results are printed to console and saved to `<model_dir>/eval_checkpoints/final_results.pkl`

---

### 4. Distance Matrix Computation

Compute pairwise distance matrices for trained models. These matrices can be used for:
- Detailed analysis and debugging
- Multimodal fusion (combining CLEWS + WEALY)
- Custom retrieval experiments

#### WEALY Distance Matrix (Overlapping Chunks - Recommended)

Compute WEALY distance matrix with overlapping chunks for robust evaluation:

```bash
python scripts/compute_distance_matrix.py \
    checkpoint=/path/to/logs/wealy-whisper-shs \
    partition=test \
    use_overlapping_chunks=true \
    overlap_percentage=0.9 \
    topk_distance=3 \
    ngpus=4
```

**Output:**
```
logs/wealy-whisper-shs/distance_matrix/
└── test_overlapping_topk3_distances.pkl
```

#### Standard Mode (Faster, Less Robust)

Compute distance matrix with single embedding per song:

```bash
python scripts/compute_distance_matrix.py \
    checkpoint=logs/wealy_shs/best.ckpt \
    partition=test \
    save_distance_matrix=distances/wealy_shs_test.pkl \
    ngpus=4
```

#### Output Format

Distance matrices are saved as pickle files with metadata:

```python
{
    'distance_matrix': np.ndarray,  # (n_queries, n_candidates)
    'query_references': [
        {'clique': int, 'version': int, 'matrix_row': int},
        ...
    ],
    'candidate_references': [
        {'clique': int, 'version': int, 'matrix_col': int},
        ...
    ],
    'metadata': {
        'checkpoint': str,
        'partition': str,
        'use_overlapping_chunks': bool,
        'topk_distance': int,
        ...
    }
}
```

The `matrix_row` and `matrix_col` fields ensure exact correspondence between the distance matrix and the song metadata.

---

### 5. Multimodal Fusion

Combine distance matrices from different modalities (e.g., CLEWS audio + WEALY lyrics) to find optimal fusion weights. This allows combining the strengths of audio-based and lyrics-based approaches.

#### Complete Fusion Workflow

**Prerequisite:** Compute both CLEWS and WEALY distance matrices (see Step 3 in CLEWS workflow and Step 4 above).

**Basic Fusion:**
```bash
python scripts/multimodal_fusion.py \
    --matrix1 /path/to/logs/clews-shs/distance_matrix/shs_clews_distance_matrix_test.pkl \
    --matrix2 /path/to/logs/wealy-whisper-shs/distance_matrix/test_overlapping_topk3_distances.pkl \
    --output /path/to/logs/fusion-wealy-clews/test_results.csv
```

This evaluates the fusion: `combined_distance = matrix1 + alpha * matrix2` across different alpha values.

#### Custom Alpha Range

Fine-grained search around a specific range:

```bash
python scripts/multimodal_fusion.py \
    --matrix1 distances/clews_shs_test.pkl \
    --matrix2 distances/wealy_shs_test.pkl \
    --alpha_range 0.0 2.0 0.1 \
    --output fusion_results.csv
```

Or specify exact values:

```bash
python scripts/multimodal_fusion.py \
    --matrix1 distances/clews_shs_test.pkl \
    --matrix2 distances/wealy_shs_test.pkl \
    --alphas 0.0 0.5 1.0 1.5 2.0 \
    --output fusion_results.csv
```

#### Output

The script prints a table showing MAP/MR1/ARP for each alpha value.

Results are also saved to the CSV file for further analysis.

#### Complete Workflow Example

```bash
# 1. Extract CLEWS embeddings (separate environment)
conda activate clews
python scripts/feature_extraction.py \
    conf=configs/extraction/clews.yaml \
    data.dataset_name=shs

# 2. Extract WEALY embeddings (main environment)
conda activate wealy
python scripts/feature_extraction.py \
    conf=configs/extraction/whisper.yaml \
    data.dataset_name=shs

# 3. Train WEALY model
python scripts/train.py \
    conf=configs/training/wealy.yaml \
    data.dataset_name=shs

# 4. Compute CLEWS distance matrix (requires CLEWS preprocessing)
# (CLEWS distances are typically pre-computed during CLEWS workflow)
python path/to/clews/compute_distance_matrix.py \
    checkpoint_dir=/path/to/clews/checkpoints \
    dataset=shs \
    output_dir=/path/to/audio-based-lyrics-matching/logs/clews-shs/distance_matrix \
    path_meta=/path/to/cache/clews/metadata-shs.pt \
    path_audio=/path/to/data/SHS100K/audio/ \
    partition=test \
    ngpus=4

# 5. Compute WEALY distance matrix
python scripts/compute_distance_matrix.py \
    checkpoint=logs/wealy_shs/best.ckpt \
    partition=test \
    save_distance_matrix=/path/to/audio-based-lyrics-matching/logs/wealy-whisper-shs/distance_matrix \

# 6. Find optimal fusion
python scripts/multimodal_fusion.py \
    --matrix1 /path/to/audio-based-lyrics-matching/logs/clews-shs/clews_shs_test.pkl \
    --matrix2 /path/to/audio-based-lyrics-matching/logs/wealy-whisper-shs/distance_matrix/wealy_shs_test.pkl \
    --alpha_range 0.0 2.5 0.1 \
    --output /path/to/audio-based-lyrics-matching/logs/fusion-wealy-clews/fusion_results.csv
```

---

### 6. Baseline Evaluation

Evaluate transcription-based baselines (SBERT, TF-IDF) and theoretical bounds on Whisper transcriptions.

#### Prerequisites
- Pre-extracted Whisper transcriptions (generated during feature extraction)
- Transcriptions stored in: `<data_folder>/<Dataset>-transcriptions/transcriptions/`

#### Command Template

```bash
python scripts/compute_baselines.py \
    jobname=<JOB_NAME> \
    conf=configs/evaluation/baselines.yaml \
    data.dataset_name=<DATASET_NAME> \
    data.split=<SPLIT> \
    data.whisper_set=<WHISPER_SET> \
    path.base_path=<PATH_TO_DATASETS> \
    path.data=<PATH_TO_DATA> \
    path.save_results_path=<PATH_TO_SAVE_RESULTS> \
    'baselines.compute=[sbert,tfidf-cosine,tfidf-lucene,ideal,random,modified_ideal,modified-random]'
```

#### Example: SHS100K Test Set

```bash
python scripts/compute_baselines.py \
    jobname=shs_test_baselines \
    conf=configs/evaluation/baselines.yaml \
    data.dataset_name=shs \
    data.split=test \
    data.whisper_set=prompt_whisper_42_transcribe \
    path.base_path=/path/to/audio-based-lyrics-matching/datasets \
    path.data=/path/to/data \
    path.save_results_path=/path/to/logs/baselines \
    'baselines.compute=[sbert,tfidf-cosine,tfidf-lucene,ideal,random,modified_ideal,modified-random]'
```

#### Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `jobname` | Descriptive name for this baseline run | `shs_test_baselines` |
| `data.dataset_name` | Dataset identifier | `shs`, `lyric-covers`, `discogs-vi` |
| `data.split` | Dataset split to evaluate | `test`, `val`, `train` |
| `data.whisper_set` | Whisper configuration identifier (without dataset prefix) | `prompt_whisper_42_transcribe` |
| `path.base_path` | Parent directory containing dataset metadata folders | `/path/to/datasets/` |
| `path.data` | Root directory with audio files and transcriptions | `/path/to/data/` |
| `path.save_results_path` | Where to save baseline results | `/path/to/logs/baselines/` |
| `baselines.compute` | List of baselines to compute | See available baselines below |

#### Available Baselines

- **`sbert`**: Sentence-BERT embeddings with cosine similarity
- **`tfidf-cosine`**: TF-IDF with cosine similarity
- **`tfidf-lucene`**: TF-IDF with Lucene-style scoring
- **`ideal`**: Theoretical upper bound (perfect clique matching)
- **`random`**: Random baseline (lower bound)
- **`modified_ideal`**: Perfect matching only for valid transcriptions
- **`modified-random`**: Upper bound for transcription-based methods

#### Whisper Set Naming

The `whisper_set` parameter should match your transcription files without the dataset prefix. For example:
- If files are named `shs_prompt_whisper_42_transcribe.txt`
- Use `data.whisper_set=prompt_whisper_42_transcribe`

The dataset name (`shs_`) is automatically prepended by the dataloader.

#### Output

Results are saved to `<save_results_path>/<jobname>/`:
```
logs/baselines/shs_test_baselines/
├── baseline_results.json           # All baseline metrics
├── detailed_results.pkl            # Full result tensors (if enabled)
└── comparison_table.txt            # Human-readable comparison
```

**Metrics computed:**
- **MAP** (Mean Average Precision): Primary ranking quality metric
- **MR1** (Mean Rank @ 1): Percentage of queries with top-1 correct
- **ARP** (Average Rank Percentile): Normalized ranking (0-100)

**Config**: See [configs/evaluation/baselines.yaml](configs/evaluation/baselines.yaml) for all configuration options.

---

## Pre-trained Models

We provide **8 pre-trained WEALY models** on [Hugging Face Hub](https://huggingface.co/audio-based-lyrics-matching) trained on different datasets and embedding types.

### Available Models

| Model Name | Dataset | Embedding Type | HF Repository |
|------------|---------|----------------|---------------|
| `wealy-whisper-shs` | SHS100K | Whisper (auto-lang) | [audio-based-lyrics-matching/wealy-whisper-shs](https://huggingface.co/audio-based-lyrics-matching/wealy-whisper-shs) |
| `wealy-sbert-shs` | SHS100K | SBERT | [audio-based-lyrics-matching/wealy-sbert-shs](https://huggingface.co/audio-based-lyrics-matching/wealy-sbert-shs) |
| `wealy-whisper-en-shs` | SHS100K | Whisper (English) | [audio-based-lyrics-matching/wealy-whisper-en-shs](https://huggingface.co/audio-based-lyrics-matching/wealy-whisper-en-shs) |
| `wealy-avgembmlp-shs` | SHS100K | Avg Embedding MLP | [audio-based-lyrics-matching/wealy-avgembmlp-shs](https://huggingface.co/audio-based-lyrics-matching/wealy-avgembmlp-shs) |
| `wealy-cls-shs` | SHS100K | CLS Token | [audio-based-lyrics-matching/wealy-cls-shs](https://huggingface.co/audio-based-lyrics-matching/wealy-cls-shs) |
| `wealy-whisper-lyc` | LyricCovers | Whisper (auto-lang) | [audio-based-lyrics-matching/wealy-whisper-lyc](https://huggingface.co/audio-based-lyrics-matching/wealy-whisper-lyc) |
| `wealy-sbert-lyc` | LyricCovers | SBERT | [audio-based-lyrics-matching/wealy-sbert-lyc](https://huggingface.co/audio-based-lyrics-matching/wealy-sbert-lyc) |
| `wealy-whisper-dvi` | Discogs-VI | Whisper (auto-lang) | [audio-based-lyrics-matching/wealy-whisper-dvi](https://huggingface.co/audio-based-lyrics-matching/wealy-whisper-dvi) |

### Quick Start Examples

**SHS100K model:**
```bash
torchrun --nproc_per_node=4 --standalone scripts/inference.py \
    model_name=wealy-whisper-shs \
    hidden_states=/path/to/SHS100K-hidden-states \
    shs_data=/path/to/datasets/shs/shs_data.csv \
    shs_splits=/path/to/datasets/shs \
    cache=/path/to/cache \
    data=/path/to/data \
    partition=test \
    use_overlapping_chunks=true \
    ngpus=4
```

**LyricCovers model:**
```bash
torchrun --nproc_per_node=4 --standalone scripts/inference.py \
    model_name=wealy-whisper-lyc \
    hidden_states=/path/to/LyricCovers-hidden-states \
    lyric_covers_data=/path/to/datasets/lyric-covers \
    cache=/path/to/cache \
    data=/path/to/data \
    partition=test \
    use_overlapping_chunks=true \
    ngpus=4
```

**Discogs-VI model:**
```bash
torchrun --nproc_per_node=4 --standalone scripts/inference.py \
    model_name=wealy-whisper-dvi \
    hidden_states=/path/to/DiscogsVI-hidden-states \
    discogs_vi_data=/path/to/datasets/discogs-vi \
    cache=/path/to/cache \
    data=/path/to/data \
    partition=test \
    use_overlapping_chunks=true \
    ngpus=4
```

### Listing Available Models

To see all available models programmatically:

```python
from utils.hf_utils import print_available_models
print_available_models()
```

Or use the upload script with `--list`:
```bash
python scripts/upload_to_hf.py --list
```

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

---

## Code Organization
```
audio-based-lyrics-matching/
├── configs/                        # Configuration files
│   ├── extraction/
│   │   ├── whisper.yaml            # Whisper extraction config
│   │   ├── sbert.yaml              # SBERT extraction config
│   │   └── clews.yaml              # CLEWS extraction config
│   ├── training/
│   │   └── wealy.yaml              # Complete WEALY training config
│   └── evaluation/
│       └── baselines.yaml          # Baseline evaluation config
│
├── datasets/                       # Dataset metadata
│   ├── shs/
│   │   ├── shs_data.csv            # SHS100K metadata
│   │   ├── SHS100K-TRAIN           # Train split clique list
│   │   ├── SHS100K-VAL             # Validation split clique list
│   │   ├── SHS100K-TEST            # Test split clique list
│   │   └── list                    # Full clique list
│   ├── discogs-vi/
│   │   ├── id-to-file-mapping.csv  # Song ID to filename mapping
│   │   ├── DiscogsVI-YT-20240701-light.json.train  # Train split
│   │   ├── DiscogsVI-YT-20240701-light.json.val    # Validation split
│   │   └── DiscogsVI-YT-20240701-light.json.test   # Test split
│   └── lyric-covers/
│       ├── data.csv                # LyricCovers metadata
│       ├── train_no_dup.csv        # Train split
│       ├── val_no_dup.csv          # Validation split
│       └── test_no_dup.csv         # Test split
│
├── lib/                            # Core library
│   ├── audio_dataset/              # Audio data loading
│   │   ├── dataloader.py           # DataLoader creation
│   │   ├── dataset.py              # AudioDataset for feature extraction
│   │   ├── cache.py                # TranscriptionCache
│   │   ├── validator.py            # Transcription validation
│   │   ├── data_processing.py      # Dataset-specific processing
│   │   └── utils.py                # Audio dataset utilities
│   ├── embedding_dataset/          # Embedding data loading
│   │   ├── base_dataset.py         # EmbeddingDataset (training/eval)
│   │   ├── multimodal_dataset.py   # Multimodal dataset handling
│   │   ├── collate_functions.py    # Batch collation functions
│   │   ├── cache_manager.py        # Embedding cache management
│   │   ├── data_processing.py      # Embedding data processing
│   │   └── utils.py                # Embedding dataset utilities
│   ├── models/
│   │   └── wealy.py                # WEALY model architecture
│   ├── evaluation/
│   │   ├── eval.py                 # Evaluation metrics (MAP, MR1, ARP)
│   │   ├── distances.py            # Distance functions (cosine, euclidean)
│   │   ├── fusion.py               # Multimodal fusion utilities
│   │   └── baselines.py            # Baseline implementations
│   ├── extractors.py               # Feature extraction classes
│   ├── layers.py                   # Neural network layers (documented)
│   ├── losses.py                   # Loss definitions
│   └── tensor_ops.py               # Tensor operations
│
├── utils/                          # Utility modules (all documented with type hints)
│   ├── training_utils.py           # Training logic and loops
│   ├── inference_utils.py          # Evaluation logic
│   ├── extraction_utils.py         # Feature extraction helpers
│   ├── latents_extraction_utils.py # Latent representation extraction
│   ├── evaluation_utils.py         # Metric computation (documented)
│   ├── distance_matrix_utils.py    # Distance matrix computation (documented)
│   ├── baselines_utils.py          # Baseline evaluation utilities
│   ├── hf_utils.py                 # HuggingFace Hub utilities
│   ├── clews_utils.py              # CLEWS-specific utilities
│   ├── print_utils.py              # Logging utilities (documented)
│   └── pytorch_utils.py            # PyTorch helpers (documented)
│
├── scripts/                        # Executable scripts
│   ├── feature_extraction.py       # Extract embeddings (Whisper/SBERT/CLEWS)
│   ├── train.py                    # Train models
│   ├── inference.py                # Evaluate models
│   ├── compute_distance_matrix.py  # Compute pairwise distance matrices
│   ├── compute_baselines.py        # Compute transcription baselines
│   ├── multimodal_fusion.py        # Combine distance matrices from different modalities
│   └── upload_to_hf.py             # Upload models to HuggingFace Hub
│
├── requirements.txt                # Python dependencies
├── LICENSE                         # License file
└── README.md                       # This file (comprehensive documentation)
```

### Key Components

**Configuration System**: OmegaConf-based YAML configs with CLI overrides

**Data Pipeline**:
- `AudioDataset`: Raw audio loading for feature extraction
- `EmbeddingDataset`: Pre-extracted embedding loading for training
- `TranscriptionCache`: Efficient transcription caching with validation

**Model Architecture**:
- Transformer-based encoder with contrastive learning
- Support for multiple embedding types (Whisper, SBERT, CLEWS)
- Flexible pooling strategies (mean, max, attention)

**Evaluation Framework**:
- Standard metrics: MAP, MR1, ARP
- Baseline methods: SBERT, TF-IDF, theoretical bounds
- Multimodal fusion with grid search

**Documentation**:
- All utility modules have Google-style docstrings
- Type hints throughout the codebase
- Comprehensive README with detailed usage examples

---

## Citation

If you use this code in your research, please cite our paper:
```bibtex
@article{mancini2025wealy,
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
- **Open an issue**: [GitHub Issues](https://github.com/helemanc/audio-based-lyrics-matching/issues)
- **Email**: e.mancini@unibo.it


---

<div align="center">
<p><b>⭐ Star this repository if you find it useful!</b></p>
<p>Watch for updates as we continue adding features and improvements.</p>
</div>

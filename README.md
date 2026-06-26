# wDGA-BERT

**Detecting Word-Based DGA Botnets Using BERT Model and Features of Domain Names**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the source code for the paper *"Detecting Word-Based DGA Botnets Using BERT Model and Features of Domain Names"* (ICECCME 2026).

## Overview

Word-based Domain Generation Algorithms (wDGA) produce domain names by concatenating meaningful English words (e.g., `superior-generators.it`), making them significantly harder to detect than character-based DGAs. This project fine-tunes three BERT-based models — **BERT-base**, **DistilBERT**, and **SecureBERT** — under three input scenarios to evaluate the contribution of TLD and handcrafted lexical features.

| Scenario | Input format | Example |
|----------|-------------|---------|
| S1 | domain without TLD | `superior-generators` |
| S2 | domain with TLD | `superior-generators.it` |
| S3 | domain + TLD + 11 features | `superior-generators.it [SEP] f1=21 f2=generators... f11=0` |

## Results

**TABLE II — BERT-base across 3 scenarios**

| Metric | S1 (no TLD) | S2 (TLD) | S3 (TLD + features) |
|--------|------------|----------|---------------------|
| F1 (%) | 95.64 | 97.20 | **98.33** |
| FPR (%) | 4.51 | 3.26 | **1.76** |
| FNR (%) | 4.22 | 2.36 | **1.58** |

**TABLE III — 3 models × Scenario 3**

| Model | F1 (%) | AUC (%) | FPR (%) | FNR (%) |
|-------|--------|---------|---------|---------|
| DistilBERT | 98.15 | 99.13 | 1.66 | 2.03 |
| BERT-base | 98.33 | **99.66** | 1.76 | **1.58** |
| SecureBERT | **98.37** | 99.52 | **1.56** | 1.69 |

## Project Structure

```
wDGA-BERT/
├── build_dataset.py        # Build train/val/test split (leakage-free by e2LD)
├── train.py                # Train S1 & S2 (3 models × 2 scenarios)
├── train_3.py              # Train S3 (3 models × domain + TLD + 11 features)
├── test_external.py        # Evaluate S3 models on external wDGA families
├── genDomain.py            # Generate synthetic wDGA domains for external test
├── extractWord.py          # Meaningful word extraction from domain names
├── requirements.txt
├── README.md
│
├── dictionary/             # Word lists for feature extraction
│   ├── dictDGA.txt         # 9,274 words from 13 wDGA families
│   ├── dictOnlyDGA.txt     # 3,506 words unique to DGA (not in NLTK)
│   ├── dictOnlyNLTK.txt    # 228,630 words in NLTK but not in DGA
│   └── dictEng.txt         # 234,351 words (full NLTK English corpus)
│
├── Dataset/wordBased/      # Raw wDGA domain files (1 file per family)
│   ├── gozi_gpl.txt
│   ├── matsnu.txt
│   ├── suppobox.txt
│   └── ... (13 families)
│
├── dataout/                # Output of build_dataset.py
│   ├── train.csv           # 109,200 domains (domain;label)
│   ├── val.csv             # 23,400 domains
│   ├── test.csv            # 23,400 domains
│   └── stats.json
│
├── results/                # Output of train.py (S1/S2)
├── results_s3/             # Output of train_3.py (S3)
└── results_external/       # Output of test_external.py
```

## Setup

### 1. Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (adjust for your CUDA version)
# See https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. NLTK Data

```bash
python -c "import nltk; nltk.download('words')"
```

### 3. Dictionaries

Place the four dictionary files in `dictionary/`. These are derived from:
- **dictDGA**: union of word lists from 13 wDGA families (DGArchive)
- **dictEng**: NLTK `words` corpus (English dictionary)
- **dictOnlyDGA**: dictDGA − dictEng
- **dictOnlyNLTK**: dictEng − dictDGA

## Usage

### Step 1 — Build Dataset

```bash
# Edit DATA paths in build_dataset.py, then:
python build_dataset.py
```

Produces `train.csv`, `val.csv`, `test.csv` with leakage-free split by e2LD. Dataset: 13 wDGA families × 6,000 + 78,000 benign = 156,000 domains (70/15/15 split).

### Step 2 — Train S1 & S2

```bash
# Edit CONFIG in train.py, then:
python train.py
```

Trains 3 models × 2 scenarios (with/without TLD). Uses early stopping on validation F1. Energy tracked via CodeCarbon.

### Step 3 — Train S3

```bash
# Edit CONFIG in train_3.py, then:
python train_3.py
```

Trains 3 models on domain + TLD + 11 handcrafted features. Hyperparameters: MAX_LEN=128, batch=32, lr=2e-5, patience=5.

### Step 4 — External Evaluation

```bash
# Generate synthetic external domains:
python genDomain.py

# Test S3 models on external families:
python test_external.py
```

## 11 Features

| # | Feature | Description |
|---|---------|-------------|
| f1 | lenDomain | Length of full domain name |
| f2 | lsDGA | List of matching dictDGA words (or "none") |
| f3 | word_dga | Count of words in dictDGA |
| f4 | word_norm | Count of words in dictEng |
| f5 | lenDGA | Total character length of dictDGA-matched words |
| f6 | lenEng | Total character length of dictEng-matched words |
| f7 | lenOnlyNLTK | Character length of words only in NLTK (not in dictDGA) |
| f8 | maxDGA | Longest dictDGA word found |
| f9 | lenOnlyDGA | Character length of words only in dictDGA (not in NLTK) |
| f10 | dictDGA | Binary: 1 if any word matches dictDGA |
| f11 | hasDigit | Binary: 1 if domain contains digits |

## Hyperparameters

| Parameter | S1 / S2 | S3 |
|-----------|---------|-----|
| MAX_LEN | 64 | 128 |
| Batch size | 64 | 32 |
| Learning rate | 2e-5 | 2e-5 |
| Weight decay | 0.01 | 0.01 |
| Warmup ratio | 0.1 | 0.1 |
| Patience (early stop) | 5 | 5 |
| AMP | BF16 | BF16 |
| Seed | 42 | 42 |

## Hardware

Experiments were conducted on:
- GPU: NVIDIA GeForce RTX 5070 Ti
- CUDA: 12.8
- PyTorch: 2.11.0+cu128

## Citation

```bibtex
@inproceedings{vu2026wdga,
  title     = {Detecting Word-Based {DGA} Botnets Using {BERT} Model
               and Features of Domain Names},
  author    = {Vu, Xuan Hanh and Dau, Xuan Hoang and Trang, Thi Thu Ninh},
  booktitle = {Proc. International Conference on Electrical, Computer,
               Communications and Mechatronics Engineering (ICECCME)},
  year      = {2026},
  address   = {Bali, Indonesia}
}
```

## License

This project is released for academic research purposes.

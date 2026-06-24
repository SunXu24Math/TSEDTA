**TSEDTA**: a Transformer-based neural network with SMILES Transformer and ESM2 Embeddings for Drug-Target binding Affinity prediction.

To ensure long-term reproducibility, a persistent snapshot of the code and datasets used in the manuscript is archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19103249.svg)](https://doi.org/10.5281/zenodo.19103249)

## Dependencies

Key packages:

- Python 3.11
- PyTorch 2.12+ (CUDA 12.6)
- fair-esm 2.0
- scikit-learn
- RDKit
- pandas, numpy

See [`environment.yml`](environment.yml) for the full environment specification.

## Overview

This repository provides the implementation of TSEDTA, which combines:
- **SMILES Transformer** (Honda et al., 2019) for drug/molecular representation (embedding dimension: **256**)
- **ESM-2** (Lin et al., 2022) for protein sequence representation (embedding dimension: **1280**, using layer **33** output)

The two pretrained embeddings are fused through a Transformer encoder to predict drug-target binding affinity.

## Repository Structure

| File / Directory | Description |
|---|---|
| `README.md` | Project documentation (this file) |
| `environment.yml` | Conda environment specification |
| `data/` | Datasets and preprocessed files |
| `models/` | Trained model checkpoints |
| `dataset.py` | PyTorch `Dataset` class for drug-target pairs |
| `metrics.py` | Evaluation metrics (CI, MSE, RMSE, AUPR, $r_m^2$) |
| `preprocess_drug.py` | Extract drug embeddings via SMILES Transformer |
| `preprocess_protein.py` | Extract protein embeddings via ESM-2 |
| `process_data.py` | Tokenize sequences, encode with vocabularies, and split train/test |
| `main.py` | Model definition, training, and evaluation |
| `test.py` | Inference example on a single drug-target pair |

### Data File Format

Each `.txt` file in `data/` is space-separated with the following columns:

```
Drug_ID Protein_ID Drug_SMILES Amino_acid_sequence affinity
```

Example:

```
CHEMBL1087421 O00141 COC1=C... MTVKTEAAKGTLT... 11.1
```

## Environment Setup

### 1. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate tsedta
```

### 2. Install SMILES Transformer

```bash
cd TSEDTA
git clone https://github.com/DSPsleeporg/smiles-transformer.git
```

The pretrained checkpoint used is `smiles_transformer/trfm_12_23000.pkl` (embedding dimension: **256**).

> **Reference:**
>
> ```bibtex
> @article{honda2019smiles,
>      title={SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery},
>      author={Shion Honda and Shoi Shi and Hiroki R. Ueda},
>      year={2019},
>      eprint={1911.04738},
>      archivePrefix={arXiv},
>      primaryClass={cs.LG}
> }
> ```

### 3. Install ESM-2

```bash
pip install fair-esm
```

> **Note:** `fair-esm` is already listed in [`environment.yml`](environment.yml) under `pip` dependencies, so if you created the environment from the YAML file in Step 1, it is already installed and this step can be skipped.

The pretrained checkpoint used is `esm2_t33_650M_UR50D.pt`. Outputs from layer **33** are used as protein embeddings (dimension: **1280**).

> **Reference:**
> ```bibtex
> @article{lin2022language,
>     title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
>     author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
>     journal={bioRxiv},
>     year={2022},
>     publisher={Cold Spring Harbor Laboratory}
> }
> ```

## Training

Run the following scripts **in order**:

```bash
python preprocess_drug.py    # Step 1: Extract SMILES Transformer embeddings
python preprocess_protein.py # Step 2: Extract ESM-2 protein embeddings
python process_data.py       # Step 3: Tokenize and split train/test
python main.py               # Step 4: Train and evaluate the model
```

## Inference Example

```bash
python test.py
```

This runs inference on a sample drug-target pair using a trained checkpoint from `models/`.


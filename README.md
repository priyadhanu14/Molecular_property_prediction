# Molecular Property Prediction Pipeline

This project turns SMILES strings into molecular graphs with **RDKit**, trains baseline graph neural networks with **PyTorch Geometric**, and tracks reproducible experiments in **MLflow**.  
The implementation is based on the logic from `test_proj.ipynb`, but organized as a scriptable pipeline for repeatable training/inference.

## What this includes

- SMILES -> graph conversion (atom features, bond features, RDKit descriptors)
- Baseline GNNs:
  - `attentivefp` (PyG `AttentiveFP`)
  - `gine` (custom GINE classifier)
- Optional classical baseline:
  - RandomForest on ECFP4 + descriptors
- Reproducible train/valid/test split with fixed seed
- MLflow logging for params, losses, classification metrics, checkpoints, and predictions
- CLI inference from a saved checkpoint
- Simple Streamlit web UI for prediction demo

## Project layout

```text
.
|-- app.py                       # Streamlit UI
|-- run_demo.ps1                 # One-command setup/train/UI launcher (PowerShell)
|-- train.py                     # End-to-end training + MLflow logging
|-- predict.py                   # Batch/CLI inference
|-- requirements.txt
|-- data/
|   `-- sample_molecules.csv
|-- molprop/
|   |-- checkpoints.py
|   |-- data.py
|   |-- featurization.py
|   |-- models.py
|   `-- training.py
`-- test_proj.ipynb              # Reference notebook
```

## 1) Quick start (Windows PowerShell, one command)

```powershell
.\run_demo.ps1
```

This script:

1. creates `.venv` (if needed)
2. installs dependencies
3. trains a model (`attentivefp` by default)
4. launches MLflow UI (`http://127.0.0.1:5000`) and Streamlit (`http://127.0.0.1:8501`)

Useful flags:

```powershell
.\run_demo.ps1 -SkipInstall -SkipTrain          # just launch MLflow + Streamlit
.\run_demo.ps1 -Model gine -Epochs 40           # change training config
.\run_demo.ps1 -PythonCmd py -PythonVersion 3.11 # force a compatible Python
.\run_demo.ps1 -Headless                        # headless Streamlit mode
```

## 2) Environment setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PyTorch first (choose your CUDA/CPU build from pytorch.org), then PyG wheels matching your torch version:

```bash
python -m pip install torch torchvision torchaudio
python - <<'PY'
import torch
print(torch.__version__)
PY
# Replace <torch_version> with printed version (e.g. 2.5.1+cu121)
python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-<torch_version>.html
python -m pip install torch-geometric
```

Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

Compatibility note: use Python **3.9-3.12** (recommended: **3.11**).  
RDKit and some PyG dependency wheels are commonly unavailable on newer interpreters (e.g. 3.13+ / 3.14).

## 3) Data format

Training expects a CSV with:

- `smiles` column
- binary label column (default: `response`)

Example: `data/sample_molecules.csv`

You can also auto-download the Tox21 split used in the notebook:

```bash
python train.py --download-tox21 --data-path data/tox21_nr_ar.csv
```

## 4) Train a model (MLflow tracked)

### AttentiveFP baseline

```bash
python train.py \
  --data-path data/tox21_nr_ar.csv \
  --model attentivefp \
  --epochs 60 \
  --batch-size 256 \
  --experiment-name molecular-property-prediction
```

### GINE baseline

```bash
python train.py \
  --data-path data/tox21_nr_ar.csv \
  --model gine \
  --epochs 60 \
  --batch-size 256 \
  --experiment-name molecular-property-prediction
```

By default, `train.py` also trains/logs a RandomForest baseline (`--no-rf-baseline` disables it).

Artifacts are saved to:

- local MLflow store (`./mlruns` by default)
- checkpoints and prediction files (`./models`)

## 5) View experiments in MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open `http://127.0.0.1:5000`.

## 6) Run inference from checkpoint

```bash
python predict.py \
  --checkpoint models/<run_id>_attentivefp_best.pt \
  --smiles "CCO" \
  --smiles "c1ccccc1"
```

Or from CSV:

```bash
python predict.py \
  --checkpoint models/<run_id>_attentivefp_best.pt \
  --input-csv data/sample_molecules.csv \
  --smiles-col smiles \
  --output-csv outputs/predictions.csv
```

## 7) Run the simple web UI

```bash
streamlit run app.py
```

In the UI:

1. Enter a checkpoint path (e.g. `models/<run_id>_attentivefp_best.pt`)
2. Paste SMILES (one per line) or upload CSV
3. Click **Predict**

## Reproducibility details

- Deterministic seeds set for Python/NumPy/PyTorch
- Split indices are saved as JSON artifacts
- Full hyperparameters are logged in MLflow
- Best checkpoint is selected by validation loss and stored per run

## Notes

- Current training pipeline is set up for **binary classification** tasks.
- Invalid SMILES are dropped during preprocessing and counted in MLflow logs.
- `test_proj.ipynb` remains the exploratory reference; scripts in this repo are the productionized version.

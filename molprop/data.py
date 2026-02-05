from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from .featurization import smiles_to_data


class MoleculeGraphDataset(Dataset):
    """PyTorch dataset of molecular graphs built from SMILES strings."""

    def __init__(
        self,
        smiles: Iterable[str],
        labels: Optional[Iterable[int]] = None,
        show_progress: bool = True,
    ) -> None:
        smiles_list = list(smiles)
        labels_list = list(labels) if labels is not None else [None] * len(smiles_list)
        if len(smiles_list) != len(labels_list):
            raise ValueError("smiles and labels must have the same length")

        self.graphs = []
        self.source_indices = []
        self.dropped_indices = []

        iterator = zip(smiles_list, labels_list)
        if show_progress:
            iterator = tqdm(iterator, total=len(smiles_list), desc="Building graphs")

        for idx, (smi, label) in enumerate(iterator):
            graph = smiles_to_data(smi, label)
            if graph is None:
                self.dropped_indices.append(idx)
                continue
            self.graphs.append(graph)
            self.source_indices.append(idx)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def labels(self) -> np.ndarray:
        ys = [int(graph.y.item()) for graph in self.graphs if hasattr(graph, "y")]
        return np.asarray(ys, dtype=np.int64)


def load_tox21_dataset(destination_csv: str | Path, key: str = "nr-arsmiles") -> pd.DataFrame:
    destination_csv = Path(destination_csv)
    destination_csv.parent.mkdir(parents=True, exist_ok=True)

    input_smi_path = destination_csv.with_suffix(".smi")
    url = f"https://tripod.nih.gov/tox21/challenge/download?id={key}"
    urllib.request.urlretrieve(url, input_smi_path)

    df = pd.read_csv(input_smi_path, sep=r"\s+", header=None, engine="python")
    df.columns = ["smiles", "id", "response"]
    df.to_csv(destination_csv, index=False)
    return df


def load_smiles_dataframe(
    data_path: str | Path,
    smiles_col: str = "smiles",
    label_col: Optional[str] = "response",
) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.suffix.lower() == ".smi":
        df = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")
        if label_col is None:
            df.columns = [smiles_col]
        else:
            df.columns = [smiles_col, "id", label_col]
        return df

    df = pd.read_csv(data_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in {data_path}")

    if label_col is not None and label_col not in df.columns:
        if "label" in df.columns:
            df = df.rename(columns={"label": label_col})
        else:
            raise ValueError(f"Missing label column '{label_col}' in {data_path}")

    return df


def make_split_indices(
    num_examples: int,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    if num_examples < 3:
        raise ValueError("Need at least 3 examples to build train/valid/test splits")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError("valid_ratio must be between 0 and 1")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_examples)

    train_count = max(int(num_examples * train_ratio), 1)
    valid_count = max(int(num_examples * valid_ratio), 1)
    test_count = num_examples - train_count - valid_count

    if test_count < 1:
        deficit = 1 - test_count
        train_reduce = min(deficit, max(train_count - 1, 0))
        train_count -= train_reduce
        deficit -= train_reduce
        if deficit > 0:
            valid_reduce = min(deficit, max(valid_count - 1, 0))
            valid_count -= valid_reduce
            deficit -= valid_reduce
        test_count = num_examples - train_count - valid_count

    train_end = train_count
    valid_end = train_count + valid_count

    return {
        "train": indices[:train_end],
        "valid": indices[train_end:valid_end],
        "test": indices[valid_end:],
    }


def create_data_loaders(
    dataset: Dataset,
    split_indices: dict[str, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 0,
) -> tuple[dict[str, Subset], dict[str, DataLoader]]:
    subsets = {
        split: Subset(dataset, indices.tolist()) for split, indices in split_indices.items()
    }
    loaders = {
        split: DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
        for split, subset in subsets.items()
    }
    return subsets, loaders

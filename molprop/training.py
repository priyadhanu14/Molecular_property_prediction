from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "f1": float("nan"),
            "roc_auc": float("nan"),
        }

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1)

        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = batch.num_graphs
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    y_true = []
    y_pred = []
    y_prob = []

    for batch in loader:
        batch = batch.to(device)
        targets = batch.y.view(-1)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        batch_size = batch.num_graphs
        loss = criterion(logits, targets)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)

    metrics = compute_classification_metrics(y_true_arr, y_pred_arr, y_prob_arr)
    avg_loss = float("nan") if total_examples == 0 else total_loss / total_examples
    return avg_loss, metrics, y_true_arr, y_pred_arr, y_prob_arr


@torch.no_grad()
def predict_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_pred = []
    y_prob = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs = torch.softmax(logits, dim=1)
        y_pred.extend(probs.argmax(dim=1).cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())
    return np.asarray(y_pred, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)

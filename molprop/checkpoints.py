from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from .models import build_model


def save_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    model_name: str,
    model_kwargs: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[str | torch.device] = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    payload = torch.load(checkpoint_path, map_location=device)
    model = build_model(payload["model_name"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


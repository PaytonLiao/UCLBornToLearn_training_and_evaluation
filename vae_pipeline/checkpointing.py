from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_metrics: Dict[str, float],
    subset_name: str,
    config: Dict[str, Any],
    subset_names: Optional[List[str]] = None,
) -> None:
    """Persist checkpoint. subset_name remains the dataset slug for backward compatibility."""
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "best_metrics": best_metrics,
        "subset_name": subset_name,
        "subset_names": subset_names if subset_names is not None else [],
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

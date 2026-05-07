from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_STEP_BUDGETS = {
    "combined_recommended": 100_000,
    "combined_all": 100_000,
    "qdac_best_3of5_hits": 100_000,
    "qdac_baseline_1.5M": 100_000,
    "qdac_other_checkpoints": 100_000,
    "moe": 100_000,
    "random": 100_000,
}


@dataclass
class DataConfig:
    hf_repo_id: str
    subset_names: List[str] = field(default_factory=lambda: ["combined_recommended"])
    local_snapshot_root: str = ""
    cache_dir: str = ".hf_cache"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    normalization_stride: int = 10
    normalization_max_episodes: int = 400
    eval_samples_per_split: int = 50_000
    h5_pool_max_open_files: int = 64


@dataclass
class ModelConfig:
    input_dim: int = 380 + 17175
    latent_dim: int = 64
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [1024, 2048])


@dataclass
class TrainConfig:
    output_dir: str = "outputs/vae"
    experiment_name: str = "baseline"
    batch_size: int = 256
    num_workers: int = 0
    learning_rate: float = 3e-4
    max_steps: int = 100_000
    eval_every_steps: int = 2_000
    save_every_steps: int = 10_000
    log_every_steps: int = 100
    grad_clip_norm: float = 5.0
    patience_evals: int = 15
    min_delta: float = 1e-4
    pin_memory: bool = True
    deterministic: bool = True


def dataset_slug(subset_names: List[str]) -> str:
    """Filesystem-safe identifier for an experiment output directory."""
    if len(subset_names) == 1:
        return subset_names[0]
    return "__".join(subset_names)


def get_default_steps_for_subset(subset_name: str) -> int:
    return DEFAULT_STEP_BUDGETS.get(subset_name, 100_000)


def get_default_steps_for_subsets(subset_names: List[str]) -> int:
    """When combining subsets, use the maximum per-subset budget as a conservative default."""
    if not subset_names:
        return 100_000
    return max(get_default_steps_for_subset(name) for name in subset_names)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def serialize_config(
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
) -> Dict[str, Dict]:
    return {
        "data": asdict(data_config),
        "model": asdict(model_config),
        "train": asdict(train_config),
    }


def data_config_from_dict(data: Dict[str, Any]) -> DataConfig:
    """Build DataConfig from serialized dict; supports legacy checkpoints with only subset_name."""
    payload = dict(data)
    if "subset_names" not in payload and "subset_name" in payload:
        payload["subset_names"] = [str(payload.pop("subset_name"))]
    elif "subset_names" not in payload:
        payload["subset_names"] = ["combined_recommended"]
    payload.pop("subset_name", None)

    kwargs: Dict[str, Any] = {}
    for f in fields(DataConfig):
        if f.name in payload:
            kwargs[f.name] = payload[f.name]
            continue
        if f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required DataConfig field {f.name!r} in checkpoint/config dict.")
    return DataConfig(**kwargs)

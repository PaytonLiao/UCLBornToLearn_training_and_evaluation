from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset, IterableDataset

from vae_pipeline.config import DataConfig


@dataclass(frozen=True)
class EpisodeRef:
    key: str
    length: int


@dataclass
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_dict(payload: Dict) -> "NormalizationStats":
        return NormalizationStats(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

# deprecated because the path parsing was not robust enough
# def resolve_subset_h5_path(data_cfg: DataConfig) -> Path:
#     if data_cfg.local_snapshot_root:
#         root = Path(data_cfg.local_snapshot_root)
#         direct = root / data_cfg.subset_name / "dataset.h5"
#         if direct.exists():
#             return direct

#     snapshot_path = snapshot_download(
#         repo_id=data_cfg.hf_repo_id,
#         repo_type="dataset",
#         cache_dir=data_cfg.cache_dir,
#         allow_patterns=[f"{data_cfg.subset_name}/**"],
#     )
#     h5_path = Path(snapshot_path) / data_cfg.subset_name / "dataset.h5"
#     if not h5_path.exists():
#         raise FileNotFoundError(f"Could not locate dataset.h5 for subset={data_cfg.subset_name} at {h5_path}.")
#     return h5_path


def resolve_subset_h5_path(data_cfg) -> Path:
    # 1. Check local snapshot first
    if data_cfg.local_snapshot_root:
        root = Path(data_cfg.local_snapshot_root)
        candidates = list((root / data_cfg.subset_name).rglob("dataset.h5"))
        if candidates:
            return candidates[0]

    # 2. Download snapshot
    snapshot_path = snapshot_download(
        repo_id=data_cfg.hf_repo_id,
        repo_type="dataset",
        cache_dir=data_cfg.cache_dir,
        allow_patterns=[f"{data_cfg.subset_name}/**"],
    )

    subset_path = Path(snapshot_path) / data_cfg.subset_name

    # 3. Search recursively (handles seed/step subfolders)
    candidates = list(subset_path.rglob("dataset.h5"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not locate dataset.h5 for subset={data_cfg.subset_name} under {subset_path}"
        )

    # Optional: deterministic selection (important if multiple variants exist)
    candidates = sorted(candidates)

    return candidates[0]


def list_episodes(h5_path: Path) -> List[EpisodeRef]:
    episodes: List[EpisodeRef] = []
    with h5py.File(h5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        for k in keys:
            g = f[k]
            length = int(g.attrs.get("length", g["proprioception"].shape[0]))
            episodes.append(EpisodeRef(key=k, length=length))
    return episodes


def deterministic_episode_split(
    episodes: List[EpisodeRef],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[EpisodeRef]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(episodes))
    rng.shuffle(indices)

    n = len(episodes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_eps = [episodes[i] for i in indices[:n_train]]
    val_eps = [episodes[i] for i in indices[n_train : n_train + n_val]]
    test_eps = [episodes[i] for i in indices[n_train + n_val : n_train + n_val + n_test]]
    return {"train": train_eps, "val": val_eps, "test": test_eps}


def _concat_modalities(group: h5py.Group, timestep: int) -> np.ndarray:
    proprio = group["proprioception"][timestep].astype(np.float32)
    touch = group["touch"][timestep].astype(np.float32)
    return np.concatenate([proprio, touch], axis=0)


def compute_normalization_stats(
    h5_path: Path,
    episodes: List[EpisodeRef],
    stride: int,
    max_episodes: int,
) -> NormalizationStats:
    selected = episodes[: min(max_episodes, len(episodes))]
    running_sum = None
    running_sq_sum = None
    total = 0
    with h5py.File(h5_path, "r") as f:
        for ep in selected:
            g = f[ep.key]
            t_max = g["proprioception"].shape[0]
            for t in range(0, t_max, max(stride, 1)):
                x = _concat_modalities(g, t)
                if running_sum is None:
                    running_sum = np.zeros_like(x, dtype=np.float64)
                    running_sq_sum = np.zeros_like(x, dtype=np.float64)
                running_sum += x
                running_sq_sum += x * x
                total += 1
    if total == 0:
        raise ValueError("No samples collected for normalization.")
    mean = running_sum / total
    var = np.maximum(running_sq_sum / total - (mean * mean), 1e-8)
    std = np.sqrt(var)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


class RandomTimestepIterableDataset(IterableDataset):
    def __init__(
        self,
        h5_path: Path,
        episodes: List[EpisodeRef],
        norm_stats: NormalizationStats,
        seed: int,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.episodes = episodes
        self.mean = norm_stats.mean
        self.std = norm_stats.std
        self.seed = seed
        lengths = np.asarray([ep.length for ep in episodes], dtype=np.float64)
        self.probs = lengths / lengths.sum()

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(self.seed + worker_id)
        with h5py.File(self.h5_path, "r") as f:
            while True:
                ep_idx = int(rng.choice(len(self.episodes), p=self.probs))
                ep = self.episodes[ep_idx]
                t = int(rng.integers(0, ep.length))
                x = _concat_modalities(f[ep.key], t)
                x = (x - self.mean) / self.std
                yield torch.from_numpy(x.astype(np.float32))


class FixedTimestepDataset(Dataset):
    def __init__(
        self,
        h5_path: Path,
        episodes: List[EpisodeRef],
        norm_stats: NormalizationStats,
        sample_count: int,
        seed: int,
    ) -> None:
        self.h5_path = str(h5_path)
        self.episodes = episodes
        self.mean = norm_stats.mean
        self.std = norm_stats.std
        self.sample_index = self._build_sample_index(sample_count, seed)
        self._h5_file = None

    def _build_sample_index(self, sample_count: int, seed: int) -> List[Tuple[int, int]]:
        rng = np.random.default_rng(seed)
        lengths = np.asarray([ep.length for ep in self.episodes], dtype=np.float64)
        probs = lengths / lengths.sum()
        index: List[Tuple[int, int]] = []
        for _ in range(sample_count):
            ep_idx = int(rng.choice(len(self.episodes), p=probs))
            t = int(rng.integers(0, self.episodes[ep_idx].length))
            index.append((ep_idx, t))
        return index

    def _get_file(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r")
        return self._h5_file

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep_idx, t = self.sample_index[idx]
        print(ep_idx, t)
        ep = self.episodes[ep_idx]
        x = _concat_modalities(self._get_file()[ep.key], t)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32))


def build_dataloaders(
    h5_path: Path,
    split_eps: Dict[str, List[EpisodeRef]],
    norm_stats: NormalizationStats,
    batch_size: int,
    num_workers: int,
    eval_samples_per_split: int,
    seed: int,
    pin_memory: bool,
) -> Dict[str, DataLoader]:
    train_dataset = RandomTimestepIterableDataset(
        h5_path=h5_path,
        episodes=split_eps["train"],
        norm_stats=norm_stats,
        seed=seed,
    )
    val_dataset = FixedTimestepDataset(
        h5_path=h5_path,
        episodes=split_eps["val"],
        norm_stats=norm_stats,
        sample_count=eval_samples_per_split,
        seed=seed + 1,
    )
    test_dataset = FixedTimestepDataset(
        h5_path=h5_path,
        episodes=split_eps["test"],
        norm_stats=norm_stats,
        sample_count=eval_samples_per_split,
        seed=seed + 2,
    )
    return {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }


def save_split_manifest(path: Path, split_eps: Dict[str, List[EpisodeRef]]) -> None:
    payload = {k: [e.key for e in v] for k, v in split_eps.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_norm_stats(path: Path, stats: NormalizationStats) -> None:
    path.write_text(json.dumps(stats.to_dict()), encoding="utf-8")


def load_norm_stats(path: Path) -> NormalizationStats:
    return NormalizationStats.from_dict(json.loads(path.read_text(encoding="utf-8")))


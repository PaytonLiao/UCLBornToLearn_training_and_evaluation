from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import h5py
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset, IterableDataset

from vae_pipeline.config import DataConfig


@dataclass(frozen=True)
class EpisodeRef:
    """Uniquely identifies one episode inside one HDF5 file (keys may repeat across files)."""

    h5_path: Path
    episode_key: str
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


class H5FilePool:
    """Lazy HDF5 handles with a bounded number of simultaneously open files."""

    def __init__(self, max_open_files: int = 64) -> None:
        self.max_open_files = max(0, max_open_files)
        self._files: "OrderedDict[str, h5py.File]" = OrderedDict()

    def get(self, path: Path | str) -> h5py.File:
        key = str(Path(path).resolve())
        if key in self._files:
            self._files.move_to_end(key)
            return self._files[key]
        while self.max_open_files > 0 and len(self._files) >= self.max_open_files:
            _, old = self._files.popitem(last=False)
            old.close()
        handle = h5py.File(key, "r")
        self._files[key] = handle
        return handle

    def close_all(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()


def discover_h5_paths_for_subset(
    hf_repo_id: str,
    subset: str,
    *,
    local_snapshot_root: str = "",
    cache_dir: str = ".hf_cache",
) -> List[Path]:
    """Return all dataset.h5 paths under one logical subset (nested dirs supported)."""
    if local_snapshot_root:
        root = Path(local_snapshot_root)
        subset_root = root / subset
        if subset_root.is_dir():
            found = sorted(subset_root.rglob("dataset.h5"))
            if found:
                return [p.resolve() for p in found]

    snapshot_path = snapshot_download(
        repo_id=hf_repo_id,
        repo_type="dataset",
        cache_dir=cache_dir,
        allow_patterns=[f"{subset}/**"],
    )
    subset_path = Path(snapshot_path) / subset
    if not subset_path.is_dir():
        raise FileNotFoundError(f"Subset directory not found after download: {subset_path}")
    found = sorted(subset_path.rglob("dataset.h5"))
    if not found:
        raise FileNotFoundError(f"No dataset.h5 under subset={subset!r} at {subset_path}")
    return [p.resolve() for p in found]


def discover_all_h5_paths(data_cfg: DataConfig) -> List[Path]:
    """
    Resolve HDF5 paths for each subset name (separate HF downloads / cache lookups),
    then merge into one ordered list with duplicate paths removed.
    """
    seen: set[str] = set()
    ordered: List[Path] = []
    for subset in data_cfg.subset_names:
        for p in discover_h5_paths_for_subset(
            data_cfg.hf_repo_id,
            subset,
            local_snapshot_root=data_cfg.local_snapshot_root,
            cache_dir=data_cfg.cache_dir,
        ):
            key = str(p.resolve())
            if key not in seen:
                seen.add(key)
                ordered.append(p)
    return ordered


def list_episodes_for_h5_file(h5_path: Path) -> List[EpisodeRef]:
    path_resolved = h5_path.resolve()
    episodes: List[EpisodeRef] = []
    with h5py.File(path_resolved, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("episode_"))
        for k in keys:
            g = f[k]
            length = int(g.attrs.get("length", g["proprioception"].shape[0]))
            episodes.append(EpisodeRef(h5_path=path_resolved, episode_key=k, length=length))
    return episodes


def collect_episodes_from_h5_files(h5_paths: List[Path]) -> List[EpisodeRef]:
    """Deterministic global ordering: sorted unique files, then sorted episode keys within each file."""
    unique_sorted = sorted({p.resolve() for p in h5_paths})
    out: List[EpisodeRef] = []
    for p in unique_sorted:
        out.extend(list_episodes_for_h5_file(p))
    return out


def list_episodes(h5_path: Path) -> List[EpisodeRef]:
    """Backward-compatible helper for a single HDF5 file."""
    return list_episodes_for_h5_file(h5_path)


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
    episodes: List[EpisodeRef],
    stride: int,
    max_episodes: int,
    pool_max_open_files: int,
) -> NormalizationStats:
    selected = episodes[: min(max_episodes, len(episodes))]
    pool = H5FilePool(pool_max_open_files)
    running_sum: np.ndarray | None = None
    running_sq_sum: np.ndarray | None = None
    total = 0
    try:
        for ep in selected:
            g = pool.get(ep.h5_path)[ep.episode_key]
            t_max = g["proprioception"].shape[0]
            for t in range(0, t_max, max(stride, 1)):
                x = _concat_modalities(g, t)
                if running_sum is None:
                    running_sum = np.zeros_like(x, dtype=np.float64)
                    running_sq_sum = np.zeros_like(x, dtype=np.float64)
                running_sum += x
                running_sq_sum += x * x
                total += 1
    finally:
        pool.close_all()
    if total == 0:
        raise ValueError("No samples collected for normalization.")
    mean = running_sum / total  # type: ignore[operator]
    var = np.maximum(running_sq_sum / total - (mean * mean), 1e-8)  # type: ignore[operator]
    std = np.sqrt(var)
    return NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))


class RandomTimestepIterableDataset(IterableDataset):
    def __init__(
        self,
        episodes: List[EpisodeRef],
        norm_stats: NormalizationStats,
        seed: int,
        pool_max_open_files: int,
    ) -> None:
        super().__init__()
        self.episodes = episodes
        self.mean = norm_stats.mean
        self.std = norm_stats.std
        self.seed = seed
        self.pool_max_open_files = pool_max_open_files
        lengths = np.asarray([ep.length for ep in episodes], dtype=np.float64)
        self.probs = lengths / lengths.sum()

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rng = np.random.default_rng(self.seed + worker_id)
        pool = H5FilePool(self.pool_max_open_files)
        try:
            while True:
                ep_idx = int(rng.choice(len(self.episodes), p=self.probs))
                ep = self.episodes[ep_idx]
                t = int(rng.integers(0, ep.length))
                g = pool.get(ep.h5_path)[ep.episode_key]
                x = _concat_modalities(g, t)
                x = (x - self.mean) / self.std
                yield torch.from_numpy(x.astype(np.float32))
        finally:
            pool.close_all()


class FixedTimestepDataset(Dataset):
    def __init__(
        self,
        episodes: List[EpisodeRef],
        norm_stats: NormalizationStats,
        sample_count: int,
        seed: int,
        pool_max_open_files: int,
    ) -> None:
        self.episodes = episodes
        self.mean = norm_stats.mean
        self.std = norm_stats.std
        self.sample_index = self._build_sample_index(sample_count, seed)
        self.pool_max_open_files = pool_max_open_files
        self._pool: H5FilePool | None = None

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

    def _get_pool(self) -> H5FilePool:
        if self._pool is None:
            self._pool = H5FilePool(self.pool_max_open_files)
        return self._pool

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["_pool"] = None
        return state

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep_idx, t = self.sample_index[idx]
        ep = self.episodes[ep_idx]
        pool = self._get_pool()
        g = pool.get(ep.h5_path)[ep.episode_key]
        x = _concat_modalities(g, t)
        x = (x - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32))


def build_dataloaders(
    split_eps: Dict[str, List[EpisodeRef]],
    norm_stats: NormalizationStats,
    batch_size: int,
    num_workers: int,
    eval_samples_per_split: int,
    seed: int,
    pin_memory: bool,
    pool_max_open_files: int,
) -> Dict[str, DataLoader]:
    train_dataset = RandomTimestepIterableDataset(
        episodes=split_eps["train"],
        norm_stats=norm_stats,
        seed=seed,
        pool_max_open_files=pool_max_open_files,
    )
    val_dataset = FixedTimestepDataset(
        episodes=split_eps["val"],
        norm_stats=norm_stats,
        sample_count=eval_samples_per_split,
        seed=seed + 1,
        pool_max_open_files=pool_max_open_files,
    )
    test_dataset = FixedTimestepDataset(
        episodes=split_eps["test"],
        norm_stats=norm_stats,
        sample_count=eval_samples_per_split,
        seed=seed + 2,
        pool_max_open_files=pool_max_open_files,
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


def episode_ref_to_manifest(ep: EpisodeRef) -> Dict[str, object]:
    return {"h5_path": str(ep.h5_path), "episode_key": ep.episode_key, "length": ep.length}


def save_split_manifest(path: Path, split_eps: Dict[str, List[EpisodeRef]]) -> None:
    payload = {split: [episode_ref_to_manifest(e) for e in eps] for split, eps in split_eps.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_norm_stats(path: Path, stats: NormalizationStats) -> None:
    path.write_text(json.dumps(stats.to_dict()), encoding="utf-8")


def load_norm_stats(path: Path) -> NormalizationStats:
    return NormalizationStats.from_dict(json.loads(path.read_text(encoding="utf-8")))

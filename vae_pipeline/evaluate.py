from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import torch

from vae_pipeline.checkpointing import load_checkpoint
from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig
from vae_pipeline.data import (
    build_dataloaders,
    deterministic_episode_split,
    list_episodes,
    load_norm_stats,
    resolve_subset_h5_path,
)
from vae_pipeline.model import VanillaVAE
from vae_pipeline.train import evaluate_split, get_device


def evaluate_checkpoint(
    checkpoint_path: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> Dict:
    device = get_device()
    h5_path = resolve_subset_h5_path(data_cfg)
    episodes = list_episodes(h5_path)
    split_eps = deterministic_episode_split(
        episodes=episodes,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        test_ratio=data_cfg.test_ratio,
        seed=data_cfg.seed,
    )

    ckpt_path_obj = Path(checkpoint_path)
    norm_stats_path = ckpt_path_obj.parents[1] / "norm_stats.json"
    norm_stats = load_norm_stats(norm_stats_path)
    loaders = build_dataloaders(
        h5_path=h5_path,
        split_eps=split_eps,
        norm_stats=norm_stats,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        eval_samples_per_split=data_cfg.eval_samples_per_split,
        seed=data_cfg.seed,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
    )

    model = VanillaVAE(
        input_dim=model_cfg.input_dim,
        latent_dim=model_cfg.latent_dim,
        encoder_hidden_dims=model_cfg.encoder_hidden_dims,
        decoder_hidden_dims=model_cfg.decoder_hidden_dims,
    ).to(device)
    state = load_checkpoint(ckpt_path_obj, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    val_metrics = evaluate_split(model, loaders["val"], device)
    test_metrics = evaluate_split(model, loaders["test"], device)
    return {
        "checkpoint_path": checkpoint_path,
        "subset_name": data_cfg.subset_name,
        "device": str(device),
        "val": val_metrics,
        "test": test_metrics,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "data_config": asdict(data_cfg),
    }


def export_results(result_rows: List[Dict], output_json: str = "", output_csv: str = "") -> None:
    if output_json:
        Path(output_json).write_text(json.dumps(result_rows, indent=2), encoding="utf-8")
    if output_csv:
        rows: List[Dict] = []
        for r in result_rows:
            rows.append(
                {
                    "subset_name": r["subset_name"],
                    "checkpoint_path": r["checkpoint_path"],
                    "val_total_loss": r["val"]["total_loss"],
                    "val_recon_loss": r["val"]["recon_loss"],
                    "val_kl_div": r["val"]["kl_div"],
                    "val_elbo": r["val"]["elbo"],
                    "test_total_loss": r["test"]["total_loss"],
                    "test_recon_loss": r["test"]["recon_loss"],
                    "test_kl_div": r["test"]["kl_div"],
                    "test_elbo": r["test"]["elbo"],
                }
            )
        with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


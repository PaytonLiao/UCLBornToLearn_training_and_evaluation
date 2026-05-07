from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List



import torch

from vae_pipeline.checkpointing import load_checkpoint
from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig, dataset_slug
from vae_pipeline.data import (
    build_dataloaders,
    collect_episodes_from_h5_files,
    deterministic_episode_split,
    discover_all_h5_paths,
    load_norm_stats,
)
from vae_pipeline.model import VanillaVAE
from vae_pipeline.train import evaluate_split, get_device

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def compute_pca_latents(model, dataloader, device, max_samples):
    model.eval()

    latents = []

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_samples is not None and step >= max_samples:
                break
            # adapt depending on your dataset format
            x = batch.to(device)

            out = model(x)

            mu = out["mu"]  # IMPORTANT: use mu, not z

            latents.append(mu.cpu().numpy())

    latents = np.concatenate(latents, axis=0)

    pca = PCA(n_components=2)
    z_2d = pca.fit_transform(latents)

    return z_2d, pca.explained_variance_ratio_

def plot_pca_latent(z_2d, var_ratio, save_path):

    plt.figure(figsize=(6, 6))
    plt.scatter(z_2d[:, 0], z_2d[:, 1], s=2)

    plt.title(
        "PCA of VAE latent (test split)\n"
        f"Explained var: {var_ratio[0]:.3f}, {var_ratio[1]:.3f}"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def evaluate_checkpoint(
    checkpoint_path: str,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    max_samples: int | None = None,
) -> Dict:
    device = get_device()
    h5_paths = discover_all_h5_paths(data_cfg)
    episodes = collect_episodes_from_h5_files(h5_paths)
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
        split_eps=split_eps,
        norm_stats=norm_stats,
        batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers,
        eval_samples_per_split=data_cfg.eval_samples_per_split,
        seed=data_cfg.seed,
        pin_memory=train_cfg.pin_memory and device.type == "cuda",
        pool_max_open_files=data_cfg.h5_pool_max_open_files,
    )

    model = VanillaVAE(
        input_dim=model_cfg.input_dim,
        latent_dim=model_cfg.latent_dim,
        encoder_hidden_dims=model_cfg.encoder_hidden_dims,
        decoder_hidden_dims=model_cfg.decoder_hidden_dims,
    ).to(device)
    print("\n[INFO] Loading checkpoint...")
    print(f"[INFO] ckpt path: {ckpt_path_obj}")

    state = load_checkpoint(ckpt_path_obj, map_location=device)
    print("[INFO] Checkpoint loaded successfully")

    print("[INFO] Loading model weights...")
    model.load_state_dict(state["model_state_dict"])
    print("[INFO] Model state_dict loaded")

    slug = dataset_slug(data_cfg.subset_names)
    print(f"[INFO] Dataset slug: {slug}")

    print(f"[INFO] Computing PCA latents (max_samples={max_samples})...")
    z_2d, var_ratio = compute_pca_latents(model, loaders["test"], device, max_samples)
    print("[INFO] PCA computation complete")

    ckpt_path = Path(checkpoint_path)
    ckpt_dir = ckpt_path.parent              # .../checkpoints
    ckpt_name = ckpt_path.stem               # best_val_elbo
    out_dir = ckpt_dir / f"{ckpt_name}_pca_test_latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{max_samples}_samples.png"

    print(f"[INFO] Saving PCA plot to: {out_path}")
    plot_pca_latent(z_2d, var_ratio, out_path)
    print("[INFO] PCA plot saved")

    print(f"[INFO] Evaluating validation set (max_samples={max_samples})...")
    val_metrics = evaluate_split(model, loaders["val"], device, max_samples)
    print("[INFO] Validation evaluation complete")

    print(f"[INFO] Evaluating test set (max_samples={max_samples})...")
    test_metrics = evaluate_split(model, loaders["test"], device, max_samples)
    print("[INFO] Test evaluation complete")
    
    return {
        "checkpoint_path": checkpoint_path,
        "subset_names": data_cfg.subset_names,
        "subset_name": slug,
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
        if not result_rows:
            return
        rows: List[Dict] = []
        for r in result_rows:
            slug = r.get("subset_name") or ""
            rows.append(
                {
                    "subset_name": slug,
                    "subset_names": "|".join(r.get("subset_names") or []),
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

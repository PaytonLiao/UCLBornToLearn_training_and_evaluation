from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vae_pipeline.checkpointing import load_checkpoint, save_checkpoint, save_json
from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig, dataset_slug, ensure_dir, serialize_config
from vae_pipeline.data import (
    build_dataloaders,
    collect_episodes_from_h5_files,
    compute_normalization_stats,
    deterministic_episode_split,
    discover_all_h5_paths,
    save_norm_stats,
    save_split_manifest,
)
from vae_pipeline.model import VanillaVAE


def set_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_split(
    model: VanillaVAE,
    data_loader: Iterable,
    device: torch.device,
    max_steps: int | None = None,
) -> Dict[str, float]:
    model.eval()
    agg = {"total_loss": 0.0, "recon_loss": 0.0, "kl_div": 0.0, "elbo": 0.0}
    total_batches = 0

    with torch.no_grad():
        for step, x in enumerate(data_loader):
            if max_steps is not None and step >= max_steps:
                break

            x = x.to(device, non_blocking=True)
            out = model(x)
            losses = VanillaVAE.loss_function(x, out["recon"], out["mu"], out["logvar"])

            agg["total_loss"] += float(losses.total_loss.item())
            agg["recon_loss"] += float(losses.recon_loss.item())
            agg["kl_div"] += float(losses.kl_div.item())
            agg["elbo"] += float(losses.elbo.item())

            total_batches += 1

    if total_batches == 0:
        return {k: float("nan") for k in agg}

    return {k: v / total_batches for k, v in agg.items()}


def _log_metrics(writer: SummaryWriter, split: str, metrics: Dict[str, float], step: int) -> None:
    writer.add_scalar(f"{split}/total_loss", metrics["total_loss"], step)
    writer.add_scalar(f"{split}/reconstruction_loss", metrics["recon_loss"], step)
    writer.add_scalar(f"{split}/kl_divergence", metrics["kl_div"], step)
    writer.add_scalar(f"{split}/elbo", metrics["elbo"], step)


def train_vae(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    resume_ckpt: str = "",
) -> Dict:
    set_determinism(data_cfg.seed, train_cfg.deterministic)
    device = get_device()

    slug = dataset_slug(data_cfg.subset_names)
    output_root = ensure_dir(Path(train_cfg.output_dir) / slug / train_cfg.experiment_name)
    ckpt_dir = ensure_dir(output_root / "checkpoints")
    tb_dir = ensure_dir(output_root / "tensorboard")

    h5_paths = discover_all_h5_paths(data_cfg)
    episodes = collect_episodes_from_h5_files(h5_paths)
    split_eps = deterministic_episode_split(
        episodes,
        train_ratio=data_cfg.train_ratio,
        val_ratio=data_cfg.val_ratio,
        test_ratio=data_cfg.test_ratio,
        seed=data_cfg.seed,
    )
    norm_stats = compute_normalization_stats(
        episodes=split_eps["train"],
        stride=data_cfg.normalization_stride,
        max_episodes=data_cfg.normalization_max_episodes,
        pool_max_open_files=data_cfg.h5_pool_max_open_files,
    )

    save_split_manifest(output_root / "episode_split.json", split_eps)
    save_norm_stats(output_root / "norm_stats.json", norm_stats)
    save_json(
        output_root / "data_summary.json",
        {
            "subset_names": data_cfg.subset_names,
            "dataset_slug": slug,
            "h5_paths": [str(p) for p in h5_paths],
            "h5_file_count": len(h5_paths),
            "episode_counts": {k: len(v) for k, v in split_eps.items()},
            "total_timesteps_per_split": {
                k: int(sum(ep.length for ep in eps)) for k, eps in split_eps.items()
            },
        },
    )

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
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    start_epoch = 0
    start_step = 0
    best_metrics = {"val_elbo": float("inf"), "val_recon": float("inf")}
    if resume_ckpt:
        state = load_checkpoint(Path(resume_ckpt), map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"])
        start_step = int(state["step"])
        best_metrics.update(state.get("best_metrics", {}))

    config_payload = serialize_config(data_cfg, model_cfg, train_cfg)
    writer = SummaryWriter(log_dir=str(tb_dir))
    train_iter = iter(loaders["train"])

    patience_counter = 0
    global_step = start_step
    epoch = start_epoch
    max_steps = train_cfg.max_steps
    est_epochs = max(1, int(max_steps / max(1, len(split_eps["train"]))))

    progress = tqdm(total=max_steps, initial=global_step, desc=f"train:{slug}")
    while global_step < max_steps:
        model.train()
        x = next(train_iter).to(device, non_blocking=True)
        out = model(x)
        losses = VanillaVAE.loss_function(x, out["recon"], out["mu"], out["logvar"])
        optimizer.zero_grad(set_to_none=True)
        losses.total_loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()

        global_step += 1
        progress.update(1)
        if global_step % train_cfg.log_every_steps == 0:
            train_metrics = {
                "total_loss": float(losses.total_loss.item()),
                "recon_loss": float(losses.recon_loss.item()),
                "kl_div": float(losses.kl_div.item()),
                "elbo": float(losses.elbo.item()),
            }
            _log_metrics(writer, "train", train_metrics, global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            writer.add_scalar("meta/epoch", epoch, global_step)
            writer.add_scalar("meta/global_step", global_step, global_step)

        if global_step % train_cfg.eval_every_steps == 0:
            val_metrics = evaluate_split(model, loaders["val"], device)
            _log_metrics(writer, "val", val_metrics, global_step)

            improved_elbo = val_metrics["elbo"] < (best_metrics["val_elbo"] - train_cfg.min_delta)
            improved_recon = val_metrics["recon_loss"] < best_metrics["val_recon"]

            if improved_elbo:
                best_metrics["val_elbo"] = val_metrics["elbo"]
                save_checkpoint(
                    path=ckpt_dir / "best_val_elbo.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_metrics=best_metrics,
                    subset_name=slug,
                    subset_names=list(data_cfg.subset_names),
                    config=config_payload,
                )
                patience_counter = 0
            else:
                patience_counter += 1

            if improved_recon:
                best_metrics["val_recon"] = val_metrics["recon_loss"]
                save_checkpoint(
                    path=ckpt_dir / "best_val_recon.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                    best_metrics=best_metrics,
                    subset_name=slug,
                    subset_names=list(data_cfg.subset_names),
                    config=config_payload,
                )

            if patience_counter >= train_cfg.patience_evals:
                break

        if global_step % train_cfg.save_every_steps == 0:
            save_checkpoint(
                path=ckpt_dir / f"step_{global_step:08d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                best_metrics=best_metrics,
                subset_name=slug,
                subset_names=list(data_cfg.subset_names),
                config=config_payload,
            )

        if global_step % max(1, int(max_steps / est_epochs)) == 0:
            epoch += 1

    progress.close()

    best_elbo_path = ckpt_dir / "best_val_elbo.pt"
    if best_elbo_path.exists():
        state = load_checkpoint(best_elbo_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])

    final_val = evaluate_split(model, loaders["val"], device)
    final_test = evaluate_split(model, loaders["test"], device)
    _log_metrics(writer, "val_final", final_val, global_step)
    _log_metrics(writer, "test", final_test, global_step)

    results = {
        "device": str(device),
        "subset_names": data_cfg.subset_names,
        "dataset_slug": slug,
        "subset_name": slug,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "final_val": final_val,
        "final_test": final_test,
        "model_config": asdict(model_cfg),
        "train_config": asdict(train_cfg),
        "data_config": asdict(data_cfg),
    }
    save_json(output_root / "results.json", results)
    writer.close()
    return results

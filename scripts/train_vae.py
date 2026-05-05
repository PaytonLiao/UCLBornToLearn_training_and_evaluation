from __future__ import annotations

import argparse
import json

from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig, get_default_steps_for_subset
from vae_pipeline.train import train_vae


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train vanilla VAE on robot dataset subsets.")
    p.add_argument("--hf-repo-id", required=True, type=str)
    p.add_argument("--subset", type=str, default="combined_recommended")
    p.add_argument("--local-snapshot-root", type=str, default="")
    p.add_argument("--experiment-name", type=str, default="baseline")
    p.add_argument("--output-dir", type=str, default="outputs/vae")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-every-steps", type=int, default=2000)
    p.add_argument("--save-every-steps", type=int, default=10000)
    p.add_argument("--patience-evals", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume-ckpt", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_steps = args.max_steps if args.max_steps > 0 else get_default_steps_for_subset(args.subset)
    data_cfg = DataConfig(
        hf_repo_id=args.hf_repo_id,
        subset_name=args.subset,
        local_snapshot_root=args.local_snapshot_root,
        seed=args.seed,
    )
    model_cfg = ModelConfig(latent_dim=args.latent_dim)
    train_cfg = TrainConfig(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=max_steps,
        eval_every_steps=args.eval_every_steps,
        save_every_steps=args.save_every_steps,
        patience_evals=args.patience_evals,
        min_delta=args.min_delta,
        num_workers=args.num_workers,
    )
    result = train_vae(data_cfg, model_cfg, train_cfg, resume_ckpt=args.resume_ckpt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


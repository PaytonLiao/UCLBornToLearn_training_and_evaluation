from __future__ import annotations

import argparse
import json

from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig
from vae_pipeline.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained vanilla VAE checkpoint.")
    p.add_argument("--hf-repo-id", required=True, type=str)
    p.add_argument(
        "--subsets",
        nargs="+",
        default=None,
        help="Subset names used for training (must match checkpoint experiment for correct split + norm stats).",
    )
    p.add_argument("--subset", type=str, default=None, help="Deprecated: use --subsets with one name.")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--local-snapshot-root", type=str, default="")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--h5-pool-max-open-files", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.subsets is not None:
        subset_names = list(args.subsets)
    elif args.subset is not None:
        subset_names = [args.subset]
    else:
        raise ValueError("Provide --subsets or deprecated --subset.")

    data_cfg = DataConfig(
        hf_repo_id=args.hf_repo_id,
        subset_names=subset_names,
        local_snapshot_root=args.local_snapshot_root,
        seed=args.seed,
        h5_pool_max_open_files=args.h5_pool_max_open_files,
    )
    model_cfg = ModelConfig(latent_dim=args.latent_dim)
    train_cfg = TrainConfig(batch_size=args.batch_size, num_workers=args.num_workers)
    result = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

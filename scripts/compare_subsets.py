from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig
from vae_pipeline.evaluate import evaluate_checkpoint, export_results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare trained VAE checkpoints across subsets.")
    p.add_argument("--hf-repo-id", required=True, type=str)
    p.add_argument("--subsets", nargs="+", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--local-snapshot-root", type=str, default="")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=str, default="outputs/vae/subset_comparison.json")
    p.add_argument("--output-csv", type=str, default="outputs/vae/subset_comparison.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.subsets) != len(args.checkpoints):
        raise ValueError("subsets and checkpoints must have equal lengths.")

    rows: List[dict] = []
    for subset, checkpoint in zip(args.subsets, args.checkpoints):
        data_cfg = DataConfig(
            hf_repo_id=args.hf_repo_id,
            subset_name=subset,
            local_snapshot_root=args.local_snapshot_root,
            seed=args.seed,
        )
        model_cfg = ModelConfig(latent_dim=args.latent_dim)
        train_cfg = TrainConfig(batch_size=args.batch_size)
        rows.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
            )
        )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    export_results(rows, output_json=args.output_json, output_csv=args.output_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


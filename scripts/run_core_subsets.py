from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from vae_pipeline.config import DataConfig, ModelConfig, TrainConfig, get_default_steps_for_subset
from vae_pipeline.evaluate import export_results
from vae_pipeline.train import train_vae


CORE_SUBSETS = ["combined_recommended", "combined_all", "qdac_best_3of5_hits"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train core VAE subset baselines sequentially and export a consolidated leaderboard."
    )
    p.add_argument("--hf-repo-id", required=True, type=str)
    p.add_argument("--local-snapshot-root", type=str, default="")
    p.add_argument("--output-dir", type=str, default="outputs/vae")
    p.add_argument("--experiment-prefix", type=str, default="core_baseline")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--eval-every-steps", type=int, default=2000)
    p.add_argument("--save-every-steps", type=int, default=10000)
    p.add_argument("--patience-evals", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--subset-max-steps",
        nargs="*",
        default=[],
        help="Optional overrides as subset=steps (e.g. combined_all=160000).",
    )
    p.add_argument("--summary-json", type=str, default="outputs/vae/core_subsets_summary.json")
    p.add_argument("--summary-csv", type=str, default="outputs/vae/core_subsets_summary.csv")
    p.add_argument("--full-results-json", type=str, default="outputs/vae/core_subsets_full_results.json")
    return p.parse_args()


def parse_subset_max_steps(pairs: List[str]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Invalid subset-max-steps entry: {item}. Expected subset=steps.")
        subset, value = item.split("=", 1)
        overrides[subset.strip()] = int(value)
    return overrides


def main() -> None:
    args = parse_args()
    max_step_overrides = parse_subset_max_steps(args.subset_max_steps)

    rows = []
    eval_export_rows = []
    for subset in CORE_SUBSETS:
        max_steps = max_step_overrides.get(subset, get_default_steps_for_subset(subset))
        experiment_name = f"{args.experiment_prefix}_{subset}"

        data_cfg = DataConfig(
            hf_repo_id=args.hf_repo_id,
            subset_name=subset,
            local_snapshot_root=args.local_snapshot_root,
            seed=args.seed,
        )
        model_cfg = ModelConfig(latent_dim=args.latent_dim)
        train_cfg = TrainConfig(
            output_dir=args.output_dir,
            experiment_name=experiment_name,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=max_steps,
            eval_every_steps=args.eval_every_steps,
            save_every_steps=args.save_every_steps,
            patience_evals=args.patience_evals,
            min_delta=args.min_delta,
            num_workers=args.num_workers,
        )

        result = train_vae(data_cfg=data_cfg, model_cfg=model_cfg, train_cfg=train_cfg)
        rows.append(result)

        best_ckpt = (
            Path(args.output_dir)
            / subset
            / experiment_name
            / "checkpoints"
            / "best_val_elbo.pt"
        )
        eval_export_rows.append(
            {
                "subset_name": subset,
                "checkpoint_path": str(best_ckpt),
                "val": result["final_val"],
                "test": result["final_test"],
            }
        )

    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.full_results_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    export_results(eval_export_rows, output_json=args.summary_json, output_csv=args.summary_csv)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()


# Vanilla VAE Pipeline for Robot Sensor Episodes

## 1) Training Plan

1. For each requested subset name, resolve **all** nested `dataset.h5` files under that subset directory (Hugging Face cache download per subset, or local snapshot). Optionally combine multiple subsets into one logical training corpus.
2. Enumerate episodes across **all** HDF5 files (each episode is `(file_path, episode_key)`), then apply one deterministic global episode-level split (`train/val/test`) with a seed to avoid timestep leakage.
3. Compute normalization statistics from training episodes only (streaming, strided sampling), then apply z-score normalization to all splits.
4. Train a **vanilla VAE** on concatenated `proprioception + touch` vectors (dim `380 + 17175 = 17555`) with step-based optimization.
5. Evaluate periodically on validation split; track and checkpoint:
   - best validation ELBO (used for early stopping and restoration)
   - best validation reconstruction loss
   - periodic snapshots every N steps
6. Stop when max steps is reached or early stopping patience is exceeded; restore best ELBO weights.
7. Evaluate final model on validation and test splits.
8. Compare different subset-trained checkpoints with a single comparison script and export JSON/CSV.

## 2) Project Structure

```text
.
├── requirements.txt
├── README.md
├── scripts
│   ├── train_vae.py
│   ├── eval_vae.py
│   ├── compare_subsets.py
│   └── run_core_subsets.py
└── vae_pipeline
    ├── __init__.py
    ├── config.py
    ├── data.py
    ├── model.py
    ├── checkpointing.py
    ├── train.py
    └── evaluate.py
```

## 3) Implementation Notes

- **Vanilla VAE only:** reconstruction MSE + KL divergence, with `ELBO = recon + KL`.
- **GPU-agnostic:** automatic `torch.device("cuda" if available else "cpu")`; all tensors/models moved with `.to(device)`.
- **Memory efficiency:** random timestep sampling from HDF5 episodes using lazy file handles; no full dataset materialization.
- **Episode-based split:** deterministic shuffling at episode level with fixed seed.
- **Determinism:** seeds for Python, NumPy, and PyTorch; optional deterministic cuDNN.
- **TensorBoard logs:** train/val/test total loss, reconstruction loss, KL, ELBO, LR, grad norm, step, epoch.
- **Checkpoint metadata:** epoch, step, best metrics, dataset slug (`subset_name` field), optional `subset_names` list, and full config payload.
- **Multiple HDF5 files:** LRU-bounded pool of open `h5py.File` handles (`h5_pool_max_open_files`) to stay scalable without exhausting OS file descriptors.
- **Output directories:** `outputs/vae/<dataset_slug>/<experiment_name>/` where `dataset_slug` is a single subset name or `name__name__...` when combining subsets.

## 4) Recommended Hyperparameters

Baseline defaults (configured in code/CLI):

- `latent_dim`: `64`
- `batch_size`: `256` (reduce for CPU or low-memory GPUs)
- `learning_rate`: `3e-4` (Adam)
- `grad_clip_norm`: `5.0`
- `eval_every_steps`: `2000`
- `save_every_steps`: `10000`
- `patience_evals`: `15`
- `min_delta`: `1e-4`
- `normalization_stride`: `10`
- `eval_samples_per_split`: `50_000`

Step-budget defaults per subset name are listed in `vae_pipeline/config.py`. When you pass `--subsets a b c`, the training script defaults to `max(default_steps(a), default_steps(b), default_steps(c))`. Override globally with `--max-steps`.

Relation to episodes:

- Step-based training samples random timesteps from train episodes, so larger subsets expose more diversity per step.
- For roughly fixed batch size, total optimizer updates scale with subset complexity rather than raw episode length alone.
- Early stopping on validation ELBO prevents overtraining when the subset saturates before budget.

## 5) Run Instructions (CPU or GPU)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train on one or more HF subset directories; every nested `dataset.h5` under those directories is merged (CUDA is used automatically when available).

### Bash / WSL (Linux / macOS)

```bash
python -m scripts.train_vae \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --subsets combined_recommended moe random \
  --experiment-name vae_concat_baseline
```

Single subset (equivalent to one directory tree):

```bash
python -m scripts.train_vae \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --subsets combined_recommended \
  --experiment-name vae_concat_baseline
```

Legacy flag (deprecated): `--subset combined_recommended` still maps to `subset_names=[combined_recommended]`.

### PowerShell (Windows)

```powershell
python -m scripts.train_vae `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --subsets combined_recommended moe random `
  --experiment-name vae_concat_baseline
```

---

## Using a Local Snapshot

Use a local dataset snapshot instead of downloading:

### Bash / WSL

```bash
python -m scripts.train_vae \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --local-snapshot-root /path/to/local_snapshot \
  --subsets combined_all \
  --experiment-name vae_combined_all
```

### PowerShell

```powershell
python -m scripts.train_vae `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --local-snapshot-root C:\path\to\local_snapshot `
  --subsets combined_all `
  --experiment-name vae_combined_all
```

---

## Evaluation

Evaluate a checkpoint; **`--subsets` must match the training run** (same split + `norm_stats.json` beside the checkpoint).

### Bash / WSL

```bash
python -m scripts.eval_vae \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --subsets combined_recommended moe random \
  --checkpoint outputs/vae/combined_recommended__moe__random/vae_concat_baseline/checkpoints/best_val_elbo.pt
```

### PowerShell

```powershell
python -m scripts.eval_vae `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --subsets combined_recommended moe random `
  --checkpoint outputs/vae/combined_recommended__moe__random/vae_concat_baseline/checkpoints/best_val_elbo.pt
```

---

## Compare Subsets

Compare subset-trained models:

### Bash / WSL

```bash
python -m scripts.compare_subsets \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --subsets combined_recommended combined_all qdac_best_3of5_hits \
  --checkpoints \
    outputs/vae/combined_recommended/exp/checkpoints/best_val_elbo.pt \
    outputs/vae/combined_all/exp/checkpoints/best_val_elbo.pt \
    outputs/vae/qdac_best_3of5_hits/exp/checkpoints/best_val_elbo.pt
```

### PowerShell

```powershell
python -m scripts.compare_subsets `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --subsets combined_recommended combined_all qdac_best_3of5_hits `
  --checkpoints `
    outputs/vae/combined_recommended/exp/checkpoints/best_val_elbo.pt `
    outputs/vae/combined_all/exp/checkpoints/best_val_elbo.pt `
    outputs/vae/qdac_best_3of5_hits/exp/checkpoints/best_val_elbo.pt
```

---

## TensorBoard

```bash
tensorboard --logdir outputs/vae
```

---

## Run Core Subsets

Train all three core subsets sequentially and export a consolidated leaderboard:

### Bash / WSL

```bash
python -m scripts.run_core_subsets \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --experiment-prefix core_baseline
```

### PowerShell

```powershell
python -m scripts.run_core_subsets `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --experiment-prefix core_baseline
```

---

## Optional Step Overrides

Override max training steps per subset:

### Bash / WSL

```bash
python -m scripts.run_core_subsets \
  --hf-repo-id BornToLearnUCL/born_to_learn \
  --subset-max-steps combined_recommended=90000 combined_all=160000 qdac_best_3of5_hits=45000
```

### PowerShell

```powershell
python -m scripts.run_core_subsets `
  --hf-repo-id BornToLearnUCL/born_to_learn `
  --subset-max-steps combined_recommended=90000 combined_all=160000 qdac_best_3of5_hits=45000
```

Outputs per experiment:

- `results.json`
- `episode_split.json`
- `norm_stats.json`
- `data_summary.json`
- `checkpoints/*.pt`
- `tensorboard/*`
- consolidated files from `run_core_subsets.py`:
  - `core_subsets_summary.json`
  - `core_subsets_summary.csv`
  - `core_subsets_full_results.json`

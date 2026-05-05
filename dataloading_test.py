from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np


def iter_episodes(h5_path):
    with h5py.File(h5_path, "r") as f:
        for k in sorted(k for k in f.keys() if k.startswith("episode_")):
            g = f[k]
            yield {
                "proprio":  g["proprioception"][:],
                "touch":    g["touch"][:],
                "action":   g["action"][:],
                "reward":   g["reward"][:],
                "timestamp": g["timestamp"][:],
                "seed": int(g.attrs.get("seed", -1)),
                "length": int(g.attrs.get("length", g["proprioception"].shape[0])),
                "total_reward": float(g.attrs.get("total_reward", 0.0)),
            }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--max-eps", type=int, default=3)
    args = p.parse_args()
    for i, ep in enumerate(iter_episodes(args.h5)):
        print(f"ep {i}: T={ep['length']}, "
              f"proprio.shape={ep['proprio'].shape}, "
              f"touch.shape={ep['touch'].shape}, "
              f"any_touch={(ep['touch'].sum(axis=1) > 1e-2).mean():.3f}, "
              f"total_reward={ep['total_reward']:.2f}")
        if i + 1 >= args.max_eps:
            break


if __name__ == "__main__":
    main()
    
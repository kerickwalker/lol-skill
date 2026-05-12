#!/usr/bin/env python
"""
Training runner for LCK skill models.

Examples:
    python train.py                              # train fast model, 1500 steps
    python train.py --model corr --n-steps 3000
    python train.py --model relationship
    python train.py --output my_run             # saves params/my_run.pt, elbo/my_run.png
    python train.py --load params/fast_20240101_120000.pt  # load and print rankings
    python train.py --use-diff-stats            # fast model with diff stats
"""

import argparse
import importlib
import inspect
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pyro
import torch


def _load_params(path):
    """Load a Pyro param store from disk. Works around PyTorch 2.6+ weights_only default."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)

MODELS = ["fast", "corr", "relationship"]
DEFAULT_MODEL = "fast"
CSV_PATH = "data/lck_s15_games_MODEL-READY.csv"


def main():
    parser = argparse.ArgumentParser(description="Train an LCK skill model")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODELS,
        help=f"Model to train (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-n", "--n-steps",
        type=int,
        default=1500,
        help="Number of SVI training steps (default: 1500)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output name for params/elbo files (default: {model}_{timestamp})",
    )
    parser.add_argument(
        "--csv-path",
        default=CSV_PATH,
        help=f"Path to model-ready CSV (default: {CSV_PATH})",
    )
    parser.add_argument(
        "--load",
        metavar="FILE",
        help="Load saved params from FILE and print rankings (skips training)",
    )
    # fast-model flags
    parser.add_argument(
        "--use-team-stats",
        action="store_true",
        help="(fast model) Use team-level output stats as context",
    )
    parser.add_argument(
        "--use-diff-stats",
        action="store_true",
        help="(fast model) Use same-role opponent difference stats",
    )
    args = parser.parse_args()

    module = importlib.import_module(f"models.{args.model}")
    batch, n_players, idx_to_name, primary_role = module.load_data(args.csv_path)

    if args.load:
        pyro.clear_param_store()
        _load_params(args.load)
        print(f"Loaded params from {args.load}")
    else:
        output_name = args.output or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        params_path = Path("params") / f"{output_name}.pt"
        elbo_path = Path("elbo") / f"{output_name}.png"
        params_path.parent.mkdir(exist_ok=True)
        elbo_path.parent.mkdir(exist_ok=True)

        # Pass model-specific kwargs only if the train function accepts them
        train_sig = inspect.signature(module.train)
        extra = {}
        if "use_team_stats" in train_sig.parameters:
            extra["use_team_stats"] = args.use_team_stats
        if "use_diff_stats" in train_sig.parameters:
            extra["use_diff_stats"] = args.use_diff_stats

        start = time.perf_counter()
        losses = module.train(batch, n_players, n_steps=args.n_steps, lr=args.lr, **extra)
        elapsed = time.perf_counter() - start

        pyro.get_param_store().save(str(params_path))
        print(f"Saved params to {params_path}")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("ELBO loss")
        plt.title(f"SVI convergence ({args.model})")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(elbo_path)
        plt.close()
        print(f"Saved loss curve to {elbo_path}")
        print(f"Training wall time: {elapsed:.2f}s")

    module.print_rankings(n_players, idx_to_name, primary_role)


if __name__ == "__main__":
    main()

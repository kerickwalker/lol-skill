#!/usr/bin/env python
"""Train LCK S15 skill models."""

from __future__ import annotations

import argparse
import importlib
import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch


MODELS = ["baseline", "game_rules", "role_alpha_tau", "role_corr"]
DEFAULT_MODEL = "baseline"
DEFAULT_CSV_PATH = "data/lck_s15_games_MODEL-READY_train.csv"


def _load_params(path):
    """Load a Pyro param store from disk despite PyTorch's weights_only default."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train an LCK skill model")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=MODELS,
        help=f"Model to train (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-n",
        "--n-steps",
        type=int,
        default=1500,
        help="Number of SVI training steps (default: 1500)",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument(
        "--output",
        default="",
        help="Output name for params/elbo/score files (default: {model}_{timestamp})",
    )
    parser.add_argument(
        "--csv-path",
        default=DEFAULT_CSV_PATH,
        help=f"Path to model-ready CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument("--load", metavar="FILE", help="Load saved params and print scores")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible SVI runs")
    args = parser.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)

    module = importlib.import_module(f"models.{args.model}")
    batch, n_players, idx_to_name, primary_role = module.load_data(args.csv_path)

    output_name = args.output
    if args.load:
        pyro.clear_param_store()
        _load_params(args.load)
        print(f"Loaded params from {args.load}")
        if not output_name:
            output_name = Path(args.load).stem
    else:
        output_name = output_name or f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        params_path = Path("params") / f"{output_name}.pt"
        elbo_path = Path("elbo") / f"{output_name}.png"
        params_path.parent.mkdir(exist_ok=True)
        elbo_path.parent.mkdir(exist_ok=True)

        start = time.perf_counter()
        losses = module.train(batch, n_players, n_steps=args.n_steps, lr=args.lr)
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

    if not getattr(module, "USE_SCORE_TABLE_AS_MAIN_OUTPUT", False):
        module.print_rankings(n_players, idx_to_name, primary_role)
    if hasattr(module, "print_score_table"):
        module.print_score_table(n_players, idx_to_name, primary_role)
    if hasattr(module, "build_player_score_table"):
        score_path = Path("data") / "analysis" / f"{output_name}_player_scores.csv"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        module.build_player_score_table(n_players, idx_to_name, primary_role).to_csv(
            score_path,
            index=False,
        )
        print(f"\nSaved player scores to {score_path}")


if __name__ == "__main__":
    main()

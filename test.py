#!/usr/bin/env python
"""
Evaluate a trained LCK skill model on held-out games, or predict a single matchup.

Win probability uses the same formula as training:
    Phi((sum_skill_A - sum_skill_B) / sqrt(2))

Evaluate mode (default):
    python test.py --params params/relationship_20260512_182140.pt
    python test.py --params params/relationship_20260512_182140.pt --csv-path data/lck_s15_games_MODEL-READY_test.csv

Predict mode (--team-a and --team-b, 5 players each as Name:ROLE):
    python test.py --params params/relationship_20260512_182140.pt \\
        --team-a "Faker:MID" "Gumayusi:ADC" "Keria:SUPPORT" "Zeus:TOP" "Oner:JUNGLE" \\
        --team-b "Chovy:MID" "Peyz:ADC" "Delight:SUPPORT" "Doran:TOP" "Canyon:JUNGLE"
"""

import argparse
import math

import pandas as pd
import pyro
import torch

from models.config import ROLE_MAP


def _load_params(path):
    """Load a Pyro param store from disk. Works around PyTorch 2.6+ weights_only default."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)

BETA_O = 1.0
DEFAULT_TEST_CSV = "data/lck_s15_games_MODEL-READY_test.csv"
# Full dataset used for player ID → index mapping, which must match what was used during training.
DEFAULT_REF_CSV = "data/lck_s15_games_MODEL-READY.csv"

ROLE_ALIASES = {
    "JNG": "JUNGLE", "JG": "JUNGLE",
    "BOT": "ADC",
    "SUP": "SUPPORT", "SUPP": "SUPPORT",
}
ROLE_IDX_TO_NAME = {v: k for k, v in ROLE_MAP.items()}


def _win_prob(s_loc, pid_a, role_a, pid_b, role_b):
    skill_a = s_loc[pid_a, role_a].sum(-1)
    skill_b = s_loc[pid_b, role_b].sum(-1)
    diff = (skill_a - skill_b) / (math.sqrt(2) * BETA_O)
    return torch.distributions.Normal(0.0, 1.0).cdf(diff)


def _load_player_maps(ref_csv):
    df = pd.read_csv(ref_csv, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()
    unique_pids = sorted(df["player_id"].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    name_to_pid = (
        df[["player_name", "player_id"]]
        .drop_duplicates()
        .set_index("player_name")["player_id"]
        .to_dict()
    )
    return pid_to_idx, name_to_pid, len(unique_pids)


def evaluate(args):
    pid_to_idx, _, n_players = _load_player_maps(args.ref_csv)

    test_df = pd.read_csv(args.csv_path, encoding="utf-8-sig", sep=None, engine="python")
    test_df.columns = test_df.columns.str.strip()
    test_df["role_idx"] = test_df["role"].map(ROLE_MAP)
    test_df["pid_idx"] = test_df["player_id"].map(pid_to_idx)

    pyro.clear_param_store()
    _load_params(args.params)
    s_loc = pyro.param("s_loc")

    if s_loc.shape[0] != n_players:
        print(
            f"WARNING: params have {s_loc.shape[0]} players but ref CSV has {n_players}. "
            "Ensure --params and --ref-csv come from the same training run."
        )

    pid_a_rows, role_a_rows, pid_b_rows, role_b_rows = [], [], [], []
    skipped = 0
    for _, group in test_df.groupby("game_block_id", sort=True):
        winners = group[group["Result"] == "Victory"].sort_values("role_idx")
        losers = group[group["Result"] == "Defeat"].sort_values("role_idx")
        if len(winners) != 5 or len(losers) != 5:
            skipped += 1
            continue
        if winners["pid_idx"].isna().any() or losers["pid_idx"].isna().any():
            skipped += 1
            continue
        pid_a_rows.append(torch.tensor(winners["pid_idx"].to_numpy(int), dtype=torch.long))
        role_a_rows.append(torch.tensor(winners["role_idx"].to_numpy(int), dtype=torch.long))
        pid_b_rows.append(torch.tensor(losers["pid_idx"].to_numpy(int), dtype=torch.long))
        role_b_rows.append(torch.tensor(losers["role_idx"].to_numpy(int), dtype=torch.long))

    if not pid_a_rows:
        print("No valid games found in the test CSV.")
        return

    pid_a = torch.stack(pid_a_rows)
    role_a = torch.stack(role_a_rows)
    pid_b = torch.stack(pid_b_rows)
    role_b = torch.stack(role_b_rows)

    with torch.no_grad():
        probs = _win_prob(s_loc, pid_a, role_a, pid_b, role_b)

    # Team A is always the winner (Result == Victory) in the dataset.
    n_games = len(pid_a_rows)
    accuracy = (probs > 0.5).float().mean().item()
    brier = ((probs - 1.0) ** 2).mean().item()
    log_loss = -torch.log(probs.clamp(min=1e-7)).mean().item()

    print(f"Params : {args.params}")
    print(f"Test   : {args.csv_path}  ({n_games} games{f', {skipped} skipped' if skipped else ''})")
    print()
    print(f"  Accuracy   : {accuracy:.1%}")
    print(f"  Brier score: {brier:.4f}  (0.00 = perfect, 0.25 = random)")
    print(f"  Log-loss   : {log_loss:.4f}  (0.00 = perfect, 0.693 = random)")


def predict(args):
    pid_to_idx, name_to_pid, _ = _load_player_maps(args.ref_csv)

    pyro.clear_param_store()
    _load_params(args.params)
    s_loc = pyro.param("s_loc")

    def parse_team(entries):
        players = []
        for entry in entries:
            if ":" not in entry:
                raise ValueError(f"Expected 'Name:ROLE', got '{entry!r}'")
            name, role_str = entry.rsplit(":", 1)
            role_str = role_str.upper()
            role_str = ROLE_ALIASES.get(role_str, role_str)
            if name not in name_to_pid:
                raise ValueError(
                    f"Unknown player {name!r}.\nKnown players: {sorted(name_to_pid)}"
                )
            if role_str not in ROLE_MAP:
                raise ValueError(
                    f"Unknown role {role_str!r}. Valid: {list(ROLE_MAP)}"
                )
            players.append((name, pid_to_idx[name_to_pid[name]], ROLE_MAP[role_str]))
        return players

    team_a = parse_team(args.team_a)
    team_b = parse_team(args.team_b)

    pid_a = torch.tensor([p[1] for p in team_a], dtype=torch.long)
    role_a = torch.tensor([p[2] for p in team_a], dtype=torch.long)
    pid_b = torch.tensor([p[1] for p in team_b], dtype=torch.long)
    role_b = torch.tensor([p[2] for p in team_b], dtype=torch.long)

    with torch.no_grad():
        prob = _win_prob(s_loc, pid_a, role_a, pid_b, role_b).item()

    def fmt_team(players):
        return "  " + "\n  ".join(
            f"{name:<20} ({ROLE_IDX_TO_NAME[role_idx]})"
            for name, _, role_idx in players
        )

    print(f"\nTeam A:\n{fmt_team(team_a)}")
    print(f"\nTeam B:\n{fmt_team(team_b)}")
    print(f"\nTeam A win probability: {prob:.1%}")
    print(f"Team B win probability: {1 - prob:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate or predict with a trained LCK skill model")
    parser.add_argument("--params", required=True, metavar="FILE",
                        help="Saved .pt params file from train.py")
    parser.add_argument("--csv-path", default=DEFAULT_TEST_CSV,
                        help=f"Test CSV for evaluation (default: {DEFAULT_TEST_CSV})")
    parser.add_argument("--ref-csv", default=DEFAULT_REF_CSV,
                        help=f"Full dataset CSV for consistent player ID→index mapping (default: {DEFAULT_REF_CSV})")
    parser.add_argument("--team-a", nargs=5, metavar="NAME:ROLE",
                        help="Predict mode: 5 players for team A as Name:ROLE")
    parser.add_argument("--team-b", nargs=5, metavar="NAME:ROLE",
                        help="Predict mode: 5 players for team B as Name:ROLE")
    args = parser.parse_args()

    if bool(args.team_a) != bool(args.team_b):
        parser.error("--team-a and --team-b must be provided together")

    if args.team_a:
        predict(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

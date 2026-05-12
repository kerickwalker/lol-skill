"""
Split the model-ready dataset into train/test sets by randomly sampling game_block_ids.

Each game_block_id has 10 rows (5 players per team) and is kept intact — rows are
never split across the two sets.

Examples:
    python "data scripts/split_train_test.py"
    python "data scripts/split_train_test.py" --test-ratio 0.2 --seed 42
    python "data scripts/split_train_test.py" --csv-path data/lck_s15_games_MODEL-READY_role_zscored.csv
"""

import argparse
import random
from pathlib import Path

import pandas as pd

CSV_PATH = "data/lck_s15_games_MODEL-READY.csv"
DEFAULT_TEST_RATIO = 0.2
DEFAULT_SEED = 42


def main():
    parser = argparse.ArgumentParser(description="Train/test split for LCK model data")
    parser.add_argument("--csv-path", default=CSV_PATH, help=f"Input CSV (default: {CSV_PATH})")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help=f"Fraction of games held out for test (default: {DEFAULT_TEST_RATIO})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()

    all_game_ids = sorted(df["game_block_id"].unique())
    n_total = len(all_game_ids)
    n_test = max(1, round(n_total * args.test_ratio))
    n_train = n_total - n_test

    rng = random.Random(args.seed)
    shuffled = all_game_ids.copy()
    rng.shuffle(shuffled)
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])

    train_df = df[df["game_block_id"].isin(train_ids)].copy()
    test_df = df[df["game_block_id"].isin(test_ids)].copy()

    # Cold-start check: players in test but never in train
    train_players = set(train_df["player_id"].unique())
    test_players = set(test_df["player_id"].unique())
    cold_start = test_players - train_players
    if cold_start:
        cold_names = (
            test_df[test_df["player_id"].isin(cold_start)][["player_id", "player_name"]]
            .drop_duplicates()
            .set_index("player_id")["player_name"]
            .to_dict()
        )
        print(f"WARNING: {len(cold_start)} player(s) appear only in the test set (no learned skill params):")
        for pid, name in sorted(cold_names.items(), key=lambda x: x[1]):
            print(f"  {name} (id={pid})")

    out_dir = Path(args.csv_path).parent
    stem = Path(args.csv_path).stem
    train_path = out_dir / f"{stem}_train.csv"
    test_path = out_dir / f"{stem}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplit summary (seed={args.seed}):")
    print(f"  Total games : {n_total}")
    print(f"  Train games : {n_train}  ({n_train / n_total:.0%})  → {train_path}")
    print(f"  Test  games : {n_test}  ({n_test / n_total:.0%})  → {test_path}")
    print(f"  Train players : {len(train_players)}")
    print(f"  Test  players : {len(test_players)}")
    date_col = "Date" if "Date" in df.columns else None
    if date_col:
        print(f"  Train date range : {train_df[date_col].min()}  to  {train_df[date_col].max()}")
        print(f"  Test  date range : {test_df[date_col].min()}  to  {test_df[date_col].max()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inspect one trained-game case study with inferred per-game performances.

This script is meant for post-game sanity checks. It prints the observed player
stats beside the guide's inferred per-game performance values (`pa_loc`/`pb_loc`)
for one `game_block_id`.

Example:
    python case_study.py --params params/baseline.pt --game-block-id 123
"""

import argparse
import importlib
from pathlib import Path

import pandas as pd
import pyro
import torch

from models.config import ROLE_MAP
from models.baseline import DIFF_STATS, TEAM_CONTEXT_STATS


DEFAULT_CSV = "data/lck_s15_games_MODEL-READY_train.csv"
DEFAULT_MODEL = "baseline"
DEFAULT_PARAMS = "params/baseline.pt"

DISPLAY_STATS = [
    "level",
    "kills",
    "deaths",
    "assists",
    "cs",
    "golds",
    "vision_score",
    "total_damage_to_champion",
]


def _load_params(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    pyro.get_param_store().set_state(state)


def _valid_game_ids(df: pd.DataFrame) -> list[int]:
    game_ids = []
    for game_block_id, group in df.groupby("game_block_id", sort=True):
        winners = group[group["Result"] == "Victory"]
        losers = group[group["Result"] == "Defeat"]
        if len(winners) == 5 and len(losers) == 5:
            game_ids.append(game_block_id)
    return game_ids


def _player_indexed_df(df: pd.DataFrame) -> pd.DataFrame:
    unique_pids = sorted(df["player_id"].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    out = df.copy()
    out["role_idx"] = out["role"].map(ROLE_MAP)
    out["pid_idx"] = out["player_id"].map(pid_to_idx)
    return out


def _first_existing(columns: pd.Index, candidates: list[str]) -> list[str]:
    return [col for col in candidates if col in columns]


def _format_duration(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _rows_for_side(
    group: pd.DataFrame,
    pids: torch.Tensor,
    roles: torch.Tensor,
    performances: torch.Tensor,
) -> pd.DataFrame:
    rows = []
    for pid_idx, role_idx, performance in zip(
        pids.detach().cpu().tolist(),
        roles.detach().cpu().tolist(),
        performances.detach().cpu().tolist(),
    ):
        match = group[(group["pid_idx"] == pid_idx) & (group["role_idx"] == role_idx)]
        if match.empty:
            continue
        row = match.iloc[0].copy()
        row["inferred_performance"] = performance
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _print_table(df: pd.DataFrame, columns: list[str], sort_by: str | None = None):
    table = df.copy()
    if sort_by and sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False)
    for col in table.columns:
        if pd.api.types.is_numeric_dtype(table[col]):
            table[col] = table[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    print(table[columns].to_string(index=False))


def inspect_game(args):
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df = _player_indexed_df(df)

    valid_game_ids = _valid_game_ids(df)
    if args.game_block_id not in valid_game_ids:
        preview = ", ".join(str(x) for x in valid_game_ids[:10])
        raise ValueError(
            f"game_block_id {args.game_block_id} is not a complete game in {csv_path}. "
            f"First valid ids: {preview}"
        )
    match_idx = valid_game_ids.index(args.game_block_id)

    module = importlib.import_module(f"models.{args.model}")
    batch, _, _, _ = module.load_data(str(csv_path))

    pyro.clear_param_store()
    _load_params(args.params)
    params = pyro.get_param_store()
    required = ["pa_loc", "pb_loc", "pa_scale", "pb_scale"]
    missing = [name for name in required if name not in params]
    if missing:
        raise ValueError(f"Params file does not contain required values: {missing}")
    if match_idx >= params["pa_loc"].shape[0]:
        raise ValueError(
            "The params file has fewer matches than this CSV. "
            "Use the same CSV that was used for training."
        )

    group = df[df["game_block_id"] == args.game_block_id].copy()
    winners = group[group["Result"] == "Victory"]
    losers = group[group["Result"] == "Defeat"]
    meta = winners.iloc[0] if not winners.empty else group.iloc[0]

    team_a = _rows_for_side(
        group,
        batch["team_a_pid"][match_idx],
        batch["team_a_role"][match_idx],
        params["pa_loc"][match_idx],
    )
    team_b = _rows_for_side(
        group,
        batch["team_b_pid"][match_idx],
        batch["team_b_role"][match_idx],
        params["pb_loc"][match_idx],
    )
    team_a["performance_uncertainty"] = params["pa_scale"][match_idx].detach().cpu().numpy()
    team_b["performance_uncertainty"] = params["pb_scale"][match_idx].detach().cpu().numpy()
    table = pd.concat([team_a, team_b], ignore_index=True)

    context_cols = _first_existing(group.columns, ["Duration", *TEAM_CONTEXT_STATS])
    stat_cols = _first_existing(table.columns, DISPLAY_STATS)
    diff_cols = _first_existing(table.columns, DIFF_STATS)
    table_cols = [
        "Result",
        "role",
        "player_name",
        "Champion",
        "inferred_performance",
        "performance_uncertainty",
        *stat_cols,
        *diff_cols,
    ]

    print()
    print("=" * 100)
    print(f"CASE STUDY: game_block_id={args.game_block_id}")
    print("=" * 100)
    print(f"Model : {args.model}")
    print(f"Params: {args.params}")
    print(f"CSV   : {csv_path}")
    print(
        f"Match : {meta.get('Tournament', '')} | {meta.get('Date', '')} | "
        f"Game {meta.get('Game', '')} | Duration {_format_duration(meta.get('Duration', ''))}"
    )
    print()

    print("Team/game context")
    context = group.groupby("Result", sort=False)[context_cols].first().reset_index()
    _print_table(context, ["Result", *context_cols])
    print()

    print("Players sorted by inferred per-game performance")
    _print_table(table, table_cols, sort_by="inferred_performance")
    print()

    print("Interpretation prompts")
    print("- Compare inferred_performance against raw stats: who looks over/under credited?")
    print("- Compare kills/golds/damage to team totals: were raw numbers inflated by team output?")
    print("- Compare *_diff_vs_role_opp: did the model reward same-role outperformance?")
    print("- Compare role-wise z-scores mentally: is each stat being rewarded fairly by role?")


def main():
    parser = argparse.ArgumentParser(description="Inspect a trained-game model case study")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model module to inspect (default: {DEFAULT_MODEL})")
    parser.add_argument("--params", default=DEFAULT_PARAMS, help=f"Saved params file (default: {DEFAULT_PARAMS})")
    parser.add_argument("--csv-path", default=DEFAULT_CSV, help=f"Training CSV path (default: {DEFAULT_CSV})")
    parser.add_argument("--game-block-id", type=int, required=True, help="game_block_id to inspect")
    args = parser.parse_args()
    inspect_game(args)


if __name__ == "__main__":
    main()

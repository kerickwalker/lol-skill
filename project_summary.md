# Project Summary

## Purpose

This project collects and prepares LCK 2025 (S15) professional League of Legends match data from `gol.gg` for skill modeling.

The current workflow is centered around one canonical match-level CSV:

- `data/lck_s15_games_blocked.csv`

and one derived player-level CSV:

- `data/lck_s15_player_aggregated.csv`

## Current Data Files

### `data/lck_s15_games_blocked.csv`

This is the main dataset for modeling and analysis.

- One row per player-game
- One complete game block contains 10 rows
- `KDA` is stored in an Excel-safe form like `="6/3/4"` so spreadsheet software does not convert it into dates
- Includes the base matchlist stats and additional fields scraped from each game's `page-fullstats`

Core columns:

- `game_block_id`
- `game_id`
- `Date`
- `Tournament`
- `Game`
- `Duration`
- `player_id`
- `player_name`
- `role`
- `Champion`
- `Result`
- `KDA`
- `CSM`
- `DPM`
- `KP%`

Additional full-stats fields include examples such as:

- `level`
- `kills`, `deaths`, `assists`
- `cs`, `golds`, `gpm`
- `vision_score`, `wards_placed`, `wards_destroyed`, `control_wards_purchased`
- `solo_kills`
- `gd_at_15`, `csd_at_15`, `xpd_at_15`, `lvld_at_15`
- `damage_dealt_to_turrets`, `damage_dealt_to_buildings`
- `total_damage_taken`, `total_time_spent_dead`

### `data/lck_s15_player_aggregated.csv`

This file is derived from `data/lck_s15_games_blocked.csv`.

- One row per player and role
- Includes game count, wins/losses, win rate, and average core stats
- Uses only the games present in `data/lck_s15_games_blocked.csv`

### `data/analysis/`

This folder contains exploratory analysis outputs generated from `data/lck_s15_games_blocked.csv`.

Key files:

- `analysis_summary.md`
- `lck_s15_model_features.csv`
- `lck_s15_player_summary.csv`
- `lck_s15_teammate_pairs.csv`
- `plots/`

## Current Scripts

### `data scripts/build_lck_s15_data.py`

This is the main data collection and preprocessing script.

It:

- scrapes player and match data directly from `gol.gg`
- collects these LCK S15 tournaments:
  `LCK Cup 2025`, `LCK 2025 Rounds 1-2`, `LCK 2025 Rounds 3-5`, `LCK 2025 Road to MSI`, `LCK 2025 Season Play-In`, `LCK 2025 Season Playoffs`
- keeps only complete 10-player games
- extracts each row's `game_id` from the matchlist page
- fetches the `page-fullstats` table for each unique game and joins those player-level fields back into the blocked CSV
- writes `data/lck_s15_games_blocked.csv`
- writes `data/lck_s15_player_aggregated.csv`

Important behavior:

- roles are filled automatically
- the script first takes `role` from the tournament player list page when available
- if that is missing, it backfills `role` from the game's `page-fullstats` table
- the blocked CSV is made Excel-safe at write time
- the player aggregate is built from the same scraped match rows used to build the blocked CSV
- HTTP requests use retry/backoff to reduce rebuild failures from temporary `gol.gg` timeouts

### `data scripts/analyze_lck_s15_data.py`

This script performs exploratory analysis on `data/lck_s15_games_blocked.csv`.

It:

- parses `KDA` into numeric kills/deaths/assists
- derives context and matchup features
- writes analysis tables into `data/analysis/`
- writes analysis plots into `data/analysis/plots/`

This script is for analysis only. It does not define the final modeling approach.

## Expected Workflow

From the project root:

```bash
conda run -n modelbased-ml python "data scripts/build_lck_s15_data.py"
conda run -n modelbased-ml python "data scripts/analyze_lck_s15_data.py"
```

Suggested usage:

1. Rebuild the canonical data files with `build_lck_s15_data.py`
2. Inspect or update the analysis with `analyze_lck_s15_data.py`
3. Use `data/lck_s15_games_blocked.csv` as the base input for modeling

## Notes

- `data/lck_s15_games_blocked.csv` is the source of truth inside this project
- `data/lck_s15_player_aggregated.csv` should be treated as a convenience summary, not an independent source
- `data/LCK_S15_games/` may contain scraped per-game files, but the main project workflow should rely on the canonical CSVs above
- The current analysis includes engineered features for exploration; not all of them should automatically be used in the final skill model

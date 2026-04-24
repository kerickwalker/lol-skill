# LCK S15 Data Pipeline - Work Summary

## Goal

Create a clean LCK S15 dataset from gol.gg, organized by game blocks (not grouped by player name), and keep a simple final file for modeling.

## What Was Implemented

- Built a scraper for gol.gg player/tournament data:
  - `scrape_golgg_lck_s15.py`
- Built cleaning/normalization script:
  - `clean_golgg_data.py`
- Built one-row-per-player aggregate export script:
  - `build_lck_s15_player_excel.py`
- Built game-block export script (final structure requested):
  - `build_lck_s15_game_blocks_excel.py`

## Key Decisions Applied

- Scraping is done from HTML tables directly (`requests` + `pandas.read_html`) rather than clipboard/Selenium.
- Matchlists are fetched with ALL filters and then filtered to **LCK S15**.
- Duplicate player collection across tournaments was fixed by using a **unique player list** before fetching last-200 matchlists.
- Role inference was attempted and then **reverted** by request (only confirmed roles should be used).

## Final Files To Use

- **Primary final dataset**:
  - `data/lck_s15_CLEAN.xlsx`
  - Contains S15 LCK game stat lines in game blocks.
  - Columns:
    - `game_block_id`, `Game`, `Duration`, `player_id`, `player_name`, `Champion`, `Result`, `KDA`, `CSM`, `DPM`, `KP%`
- Additional game-block export:
  - `data/lck_s15_games_blocked.xlsx`
- Player-level aggregate (one row per player):
  - `data/lck_s15_player_aggregated.xlsx`

## Archived Intermediate Data

All raw/processed intermediate outputs were moved to:

- `data/legacy/snapshot_2026-04-15/`

This includes previous CSV outputs and cleaned variants used during iteration.

## Reverted / Cleaned Up

- Removed unconfirmed `role` column from:
  - `data/legacy/snapshot_2026-04-15/processed/lck_s15_unique_players.csv`
  - `data/lck_s15_CLEAN.xlsx`
- Deleted temporary role script:
  - `attach_player_roles.py`

## Environment Notes

- Conda environment used:
  - `modelbased-ml`
- Packages installed during setup:
  - `pandas`, `requests`, `beautifulsoup4`, `lxml`, `html5lib`, `openpyxl`

## Re-run Commands

From project root:

```bash
conda run -n modelbased-ml python scrape_golgg_lck_s15.py
conda run -n modelbased-ml python clean_golgg_data.py
conda run -n modelbased-ml python build_lck_s15_player_excel.py
conda run -n modelbased-ml python build_lck_s15_game_blocks_excel.py
```

## Current Status

- Data collection completed.
- Final S15 LCK block-structured Excel prepared.
- Intermediate files archived.
- No inferred/uncertain role labels remain in final files.

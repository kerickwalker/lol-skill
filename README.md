# lol-skill

A TrueSkill-like Bayesian model for rating individual skill of professional League of Legends players, fit using Pyro (SVI). Models infer per-player, per-role skill distributions from LCK Season 15 match data while trying to separate individual performance from game/team context.

---

## Setup

**With uv (recommended):**
```bash
git clone <repo-url>
cd lol-skill
uv sync
```

**Without uv:**
```bash
git clone <repo-url>
cd lol-skill
pip install .
```

---

## Data preparation

Split the model-ready dataset into train and test sets (80/20 random split by game):

```bash
# with uv
uv run python "data scripts/split_train_test.py"

# without uv
python "data scripts/split_train_test.py"
```

Options:
```
--test-ratio 0.2    fraction of games held out for test (default: 0.2)
--seed 42           random seed (default: 42)
--csv-path ...      input CSV (default: data/lck_s15_games_MODEL-READY.csv)
```

Outputs:
- `data/lck_s15_games_MODEL-READY_train.csv`
- `data/lck_s15_games_MODEL-READY_test.csv`

The model-ready CSV stays raw. Role-wise z-scores for individual and same-role opponent diff stats are calculated inside the model scripts, not stored as a separate training CSV.

---

## Training

Train a model on the train split and save the learned parameters:

```bash
# with uv
uv run python train.py --model baseline       --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output baseline_5000_seed1
uv run python train.py --model role_alpha_tau --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_alpha_tau_5000_seed1
uv run python train.py --model role_corr      --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_corr_5000_seed1

# without uv
python train.py --model baseline       --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output baseline_5000_seed1
python train.py --model role_alpha_tau --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_alpha_tau_5000_seed1
python train.py --model role_corr      --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_corr_5000_seed1
```

Saves to:
- `params/<output>.pt`
- `elbo/<output>.png`
- `data/analysis/<output>_player_scores.csv`

After training, the script prints a full player score table with each player's inferred mean skill (`mu`), uncertainty (`sigma`), uncertainty-adjusted skill, and a 0-100 rating for easier ranking.

Options:
```
--model     baseline | game_rules | role_alpha_tau | role_corr (default: baseline)
--n-steps   number of SVI training steps (default: 1500); short form: -n
--lr        learning rate (default: 0.01)
--output    name for params/elbo/score files (default: {model}_{timestamp})
--csv-path  path to input CSV (default: data/lck_s15_games_MODEL-READY_train.csv)
--load      load existing params and print scores, skipping training
--seed      random seed for reproducible SVI runs
```

### Available models

| Model | Description |
|-------|-------------|
| `baseline` | Current clean baseline. Uses role-wise z-scored individual stats and same-role opponent diff stats, plus separately standardized duration and team context. Skills are centered around 0 with unit-scale interpretation. |
| `game_rules` | Experimental extension of `baseline` using only the "directly increases" feature relationship edges. Kept for exploration, but not the current preferred next model because z-scored relationship weights are harder to interpret and may double-count evidence. |
| `role_alpha_tau` | Preferred baseline extension. Builds on `baseline` with role-specific alpha/tau priors so, for example, kills/golds can matter differently for ADC, MID, JUNGLE, TOP, and SUPPORT. |
| `role_corr` | Builds on `role_alpha_tau` with learned cross-role performance correlations within a team, using the role-pair matrix discussed in `project_summary.md` as an informative prior. |

---

## Evaluation and prediction

### Evaluate on held-out test games

```bash
# with uv
uv run python test.py --params params/baseline_5000_seed1.pt
uv run python test.py --params params/role_alpha_tau_5000_seed1.pt
uv run python test.py --params params/role_corr_5000_seed1.pt

# without uv
python test.py --params params/baseline_5000_seed1.pt
python test.py --params params/role_alpha_tau_5000_seed1.pt
python test.py --params params/role_corr_5000_seed1.pt
```

Reports accuracy, Brier score, and log-loss on the held-out test games. By default, team skill is the mean of the five player-role skills.

### Predict a specific matchup

```bash
# with uv
uv run python test.py --params params/role_corr_5000_seed1.pt \
    --team-a "Faker:MID" "Gumayusi:ADC" "Keria:SUPPORT" "Zeus:TOP" "Oner:JUNGLE" \
    --team-b "Chovy:MID" "Ruler:ADC" "Delight:SUPPORT" "Doran:TOP" "Canyon:JUNGLE"

# without uv
python test.py --params params/role_corr_5000_seed1.pt \
    --team-a "Faker:MID" "Gumayusi:ADC" "Keria:SUPPORT" "Zeus:TOP" "Oner:JUNGLE" \
    --team-b "Chovy:MID" "Ruler:ADC" "Delight:SUPPORT" "Doran:TOP" "Canyon:JUNGLE"
```

Roles accepted: `TOP`, `JUNGLE` (or `JNG`/`JG`), `MID`, `ADC` (or `BOT`), `SUPPORT` (or `SUP`/`SUPP`).
If a player name is not recognised, the error message lists all known players.

Options:
```
--params            path to saved .pt params file (required)
--csv-path          test CSV for evaluation (default: data/lck_s15_games_MODEL-READY_test.csv)
--ref-csv           training CSV for player ID to index mapping (default: data/lck_s15_games_MODEL-READY_train.csv)
--team-aggregation  sum | mean for combining five players into team skill (default: mean)
--team-a            5 players for team A as Name:ROLE (predict mode)
--team-b            5 players for team B as Name:ROLE (predict mode)
```

---

## Case studies

Inspect one training game to compare observed raw stats, team/game context, same-role opponent diff stats, and the inferred per-game performance values:

```bash
# with uv
uv run python case_study.py --model role_corr --params params/role_corr_5000_seed1.pt --game-block-id 123

# without uv
python case_study.py --model role_corr --params params/role_corr_5000_seed1.pt --game-block-id 123
```

Use this for sanity checks such as: did the model over-credit a player because the game had unusually high team kills, did same-role outperformance matter, or do the inferred performances agree with League intuition?

---

## Typical end-to-end workflow

```bash
# 1. Split the data
uv run python "data scripts/split_train_test.py"

# 2. Train the current preferred model family
uv run python train.py --model baseline       --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output baseline_5000_seed1
uv run python train.py --model role_alpha_tau --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_alpha_tau_5000_seed1
uv run python train.py --model role_corr      --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 5000 --seed 1 --output role_corr_5000_seed1

# 3. Evaluate on the test split
uv run python test.py --params params/baseline_5000_seed1.pt
uv run python test.py --params params/role_alpha_tau_5000_seed1.pt
uv run python test.py --params params/role_corr_5000_seed1.pt

# 4. Inspect one game manually
uv run python case_study.py --model role_corr --params params/role_corr_5000_seed1.pt --game-block-id 123
```

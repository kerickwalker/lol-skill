# lol-skill

A TrueSkill-like Bayesian model for rating individual skill of professional League of Legends players, fit using Pyro (SVI). Models infer per-player, per-role skill distributions from LCK Season 15 match data.

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

---

## Training

Train a model on the train split and save the learned parameters:

```bash
# with uv
uv run python train.py --model baseline           --csv-path data/lck_s15_games_MODEL-READY_train.csv --output baseline
uv run python train.py --model baseline_team_diff --csv-path data/lck_s15_games_MODEL-READY_train.csv --output baseline_team_diff
uv run python train.py --model corr               --csv-path data/lck_s15_games_MODEL-READY_train.csv --output corr
uv run python train.py --model relationship       --csv-path data/lck_s15_games_MODEL-READY_train.csv --output relationship

# without uv
python train.py --model baseline           --csv-path data/lck_s15_games_MODEL-READY_train.csv --output baseline
python train.py --model baseline_team_diff --csv-path data/lck_s15_games_MODEL-READY_train.csv --output baseline_team_diff
python train.py --model corr               --csv-path data/lck_s15_games_MODEL-READY_train.csv --output corr
python train.py --model relationship       --csv-path data/lck_s15_games_MODEL-READY_train.csv --output relationship
```

Saves to `params/<output>.pt` and `elbo/<output>.png`. After training, prints per-role and overall skill rankings with each player's inferred mean (`mu`), uncertainty (`sigma`), and conservative rating (`mu - 3*sigma`).

Options:
```
--model     baseline | baseline_team_diff | corr | relationship (default: baseline)
--n-steps   number of SVI training steps (default: 1500)
--lr        learning rate (default: 0.01)
--output    name for output files (default: {model}_{timestamp})
--csv-path  path to input CSV
--load      load existing params and print rankings, skipping training
```

### Available models

| Model | Description |
|-------|-------------|
| `baseline` | Independent per-role skill; individual stat observations with duration context |
| `baseline_team_diff` | Adds team-level output context and same-role opponent diff stats |
| `corr` | Team performances sampled jointly from a correlated MVN (bot lane, jungle/mid, etc.) |
| `relationship` | Causal stat structure: gold depends on CS/kills/assists, time dead depends on deaths, team aggregate constraints |

---

## Evaluation and prediction

### Evaluate on held-out test games

```bash
# with uv
uv run python test.py --params params/baseline.pt
uv run python test.py --params params/baseline_team_diff.pt
uv run python test.py --params params/corr.pt
uv run python test.py --params params/relationship.pt

# without uv
python test.py --params params/baseline.pt
python test.py --params params/baseline_team_diff.pt
python test.py --params params/corr.pt
python test.py --params params/relationship.pt
```

Reports accuracy, Brier score, and log-loss on the 111 held-out test games.

### Predict a specific matchup

```bash
# with uv
uv run python test.py --params params/baseline.pt \
    --team-a "Faker:MID" "Gumayusi:ADC" "Keria:SUPPORT" "Zeus:TOP" "Oner:JUNGLE" \
    --team-b "Chovy:MID" "Ruler:ADC" "Delight:SUPPORT" "Doran:TOP" "Canyon:JUNGLE"

# without uv
python test.py --params params/baseline.pt \
    --team-a "Faker:MID" "Gumayusi:ADC" "Keria:SUPPORT" "Zeus:TOP" "Oner:JUNGLE" \
    --team-b "Chovy:MID" "Ruler:ADC" "Delight:SUPPORT" "Doran:TOP" "Canyon:JUNGLE"
```

Roles accepted: `TOP`, `JUNGLE` (or `JNG`), `MID`, `ADC` (or `BOT`), `SUPPORT` (or `SUP`).
If a player name is not recognised, the error message lists all known players.

Options:
```
--params      path to saved .pt params file (required)
--csv-path    test CSV for evaluation (default: data/lck_s15_games_MODEL-READY_test.csv)
--ref-csv     full dataset CSV for player ID→index mapping (default: data/lck_s15_games_MODEL-READY.csv)
--team-a      5 players for team A as Name:ROLE (predict mode)
--team-b      5 players for team B as Name:ROLE (predict mode)
```

---

## Typical end-to-end workflow

```bash
# 1. Split the data
uv run python "data scripts/split_train_test.py"

# 2. Train a model
uv run python train.py --model relationship --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 3000 --output relationship

# 3. Evaluate on the test split
uv run python test.py --params params/relationship.pt
```

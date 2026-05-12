# Project Summary

## Purpose

This project collects and prepares LCK 2025 (S15) professional League of Legends match data from `gol.gg` for skill modeling.

The current workflow is centered around one canonical match-level CSV:

- `data/lck_s15_games.csv`

and one derived player-level CSV:

- `data/player_stats.csv`

There is also a model-facing export for feature selection experiments:

- `data/lck_s15_games_MODEL-READY.csv`

And a separate comparison export for role normalization experiments:

- `data/role_normalization_comparison.csv`

And a separate standardized modeling export:

- `data/lck_s15_games_MODEL-READY_role_zscored.csv`

## Current Data Files

### `data/lck_s15_games.csv`

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
- `gd_at_15`, `csd_at_15`, `xpd_at_15`
- `damage_dealt_to_turrets`, `damage_dealt_to_buildings`
- `total_damage_taken`, `total_time_spent_dead`

### `data/player_stats.csv`

This file is derived from `data/lck_s15_games.csv`.

- One row per player and role
- Includes game count, wins/losses, win rate, and average core stats
- Uses only the games present in `data/lck_s15_games.csv`

### `data/lck_s15_games_MODEL-READY.csv`

This file is a filtered modeling export derived from `data/lck_s15_games.csv`.

- One row per player-game
- Keeps selected raw player stats plus a small set of explicit context variables
- Designed to support feature-selection discussions before the final model specification is fixed
- `Duration` is kept as an explicit input rather than relying on per-minute summaries

### `data/role_normalization_comparison.csv`

This file is separate from the training-oriented model-ready export.

- One row per player-game
- Keeps the raw selected comparison stats together with `*_minus_role_mean` and `*_role_z`
- Exists only to compare role-centering versus role-z-scoring before deciding what should enter the actual model

### `data/lck_s15_games_MODEL-READY_role_zscored.csv`

This file is a standardized counterpart to the raw model-ready export.

- One row per player-game
- Keeps the same overall structure as `data/lck_s15_games_MODEL-READY.csv`
- Player-centric stats are z-scored within role
- `Duration` and team-total context columns are z-scored globally
- `Result` is encoded as `1` for `Victory` and `0` for `Defeat`

### `data/analysis/`

This folder contains exploratory analysis outputs generated from `data/lck_s15_games.csv`.

Key files:

- `analysis.md`
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
- writes `data/lck_s15_games.csv`
- writes `data/player_stats.csv`

Important behavior:

- roles are filled automatically
- the script first takes `role` from the tournament player list page when available
- if that is missing, it backfills `role` from the game's `page-fullstats` table
- the blocked CSV is made Excel-safe at write time
- the player aggregate is built from the same scraped match rows used to build the blocked CSV
- HTTP requests use retry/backoff to reduce rebuild failures from temporary `gol.gg` timeouts

### `data scripts/analyze_lck_s15_data.py`

This script now acts as the shared analysis implementation module.

It contains the common loading, plotting, feature-engineering, and summary-writing logic used by the two runnable analysis entry scripts below.

### `data scripts/analyze_lck_s15_comprehension.py`

This script performs exploratory analysis for comprehension.

It:

- parses `KDA` into numeric kills/deaths/assists
- writes analysis tables into `data/analysis/`
- writes analysis plots into `data/analysis/plots/`
- writes tables and plots used by the unified `data/analysis/analysis.md`

### `data scripts/select_lck_s15_features.py`

This script performs feature selection / model-ready export work.

It:

- derives the current selected model-ready feature table
- writes `data/lck_s15_games_MODEL-READY.csv`
- writes `data/role_normalization_comparison.csv`
- writes `data/lck_s15_games_MODEL-READY_role_zscored.csv`
- writes feature-selection plots into `data/analysis/plots/`
- writes tables and plots used by the unified `data/analysis/analysis.md`
 
The unified markdown summary is kept as:

- `data/analysis/analysis.md`

## Expected Workflow

From the project root:

```bash
conda run -n modelbased-ml python "data scripts/build_lck_s15_data.py"
conda run -n modelbased-ml python "data scripts/analyze_lck_s15_comprehension.py"
conda run -n modelbased-ml python "data scripts/select_lck_s15_features.py"
```

Suggested usage:

1. Rebuild the canonical data files with `build_lck_s15_data.py`
2. Refresh the comprehension analysis with `analyze_lck_s15_comprehension.py`
3. Refresh the model-ready export and feature-selection visuals with `select_lck_s15_features.py`
4. Use `data/lck_s15_games.csv` as the base input for modeling

## Discussion Points

These are active modeling decisions that should be discussed before the final skill model is locked in.

### 1. Raw Inputs vs Engineered Features

There are currently two reasonable directions:

- Feed the model mostly raw observed stats such as `Duration`, `kills`, `deaths`, `assists`, `cs`, `golds`, and damage-related totals, and let the model learn the useful structure itself
- Precompute more informative derived features and train the model on those instead

The first approach is simpler and avoids hard-coding assumptions. The second approach can make the signal easier for a simpler model to learn, but it also risks baking in human choices too early.

### 2. Whether Team Context Should Be Explicit Model Input

Team-level context such as `team_kills`, `team_deaths`, `team_assists`, `team_cs`, `team_golds`, `team_vision_score`, and `team_total_damage_to_champion` is only available to the model if those columns are explicitly included in the training data.

This matters conceptually:

- If the model only sees one player's own row-level stats, then it does not automatically know the team's total output in that game
- If team totals are included as columns, then the model can evaluate a player's performance with some built-in game/team context without needing the full set of all 10 player rows at inference time

The current `data/lck_s15_games_MODEL-READY.csv` includes those team-total columns explicitly for that reason.

### 3. Current Fast Model Context Interpretation

The current `model-fast.py` splits non-player context into two parts:

- `Duration` is always used as game-length context
- `team_*` columns are optional team-output context, disabled by default and enabled with `--use-team-stats`

Both are contextual covariates in the observation layer, not individual player stats.

In model-based terms, the generative direction is:

```text
player-role skill -> game performance -> observed player stats
game length ---------------------------> observed player stats
team output ---------------------------> observed player stats
```

This can feel backwards at first, because intuitively we often say that observed stats "feed into" performance. In the generative model, however, latent performance generates observed stats. During inference, the information flows the other way: observed stats update the posterior belief about latent game performance and player skill.

The current observation model can be summarized as:

```text
observed_stat =
    player performance contribution
  + role baseline
  + game-length context contribution
  + optional team-output context contribution
  + noise
```

or more formally:

```text
x_i,g,k ~ Normal(
    alpha_k * p_i,g
  + gamma_role(i),k
  + beta_duration[:,k] * duration_context(g)
  + beta_team[:,k] * team_output_context(team,g)
  , tau_k
)
```

where:

- `p_i,g` is the latent performance of player `i` in game `g`
- `alpha_k` controls how strongly latent performance affects stat `k`
- `gamma_role(i),k` is the expected role-specific baseline for stat `k`
- `duration_context(g)` is standardized game duration
- `team_output_context(team,g)` contains standardized team-level output for that team in that game
- `beta_duration` and `beta_team` are learned, and control how much each context variable shifts each observed stat
- `tau_k` is observation noise for stat `k`

Why `gamma_role` is needed:

- roles have very different stat baselines
- without role baselines, SUPPORT players would be punished for low CS or damage, and ADC/MID players would be rewarded for role-typical farm and damage
- `gamma_role` lets the model compare a player against what is normal for their role

Why the context effects are learned:

- raw counting stats are affected by measurement conditions such as game duration and team tempo
- a 45-minute game naturally creates more opportunities for damage, kills, deaths, healing, shielding, and gold than a 25-minute game
- high-output team games also inflate many individual raw stats
- learned context effects let the model explain those stat shifts without forcing latent player performance to absorb them

Example:

```text
Player A: 28k damage in a 45-minute game
Player B: 24k damage in a 25-minute game
```

Without context, Player A may look better from raw damage alone. With duration context, the model can learn that Player A's damage was partly inflated by game length, so Player B may still have the stronger inferred performance.

This means the current context implementation is mostly a measurement-context correction:

- `Duration` explains game length and exposure time
- `team_*` explains broader team-output environment when enabled
- they do not directly increase or decrease a player's underlying skill
- they help prevent the model from mistaking long/high-action games for better individual performance

The same-role opponent difference columns can also be used as optional observed performance evidence with `--use-diff-stats`.

Those columns are:

- `kills_diff_vs_role_opp`
- `deaths_diff_vs_role_opp`
- `assists_diff_vs_role_opp`
- `cs_diff_vs_role_opp`
- `golds_diff_vs_role_opp`
- `vision_diff_vs_role_opp`
- `damage_diff_vs_role_opp`

When enabled, these are modeled like additional player-level observations of latent game performance. They are not part of the default run yet because they are derived comparison variables and may duplicate information already present in the raw individual stats.

There is an important alternative model structure that may better capture teammate influence:

```text
player skill + teammate/team influence -> latent game performance -> observed stats
```

That would make team or teammate context affect the latent performance variable itself. This has a different interpretation: it says context changes how well the player actually performs in the game, rather than only changing how raw stats should be interpreted.

Likely future direction:

- use `Duration` and broad team tempo variables as observation-level measurement context
- use explicit teammate/player interaction terms as performance-level influence context
- keep those two ideas separate, because they answer different modeling questions

Useful fast-model commands:

```bash
python model-fast.py -n 1000
python model-fast.py -n 1000 --use-team-stats
python model-fast.py -n 1000 --use-diff-stats
```

### 4. Whether And How To Normalize Inputs

There are now three distinct options available in the project:

- use the raw training table in `data/lck_s15_games_MODEL-READY.csv`
- compare role-centering versus role-z-scoring in `data/role_normalization_comparison.csv`
- try a standardized training table in `data/lck_s15_games_MODEL-READY_role_zscored.csv`

Recommendation:

- keep the raw file as the main baseline
- use role-wise normalization for player-centric stats when the problem is mainly role mismatch
- do not role-normalize `Duration` or team-total context columns
- if the model benefits from standardized scales for optimization, standardize `Duration` and team-total columns globally instead

Reasoning:

- role normalization makes sense for stats whose scale and interpretation differ strongly by role
- `Duration` is a game-level quantity, not a role quantity
- `team_*` totals are shared context variables, so role-wise z-scores would be conceptually odd there
- global standardization is a cleaner way to keep those context columns on comparable numeric scales without pretending they are role-dependent

## Notes

- `data/lck_s15_games.csv` is the source of truth inside this project
- `data/player_stats.csv` should be treated as a convenience summary, not an independent source
- `data/lck_s15_games_MODEL-READY.csv` is a working modeling export, not the final locked feature specification
- `data/role_normalization_comparison.csv` is a comparison artifact and should not be mixed into the training table by default
- `data/lck_s15_games_MODEL-READY_role_zscored.csv` is a standardized alternative export, not the default training table
- `data/LCK_S15_games/` may contain scraped per-game files, but the main project workflow should rely on the canonical CSVs above
- The current analysis includes engineered features for exploration; not all of them should automatically be used in the final skill model

## Feature Relationship Notes

These notes are working modeling assumptions and game-mechanic observations for discussion with the team.

### 1. Duration Effects

Almost all raw counting stats that are not explicitly tied to a fixed timestamp or direct opponent difference tend to increase with game duration.

Examples:

- `kills`, `deaths`, `assists`
- `cs`
- `golds`
- `damage_dealt_to_buildings`
- `total_heal`
- `total_heals_on_teammates`
- `damage_self_mitigated`
- `total_damage_shielded_on_teammates`
- `total_time_cc_dealt`
- `total_damage_taken`
- `total_time_spent_dead`

Reasoning:

- Longer games create more opportunities for combat, farming, rotations, sieging, healing, shielding, and deaths
- Because of that, these stats often scale upward with `Duration` even when underlying player strength is unchanged

Counterexamples or weaker duration dependence:

- fixed-time lane stats such as `gd_at_15`, `csd_at_15`, `xpd_at_15`
- event-count stats like `solo_kills`, which can still depend on game flow but are not expected to grow as smoothly with duration

### 2. Gold as a Central Causal Variable

Gold is one of the central game resources and likely has causal links to many other observed variables.

Direct gold generation:

- `cs` generates gold
- `kills` generate gold
- `assists` contribute to gold gain

Likely causal chain:

- higher `cs`, `kills`, and `assists` lead to higher `golds`
- higher `golds` allows stronger item purchases
- stronger items often increase damage, survivability, healing, shielding, and general fight impact
- higher player and team gold therefore likely increases the chance of `Victory`

This same logic should also apply at team level:

- `team_kills`, `team_assists`, and `team_cs` contribute to `team_golds`

### 3. Early-Game Gold and Lane-State Links

The 15-minute lane-state variables are expected to be strongly related.

Likely causal links:

- `csd_at_15` contributes to `gd_at_15`
- kill or assist advantage before 15 minutes can also contribute to `gd_at_15`
- stronger early gold and farm state often also appears together with `xpd_at_15`

So these 15-minute variables are not independent pieces of evidence. They are different views of early-game advantage.

### 4. Same-Role Difference Variables

The same-role opponent difference features are expected to be tightly linked because they partly describe the same matchup advantage.

Likely causal links:

- `kills_diff_vs_role_opp`, `assists_diff_vs_role_opp`, and `cs_diff_vs_role_opp` all tend to contribute to `golds_diff_vs_role_opp`
- higher `golds_diff_vs_role_opp` can then contribute to stronger damage output, so it may also link to `damage_diff_vs_role_opp`

These are closer to causal gameplay relationships than simple statistical overlap.

### 5. Team Totals and Raw Player Stats

Team totals are related to player raw stats, but this is usually a structural relationship rather than a clean causal one.

Examples:

- `kills` is one component of `team_kills`
- `assists` is one component of `team_assists`
- `cs` is one component of `team_cs`
- `golds` is one component of team-level gold generation

This is important because those variables are partly "baked into each other." The relationship is not only that one causes the other, but also that one is mathematically contained inside the other.

### 6. Gold, Items, and Downstream Combat Stats

Gold likely has an indirect causal effect on many later-game outcome stats through itemization.

Likely causal chain:

- more `golds` leads to more or stronger items
- stronger items often increase damage
- stronger items can also improve survivability, healing, shielding, and siege pressure

So correlations from `golds` to stats like:

- `total_damage_to_champion`
- `total_heal`
- `total_heals_on_teammates`
- `damage_self_mitigated`
- `total_damage_shielded_on_teammates`
- `damage_dealt_to_buildings`

may reflect real game causality rather than mere coincidence.

### 7. Role Differences

Roles behave very differently, so raw stat scales are not directly comparable across the full dataset.

Examples:

- ADCs and MIDs often have much higher damage and farm totals than SUPPORT
- SUPPORT typically has much higher vision-related responsibility
- JUNGLE has different kill participation, farm pattern, and early-game responsibilities than lane roles

Because of this, it may be important to compare players to the baseline of their own role rather than to the full league-wide distribution.

Open normalization question for the model:

- subtract the role mean from each stat
- or convert to role-wise z-scores by subtracting the role mean and dividing by the role standard deviation

Current comparison set in `data/role_normalization_comparison.csv`:

- `kills`
- `cs`
- `golds`
- `vision_score`
- `total_heals_on_teammates`
- `damage_self_mitigated`
- `total_damage_to_champion`

For each of these, the analysis currently includes:

- `*_minus_role_mean`
- `*_role_z`

This is meant as a side-by-side comparison tool, not yet a final commitment to one normalization scheme.

## Relationship Types and Interpretation

When constructing feature relationships, it is important to distinguish between different kinds of dependencies rather than treating all edges as direct causal effects.

**1. Precondition (Enabling Condition)**  
In some cases, one variable must occur before another can occur, but does not guarantee it.

- Example: `kills → shutdown_bounty_collected`  
  A kill is required to collect a shutdown bounty, but not every kill results in a shutdown.

This type of relationship should not be interpreted as a causal effect in the sense of “increasing one increases the other.”  
Instead, it represents a structural or logical dependency in the game mechanics.

**2. Scale / Exposure Effect**  
Some variables tend to increase together because one creates more opportunities for the other.

- Example: `Duration → kills`  
  Longer games allow more time for kills to occur, but duration does not directly cause kills in a mechanistic sense.

This type of relationship reflects **positive association due to exposure**, not direct causality.

Furthermore...

Raw counting stats often increase in longer games because there is more time for events to occur. This applies naturally to team-level totals such as `team_kills`, `team_deaths`, `team_assists`, `team_cs`, `team_golds`, and total team damage. It can also apply to player-level stats, but the relationship is weaker because the additional events are distributed across five players and depend heavily on role, champion, game state, and team strategy.

Because of this, it is safer to interpret duration relationships as **exposure effects** rather than direct causal effects. A possible graph edge would be `Duration -> team_kills` or `Duration -> team_deaths` labeled as `scales_with`, while player-level edges such as `Duration -> kills` should be treated more cautiously.

**3. Player raw stats and same-role difference stats**

A same-role difference stat is partly derived from the player stat and the opposing same-role player's stat. For example, `kills_diff` is related to the player's `kills`, but it is not simply the same variable because it also depends on the opponent's kills.

This means raw stats and same-role difference stats should not be treated as independent evidence. However, the relationship is not a clean direct causal mechanism. It is better described as a **derived comparison** or **relative-performance relationship**.

A possible edge type for this would be `relative_to_opponent`, for example `kills -> kills_diff`, but this should be visually distinct from `directly increases` because the value of `kills_diff` also depends on the opponent.

**4. Fifteen-minute checkpoint stats and full-game difference stats**

The `*_at_15` variables are early-game checkpoints, while the role-opponent difference variables describe a broader same-role performance difference. For example, `csd_at_15` is an early checkpoint related to lane or role advantage, while `cs_diff` describes the final or full-game CS difference against the same-role opponent.

In League of Legends, early advantages can snowball into later advantages through gold, experience, map control, and item timing. Therefore, it is reasonable to discuss a soft relationship from `*_at_15` features to later `*_diff` features.

However, this is not deterministic. A player can have an early advantage and lose it later, or recover from an early deficit. These edges should therefore be labeled as **snowball tendency** or `snowballs_into`, not as direct causal effects.

**5. Conservative graph**

For the conservative graph, keep only strong direct mechanisms, preconditions, and component relationships. Duration effects, raw-to-difference relationships, and early-to-late snowball relationships are useful discussion points, but they should be added later using softer edge types such as:

- `scales_with` for duration and exposure effects
- `relative_to_opponent` for raw stat to same-role difference relationships
- `snowballs_into` for early-game checkpoint advantages influencing later advantages

This keeps the graph useful without overstating causal certainty.

---

These distinctions are important to avoid over-interpreting relationships in the model.  
Only a subset of edges (such as gold generation from kills or CS) should be treated as strong causal mechanisms, while others reflect constraints or scaling effects.

# LCK S15 Data Comprehension Summary

- Rows: 5550
- Game blocks: 555
- Players: 68
- Roles: ADC, JUNGLE, MID, SUPPORT, TOP

## Scope
- All engineered columns come from `data/lck_s15_games.csv` only.
- No external gold, vision, economy, draft, or timeline data is used in this analysis.
- Each derived feature is either a direct transformation of one CSV column or an aggregation within the same `game_block_id`.

## Raw Columns Used
- `game_block_id`, `Game`, `Duration`, `player_id`, `player_name`, `role`, `Champion`, `Result`, `KDA`, `CSM`, `DPM`, `KP%`

## Derived Feature Definitions
- `kills`, `deaths`, `assists`: split directly from `KDA = kills/deaths/assists`.
- `kp_pct`: numeric version of `KP%` after removing the percent sign.
- `duration_min`: match duration converted from `MM:SS` to decimal minutes.
- `win`: `1` for `Victory`, `0` for `Defeat`.
- `team_id`: `game_block_id` combined with `Result`, so each game has one winning team row-group and one losing team row-group.
- `kills_per_min`, `deaths_per_min`, `assists_per_min`: per-minute rates from the parsed KDA counts and `duration_min`.
- `kda_index`: `log(1 + kills + assists) - log(1 + deaths)`.
- `dpm_per_csm`: `DPM / CSM`. This is only a damage-per-farm proxy; it is not gold-based and should not be interpreted as gold efficiency.
- `*_role_z`: z-score of a metric within the same role only.
- `performance_score`: hand-crafted exploratory score, not a learned skill estimate:
  `0.35*dpm_role_z + 0.20*csm_role_z + 0.20*kp_role_z + 0.20*kda_role_z - 0.15*deaths_role_z + 0.10*(2*win - 1)`
- `team_kills`, `team_deaths`, `team_assists`, `team_dpm`, `team_csm`: sums across the five players with the same `team_id`.
- `kill_share`, `death_share`, `assist_share`, `dpm_share`, `csm_share`: player share of the corresponding team total.
- `teammate_mean_dpm`, `teammate_mean_csm`, `teammate_mean_perf`: average of the other four players on the same team in the same game.
- `opponent_*`: stats of the same-role opponent in the same `game_block_id` on the opposite result side.
- `*_diff_vs_role_opp`: player stat minus same-role opponent stat in that match.

## Modeling Notes
- `kda_index` uses a log-smoothed form so 0-death games are rewarded without producing infinite values.
- Role normalization is used because raw DPM, CSM, and KP differ structurally by role.
- `performance_score` is best treated as an EDA convenience, not as the target latent skill variable.

## Top Composite Skill Proxies
| player_name | role | games | win_rate | avg_performance | avg_dpm | avg_kp_pct |
| --- | --- | --- | --- | --- | --- | --- |
| Chovy | MID | 121 | 0.744 | 0.52 | 775.223 | 68.02 |
| Oner | JUNGLE | 115 | 0.626 | 0.346 | 495.852 | 73.977 |
| Keria | SUPPORT | 115 | 0.626 | 0.342 | 248.165 | 75.622 |
| Ruler | ADC | 121 | 0.744 | 0.332 | 787.62 | 70.16 |
| Canyon | JUNGLE | 121 | 0.744 | 0.277 | 456.521 | 69.623 |
| Kiin | TOP | 121 | 0.744 | 0.253 | 634.471 | 59.512 |
| Aiming | ADC | 131 | 0.557 | 0.24 | 742.336 | 72.939 |
| Delight | SUPPORT | 121 | 0.628 | 0.233 | 205.529 | 75.872 |
| Gumayusi | ADC | 91 | 0.626 | 0.19 | 769.516 | 69.816 |
| Viper | ADC | 121 | 0.628 | 0.188 | 778.116 | 69.803 |

## Strongest Positive Pair Contexts
| focal_player_name | focal_role | teammate_player_name | teammate_role | games_together | win_rate_together | focal_perf_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Teddy | ADC | Kyeahoo | MID | 34 | 0.441 | 0.301 |
| GIDEON | JUNGLE | Fisher | MID | 40 | 0.55 | 0.261 |
| kingen | TOP | Fisher | MID | 40 | 0.55 | 0.253 |
| Rich | TOP | Kyeahoo | MID | 39 | 0.436 | 0.18 |
| Jiwoo | ADC | Fisher | MID | 40 | 0.55 | 0.155 |
| deokdam | ADC | Peter | SUPPORT | 84 | 0.524 | 0.123 |
| Hype | ADC | HamBak | JUNGLE | 27 | 0.333 | 0.104 |
| Andil | SUPPORT | Kyeahoo | MID | 39 | 0.436 | 0.097 |
| Berserker | ADC | Bulldog | MID | 76 | 0.25 | 0.09 |
| Clozer | MID | HamBak | JUNGLE | 32 | 0.344 | 0.086 |
| Sponge | JUNGLE | Kyeahoo | MID | 39 | 0.436 | 0.07 |
| Pollu | SUPPORT | Croco | JUNGLE | 64 | 0.422 | 0.069 |

## Notes
- `performance_score` is a role-normalized proxy, not a learned latent skill.
- `focal_perf_delta_vs_baseline` compares a player's average performance with a given teammate against that player's overall average.
- Opponent-adjusted columns use the same-role opponent in the same game block as a simple matchup control.

---

# LCK S15 Feature Selection Summary

## Purpose
- This file records the current model-ready feature choices.
- It is meant to stay separate from the general comprehension/EDA notes.

## Current Model-Ready Export
- `data/lck_s15_games_MODEL-READY.csv` is produced by `data scripts/analyze_lck_s15_data.py` from `data/lck_s15_games.csv`.
- One row per player-game.
- `Duration` is kept as an explicit raw input rather than replacing it with per-minute derived stats.
- This file is intentionally kept raw for now.

## Included Raw Player Stats
- `Result`, `Duration`, `level`, `kills`, `deaths`, `assists`, `cs`, `golds`, `vision_score`
- `solo_kills`, `double_kills`, `triple_kills`, `quadra_kills`, `penta_kills`
- `gd_at_15`, `csd_at_15`, `xpd_at_15`
- `objectives_stolen`, `damage_dealt_to_buildings`
- `total_heal`, `total_heals_on_teammates`, `damage_self_mitigated`, `total_damage_shielded_on_teammates`
- `total_time_cc_dealt`, `total_damage_taken`, `total_time_spent_dead`
- `shutdown_bounty_collected`, `shutdown_bounty_lost`, `total_damage_to_champion`

## Excluded Raw / Derived Stats
- `KDA`
- `CSM`, `DPM`, `gpm` and similar per-minute rate features in the model-ready export
- `consumables_purchased`, `items_purchased`
- ward subcomponent fields such as `wards_placed`, `wards_destroyed`, `control_wards_purchased`
- `damage_dealt_to_turrets`, `time_ccing_others`

## Included Team Context
- `team_kills`, `team_deaths`, `team_assists`, `team_cs`, `team_golds`, `team_vision_score`, `team_total_damage_to_champion`
- These are included explicitly so the model can see team-level context without needing the full 10-player game table at inference time.

## Included Same-Role Opponent Differences
- `kills_diff_vs_role_opp`, `deaths_diff_vs_role_opp`, `assists_diff_vs_role_opp`, `cs_diff_vs_role_opp`
- `golds_diff_vs_role_opp`, `vision_diff_vs_role_opp`, `damage_diff_vs_role_opp`

## Role-Normalization Comparison File
- `data/role_normalization_comparison.csv` is a separate comparison file and is not meant to be mixed into the training table by default.
- For `kills`, `cs`, `golds`, `vision_score`, `total_heals_on_teammates`, `damage_self_mitigated`, and `total_damage_to_champion`, it includes:
- `*_minus_role_mean`
- `*_role_z`
- These are included for comparison and discussion, not yet as a final commitment to one normalization strategy.

## Z-Scored Modeling File
- `data/lck_s15_games_MODEL-READY_role_zscored.csv` is a separate standardized version of the model-ready file.
- Player-centric stats are z-scored within role.
- `Duration` is z-scored at the unique-game level.
- Team-total context columns are z-scored at the unique-team level rather than by role.
- `Result` is encoded as `1` for `Victory` and `0` for `Defeat` in that file.

## Analysis Plots Relevant To Selection
- `plots/model_ready_feature_correlation_heatmap.png`: correlation matrix over all columns from `Result` onward in the model-ready CSV, with `Result` encoded as `1/0` and `Duration` converted to decimal minutes.
- `plots/role_normalization_comparison.png`: compares raw, role-centered, and role-z-scored versions of the selected normalization stats.
# LCK S15 Games Blocked Analysis

- Rows: 5160
- Game blocks: 516
- Players: 68
- Roles: ADC, JUNGLE, MID, SUPPORT, TOP

## Scope
- All engineered columns come from `data/lck_s15_games_blocked.csv` only.
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
| Chovy | MID | 116 | 0.75 | 0.511 | 772.181 | 67.755 |
| Ruler | ADC | 116 | 0.75 | 0.341 | 785.147 | 69.624 |
| Keria | SUPPORT | 108 | 0.611 | 0.335 | 247.269 | 75.844 |
| Oner | JUNGLE | 108 | 0.611 | 0.332 | 494.019 | 74.049 |
| Canyon | JUNGLE | 116 | 0.75 | 0.294 | 460.302 | 69.596 |
| Kiin | TOP | 116 | 0.75 | 0.267 | 637.181 | 59.409 |
| Aiming | ADC | 116 | 0.56 | 0.253 | 731.75 | 73.71 |
| Delight | SUPPORT | 113 | 0.655 | 0.245 | 206.018 | 75.2 |
| Viper | ADC | 113 | 0.655 | 0.234 | 782.159 | 69.801 |
| Zeus | TOP | 113 | 0.655 | 0.16 | 627.92 | 58.967 |

## Strongest Positive Pair Contexts
| focal_player_name | focal_role | teammate_player_name | teammate_role | games_together | win_rate_together | focal_perf_delta_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Teddy | ADC | Kyeahoo | MID | 34 | 0.441 | 0.303 |
| kingen | TOP | Fisher | MID | 40 | 0.55 | 0.229 |
| GIDEON | JUNGLE | Fisher | MID | 40 | 0.55 | 0.22 |
| Rich | TOP | Kyeahoo | MID | 39 | 0.436 | 0.18 |
| Jiwoo | ADC | Fisher | MID | 40 | 0.55 | 0.137 |
| deokdam | ADC | Peter | SUPPORT | 74 | 0.5 | 0.129 |
| Andil | SUPPORT | Kyeahoo | MID | 39 | 0.436 | 0.098 |
| Hype | ADC | HamBak | JUNGLE | 27 | 0.333 | 0.094 |
| Berserker | ADC | Bulldog | MID | 76 | 0.25 | 0.09 |
| Pollu | SUPPORT | Croco | JUNGLE | 55 | 0.436 | 0.085 |
| Clozer | MID | HamBak | JUNGLE | 32 | 0.344 | 0.082 |
| Peter | SUPPORT | Casting | TOP | 19 | 0.368 | 0.08 |

## Notes
- `performance_score` is a role-normalized proxy, not a learned latent skill.
- `focal_perf_delta_vs_baseline` compares a player's average performance with a given teammate against that player's overall average.
- Opponent-adjusted columns use the same-role opponent in the same game block as a simple matchup control.
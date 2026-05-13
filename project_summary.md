# Project Summary

## Current Branch Note: `baseline_v2`

The long project memory below is intentionally preserved, including older experiments and decision history. On the `baseline_v2` cleanup branch, the executable model code is being simplified to one canonical baseline model:

- `models/baseline.py`
- `models/config.py`
- `train.py`
- `test.py`
- `case_study.py`

The current baseline uses `data/lck_s15_games_MODEL-READY.csv` / `data/lck_s15_games_MODEL-READY_train.csv` as raw, human-readable input and calculates z-scores inside `models/baseline.py`.

The separate exported file `data/lck_s15_games_MODEL-READY_role_zscored.csv` was removed on `baseline_v2` because it was outdated and is no longer used by the model. The data scripts were updated so they do not recreate that file. Older sections below may still mention the role-zscored export because they record the historical reasoning that led to the current design.

Old executable model variants such as `baseline_team_diff`, old `relationship`, and old `corr` were removed from the cleaned branch because they no longer match the agreed plan. The future plan is still:

- keep `baseline` as the clean full-context z-score baseline;
- build `game_rules` from this baseline using selected feature relationships;
- build future `corr` on top of the baseline/game-rules direction.

The agreed model ladder is:

1. `baseline`: full-context z-score model.
2. `game_rules`: `baseline` plus selected direct gameplay relationships.
3. `role_alpha_tau`: `game_rules` plus role-specific `alpha` and `tau`, so each role can learn which stats matter most and how noisy those stats are.
4. `role_corr`: `role_alpha_tau` plus correlations between role performances within the same team/game.

The old `relationship` and `corr` files are treated as historical references, not as code to preserve. They did not build cleanly on the final full-context baseline and did not match the intended additive sequence above. Starting the future models from the current baseline is cleaner and easier to explain in the report.

### Distribution Decision: Keep Gaussian for Now

We discussed whether Gaussian likelihoods are logical for League of Legends stats, since many raw stats are counts and some are snowbally. Examples:

- kills, deaths, assists, solo kills, and objectives are count-like;
- many count stats are skewed and zero-heavy;
- snowballing means once a player/team is ahead, it may become easier to get additional kills, gold, damage, and objectives.

A more literal statistical model could use Poisson or Negative Binomial likelihoods for counts, perhaps with duration as an exposure term, and LogNormal/Gamma-like likelihoods for positive continuous totals such as gold, damage, healing, or damage mitigated.

For the current model family, we decided to keep Gaussian observations because the baseline does not model raw counts directly. It first converts individual and diff stats into role-wise z-scores, duration into a game-level z-score, and team context into a team-game z-score. The model therefore observes "above or below normal for this role/context" rather than raw count values. On that standardized scale, a Gaussian likelihood is a defensible simplifying approximation and keeps `alpha`, `tau`, and importance diagnostics interpretable.

Snowball effects are real, but modeling them properly would likely require timeline data or a more complex dynamic model. For now, snowballing should be represented conservatively through `game_rules` relationships, not by changing every likelihood family at once. This keeps comparisons between `baseline`, `game_rules`, `role_alpha_tau`, and `role_corr` cleaner.

### `game_rules` Implementation Decision

The new `game_rules` model should only use edges from `feature-relationships/simple/lck_s15_feature_relationship_edges_simplified.csv` where `kind == directly_increases`.

It should not use:

- `component_of` edges;
- `precondition` edges;
- softer duration/exposure or snowball-tendency relationships;
- raw-to-diff structural overlap relationships.

This is intentionally conservative. `directly_increases` edges are modeled as additional positive-weight Gaussian rule constraints on top of the baseline observations. They do not replace the baseline observations. The positive weights encode the domain assumption that the source tends to increase the target, while the learned rule noise controls how strict or loose that rule should be.

Directly-increases edges currently used by `game_rules`:

- `cs -> golds`
- `kills -> golds`
- `assists -> golds`
- `shutdown_bounty_collected -> golds`
- `team_cs -> team_golds`
- `team_kills -> team_golds`
- `team_assists -> team_golds`
- `csd_at_15 -> gd_at_15`
- `cs_diff_vs_role_opp -> golds_diff_vs_role_opp`
- `kills_diff_vs_role_opp -> golds_diff_vs_role_opp`
- `assists_diff_vs_role_opp -> golds_diff_vs_role_opp`
- `deaths -> total_time_spent_dead`

### Baseline vs `game_rules` 1500-Step Check

On `baseline_v2`, both models were trained for 1500 SVI steps with seed `1` on `data/lck_s15_games_MODEL-READY_train.csv` and evaluated on `data/lck_s15_games_MODEL-READY_test.csv`.

Environment note:

- the runnable Python was `C:\Users\pokem\miniconda3\envs\modelbased-ml\python.exe`;
- the Windows OpenMP workaround `KMP_DUPLICATE_LIB_OK=TRUE` was required for this shell.

Held-out results:

| model | train time | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|
| `baseline` | 46.61s | 59.5% | 0.2321 | 0.6563 |
| `game_rules` | 74.85s | 59.5% | 0.2330 | 0.6581 |

Interpretation:

- `game_rules` did not improve held-out win prediction in this first one-seed, 1500-step check;
- calibration was very slightly worse than baseline;
- this does not necessarily mean the rules are useless, because win prediction is only a partial proxy for individual performance quality;
- the next useful checks are case studies and learned rule-weight inspection.

Learned `game_rules` guide-location weights from this run:

- `cs -> golds`: `0.605`
- `kills -> golds`: `0.340`
- `assists -> golds`: `0.318`
- `shutdown_bounty_collected -> golds`: `0.122`
- `csd_at_15 -> gd_at_15`: `0.546`
- `deaths -> total_time_spent_dead`: `0.914`
- `cs_diff_vs_role_opp -> golds_diff_vs_role_opp`: `0.331`
- `kills_diff_vs_role_opp -> golds_diff_vs_role_opp`: `0.528`
- `assists_diff_vs_role_opp -> golds_diff_vs_role_opp`: `0.389`
- `team_cs -> team_golds`: `0.763`
- `team_kills -> team_golds`: `0.370`
- `team_assists -> team_golds`: `0.091`

Effective learned rule tau values from this run:

- approximately `0.445`, `0.833`, `0.396`, `0.393`, `0.216`

The learned values are broadly sensible: CS is the strongest contributor to player/team gold, deaths strongly explain time spent dead, and team assists contribute less to team gold than team CS/team kills. However, because these rule factors are additional likelihoods on already-observed stats, they can also double-count relationships if the rule tau becomes too small.

### Decision After Inspecting `game_rules`: Pause It

After the first `game_rules` run and case-study inspection, we decided not to continue building the next model on top of `game_rules` yet.

Reasons:

- the held-out win metrics did not improve over `baseline`;
- the rules are extra likelihood factors over stats that are already observed by the baseline, so they can double-count evidence;
- the relationships are harder to parameterize after z-scoring than they would be in a raw-count model;
- in a raw model, a domain prior like "one kill is roughly 300 gold" has an interpretable scale;
- in the z-score model, a rule like `kills_z -> golds_z` no longer means "300 gold per kill"; it means "a one-role-standard-deviation increase in kills predicts some fraction of a role-standard-deviation increase in gold";
- choosing weights such as `0.2`, `0.35`, or `0.6` is therefore much more abstract and risks becoming manual tuning rather than clear domain knowledge.

Current decision:

- keep `game_rules` as an experiment/reference;
- do not use it as the parent for the next model;
- build `role_alpha_tau` directly on top of the clean `baseline`;
- revisit `game_rules` later only if we find a principled way to handle rule weights, rule noise, and double-counting.

Updated model ladder:

1. `baseline`: full-context z-score model.
2. `role_alpha_tau`: `baseline` plus role-specific `alpha` and `tau`.
3. `role_corr`: `role_alpha_tau` plus correlations between role performances.
4. Optional future `game_rules`: reintroduced only after a clearer rule-weight strategy is agreed.

### `role_alpha_tau` First Implementation

`role_alpha_tau` builds directly on `baseline`, not on `game_rules`.

The baseline observation model is:

```text
stat_z ~ Normal(alpha_stat * performance + context, tau_stat)
```

The role-specific version is:

```text
stat_z ~ Normal(alpha_role,stat * performance + context, tau_role,stat)
```

For same-role diff stats:

```text
diff_stat_z ~ Normal(alpha_role,diff_stat * performance_difference, tau_role,diff_stat)
```

The implementation is hierarchical:

```text
alpha_role,stat = alpha_global,stat + role_offset_role,stat
role_offset_role,stat ~ Normal(0, 0.35)
```

and:

```text
tau_role,stat = 0.05 + tau_global_raw_stat * role_multiplier_role,stat
role_multiplier_role,stat ~ LogNormal(0, 0.35)
```

This lets roles differ while still sharing statistical strength. It avoids manually setting hard values such as "kills matter twice as much for ADC", because in z-score space that would be hard to justify directly.

First 1500-step check with seed `1`:

| model | train time | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|
| `baseline` | 46.61s | 59.5% | 0.2321 | 0.6563 |
| `role_alpha_tau` | 71.59s | 58.6% | 0.2326 | 0.6573 |

Interpretation:

- `role_alpha_tau` was slightly worse than baseline on held-out win prediction in this first run;
- this does not automatically invalidate it, because the goal is interpretable individual-performance inference, not only win prediction;
- however, the added flexibility may need stronger regularization or more steps/seeds to avoid chasing noise.

Largest player score increases versus baseline:

- `Soboro`: `+13.4` score points
- `Ellim`: `+11.9`
- `Diable`: `+6.0`
- `deokdam`: `+6.0`
- `Clozer`: `+6.0`

Largest player score decreases versus baseline:

- `Pollu`: `-6.0`
- `Berserker`: `-6.0`
- `DuDu`: `-6.0`
- `Life`: `-6.0`
- `Vicla`: `-6.0`
- `Bdd`: `-6.0`
- `PerfecT`: `-6.0`

Interesting learned role-specific differences:

- ADC `kills` became more important than support `kills`:
  - ADC `kills`: `alpha=0.724`, `tau=0.447`, importance `1.620`
  - support `kills`: `alpha=0.533`, `tau=0.828`, importance `0.644`
- Jungle `cs` became more important than support `cs`:
  - jungle `cs`: `alpha=0.534`, `tau=0.448`, importance `1.191`
  - support `cs`: `alpha=0.312`, `tau=1.031`, importance `0.303`
- ADC `kills_diff_vs_role_opp` became more important than support `kills_diff_vs_role_opp`:
  - ADC: `alpha=0.421`, `tau=0.428`, importance `0.984`
  - support: `alpha=0.258`, `tau=0.816`, importance `0.317`
- `golds` remained extremely important across roles because tau stayed near the floor:
  - jungle `golds`: importance `10.746`
  - support `golds`: importance `10.353`
  - ADC `golds`: importance `9.664`
- Support multikill tau values collapsed near the `0.05` floor for rare multikill stats such as triple/quadra/penta kills. This is probably an overconfidence artifact caused by sparse support multikill observations rather than meaningful domain knowledge.

Important caution:

- the tau floor may still be too permissive for rare/zero-heavy stats;
- role-specific tau can create misleading certainty for sparse events;
- before trusting `role_alpha_tau`, inspect whether rare stats such as `triple_kills`, `quadra_kills`, and `penta_kills` should have stronger tau floors, stronger priors, or be grouped/removed.

### Category-Informed `role_alpha_tau`

After inspecting the fully free role-specific run, we decided to add explicit role/stat categories for core stats where we have strong League of Legends domain beliefs.

The motivation:

- role-wise z-scoring makes stats comparable within roles, but does not say which stats should matter most for each role;
- a support with high CS is usually not "performing well" in the same way that an ADC or mid with high CS is;
- kills and low deaths are much more central to ADC/mid carry performance than to support performance;
- jungle CS matters, but usually less directly than lane CS for top/mid/ADC;
- gold is central, but support gold should not be interpreted the same way as carry/solo-lane gold.

The category-informed priors are still learnable. They are not fixed coefficients.

Alpha category values:

| category | alpha prior mean magnitude | alpha prior std |
|---|---:|---:|
| `very_low` | 0.00 | 0.20 |
| `low` | 0.15 | 0.25 |
| `medium` | 0.35 | 0.35 |
| `high` | 0.65 | 0.35 |
| `very_high` | 0.80 | 0.40 |

Tau category values:

| category | tau prior center |
|---|---:|
| `very_small` | 0.35 |
| `small` | 0.60 |
| `medium` | 0.90 |
| `high` | 1.30 |
| `very_high` | 1.70 |

The selected role assumptions:

| stat | top | jungle | mid | adc | support |
|---|---|---|---|---|---|
| `kills` | medium alpha, small tau | medium alpha, small tau | very_high alpha, very_small tau | very_high alpha, very_small tau | very_low alpha, high tau |
| `deaths` | medium negative alpha, small tau | medium negative alpha, small tau | high negative alpha, very_small tau | high negative alpha, very_small tau | very_low negative alpha, high tau |
| `cs` | high alpha, very_small tau | medium alpha, medium tau | high alpha, very_small tau | high alpha, very_small tau | very_low alpha, very_high tau |
| `golds` | very_high alpha, very_small tau | medium alpha, medium tau | very_high alpha, very_small tau | very_high alpha, very_small tau | very_low alpha, very_high tau |

First 1500-step check with seed `1`, compared against the same baseline:

| model | train time | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|
| `baseline` | 46.61s | 59.5% | 0.2321 | 0.6563 |
| free `role_alpha_tau` | 71.59s | 58.6% | 0.2326 | 0.6573 |
| category-informed `role_alpha_tau` | 51.48s | 59.5% | 0.2324 | 0.6570 |

Interpretation:

- the category-informed version recovered baseline accuracy and was closer to baseline calibration than the fully free role-specific version;
- it still did not beat baseline on Brier/log-loss;
- the category priors made the model more aligned with domain expectations, but the data can still pull against them.

Learned core stat values from the category-informed run:

| stat | role | alpha | tau | importance |
|---|---|---:|---:|---:|
| `kills` | top | 0.656 | 0.593 | 1.105 |
| `kills` | jungle | 0.698 | 0.517 | 1.350 |
| `kills` | mid | 0.622 | 0.578 | 1.075 |
| `kills` | adc | 0.712 | 0.439 | 1.622 |
| `kills` | support | 0.497 | 0.832 | 0.598 |
| `deaths` | top | -0.208 | 0.653 | 0.319 |
| `deaths` | jungle | -0.269 | 0.581 | 0.463 |
| `deaths` | mid | -0.241 | 0.660 | 0.365 |
| `deaths` | adc | -0.210 | 0.630 | 0.333 |
| `deaths` | support | -0.130 | 0.643 | 0.203 |
| `cs` | top | 0.429 | 0.390 | 1.101 |
| `cs` | jungle | 0.543 | 0.452 | 1.202 |
| `cs` | mid | 0.377 | 0.382 | 0.987 |
| `cs` | adc | 0.350 | 0.356 | 0.982 |
| `cs` | support | 0.288 | 1.058 | 0.272 |
| `golds` | top | 0.527 | 0.083 | 6.331 |
| `golds` | jungle | 0.569 | 0.069 | 8.215 |
| `golds` | mid | 0.489 | 0.087 | 5.624 |
| `golds` | adc | 0.481 | 0.088 | 5.490 |
| `golds` | support | 0.526 | 0.072 | 7.326 |

Important observations:

- ADC kills became the strongest kill signal, which matches the domain prior.
- Support kills were reduced relative to ADC/jungle/top, but not as much as expected; the data still pulled support kills upward.
- Support CS had much lower importance than lane/jungle CS, which matches the domain prior.
- Deaths were negative for every role, but the role separation was weaker than expected.
- Gold remained extremely important for every role, including support. This conflicts with the prior that support gold should be very low importance and suggests either:
  - gold is acting as a broad proxy for team/game advantage even for support;
  - the prior is too weak;
  - gold may be too dominant and should be modeled differently or decomposed later.
- Support rare multikill tau still collapsed near the `0.05` floor for triple/quadra/penta kills, because these stats were not part of the new core category priors. This remains a warning that rare zero-heavy stats may need special treatment.

Largest player score increases versus baseline in the category-informed run:

- `Soboro`: `+10.4`
- `Peanut`: `+9.0`
- `Ellim`: `+9.0`
- `Bdd`: `+7.5`
- `Rich`: `+6.0`
- `Vital`: `+6.0`

Largest player score decreases versus baseline:

- `Life`: `-10.4`
- `Berserker`: `-7.5`
- `Vicla`: `-7.5`
- `Siwoo`: `-6.0`
- `BeryL`: `-6.0`

### `role_corr` Role-Performance Correlation Plan

The next model, `role_corr`, builds on the category-informed `role_alpha_tau` model and changes only the per-game performance layer.

Instead of independent per-player performances:

```text
p_role ~ Normal(skill_role, performance_beta)
```

the five teammate performances are sampled jointly:

```text
[p_top, p_jng, p_mid, p_adc, p_sup] ~ MVN(skill_means, performance_covariance)
```

The user-provided role influence matrix is:

| source/target | TOP | JNG | MID | ADC | SUP |
|---|---:|---:|---:|---:|---:|
| TOP | x | 0.30 | 0.15 | 0.05 | 0.10 |
| JNG | 0.30 | x | 0.60 | 0.40 | 0.80 |
| MID | 0.15 | 0.60 | x | 0.20 | 0.50 |
| ADC | 0.05 | 0.40 | 0.20 | x | 1.00 |
| SUP | 0.10 | 0.80 | 0.50 | 1.00 | x |

Interpretation:

- values are relative influence/importance strengths between role performances;
- they are not used directly as an unconstrained covariance matrix;
- the off-diagonal values are globally rescaled to produce a valid positive-definite correlation matrix while preserving relative differences.

The raw matrix was not positive definite at full scale. In particular, ADC-SUP at `1.00` implies perfect correlation and creates a near/singular multivariate Normal. The implementation therefore uses global scale `0.8`.

Scaled role correlation matrix used in `models/role_corr.py`:

| role | TOP | JNG | MID | ADC | SUP |
|---|---:|---:|---:|---:|---:|
| TOP | 1.00 | 0.24 | 0.12 | 0.04 | 0.08 |
| JNG | 0.24 | 1.00 | 0.48 | 0.32 | 0.64 |
| MID | 0.12 | 0.48 | 1.00 | 0.16 | 0.40 |
| ADC | 0.04 | 0.32 | 0.16 | 1.00 | 0.80 |
| SUP | 0.08 | 0.64 | 0.40 | 0.80 | 1.00 |

This preserves the intended relative ordering:

- ADC-SUP is strongest;
- JNG-SUP and JNG-MID are very strong;
- JNG-ADC and MID-SUP are medium-high;
- TOP-ADC is weakest.

The current `role_corr` model keeps these correlations fixed. They are not learned yet. Learning them would introduce another major layer of assumptions and identifiability concerns, so fixed correlations are the clearer first version.

### `role_corr` Training Results

Both `role_corr` variants were trained for 1500 steps with seed `1`.

| model | train time | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|
| `baseline` | 46.61s | 59.5% | 0.2321 | 0.6563 |
| category-informed `role_alpha_tau` | 51.48s | 59.5% | 0.2324 | 0.6570 |
| fixed `role_corr` | 56.69s | 57.7% | 0.2335 | 0.6594 |
| learned `role_corr` | 58.35s | 58.6% | 0.2300 | 0.6516 |

Interpretation:

- fixed `role_corr` made held-out performance worse in this run;
- learned `role_corr` improved Brier and log-loss compared with all previous models in this small comparison, even though accuracy was only `58.6%`;
- this suggests the correlation structure may help calibration when it is allowed to adapt, but fixed correlations may be too rigid.

Learned-correlation implementation:

- `role_corr_learned` does not sample every correlation entry independently, because that can create invalid covariance matrices;
- instead, it samples the Cholesky factor of the performance covariance around the fixed target matrix;
- this guarantees a positive-definite covariance matrix;
- prior standard deviations were small:
  - Cholesky off-diagonal prior std: `0.08`
  - Cholesky diagonal log prior std: `0.05`
- the reported learned matrix below is the implied correlation matrix after reconstructing the covariance.

Fixed scaled correlation matrix:

| role | TOP | JNG | MID | ADC | SUP |
|---|---:|---:|---:|---:|---:|
| TOP | 1.000 | 0.240 | 0.120 | 0.040 | 0.080 |
| JNG | 0.240 | 1.000 | 0.480 | 0.320 | 0.640 |
| MID | 0.120 | 0.480 | 1.000 | 0.160 | 0.400 |
| ADC | 0.040 | 0.320 | 0.160 | 1.000 | 0.800 |
| SUP | 0.080 | 0.640 | 0.400 | 0.800 | 1.000 |

Learned implied correlation matrix:

| role | TOP | JNG | MID | ADC | SUP |
|---|---:|---:|---:|---:|---:|
| TOP | 1.000 | 0.274 | 0.247 | 0.230 | 0.399 |
| JNG | 0.274 | 1.000 | 0.385 | 0.406 | 0.584 |
| MID | 0.247 | 0.385 | 1.000 | 0.329 | 0.522 |
| ADC | 0.230 | 0.406 | 0.329 | 1.000 | 0.583 |
| SUP | 0.399 | 0.584 | 0.522 | 0.583 | 1.000 |

Largest changes:

- `ADC-SUP`: `0.800 -> 0.583`, reduced substantially;
- `JNG-SUP`: `0.640 -> 0.584`, reduced slightly;
- `TOP-SUP`: `0.080 -> 0.399`, increased substantially;
- `TOP-ADC`: `0.040 -> 0.230`, increased;
- `MID-ADC`: `0.160 -> 0.329`, increased;
- `JNG-MID`: `0.480 -> 0.385`, reduced.

Caution:

- because the learned model uses a Cholesky prior, the learned pairwise correlations are not independent parameters;
- moving one Cholesky entry can affect multiple final correlations;
- the learned values should be interpreted as an adaptive covariance structure near the prior, not as direct posterior estimates for independently modeled pair strengths.

### 5000-Step Baseline vs Role Models

After deciding to scrap the fixed-correlation version, `models/role_corr.py` now refers to the learned-correlation model. The separate `role_corr_learned` module was removed.

All three main models were trained for 5000 SVI steps with seed `1`:

| model | train time | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|
| `baseline` | 165.02s | 58.6% | 0.2338 | 0.6600 |
| `role_alpha_tau` | 184.07s | 59.5% | 0.2332 | 0.6587 |
| `role_corr` | 236.07s | 59.5% | 0.2310 | 0.6537 |

Interpretation:

- `role_corr` is the best of the three on Brier score and log-loss after 5000 steps;
- `role_alpha_tau` and `role_corr` tie on accuracy;
- `baseline` is slightly worse after 5000 steps than it was after 1500 steps, which suggests longer training is not automatically better for held-out win prediction under this SVI setup;
- `role_corr` remains the most promising model by calibration, but it should still be checked across multiple seeds before making a final claim.

Learned implied correlation matrix after 5000 steps:

| role | TOP | JNG | MID | ADC | SUP |
|---|---:|---:|---:|---:|---:|
| TOP | 1.000 | 0.318 | 0.266 | 0.234 | 0.400 |
| JNG | 0.318 | 1.000 | 0.418 | 0.433 | 0.597 |
| MID | 0.266 | 0.418 | 1.000 | 0.350 | 0.551 |
| ADC | 0.234 | 0.433 | 0.350 | 1.000 | 0.582 |
| SUP | 0.400 | 0.597 | 0.551 | 0.582 | 1.000 |

Pairwise comparison against the prior matrix:

| pair | prior | learned | delta |
|---|---:|---:|---:|
| JNG-SUP | 0.640 | 0.597 | -0.043 |
| ADC-SUP | 0.800 | 0.582 | -0.218 |
| MID-SUP | 0.400 | 0.551 | +0.151 |
| JNG-ADC | 0.320 | 0.433 | +0.113 |
| JNG-MID | 0.480 | 0.418 | -0.062 |
| TOP-SUP | 0.080 | 0.400 | +0.320 |
| MID-ADC | 0.160 | 0.350 | +0.190 |
| TOP-JNG | 0.240 | 0.318 | +0.078 |
| TOP-MID | 0.120 | 0.266 | +0.146 |
| TOP-ADC | 0.040 | 0.234 | +0.194 |

The learned matrix continues the same pattern as the 1500-step run:

- ADC-SUP is pulled down substantially from the prior;
- TOP-SUP, TOP-ADC, TOP-MID, and MID-ADC are pulled upward;
- JNG-SUP remains high but slightly below the prior;
- the learned structure suggests broader teamwide role coupling than the original hand-specified matrix.

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

### 0. Modeling Philosophy And Evaluation Target

The main goal of this project is not only to predict which team wins a game. The deeper goal is to infer a player's latent "true skill" from noisy, context-dependent observations of how they performed in professional games.

Current modeling philosophy:

- each player has a latent player-role skill
- in each game, each player has a latent game performance drawn around that skill
- team performance is the sum of the five player performances
- win/loss is generated from the difference between the two team performances
- observed player stats are noisy evidence of each player's latent game performance

This means the match result is useful because winning is the clearest objective outcome. However, the result is a team-level outcome. A win can happen because one player carried, because all five players were slightly better, because the opponent made mistakes, or because a teammate performed exceptionally well. Therefore, result alone is not enough to identify individual performance cleanly.

The desired inference question is:

```text
Given the observed game stats, what individual player performances best explain both
the player stats and the final team result?
```

This is different from the easier evaluation question:

```text
Given the learned player skills only, can we predict the winner of a held-out game?
```

The project currently uses held-out win prediction as a proxy evaluation because `Result` is the only objective target label. There is no ground-truth label saying that a specific player had exactly some true performance value in a game. This makes win prediction defensible, but incomplete.

Important limitation of the current `test.py`:

- it loads the learned `s_loc` player skill values
- it predicts held-out games from the sum of the five player skills on each team
- it does not use the held-out game's observed player stats
- it does not use `Duration`, `team_*`, or `*_diff_vs_role_opp` at evaluation time

So the current test is mostly a pre-game skill-based winner prediction proxy. It does not directly answer whether the model inferred more reasonable post-game individual performances.

This explains why adding team/diff context may not strongly change accuracy. Those context variables can improve how player performances are inferred during training, but the current evaluation only sees the final learned skill estimates. If the final skill estimates do not change enough to flip held-out match predictions, accuracy may stay the same even if the internal performance attribution became more sensible.

The ideal future evaluation should include qualitative and quantitative checks for post-game performance inference:

- context sensitivity: the same raw stat should imply different performance depending on game/team context
- example: 10 kills in a 100-kill game should be less exceptional than 10 kills in a 12-kill game
- role fairness: supports should not be punished for low CS or low damage when those are normal for support
- opponent comparison: outperforming the same-role opponent should increase inferred performance more clearly than raw totals alone
- ablations: compare `baseline` versus `baseline_team_diff` and inspect whether inferred per-game performances become more sensible
- case studies: manually inspect selected games with domain knowledge and check whether inferred performances match expert intuition

This should be documented carefully in reports:

- held-out win prediction is a proxy metric
- the model's philosophical target is post-game individual performance attribution
- team and diff stats are included to interpret observed stats in context, not because they are individual skill by themselves

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

### 3. Current Model Context Interpretation

The current main workflow uses `train.py`, `test.py`, and model modules in `models/`.

Current relevant models:

- `models/baseline.py`
- `models/baseline_team_diff.py`
- `models/corr.py`
- `models/relationship.py`

`baseline` is now intended to be the full context baseline: individual player stats, duration context, team-output context, and same-role opponent difference observations. The old `baseline_team_diff` name referred to this idea before we decided that team/diff context should always be part of the baseline.

Current model-context summary:

- `baseline`: individual stats plus `Duration`; full selected `team_*` context; selected `*_diff_vs_role_opp` observations
- `baseline_team_diff`: legacy/older name for the full-context baseline idea; should no longer be treated as the conceptual default
- `corr`: individual stats plus full selected `team_*` context; no `Duration`; no `*_diff_vs_role_opp`; adds fixed role-correlation structure between teammates' per-game performances
- `relationship`: structured causal/stat relationship model; includes selected team aggregate constraints (`team_kills`, `team_golds`, `team_cs`) and selected diff stats (`golds_diff_vs_role_opp`, `damage_diff_vs_role_opp`); no `Duration`; not the full team-context set

Context variables are used in the observation layer, not as direct individual player stats.

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
  + team-output context contribution
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

Important example:

```text
Player A has 10 kills in a game with 100 total kills.
Player B has 10 kills in a game with 12 total kills.
```

Without team context, both players can look similar from `kills` alone. With team context such as `team_kills`, the model can learn that Player A's 10 kills occurred in a high-kill environment, while Player B captured a much larger share of a low-kill game. This is exactly the kind of distinction the project wants the model to be able to make.

Example:

```text
Player A: 28k damage in a 45-minute game
Player B: 24k damage in a 25-minute game
```

Without context, Player A may look better from raw damage alone. With duration context, the model can learn that Player A's damage was partly inflated by game length, so Player B may still have the stronger inferred performance.

This means the current context implementation is mostly a measurement-context correction:

- `Duration` explains game length and exposure time
- `team_*` explains broader team-output environment
- they do not directly increase or decrease a player's underlying skill
- they help prevent the model from mistaking long/high-action games for better individual performance

Important team-context distinction:

- during training, the model sees all ten player rows from a game
- therefore team totals such as `team_kills` are mathematically redundant with the five player rows for that team
- however, they are not automatically redundant inside the model structure

Reason:

```text
player_i performance -> player_i observed stats
```

does not by itself create:

```text
sum(teammate stats) -> interpretation of player_i observed stats
```

The other four teammates' stats are present elsewhere in the likelihood, but unless the model explicitly computes and uses their sum, they do not directly contextualize one player's raw stat observation. Including `team_*` columns is therefore a simple way to expose that context:

```text
team_kills -> expected individual kills
team_golds -> expected individual golds
team_total_damage_to_champion -> expected individual damage
```

Alternative future design:

- compute team totals internally from the five player rows instead of reading precomputed `team_*` columns
- this would remove data redundancy, but the model would still need explicit edges saying that team totals affect expected individual stat observations

Current recommendation:

- keep `team_*` as observation-level context for now
- do not treat `team_*` as extra individual skill evidence
- avoid adding separate `component_of` likelihood constraints unless there is a clear reason, because this may double-count the same observed information

The same-role opponent difference columns are used in the full-context baseline as additional observed performance evidence.

Those columns are:

- `kills_diff_vs_role_opp`
- `deaths_diff_vs_role_opp`
- `assists_diff_vs_role_opp`
- `cs_diff_vs_role_opp`
- `golds_diff_vs_role_opp`
- `vision_diff_vs_role_opp`
- `damage_diff_vs_role_opp`

These are modeled like additional player-level observations of latent game performance. They are derived comparison variables and may duplicate information already present in the raw individual stats, but they also provide a very useful relative signal: how the player did compared with the same-role opponent.

Important diff-stat interpretation:

- same-role diff stats should ideally be modeled as evidence about performance difference, not just one player's absolute performance
- for example, `kills_diff_vs_role_opp` is not only about the player's kills; it is the player's kills minus the same-role opponent's kills

Preferred structure:

```text
diff_stat_for_A_role ~ alpha * (p_A_role - p_B_same_role) + noise
diff_stat_for_B_role ~ alpha * (p_B_role - p_A_same_role) + noise
```

This is more aligned with the modeling philosophy than:

```text
diff_stat_for_A_role ~ alpha * p_A_role + noise
```

because same-role diff stats are explicitly relative-performance evidence.

Why not compare every player to every teammate and opponent:

- same-role opponent comparison is clean because TOP is compared to TOP, MID to MID, etc.
- teammate comparisons are harder because roles have different jobs, so ADC-vs-SUPPORT or JUNGLE-vs-MID raw stat comparisons can be misleading
- all-opponent comparisons are possible but noisier and more dependent on role, champion, team composition, and game state
- share-style features such as kill share, gold share, damage share, or vision share may be useful later, but they are derived features and should be added deliberately

Current recommendation:

- use same-role opponent diff stats as the main relative-performance evidence
- keep team totals as context
- avoid full pairwise teammate/opponent comparisons until the simpler model is understood

### 3.1 Current Alpha, Gamma, And Tau Assumptions

The observation model connects latent per-game performance to observed stats using:

```text
observed_stat ~ Normal(
    alpha_stat * latent_performance
  + gamma_role,stat
  + duration/team context effects
  , tau_stat
)
```

There are now multiple baseline variants, so the exact treatment depends on the model:

- `baseline_fixed_raw`: `alpha`, `gamma`, and `tau` are fixed from `models/config.py`
- `baseline_free_raw`: `gamma` is fixed, while `alpha` and `tau` are learned around the old config values as priors
- `baseline_zscore`: `gamma` is removed, individual/diff stats are role-wise z-scored, and `alpha`/`tau` are learned

This section documents the raw config assumptions because they still matter for `baseline_fixed_raw` and as prior centers for `baseline_free_raw`.

Important interpretation:

- fixed alpha values are strongest assumptions
- learned alpha values are weaker assumptions, but the prior center still matters
- learned tau values let the model adjust how noisy/reliable each stat is
- z-scored stats make alpha/tau comparisons easier because the observed inputs are on a common role-relative scale

Important special case:

- `total_damage_taken` is ambiguous by domain knowledge
- taking damage can indicate useful frontline/tanking/participation in some contexts, but poor positioning or being caught in others
- the raw config still has a positive alpha prior for `total_damage_taken`
- this should be reviewed in future role-specific models

Older experimental implementation detail:

```text
alpha_stat = sign_stat * prior_alpha_magnitude_stat * exp(alpha_log_multiplier_stat)
```

This was used in the earlier experimental `models/baseline.py` learned-alpha version. The current comparison models are now separated into `baseline_fixed_raw`, `baseline_free_raw`, and `baseline_zscore`.

In that older experiment, sign-free stats used:

```text
alpha_stat = prior_alpha_magnitude_stat * unconstrained_alpha_multiplier_stat
```

where `unconstrained_alpha_multiplier_stat` can become positive, negative, or remain close to zero.

Why the separated comparison models are better:

- they let us compare fixed assumptions, raw learned assumptions, and z-scored learned assumptions explicitly
- they avoid hiding several design choices inside one changing `baseline` file
- they make it easier to explain to teammates exactly what changed between runs

Current raw config alpha signs/prior centers and tau values:

| stat | alpha sign | prior alpha magnitude | fixed tau |
|---|---:|---:|---:|
| `level` | positive | 0.02 | 1.5 |
| `kills` | positive | 0.05 | 2.5 |
| `deaths` | negative | 0.05 | 2 |
| `assists` | positive | 0.05 | 4 |
| `cs` | positive | 1 | 50 |
| `golds` | positive | 30 | 2500 |
| `vision_score` | positive | 0.3 | 15 |
| `solo_kills` | positive | 0.1 | 0.5 |
| `double_kills` | positive | 0.005 | 0.5 |
| `triple_kills` | positive | 0.002 | 0.25 |
| `quadra_kills` | positive | 0.001 | 0.15 |
| `penta_kills` | positive | 0.0005 | 0.05 |
| `gd_at_15` | positive | 10 | 700 |
| `csd_at_15` | positive | 0.2 | 15 |
| `xpd_at_15` | positive | 10 | 650 |
| `objectives_stolen` | positive | 0.005 | 0.3 |
| `damage_dealt_to_buildings` | positive | 30 | 3000 |
| `total_heal` | positive | 20 | 5000 |
| `total_heals_on_teammates` | positive | 5 | 2000 |
| `damage_self_mitigated` | positive | 100 | 12000 |
| `total_damage_shielded_on_teammates` | positive | 5 | 1200 |
| `total_time_cc_dealt` | positive | 2 | 300 |
| `total_damage_taken` | positive config prior, domain-ambiguous | 50 | 8000 |
| `total_time_spent_dead` | negative | 1 | 65 |
| `shutdown_bounty_collected` | positive | 2 | 250 |
| `shutdown_bounty_lost` | negative | 2 | 200 |
| `total_damage_to_champion` | positive | 100 | 7000 |

Potential sign assumptions to review carefully:

- `damage_self_mitigated` is currently positive, which mostly rewards tanking/survivability and may be role/champion-dependent
- `total_heal` is currently positive, but jungle/self-healing can be champion-kit dependent rather than pure skill
- `total_time_cc_dealt` is currently positive, but champion kit matters heavily
- multi-kill stats are positive but very sparse, so their small tau values may make them surprisingly strong unless checked

Diff stat handling in the fixed raw config:

- diff stats inherit the sign of their base stat in `baseline_fixed_raw`
- `deaths_diff_vs_role_opp` therefore has a negative alpha sign
- `shutdown_bounty_lost` is not currently one of the selected diff stats
- diff stat tau is currently `sqrt(2) * base_stat_tau`, because a difference of two noisy observed quantities is expected to be noisier than either raw stat alone
- in `baseline_free_raw`, diff alpha and diff tau are learned around these raw config values
- in `baseline_zscore`, diff stats are role-wise z-scored and use low/medium/high tau category priors

Current selected diff stat signs and tau values:

| diff stat | inherited sign | fixed diff tau |
|---|---:|---:|
| `kills_diff_vs_role_opp` | positive | 3.54 |
| `deaths_diff_vs_role_opp` | negative | 2.83 |
| `assists_diff_vs_role_opp` | positive | 5.66 |
| `cs_diff_vs_role_opp` | positive | 70.71 |
| `golds_diff_vs_role_opp` | positive | 3535.53 |
| `vision_diff_vs_role_opp` | positive | 21.21 |
| `damage_diff_vs_role_opp` | positive | 9899.49 |

Open decision:

- whether `gamma` role baselines should also become learned later
- whether `tau` should become learned later
- whether some signs should be changed, removed, or made role-specific
- whether some noisy/champion-dependent stats should have larger tau or be excluded

Important role-specific-importance distinction:

- `gamma` only says what value is normal for a role
- `gamma` does not say how important that stat is for performance in that role

Example:

```text
expected_cs = gamma_role,cs + alpha_cs * performance + context
```

In this structure:

- SUPPORT has a low CS baseline through `gamma`
- ADC has a high CS baseline through `gamma`
- but `alpha_cs` is still shared across roles

So the model knows that SUPPORT normally farms less than ADC, but it does not know that extra CS may be much more important evidence for ADC than for SUPPORT.

To model role-specific stat importance, we need a future model with:

```text
alpha_role,stat
```

instead of:

```text
alpha_stat
```

This would allow assumptions such as:

- `alpha[ADC, cs] > alpha[SUPPORT, cs]`
- `alpha[SUPPORT, vision_score] > alpha[ADC, vision_score]`
- `alpha[TOP/MID/JUNGLE/ADC, solo_kills] > alpha[SUPPORT, solo_kills]`
- `alpha[JUNGLE, objectives_stolen] > alpha[other_roles, objectives_stolen]`

This is considered necessary for the long-term model, but should be implemented as a separate model after the `game_rules` model is finished. It should not be quietly folded into the current baseline.

Planned model hierarchy:

1. `baseline`
   Full-context baseline using raw individual stats, `Duration`, `team_*` context, and same-role diff stats.

2. `game_rules`
   Builds on `baseline`, keeps all full-context baseline inputs, and adds explicit game-rule relationships selected from the feature-relationship work. Precondition edges are skipped for now. Component-of team constraints are treated cautiously to avoid double-counting.

3. future role-specific-importance model
   Builds after `game_rules`, replacing shared `alpha_stat` with role-specific `alpha_role,stat` so each role can learn which stats matter most for its performance.

4. future `corr` model
   Builds after the above, adding correlated teammate performance structure on top of the richer baseline/game-rules setup.

Current baseline comparison plan:

The project now has three full-context baseline variants for comparing the modeling assumptions cleanly before moving on to `game_rules`.

All three variants include:

- individual selected player stats
- `Duration` as game-length context
- all selected `team_*` context stats
- selected same-role opponent `*_diff_vs_role_opp` stats
- team result generated from the sum of the five latent player performances

The three variants differ only in how observed stats are scaled and how much freedom `alpha`, `gamma`, and `tau` have.

#### `baseline_fixed_raw`

File:

- `models/baseline_fixed_raw.py`

This is the most conservative comparison point.

```text
raw observed stat
    ~ Normal(gamma_role,stat + alpha_stat * performance + context, tau_stat)
```

Properties:

- uses raw stat values from `data/lck_s15_games_MODEL-READY.csv`
- keeps the hand-set `alpha` values from `models/config.py` fixed
- keeps the hand-set `tau` values from `models/config.py` fixed
- keeps `gamma_role,stat` as a fixed role baseline/intercept
- models diff stats as evidence about the same-role performance gap, not only absolute performance

Diff stat form:

```text
diff_stat_A_role ~ Normal(alpha_diff_stat * (p_A_role - p_B_role), tau_diff_stat)
diff_stat_B_role ~ Normal(alpha_diff_stat * (p_B_role - p_A_role), tau_diff_stat)
```

#### `baseline_free_raw`

File:

- `models/baseline_free_raw.py`

This keeps raw stats but lets the model learn how important/noisy each stat is.

```text
raw observed stat
    ~ Normal(gamma_role,stat + learned_alpha_stat * performance + context, learned_tau_stat)
```

Properties:

- uses raw stat values from `data/lck_s15_games_MODEL-READY.csv`
- keeps `gamma_role,stat` as the role baseline/intercept
- learns `alpha_stat` around the old config value as a prior center
- learns `tau_stat` around the old config value as a prior center
- learns separate `alpha` and `tau` for same-role diff stats

Important caveat:

- because raw stats have different units, learned raw `alpha` and raw `tau` are still not directly comparable across stats
- a useful rough importance signal is `abs(alpha_stat) / tau_stat`, but even that should be interpreted carefully unless the inputs are standardized

#### `baseline_zscore`

File:

- `models/baseline_zscore.py`

This uses standardized observed stats and removes `gamma`.

```text
role-wise z-scored observed stat
    ~ Normal(alpha_stat * (performance - 25) + standardized_context, learned_tau_stat)
```

Properties:

- z-scores individual player stats within role
- z-scores same-role opponent diff stats within role
- standardizes `Duration` globally by game
- standardizes `team_*` context globally by team-game
- removes `gamma_role,stat`, because role-wise z-scoring already subtracts the role mean
- learns `alpha_stat` freely around zero
- learns `tau_stat` from low/medium/high category priors

Why `gamma` is removed here:

- in the raw models, `gamma_role,stat` says what value is normal for a role
- in the z-score model, role-wise z-scoring has already centered each role/stat around zero
- keeping both role z-scoring and `gamma` would duplicate the same role-baseline idea

Tau category priors in `baseline_zscore`:

| category | prior tau | interpretation |
|---|---:|---|
| low | 0.5 | relatively reliable evidence after standardization |
| medium | 1.0 | normal/noisy evidence |
| high | 2.0 | sparse, champion-dependent, or context-dependent evidence |

Current category choices:

- low tau: `level`, `cs`, `golds`, `gd_at_15`, `csd_at_15`, `xpd_at_15`, `cs_diff_vs_role_opp`, `golds_diff_vs_role_opp`
- medium tau: `kills`, `deaths`, `assists`, `vision_score`, `damage_dealt_to_buildings`, `total_time_spent_dead`, `total_damage_to_champion`, most combat/vision diff stats
- high tau: `solo_kills`, multi-kill stats, `objectives_stolen`, healing/shielding/mitigation stats, `total_damage_taken`, shutdown bounty stats

Why this is less rigid than the original model:

- the old model fixed every `tau_stat` numerically by hand
- the z-score model only gives broad prior categories
- `tau_stat` is still learned during training

Important limitation:

- `baseline_zscore` makes `alpha` and `tau` more comparable than the raw models, but it also changes the model's interpretation: each stat now means "above/below role average" rather than its original raw count
- `Duration` and `team_*` are not role-wise z-scored because they are not role-specific player stats

```text
raw model idea:
  "How many kills/gold/damage did the player have?"

z-score model idea:
  "How far above or below normal for this role was the player?"
```

### 3.2 Current Baseline Comparison Results

All three comparison models were trained for 1500 SVI steps on:

- `data/lck_s15_games_MODEL-READY_train.csv`

They were evaluated with the current held-out skill-only proxy test:

- `data/lck_s15_games_MODEL-READY_test.csv`
- 111 games
- `test.py` uses learned long-term skills only
- `test.py` does not use held-out observed stats, `Duration`, `team_*`, or diff stats at prediction time

Results:

| model | steps | train time | accuracy | Brier | log-loss | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `baseline_fixed_raw` | 1500 | 40.41s | 59.5% | 0.2480 | 0.7565 | best calibration of the three; close to random Brier/log-loss but slightly above random accuracy |
| `baseline_free_raw` | 1500 | 54.95s | 62.2% | 0.3602 | 3.2314 | higher accuracy but very overconfident/miscalibrated |
| `baseline_zscore` | 1500 | 57.35s | 62.2% | 0.3420 | 2.3524 | same accuracy as free raw, less overconfident than free raw, still poorly calibrated |
| `baseline_zscore_standardized` initial | 1500 | 66.33s | 62.2% | 0.2380 | 0.6689 | `beta=0.5`, `alpha_prior_std=0.5`, no tau floor |
| `baseline_zscore_standardized` current best sweep | 1500 | not recorded in table | 59.5% | 0.2325 | 0.6572 | `beta=1.0`, `alpha_prior_std=1.0`, `tau=0.05+raw_tau`; best calibration in the sweep |

Interpretation:

- freeing `alpha` and `tau` appears to increase held-out winner hit rate from 59.5% to 62.2%
- however, both free models produce much worse Brier/log-loss, meaning the probabilities are overconfident
- the z-score model is better calibrated than free raw, but still worse than fixed raw on Brier/log-loss
- because this test ignores held-out observed stats, these results do not directly tell us which model gives better post-game performance attribution

Practical conclusion:

- keep all three models for now
- use `baseline_fixed_raw` as the conservative default
- use `baseline_free_raw` to test whether learning raw alpha/tau helps
- use `baseline_zscore` as the main candidate for interpretable/comparable learned stat importance
- prefer `baseline_zscore_standardized` for new z-score modeling work because its latent skill scale is centered at `0` and easier to interpret
- do not judge the project only by this held-out win prediction proxy; case studies remain necessary

Reason for preferring `baseline_zscore` going forward:

- raw learned `alpha` and `tau` are hard to compare across stats because raw stats have different units
- role-wise z-scores make player-centric stats mean "above/below normal for this role"
- this makes learned importance much easier to discuss with domain knowledge
- it removes the need for fixed `gamma` role baselines because role centering is already handled by the z-score transform
- it better matches the project philosophy that roles should be compared to their own role expectations, not to a generic player average
- the calibration is still not good enough, so this is a modeling direction rather than a final validated model

Additional reason for preferring `baseline_zscore_standardized`:

- the old `baseline_zscore` still used the arbitrary TrueSkill-like scale centered at `25`
- `baseline_zscore_standardized` uses `0` as average latent skill/performance and `1` as the prior skill standard deviation
- this makes learned skill values easier to read directly: positive means above average, negative means below average
- the model averages five player performances into team performance instead of summing them, keeping team performance on the same standardized scale
- this change materially improved calibration in the current proxy test
- a small additive tau floor improved the interpretability of very small learned tau values without destroying the gold signal

Useful commands:

```bash
python train.py --model baseline_fixed_raw --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 1500 --output baseline_fixed_raw_1500
python train.py --model baseline_free_raw --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 1500 --output baseline_free_raw_1500
python train.py --model baseline_zscore --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 1500 --output baseline_zscore_1500
python train.py --model baseline_zscore_standardized --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 1500 --output baseline_zscore_standardized_1500
python test.py --params params/baseline_fixed_raw_1500.pt
python test.py --params params/baseline_free_raw_1500.pt
python test.py --params params/baseline_zscore_1500.pt
python test.py --params params/baseline_zscore_standardized_1500.pt --team-aggregation mean
```

The standardized model can be swept with environment variables:

```bash
LCK_PERFORMANCE_BETA=1.0 LCK_ALPHA_PRIOR_STD=1.0 LCK_TAU_FLOOR=0.05 python train.py --model baseline_zscore_standardized --csv-path data/lck_s15_games_MODEL-READY_train.csv --n-steps 1500 --output std_beta1p0_alpha1p0_tau0p05
python test.py --params params/std_beta1p0_alpha1p0_tau0p05.pt --team-aggregation mean
```

### 3.2.1 Alpha, Tau, And Importance Comparison

The current comparison output files are:

- `data/analysis/model_alpha_tau_importance_comparison.csv`
- `data/analysis/model_alpha_tau_importance_individual.csv`
- `data/analysis/model_alpha_tau_importance_diff.csv`

The comparison uses:

```text
importance = abs(alpha) / tau
```

Interpretation:

- `alpha` is the learned/fixed slope from latent performance to an observed stat
- `tau` is the learned/fixed observation noise for that stat
- a larger `abs(alpha) / tau` means a stat is stronger evidence of latent performance
- for raw models this is only a rough comparison because stats are in different units
- for the z-score model this is much more interpretable because player-centric stats are standardized

For learned models, the reported values use the variational guide locations:

- `alpha = alpha_loc`
- `tau = exp(tau_log_loc)`

Top individual stat importance after 1500 steps:

| model | strongest individual signals by `abs(alpha)/tau` |
|---|---|
| `baseline_fixed_raw` | `solo_kills`, `deaths`, `kills`, `vision_score`, `cs`, `objectives_stolen`, `xpd_at_15`, `total_time_spent_dead` |
| `baseline_free_raw` | `solo_kills`, `deaths`, `cs`, `objectives_stolen`, `golds`, `vision_score`, `total_time_spent_dead`, `level` |
| `baseline_zscore` | `golds`, `kills`, `cs`, `total_damage_to_champion`, `damage_dealt_to_buildings`, `shutdown_bounty_collected`, `level`, `gd_at_15` |
| `baseline_zscore_standardized` | `golds`, `kills`, `cs`, `total_damage_to_champion`, `shutdown_bounty_collected`, `damage_dealt_to_buildings`, `level`, `gd_at_15` |

Top diff stat importance after 1500 steps:

| model | strongest same-role diff signals by `abs(alpha)/tau` |
|---|---|
| `baseline_fixed_raw` | `deaths_diff_vs_role_opp`, `kills_diff_vs_role_opp`, `cs_diff_vs_role_opp`, `vision_diff_vs_role_opp` |
| `baseline_free_raw` | `deaths_diff_vs_role_opp`, `assists_diff_vs_role_opp`, `kills_diff_vs_role_opp`, `cs_diff_vs_role_opp` |
| `baseline_zscore` | `golds_diff_vs_role_opp`, `kills_diff_vs_role_opp`, `deaths_diff_vs_role_opp`, `assists_diff_vs_role_opp` |
| `baseline_zscore_standardized` | `golds_diff_vs_role_opp`, `kills_diff_vs_role_opp`, `deaths_diff_vs_role_opp`, `assists_diff_vs_role_opp` |

Important observations:

- `baseline_zscore` makes `golds` and `golds_diff_vs_role_opp` the strongest signals, which matches the domain idea that gold is central to League of Legends
- `baseline_zscore_standardized` keeps the same qualitative importance ordering, but the absolute `alpha/tau` values are much larger because the latent performance scale changed from around `25` to around `0`
- the standardized model learned very small tau for `golds` and `golds_diff_vs_role_opp`; this strongly reinforces that gold is central, but it should be monitored because tiny tau can make the model overconfident
- `baseline_free_raw` learned a negative `solo_kills` alpha, which should not be interpreted literally as "solo kills are bad"; it is more likely a redundancy/instability issue caused by overlap with kills, gold, damage, and diff stats
- `baseline_zscore` gives the most useful importance ranking for discussion because its inputs are on comparable role-relative scales
- `baseline_fixed_raw` remains useful as a sanity-check baseline because it is less overconfident in held-out win prediction

### 3.2.2 Standardized Model Sweep Results

The standardized z-score model was swept over:

- `beta_performance`: `0.25`, `0.5`, `0.75`, `1.0`
- `alpha_prior_std`: `0.5`, `1.0`
- `tau_floor`: `0.0`, `0.05`

The sweep results are saved in:

- `data/analysis/baseline_zscore_standardized_sweep_results.csv`

Best single run by Brier/log-loss:

| run | beta | alpha prior std | tau floor | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|---:|---:|
| `std_beta1p0_alpha1p0_tau0p05` | 1.0 | 1.0 | 0.05 | 59.5% | 0.2325 | 0.6572 |

Interpretation:

- `beta=0.25` performed badly, suggesting per-game performance was too tightly tied to long-term skill
- larger `beta` values improved calibration in the current skill-only proxy test
- `alpha_prior_std=1.0` was slightly better than `0.5` in the best tau-floor run, but the difference was small
- adding `tau_floor=0.05` did not remove the gold signal; it simply prevents tau from collapsing all the way toward zero
- the current default in `models/baseline_zscore_standardized.py` is therefore `beta=1.0`, `alpha_prior_std=1.0`, and `tau_floor=0.05`

Important caveat:

- these are single SVI runs without fixed random seed replication
- use them as directional evidence, not as a final hyperparameter proof
- a later robust comparison should repeat promising settings over several seeds

Seed-repeat check:

After adding `--seed` to `train.py`, the two main standardized candidates were rerun for 1500 steps each over seeds `1` through `5`.

The detailed per-seed results are saved in:

- `data/analysis/baseline_zscore_standardized_seed_repeats.csv`

The aggregated summary is saved in:

- `data/analysis/baseline_zscore_standardized_seed_repeat_summary.csv`

| setting | beta | alpha prior std | tau floor | seeds | mean accuracy | accuracy range | mean Brier | mean log-loss |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| accuracy candidate | 0.5 | 0.5 | 0.0 | 5 | 59.68% | 58.6%-62.2% | 0.2383 | 0.6695 |
| calibration candidate | 1.0 | 1.0 | 0.05 | 5 | 60.04% | 59.5%-62.2% | 0.2325 | 0.6572 |

Interpretation of the seed-repeat check:

- the earlier 62.2% accuracy was not stable evidence for the accuracy candidate
- seed `5` reached 62.2% for both candidate settings
- the calibration candidate had slightly higher mean accuracy and clearly better Brier/log-loss across seeds
- this supports keeping `beta=1.0`, `alpha_prior_std=1.0`, and `tau_floor=0.05` as the current default

Learned quantities for the best run are saved in:

- `data/analysis/std_beta1p0_alpha1p0_tau0p05_learned_quantities.csv`

Top signals for the best run:

| signal | effective tau | importance |
|---|---:|---:|
| `golds` | 0.0603 | 8.6613 |
| `golds_diff_vs_role_opp` | 0.0608 | 7.6714 |
| `kills` | 0.6078 | 1.1088 |
| `kills_diff_vs_role_opp` | 0.6282 | 0.5903 |
| `deaths_diff_vs_role_opp` | 0.6534 | 0.5418 |

The gold variables are still strongest, but the additive floor makes their effective tau about `0.06` instead of allowing values around `0.01` or lower.

Two candidate settings were then rerun for 4000 steps:

| run | beta | alpha prior std | tau floor | steps | accuracy | Brier | log-loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| `std_accuracy_candidate_4000` | 0.5 | 0.5 | 0.0 | 4000 | 59.5% | 0.2394 | 0.6717 |
| `std_calibration_candidate_4000` | 1.0 | 1.0 | 0.05 | 4000 | 59.5% | 0.2334 | 0.6592 |

Interpretation of the 4000-step rerun:

- the earlier 62.2% accuracy for the `beta=0.5`, `alpha_prior_std=0.5`, no-floor setting did not persist in this longer run
- both candidate settings reached the same held-out accuracy after 4000 steps
- the `beta=1.0`, `alpha_prior_std=1.0`, `tau_floor=0.05` setting remained better calibrated
- based on current evidence, the calibration candidate is the better default, but repeated random seeds are still needed before treating this as final

### 3.3 Future Alpha/Tau Relaxation Plan

Important interpretation:

- `alpha` is the slope, direction, and importance signal
- `tau` is the noise and reliability signal
- a stat is strong evidence when `abs(alpha)` is large relative to `tau`
- raw `alpha` and raw `tau` are not directly comparable across stats because stats have different units
- the z-score model makes comparison easier because each player-centric stat is on a role-relative standardized scale

Potential tau designs:

1. Fixed numeric tau
   Most rigid and assumption-heavy.

2. Role-specific low/medium/high tau categories
   Recommended next step after the current shared-stat baselines. This is still interpretable but less rigid than fixed numeric tau. It lets the model or model designer say that a stat is reliable, medium-noisy, or highly context-dependent for a given role.

3. Fully learned free tau
   Most flexible, but easiest to destabilize or make hard to explain.

The low/medium/high approach should not be described as "more assumptions than the original model." It is more structured than fully free tau, but compared to fixed numeric tau values, it relaxes assumptions because tau can vary by role/stat category.

Examples of why role-specific tau matters:

- `total_damage_taken` may be medium-noise evidence for TOP tanks but high-noise evidence for ADC
- `vision_score` may be more reliable for SUPPORT/JUNGLE than ADC
- `cs` may be reliable for ADC/MID/TOP but much less meaningful for SUPPORT
- `solo_kills` may be important for TOP/MID/JUNGLE/ADC but less relevant for SUPPORT

This future model should make the project better aligned with League of Legends domain knowledge: roles do not only have different average stat values; they also differ in which stats are meaningful and how noisy those stats are.

Important clarification for the raw-free model:

- the old alpha values in `models/config.py` are still used
- they are not fixed coefficients in `baseline_free_raw`
- instead, they are used as the prior center for learned alpha
- old tau values are also used as the prior center for learned tau

In the z-score model:

```text
alpha starts around 0
tau starts from low/medium/high category priors
gamma is removed
```

Future role-specific importance model:

```text
alpha_role,stat = learned importance of stat for that role
tau_role,stat = learned or category-constrained reliability/noise for that role/stat
```

This is still planned after `game_rules`. It should let the model express assumptions such as:

- `alpha[ADC, cs] > alpha[SUPPORT, cs]`
- `alpha[SUPPORT, vision_score] > alpha[ADC, vision_score]`
- `alpha[TOP/MID/JUNGLE/ADC, solo_kills] > alpha[SUPPORT, solo_kills]`
- `tau[ADC, total_damage_taken]` may be higher than `tau[TOP, total_damage_taken]`

### 3.4 Learned-Alpha Experiment Findings

Several learned-alpha baseline runs were tested to understand how the model changes the old hand-set alpha assumptions.

Runs:

- `baseline_unconstrained_damage_taken_1500`
- `baseline_unconstrained_damage_taken_4000`
- `baseline_solo_kills_alpha_0p1_1500`

The current held-out evaluation is still only the skill-based win-prediction proxy, so these results should not be overinterpreted as direct proof of individual performance attribution quality.

Evaluation summary:

| run | steps | accuracy | Brier | log-loss | note |
|---|---:|---:|---:|---:|---|
| fixed-alpha old context run | 1500 | 61.3% | 0.3172 | 1.8175 | old `baseline_team_diff_1500` comparison point |
| learned alpha, `total_damage_taken` unconstrained | 1500 | 61.3% | 0.3295 | 2.2181 | same accuracy, worse calibration |
| learned alpha, `total_damage_taken` unconstrained | 4000 | 61.3% | 0.3683 | 4.4138 | much more overconfident on test |
| learned alpha, boosted `solo_kills` prior | 1500 | 60.4% | 0.3198 | 1.9869 | slightly lower accuracy, better calibration than learned-alpha 1500 |

Learned alpha comparison after 4000 steps:

| stat | old alpha | learned alpha | ratio |
|---|---:|---:|---:|
| `kills` | 0.05 | 0.0664 | 1.33 |
| `deaths` | -0.05 | -0.0633 | 1.27 |
| `assists` | 0.05 | 0.0662 | 1.32 |
| `cs` | 1.00 | 1.106 | 1.11 |
| `golds` | 30.0 | 45.03 | 1.50 |
| `total_damage_to_champion` | 100.0 | 128.70 | 1.29 |
| `total_damage_taken` | 50.0 | 50.60 | 1.01 |
| `solo_kills` | 0.01 | 0.000044 | 0.004 |
| `shutdown_bounty_lost` | -2.0 | -1.80 | 0.90 |

Interpretation:

- the model increased the strength of golds, kills/deaths/assists, and champion damage
- `total_damage_taken` remained close to the old positive value even when allowed to learn either sign
- `solo_kills` collapsed almost to zero in the learned-alpha setup
- 4000 steps made rankings sharper but test calibration worse, suggesting overconfidence or overfitting under the current evaluation

Solo-kill experiment:

- domain expectation: solo kills are likely very important for TOP, JUNGLE, MID, and ADC
- test change: `solo_kills` prior alpha was raised from `0.01` to `0.1`, making it start twice as large as ordinary `kills`
- result after 1500 steps: learned `solo_kills` alpha still fell to about `0.0041`

Key caveat:

- this does not prove solo kills are unimportant in League of Legends
- it means that in the current model structure, `solo_kills` is not being used as a separate explanatory signal
- likely reasons include sparsity, overlap with `kills`, `golds`, `damage`, and same-role diff stats, and the current fixed `tau`/role-baseline setup

Possible next tests for solo kills:

- fix `solo_kills` alpha instead of learning it
- use a much tighter prior so it cannot collapse toward zero
- make `solo_kills` role-specific, since it may matter much more for TOP/MID/JUNGLE/ADC than SUPPORT
- inspect case studies where solo kills occurred and compare inferred performance with and without stronger solo-kill assumptions
- reconsider `tau` for sparse event stats like solo/multi-kills

There is an important alternative model structure that may better capture teammate influence:

```text
player skill + teammate/team influence -> latent game performance -> observed stats
```

That would make team or teammate context affect the latent performance variable itself. This has a different interpretation: it says context changes how well the player actually performs in the game, rather than only changing how raw stats should be interpreted.

Likely future direction:

- use `Duration` and broad team tempo variables as observation-level measurement context
- use explicit teammate/player interaction terms as performance-level influence context
- keep those two ideas separate, because they answer different modeling questions

Useful current model commands are listed in the baseline comparison section above.

### 3.5 Case-Study Inspection

Because the ideal target is post-game individual performance inference, not only held-out winner prediction, the project now includes a case-study script:

- `case_study.py`

The script takes a `game_block_id`, a model name, a saved params file, and the CSV used for training. It prints:

- game metadata
- team/game context
- raw player stats
- same-role opponent diff stats, when present
- inferred per-game latent performance values from `pa_loc` and `pb_loc`
- performance uncertainty from `pa_scale` and `pb_scale`

Important interpretation:

- `pa_loc` contains the inferred per-game performances for the five players on Team A
- `pb_loc` contains the inferred per-game performances for the five players on Team B
- in the current loaders, Team A is the winning team and Team B is the losing team
- these are not long-term skill values
- long-term player-role skill is stored in `s_loc`

The relationship is:

```text
long-term player-role skill s_i,r
    -> single-game player performance p_i,g
    -> observed player stats and team result
```

Useful commands:

```bash
python case_study.py --model baseline_fixed_raw --params params/baseline_fixed_raw_1500.pt --csv-path data/lck_s15_games_MODEL-READY_train.csv --game-block-id 70
python case_study.py --model baseline_free_raw --params params/baseline_free_raw_1500.pt --csv-path data/lck_s15_games_MODEL-READY_train.csv --game-block-id 70
python case_study.py --model baseline_zscore --params params/baseline_zscore_1500.pt --csv-path data/lck_s15_games_MODEL-READY_train.csv --game-block-id 70
```

Example case-study choices:

- `game_block_id=70`: high-action T1 vs NS game with 54 total kills
- `game_block_id=109`: low-kill HLE vs GEN game with 9 total kills

These are useful together because they help inspect whether the model distinguishes raw stat totals from game/team context. For example, 10 kills in a 54-kill game should not necessarily mean the same thing as a much smaller kill total in a very low-kill game.

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

### 4.1 How `baseline_zscore` Calculates Standardized Inputs

The current z-score model calculates standardization inside `models/baseline_zscore.py` when loading the CSV.

Important implementation detail:

- the means and standard deviations are calculated from whichever CSV is passed to `train.py`
- when training with `data/lck_s15_games_MODEL-READY_train.csv`, the z-score reference values come from the training split
- future post-game/stat-based evaluation on held-out games should reuse training-set means/stds rather than recomputing them on the held-out set
- the current `test.py` does not use observed held-out stats, so this issue does not affect the current skill-only test

#### Individual Player Stats

Individual player stats are z-scored within role.

Formula:

```text
player_stat_role_z =
    (player_stat - mean(player_stat for same role))
    / std(player_stat for same role)
```

Examples:

- TOP `cs` is compared against other TOP `cs` rows
- SUPPORT `vision_score` is compared against other SUPPORT `vision_score` rows
- ADC `total_damage_to_champion` is compared against other ADC `total_damage_to_champion` rows

This means:

- `0` means approximately role-average
- `+1` means one role-specific standard deviation above average
- `-1` means one role-specific standard deviation below average

The model uses these as direct noisy observations of individual latent performance:

```text
stat_z_i,k ~ Normal(
    alpha_k * (performance_i - 25)
  + duration_context_effect_k
  + team_context_effect_k
  , tau_k
)
```

Why `performance_i - 25` is used:

- the skill/performance prior is centered at `25`
- after z-scoring, an average stat has expected value near `0`
- subtracting `25` means an average latent performance contributes roughly zero stat deviation

#### Same-Role Diff Stats

Diff stats are also z-scored within role.

Formula:

```text
diff_stat_role_z =
    (diff_stat - mean(diff_stat for same role))
    / std(diff_stat for same role)
```

Examples:

- TOP `golds_diff_vs_role_opp` is compared against other TOP gold-diff rows
- JUNGLE `kills_diff_vs_role_opp` is compared against other JUNGLE kill-diff rows
- SUPPORT `vision_diff_vs_role_opp` is compared against other SUPPORT vision-diff rows

The model uses diff stats differently from raw individual stats. They are evidence about relative same-role performance:

```text
diff_stat_z_A_role ~ Normal(
    diff_alpha_k * (performance_A_role - performance_B_role),
    diff_tau_k
)

diff_stat_z_B_role ~ Normal(
    diff_alpha_k * (performance_B_role - performance_A_role),
    diff_tau_k
)
```

Important distinction:

- individual stats explain how well one player performed relative to role-normal expectations
- diff stats explain how much better/worse that player performed than the same-role opponent
- duration and team context are not added to the diff-stat likelihood in the current z-score model

#### Duration

`Duration` is not role-wise z-scored.

Instead, it is parsed into minutes and standardized globally once per game:

```text
duration_minutes_z =
    (duration_minutes - mean(duration_minutes over games))
    / std(duration_minutes over games)
```

Counting rule:

- each game contributes one duration value
- duration is not counted once per player
- this avoids shrinking or distorting the duration standard deviation by repeating the same value ten times per game

The model uses duration as context, not as an individual performance observation:

```text
expected_stat_z_i,k += duration_minutes_z * duration_effect_k
```

Interpretation:

- longer games naturally inflate many raw event totals
- duration context lets the model explain some stat increases as game length rather than player performance
- duration affects the expected observed stats, not the player's latent skill directly

#### Team Context Stats

The selected `team_*` stats are not role-wise z-scored.

They are standardized globally across team-game rows:

```text
team_stat_z =
    (team_stat - mean(team_stat over team-games))
    / std(team_stat over team-games)
```

Counting rule:

- each game contributes two team rows: one winning team and one losing team
- each team stat is not counted once per player
- this avoids counting the same team total five times

Examples:

- `team_kills`
- `team_deaths`
- `team_assists`
- `team_cs`
- `team_golds`
- `team_vision_score`
- `team_total_damage_to_champion`

The model uses team stats as team-output context:

```text
expected_stat_z_i,k += sum_j(team_context_z_j * team_output_effect_j,k)
```

Interpretation:

- team stats are not treated as individual player achievements
- they are contextual variables that help explain the environment in which individual stats occurred
- example: 10 kills in a very high-kill team game can be interpreted differently from 10 kills in a low-kill team game

Current conceptual split:

| input type | standardization | model role |
|---|---|---|
| individual player stats | role-wise z-score | direct noisy evidence of individual latent performance |
| same-role diff stats | role-wise z-score | direct noisy evidence of same-role performance gap |
| `Duration` | global z-score once per game | context effect on expected observed stats |
| `team_*` stats | global z-score once per team-game | team-output context effect on expected observed stats |

This split is important because not all inputs mean the same thing. Individual and diff stats are evidence about performance; duration and team totals are context that changes how observed player stats should be interpreted.

### 4.2 Standardized Latent Skill Scale

The newer z-score baseline is:

- `models/baseline_zscore_standardized.py`

This keeps the same z-scored inputs as `baseline_zscore`, but changes the latent skill/performance scale to be easier to interpret.

Old z-score scale:

```text
skill ~ Normal(25, 25/3)
performance ~ Normal(skill, 25/6)
stat_z ~ Normal(alpha * (performance - 25) + context, tau)
team_performance = sum(five player performances)
```

New standardized scale:

```text
skill ~ Normal(0, 1)
performance ~ Normal(skill, 0.5)
stat_z ~ Normal(alpha * performance + context, tau)
team_performance = mean(five player performances)
```

Why this is easier:

- `0` means average latent skill/performance before observing data
- positive skill means above average
- negative skill means below average
- one skill unit is one prior skill standard deviation
- there is no need for `(performance - 25)` because the latent performance scale and z-scored stat scale are both centered at zero

Current priors in `baseline_zscore_standardized`:

| quantity | prior / fixed value | meaning |
|---|---|---|
| player-role skill `s` | `Normal(0, 1)` | all players start equal at average skill |
| per-game performance `p` | `Normal(skill, 1.0)` | game performance can vary around long-term skill |
| `beta_performance` | fixed `1.0` | not learned yet |
| result noise | fixed `1.0` | uncertainty in mapping team performance gap to win/loss |
| individual alpha | `Normal(0, 1.0)` | no assumed direction; broad prior freedom |
| individual tau | `0.05 + LogNormal(log(0.95), 0.35)` | every individual stat starts with effective noise centered near `1`, with a hard lower floor |
| diff alpha | `Normal(0, 1.0)` | no assumed direction for diff effects |
| diff tau | `0.05 + LogNormal(log(0.95), 0.35)` | every diff stat starts with effective noise centered near `1`, with a hard lower floor |
| duration effects | `Normal(0, 0.5)` | no assumed direction for duration context |
| team-context effects | `Normal(0, 0.5)` | no assumed direction for team-output context |

Important clarification about `beta_performance`:

- it is currently fixed, not learned
- it controls how much a player's single-game performance can deviate from their long-term skill
- small `beta_performance` means players are assumed to perform close to their skill every game
- large `beta_performance` means single-game performances can swing much more around long-term skill
- the current value `1.0` was chosen after a small sweep where larger beta values improved calibration in the current proxy test

Future option:

- learn `beta_performance` globally with a positive prior
- or learn role-specific `beta_performance_role`, because some roles may have more volatile game-to-game observed impact than others

This should be a deliberate later change, because learning beta changes how strongly individual games update long-term skill.

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

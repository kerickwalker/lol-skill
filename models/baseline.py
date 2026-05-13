"""Canonical full-context baseline for LCK S15 player skill.

The model reads raw rows from ``data/lck_s15_games_MODEL-READY*.csv`` and
standardizes the modeling inputs internally:

- individual player stats and same-role opponent diff stats are z-scored
  within role;
- duration is z-scored once per game;
- team context stats are z-scored once per team-game.

This keeps the CSV human-readable while letting the Bayesian model work on
comparable scales.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models.config import INDIVIDUAL_STATS, N_ROLES, ROLE_MAP, ROLES


DURATION_CONTEXT_STATS = ["duration_minutes"]
TEAM_CONTEXT_STATS = [
    "team_kills",
    "team_deaths",
    "team_assists",
    "team_cs",
    "team_golds",
    "team_vision_score",
    "team_total_damage_to_champion",
]
DIFF_STATS = [
    "kills_diff_vs_role_opp",
    "deaths_diff_vs_role_opp",
    "assists_diff_vs_role_opp",
    "cs_diff_vs_role_opp",
    "golds_diff_vs_role_opp",
    "vision_diff_vs_role_opp",
    "damage_diff_vs_role_opp",
]

SKILL_PRIOR_MEAN = 0.0
SKILL_PRIOR_STD = 1.0
PERFORMANCE_BETA = float(os.environ.get("LCK_PERFORMANCE_BETA", "1.0"))
RESULT_NOISE = 1.0
ALPHA_PRIOR_STD = float(os.environ.get("LCK_ALPHA_PRIOR_STD", "1.0"))
TAU_PRIOR_CENTER = 1.0
TAU_PRIOR_LOG_STD = 0.35
TAU_FLOOR = float(os.environ.get("LCK_TAU_FLOOR", "0.05"))
CONTEXT_PRIOR_STD = 0.5
USE_SCORE_TABLE_AS_MAIN_OUTPUT = True

RAW_TAU_PRIOR_CENTER = max(TAU_PRIOR_CENTER - TAU_FLOOR, 1e-3)
BASE_TAU_VEC = RAW_TAU_PRIOR_CENTER * torch.ones(len(INDIVIDUAL_STATS), dtype=torch.float32)
BASE_DIFF_TAU_VEC = RAW_TAU_PRIOR_CENTER * torch.ones(len(DIFF_STATS), dtype=torch.float32)
ALPHA_PRIOR_SCALE = ALPHA_PRIOR_STD * torch.ones(len(INDIVIDUAL_STATS), dtype=torch.float32)
DIFF_ALPHA_PRIOR_SCALE = ALPHA_PRIOR_STD * torch.ones(len(DIFF_STATS), dtype=torch.float32)


def parse_duration_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", maxsplit=1)
        return float(minutes) + float(seconds) / 60.0
    return float(value)


def _role_zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        values = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        mean = values.groupby(out["role_idx"]).transform("mean")
        std = values.groupby(out["role_idx"]).transform("std").replace(0, 1.0).fillna(1.0)
        out[col] = (values - mean) / std
    return out


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std, mean, std


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df["role_idx"] = df["role"].map(ROLE_MAP)
    if df["role_idx"].isna().any():
        unknown = sorted(df.loc[df["role_idx"].isna(), "role"].dropna().unique())
        raise ValueError(f"Unknown roles in {csv_path}: {unknown}")

    df = _role_zscore(df, [*INDIVIDUAL_STATS, *DIFF_STATS])

    unique_pids = sorted(df["player_id"].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    df["pid_idx"] = df["player_id"].map(pid_to_idx)
    n_players = len(unique_pids)

    idx_to_name = (
        df[["pid_idx", "player_name"]]
        .drop_duplicates()
        .set_index("pid_idx")["player_name"]
        .to_dict()
    )
    primary_role = df.groupby("pid_idx")["role_idx"].agg(lambda x: x.mode().iloc[0]).to_dict()

    team_a_pid, team_a_role, team_b_pid, team_b_role, winners = [], [], [], [], []
    duration_context, team_output_context_a, team_output_context_b = [], [], []
    stats_a_by_name = {stat: [] for stat in INDIVIDUAL_STATS}
    stats_b_by_name = {stat: [] for stat in INDIVIDUAL_STATS}
    diff_stats_a_by_name = {stat: [] for stat in DIFF_STATS}
    diff_stats_b_by_name = {stat: [] for stat in DIFF_STATS}

    for _, group in df.groupby("game_block_id", sort=True):
        winners_df = group[group["Result"] == "Victory"].sort_values("role_idx")
        losers_df = group[group["Result"] == "Defeat"].sort_values("role_idx")
        if len(winners_df) != 5 or len(losers_df) != 5:
            continue

        team_a_pid.append(winners_df["pid_idx"].to_numpy(dtype=np.int64))
        team_a_role.append(winners_df["role_idx"].to_numpy(dtype=np.int64))
        team_b_pid.append(losers_df["pid_idx"].to_numpy(dtype=np.int64))
        team_b_role.append(losers_df["role_idx"].to_numpy(dtype=np.int64))
        winners.append(1.0)
        duration_context.append(
            np.array([parse_duration_minutes(winners_df.iloc[0]["Duration"])], dtype=np.float32)
        )
        team_output_context_a.append(
            pd.to_numeric(winners_df.iloc[0][TEAM_CONTEXT_STATS], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        team_output_context_b.append(
            pd.to_numeric(losers_df.iloc[0][TEAM_CONTEXT_STATS], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )
        for stat in INDIVIDUAL_STATS:
            stats_a_by_name[stat].append(winners_df[stat].to_numpy(dtype=np.float32))
            stats_b_by_name[stat].append(losers_df[stat].to_numpy(dtype=np.float32))
        for stat in DIFF_STATS:
            diff_stats_a_by_name[stat].append(winners_df[stat].to_numpy(dtype=np.float32))
            diff_stats_b_by_name[stat].append(losers_df[stat].to_numpy(dtype=np.float32))

    stats_a = np.stack([np.stack(stats_a_by_name[stat], axis=0) for stat in INDIVIDUAL_STATS], axis=-1)
    stats_b = np.stack([np.stack(stats_b_by_name[stat], axis=0) for stat in INDIVIDUAL_STATS], axis=-1)
    diff_stats_a = np.stack([np.stack(diff_stats_a_by_name[stat], axis=0) for stat in DIFF_STATS], axis=-1)
    diff_stats_b = np.stack([np.stack(diff_stats_b_by_name[stat], axis=0) for stat in DIFF_STATS], axis=-1)

    duration_context = np.stack(duration_context, axis=0)
    duration_context, duration_mean, duration_std = _standardize(duration_context)

    team_output_context_a = np.stack(team_output_context_a, axis=0)
    team_output_context_b = np.stack(team_output_context_b, axis=0)
    team_output_context_all = np.concatenate([team_output_context_a, team_output_context_b], axis=0)
    _, team_output_mean, team_output_std = _standardize(team_output_context_all)

    batch = {
        "team_a_pid": torch.tensor(np.stack(team_a_pid, axis=0), dtype=torch.long),
        "team_a_role": torch.tensor(np.stack(team_a_role, axis=0), dtype=torch.long),
        "team_b_pid": torch.tensor(np.stack(team_b_pid, axis=0), dtype=torch.long),
        "team_b_role": torch.tensor(np.stack(team_b_role, axis=0), dtype=torch.long),
        "winner": torch.tensor(winners, dtype=torch.float32),
        "stats_a": torch.tensor(stats_a, dtype=torch.float32),
        "stats_b": torch.tensor(stats_b, dtype=torch.float32),
        "diff_stats_a": torch.tensor(diff_stats_a, dtype=torch.float32),
        "diff_stats_b": torch.tensor(diff_stats_b, dtype=torch.float32),
        "duration_context": torch.tensor(duration_context, dtype=torch.float32),
        "duration_context_mean": torch.tensor(duration_mean, dtype=torch.float32),
        "duration_context_std": torch.tensor(duration_std, dtype=torch.float32),
        "team_output_context_a": torch.tensor(
            (team_output_context_a - team_output_mean) / team_output_std,
            dtype=torch.float32,
        ),
        "team_output_context_b": torch.tensor(
            (team_output_context_b - team_output_mean) / team_output_std,
            dtype=torch.float32,
        ),
        "team_output_context_mean": torch.tensor(team_output_mean, dtype=torch.float32),
        "team_output_context_std": torch.tensor(team_output_std, dtype=torch.float32),
    }

    print(f"Loaded {batch['winner'].shape[0]} matches, {n_players} players")
    return batch, n_players, idx_to_name, primary_role


def model(batch, n_players):
    n_matches = batch["winner"].shape[0]

    s = pyro.sample(
        "s",
        dist.Normal(SKILL_PRIOR_MEAN, SKILL_PRIOR_STD)
        .expand([n_players, N_ROLES])
        .to_event(2),
    )
    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]

    alpha_vec = pyro.sample(
        "alpha_vec",
        dist.Normal(torch.zeros(len(INDIVIDUAL_STATS)), ALPHA_PRIOR_SCALE).to_event(1),
    )
    tau_raw_vec = pyro.sample(
        "tau_vec",
        dist.LogNormal(
            torch.log(BASE_TAU_VEC),
            TAU_PRIOR_LOG_STD * torch.ones_like(BASE_TAU_VEC),
        ).to_event(1),
    )
    tau_vec = tau_raw_vec + TAU_FLOOR
    diff_alpha_vec = pyro.sample(
        "diff_alpha_vec",
        dist.Normal(torch.zeros(len(DIFF_STATS)), DIFF_ALPHA_PRIOR_SCALE).to_event(1),
    )
    diff_tau_raw_vec = pyro.sample(
        "diff_tau_vec",
        dist.LogNormal(
            torch.log(BASE_DIFF_TAU_VEC),
            TAU_PRIOR_LOG_STD * torch.ones_like(BASE_DIFF_TAU_VEC),
        ).to_event(1),
    )
    diff_tau_vec = diff_tau_raw_vec + TAU_FLOOR
    duration_effect = pyro.sample(
        "duration_effect",
        dist.Normal(
            torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            CONTEXT_PRIOR_STD,
        ).to_event(2),
    )
    team_output_effect = pyro.sample(
        "team_output_effect",
        dist.Normal(
            torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            CONTEXT_PRIOR_STD,
        ).to_event(2),
    )

    with pyro.plate("matches", n_matches):
        p_a = pyro.sample("p_a", dist.Normal(skill_a, PERFORMANCE_BETA).to_event(1))
        p_b = pyro.sample("p_b", dist.Normal(skill_b, PERFORMANCE_BETA).to_event(1))

        team_perf_a = p_a.mean(-1)
        team_perf_b = p_b.mean(-1)
        win_prob = dist.Normal(0.0, 1.0).cdf(
            (team_perf_a - team_perf_b) / (torch.sqrt(torch.tensor(2.0)) * RESULT_NOISE)
        )
        pyro.sample("y", dist.Bernoulli(win_prob), obs=batch["winner"])

        mean_a = p_a.unsqueeze(-1) * alpha_vec.view(1, 1, -1)
        mean_b = p_b.unsqueeze(-1) * alpha_vec.view(1, 1, -1)
        duration_shift = batch["duration_context"] @ duration_effect
        mean_a = mean_a + duration_shift.unsqueeze(1)
        mean_b = mean_b + duration_shift.unsqueeze(1)
        mean_a = mean_a + (batch["team_output_context_a"] @ team_output_effect).unsqueeze(1)
        mean_b = mean_b + (batch["team_output_context_b"] @ team_output_effect).unsqueeze(1)

        pyro.sample(
            "obs_a",
            dist.Normal(mean_a, tau_vec.view(1, 1, -1)).to_event(2),
            obs=batch["stats_a"],
        )
        pyro.sample(
            "obs_b",
            dist.Normal(mean_b, tau_vec.view(1, 1, -1)).to_event(2),
            obs=batch["stats_b"],
        )

        diff_mean_a = (p_a - p_b).unsqueeze(-1) * diff_alpha_vec.view(1, 1, -1)
        diff_mean_b = (p_b - p_a).unsqueeze(-1) * diff_alpha_vec.view(1, 1, -1)
        pyro.sample(
            "diff_obs_a",
            dist.Normal(diff_mean_a, diff_tau_vec.view(1, 1, -1)).to_event(2),
            obs=batch["diff_stats_a"],
        )
        pyro.sample(
            "diff_obs_b",
            dist.Normal(diff_mean_b, diff_tau_vec.view(1, 1, -1)).to_event(2),
            obs=batch["diff_stats_b"],
        )


def guide(batch, n_players):
    n_matches = batch["winner"].shape[0]

    s_loc = pyro.param("s_loc", SKILL_PRIOR_MEAN * torch.ones(n_players, N_ROLES))
    s_scale = pyro.param(
        "s_scale",
        SKILL_PRIOR_STD * torch.ones(n_players, N_ROLES),
        constraint=dist.constraints.positive,
    )
    pyro.sample("s", dist.Normal(s_loc, s_scale).to_event(2))

    alpha_loc = pyro.param("alpha_loc", torch.zeros(len(INDIVIDUAL_STATS)))
    alpha_scale = pyro.param(
        "alpha_scale",
        0.1 * ALPHA_PRIOR_SCALE,
        constraint=dist.constraints.positive,
    )
    pyro.sample("alpha_vec", dist.Normal(alpha_loc, alpha_scale).to_event(1))

    tau_log_loc = pyro.param("tau_log_loc", torch.log(BASE_TAU_VEC))
    tau_log_scale = pyro.param(
        "tau_log_scale",
        0.1 * torch.ones_like(BASE_TAU_VEC),
        constraint=dist.constraints.positive,
    )
    pyro.sample("tau_vec", dist.LogNormal(tau_log_loc, tau_log_scale).to_event(1))

    diff_alpha_loc = pyro.param("diff_alpha_loc", torch.zeros(len(DIFF_STATS)))
    diff_alpha_scale = pyro.param(
        "diff_alpha_scale",
        0.1 * DIFF_ALPHA_PRIOR_SCALE,
        constraint=dist.constraints.positive,
    )
    pyro.sample("diff_alpha_vec", dist.Normal(diff_alpha_loc, diff_alpha_scale).to_event(1))

    diff_tau_log_loc = pyro.param("diff_tau_log_loc", torch.log(BASE_DIFF_TAU_VEC))
    diff_tau_log_scale = pyro.param(
        "diff_tau_log_scale",
        0.1 * torch.ones_like(BASE_DIFF_TAU_VEC),
        constraint=dist.constraints.positive,
    )
    pyro.sample("diff_tau_vec", dist.LogNormal(diff_tau_log_loc, diff_tau_log_scale).to_event(1))

    duration_loc = pyro.param(
        "duration_effect_loc",
        torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    duration_scale = pyro.param(
        "duration_effect_scale",
        0.1 * torch.ones(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("duration_effect", dist.Normal(duration_loc, duration_scale).to_event(2))

    team_output_loc = pyro.param(
        "team_output_effect_loc",
        torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    team_output_scale = pyro.param(
        "team_output_effect_scale",
        0.1 * torch.ones(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("team_output_effect", dist.Normal(team_output_loc, team_output_scale).to_event(2))

    pa_loc = pyro.param("pa_loc", SKILL_PRIOR_MEAN * torch.ones(n_matches, 5))
    pa_scale = pyro.param(
        "pa_scale",
        PERFORMANCE_BETA * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )
    pb_loc = pyro.param("pb_loc", SKILL_PRIOR_MEAN * torch.ones(n_matches, 5))
    pb_scale = pyro.param(
        "pb_scale",
        PERFORMANCE_BETA * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("matches", n_matches):
        pyro.sample("p_a", dist.Normal(pa_loc, pa_scale).to_event(1))
        pyro.sample("p_b", dist.Normal(pb_loc, pb_scale).to_event(1))


def train(batch, n_players, n_steps=1500, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    disable_progress = os.environ.get("LCK_DISABLE_TQDM", "0") == "1"
    with trange(n_steps, desc="Training", unit="step", disable=disable_progress) as bar:
        for _ in bar:
            loss = svi.step(batch, n_players)
            losses.append(loss)
            if not disable_progress:
                bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses


def build_player_score_table(n_players, idx_to_name, primary_role):
    params = pyro.get_param_store()
    mu = params["s_loc"].detach().cpu().numpy()
    sigma = params["s_scale"].detach().cpu().numpy()

    rows = []
    for pid_idx in range(n_players):
        role_idx = primary_role[pid_idx]
        rows.append(
            {
                "player": idx_to_name[pid_idx],
                "role": ROLES[role_idx],
                "mu": mu[pid_idx, role_idx],
                "sigma": sigma[pid_idx, role_idx],
            }
        )

    scores = pd.DataFrame(rows)
    median_sigma = scores["sigma"].median()
    scores["uncertainty_adjusted_skill"] = scores["mu"] - 2.0 * (
        scores["sigma"] - median_sigma
    )
    ranks = scores["uncertainty_adjusted_skill"].rank(method="average")
    scores["score_0_100"] = 100.0 * (ranks - 1.0) / (len(scores) - 1.0)
    return scores.sort_values("score_0_100", ascending=False).reset_index(drop=True)


def print_score_table(n_players, idx_to_name, primary_role, limit=None):
    scores = build_player_score_table(n_players, idx_to_name, primary_role)
    display = scores if limit is None else scores.head(limit)

    print("\n" + "=" * 92)
    print("PLAYER SCORES (0-100 rank of mu - 2*(sigma - median_sigma))")
    print("=" * 92)
    print(
        display[
            ["player", "role", "mu", "sigma", "uncertainty_adjusted_skill", "score_0_100"]
        ].to_string(
            index=False,
            formatters={
                "mu": "{:.3f}".format,
                "sigma": "{:.3f}".format,
                "uncertainty_adjusted_skill": "{:.3f}".format,
                "score_0_100": "{:.1f}".format,
            },
        )
    )


def print_rankings(n_players, idx_to_name, primary_role):
    print_score_table(n_players, idx_to_name, primary_role)

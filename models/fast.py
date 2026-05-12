import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models.config import INDIVIDUAL_STATS, N_ROLES, ROLE_MAP, ROLES, STAT_CONFIG

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
DIFF_STAT_BASE = {
    "kills_diff_vs_role_opp": "kills",
    "deaths_diff_vs_role_opp": "deaths",
    "assists_diff_vs_role_opp": "assists",
    "cs_diff_vs_role_opp": "cs",
    "golds_diff_vs_role_opp": "golds",
    "vision_diff_vs_role_opp": "vision_score",
    "damage_diff_vs_role_opp": "total_damage_to_champion",
}


def parse_duration_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", maxsplit=1)
        return float(minutes) + float(seconds) / 60.0
    return float(value)


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["role_idx"] = df["role"].map(ROLE_MAP)

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
    primary_role = (
        df.groupby("pid_idx")["role_idx"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    team_a_pid = []
    team_a_role = []
    team_b_pid = []
    team_b_role = []
    winners = []
    duration_context = []
    team_output_context_a = []
    team_output_context_b = []
    stats_a_by_name = {stat: [] for stat in INDIVIDUAL_STATS}
    stats_b_by_name = {stat: [] for stat in INDIVIDUAL_STATS}
    diff_stats_a_by_name = {stat: [] for stat in DIFF_STATS}
    diff_stats_b_by_name = {stat: [] for stat in DIFF_STATS}

    for _, group in df.groupby("game_block_id", sort=True):
        winners_df = group[group["Result"] == "Victory"]
        losers_df = group[group["Result"] == "Defeat"]
        if len(winners_df) != 5 or len(losers_df) != 5:
            continue

        team_a_pid.append(winners_df["pid_idx"].to_numpy(dtype=np.int64))
        team_a_role.append(winners_df["role_idx"].to_numpy(dtype=np.int64))
        team_b_pid.append(losers_df["pid_idx"].to_numpy(dtype=np.int64))
        team_b_role.append(losers_df["role_idx"].to_numpy(dtype=np.int64))
        winners.append(1.0)
        duration_minutes = parse_duration_minutes(winners_df.iloc[0]["Duration"])
        duration_context.append(np.array([duration_minutes], dtype=np.float32))
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
            stats_a_by_name[stat].append(
                pd.to_numeric(winners_df[stat], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )
            stats_b_by_name[stat].append(
                pd.to_numeric(losers_df[stat], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )
        for stat in DIFF_STATS:
            diff_stats_a_by_name[stat].append(
                pd.to_numeric(winners_df[stat], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )
            diff_stats_b_by_name[stat].append(
                pd.to_numeric(losers_df[stat], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )

    stats_a = np.stack(
        [np.stack(stats_a_by_name[stat], axis=0) for stat in INDIVIDUAL_STATS],
        axis=-1,
    )
    stats_b = np.stack(
        [np.stack(stats_b_by_name[stat], axis=0) for stat in INDIVIDUAL_STATS],
        axis=-1,
    )
    diff_stats_a = np.stack(
        [np.stack(diff_stats_a_by_name[stat], axis=0) for stat in DIFF_STATS],
        axis=-1,
    )
    diff_stats_b = np.stack(
        [np.stack(diff_stats_b_by_name[stat], axis=0) for stat in DIFF_STATS],
        axis=-1,
    )
    duration_context = np.stack(duration_context, axis=0)
    duration_mean = duration_context.mean(axis=0)
    duration_std = duration_context.std(axis=0)
    duration_std = np.where(duration_std == 0, 1.0, duration_std)

    team_output_context_a = np.stack(team_output_context_a, axis=0)
    team_output_context_b = np.stack(team_output_context_b, axis=0)
    team_output_context_all = np.concatenate([team_output_context_a, team_output_context_b], axis=0)
    team_output_mean = team_output_context_all.mean(axis=0)
    team_output_std = team_output_context_all.std(axis=0)
    team_output_std = np.where(team_output_std == 0, 1.0, team_output_std)

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
        "duration_context": torch.tensor((duration_context - duration_mean) / duration_std, dtype=torch.float32),
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


def build_stat_tensors():
    alpha = torch.tensor([STAT_CONFIG[stat]["alpha"] for stat in INDIVIDUAL_STATS], dtype=torch.float32)
    tau = torch.tensor([STAT_CONFIG[stat]["tau"] for stat in INDIVIDUAL_STATS], dtype=torch.float32)
    gamma = torch.stack([STAT_CONFIG[stat]["gamma"].float() for stat in INDIVIDUAL_STATS], dim=0)
    return alpha, tau, gamma


def build_diff_stat_tensors():
    alpha = torch.tensor(
        [STAT_CONFIG[DIFF_STAT_BASE[stat]]["alpha"] for stat in DIFF_STATS],
        dtype=torch.float32,
    )
    tau = torch.tensor(
        [STAT_CONFIG[DIFF_STAT_BASE[stat]]["tau"] * np.sqrt(2.0) for stat in DIFF_STATS],
        dtype=torch.float32,
    )
    gamma = torch.zeros(len(DIFF_STATS), N_ROLES, dtype=torch.float32)
    return alpha, tau, gamma


ALPHA_VEC, TAU_VEC, GAMMA_MAT = build_stat_tensors()
DIFF_ALPHA_VEC, DIFF_TAU_VEC, DIFF_GAMMA_MAT = build_diff_stat_tensors()


def role_intercepts(gamma_mat: torch.Tensor, role_idx: torch.Tensor) -> torch.Tensor:
    # gamma_mat: [S, R], role_idx: [M, 5] -> [M, 5, S]
    return gamma_mat[:, role_idx].permute(1, 2, 0)


def model(batch, n_players, use_team_stats=False, use_diff_stats=False):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    beta = 25.0 / 6
    beta_o = 1.0

    n_matches = batch["winner"].shape[0]

    s = pyro.sample(
        "s",
        dist.Normal(mu_0, sigma_0).expand([n_players, N_ROLES]).to_event(2),
    )

    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]

    duration_effect_z = pyro.sample(
        "duration_effect_z",
        dist.Normal(
            torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            0.25 * torch.ones(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        ).to_event(2),
    )
    duration_effect = duration_effect_z * TAU_VEC.view(1, -1)

    if use_team_stats:
        team_output_effect_z = pyro.sample(
            "team_output_effect_z",
            dist.Normal(
                torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
                0.25 * torch.ones(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            ).to_event(2),
        )
        team_output_effect = team_output_effect_z * TAU_VEC.view(1, -1)

    with pyro.plate("matches", n_matches):
        p_a = pyro.sample("p_a", dist.Normal(skill_a, beta).to_event(1))
        p_b = pyro.sample("p_b", dist.Normal(skill_b, beta).to_event(1))

        t_a = p_a.sum(-1)
        t_b = p_b.sum(-1)
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample("y", dist.Bernoulli(win_prob), obs=batch["winner"])

        gamma_a = role_intercepts(GAMMA_MAT, batch["team_a_role"])
        gamma_b = role_intercepts(GAMMA_MAT, batch["team_b_role"])
        mean_a = p_a.unsqueeze(-1) * ALPHA_VEC.view(1, 1, -1) + gamma_a
        mean_b = p_b.unsqueeze(-1) * ALPHA_VEC.view(1, 1, -1) + gamma_b
        duration_shift = batch["duration_context"] @ duration_effect
        mean_a = mean_a + duration_shift.unsqueeze(1)
        mean_b = mean_b + duration_shift.unsqueeze(1)
        if use_team_stats:
            team_output_shift_a = batch["team_output_context_a"] @ team_output_effect
            team_output_shift_b = batch["team_output_context_b"] @ team_output_effect
            mean_a = mean_a + team_output_shift_a.unsqueeze(1)
            mean_b = mean_b + team_output_shift_b.unsqueeze(1)

        pyro.sample(
            "obs_a",
            dist.Normal(mean_a, TAU_VEC.view(1, 1, -1)).to_event(2),
            obs=batch["stats_a"],
        )
        pyro.sample(
            "obs_b",
            dist.Normal(mean_b, TAU_VEC.view(1, 1, -1)).to_event(2),
            obs=batch["stats_b"],
        )

        if use_diff_stats:
            diff_gamma_a = role_intercepts(DIFF_GAMMA_MAT, batch["team_a_role"])
            diff_gamma_b = role_intercepts(DIFF_GAMMA_MAT, batch["team_b_role"])
            diff_mean_a = p_a.unsqueeze(-1) * DIFF_ALPHA_VEC.view(1, 1, -1) + diff_gamma_a
            diff_mean_b = p_b.unsqueeze(-1) * DIFF_ALPHA_VEC.view(1, 1, -1) + diff_gamma_b
            pyro.sample(
                "diff_obs_a",
                dist.Normal(diff_mean_a, DIFF_TAU_VEC.view(1, 1, -1)).to_event(2),
                obs=batch["diff_stats_a"],
            )
            pyro.sample(
                "diff_obs_b",
                dist.Normal(diff_mean_b, DIFF_TAU_VEC.view(1, 1, -1)).to_event(2),
                obs=batch["diff_stats_b"],
            )


def guide(batch, n_players, use_team_stats=False, use_diff_stats=False):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    n_matches = batch["winner"].shape[0]

    s_loc = pyro.param("s_loc", mu_0 * torch.ones(n_players, N_ROLES))
    s_scale = pyro.param(
        "s_scale",
        sigma_0 * torch.ones(n_players, N_ROLES),
        constraint=dist.constraints.positive,
    )
    pyro.sample("s", dist.Normal(s_loc, s_scale).to_event(2))

    duration_loc = pyro.param(
        "duration_effect_z_loc",
        torch.zeros(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    duration_scale = pyro.param(
        "duration_effect_z_scale",
        0.1 * torch.ones(len(DURATION_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("duration_effect_z", dist.Normal(duration_loc, duration_scale).to_event(2))

    if use_team_stats:
        team_output_loc = pyro.param(
            "team_output_effect_z_loc",
            torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        )
        team_output_scale = pyro.param(
            "team_output_effect_z_scale",
            0.1 * torch.ones(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            constraint=dist.constraints.positive,
        )
        pyro.sample(
            "team_output_effect_z",
            dist.Normal(team_output_loc, team_output_scale).to_event(2),
        )

    pa_loc = pyro.param("pa_loc", 25.0 * torch.ones(n_matches, 5))
    pa_scale = pyro.param(
        "pa_scale",
        4.0 * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )
    pb_loc = pyro.param("pb_loc", 25.0 * torch.ones(n_matches, 5))
    pb_scale = pyro.param(
        "pb_scale",
        4.0 * torch.ones(n_matches, 5),
        constraint=dist.constraints.positive,
    )

    with pyro.plate("matches", n_matches):
        pyro.sample("p_a", dist.Normal(pa_loc, pa_scale).to_event(1))
        pyro.sample("p_b", dist.Normal(pb_loc, pb_scale).to_event(1))


def train(batch, n_players, n_steps=1500, lr=0.01, use_team_stats=False, use_diff_stats=False):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    with trange(n_steps, desc="Training", unit="step") as bar:
        for _ in bar:
            loss = svi.step(batch, n_players, use_team_stats, use_diff_stats)
            losses.append(loss)
            bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses


def print_rankings(n_players, idx_to_name, primary_role):
    params = pyro.get_param_store()
    mu = params["s_loc"].detach().cpu().numpy()
    sigma = params["s_scale"].detach().cpu().numpy()
    conservative = mu - 3 * sigma

    print("\n" + "=" * 65)
    print("SKILL RANKINGS BY ROLE  (conservative = mu - 3*sigma)")
    print("=" * 65)

    for r, role in enumerate(ROLES):
        role_players = [p for p, pr in primary_role.items() if pr == r]
        ranking = sorted(role_players, key=lambda p: conservative[p, r], reverse=True)
        print(f"\n  {role.upper()}")
        print(f"  {'-' * 55}")
        for rank, p in enumerate(ranking, 1):
            print(
                f"  {rank:2d}. {idx_to_name[p]:12s}  "
                f"mu={mu[p, r]:6.2f}  sigma={sigma[p, r]:5.2f}  "
                f"rating={conservative[p, r]:6.2f}"
            )

    print(f"\n{'=' * 65}")
    print("OVERALL RANKINGS (each player at their primary role)")
    print(f"{'=' * 65}")

    all_ratings = []
    for pid_idx in range(n_players):
        r = primary_role[pid_idx]
        all_ratings.append(
            (
                idx_to_name[pid_idx],
                ROLES[r],
                mu[pid_idx, r],
                sigma[pid_idx, r],
                conservative[pid_idx, r],
            )
        )
    all_ratings.sort(key=lambda x: x[4], reverse=True)

    for rank, (name, role, m, s, c) in enumerate(all_ratings, 1):
        print(
            f"  {rank:2d}. {name:12s} ({role:3s})  "
            f"mu={m:6.2f}  sigma={s:5.2f}  rating={c:6.2f}"
        )

"""
LCK Season 15 TrueSkill model — structured stat relationships.

Factor graph (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)
    Layer 2 — per-game performance:     p_i     ~ N(s_{i,r_i}, beta^2)
    Layer 3 — team sum:                 t_A = sum(p_i for i in A)
    Layer 4 — match outcome:            y ~ Bernoulli(Phi((t_A - t_B) / sqrt(2)*beta_o))
    Layer 5 — structured stat observations with causal edges:
                - golds depends on cs, kills, assists, shutdown bounty
                - time_dead depends on deaths
                - diff stats depend on performance gap vs same-role opponent
                - team aggregates (kills/golds/cs) constrain individual sums

Inference: SVI with mean-field Normal guide.
"""

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models.config import N_ROLES, ROLE_MAP, STAT_CONFIG as BASE_STAT_CONFIG
from models.fast import print_rankings

STAT_CONFIG = {
    **BASE_STAT_CONFIG,
    "golds_diff_vs_role_opp": {
        "alpha": 50.0,
        "gamma": torch.zeros(5),
        "tau": 1500.0,
    },
    "damage_diff_vs_role_opp": {
        "alpha": 150.0,
        "gamma": torch.zeros(5),
        "tau": 5000.0,
    },
}

# Stats loaded per-player per-match (shape [M, 5] each)
PLAYER_STATS = [
    "cs", "kills", "deaths", "assists",
    "shutdown_bounty_collected", "total_damage_to_champion",
    "golds", "total_time_spent_dead",
    "golds_diff_vs_role_opp", "damage_diff_vs_role_opp",
]
# Team-level aggregates (shape [M] each)
TEAM_STATS = ["team_kills", "team_golds", "team_cs"]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()
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

    team_a_pid, team_a_role = [], []
    team_b_pid, team_b_role = [], []
    winners = []
    player_stats_a = {s: [] for s in PLAYER_STATS}
    player_stats_b = {s: [] for s in PLAYER_STATS}
    team_stats_a = {s: [] for s in TEAM_STATS}
    team_stats_b = {s: [] for s in TEAM_STATS}

    for _, group in df.groupby("game_block_id", sort=True):
        # Sort by role so element-wise diff stats align across teams
        w = group[group["Result"] == "Victory"].sort_values("role_idx")
        l = group[group["Result"] == "Defeat"].sort_values("role_idx")
        if len(w) != 5 or len(l) != 5:
            continue

        team_a_pid.append(w["pid_idx"].to_numpy(dtype=np.int64))
        team_a_role.append(w["role_idx"].to_numpy(dtype=np.int64))
        team_b_pid.append(l["pid_idx"].to_numpy(dtype=np.int64))
        team_b_role.append(l["role_idx"].to_numpy(dtype=np.int64))
        winners.append(1.0)

        for s in PLAYER_STATS:
            player_stats_a[s].append(pd.to_numeric(w[s], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
            player_stats_b[s].append(pd.to_numeric(l[s], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
        for s in TEAM_STATS:
            team_stats_a[s].append(float(pd.to_numeric(w.iloc[0][s], errors="coerce") or 0))
            team_stats_b[s].append(float(pd.to_numeric(l.iloc[0][s], errors="coerce") or 0))

    batch = {
        "team_a_pid": torch.tensor(np.stack(team_a_pid), dtype=torch.long),
        "team_a_role": torch.tensor(np.stack(team_a_role), dtype=torch.long),
        "team_b_pid": torch.tensor(np.stack(team_b_pid), dtype=torch.long),
        "team_b_role": torch.tensor(np.stack(team_b_role), dtype=torch.long),
        "winner": torch.tensor(winners, dtype=torch.float32),
    }
    for s in PLAYER_STATS:
        batch[f"{s}_a"] = torch.tensor(np.stack(player_stats_a[s]), dtype=torch.float32)
        batch[f"{s}_b"] = torch.tensor(np.stack(player_stats_b[s]), dtype=torch.float32)
    for s in TEAM_STATS:
        batch[f"{s}_a"] = torch.tensor(team_stats_a[s], dtype=torch.float32)
        batch[f"{s}_b"] = torch.tensor(team_stats_b[s], dtype=torch.float32)

    print(f"Loaded {batch['winner'].shape[0]} matches, {n_players} players")
    return batch, n_players, idx_to_name, primary_role


def _sample_team_stats(p, role_idx, prefix, opp_p, batch):
    """Sample all observed stats for one team inside a pyro.plate("matches") context.

    p, opp_p: [M, 5] per-game performances
    role_idx: [M, 5] role indices (sorted, so position i matches same-role opponent)
    """
    def mean(stat):
        cfg = STAT_CONFIG[stat]
        return cfg["alpha"] * p + cfg["gamma"][role_idx]

    v_cs = pyro.sample(
        f"cs_{prefix}",
        dist.Normal(mean("cs"), STAT_CONFIG["cs"]["tau"]).to_event(1),
        obs=batch[f"cs_{prefix}"],
    )
    v_kills = pyro.sample(
        f"kills_{prefix}",
        dist.Normal(mean("kills"), STAT_CONFIG["kills"]["tau"]).to_event(1),
        obs=batch[f"kills_{prefix}"],
    )
    v_deaths = pyro.sample(
        f"deaths_{prefix}",
        dist.Normal(mean("deaths"), STAT_CONFIG["deaths"]["tau"]).to_event(1),
        obs=batch[f"deaths_{prefix}"],
    )
    v_assists = pyro.sample(
        f"assists_{prefix}",
        dist.Normal(mean("assists"), STAT_CONFIG["assists"]["tau"]).to_event(1),
        obs=batch[f"assists_{prefix}"],
    )
    v_shutdown = pyro.sample(
        f"shutdown_bounty_collected_{prefix}",
        dist.Normal(mean("shutdown_bounty_collected"), STAT_CONFIG["shutdown_bounty_collected"]["tau"]).to_event(1),
        obs=batch[f"shutdown_bounty_collected_{prefix}"],
    )
    pyro.sample(
        f"total_damage_to_champion_{prefix}",
        dist.Normal(mean("total_damage_to_champion"), STAT_CONFIG["total_damage_to_champion"]["tau"]).to_event(1),
        obs=batch[f"total_damage_to_champion_{prefix}"],
    )

    # Golds: caused by cs, kills, assists, shutdown bounty
    cfg_g = STAT_CONFIG["golds"]
    v_golds = pyro.sample(
        f"golds_{prefix}",
        dist.Normal(
            cfg_g["alpha"] * p + cfg_g["gamma"][role_idx]
            + 19.0 * v_cs + 300.0 * v_kills + 150.0 * v_assists + 1.0 * v_shutdown,
            cfg_g["tau"],
        ).to_event(1),
        obs=batch[f"golds_{prefix}"],
    )

    # Time dead: caused by deaths
    cfg_td = STAT_CONFIG["total_time_spent_dead"]
    pyro.sample(
        f"total_time_spent_dead_{prefix}",
        dist.Normal(
            cfg_td["alpha"] * p + cfg_td["gamma"][role_idx] + 35.0 * v_deaths,
            cfg_td["tau"],
        ).to_event(1),
        obs=batch[f"total_time_spent_dead_{prefix}"],
    )

    # Opponent diff stats (element-wise: both teams sorted by role, so position i = same role)
    pyro.sample(
        f"golds_diff_vs_role_opp_{prefix}",
        dist.Normal(
            STAT_CONFIG["golds_diff_vs_role_opp"]["alpha"] * (p - opp_p),
            STAT_CONFIG["golds_diff_vs_role_opp"]["tau"],
        ).to_event(1),
        obs=batch[f"golds_diff_vs_role_opp_{prefix}"],
    )
    pyro.sample(
        f"damage_diff_vs_role_opp_{prefix}",
        dist.Normal(
            STAT_CONFIG["damage_diff_vs_role_opp"]["alpha"] * (p - opp_p),
            STAT_CONFIG["damage_diff_vs_role_opp"]["tau"],
        ).to_event(1),
        obs=batch[f"damage_diff_vs_role_opp_{prefix}"],
    )

    # Team aggregate constraints (sum over the 5-player dim)
    pyro.sample(
        f"team_kills_{prefix}",
        dist.Normal(v_kills.sum(-1), 1.0),
        obs=batch[f"team_kills_{prefix}"],
    )
    pyro.sample(
        f"team_golds_{prefix}",
        dist.Normal(v_golds.sum(-1), 10.0),
        obs=batch[f"team_golds_{prefix}"],
    )
    pyro.sample(
        f"team_cs_{prefix}",
        dist.Normal(v_cs.sum(-1), 1.0),
        obs=batch[f"team_cs_{prefix}"],
    )


def model(batch, n_players):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    beta = 25.0 / 6
    beta_o = 1.0
    n_matches = batch["winner"].shape[0]

    s = pyro.sample(
        "s",
        dist.Normal(mu_0, sigma_0).expand([n_players, N_ROLES]).to_event(2),
    )
    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]  # [M, 5]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]  # [M, 5]

    with pyro.plate("matches", n_matches):
        p_a = pyro.sample("p_a", dist.Normal(skill_a, beta).to_event(1))
        p_b = pyro.sample("p_b", dist.Normal(skill_b, beta).to_event(1))

        t_a = p_a.sum(-1)
        t_b = p_b.sum(-1)
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample("y", dist.Bernoulli(win_prob), obs=batch["winner"])

        _sample_team_stats(p_a, batch["team_a_role"], "a", p_b, batch)
        _sample_team_stats(p_b, batch["team_b_role"], "b", p_a, batch)


def guide(batch, n_players):
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


def train(batch, n_players, n_steps=1500, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    with trange(n_steps, desc="Training", unit="step") as bar:
        for _ in bar:
            loss = svi.step(batch, n_players)
            losses.append(loss)
            bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses

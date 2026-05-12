"""
LCK Season 15 TrueSkill model — structured stat relationships.

Factor graph (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)
    Layer 2 — per-game performance:     p_i     ~ N(s_{i,r_i}, beta^2)
    Layer 3 — team sum:                 t_A = sum(p_i for i in A)
    Layer 4 — match outcome:            y ~ Bernoulli(Phi((t_A - t_B) / sqrt(2)*beta_o))
    Layer 5 — per-player stats:         structured with causal edges between stats
              (golds depends on cs/kills/assists, time_dead depends on deaths, etc.)

Inference: SVI with mean-field Normal guide.
"""

import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models.config import ROLE_MAP, ROLES, N_ROLES, STAT_CONFIG as BASE_STAT_CONFIG

# Superset of INDIVIDUAL_STATS — includes opponent-diff stats sampled in Layer 5
INDIVIDUAL_STATS = [
    "level", "kills", "deaths", "assists", "cs", "golds", "vision_score",
    "solo_kills", "double_kills", "triple_kills", "quadra_kills", "penta_kills",
    "gd_at_15", "csd_at_15", "xpd_at_15", "objectives_stolen",
    "damage_dealt_to_buildings", "total_heal", "total_heals_on_teammates",
    "damage_self_mitigated", "total_damage_shielded_on_teammates",
    "total_time_cc_dealt", "total_damage_taken", "total_time_spent_dead",
    "shutdown_bounty_collected", "shutdown_bounty_lost", "total_damage_to_champion",
    "golds_diff_vs_role_opp",
    "damage_diff_vs_role_opp",
]

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


def load_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding="utf-8-sig", sep=None, engine="python")
    df.columns = df.columns.str.strip()
    df["role_idx"] = df["role"].map(ROLE_MAP)

    unique_pids = sorted(df["player_id"].unique())
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    df["pid_idx"] = df["player_id"].map(pid_to_idx)
    n_players = len(unique_pids)

    idx_to_name = df[["pid_idx", "player_name"]].drop_duplicates().set_index("pid_idx")["player_name"].to_dict()
    primary_role = df.groupby("pid_idx")["role_idx"].agg(lambda x: x.mode().iloc[0]).to_dict()

    matches = []
    for gid, group in df.groupby("game_block_id"):
        winners = group[group["Result"] == "Victory"]
        losers = group[group["Result"] == "Defeat"]
        if len(winners) != 5 or len(losers) != 5:
            continue

        match = {
            "team_a": list(zip(winners["pid_idx"].tolist(), winners["role_idx"].astype(int).tolist())),
            "team_b": list(zip(losers["pid_idx"].tolist(), losers["role_idx"].astype(int).tolist())),
            "winner": 1,
            "team_kills_a": torch.tensor(winners["team_kills"].iloc[0], dtype=torch.float32),
            "team_kills_b": torch.tensor(losers["team_kills"].iloc[0], dtype=torch.float32),
            "team_golds_a": torch.tensor(winners["team_golds"].iloc[0], dtype=torch.float32),
            "team_golds_b": torch.tensor(losers["team_golds"].iloc[0], dtype=torch.float32),
            "team_cs_a": torch.tensor(winners["team_cs"].iloc[0], dtype=torch.float32),
            "team_cs_b": torch.tensor(losers["team_cs"].iloc[0], dtype=torch.float32),
        }
        for stat in INDIVIDUAL_STATS:
            match[f"{stat}_a"] = torch.tensor(winners[stat].fillna(0).values, dtype=torch.float32)
            match[f"{stat}_b"] = torch.tensor(losers[stat].fillna(0).values, dtype=torch.float32)
        matches.append(match)

    return matches, n_players, idx_to_name, primary_role


def model(matches, n_players):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    beta = 25.0 / 6
    beta_o = 1.0

    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            s = pyro.sample("s", dist.Normal(mu_0, sigma_0))
    s = s.T  # -> (n_players, N_ROLES)

    for m_idx, match in enumerate(matches):
        team_a = match["team_a"]
        team_b = match["team_b"]

        p_a = torch.stack([
            pyro.sample(f"pa_{m_idx}_{i}", dist.Normal(s[pid, r], beta))
            for i, (pid, r) in enumerate(team_a)
        ])
        p_b = torch.stack([
            pyro.sample(f"pb_{m_idx}_{i}", dist.Normal(s[pid, r], beta))
            for i, (pid, r) in enumerate(team_b)
        ])

        t_a = p_a.sum()
        t_b = p_b.sum()
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample(
            f"y_{m_idx}",
            dist.Bernoulli(win_prob),
            obs=torch.tensor(float(match["winner"])),
        )

        roles_a = torch.tensor([r for _, r in team_a])
        roles_b = torch.tensor([r for _, r in team_b])

        def sample_structured_stats(perf, roles, prefix, m_idx, match, opp_perf=None):
            v_cs = pyro.sample(
                f"cs_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["cs"]["alpha"] * perf + STAT_CONFIG["cs"]["gamma"][roles],
                    STAT_CONFIG["cs"]["tau"],
                ).to_event(1),
                obs=match[f"cs_{prefix}"],
            )
            v_kills = pyro.sample(
                f"kills_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["kills"]["alpha"] * perf + STAT_CONFIG["kills"]["gamma"][roles],
                    STAT_CONFIG["kills"]["tau"],
                ).to_event(1),
                obs=match[f"kills_{prefix}"],
            )
            v_deaths = pyro.sample(
                f"deaths_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["deaths"]["alpha"] * perf + STAT_CONFIG["deaths"]["gamma"][roles],
                    STAT_CONFIG["deaths"]["tau"],
                ).to_event(1),
                obs=match[f"deaths_{prefix}"],
            )
            v_assists = pyro.sample(
                f"assists_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["assists"]["alpha"] * perf + STAT_CONFIG["assists"]["gamma"][roles],
                    STAT_CONFIG["assists"]["tau"],
                ).to_event(1),
                obs=match[f"assists_{prefix}"],
            )
            v_shutdown = pyro.sample(
                f"shutdown_bounty_collected_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["shutdown_bounty_collected"]["alpha"] * perf
                    + STAT_CONFIG["shutdown_bounty_collected"]["gamma"][roles],
                    STAT_CONFIG["shutdown_bounty_collected"]["tau"],
                ).to_event(1),
                obs=match[f"shutdown_bounty_collected_{prefix}"],
            )
            v_dmg = pyro.sample(
                f"total_damage_to_champion_{prefix}_{m_idx}",
                dist.Normal(
                    STAT_CONFIG["total_damage_to_champion"]["alpha"] * perf
                    + STAT_CONFIG["total_damage_to_champion"]["gamma"][roles],
                    STAT_CONFIG["total_damage_to_champion"]["tau"],
                ).to_event(1),
                obs=match[f"total_damage_to_champion_{prefix}"],
            )

            # Golds depends on CS, kills, assists, and shutdowns
            gold_mean = (
                STAT_CONFIG["golds"]["alpha"] * perf
                + STAT_CONFIG["golds"]["gamma"][roles]
                + 19.0 * v_cs
                + 300.0 * v_kills
                + 150.0 * v_assists
                + 1.0 * v_shutdown
            )
            v_golds = pyro.sample(
                f"golds_{prefix}_{m_idx}",
                dist.Normal(gold_mean, STAT_CONFIG["golds"]["tau"]).to_event(1),
                obs=match[f"golds_{prefix}"],
            )

            # Time dead depends on deaths
            dead_timer_mean = (
                STAT_CONFIG["total_time_spent_dead"]["alpha"] * perf
                + STAT_CONFIG["total_time_spent_dead"]["gamma"][roles]
                + 35.0 * v_deaths
            )
            pyro.sample(
                f"total_time_spent_dead_{prefix}_{m_idx}",
                dist.Normal(dead_timer_mean, STAT_CONFIG["total_time_spent_dead"]["tau"]).to_event(1),
                obs=match[f"total_time_spent_dead_{prefix}"],
            )

            if opp_perf is not None:
                pyro.sample(
                    f"golds_diff_vs_role_opp_{prefix}_{m_idx}",
                    dist.Normal(
                        STAT_CONFIG["golds_diff_vs_role_opp"]["alpha"] * (perf - opp_perf),
                        STAT_CONFIG["golds_diff_vs_role_opp"]["tau"],
                    ).to_event(1),
                    obs=match[f"golds_diff_vs_role_opp_{prefix}"],
                )
                pyro.sample(
                    f"damage_diff_vs_role_opp_{prefix}_{m_idx}",
                    dist.Normal(
                        STAT_CONFIG["damage_diff_vs_role_opp"]["alpha"] * (perf - opp_perf),
                        STAT_CONFIG["damage_diff_vs_role_opp"]["tau"],
                    ).to_event(1),
                    obs=match[f"damage_diff_vs_role_opp_{prefix}"],
                )

            pyro.sample(
                f"team_kills_{prefix}_{m_idx}",
                dist.Normal(v_kills.sum(), 1.0),
                obs=match[f"team_kills_{prefix}"],
            )
            pyro.sample(
                f"team_golds_{prefix}_{m_idx}",
                dist.Normal(v_golds.sum(), 10.0),
                obs=match[f"team_golds_{prefix}"],
            )
            pyro.sample(
                f"team_cs_{prefix}_{m_idx}",
                dist.Normal(v_cs.sum(), 1.0),
                obs=match[f"team_cs_{prefix}"],
            )

        sample_structured_stats(p_a, roles_a, "a", m_idx, match, opp_perf=p_b)
        sample_structured_stats(p_b, roles_b, "b", m_idx, match, opp_perf=p_a)


def guide(matches, n_players):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3

    s_loc = pyro.param("s_loc", mu_0 * torch.ones(N_ROLES, n_players))
    s_scale = pyro.param(
        "s_scale",
        sigma_0 * torch.ones(N_ROLES, n_players),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            pyro.sample("s", dist.Normal(s_loc, s_scale))

    for m_idx, match in enumerate(matches):
        for i in range(5):
            for prefix in ("pa", "pb"):
                name = f"{prefix}_{m_idx}_{i}"
                loc = pyro.param(f"{name}_loc", torch.tensor(25.0))
                scl = pyro.param(
                    f"{name}_scale",
                    torch.tensor(4.0),
                    constraint=dist.constraints.positive,
                )
                pyro.sample(name, dist.Normal(loc, scl))


def train(matches, n_players, n_steps=1500, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    with trange(n_steps, desc="Training", unit="step") as bar:
        for _ in bar:
            loss = svi.step(matches, n_players)
            losses.append(loss)
            bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses


def print_rankings(n_players, idx_to_name, primary_role):
    params = pyro.get_param_store()
    mu = params["s_loc"].detach().T.numpy()
    sigma = params["s_scale"].detach().T.numpy()
    conservative = mu - 3 * sigma

    print("\n" + "=" * 65)
    print("SKILL RANKINGS BY ROLE  (conservative = mu - 3*sigma)")
    print("=" * 65)

    for r, role in enumerate(ROLES):
        role_players = [p for p, pr in primary_role.items() if pr == r]
        ranking = sorted(role_players, key=lambda p: conservative[p, r], reverse=True)
        print(f"\n  {role.upper()}")
        print(f"  {'─' * 55}")
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
        all_ratings.append((
            idx_to_name[pid_idx],
            ROLES[r],
            mu[pid_idx, r],
            sigma[pid_idx, r],
            conservative[pid_idx, r],
        ))
    all_ratings.sort(key=lambda x: x[4], reverse=True)

    for rank, (name, role, m, s, c) in enumerate(all_ratings, 1):
        print(
            f"  {rank:2d}. {name:12s} ({role:3s})  "
            f"mu={m:6.2f}  sigma={s:5.2f}  rating={c:6.2f}"
        )

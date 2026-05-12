import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from model import INDIVIDUAL_STATS, N_ROLES, ROLE_MAP, ROLES, STAT_CONFIG

TEAM_CONTEXT_STATS = [
    "team_kills",
    "team_deaths",
    "team_assists",
    "team_cs",
    "team_golds",
    "team_vision_score",
    "team_total_damage_to_champion",
]
CONTEXT_STATS = ["duration_minutes", *TEAM_CONTEXT_STATS]


def parse_duration_minutes(value) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, str) and ":" in value:
        minutes, seconds = value.split(":", maxsplit=1)
        return float(minutes) + float(seconds) / 60.0
    return float(value)


def load_data(csv_path: str):
    """
    Load the LCK MODEL-READY CSV into batched tensors.

    Returns:
        batch: dict of tensors with match-major shapes
        n_players: total distinct players
        idx_to_name: pid_idx -> player name
        primary_role: pid_idx -> main role index
    """
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
    team_context_a = []
    team_context_b = []
    stats_a_by_name = {stat: [] for stat in INDIVIDUAL_STATS}
    stats_b_by_name = {stat: [] for stat in INDIVIDUAL_STATS}

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
        team_context_a.append(
            np.concatenate(
                [
                    np.array([duration_minutes], dtype=np.float32),
                    pd.to_numeric(winners_df.iloc[0][TEAM_CONTEXT_STATS], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=np.float32),
                ]
            )
        )
        team_context_b.append(
            np.concatenate(
                [
                    np.array([duration_minutes], dtype=np.float32),
                    pd.to_numeric(losers_df.iloc[0][TEAM_CONTEXT_STATS], errors="coerce")
                    .fillna(0)
                    .to_numpy(dtype=np.float32),
                ]
            )
        )
        for stat in INDIVIDUAL_STATS:
            stats_a_by_name[stat].append(
                pd.to_numeric(winners_df[stat], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
            )
            stats_b_by_name[stat].append(
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
    team_context_a = np.stack(team_context_a, axis=0)
    team_context_b = np.stack(team_context_b, axis=0)
    team_context_all = np.concatenate([team_context_a, team_context_b], axis=0)
    context_mean = team_context_all.mean(axis=0)
    context_std = team_context_all.std(axis=0)
    context_std = np.where(context_std == 0, 1.0, context_std)

    batch = {
        "team_a_pid": torch.tensor(np.stack(team_a_pid, axis=0), dtype=torch.long),
        "team_a_role": torch.tensor(np.stack(team_a_role, axis=0), dtype=torch.long),
        "team_b_pid": torch.tensor(np.stack(team_b_pid, axis=0), dtype=torch.long),
        "team_b_role": torch.tensor(np.stack(team_b_role, axis=0), dtype=torch.long),
        "winner": torch.tensor(winners, dtype=torch.float32),
        "stats_a": torch.tensor(stats_a, dtype=torch.float32),
        "stats_b": torch.tensor(stats_b, dtype=torch.float32),
        "team_context_a": torch.tensor((team_context_a - context_mean) / context_std, dtype=torch.float32),
        "team_context_b": torch.tensor((team_context_b - context_mean) / context_std, dtype=torch.float32),
        "team_context_mean": torch.tensor(context_mean, dtype=torch.float32),
        "team_context_std": torch.tensor(context_std, dtype=torch.float32),
    }

    print(f"Loaded {batch['winner'].shape[0]} matches, {n_players} players")
    return batch, n_players, idx_to_name, primary_role


def build_stat_tensors():
    alpha = torch.tensor([STAT_CONFIG[stat]["alpha"] for stat in INDIVIDUAL_STATS], dtype=torch.float32)
    tau = torch.tensor([STAT_CONFIG[stat]["tau"] for stat in INDIVIDUAL_STATS], dtype=torch.float32)
    gamma = torch.stack([STAT_CONFIG[stat]["gamma"].float() for stat in INDIVIDUAL_STATS], dim=0)
    return alpha, tau, gamma


ALPHA_VEC, TAU_VEC, GAMMA_MAT = build_stat_tensors()


def role_intercepts(gamma_mat: torch.Tensor, role_idx: torch.Tensor) -> torch.Tensor:
    # gamma_mat: [S, R], role_idx: [M, 5] -> [M, 5, S]
    return gamma_mat[:, role_idx].permute(1, 2, 0)


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

    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]

    # Effects are in units of each stat's observation noise. This keeps the
    # prior scale comparable for kills, gold, damage, and the other stats.
    team_context_effect_z = pyro.sample(
        "team_context_effect_z",
        dist.Normal(
            torch.zeros(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            0.25 * torch.ones(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        ).to_event(2),
    )
    team_context_effect = team_context_effect_z * TAU_VEC.view(1, -1)

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
        context_shift_a = batch["team_context_a"] @ team_context_effect
        context_shift_b = batch["team_context_b"] @ team_context_effect
        mean_a = mean_a + context_shift_a.unsqueeze(1)
        mean_b = mean_b + context_shift_b.unsqueeze(1)

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

    context_loc = pyro.param(
        "team_context_effect_z_loc",
        torch.zeros(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    context_scale = pyro.param(
        "team_context_effect_z_scale",
        0.1 * torch.ones(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("team_context_effect_z", dist.Normal(context_loc, context_scale).to_event(2))

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


def print_rankings_fast(n_players, idx_to_name, primary_role):
    params = pyro.get_param_store()
    mu = params["s_loc"].detach().cpu().numpy()          # (n_players, 5)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-steps", type=int, default=1500)
    parser.add_argument("--load", metavar="FILE", help="Load saved params and print rankings without retraining")
    parser.add_argument("--csv-path", default="data/lck_s15_games_MODEL-READY.csv")
    args = parser.parse_args()

    batch, n_players, idx_to_name, primary_role = load_data(args.csv_path)

    if args.load:
        pyro.clear_param_store()
        pyro.get_param_store().load(args.load)
        print(f"Loaded params from {args.load}")
    else:
        start = time.perf_counter()
        losses = train(
            batch,
            n_players,
            n_steps=args.n_steps,
            lr=0.01,
        )
        elapsed = time.perf_counter() - start

        pyro.get_param_store().save("params_fast.pt")
        print("Saved params to params_fast.pt")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("ELBO loss")
        plt.title("SVI convergence (fast)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("elbo_fast.png")
        print("Saved loss curve to elbo_fast.png")
        print(f"Training wall time: {elapsed:.2f}s")

    print_rankings_fast(n_players, idx_to_name, primary_role)

"""
LCK Season 15 TrueSkill model — Pyro implementation.

Factor graph (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)
    Layer 2 — per-game performance:     p_i     ~ N(s_{i,r_i}, beta^2)
    Layer 3 — team sum:                 t_A = sum(p_i for i in A)
    Layer 4 — match outcome:            y ~ Bernoulli(Phi((t_A - t_B) / sqrt(2)*beta_o))
    Layer 5 — per-player stats:         stat_i ~ N(alpha * p_i + gamma_{role}, tau^2)
              for stat in {level, kills, deaths, assists, cs, golds, vision_score,
                           solo_kills, double_kills, triple_kills, quadra_kills, penta_kills,
                           gd_at_15, csd_at_15, xpd_at_15, objectives_stolen,
                           damage_dealt_to_buildings, total_heal, total_heals_on_teammates,
                           damage_self_mitigated, total_damage_shielded_on_teammates,
                           total_time_cc_dealt, total_damage_taken, total_time_spent_dead,
                           shutdown_bounty_collected, shutdown_bounty_lost, total_damage_to_champion}

Inference: SVI with mean-field Normal guide.

Usage:
    python lck_trueskill_model.py
"""

import argparse
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import matplotlib.pyplot as plt
from tqdm import trange


# ─────────────────────────────────────────────────────────────
# Data loading and preprocessing
# ─────────────────────────────────────────────────────────────

ROLE_MAP = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
ROLES = ["top", "jng", "mid", "adc", "sup"]
N_ROLES = 5

INDIVIDUAL_STATS = [
    "level", "kills", "deaths", "assists", "cs", "golds", "vision_score",
    "solo_kills", "double_kills", "triple_kills", "quadra_kills", "penta_kills",
    "gd_at_15", "csd_at_15", "xpd_at_15", "objectives_stolen",
    "damage_dealt_to_buildings", "total_heal", "total_heals_on_teammates",
    "damage_self_mitigated", "total_damage_shielded_on_teammates",
    "total_time_cc_dealt", "total_damage_taken", "total_time_spent_dead",
    "shutdown_bounty_collected", "shutdown_bounty_lost", "total_damage_to_champion",
]


def load_data(csv_path: str):
    """
    Load the LCK MODEL-READY CSV and return (matches, n_players, idx_to_name).

    matches:     list of dicts ready for the Pyro model
    n_players:   int, total distinct players
    idx_to_name: dict mapping contiguous player index -> player name
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    df["role_idx"] = df["role"].map(ROLE_MAP)

    # --- re-index player IDs to 0..N-1 ---
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

    # --- build a role lookup: pid_idx -> primary role_idx ---
    primary_role = (
        df.groupby("pid_idx")["role_idx"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    # --- build match dicts ---
    matches = []
    for gid, group in df.groupby("game_block_id"):
        winners = group[group["Result"] == "Victory"]
        losers = group[group["Result"] == "Defeat"]
        if len(winners) != 5 or len(losers) != 5:
            continue

        match = {
            "team_a": list(zip(winners["pid_idx"].tolist(),
                               winners["role_idx"].astype(int).tolist())),
            "team_b": list(zip(losers["pid_idx"].tolist(),
                               losers["role_idx"].astype(int).tolist())),
            "winner": 1,  # team_a = winners by construction
        }
        for stat in INDIVIDUAL_STATS:
            match[f"{stat}_a"] = torch.tensor(winners[stat].fillna(0).values, dtype=torch.float32)
            match[f"{stat}_b"] = torch.tensor(losers[stat].fillna(0).values, dtype=torch.float32)
        matches.append(match)

    print(f"Loaded {len(matches)} matches, {n_players} players")
    return matches, n_players, idx_to_name, primary_role


# ─────────────────────────────────────────────────────────────
# Pyro model and guide
# ─────────────────────────────────────────────────────────────

# Per-stat observation parameters: slope (alpha), role intercepts (gamma), noise std (tau).
# gamma order: [TOP, JUNGLE, MID, ADC, SUPPORT]
# gamma = role_mean - alpha * 25, so that E[stat | p=25] matches the observed role average.
# alpha scaled so a 10-unit performance swing causes ~0.1-0.3 std change in the stat.
# tau set to approximately the observed per-role std.
STAT_CONFIG = {
    "level": {
        "alpha": 0.02,
        "gamma": torch.tensor([15.9, 14.7, 16.2, 15.3, 11.9]),
        "tau": 1.5,
    },
    "kills": {
        "alpha": 0.05,
        "gamma": torch.tensor([1.6, 1.8, 2.1, 3.4, -0.6]),
        "tau": 2.5,
    },
    "deaths": {
        "alpha": -0.05,
        "gamma": torch.tensor([4.3, 4.3, 3.8, 3.6, 4.9]),
        "tau": 2.0,
    },
    "assists": {
        "alpha": 0.05,
        "gamma": torch.tensor([4.2, 6.3, 5.2, 4.5, 8.8]),
        "tau": 4.0,
    },
    "cs": {
        "alpha": 1.0,
        "gamma": torch.tensor([227.4, 183.0, 256.8, 280.3, 8.0]),
        "tau": 50.0,
    },
    "golds": {
        "alpha": 30.0,
        "gamma": torch.tensor([11628.0, 10947.0, 12453.0, 13569.0, 7224.0]),
        "tau": 2500.0,
    },
    "vision_score": {
        "alpha": 0.3,
        "gamma": torch.tensor([26.4, 44.3, 29.2, 31.1, 107.4]),
        "tau": 15.0,
    },
    "solo_kills": {
        "alpha": 0.01,
        "gamma": torch.tensor([1.05, 0.85, 0.85, 0.95, 0.85]),
        "tau": 0.5,
    },
    "double_kills": {
        "alpha": 0.005,
        "gamma": torch.tensor([0.175, 0.175, 0.175, 0.375, -0.125]),
        "tau": 0.5,
    },
    "triple_kills": {
        "alpha": 0.002,
        "gamma": torch.tensor([-0.05, -0.05, 0.05, 0.15, -0.05]),
        "tau": 0.25,
    },
    "quadra_kills": {
        "alpha": 0.001,
        "gamma": torch.tensor([-0.025, -0.025, -0.025, -0.025, -0.025]),
        "tau": 0.15,
    },
    "penta_kills": {
        "alpha": 0.0005,
        "gamma": torch.tensor([-0.0125, -0.0125, -0.0125, -0.0125, -0.0125]),
        "tau": 0.05,
    },
    "gd_at_15": {
        "alpha": 10.0,
        "gamma": torch.tensor([-250.0, -250.0, -250.0, -250.0, -250.0]),
        "tau": 700.0,
    },
    "csd_at_15": {
        "alpha": 0.2,
        "gamma": torch.tensor([-5.0, -5.0, -5.0, -5.0, -5.0]),
        "tau": 15.0,
    },
    "xpd_at_15": {
        "alpha": 10.0,
        "gamma": torch.tensor([-250.0, -250.0, -250.0, -250.0, -250.0]),
        "tau": 650.0,
    },
    "objectives_stolen": {
        "alpha": 0.005,
        "gamma": torch.tensor([-0.125, -0.025, -0.125, -0.125, -0.125]),
        "tau": 0.3,
    },
    "damage_dealt_to_buildings": {
        "alpha": 30.0,
        "gamma": torch.tensor([3367.0, 1099.0, 2934.0, 4005.0, 120.0]),
        "tau": 3000.0,
    },
    "total_heal": {
        "alpha": 20.0,
        "gamma": torch.tensor([5963.0, 21828.0, 4251.0, 4944.0, 4699.0]),
        "tau": 5000.0,
    },
    "total_heals_on_teammates": {
        "alpha": 5.0,
        "gamma": torch.tensor([-107.0, 144.0, -83.0, -12.0, 1935.0]),
        "tau": 2000.0,
    },
    "damage_self_mitigated": {
        "alpha": 100.0,
        "gamma": torch.tensor([29129.0, 32430.0, 14275.0, 8382.0, 15368.0]),
        "tau": 12000.0,
    },
    "total_damage_shielded_on_teammates": {
        "alpha": 5.0,
        "gamma": torch.tensor([-39.0, -45.0, 266.0, -57.0, 1189.0]),
        "tau": 1200.0,
    },
    "total_time_cc_dealt": {
        "alpha": 2.0,
        "gamma": torch.tensor([332.0, 388.0, 281.0, 147.0, 81.0]),
        "tau": 300.0,
    },
    "total_damage_taken": {
        "alpha": 50.0,
        "gamma": torch.tensor([26094.0, 37897.0, 19227.0, 15053.0, 16378.0]),
        "tau": 8000.0,
    },
    "total_time_spent_dead": {
        "alpha": -1.0,
        "gamma": torch.tensor([122.4, 119.0, 108.8, 94.2, 113.2]),
        "tau": 65.0,
    },
    "shutdown_bounty_collected": {
        "alpha": 2.0,
        "gamma": torch.tensor([83.0, 75.0, 96.0, 126.0, -23.0]),
        "tau": 250.0,
    },
    "shutdown_bounty_lost": {
        "alpha": -2.0,
        "gamma": torch.tensor([181.0, 191.0, 201.0, 211.0, 74.0]),
        "tau": 200.0,
    },
    "total_damage_to_champion": {
        "alpha": 100.0,
        "gamma": torch.tensor([16978.0, 11506.0, 19187.0, 21042.0, 4330.0]),
        "tau": 7000.0,
    },
}


def model(matches, n_players):
    # --- hyperparameters ---
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    beta = 25.0 / 6  # performance noise
    beta_o = 1.0      # outcome-link scale

    # Layer 1: per-role player skill priors
    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            s = pyro.sample("s", dist.Normal(mu_0, sigma_0))
    s = s.T  # -> (n_players, N_ROLES)

    for m_idx, match in enumerate(matches):
        team_a = match["team_a"]
        team_b = match["team_b"]

        # Layer 2: per-game performance
        p_a = torch.stack([
            pyro.sample(f"pa_{m_idx}_{i}", dist.Normal(s[pid, r], beta))
            for i, (pid, r) in enumerate(team_a)
        ])
        p_b = torch.stack([
            pyro.sample(f"pb_{m_idx}_{i}", dist.Normal(s[pid, r], beta))
            for i, (pid, r) in enumerate(team_b)
        ])

        # Layer 3: team sums (deterministic)
        t_a = p_a.sum()
        t_b = p_b.sum()

        # Layer 4: outcome with probit link
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample(
            f"y_{m_idx}",
            dist.Bernoulli(win_prob),
            obs=torch.tensor(float(match["winner"])),
        )

        # Layer 5: per-player stat observations (27 raw stats)
        roles_a = torch.tensor([r for _, r in team_a])
        roles_b = torch.tensor([r for _, r in team_b])

        for stat_name, cfg in STAT_CONFIG.items():
            mean_a = cfg["alpha"] * p_a + cfg["gamma"][roles_a]
            mean_b = cfg["alpha"] * p_b + cfg["gamma"][roles_b]

            pyro.sample(
                f"{stat_name}_a_{m_idx}",
                dist.Normal(mean_a, cfg["tau"]).to_event(1),
                obs=match[f"{stat_name}_a"],
            )
            pyro.sample(
                f"{stat_name}_b_{m_idx}",
                dist.Normal(mean_b, cfg["tau"]).to_event(1),
                obs=match[f"{stat_name}_b"],
            )


def guide(matches, n_players):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3

    # Variational params for player-role skills
    s_loc = pyro.param("s_loc", mu_0 * torch.ones(N_ROLES, n_players))
    s_scale = pyro.param(
        "s_scale",
        sigma_0 * torch.ones(N_ROLES, n_players),
        constraint=dist.constraints.positive,
    )
    with pyro.plate("players", n_players):
        with pyro.plate("roles", N_ROLES):
            pyro.sample("s", dist.Normal(s_loc, s_scale))

    # Variational params for per-match performances
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


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train(matches, n_players, n_steps=1500, lr=0.01):
    pyro.clear_param_store()
    svi = SVI(model, guide, Adam({"lr": lr}), loss=Trace_ELBO())
    losses = []
    with trange(n_steps, desc="Training", unit="step") as bar:
        for step in bar:
            loss = svi.step(matches, n_players)
            losses.append(loss)
            bar.set_postfix(elbo=f"{loss:,.0f}")
    return losses


# ─────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────

def print_rankings(n_players, idx_to_name, primary_role):
    params = pyro.get_param_store()
    mu = params["s_loc"].detach().T.numpy()       # (n_players, 5)
    sigma = params["s_scale"].detach().T.numpy()
    conservative = mu - 3 * sigma

    # Per-role rankings
    print("\n" + "=" * 65)
    print("SKILL RANKINGS BY ROLE  (conservative = mu - 3*sigma)")
    print("=" * 65)

    for r, role in enumerate(ROLES):
        role_players = [p for p, pr in primary_role.items() if pr == r]
        ranking = sorted(role_players, key=lambda p: conservative[p, r], reverse=True)
        print(f"\n  {role.upper()}")
        print(f"  {'─' * 55}")
        for rank, p in enumerate(ranking, 1):
            print(f"  {rank:2d}. {idx_to_name[p]:12s}  "
                  f"mu={mu[p,r]:6.2f}  sigma={sigma[p,r]:5.2f}  "
                  f"rating={conservative[p,r]:6.2f}")

    # Overall
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
        print(f"  {rank:2d}. {name:12s} ({role:3s})  "
              f"mu={m:6.2f}  sigma={s:5.2f}  rating={c:6.2f}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n-steps", type=int, default=1500)
    parser.add_argument("--load", metavar="FILE", help="Load saved params and print rankings without retraining")
    args = parser.parse_args()

    CSV_PATH = "data/lck_s15_games_MODEL-READY.csv"

    matches, n_players, idx_to_name, primary_role = load_data(CSV_PATH)

    if args.load:
        pyro.clear_param_store()
        pyro.get_param_store().load(args.load)
        print(f"Loaded params from {args.load}")
    else:
        losses = train(matches, n_players, n_steps=args.n_steps, lr=0.01)

        pyro.get_param_store().save("params.pt")
        print("Saved params to params.pt")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("ELBO loss")
        plt.title("SVI convergence")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("elbo.png")
        print("Saved loss curve to elbo.png")

    print_rankings(n_players, idx_to_name, primary_role)

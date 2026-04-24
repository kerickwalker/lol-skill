"""
LCK Season 15 TrueSkill model — Pyro implementation.

Factor graph (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)
    Layer 2 — per-game performance:     p_i     ~ N(s_{i,r_i}, beta^2)
    Layer 3 — team sum:                 t_A = sum(p_i for i in A)
    Layer 4 — match outcome:            y ~ Bernoulli(Phi((t_A - t_B) / sqrt(2)*beta_o))
    Layer 5 — per-player stats:         stat_i ~ N(alpha * p_i + gamma_{role}, tau^2)
              for stat in {KDA ratio, CSM, DPM, KP%}

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


# ─────────────────────────────────────────────────────────────
# Data loading and preprocessing
# ─────────────────────────────────────────────────────────────

ROLE_MAP = {"TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "SUPPORT": 4}
ROLES = ["top", "jng", "mid", "adc", "sup"]
N_ROLES = 5


def parse_kda(kda_str: str) -> float:
    """Convert 'K/D/A' string to numeric ratio (K+A)/max(D,1)."""
    kda_str = str(kda_str).strip()
    if kda_str.startswith('="') and kda_str.endswith('"'):
        kda_str = kda_str[2:-1]
    k, d, a = (int(x) for x in kda_str.split("/"))
    return (k + a) / max(d, 1)


def load_data(csv_path: str):
    """
    Load the LCK CSV and return (matches, n_players, idx_to_name).

    matches:     list of dicts ready for the Pyro model
    n_players:   int, total distinct players
    idx_to_name: dict mapping contiguous player index -> player name
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # --- parse stats ---
    df["kda_ratio"] = df["KDA"].apply(parse_kda)
    df["kp"] = df["KP%"].str.rstrip("%").astype(float) / 100.0
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
            # 4 observed stats per side
            "kda_a": torch.tensor(winners["kda_ratio"].values, dtype=torch.float32),
            "kda_b": torch.tensor(losers["kda_ratio"].values, dtype=torch.float32),
            "csm_a": torch.tensor(winners["CSM"].values, dtype=torch.float32),
            "csm_b": torch.tensor(losers["CSM"].values, dtype=torch.float32),
            "dpm_a": torch.tensor(winners["DPM"].values, dtype=torch.float32),
            "dpm_b": torch.tensor(losers["DPM"].values, dtype=torch.float32),
            "kp_a":  torch.tensor(winners["kp"].values, dtype=torch.float32),
            "kp_b":  torch.tensor(losers["kp"].values, dtype=torch.float32),
        }
        matches.append(match)

    print(f"Loaded {len(matches)} matches, {n_players} players")
    return matches, n_players, idx_to_name, primary_role


# ─────────────────────────────────────────────────────────────
# Pyro model and guide
# ─────────────────────────────────────────────────────────────

# Per-stat observation parameters: slope, role intercepts, noise std.
# These are rough but reasonable priors based on LCK averages.
STAT_CONFIG = {
    "kda": {
        "alpha": 0.10,
        "gamma": torch.tensor([3.0, 3.5, 3.5, 4.0, 3.0]),
        "tau": 2.5,
    },
    "csm": {
        "alpha": 0.05,
        "gamma": torch.tensor([7.8, 6.5, 8.7, 9.5, 1.0]),
        "tau": 1.0,
    },
    "dpm": {
        "alpha": 2.0,
        "gamma": torch.tensor([500.0, 400.0, 550.0, 600.0, 150.0]),
        "tau": 150.0,
    },
    "kp": {
        "alpha": 0.005,
        "gamma": torch.tensor([0.60, 0.70, 0.65, 0.70, 0.75]),
        "tau": 0.12,
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

        # Layer 5: per-player stat observations (4 stats)
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
    for step in range(n_steps):
        loss = svi.step(matches, n_players)
        losses.append(loss)
        if step % 100 == 0:
            print(f"  step {step:5d}   ELBO loss = {loss:,.0f}")
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
    args = parser.parse_args()

    CSV_PATH = "/Users/kerickwalker/Desktop/mbml/lol-skill/lck_s15_games_blocked.csv"

    matches, n_players, idx_to_name, primary_role = load_data(CSV_PATH)
    losses = train(matches, n_players, n_steps=args.n_steps, lr=0.01)
    print_rankings(n_players, idx_to_name, primary_role)

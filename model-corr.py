"""
LCK Season 15 TrueSkill model — correlated team performance (Option A).

Differs from model-fast.py only in Layer 2: each team's 5 per-game performances
are sampled jointly from a MultivariateNormal with a fixed role-structured
covariance matrix, rather than independently.

Factor graph (per match):
    Layer 1 — per-role player skill:    s_{i,r} ~ N(mu_0, sigma_0^2)
    Layer 2 — per-game performance:     [p_top,..,p_sup] ~ MVN(skill_means, Sigma_roles)
    Layer 3 — team sum:                 t_A = sum(p_i for i in A)
    Layer 4 — match outcome:            y ~ Bernoulli(Phi((t_A - t_B) / sqrt(2)*beta_o))
    Layer 5 — per-player stats:         stat_i ~ N(alpha * p_i + gamma_{role}, tau^2)

Sigma_roles = beta^2 * ROLE_CORR, where ROLE_CORR encodes prior beliefs about
which roles' performances are correlated within a game:
    - ADC/Support:    rho=0.50  (bot lane plays as a unit)
    - Jungle/Mid:     rho=0.30  (jungler frequently enables mid)
    - Jungle/Top:     rho=0.20  (topside jungle pressure)
    - Jungle/ADC,SUP: rho=0.10
    - Top/Mid:        rho=0.10

The guide keeps mean-field (independent Normal) posteriors for performances —
a valid approximation that keeps inference tractable.
"""

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

import importlib.util
import pathlib

from model import INDIVIDUAL_STATS, N_ROLES, ROLE_MAP, ROLES, STAT_CONFIG

# model-fast.py has a hyphenated name so we load it via importlib
_spec = importlib.util.spec_from_file_location(
    "model_fast", pathlib.Path(__file__).parent / "model-fast.py"
)
_mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mf)

ALPHA_VEC         = _mf.ALPHA_VEC
CONTEXT_STATS     = _mf.CONTEXT_STATS
GAMMA_MAT         = _mf.GAMMA_MAT
TAU_VEC           = _mf.TAU_VEC
TEAM_CONTEXT_STATS = _mf.TEAM_CONTEXT_STATS
load_data          = _mf.load_data
print_rankings_fast = _mf.print_rankings_fast
role_intercepts    = _mf.role_intercepts

# ─────────────────────────────────────────────────────────────
# Role correlation structure
# ─────────────────────────────────────────────────────────────
# Order: TOP=0, JNG=1, MID=2, ADC=3, SUP=4

ROLE_CORR = torch.tensor([
    #  TOP   JNG   MID   ADC   SUP
    [1.00, 0.20, 0.10, 0.00, 0.00],  # TOP
    [0.20, 1.00, 0.30, 0.10, 0.10],  # JNG
    [0.10, 0.30, 1.00, 0.00, 0.00],  # MID
    [0.00, 0.10, 0.00, 1.00, 0.50],  # ADC
    [0.00, 0.10, 0.00, 0.50, 1.00],  # SUP
], dtype=torch.float32)

_beta = 25.0 / 6
PERF_COV = (_beta ** 2) * ROLE_CORR
PERF_SCALE_TRIL = torch.linalg.cholesky(PERF_COV)


# ─────────────────────────────────────────────────────────────
# Model and guide
# ─────────────────────────────────────────────────────────────

def model(batch, n_players):
    mu_0 = 25.0
    sigma_0 = 25.0 / 3
    beta_o = 1.0

    n_matches = batch["winner"].shape[0]

    s = pyro.sample(
        "s",
        dist.Normal(mu_0, sigma_0).expand([n_players, N_ROLES]).to_event(2),
    )

    skill_a = s[batch["team_a_pid"], batch["team_a_role"]]  # [M, 5]
    skill_b = s[batch["team_b_pid"], batch["team_b_role"]]  # [M, 5]

    team_context_effect_z = pyro.sample(
        "team_context_effect_z",
        dist.Normal(
            torch.zeros(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
            0.25 * torch.ones(len(CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        ).to_event(2),
    )
    team_context_effect = team_context_effect_z * TAU_VEC.view(1, -1)

    with pyro.plate("matches", n_matches):
        # Layer 2: correlated per-game performance via MVN
        p_a = pyro.sample(
            "p_a",
            dist.MultivariateNormal(skill_a, scale_tril=PERF_SCALE_TRIL),
        )
        p_b = pyro.sample(
            "p_b",
            dist.MultivariateNormal(skill_b, scale_tril=PERF_SCALE_TRIL),
        )

        # Layer 3 & 4: team sums and outcome
        t_a = p_a.sum(-1)
        t_b = p_b.sum(-1)
        diff = (t_a - t_b) / (torch.sqrt(torch.tensor(2.0)) * beta_o)
        win_prob = dist.Normal(0.0, 1.0).cdf(diff)
        pyro.sample("y", dist.Bernoulli(win_prob), obs=batch["winner"])

        # Layer 5: per-player stat observations
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

    # Mean-field (diagonal) posterior over performances — valid approximation
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


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

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
        losses = train(batch, n_players, n_steps=args.n_steps, lr=0.01)
        elapsed = time.perf_counter() - start

        pyro.get_param_store().save("params_corr.pt")
        print("Saved params to params_corr.pt")

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Step")
        plt.ylabel("ELBO loss")
        plt.title("SVI convergence (correlated)")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig("elbo_corr.png")
        print("Saved loss curve to elbo_corr.png")
        print(f"Training wall time: {elapsed:.2f}s")

    print_rankings_fast(n_players, idx_to_name, primary_role)

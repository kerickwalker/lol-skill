"""
LCK Season 15 baseline model with team-context and same-role diff stats.

Extends baseline.py by adding two additional stat layers:
  - Team-level output context: normalised team aggregates (kills, golds, cs, etc.)
    shift each player's expected individual stats.
  - Same-role opponent diff stats: each player's diff vs their role counterpart
    is observed as a function of the performance gap between the two players.

Everything else (skill prior, independent per-game performances, outcome layer,
individual stat observations) is identical to baseline.py.
"""

import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from tqdm import trange

from models.config import INDIVIDUAL_STATS, N_ROLES
from models.baseline import (
    ALPHA_VEC,
    DIFF_ALPHA_VEC,
    DIFF_GAMMA_MAT,
    DIFF_TAU_VEC,
    DURATION_CONTEXT_STATS,
    GAMMA_MAT,
    TAU_VEC,
    TEAM_CONTEXT_STATS,
    load_data,
    print_rankings,
    role_intercepts,
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

    team_output_loc = pyro.param(
        "team_output_effect_z_loc",
        torch.zeros(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
    )
    team_output_scale = pyro.param(
        "team_output_effect_z_scale",
        0.1 * torch.ones(len(TEAM_CONTEXT_STATS), len(INDIVIDUAL_STATS)),
        constraint=dist.constraints.positive,
    )
    pyro.sample("team_output_effect_z", dist.Normal(team_output_loc, team_output_scale).to_event(2))

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
